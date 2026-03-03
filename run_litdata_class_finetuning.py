"""
LitData classification finetuning script.

Trains a VideoMAE classification head on a LitData-optimized (video, label) dataset.
Uses schedule-free RAdam/AdamW with layerwise LR decay, early stopping, and
weighted AUROC/F1/accuracy on val and test.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from scipy.special import softmax
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from datasets.litdata_labeled_dataset import build_litdata_finetune_datasets
from modeling.model_factory import create_videomae_finetune_model
from optim_factory import LayerDecayValueAssigner, get_parameter_groups
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer
from utils import (
    NativeScalerWithGradNormCount as NativeScaler,
    init_distributed_mode,
    is_main_process,
    get_rank,
    get_world_size,
    save_on_master,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="VideoMAE finetuning on LitData (video, label) dataset", add_help=False
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config output_dir")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Override config pretrained_path")
    parser.add_argument("--train_optimized_dir", type=str, default=None, help="Override config data.train_optimized_dir")
    known, _ = parser.parse_known_args()
    return known


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _to_namespace(d, parent=None):
    """Convert nested dict to object with attribute access."""
    if isinstance(d, dict):
        return type("Config", (), {k: _to_namespace(v) for k, v in d.items()})()
    if isinstance(d, list):
        return d
    return d


def compute_weighted_metrics(logits, targets, num_classes):
    """Compute weighted AUROC, weighted F1, weighted accuracy, and mean loss."""
    probs = softmax(logits, axis=1)
    preds = np.argmax(logits, axis=1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(
        torch.from_numpy(logits).float(),
        torch.from_numpy(targets).long(),
    ).item()

    try:
        if num_classes == 2:
            auroc = roc_auc_score(targets, probs[:, 1], average="weighted")
        else:
            auroc = roc_auc_score(
                targets, probs, multi_class="ovr", average="weighted"
            )
    except ValueError:
        auroc = float("nan")

    f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    weighted_acc = recall_score(
        targets, preds, average="weighted", zero_division=0
    )
    return {
        "loss": loss,
        "weighted_auroc": auroc,
        "weighted_f1": f1,
        "weighted_acc": weighted_acc,
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    epoch,
    loss_scaler,
    clip_grad,
    log_writer,
    print_freq=10,
    update_freq=1,
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    n = len(data_loader)
    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        if loss_scaler is not None:
            loss = loss / update_freq
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=model.parameters(),
                update_grad=(step + 1) % update_freq == 0,
            )
            if (step + 1) % update_freq == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % update_freq == 0:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        if hasattr(optimizer, "train"):
            optimizer.train()

        if step % print_freq == 0 or step == n - 1:
            print(
                f"Epoch [{epoch}] Step [{step}/{n}] loss {loss_value:.4f}"
            )
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.set_step()

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, distributed=False):
    """Run evaluation and return weighted AUROC, F1, accuracy, loss."""
    model.eval()
    all_logits = []
    all_targets = []

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
        all_logits.append(outputs.float().cpu())
        all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    if distributed and get_world_size() > 1:
        world_size = get_world_size()
        rank = get_rank()
        # Gather sizes and then data
        local_size = torch.tensor([logits.shape[0]], device=device, dtype=torch.long)
        sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)
        # Pad and gather
        if logits.shape[0] < max_size:
            pad_logits = np.zeros((max_size - logits.shape[0], logits.shape[1]), dtype=logits.dtype)
            logits = np.concatenate([logits, pad_logits], axis=0)
            pad_targets = np.zeros(max_size - targets.shape[0], dtype=targets.dtype)
            targets = np.concatenate([targets, pad_targets], axis=0)
        logits_t = torch.from_numpy(logits).to(device)
        targets_t = torch.from_numpy(targets).to(device)
        logits_gather = [torch.zeros_like(logits_t) for _ in range(world_size)]
        targets_gather = [torch.zeros_like(targets_t) for _ in range(world_size)]
        torch.distributed.all_gather(logits_gather, logits_t)
        torch.distributed.all_gather(targets_gather, targets_t)
        logits_list = [g.cpu().numpy() for g in logits_gather]
        targets_list = [g.cpu().numpy() for g in targets_gather]
        # Trim to actual sizes
        logits_list = [x[:sizes[i].item()] for i, x in enumerate(logits_list)]
        targets_list = [x[:sizes[i].item()] for i, x in enumerate(targets_list)]
        logits = np.concatenate(logits_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

    return compute_weighted_metrics(logits, targets, num_classes)


def main():
    args = get_args()
    config = load_config(args.config)

    # Overrides from CLI
    if args.output_dir is not None:
        config.setdefault("checkpoint", {})["output_dir"] = args.output_dir
    if args.pretrained_path is not None:
        config.setdefault("model", {})["pretrained_path"] = args.pretrained_path
    if args.train_optimized_dir is not None:
        config.setdefault("data", {})["train_optimized_dir"] = args.train_optimized_dir

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})
    checkpoint_config = config.get("checkpoint", {})

    # Build a minimal args object for distributed and save_model
    class RunArgs:
        pass

    run_args = RunArgs()
    run_args.output_dir = checkpoint_config.get("output_dir", "./output/finetune_litdata")
    run_args.distributed = False
    run_args.world_size = 1
    run_args.rank = 0
    run_args.gpu = 0
    run_args.dist_on_itp = False
    run_args.dist_url = "env://"
    run_args.local_rank = -1
    run_args.resume = checkpoint_config.get("resume", "")
    run_args.auto_resume = checkpoint_config.get("auto_resume", False)
    run_args.save_ckpt = checkpoint_config.get("save_ckpt", True)
    run_args.save_ckpt_freq = checkpoint_config.get("save_ckpt_freq", 5)

    init_distributed_mode(run_args)

    if is_main_process():
        Path(run_args.output_dir).mkdir(parents=True, exist_ok=True)
        log_dir = checkpoint_config.get("log_dir", os.path.join(run_args.output_dir, "logs"))
        os.makedirs(log_dir, exist_ok=True)
        try:
            from tensorboardX import SummaryWriter
            log_writer = SummaryWriter(logdir=log_dir)
        except Exception:
            log_writer = None
    else:
        log_writer = None

    device = torch.device(training_config.get("device", "cuda"))
    seed = training_config.get("seed", 0) + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Data
    train_ds, val_ds, test_ds, num_classes = build_litdata_finetune_datasets(
        data_config, seed=training_config.get("seed", 0)
    )
    num_workers = data_config.get("num_workers", 10)
    pin_mem = data_config.get("pin_memory", True)
    batch_size = training_config.get("batch_size", 32)
    update_freq = training_config.get("update_freq", 1)

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    def _make_loader(dataset, shuffle, drop_last=False):
        if run_args.distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_mem,
                drop_last=drop_last,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    data_loader_train = _make_loader(train_ds, shuffle=True, drop_last=True)
    data_loader_val = _make_loader(val_ds, shuffle=False) if val_ds else None
    data_loader_test = _make_loader(test_ds, shuffle=False) if test_ds else None

    # Model
    model = create_videomae_finetune_model(
        backbone=model_config.get("backbone", "vit-b"),
        pretrained_path=model_config.get("pretrained_path"),
        num_classes=num_classes,
        num_frames=model_config.get("num_frames", 16),
        tubelet_size=model_config.get("tubelet_size", 2),
        input_size=model_config.get("input_size", 224),
        fc_drop_rate=model_config.get("fc_drop_rate", 0.0),
        drop_rate=model_config.get("drop_rate", 0.0),
        drop_path_rate=model_config.get("drop_path", 0.1),
        attn_drop_rate=model_config.get("attn_drop_rate", 0.0),
        use_checkpoint=model_config.get("use_checkpoint", False),
        use_mean_pooling=model_config.get("use_mean_pooling", True),
        init_scale=model_config.get("init_scale", 0.001),
        mcm=model_config.get("mcm", False),
        mcm_ratio=model_config.get("mcm_ratio", 0.4),
        model_key=model_config.get("model_key", "model|module"),
        model_prefix=model_config.get("model_prefix", ""),
    )
    model.to(device)

    model_without_ddp = model
    if run_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[run_args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Layer decay + schedule-free optimizer
    num_layers = model_without_ddp.get_num_layers()
    layer_decay = optimizer_config.get("layer_decay", 0.75)
    weight_decay = optimizer_config.get("weight_decay", 0.05)
    base_lr = optimizer_config.get("lr", 1e-3)
    if isinstance(base_lr, (list, tuple)):
        base_lr = float(base_lr[0])
    else:
        base_lr = float(base_lr)

    if layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        )
        param_groups_raw = get_parameter_groups(
            model_without_ddp,
            weight_decay,
            model_without_ddp.no_weight_decay(),
            assigner.get_layer_id,
            assigner.get_scale,
        )
        param_groups = [
            {
                "params": g["params"],
                "lr": base_lr * g["lr_scale"],
                "weight_decay": g["weight_decay"],
            }
            for g in param_groups_raw
        ]
    else:
        param_groups = None

    if param_groups is not None:
        optimizer = create_schedule_free_optimizer(
            optimizer_type=optimizer_config.get("type", "radam"),
            param_groups=param_groups,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
            eps=optimizer_config.get("eps", 1e-8),
            warmup_steps=optimizer_config.get("warmup_steps", 0),
        )
    else:
        optimizer = create_schedule_free_optimizer(
            model=model_without_ddp,
            optimizer_type=optimizer_config.get("type", "radam"),
            lr=base_lr,
            weight_decay=weight_decay,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
            eps=optimizer_config.get("eps", 1e-8),
            warmup_steps=optimizer_config.get("warmup_steps", 0),
        )

    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()
    clip_grad = training_config.get("clip_grad")

    epochs = training_config.get("epochs", 30)
    patience = training_config.get("early_stopping_patience", 5)
    monitor_metric = training_config.get("early_stopping_metric", "val_weighted_acc")
    mode = training_config.get("early_stopping_mode", "max")
    best_metric = -float("inf") if mode == "max" else float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    start_epoch = 0
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        if run_args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            criterion,
            optimizer,
            device,
            epoch,
            loss_scaler,
            clip_grad,
            log_writer,
            print_freq=10,
            update_freq=update_freq,
        )

        if data_loader_val is not None:
            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                num_classes,
                distributed=run_args.distributed,
            )
            val_stats = {f"val_{k}": v for k, v in val_stats.items()}
            if is_main_process():
                print(
                    f"Epoch {epoch} val loss={val_stats['val_loss']:.4f} "
                    f"weighted_auroc={val_stats['val_weighted_auroc']:.4f} "
                    f"weighted_f1={val_stats['val_weighted_f1']:.4f} "
                    f"weighted_acc={val_stats['val_weighted_acc']:.4f}"
                )
            if log_writer is not None:
                for k, v in val_stats.items():
                    log_writer.update(**{k: v}, head="val", step=epoch)
                log_writer.flush()

            current = val_stats.get(monitor_metric)
            if current is None:
                current = val_stats.get("val_loss")
                mode_infer = "min"
            else:
                mode_infer = mode
            is_better = (
                (mode_infer == "max" and current > best_metric)
                or (mode_infer == "min" and current < best_metric)
            )
            if is_better:
                best_metric = current
                best_epoch = epoch
                epochs_without_improve = 0
                if run_args.save_ckpt and is_main_process():
                    to_save = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "scaler": loss_scaler.state_dict(),
                    }
                    save_on_master(
                        to_save,
                        os.path.join(run_args.output_dir, "checkpoint-best.pth"),
                    )
            else:
                epochs_without_improve += 1

        if run_args.save_ckpt and (epoch + 1) % run_args.save_ckpt_freq == 0 and is_main_process():
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
            }
            save_on_master(
                to_save,
                os.path.join(run_args.output_dir, f"checkpoint-{epoch}.pth"),
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if data_loader_val is not None:
            log_stats.update(val_stats)
        if is_main_process():
            with open(
                os.path.join(run_args.output_dir, "log.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if epochs_without_improve >= patience:
            if is_main_process():
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
            break

    # Load best checkpoint on all ranks for correct DDP test
    if data_loader_test is not None and run_args.save_ckpt and best_epoch >= 0:
        best_path = os.path.join(run_args.output_dir, "checkpoint-best.pth")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            model_without_ddp.load_state_dict(ckpt["model"])
            if is_main_process():
                print(f"Loaded best checkpoint from epoch {best_epoch} for final test.")

    if data_loader_test is not None:
        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            num_classes,
            distributed=run_args.distributed,
        )
        if is_main_process():
            print(
                f"Test loss={test_stats['loss']:.4f} "
                f"weighted_auroc={test_stats['weighted_auroc']:.4f} "
                f"weighted_f1={test_stats['weighted_f1']:.4f} "
                f"weighted_acc={test_stats['weighted_acc']:.4f}"
            )
            with open(
                os.path.join(run_args.output_dir, "log.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps({"test": test_stats}) + "\n")

    total_time = time.time() - start_time
    if is_main_process():
        print(f"Training time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
