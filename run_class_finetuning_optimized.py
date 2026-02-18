"""
EVEREST-style classification finetuning with litdata optimized dataset (video + integer label).
Uses S3-streamed data, YAML config, schedule-free optimizers (RAdam/AdamW), and layer-wise LR decay.
Reuses engine_for_finetuning train/val/test loops and run_class_finetuning model/checkpoint logic.
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import yaml

import modeling_finetune  # noqa: F401 - register models
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

from datasets import FinetuningVideoDataModule
from engine_for_finetuning import final_test, train_one_epoch, validation_one_epoch
from mixup import Mixup
from optim_factory import LayerDecayValueAssigner, get_parameter_groups
from collections import OrderedDict

from litdata import StreamingDataLoader

from optimizers.schedule_free_optimizer import create_schedule_free_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import load_state_dict
import utils


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ScheduleFreeOptimizerWrapper:
    """Calls optimizer.train() before step() so schedule-free optimizers work correctly in the engine loop."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self, *args, **kwargs):
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        return self.optimizer.step(*args, **kwargs)


def collate_video_label_4tuple(batch):
    """Collate (video, label) list into (videos, labels, dummy, dummy) for engine_for_finetuning train/val."""
    videos = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return videos, labels, torch.zeros(0), torch.zeros(0)


def collate_video_label_5tuple(batch):
    """Collate (video, label) list into (videos, labels, ids, chunk_nb, split_nb) for final_test."""
    videos = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    B = videos.size(0)
    return videos, labels, torch.arange(B), torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)


BACKBONE_TO_MODEL = {
    "vit-s": "vit_small_patch16_224",
    "vit-b": "vit_base_patch16_224",
    "vit-l": "vit_large_patch16_224",
    "vit-h": "vit_huge_patch16_224",
}


def build_args_from_config(config):
    """Build a minimal args-like object for utils.save_model and compatibility."""
    out_cfg = config.get("output", {})
    train_cfg = config["training"]
    opt_cfg = config.get("optimizer", {})
    model_cfg = config["model"]
    return type("Args", (), {
        "output_dir": out_cfg.get("output_dir", "output"),
        "log_dir": out_cfg.get("log_dir"),
        "save_ckpt": out_cfg.get("save_ckpt", True),
        "save_ckpt_freq": out_cfg.get("save_ckpt_freq", 5),
        "resume": "",
        "auto_resume": False,
        "clip_grad": train_cfg.get("gradient_clip_val", 0) or None,
        "distributed": False,
        "num_frames": train_cfg.get("num_frames", 16),
        "num_segments": train_cfg.get("num_segments", 1),
        "model_key": model_cfg.get("model_key", "model|module"),
        "model_prefix": model_cfg.get("model_prefix", ""),
        "window_size": None,
        "patch_size": None,
    })()


def main():
    parser = argparse.ArgumentParser(description="EVEREST-style finetuning with optimized S3 dataset (YAML, schedule-free, layer decay)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args_cli = parser.parse_args()

    config = load_config(args_cli.config)
    args = build_args_from_config(config)

    if args_cli.resume:
        args.resume = args_cli.resume

    # Single-GPU (no distributed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config["training"].get("seed", 0)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    # Data: use FinetuningVideoDataModule then build loaders with our collate
    data_module = FinetuningVideoDataModule(config)
    data_module.setup("fit")

    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    disable_eval = config.get("output", {}).get("disable_eval_during_finetuning", False)
    if disable_eval:
        val_dataset = None
        test_dataset = None

    batch_size = config["training"]["batch_size"]
    val_batch_size = config["training"].get("val_batch_size", int(1.5 * batch_size))
    num_workers = config["training"].get("num_workers", 10)
    pin_memory = config["training"].get("pin_memory", True)

    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_video_label_4tuple,
    )

    if val_dataset is not None:
        val_loader = StreamingDataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_video_label_4tuple,
        )
    else:
        val_loader = None

    if test_dataset is not None:
        test_loader = StreamingDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_video_label_5tuple,
        )
    else:
        test_loader = None

    # Steps per epoch: config, or len(dataset), or large default
    steps_per_epoch_cfg = config["training"].get("steps_per_epoch")
    try:
        _len = len(train_dataset)
        num_training_steps_per_epoch = steps_per_epoch_cfg if steps_per_epoch_cfg is not None else (_len // batch_size)
    except (TypeError, NotImplementedError):
        num_training_steps_per_epoch = steps_per_epoch_cfg if steps_per_epoch_cfg is not None else 100000
    print(f"Steps per epoch: {num_training_steps_per_epoch}")

    # Model
    model_cfg = config["model"]
    backbone = model_cfg.get("backbone", "vit-b").lower()
    model_name = BACKBONE_TO_MODEL.get(backbone)
    if model_name is None:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(BACKBONE_TO_MODEL.keys())}")

    num_classes = model_cfg.get("num_classes", 101)
    num_frames = config["training"].get("num_frames", 16)
    num_segments = config["training"].get("num_segments", 1)

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        all_frames=num_frames * num_segments,
        tubelet_size=model_cfg.get("tubelet_size", 2),
        fc_drop_rate=model_cfg.get("fc_drop_rate", 0.0),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path", 0.1),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_block_rate=None,
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        use_mean_pooling=model_cfg.get("use_mean_pooling", True),
        init_scale=model_cfg.get("init_scale", 0.001),
        mcm=model_cfg.get("mcm", False),
        mcm_ratio=model_cfg.get("mcm_ratio", 0.4),
    )

    patch_size = model.patch_embed.patch_size
    args.window_size = (num_frames // 2, model_cfg.get("input_size", 224) // patch_size[0], model_cfg.get("input_size", 224) // patch_size[1])
    args.patch_size = patch_size

    # Load pretrained
    pretrained_path = model_cfg.get("pretrained_path") or ""
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model_key = model_cfg.get("model_key", "model|module")
        checkpoint_model = None
        for key in model_key.split("|"):
            if key in checkpoint:
                checkpoint_model = checkpoint[key]
                print(f"Load state_dict by model_key = {key}")
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("backbone."):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith("encoder."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        if "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            if orig_size != new_size:
                print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed
        load_state_dict(model, checkpoint_model, prefix=model_cfg.get("model_prefix", ""))

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_parameters / 1e6:.2f}M")

    # Model EMA (optional)
    model_ema = None
    ema_cfg = config.get("model_ema", {})
    if ema_cfg.get("enable", False):
        model_ema = ModelEma(
            model,
            decay=ema_cfg.get("decay", 0.9999),
            device="cpu" if ema_cfg.get("force_cpu", False) else "",
            resume="",
        )
        print(f"Using EMA with decay = {ema_cfg.get('decay', 0.9999):.8f}")

    # Optimizer: layer-wise LR decay + schedule-free
    opt_cfg = config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    layer_decay = float(opt_cfg.get("layer_decay", 1.0))
    num_layers = model_without_ddp.get_num_layers()
    skip_list = model_without_ddp.no_weight_decay()

    if layer_decay < 1.0:
        assigner = LayerDecayValueAssigner([layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)])
        raw_groups = get_parameter_groups(model_without_ddp, weight_decay, skip_list, assigner.get_layer_id, assigner.get_scale)
        param_groups = [{"params": g["params"], "lr": lr * g["lr_scale"], "weight_decay": g["weight_decay"]} for g in raw_groups]
        print(f"Layer-wise LR decay: layer_decay={layer_decay}, num_groups={len(param_groups)}")
        opt = create_schedule_free_optimizer(
            optimizer_type=opt_cfg.get("type", "radam"),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            eps=float(opt_cfg.get("eps", 1e-8)),
            warmup_steps=int(opt_cfg.get("warmup_steps", 0)),
            param_groups=param_groups,
        )
    else:
        opt = create_schedule_free_optimizer(
            model_without_ddp,
            optimizer_type=opt_cfg.get("type", "radam"),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            eps=float(opt_cfg.get("eps", 1e-8)),
            warmup_steps=int(opt_cfg.get("warmup_steps", 0)),
        )
    optimizer = ScheduleFreeOptimizerWrapper(opt)

    loss_scaler = NativeScaler()

    # Criterion
    aug_cfg = config.get("augmentation", {})
    mixup_alpha = aug_cfg.get("mixup", 0.0)
    cutmix_alpha = aug_cfg.get("cutmix", 0.0)
    cutmix_minmax = aug_cfg.get("cutmix_minmax")
    mixup_active = mixup_alpha > 0 or cutmix_alpha > 0 or cutmix_minmax is not None
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,
            prob=aug_cfg.get("mixup_prob", 1.0),
            switch_prob=aug_cfg.get("mixup_switch_prob", 0.5),
            mode=aug_cfg.get("mixup_mode", "batch"),
            label_smoothing=aug_cfg.get("label_smoothing", 0.1),
            num_classes=num_classes,
        )
        print("Mixup is activated!")
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif aug_cfg.get("label_smoothing", 0) > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=aug_cfg["label_smoothing"])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print(f"Criterion: {criterion}")

    # Log writer (optional)
    log_dir = config.get("output", {}).get("log_dir")
    log_writer = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=log_dir)

    update_freq = config["training"].get("update_freq", 1)
    max_epochs = config["training"].get("max_epochs", 30)
    start_epoch = config["training"].get("start_epoch", 0)
    clip_grad = args.clip_grad

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Start training for {max_epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    global_rank = 0
    num_tasks = 1

    for epoch in range(start_epoch, max_epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            loss_scaler,
            clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=None,
            wd_schedule_values=None,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=update_freq,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == max_epochs:
                utils.save_model(args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
        if val_loader is not None:
            test_stats = validation_one_epoch(val_loader, model, device)
            n_val = len(val_dataset) if val_dataset is not None else 0
            print(f"Accuracy of the network on the {n_val} val videos: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(args=args, epoch="best", model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            print(f"Max accuracy: {max_accuracy:.2f}%")
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats["acc1"], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats["acc5"], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats["loss"], head="perf", step=epoch)
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in test_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}
        else:
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if test_loader is not None:
        preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
        test_stats = final_test(test_loader, model, device, preds_file)
        if global_rank == 0:
            # Single-GPU: use aggregated test_stats; merge() is for multi-process / multi-crop
            final_top1 = test_stats["acc1"]
            final_top5 = test_stats["acc5"]
            n_test = len(test_dataset) if test_dataset is not None else 0
            print(f"Accuracy of the network on the {n_test} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {"Final top-1": final_top1, "Final Top-5": final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main()
