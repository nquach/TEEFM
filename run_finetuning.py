"""
VideoMAE classification finetuning with PyTorch Lightning and LitData.

Loads config from YAML, builds FinetuningDataModule (LitData train/val/test with
optional train_test_split), encoder-only classification model from pretrained checkpoint,
schedule-free optimizer, and runs Trainer. Use --config and optional --resume.
"""

import math
import os
import warnings
from typing import Dict, Optional

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')

from datasets.finetuning_data_module import FinetuningDataModule
from lightning_module_finetune import VideoMAEFinetuneLightningModule


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_best_checkpoint(checkpoint_callback) -> Optional[str]:
    if checkpoint_callback is None:
        return None
    best_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_path and os.path.isfile(best_path):
        return best_path
    last_path = getattr(checkpoint_callback, "last_model_path", None)
    if last_path and os.path.isfile(last_path):
        return last_path
    return None


def _format_auroc_value(value) -> str:
    if value is None:
        return "nan"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _print_per_class_auroc_table(
    title: str,
    metrics: Dict,
    key_prefix: str,
    num_classes: int,
    global_rank: int = 0,
):
    if global_rank != 0:
        return
    macro_key = f"{key_prefix}_aucroc"
    macro_val = metrics.get(macro_key)
    print(f"\n{title}", end="")
    if macro_val is not None:
        print(f" — macro AUROC: {_format_auroc_value(macro_val)}")
    else:
        print()
    print("Class | AUROC")
    print("------|------")
    for class_id in range(num_classes):
        class_key = f"{key_prefix}_aucroc_class_{class_id}"
        print(f"{class_id:5d} | {_format_auroc_value(metrics.get(class_key))}")


def _print_eval_auroc_tables(
    split_label: str,
    metrics: Dict,
    key_prefix: str,
    num_classes: int,
    eval_multi_clip: bool,
    global_rank: int = 0,
):
    _print_per_class_auroc_table(
        f"{split_label} (clip-level)",
        metrics,
        key_prefix,
        num_classes,
        global_rank=global_rank,
    )
    if eval_multi_clip:
        _print_per_class_auroc_table(
            f"{split_label} (video-level)",
            metrics,
            f"{key_prefix}_video",
            num_classes,
            global_rank=global_rank,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='VideoMAE classification finetuning with PyTorch Lightning'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument(
        '--test_predictions_csv',
        type=str,
        default=None,
        help='Optional path to write clip-level test predictions CSV after test',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    ev = config.get("eval") or {}
    print(
        f"Eval: eval_protocol={ev.get('eval_protocol', 'single_clip')}, "
        f"video_id_key={config.get('data', {}).get('video_id_key', 'video_id')}"
    )

    seed = config.get('training', {}).get('seed', 0)
    pl.seed_everything(seed, workers=True)

    print("Creating data module...")
    data_module = FinetuningDataModule(config)
    data_module.setup('fit')
    if data_module.train_dataset is not None:
        print(f"Training dataset size: {len(data_module.train_dataset)}")

    print("Creating model...")
    model = VideoMAEFinetuneLightningModule(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    checkpoint_config = config.get('checkpoint', {})
    checkpoint_enabled = checkpoint_config.get('enable', True)
    checkpoint_callback = None
    if checkpoint_enabled:
        checkpoint_dir = checkpoint_config.get('dir', './output/checkpoints_finetune')
        os.makedirs(checkpoint_dir, exist_ok=True)
        prefix = checkpoint_config.get('prefix', 'finetune_')
        save_top_k = checkpoint_config.get('save_top_k', 1)
        strict_top_k = checkpoint_config.get('strict_top_k', False)
        save_last = False if strict_top_k else checkpoint_config.get('save_last', True)
        every_n_epochs = checkpoint_config.get('save_ckpt_freq')
        monitor = checkpoint_config.get('monitor', 'val_top1_acc')
        mode = checkpoint_config.get('mode', 'max')
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=checkpoint_dir,
            filename=prefix + "-{epoch:02d}-{" + monitor + ":.4f}",
            mode=mode,
            save_top_k=save_top_k,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=False,
            save_last=save_last,
        )
        print(f"Checkpoint: monitor={monitor}, mode={mode}, save_top_k={save_top_k}, save_last={save_last}")

    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', './output/logs_finetune')
    os.makedirs(log_dir, exist_ok=True)
    logger_name = checkpoint_config.get('prefix', 'finetune') if checkpoint_enabled else 'finetune'
    logger = TensorBoardLogger(save_dir=log_dir, name=logger_name)

    callbacks_list = [checkpoint_callback] if checkpoint_callback else []
    training_config = config.get('training', {})
    strategy = training_config.get('strategy', 'auto')
    if torch.cuda.device_count() > 1 and strategy == 'auto':
        strategy = 'ddp'
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 30),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy=strategy,
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        precision='16-mixed' if torch.cuda.is_available() else '32',
        gradient_clip_val=training_config.get('gradient_clip_val', 0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
    )

    print("Starting finetuning...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    print("Finetuning completed.")

    best_ckpt = _resolve_best_checkpoint(checkpoint_callback)
    if trainer.global_rank == 0:
        if best_ckpt:
            print(f"Best model checkpoint: {best_ckpt}")
        elif not checkpoint_enabled:
            print("Checkpointing disabled; per-class AUROC tables require a saved checkpoint.")
        else:
            print("No checkpoint file found; skipping per-class AUROC tables.")

    num_classes = config.get("data", {}).get("num_classes", 101)
    eval_multi_clip = (ev.get("eval_protocol") or "single_clip").lower() == "multi_clip"

    if best_ckpt and data_module.val_dataset is not None:
        if trainer.global_rank == 0:
            print("Running validation on best checkpoint for per-class AUROC...")
        val_results = trainer.validate(model, datamodule=data_module, ckpt_path=best_ckpt)
        val_metrics = val_results[0] if val_results else {}
        _print_eval_auroc_tables(
            "Validation",
            val_metrics,
            "val",
            num_classes,
            eval_multi_clip,
            global_rank=trainer.global_rank,
        )
    elif best_ckpt and data_module.val_dataset is None and trainer.global_rank == 0:
        print("Skipping validation AUROC tables: no val_dataset.")

    run_test = config.get("data", {}).get("run_test_after_fit", True)
    if run_test and data_module.test_dataset is not None:
        if best_ckpt is None:
            if trainer.global_rank == 0:
                print("Skipping test: no checkpoint available for evaluation.")
        else:
            if trainer.global_rank == 0:
                print("Running test step on best checkpoint...")
            test_predictions_csv = args.test_predictions_csv
            if test_predictions_csv is None:
                test_predictions_csv = (config.get("output") or {}).get("test_predictions_csv")
            if test_predictions_csv:
                parent = os.path.dirname(test_predictions_csv)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                model.test_predictions_csv_path = test_predictions_csv
            test_results = trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt)
            test_metrics = test_results[0] if test_results else {}
            _print_eval_auroc_tables(
                "Test",
                test_metrics,
                "test",
                num_classes,
                eval_multi_clip,
                global_rank=trainer.global_rank,
            )
    elif run_test and data_module.test_dataset is None and trainer.global_rank == 0:
        print("Skipping test: no test_dataset (set data.test_optimized_dir with val for explicit splits, or use train_test_split).")


if __name__ == '__main__':
    main()
