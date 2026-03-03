"""
LitData classification finetuning script (PyTorch Lightning).

Trains a VideoMAE classification head on a LitData-optimized (video, label) dataset.
Uses PyTorch Lightning with schedule-free RAdam/AdamW, layerwise LR decay,
early stopping, and weighted AUROC/F1/accuracy on val and test.
"""

import argparse
import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import FinetuningLitDataDataModule
from lightning_module_finetune import VideoMAEFinetuneLightningModule


def get_args():
    parser = argparse.ArgumentParser(
        description="VideoMAE finetuning on LitData (video, label) dataset", add_help=False
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config checkpoint.output_dir")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Override config model.pretrained_path")
    parser.add_argument("--train_optimized_dir", type=str, default=None, help="Override config data.train_optimized_dir")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    known, _ = parser.parse_known_args()
    return known


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = get_args()
    config = load_config(args.config)

    if args.output_dir is not None:
        config.setdefault("checkpoint", {})["output_dir"] = args.output_dir
    if args.pretrained_path is not None:
        config.setdefault("model", {})["pretrained_path"] = args.pretrained_path
    if args.train_optimized_dir is not None:
        config.setdefault("data", {})["train_optimized_dir"] = args.train_optimized_dir

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})

    pl.seed_everything(training_config.get("seed", 0), workers=True)

    data_module = FinetuningLitDataDataModule(config)
    data_module.setup()
    print(f"Train samples: {len(data_module.train_ds)}, val: {len(data_module.val_ds)}, test: {len(data_module.test_ds)}")

    model = VideoMAEFinetuneLightningModule(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M, trainable: {trainable_params / 1e6:.2f}M")

    output_dir = checkpoint_config.get("output_dir", "./output/finetune_litdata")
    log_dir = checkpoint_config.get("log_dir", os.path.join(output_dir, "logs"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    monitor_metric = training_config.get("early_stopping_metric", "val_weighted_acc")
    monitor_mode = training_config.get("early_stopping_mode", "max")
    patience = training_config.get("early_stopping_patience", 5)
    save_ckpt_freq = checkpoint_config.get("save_ckpt_freq")
    save_ckpt = checkpoint_config.get("save_ckpt", True)

    checkpoint_callback = None
    if save_ckpt:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=save_ckpt_freq,
            save_on_train_epoch_end=False,
        )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=monitor_mode,
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="finetune_litdata")

    callbacks = [early_stopping_callback]
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=training_config.get("epochs", 30),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto" if torch.cuda.device_count() <= 1 else "ddp",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=training_config.get("clip_grad") or 0.0,
        accumulate_grad_batches=training_config.get("update_freq", 1),
    )

    resume_path = args.resume or checkpoint_config.get("resume") or None
    trainer.fit(model, data_module, ckpt_path=resume_path)

    print("Training completed.")
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        trainer.test(model, data_module, ckpt_path=checkpoint_callback.best_model_path)
    else:
        print("No best checkpoint saved; skipping test.")


if __name__ == "__main__":
    main()
