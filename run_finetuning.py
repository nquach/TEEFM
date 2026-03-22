"""
VideoMAE classification finetuning with PyTorch Lightning and LitData.

Loads config from YAML, builds FinetuningDataModule (LitData train/val/test with
optional train_test_split), encoder-only classification model from pretrained checkpoint,
schedule-free optimizer, and runs Trainer. Use --config and optional --resume.
"""

import os
import warnings
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='VideoMAE classification finetuning with PyTorch Lightning'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
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
    if checkpoint_callback is not None:
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

    run_test = config.get("data", {}).get("run_test_after_fit", True)
    if run_test and data_module.test_dataset is not None:
        test_ckpt = None
        if checkpoint_callback is not None:
            bp = getattr(checkpoint_callback, "best_model_path", None)
            if bp and os.path.isfile(bp):
                test_ckpt = bp
            else:
                lp = getattr(checkpoint_callback, "last_model_path", None)
                if lp and os.path.isfile(lp):
                    test_ckpt = lp
        print("Running test step...")
        trainer.test(model, datamodule=data_module, ckpt_path=test_ckpt)
    elif run_test and data_module.test_dataset is None:
        print("Skipping test: no test_dataset (set data.test_optimized_dir with val for explicit splits, or use train_test_split).")


if __name__ == '__main__':
    main()
