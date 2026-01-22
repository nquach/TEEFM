"""
Main script for VideoMAE finetuning with PyTorch Lightning

This script handles:
- Loading configuration from YAML file
- Creating data module and model
- Setting up PyTorch Lightning Trainer
- Checkpointing and logging
- Resuming from checkpoint
"""

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from datasets import FinetuningVideoDataModule
from lightning_module_finetune import VideoMAEFinetuningLightningModule


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main finetuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Finetune VideoMAE with PyTorch Lightning')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Set random seed for reproducibility
    seed = config['training'].get('seed', 0)
    pl.seed_everything(seed, workers=True)
    
    # Create data module
    print("Creating data module...")
    data_module = FinetuningVideoDataModule(config)
    data_module.setup('fit')
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    if data_module.val_dataset is not None:
        print(f"Validation dataset size: {len(data_module.val_dataset)}")
    if data_module.test_dataset is not None:
        print(f"Test dataset size: {len(data_module.test_dataset)}")
    
    # Create Lightning module
    print("Creating model...")
    model = VideoMAEFinetuningLightningModule(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Setup checkpoint callback (if enabled)
    checkpoint_config = config['checkpoint']
    checkpoint_enabled = checkpoint_config.get('enable', True)
    
    # Determine task type for checkpoint configuration
    task_type = config['model'].get('task_type', 'classification').lower()
    
    checkpoint_callback = None
    if checkpoint_enabled:
        checkpoint_dir = checkpoint_config['dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = checkpoint_config['prefix']
        
        # Configure checkpoint monitoring based on task type
        if task_type == 'regression':
            # For regression: minimize MSE (lower is better)
            default_monitor = 'val_mse'
            checkpoint_mode = 'min'
        else:  # classification
            # For classification: maximize accuracy (higher is better)
            default_monitor = 'val_acc1'
            checkpoint_mode = 'max'
        
        # Use configured monitor or default based on task type
        monitor_metric = checkpoint_config.get('monitor', default_monitor)
        
        # Get checkpoint management options
        save_top_k = checkpoint_config.get('save_top_k', 1)
        strict_top_k = checkpoint_config.get('strict_top_k', False)
        
        # Determine save_last behavior
        if strict_top_k:
            # Strict mode: only keep top_k checkpoints, no last checkpoint
            save_last = False
            # In strict mode, every_n_epochs should not create extra checkpoints
            # PyTorch Lightning's ModelCheckpoint already respects save_top_k for every_n_epochs
            every_n_epochs = checkpoint_config.get('save_ckpt_freq', None)
        else:
            # Normal mode: configurable save_last (default False to save disk space)
            save_last = checkpoint_config.get('save_last', False)
            every_n_epochs = checkpoint_config.get('save_ckpt_freq', None)
        
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=checkpoint_dir,
            filename=checkpoint_prefix + "-{epoch:02d}",
            mode=checkpoint_mode,  # 'max' for classification, 'min' for regression
            save_top_k=save_top_k,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=False,  # Save on validation end for finetuning
            save_last=save_last
        )
        
        print(f"Checkpoint callback configured for {task_type} task:")
        print(f"  Monitor: {monitor_metric}")
        print(f"  Mode: {checkpoint_mode}")
        print(f"  Save top K: {save_top_k}")
        print(f"  Save last: {save_last}")
        print(f"  Strict top K mode: {strict_top_k}")
        if every_n_epochs:
            print(f"  Periodic saves every {every_n_epochs} epochs")
    
    # Setup logging
    logging_config = config.get('logging', {})
    logger = None
    if logging_config.get('enable', True):
        log_dir = logging_config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name=logging_config.get('experiment_name', 'finetune')
        )
    
    # Build callbacks list
    callbacks_list = []
    if checkpoint_callback is not None:
        callbacks_list.append(checkpoint_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy=config['training'].get('strategy', 'ddp') if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        precision='16-mixed' if torch.cuda.is_available() else '32',
        gradient_clip_val=config['training'].get('gradient_clip_val', 0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        val_check_interval=config['training'].get('val_check_interval', 1.0)  # Validate every epoch by default
    )
    
    # Start training
    print("Starting finetuning...")
    trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)
    
    # Run test if test dataset is available
    if data_module.test_dataset is not None:
        print("Running test...")
        trainer.test(model, data_module)
    
    print("Finetuning completed!")


if __name__ == '__main__':
    main()

