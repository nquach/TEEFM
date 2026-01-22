"""
Main Training Script for EVEREST VideoMAE

This script loads configuration from a YAML file, sets up datasets, data loaders,
PyTorch Lightning model, and trainer for training VideoMAE models.
"""

import os
import time
import warnings
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path

# Suppress torchvision warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')

from datasets import VideoDataModule
from lightning_module import VideoMAELightningModule

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
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VideoMAE with PyTorch Lightning')
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
    data_module = VideoDataModule(config)
    data_module.setup('fit')
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    
    # Create Lightning module
    print("Creating model...")
    model = VideoMAELightningModule(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Setup checkpoint callback (if enabled)
    checkpoint_config = config['checkpoint']
    checkpoint_enabled = checkpoint_config.get('enable', True)
    
    checkpoint_callback = None
    if checkpoint_enabled:
        checkpoint_dir = checkpoint_config['dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = checkpoint_config['prefix']
        
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
            monitor=checkpoint_config.get('monitor', 'train_loss_epoch'),
            dirpath=checkpoint_dir,
            filename = checkpoint_prefix + "-{epoch:02d}-{train_loss_epoch:.4f}",
            mode='min',
            save_top_k=save_top_k,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=True,
            save_last=save_last
        )
        
        # Print checkpoint configuration
        print(f"Checkpoint configuration:")
        print(f"  Save top K: {save_top_k}")
        print(f"  Save last: {save_last}")
        print(f"  Strict top K mode: {strict_top_k}")
        if every_n_epochs:
            print(f"  Periodic saves every {every_n_epochs} epochs")
    
    # Setup logging
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'output/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Use checkpoint prefix for logger name, or default name if checkpointing is disabled
    logger_name = checkpoint_config.get('prefix', 'videomae') if checkpoint_enabled else 'videomae'
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=logger_name
    )
    
    # Prepare callbacks list (only include checkpoint callback if enabled)
    callbacks_list = []
    if checkpoint_callback is not None:
        callbacks_list.append(checkpoint_callback)
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',  # Use all available GPUs
        strategy=config['training'].get('strategy', 'ddp') if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision on GPU
        gradient_clip_val=config['training'].get('gradient_clip_val', 0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1)
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)
    
    print("Training completed!")
    if checkpoint_callback is not None:
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print("No checkpoints were saved (checkpoint saving is disabled).")


if __name__ == '__main__':
    main()

