"""
Main training script for VideoMAE with EVEREST masking.

This script sets up and runs the training pipeline using PyTorch Lightning.
It supports:
- Single and multi-GPU training
- Checkpointing with customizable directory and prefix
- Gradient norm tracking (optional)
- Pretrained weight loading
- Easy hyperparameter configuration
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config
from data import VideoDataset
from models import VideoMAE
from training import VideoMAELightningModule


def create_data_loaders(config: Config):
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = VideoDataset(
        csv_file=config.data.train_csv,
        sample_frames=config.data.sample_frames,
        temporal_stride=config.data.temporal_stride,
    )
    
    # Create validation dataset
    val_dataset = VideoDataset(
        csv_file=config.data.val_csv,
        sample_frames=config.data.sample_frames,
        temporal_stride=config.data.temporal_stride,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )
    
    return train_loader, val_loader


def create_model(config: Config) -> VideoMAE:
    """
    Create VideoMAE model.
    
    Args:
        config: Configuration object
        
    Returns:
        VideoMAE model instance
    """
    model = VideoMAE(
        backbone=config.model.backbone,
        img_size=config.model.img_size,
        patch_size=config.model.patch_size,
        tubelet_size=config.model.tubelet_size,
        num_frames=config.model.num_frames,
        mask_ratio=config.model.mask_ratio,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        dropout=0.0,
        pretrained_weights=config.model.pretrained_weights,
    )
    
    return model


def create_lightning_module(model: VideoMAE, config: Config) -> VideoMAELightningModule:
    """
    Create PyTorch Lightning module.
    
    Args:
        model: VideoMAE model instance
        config: Configuration object
        
    Returns:
        Lightning module instance
    """
    lightning_module = VideoMAELightningModule(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        loss_type=config.training.loss_type,
        track_gradient_norm=config.training.track_gradient_norm,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    
    return lightning_module


def create_trainer(config: Config) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer.
    
    Args:
        config: Configuration object
        
    Returns:
        PyTorch Lightning trainer instance
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint.checkpoint_dir, exist_ok=True)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.checkpoint_dir,
        filename=f"{config.checkpoint.checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
        every_n_epochs=config.checkpoint.save_every_n_epochs,
    )
    
    # Setup learning rate monitor (optional, for logging)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config.checkpoint.checkpoint_dir,
        name='logs',
    )
    
    # Determine accelerator
    if config.trainer.gpus == 0:
        accelerator = 'cpu'
    else:
        accelerator = config.trainer.accelerator
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=config.trainer.gpus if config.trainer.gpus > 0 else None,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        precision=config.trainer.precision if config.training.use_amp else 32,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VideoMAE with EVEREST masking')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vit_s',
                       choices=['vit_s', 'vit_b', 'vit_l'],
                       help='ViT backbone type')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to pretrained weights (None for random init)')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='mp4_paths.csv',
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='val500_2023-2024.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value (None to disable)')
    parser.add_argument('--track_gradient_norm', action='store_true',
                       help='Track L2 norm of gradients (off by default)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint_prefix', type=str, default='videomae',
                       help='Checkpoint file prefix')
    
    # Trainer arguments
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = Config()
    config.model.backbone = args.backbone
    config.model.pretrained_weights = args.pretrained_weights
    config.data.train_csv = args.train_csv
    config.data.val_csv = args.val_csv
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.max_epochs = args.max_epochs
    config.training.gradient_clip_val = args.gradient_clip_val
    config.training.track_gradient_norm = args.track_gradient_norm
    config.checkpoint.checkpoint_dir = args.checkpoint_dir
    config.checkpoint.checkpoint_prefix = args.checkpoint_prefix
    config.trainer.gpus = args.gpus
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Backbone: {config.model.backbone}")
    print(f"Pretrained weights: {config.model.pretrained_weights}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Max epochs: {config.training.max_epochs}")
    print(f"GPUs: {config.trainer.gpus}")
    print(f"Checkpoint dir: {config.checkpoint.checkpoint_dir}")
    print(f"Checkpoint prefix: {config.checkpoint.checkpoint_prefix}")
    print("=" * 80)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create lightning module
    print("Creating Lightning module...")
    lightning_module = create_lightning_module(model, config)
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(config)
    
    # Train
    print("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
