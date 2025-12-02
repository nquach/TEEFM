"""
Main Training Script for VideoMAE

This script handles the training of VideoMAE models using PyTorch Lightning.
It supports:
- Multi-GPU training
- Configurable checkpointing
- Hyperparameter tuning
- Pretrained weight loading
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.video_dataset import VideoDataset
from models.videomae import VideoMAE
from training.lightning_module import VideoMAELightningModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VideoMAE with EVEREST method')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='mp4_paths.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='val500_2023-2024.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_frames_to_sample', type=int, default=32,
                        help='Number of consecutive frames to sample')
    parser.add_argument('--temporal_stride', type=int, default=2,
                        help='Temporal stride for downsampling')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vit_s',
                        choices=['vit_s', 'vit_b', 'vit_l'],
                        help='Vision Transformer backbone')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Ratio of patches to mask')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Normalize pixel loss')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--track_grad_norm', action='store_true',
                        help='Track L2 norm of gradients')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='videomae',
                        help='Prefix for checkpoint filenames')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--experiment_name', type=str, default='videomae_experiment',
                        help='Experiment name for logging')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32,
                        choices=[16, 32],
                        help='Training precision (16 or 32)')
    
    return parser.parse_args()


def create_data_loaders(args):
    """
    Create training and validation data loaders.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = VideoDataset(
        csv_file=args.train_csv,
        num_frames_to_sample=args.num_frames_to_sample,
        temporal_stride=args.temporal_stride,
        frame_size=(args.img_size, args.img_size)
    )
    
    val_dataset = VideoDataset(
        csv_file=args.val_csv,
        num_frames_to_sample=args.num_frames_to_sample,
        temporal_stride=args.temporal_stride,
        frame_size=(args.img_size, args.img_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_model(args):
    """
    Create VideoMAE model.
    
    Args:
        args: Parsed arguments
        
    Returns:
        VideoMAE model instance
    """
    model = VideoMAE(
        backbone=args.backbone,
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        pretrained=args.pretrained
    )
    
    return model


def create_lightning_module(args, model):
    """
    Create PyTorch Lightning module.
    
    Args:
        args: Parsed arguments
        model: VideoMAE model instance
        
    Returns:
        Lightning module instance
    """
    lightning_module = VideoMAELightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        track_grad_norm=args.track_grad_norm
    )
    
    return lightning_module


def create_callbacks(args):
    """
    Create training callbacks.
    
    Args:
        args: Parsed arguments
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor (for logging, even though we use schedule-free optimizer)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating VideoMAE model with {args.backbone} backbone...")
    model = create_model(args)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create lightning module
    print("Creating PyTorch Lightning module...")
    lightning_module = create_lightning_module(args, model)
    
    # Create callbacks
    print("Setting up callbacks...")
    callbacks = create_callbacks(args)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Create trainer
    print("Creating PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus if torch.cuda.is_available() else 0,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=args.resume_from_checkpoint,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        strategy='ddp' if args.gpus > 1 else None,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("Training completed!")
    print(f"Best model saved at: {callbacks[0].best_model_path}")


if __name__ == '__main__':
    main()

