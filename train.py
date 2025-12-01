"""
Main training script for VideoMAE with EVEREST masking.

This script sets up and runs training using PyTorch Lightning.
Supports single-GPU and multi-GPU training automatically.

Usage:
    python train.py --backbone ViT-S
    python train.py --backbone ViT-B --batch_size 4
    python train.py --pretrained path/to/checkpoint.ckpt
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import VideoDataset, get_video_transforms
from training import VideoMAELightningModule
from config import VideoMAEConfig, get_vit_s_config, get_vit_b_config, get_vit_l_config


def setup_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # Better performance


def create_data_loaders(config: VideoMAEConfig):
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_video_transforms(mode='train', normalize=False)
    val_transform = get_video_transforms(mode='val', normalize=False)
    
    # Create datasets
    train_dataset = VideoDataset(
        csv_file=config.csv_file,
        num_frames=config.num_frames,
        temporal_stride=config.temporal_stride,
        transform=train_transform
    )
    
    # Validation dataset from separate CSV file
    val_dataset = VideoDataset(
        csv_file=config.val_csv_file,
        num_frames=config.num_frames,
        temporal_stride=config.temporal_stride,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VideoMAE with EVEREST masking')
    
    # Model arguments
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['ViT-S', 'ViT-B', 'ViT-L'],
        default='ViT-S',
        help='Vision Transformer backbone'
    )
    
    # Data arguments
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='Path to CSV file with video paths'
    )
    parser.add_argument(
        '--val_csv_file',
        type=str,
        default=None,
        help='Path to CSV file with validation video paths'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of data loader workers'
    )
    
    # Training arguments
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--mask_ratio',
        type=float,
        default=None,
        help='Masking ratio for EVEREST'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--devices',
        type=int,
        default=None,
        help='Number of GPUs to use (None = all available)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['32', '16-mixed', 'bf16-mixed'],
        default=None,
        help='Training precision'
    )
    
    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--checkpoint_prefix',
        type=str,
        default=None,
        help='Prefix for checkpoint filenames'
    )
    
    # Other arguments
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (resumes full training state)'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained model weights to initialize from (.ckpt, .pth, or .pt)'
    )
    parser.add_argument(
        '--load_pretrained_strict',
        action='store_true',
        default=None,
        help='Use strict loading for pretrained weights (default: True)'
    )
    parser.add_argument(
        '--load_pretrained_loose',
        action='store_true',
        default=None,
        help='Use loose (non-strict) loading for pretrained weights (allows partial loading)'
    )
    parser.add_argument(
        '--log_gradient_norm',
        action='store_true',
        default=None,
        help='Log L2 norm of full loss gradient (off by default)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name for logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.backbone == 'ViT-S':
        config = get_vit_s_config()
    elif args.backbone == 'ViT-B':
        config = get_vit_b_config()
    elif args.backbone == 'ViT-L':
        config = get_vit_l_config()
    else:
        config = VideoMAEConfig()
    
    # Override with CLI arguments
    if args.csv_file:
        config.csv_file = args.csv_file
    if args.val_csv_file:
        config.val_csv_file = args.val_csv_file
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.mask_ratio:
        config.mask_ratio = args.mask_ratio
    if args.devices:
        config.devices = args.devices
    if args.precision:
        config.precision = args.precision
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.pretrained:
        config.pretrained_checkpoint = args.pretrained
    if args.load_pretrained_strict is not None:
        config.load_pretrained_strict = True
    if args.load_pretrained_loose is not None:
        config.load_pretrained_strict = False
    if args.log_gradient_norm is not None:
        config.log_gradient_norm = args.log_gradient_norm
    if args.seed:
        config.seed = args.seed
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_prefix:
        config.checkpoint_filename_prefix = args.checkpoint_prefix
    
    # Set experiment name
    experiment_name = args.name or f'videomae_{config.backbone}_{config.mask_ratio}mask'
    
    # Setup random seed
    setup_seed(config.seed, config.deterministic)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create Lightning module
    print("Initializing model...")
    model = VideoMAELightningModule(
        backbone=config.backbone,
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_frames=config.final_num_frames,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        dropout=config.dropout,
        drop_path=config.drop_path,
        mask_ratio=config.mask_ratio,
        motion_weight=config.motion_weight,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        norm_pix_loss=config.norm_pix_loss,
        pretrained_checkpoint=config.pretrained_checkpoint,
        load_pretrained_strict=config.load_pretrained_strict,
        log_gradient_norm=config.log_gradient_norm
    )
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint callback
    if config.enable_checkpointing:
        # Build checkpoint filename with prefix
        checkpoint_filename = f"{config.checkpoint_filename_prefix}-{{epoch:02d}}-{{{config.monitor_metric.replace('/', '_')}:.2f}}"
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=checkpoint_filename,
            monitor=config.monitor_metric,
            mode=config.mode,
            save_top_k=3,  # Save top 3 checkpoints
            save_last=True,  # Always save last checkpoint (saved as 'last.ckpt')
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=experiment_name
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices if config.devices is not None else "auto",
        max_epochs=config.max_epochs,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    print(f"Backbone: {config.backbone}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Mask ratio: {config.mask_ratio}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Devices: {config.devices if config.devices is not None else 'auto'}")
    print(f"Precision: {config.precision}")
    if config.pretrained_checkpoint:
        print(f"Pretrained checkpoint: {config.pretrained_checkpoint}")
        print(f"Strict loading: {config.load_pretrained_strict}")
    if config.enable_checkpointing:
        print(f"Checkpoint directory: {config.checkpoint_dir}")
        print(f"Checkpoint filename prefix: {config.checkpoint_filename_prefix}")
    if config.log_gradient_norm:
        print(f"Gradient norm logging: Enabled")
    print("="*50 + "\n")
    
    # Start training
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=config.resume_from_checkpoint
    )
    
    print("Training completed!")
    if config.enable_checkpointing:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

