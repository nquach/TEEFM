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
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.video_dataset import VideoDataset
from models.videomae import VideoMAE
from training.lightning_module import VideoMAELightningModule


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments, optionally loading from config file."""
    parser = argparse.ArgumentParser(description='Train VideoMAE with EVEREST method')
    
    # Config file argument (must be first to load defaults)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (default: config/config.yaml)')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Path to validation CSV file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--num_frames_to_sample', type=int, default=None,
                        help='Number of consecutive frames to sample')
    parser.add_argument('--temporal_stride', type=int, default=None,
                        help='Temporal stride for downsampling')
    parser.add_argument('--train_dataset_ratio', type=float, default=None,
                        help='Ratio of training dataset to use (0.0-1.0). If None, uses all data (default: None)')
    parser.add_argument('--dataset_random_seed', type=int, default=None,
                        help='Random seed for dataset sampling (default: None)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['vit_s', 'vit_b', 'vit_l'],
                        help='Vision Transformer backbone')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size')
    parser.add_argument('--mask_ratio', type=float, default=None,
                        help='Ratio of patches to mask')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Normalize pixel loss')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of warmup steps')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of epochs')
    parser.add_argument('--track_grad_norm', action='store_true',
                        help='Track L2 norm of gradients')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default=None,
                        help='Prefix for checkpoint filenames')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=None,
                        choices=[16, 32],
                        help='Training precision (16 or 32)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load config file if specified or use default
    config_path = args.config
    if config_path is None:
        # Try default config path
        default_config = os.path.join(project_root, 'config', 'config.yaml')
        if os.path.exists(default_config):
            config_path = default_config
    
    config = {}
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = load_config(config_path)
    else:
        print("No config file found, using command-line arguments and defaults")
    
    # Merge config with command-line arguments (CLI args override config)
    # Data config
    if args.train_csv is None:
        args.train_csv = config.get('data', {}).get('train_csv', 'mp4_paths.csv')
    if args.val_csv is None:
        args.val_csv = config.get('data', {}).get('val_csv', 'val500_2023-2024.csv')
    if args.batch_size is None:
        args.batch_size = config.get('data', {}).get('batch_size', 8)
    if args.num_workers is None:
        args.num_workers = config.get('data', {}).get('num_workers', 4)
    if args.num_frames_to_sample is None:
        args.num_frames_to_sample = config.get('data', {}).get('num_frames_to_sample', 32)
    if args.temporal_stride is None:
        args.temporal_stride = config.get('data', {}).get('temporal_stride', 2)
    if args.train_dataset_ratio is None:
        train_ratio = config.get('data', {}).get('train_dataset_ratio')
        args.train_dataset_ratio = train_ratio if train_ratio is not None else None
    if args.dataset_random_seed is None:
        seed = config.get('data', {}).get('dataset_random_seed')
        args.dataset_random_seed = seed if seed is not None else None
    
    # Model config
    if args.backbone is None:
        args.backbone = config.get('model', {}).get('backbone', 'vit_s')
    if args.img_size is None:
        args.img_size = config.get('model', {}).get('img_size', 224)
    if args.patch_size is None:
        args.patch_size = config.get('model', {}).get('patch_size', 16)
    if args.mask_ratio is None:
        args.mask_ratio = config.get('model', {}).get('mask_ratio', 0.75)
    if not args.norm_pix_loss:  # Only set from config if not specified via CLI
        args.norm_pix_loss = config.get('model', {}).get('norm_pix_loss', False)
    if args.pretrained is None:
        pretrained_path = config.get('model', {}).get('pretrained')
        args.pretrained = pretrained_path if pretrained_path else None
    
    # Training config
    if args.learning_rate is None:
        args.learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
    if args.weight_decay is None:
        args.weight_decay = config.get('training', {}).get('weight_decay', 0.05)
    if args.warmup_steps is None:
        args.warmup_steps = config.get('training', {}).get('warmup_steps', 1000)
    if args.max_epochs is None:
        args.max_epochs = config.get('training', {}).get('max_epochs', 100)
    if not args.track_grad_norm:  # Only set from config if not specified via CLI
        args.track_grad_norm = config.get('training', {}).get('track_grad_norm', False)
    
    # Checkpointing config
    if args.checkpoint_dir is None:
        args.checkpoint_dir = config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints')
    if args.checkpoint_prefix is None:
        args.checkpoint_prefix = config.get('checkpointing', {}).get('checkpoint_prefix', 'videomae')
    if args.resume_from_checkpoint is None:
        resume_path = config.get('checkpointing', {}).get('resume_from_checkpoint')
        args.resume_from_checkpoint = resume_path if resume_path else None
    
    # Logging config
    if args.log_dir is None:
        args.log_dir = config.get('logging', {}).get('log_dir', './logs')
    if args.experiment_name is None:
        args.experiment_name = config.get('logging', {}).get('experiment_name', 'videomae_experiment')
    
    # Hardware config
    if args.gpus is None:
        args.gpus = config.get('hardware', {}).get('gpus', 1)
    if args.precision is None:
        args.precision = config.get('hardware', {}).get('precision', 32)
    
    return args


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
        frame_size=(args.img_size, args.img_size),
        dataset_ratio=args.train_dataset_ratio,  # Only apply to training dataset
        random_seed=args.dataset_random_seed
    )
    
    val_dataset = VideoDataset(
        csv_file=args.val_csv,
        num_frames_to_sample=args.num_frames_to_sample,
        temporal_stride=args.temporal_stride,
        frame_size=(args.img_size, args.img_size),
        dataset_ratio=None,  # Always use full validation dataset
        random_seed=None
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
    
    # Determine accelerator and devices (PyTorch Lightning 2.0+ API)
    if torch.cuda.is_available() and args.gpus > 0:
        accelerator = 'gpu'
        devices = args.gpus
        strategy = 'ddp' if args.gpus > 1 else 'auto'
    else:
        accelerator = 'cpu'
        devices = 1
        strategy = 'auto'
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
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
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint  # Pass checkpoint path to fit() method
    )
    
    print("Training completed!")
    print(f"Best model saved at: {callbacks[0].best_model_path}")


if __name__ == '__main__':
    main()

