"""
Main Training Script for EVEREST VideoMAE

This script loads configuration from a YAML file, sets up datasets, data loaders,
PyTorch Lightning model, and trainer for training VideoMAE models.
"""

import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path

from datasets.custom_video_dataset import CustomVideoDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE
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


def create_datasets(config):
    """
    Create training and validation datasets.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    
    # Get normalization values
    normalize_mean = data_config.get('normalize_mean', [0.117, 0.114, 0.113])
    normalize_std = data_config.get('normalize_std', [0.208, 0.204, 0.203])
    
    # Calculate window size for masking
    num_frames = training_config.get('num_frames', 16)
    input_size = model_config.get('input_size', 224)
    patch_size = model_config.get('patch_size', 16)
    
    # Window size: (frames, height_patches, width_patches)
    # After tubelet_size=2, temporal dimension is num_frames // 2
    window_size = (
        num_frames // 2,
        input_size // patch_size,
        input_size // patch_size
    )
    
    # Create transform
    transform = DataAugmentationForVideoMAE(
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        window_size=window_size,
        mask_type=model_config.get('mask_type', 'motion-centric'),
        mask_ratio=model_config.get('mask_ratio', 0.9),
        motion_centric_masking_ratio=model_config.get('motion_centric_masking_ratio', 0.7)
    )
    
    # Create training dataset
    train_dataset = CustomVideoDataset(
        csv_file=data_config['train_csv'],
        frames_to_sample=training_config.get('frames_to_sample', 32),
        temporal_stride=training_config.get('temporal_stride', 2),
        subset_ratio=data_config.get('subset_ratio'),
        seed=training_config.get('seed', 0),
        transform=transform
    )
    
    # Create validation dataset (no subset sampling for validation)
    val_dataset = CustomVideoDataset(
        csv_file=data_config['val_csv'],
        frames_to_sample=training_config.get('frames_to_sample', 32),
        temporal_stride=training_config.get('temporal_stride', 2),
        subset_ratio=None,  # Always use full validation set
        seed=training_config.get('seed', 0),
        transform=transform
    )
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    training_config = config['training']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 10),
        pin_memory=training_config.get('pin_memory', True),
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 10),
        pin_memory=training_config.get('pin_memory', True),
        drop_last=False  # Keep all validation samples
    )
    
    return train_loader, val_loader


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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    # Create Lightning module
    print("Creating model...")
    model = VideoMAELightningModule(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Setup checkpoint callback
    checkpoint_config = config['checkpoint']
    checkpoint_dir = checkpoint_config['dir']
    checkpoint_prefix = checkpoint_config['prefix']
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode='min',  # Minimize validation loss
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=True,  # Always save last checkpoint
        verbose=True
    )
    
    # Setup logging
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'output/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=checkpoint_prefix
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',  # Use all available GPUs
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        enable_progress_bar=True,
        enable_model_summary=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision on GPU
        gradient_clip_val=config.get('training', {}).get('gradient_clip_val', None),
        profiler='advanced'
    )
    
    # Start training
    print("Starting training...")
    # Pass checkpoint path to fit() method instead of Trainer constructor
    # This is the correct way in newer PyTorch Lightning versions
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume if args.resume else None)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

