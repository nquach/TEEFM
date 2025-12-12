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

from datasets.optimized_video_dataset import OptimizedVideoDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE
from lightning_module import VideoMAELightningModule
from litdata import StreamingDataLoader

def safe_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    
    Supports both regular CSV-based datasets and optimized litdata datasets.
    
    Args:
        config (dict): Configuration dictionary
        train_cache_dir (str, optional): Unique cache directory for training dataset (for multi-GPU)
    
    Returns:
        tuple: (train_dataset, val_dataset, use_optimized) where val_dataset can be None if validation is disabled
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
        motion_centric_masking_ratio=model_config.get('motion_centric_masking_ratio', 0.7),
        frame_size=input_size,
        crop_scale=model_config.get('crop_scale', (0.75,1.0)),
        crop_aspect_ratio=model_config.get('crop_aspect_ratio', (0.8,1.2))
    )
    
    train_data_dir = data_config.get('train_optimized_dir')
    
    if train_data_dir is None:
        raise ValueError(
            "use_optimized is True but train_optimized_dir or val_optimized_dir is not specified. "
            "Please provide paths to optimized dataset directories."
        )
        
    print("Using optimized litdata datasets")
        
    train_cache = data_config.get('cache_dir')
    safe_makedir(train_cache)
    # Create training dataset from optimized data
    train_dataset = OptimizedVideoDataset(
        data_dir=train_data_dir,
        frames_to_sample=training_config.get('frames_to_sample', 16),
        temporal_stride=training_config.get('temporal_stride', 1),
        subset_ratio=data_config.get('subset_ratio'),
        seed=training_config.get('seed', 0),
        transform=transform,
        cache_dir=train_cache,
        drop_last=True
    )
    print(f'Created optimized training dataset from {train_data_dir} of length {len(train_dataset)}')
      
    return train_dataset


def create_data_loaders(train_dataset, config):
    """
    Create data loaders for training and validation.
    
    Uses StreamingDataLoader for optimized datasets, regular DataLoader otherwise.
    
    Args:
        train_dataset: Training dataset
        config (dict): Configuration dictionary
    
    Returns:
       train_loader
    """
    training_config = config['training']
    batch_size = training_config['batch_size']
    
    # Check if we're in DDP mode (will be True if multiple GPUs are available)
    # In DDP, validation must use drop_last=True to ensure all processes have same number of batches
    # This prevents deadlock when sync_dist=True synchronizes across processes
    is_ddp = torch.cuda.device_count() > 1
            
    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.get('num_workers', 10),
        pin_memory=training_config.get('pin_memory', True),
        persistent_workers=True if is_ddp else False,
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
    train_dataset = create_datasets(config)
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_data_loaders(train_dataset, config)
    
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
        checkpoint_prefix = checkpoint_config['prefix']
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_config.get('monitor', 'train_loss'),
            dirpath=checkpoint_dir,
            filename = f"{checkpoint_prefix}-{epoch:02d}-{train_loss:.4f}",
            mode='min',
            save_top_k=checkpoint_config.get('save_top_k', 1),
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=checkpoint_config.get('save_ckpt_freq', 50)
        )
    
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
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision on GPU
        gradient_clip_val=config.get('training', {}).get('gradient_clip_val', 0),
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, ckpt_path=args.resume if args.resume else None)
    
    print("Training completed!")
    if checkpoint_callback is not None:
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print("No checkpoints were saved (checkpoint saving is disabled).")


if __name__ == '__main__':
    main()

