"""
Main training script for VideoMAE with EVEREST masking.

This script:
1. Loads configuration from YAML file
2. Initializes datasets and data loaders
3. Creates the VideoMAE model
4. Sets up PyTorch Lightning Trainer
5. Starts training with checkpointing and logging
"""

import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.video_dataset import VideoDataset
from models.videomae import VideoMAE
from training.lightning_module import VideoMAELightningModule


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict) -> tuple:
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_config = config['data']
    model_config = config['model']
    
    # Training dataset
    train_dataset = VideoDataset(
        csv_file=data_config['train_csv'],
        sample_frames=data_config['sample_frames'],
        temporal_stride=data_config['temporal_stride'],
        use_subset=data_config.get('use_subset', False),
        subset_ratio=data_config.get('subset_ratio', 0.1)
    )
    
    # Validation dataset
    val_dataset = VideoDataset(
        csv_file=data_config['val_csv'],
        sample_frames=data_config['sample_frames'],
        temporal_stride=data_config['temporal_stride'],
        use_subset=False,  # Always use full validation set
        subset_ratio=1.0
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: VideoDataset,
    val_dataset: VideoDataset,
    config: dict
) -> tuple:
    """
    Create training and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config['data']
    training_config = config['training']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True if data_config.get('num_workers', 0) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True if data_config.get('num_workers', 0) > 0 else False
    )
    
    return train_loader, val_loader


def create_model(config: dict) -> VideoMAE:
    """
    Create VideoMAE model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VideoMAE model instance
    """
    model_config = config['model']
    
    model = VideoMAE(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        num_frames=model_config['num_frames'],
        backbone=model_config['backbone'],
        mask_ratio=model_config['mask_ratio'],
        pretrained_weights=model_config.get('pretrained_weights')
    )
    
    print(f"Created VideoMAE model with {model_config['backbone']} backbone")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def create_lightning_module(model: VideoMAE, config: dict) -> VideoMAELightningModule:
    """
    Create PyTorch Lightning module.
    
    Args:
        model: VideoMAE model
        config: Configuration dictionary
        
    Returns:
        VideoMAELightningModule instance
    """
    training_config = config['training']
    model_config = config['model']
    
    lightning_module = VideoMAELightningModule(
        model=model,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.05),
        track_gradient_norm=training_config.get('track_gradient_norm', False),
        patch_size=model_config['patch_size'],
        img_size=model_config['img_size']
    )
    
    return lightning_module


def create_trainer(config: dict) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch Lightning Trainer instance
    """
    training_config = config['training']
    checkpoint_config = config['checkpoint']
    logging_config = config.get('logging', {})
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config['dir'],
        filename=f"{checkpoint_config['prefix']}-{{epoch:02d}}-{{{checkpoint_config['monitor']}:.4f}}",
        monitor=checkpoint_config['monitor'],
        mode=checkpoint_config['mode'],
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=True,
        every_n_epochs=checkpoint_config.get('save_every_n_epochs', 1)
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=logging_config.get('log_dir', './logs'),
        name=logging_config.get('experiment_name', 'videomae_everest')
    )
    
    # Determine number of GPUs
    gpus = training_config.get('gpus', 1)
    if isinstance(gpus, int) and gpus > 0:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            gpus = 0
    else:
        gpus = 0
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        gpus=gpus if isinstance(gpus, int) else None,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if isinstance(gpus, int) else None,
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        gradient_clip_val=training_config.get('gradient_clip_val', 0.0),
        precision=training_config.get('precision', 32),
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=logging_config.get('log_every_n_steps', 50),
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VideoMAE with EVEREST masking')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Create Lightning module
    print("Creating Lightning module...")
    lightning_module = create_lightning_module(model, config)
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(config)
    
    # Start training
    print("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("Training completed!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

