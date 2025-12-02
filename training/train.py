"""
Main training script for VideoMAE with EVEREST method.

This script sets up and runs training using PyTorch Lightning.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import VideoDataset
from models import VideoMAE
from training import VideoMAELightning


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_dataset, val_dataset: Dataset instances
    """
    data_config = config["data"]
    train_config = config["training"]
    
    # Training dataset
    train_dataset = VideoDataset(
        csv_file=data_config["train_csv"],
        subset_ratio=data_config.get("subset_ratio"),
        num_sample_frames=data_config["sample_frames"],
        temporal_stride=data_config["temporal_stride"],
    )
    
    # Validation dataset
    val_dataset = VideoDataset(
        csv_file=data_config["val_csv"],
        subset_ratio=None,  # Always use full validation set
        num_sample_frames=data_config["sample_frames"],
        temporal_stride=data_config["temporal_stride"],
    )
    
    return train_dataset, val_dataset


def create_model(config: dict) -> VideoMAE:
    """
    Create VideoMAE model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VideoMAE model instance
    """
    model_config = config["model"]
    everest_config = config["everest"]
    data_config = config["data"]
    
    model = VideoMAE(
        backbone=model_config["backbone"],
        img_size=data_config["frame_size"],
        patch_size=everest_config["patch_size"],
        tubelet_size=everest_config["tubelet_size"],
        num_frames=data_config["num_frames"],
        mask_ratio=everest_config["mask_ratio"],
        pretrained=model_config["pretrained"],
        pretrained_path=model_config.get("pretrained_path"),
    )
    
    return model


def create_lightning_module(model: VideoMAE, config: dict) -> VideoMAELightning:
    """
    Create PyTorch Lightning module.
    
    Args:
        model: VideoMAE model instance
        config: Configuration dictionary
        
    Returns:
        VideoMAELightning module instance
    """
    training_config = config["training"]
    everest_config = config["everest"]
    logging_config = config["logging"]
    
    lightning_module = VideoMAELightning(
        model=model,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        mask_ratio=everest_config["mask_ratio"],
        norm_pix_loss=True,
        track_gradient_norm=logging_config["track_gradient_norm"],
    )
    
    return lightning_module


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VideoMAE with EVEREST method")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model: {config['model']['backbone']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create Lightning module
    lightning_module = create_lightning_module(model, config)
    
    # Setup checkpoint callback
    checkpoint_config = config["checkpoint"]
    checkpoint_dir = checkpoint_config["dir"]
    checkpoint_prefix = checkpoint_config["prefix"]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor=checkpoint_config["monitor"],
        mode=checkpoint_config["mode"],
        save_top_k=checkpoint_config["save_top_k"],
        save_last=True,
    )
    
    # Setup logger
    logging_config = config["logging"]
    logger = TensorBoardLogger(
        save_dir=logging_config["log_dir"],
        name=logging_config["name"],
    )
    
    # Setup trainer
    hardware_config = config["hardware"]
    gpus = hardware_config["gpus"]
    if gpus == -1:
        gpus = torch.cuda.device_count()
    
    precision = hardware_config.get("precision", 32)
    if precision == 16:
        precision = "16-mixed" if hasattr(pl, "__version__") and int(pl.__version__.split(".")[0]) >= 2 else 16
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=gpus if gpus > 0 else None,
        precision=precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        log_every_n_steps=10,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

