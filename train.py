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

from datasets.custom_video_dataset import CustomVideoDataset
from datasets.optimized_video_dataset import OptimizedVideoDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE
from lightning_module import VideoMAELightningModule

# Try to import litdata components
try:
    from litdata import StreamingDataLoader
    LITDATA_AVAILABLE = True
except ImportError:
    LITDATA_AVAILABLE = False
    StreamingDataLoader = None
    print("Warning: litdata package not found. Install with: pip install litdata")


class DistributedModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint callback that fixes distributed training issues.
    
    Overrides the file_exists method to avoid problematic broadcasts in distributed
    training that cause SymIntArrayRef errors. Only rank 0 performs file existence
    checks, and the result is returned directly without broadcasting.
    """
    
    def file_exists(self, filepath, trainer):
        """
        Check if a file exists, avoiding distributed broadcast issues.
        
        In distributed training, only rank 0 checks file existence and returns
        the result. Other ranks return False without checking, since only rank 0
        performs checkpoint saves anyway.
        
        Args:
            filepath: Path to the file to check
            trainer: PyTorch Lightning trainer instance
        
        Returns:
            bool: True if file exists (on rank 0), False otherwise
        """
        # Check if we're in a distributed setting
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            # Only rank 0 checks file existence
            if rank == 0:
                exists = os.path.exists(filepath)
                return exists
            else:
                # Other ranks don't need to check since only rank 0 saves
                return False
        else:
            # Single process - check normally
            return os.path.exists(filepath)


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


def setup_litdata_cache_dirs(config):
    """
    Setup shared cache directories for litdata StreamingDataset in multi-GPU training.
    
    Uses shared cache directories (same for all processes) instead of per-process caches.
    Litdata's StreamingDataset has built-in synchronization mechanisms to handle concurrent
    access safely. In ddp_spawn mode, all processes need to ensure directories exist.
    
    Args:
        config (dict): Configuration dictionary containing cache directory paths
    
    Returns:
        tuple: (train_cache_dir, val_cache_dir) - Shared cache directories for all processes
    """
    data_config = config.get('data', {})
    
    # Get base cache directories from config
    base_train_cache = data_config.get('cache_dir', './output/cache')
    base_val_cache = data_config.get('val_cache_dir', './output/val_cache')
    
    # Convert to absolute paths to ensure consistency across all processes
    # This is especially important in ddp_spawn where processes may have different working directories
    train_cache_dir = str(Path(base_train_cache).resolve())
    val_cache_dir = str(Path(base_val_cache).resolve())
    
    # Detect process rank for multi-GPU training
    # PyTorch Lightning sets LOCAL_RANK and RANK environment variables before spawning processes
    # Try environment variables first (most reliable for PyTorch Lightning DDP)
    rank_str = os.environ.get('LOCAL_RANK') or os.environ.get('RANK', '0')
    try:
        rank = int(rank_str)
    except (ValueError, TypeError):
        # Fall back to torch.distributed if available and initialized
        try:
            if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except (AttributeError, RuntimeError):
            # Single GPU or distributed not available
            rank = 0
    
    # Create cache directories on all ranks
    # os.makedirs with exist_ok=True is safe for concurrent calls from multiple processes
    # In ddp_spawn, each process is separate, so all need to ensure directories exist
    try:
        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(val_cache_dir, exist_ok=True)
    except OSError as e:
        # If directory creation fails, log and re-raise
        print(f"Process rank {rank}: Failed to create cache directories: {e}")
        raise
    
    # Verify directories exist (with a small retry for ddp_spawn synchronization)
    max_retries = 5
    for retry in range(max_retries):
        if os.path.exists(train_cache_dir) and os.path.exists(val_cache_dir):
            break
        if retry < max_retries - 1:
            time.sleep(0.1)  # Small delay before retry
        else:
            raise RuntimeError(
                f"Process rank {rank}: Cache directories do not exist after creation: "
                f"train={train_cache_dir}, val={val_cache_dir}"
            )
    
    print(f"Process rank {rank}: Using shared cache directories:")
    print(f"  Train cache: {train_cache_dir}")
    print(f"  Val cache: {val_cache_dir}")
    
    return train_cache_dir, val_cache_dir


def create_datasets(config, train_cache_dir=None, val_cache_dir=None):
    """
    Create training and validation datasets.
    
    Supports both regular CSV-based datasets and optimized litdata datasets.
    
    Args:
        config (dict): Configuration dictionary
        train_cache_dir (str, optional): Unique cache directory for training dataset (for multi-GPU)
        val_cache_dir (str, optional): Unique cache directory for validation dataset (for multi-GPU)
    
    Returns:
        tuple: (train_dataset, val_dataset, use_optimized)
    """
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    
    # Check if optimized datasets should be used
    use_optimized = data_config.get('use_optimized', False)
    
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
    
    if use_optimized:
        # Use optimized litdata datasets
        if not LITDATA_AVAILABLE:
            raise ImportError(
                "litdata package is required for optimized datasets. "
                "Install with: pip install litdata"
            )
        
        train_data_dir = data_config.get('train_optimized_dir')
        val_data_dir = data_config.get('val_optimized_dir')
        
        if train_data_dir is None or val_data_dir is None:
            raise ValueError(
                "use_optimized is True but train_optimized_dir or val_optimized_dir is not specified. "
                "Please provide paths to optimized dataset directories."
            )
        
        print("Using optimized litdata datasets")
        
        # Use unique cache directories if provided (for multi-GPU), otherwise use config values
        train_cache = train_cache_dir if train_cache_dir is not None else data_config.get('cache_dir')
        val_cache = val_cache_dir if val_cache_dir is not None else data_config.get('val_cache_dir')
        
        is_ddp = torch.cuda.device_count() > 1
        # Create validation dataset from optimized data
        val_dataset = OptimizedVideoDataset(
            data_dir=val_data_dir,
            frames_to_sample=training_config.get('frames_to_sample', 32),
            temporal_stride=training_config.get('temporal_stride', 2),
            subset_ratio=None,  # Always use full validation set
            seed=training_config.get('seed', 0),
            transform=transform,
            cache_dir=val_cache,
            custom_drop_last=True if is_ddp else False
        )
        print(f'Created optimized validation dataset from {val_data_dir} of length {len(val_dataset)}')
        
        # Create training dataset from optimized data
        train_dataset = OptimizedVideoDataset(
            data_dir=train_data_dir,
            frames_to_sample=training_config.get('frames_to_sample', 32),
            temporal_stride=training_config.get('temporal_stride', 2),
            subset_ratio=data_config.get('subset_ratio'),
            seed=training_config.get('seed', 0),
            transform=transform,
            cache_dir=train_cache,
            custom_drop_last=True
        )
        print(f'Created optimized training dataset from {train_data_dir} of length {len(train_dataset)}')
       
    else:
        # Use regular CSV-based datasets
        print("Using regular CSV-based datasets")
        
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
    
    return train_dataset, val_dataset, use_optimized


def create_data_loaders(train_dataset, val_dataset, config, use_optimized=False):
    """
    Create data loaders for training and validation.
    
    Uses StreamingDataLoader for optimized datasets, regular DataLoader otherwise.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config (dict): Configuration dictionary
        use_optimized (bool): Whether to use StreamingDataLoader for optimized datasets
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    training_config = config['training']
    batch_size = training_config['batch_size']
    
    # Check if we're in DDP mode (will be True if multiple GPUs are available)
    # In DDP, validation must use drop_last=True to ensure all processes have same number of batches
    # This prevents deadlock when sync_dist=True synchronizes across processes
    is_ddp = torch.cuda.device_count() > 1
    
    if use_optimized and LITDATA_AVAILABLE:
        # Use StreamingDataLoader for optimized datasets
        print("Using StreamingDataLoader for optimized datasets")
            
        train_loader = StreamingDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config.get('num_workers', 10),
            pin_memory=training_config.get('pin_memory', True),
            persistent_workers=True if is_ddp else False,
            drop_last=True
        )
        
        # Reduce num_workers for validation to prevent resource contention and potential deadlocks
        val_num_workers = max(1, training_config.get('num_workers', 10) // 2)
        val_loader = StreamingDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=training_config.get('pin_memory', True),
            drop_last=True if is_ddp else False
        )
    else:
        # Use regular DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config.get('num_workers', 10),
            pin_memory=training_config.get('pin_memory', True),
            drop_last=True  # Drop last incomplete batch
        )
        
        # For validation in DDP, use drop_last=True to prevent deadlock
        # This ensures all processes have the same number of batches, which is required
        # for sync_dist=True to work correctly in validation_step
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Validation should not be shuffled
            num_workers=training_config.get('num_workers', 10),
            pin_memory=training_config.get('pin_memory', True),
            drop_last=True if is_ddp else False  # Drop last batch in DDP to prevent deadlock
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
    
    # Setup unique cache directories for multi-GPU training (if using optimized datasets)
    # This prevents race conditions when multiple processes download/decompress chunks
    train_cache_dir = None
    val_cache_dir = None
    if config.get('data', {}).get('use_optimized', False):
        train_cache_dir, val_cache_dir = setup_litdata_cache_dirs(config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, use_optimized = create_datasets(
        config, 
        train_cache_dir=train_cache_dir, 
        val_cache_dir=val_cache_dir
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config, use_optimized)
    
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
        
        # Only create checkpoint directory on rank 0 to avoid race conditions
        # Check if we're in a distributed setting
        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Configure ModelCheckpoint to avoid distributed broadcast issues
        # Use save_on_train_epoch_end=False to save only after validation
        # Use custom DistributedModelCheckpoint to avoid file existence broadcast errors
        checkpoint_callback = DistributedModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            mode='min',  # Minimize validation loss
            save_top_k=checkpoint_config.get('save_top_k', 3),
            save_last=True,  # Always save last checkpoint
            verbose=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False  # Save only after validation to avoid broadcast issues
        )
        print(f"Checkpoint saving enabled. Checkpoints will be saved to: {checkpoint_dir}")
    else:
        print("Checkpoint saving disabled. No model weights will be saved.")
    
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto' if torch.cuda.is_available() else 'cpu',
        devices='auto',  # Use all available GPUs
        strategy='ddp_spawn' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        enable_progress_bar=True,
        enable_model_summary=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision on GPU
        gradient_clip_val=config.get('training', {}).get('gradient_clip_val', 0),
        check_val_every_n_epoch=config.get('training', {}).get('check_val_every_n_epoch', 1)
    )
    
    # Start training
    print("Starting training...")
    # Pass checkpoint path to fit() method instead of Trainer constructor
    # This is the correct way in newer PyTorch Lightning versions
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume if args.resume else None)
    
    print("Training completed!")
    if checkpoint_callback is not None:
        print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print("No checkpoints were saved (checkpoint saving is disabled).")


if __name__ == '__main__':
    main()

