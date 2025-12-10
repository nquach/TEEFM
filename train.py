"""
Main Training Script for EVEREST VideoMAE

This script loads configuration from a YAML file, sets up datasets, data loaders,
PyTorch Lightning model, and trainer for training VideoMAE models.
"""

import os
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


def get_process_rank():
    """
    Get the current process rank for distributed training.
    
    Returns:
        int: Process rank (0 for main process, 1+ for other processes in DDP)
    """
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
    return rank


class RankAwareModelCheckpoint(ModelCheckpoint):
    """
    A ModelCheckpoint wrapper that only executes on rank 0 to prevent DDP broadcast errors.
    
    This class wraps PyTorch Lightning's ModelCheckpoint and adds rank checks to prevent
    execution on non-zero ranks, which can cause broadcast synchronization errors in DDP.
    
    All callback methods check the process rank and only execute on rank 0, returning
    immediately (no-op) on other ranks.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the rank-aware checkpoint callback."""
        super().__init__(*args, **kwargs)
        self._rank = get_process_rank()
    
    def __getstate__(self):
        """
        Custom pickling to preserve rank-aware behavior.
        
        This ensures that when PyTorch Lightning pickles/unpickles the callback
        (which can happen in DDP), our custom rank-aware behavior is preserved.
        """
        state = super().__getstate__()
        state['_rank'] = get_process_rank()
        return state
    
    def __setstate__(self, state):
        """
        Custom unpickling to preserve rank-aware behavior.
        
        This ensures that after unpickling, the callback still knows its rank
        and can prevent execution on non-zero ranks.
        """
        super().__setstate__(state)
        # Re-check rank after unpickling to ensure correctness
        self._rank = get_process_rank()
    
    def _is_rank_zero(self):
        """Check if current process is rank 0."""
        # Re-check rank in case it changed (shouldn't happen, but defensive)
        current_rank = get_process_rank()
        return current_rank == 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Override to only execute on rank 0."""
        if not self._is_rank_zero():
            return
        super().on_train_epoch_end(trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        """Override to only execute on rank 0."""
        if not self._is_rank_zero():
            return
        super().on_validation_end(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Override to only execute on rank 0."""
        if not self._is_rank_zero():
            return
        super().on_validation_epoch_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        """Override to only execute on rank 0."""
        if not self._is_rank_zero():
            return
        super().on_train_end(trainer, pl_module)
    
    def file_exists(self, filepath, trainer):
        """
        Override file_exists to prevent broadcast on non-zero ranks.
        
        This method is the source of the broadcast error, so we need to
        ensure it only executes on rank 0.
        """
        if not self._is_rank_zero():
            # Return False on non-zero ranks to prevent broadcast
            return False
        return super().file_exists(filepath, trainer)
    
    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        """
        Override to prevent top-k checkpoint saving from executing on non-zero ranks.
        
        This method is called during on_train_epoch_end and can cause broadcast errors
        if executed on non-zero ranks. This is the entry point that calls other
        checkpoint saving methods.
        """
        if not self._is_rank_zero():
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)
    
    def _save_monitor_checkpoint(self, trainer, monitor_candidates):
        """
        Override to prevent monitor-based checkpoint saving from executing on non-zero ranks.
        
        This method is called when monitoring a metric (like val_loss) and can cause
        broadcast errors if executed on non-zero ranks.
        """
        if not self._is_rank_zero():
            return
        super()._save_monitor_checkpoint(trainer, monitor_candidates)
    
    def _save_none_monitor_checkpoint(self, trainer, monitor_candidates):
        """
        Override to prevent checkpoint saving logic from executing on non-zero ranks.
        
        This method is called during on_train_epoch_end and can cause broadcast errors
        if executed on non-zero ranks.
        """
        if not self._is_rank_zero():
            return
        super()._save_none_monitor_checkpoint(trainer, monitor_candidates)


def setup_litdata_cache_dirs(config):
    """
    Setup unique cache directories per process to prevent race conditions in multi-GPU training.
    
    Each process (rank) gets its own cache directory using format: {base_dir}_rank{rank}_pid{pid}
    This prevents FileNotFoundError when multiple processes try to download/decompress chunks simultaneously.
    
    Args:
        config (dict): Configuration dictionary containing cache directory paths
    
    Returns:
        tuple: (train_cache_dir, val_cache_dir) - Unique cache directories for this process
    """
    data_config = config.get('data', {})
    
    # Get base cache directories from config
    base_train_cache = data_config.get('cache_dir', './output/cache')
    base_val_cache = data_config.get('val_cache_dir', './output/val_cache')
    
    # Detect process rank for multi-GPU training
    rank = get_process_rank()
    
    # Get process ID for additional uniqueness
    pid = os.getpid()
    
    # Create unique cache directories
    train_cache_dir = f"{base_train_cache}_rank{rank}_pid{pid}"
    val_cache_dir = f"{base_val_cache}_rank{rank}_pid{pid}"
    
    # Create cache directories with proper permissions
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)
    
    print(f"Process rank {rank}, PID {pid}: Using cache directories:")
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
        
        # Create validation dataset from optimized data
        val_dataset = OptimizedVideoDataset(
            data_dir=val_data_dir,
            frames_to_sample=training_config.get('frames_to_sample', 32),
            temporal_stride=training_config.get('temporal_stride', 2),
            subset_ratio=None,  # Always use full validation set
            seed=training_config.get('seed', 0),
            transform=transform,
            cache_dir=val_cache
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
    
    if use_optimized and LITDATA_AVAILABLE:
        # Use StreamingDataLoader for optimized datasets
        print("Using StreamingDataLoader for optimized datasets")
        
        train_loader = StreamingDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config.get('num_workers', 10),
            pin_memory=training_config.get('pin_memory', True),
            drop_last=False  # Cannot drop last batch otherwise wont start validation step
        )
        
        val_loader = StreamingDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=training_config.get('num_workers', 10),
            pin_memory=training_config.get('pin_memory', True),
            drop_last=False  # Keep all validation samples
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
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
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
    
    # Create Lightning module with comprehensive error handling and debugging
    print("Creating model...")
    try:
        rank_before_model = get_process_rank()
        print(f"[Rank {rank_before_model}] Starting model creation...")
        
        # Check if pretrained path exists (if specified)
        pretrained_path = config.get('model', {}).get('pretrained_path')
        if pretrained_path:
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"[Rank {rank_before_model}] Pretrained path not found: {pretrained_path}"
                )
            print(f"[Rank {rank_before_model}] Pretrained path verified: {pretrained_path}")
        
        # Create the model
        model = VideoMAELightningModule(config)
        print(f"[Rank {rank_before_model}] Model object created successfully")
        
        # Verify model was created correctly
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"[Rank {rank_before_model}] Total parameters: {total_params / 1e6:.2f}M")
        print(f"[Rank {rank_before_model}] Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Verify model has reasonable number of parameters
        # VideoMAE models should have at least 10M parameters (even for small models)
        if total_params < 10_000_000:  # Less than 10M is suspicious
            error_msg = (
                f"[Rank {rank_before_model}] ERROR: Model has only {total_params} parameters. "
                f"Expected at least ~10M parameters for VideoMAE. "
                f"Model creation may have failed or returned an incomplete model."
            )
            print(error_msg)
            
            # Try to get more information about the model
            try:
                model_type = type(model).__name__
                has_model_attr = hasattr(model, 'model')
                if has_model_attr:
                    model_attr_type = type(model.model).__name__
                    print(f"[Rank {rank_before_model}] Model type: {model_type}, model.model type: {model_attr_type}")
                else:
                    print(f"[Rank {rank_before_model}] Model type: {model_type}, no 'model' attribute found")
                
                # List model attributes
                print(f"[Rank {rank_before_model}] Model attributes: {list(model.__dict__.keys())[:10]}")
            except Exception as debug_e:
                print(f"[Rank {rank_before_model}] Could not inspect model: {debug_e}")
            
            raise RuntimeError(error_msg)
        
        # Additional verification: check if model has the expected structure
        if not hasattr(model, 'model'):
            raise RuntimeError(
                f"[Rank {rank_before_model}] Model missing 'model' attribute. "
                f"Expected VideoMAELightningModule to have a 'model' attribute."
            )
        
        print(f"[Rank {rank_before_model}] Model verification passed")
        
    except FileNotFoundError as e:
        rank_error = get_process_rank()
        print(f"[Rank {rank_error}] FileNotFoundError during model creation: {e}")
        import traceback
        traceback.print_exc()
        raise
    except RuntimeError as e:
        rank_error = get_process_rank()
        print(f"[Rank {rank_error}] RuntimeError during model creation: {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        rank_error = get_process_rank()
        print(f"[Rank {rank_error}] Unexpected error during model creation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Synchronization barrier: Ensure all ranks have completed model creation before proceeding
    # This helps catch any rank-specific issues early
    if torch.cuda.device_count() > 1:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                barrier_rank = get_process_rank()
                print(f"[Rank {barrier_rank}] Model creation barrier passed - all ranks synchronized")
        except Exception as barrier_e:
            barrier_rank = get_process_rank()
            print(f"[Rank {barrier_rank}] Warning: Could not synchronize ranks after model creation: {barrier_e}")
            # Don't raise - this is just a safeguard, not critical
    
    # Setup checkpoint callback (if enabled)
    checkpoint_config = config['checkpoint']
    checkpoint_enabled = checkpoint_config.get('enable', True)
    
    checkpoint_callback = None
    if checkpoint_enabled:
        # Get process rank to determine if we should enable checkpointing
        # In DDP, only rank 0 should save checkpoints to avoid synchronization issues
        rank = get_process_rank()
        
        # Only create checkpoint callback on rank 0 to prevent any callback operations on other ranks
        # This is critical: creating the callback on all ranks can cause DDP synchronization issues
        if rank == 0:
            checkpoint_dir = checkpoint_config['dir']
            checkpoint_prefix = checkpoint_config['prefix']
            
            # Create checkpoint directory
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = RankAwareModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"{checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode='min',  # Minimize validation loss
                save_top_k=checkpoint_config.get('save_top_k', 3),
                save_last=True,  # Always save last checkpoint
                save_on_train_epoch_end=False,  # Only save after validation, not during training epochs
                verbose=True
            )
            print(f"Checkpoint saving enabled on rank 0. Checkpoints will be saved to: {checkpoint_dir}")
        else:
            print(f"Checkpoint saving disabled on rank {rank} (only rank 0 saves checkpoints in DDP)")
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
    
    # Prepare callbacks list (only include checkpoint callback if enabled and on rank 0)
    # This is critical: only register callback on rank 0 to prevent DDP synchronization issues
    callbacks_list = []
    if checkpoint_callback is not None:
        final_rank_check = get_process_rank()
        if final_rank_check == 0:
            callbacks_list.append(checkpoint_callback)
            print(f"Checkpoint callback registered on rank {final_rank_check}")
        else:
            print(f"Checkpoint callback NOT registered on rank {final_rank_check} (only rank 0 saves checkpoints)")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',  # Use all available GPUs
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks_list if callbacks_list else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        enable_progress_bar=True,
        enable_model_summary=True,
        precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision on GPU
        gradient_clip_val=config.get('training', {}).get('gradient_clip_val', 0),
        check_val_every_n_epoch=config.get('training', {}).get('check_val_every_n_epoch', 1)
    )
    # Note: val_check_interval removed - validation will run at end of each epoch only
    # Setting val_check_interval=1.0 causes validation to run after every training step
    
    # Explicit safeguard: Ensure callback is None on non-zero ranks after trainer creation
    # This prevents any callback operations on non-zero ranks even if PyTorch Lightning
    # somehow creates or synchronizes callback instances
    if torch.cuda.device_count() > 1:  # Multi-GPU training
        final_safeguard_rank = get_process_rank()
        if final_safeguard_rank != 0:
            # Explicitly ensure no checkpoint callback exists on non-zero ranks
            if checkpoint_callback is not None:
                print(f"Warning: Checkpoint callback exists on rank {final_safeguard_rank}, removing it")
                checkpoint_callback = None
            # Ensure callbacks list is empty on non-zero ranks
            if hasattr(trainer, 'callbacks') and trainer.callbacks:
                # Filter out any ModelCheckpoint callbacks on non-zero ranks
                trainer.callbacks = [cb for cb in trainer.callbacks 
                                    if not isinstance(cb, (ModelCheckpoint, RankAwareModelCheckpoint))]
                print(f"Removed checkpoint callbacks from trainer on rank {final_safeguard_rank}")
    
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

