"""
Configuration file for VideoMAE training with EVEREST masking.

This module provides a configuration class and default parameters for easy
hyperparameter tuning. Modify the default values or create new configurations
as needed.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoMAEConfig:
    """
    Configuration class for VideoMAE training.
    
    This class contains all hyperparameters needed for training.
    Modify values here or override in the training script.
    """
    
    # Data parameters
    csv_file: str = 'mp4_paths.csv'
    val_csv_file: str = 'val500_2023-2024.csv'  # Validation set CSV file
    dataset_subset_ratio: float = 1.0  # Ratio of dataset to use (1.0 = use all, 0.5 = use 50%, off by default)
    # LitData optimization (for faster data loading)
    use_litdata: bool = True  # Use litdata streaming (requires preprocessing with optimize_video_dataset)
    litdata_output_dir: str = 'litdata_optimized'  # Directory for optimized data chunks
    litdata_val_output_dir: str = 'litdata_optimized_val'  # Directory for optimized validation data
    num_frames: int = 32  # Frames to sample before downsampling
    temporal_stride: int = 2  # Stride for temporal downsampling
    final_num_frames: int = 16  # Final number of frames after downsampling
    img_size: int = 224
    batch_size: int = 8
    num_workers: int = os.cpu_count() or 4  # Use all available CPUs, fallback to 4 if count unavailable
    pin_memory: bool = True
    persistent_workers: bool = True  # Keep workers alive between epochs (faster)
    prefetch_factor: int = 2  # Number of batches to prefetch per worker
    
    # Model parameters
    backbone: str = 'ViT-S'  # Options: 'ViT-S', 'ViT-B', 'ViT-L'
    patch_size: int = 16
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    dropout: float = 0.0
    drop_path: float = 0.0
    
    # Masking parameters (EVEREST)
    mask_ratio: float = 0.9  # Ratio of patches to mask (high ratio as in VideoMAE)
    motion_weight: float = 0.7  # Weight for motion-based token selection
    random_ratio: float = 0.1  # Small random component for diversity
    
    # Training parameters
    learning_rate: float = 1.5e-4  # Base learning rate
    weight_decay: float = 0.05
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2
    warmup_epochs: int = 40
    max_epochs: int = 800
    norm_pix_loss: bool = True  # Normalize patches before loss computation
    
    # Training setup
    accelerator: str = 'gpu'  # 'gpu', 'cpu', or 'auto'
    devices: Optional[int] = None  # None = use all available GPUs
    precision: str = '16-mixed'  # Mixed precision training: '16-mixed', '32', 'bf16-mixed'
    gradient_clip_val: Optional[float] = None  # Gradient clipping value
    accumulate_grad_batches: int = 1  # Gradient accumulation
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    log_gradient_norm: bool = False  # Log L2 norm of full loss gradient (off by default)
    # Validation frequency controls (work together):
    # - check_val_every_n_epoch: Controls which epochs to validate (epoch-level frequency)
    #   Example: 10 means validate at the end of every 10th epoch
    # - val_check_interval: Controls how often to validate WITHIN an epoch
    #   - None: Only validate at epoch boundaries (respects check_val_every_n_epoch)
    #   - Float (0.0-1.0): Fraction of epoch (e.g., 0.5 = twice per epoch)
    #   - Integer: Number of batches (e.g., 100 = every 100 batches)
    # Example: check_val_every_n_epoch=10, val_check_interval=None -> validate every 10 epochs
    # Example: check_val_every_n_epoch=1, val_check_interval=0.5 -> validate twice per epoch
    val_check_interval: Optional[float] = None  # None = only at epoch boundaries
    check_val_every_n_epoch: int = 10  # Validate at the end of every N epochs
    enable_checkpointing: bool = True
    checkpoint_dir: str = 'checkpoints'  # Directory to save checkpoints
    checkpoint_filename_prefix: str = 'videomae'  # Prefix for checkpoint filenames
    # Final filename format: {prefix}-{epoch:02d}-{monitor_metric:.2f}.ckpt
    monitor_metric: str = 'train/loss'
    mode: str = 'min'  # 'min' or 'max' for checkpoint saving
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Pretrained weights (for initializing model, not resuming training)
    pretrained_checkpoint: Optional[str] = None  # Path to pretrained model weights (.ckpt, .pth, or .pt)
    load_pretrained_strict: bool = True  # If False, allows partial weight loading when architectures don't match exactly
    
    # Other
    seed: int = 42
    deterministic: bool = False  # Set to True for reproducibility (slower)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.backbone in ['ViT-S', 'ViT-B', 'ViT-L'], \
            f"Invalid backbone: {self.backbone}"
        assert 0 < self.mask_ratio < 1, \
            f"mask_ratio must be between 0 and 1, got {self.mask_ratio}"
        assert 0 <= self.motion_weight <= 1, \
            f"motion_weight must be between 0 and 1, got {self.motion_weight}"
        assert 0 < self.dataset_subset_ratio <= 1, \
            f"dataset_subset_ratio must be between 0 and 1, got {self.dataset_subset_ratio}"
        assert self.final_num_frames == self.num_frames // self.temporal_stride, \
            f"final_num_frames ({self.final_num_frames}) should equal " \
            f"num_frames // temporal_stride ({self.num_frames // self.temporal_stride})"


# Predefined configurations for different scenarios
def get_vit_s_config() -> VideoMAEConfig:
    """Get configuration for ViT-S backbone (default, fastest)."""
    return VideoMAEConfig(
        backbone='ViT-S',
        batch_size=16,  # Can use larger batch with smaller model
        learning_rate=1.5e-4
    )


def get_vit_b_config() -> VideoMAEConfig:
    """Get configuration for ViT-B backbone (balanced)."""
    return VideoMAEConfig(
        backbone='ViT-B',
        batch_size=8,
        learning_rate=1.5e-4
    )


def get_vit_l_config() -> VideoMAEConfig:
    """Get configuration for ViT-L backbone (largest, best performance)."""
    return VideoMAEConfig(
        backbone='ViT-L',
        batch_size=4,  # Smaller batch due to memory constraints
        learning_rate=1.0e-4,  # Slightly lower LR for larger model
        accumulate_grad_batches=2  # Compensate for smaller batch
    )


def get_debug_config() -> VideoMAEConfig:
    """Get configuration for debugging (small model, fast training)."""
    return VideoMAEConfig(
        backbone='ViT-S',
        batch_size=4,
        max_epochs=5,
        num_workers=2,
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )

