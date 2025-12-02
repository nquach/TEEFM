"""
Configuration module for VideoMAE training with EVEREST masking.

This module contains all hyperparameters and training settings that can be easily
tuned for different experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the VideoMAE model architecture."""
    # Backbone options: 'vit_s', 'vit_b', 'vit_l'
    backbone: str = 'vit_s'
    
    # Image and patch configuration
    img_size: int = 224  # Input image size (height and width)
    patch_size: int = 16  # Patch size for ViT
    num_frames: int = 16  # Number of frames after temporal downsampling
    
    # VideoMAE specific parameters
    tubelet_size: int = 2  # Temporal tubelet size
    mask_ratio: float = 0.9  # Masking ratio for EVEREST (90% is typical)
    
    # Pretrained weights path (None for random initialization)
    pretrained_weights: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Dataset paths
    train_csv: str = 'mp4_paths.csv'
    val_csv: str = 'val500_2023-2024.csv'
    
    # Frame sampling parameters
    sample_frames: int = 32  # Number of consecutive frames to sample
    temporal_stride: int = 2  # Temporal downsampling stride
    
    # DataLoader parameters
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    # Optimizer parameters (AdamWScheduleFree)
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    
    # Training parameters
    max_epochs: int = 100
    gradient_clip_val: Optional[float] = 1.0  # Gradient clipping value
    
    # Loss function
    loss_type: str = 'mse'  # 'mse' for mean squared error
    
    # Gradient norm tracking (off by default)
    track_gradient_norm: bool = False
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    # Checkpoint directory
    checkpoint_dir: str = './checkpoints'
    
    # Checkpoint file prefix
    checkpoint_prefix: str = 'videomae'
    
    # Checkpoint saving strategy
    save_top_k: int = 1  # Save top k checkpoints based on validation loss
    monitor: str = 'val_loss'  # Metric to monitor
    mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    
    # Save frequency
    save_every_n_epochs: int = 1
    save_last: bool = True  # Always save the last checkpoint


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning Trainer."""
    # GPU configuration
    gpus: int = 1  # Number of GPUs (0 for CPU)
    accelerator: str = 'gpu'  # 'gpu' or 'cpu'
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # Validate every N epochs (1.0 = every epoch)
    
    # Other settings
    deterministic: bool = False
    benchmark: bool = True  # cudnn benchmark for faster training
    precision: int = 16 if TrainingConfig().use_amp else 32  # 16 for mixed precision


@dataclass
class Config:
    """Main configuration class that aggregates all sub-configurations."""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    checkpoint: CheckpointConfig = None
    trainer: TrainerConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.checkpoint is None:
            self.checkpoint = CheckpointConfig()
        if self.trainer is None:
            self.trainer = TrainerConfig()
