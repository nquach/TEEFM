"""
PyTorch Lightning DataModule for VideoMAE Training

This module provides a LightningDataModule that encapsulates dataset and dataloader
creation for VideoMAE training with optimized litdata datasets.
"""

import os
import torch
import pytorch_lightning as pl
from litdata import StreamingDataLoader

from .optimized_video_dataset import OptimizedVideoDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE

import botocore

custom_storage_options = {
    "config": botocore.config.Config(
        retries={"max_attempts": 1000, "mode": "adaptive"},
        signature_version=botocore.UNSIGNED,
    )
}

def safe_makedir(path):
    """Safely create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


class VideoDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VideoMAE training.
    
    This DataModule handles:
    - Transform creation for data augmentation and masking
    - Dataset creation from optimized litdata datasets
    - DataLoader creation with proper DDP configuration
    
    Args:
        config (dict): Configuration dictionary containing all training parameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.model_config = config['model']
        self.training_config = config['training']
        
        # Will be set in setup()
        self.train_dataset = None
        self.transform = None
    
    def setup(self, stage=None):
        """
        Create datasets and transforms.
        
        Called automatically by Lightning before training starts.
        In DDP mode, this is called once per process.
        
        Args:
            stage (str, optional): Stage of training ('fit', 'validate', 'test', 'predict')
        """
        # Get normalization values
        normalize_mean = self.data_config.get('normalize_mean', [0.117, 0.114, 0.113])
        normalize_std = self.data_config.get('normalize_std', [0.208, 0.204, 0.203])
        
        # Calculate window size for masking
        num_frames = self.training_config.get('num_frames', 16)
        input_size = self.model_config.get('input_size', 224)
        patch_size = self.model_config.get('patch_size', 16)
        
        # Window size: (frames, height_patches, width_patches)
        # After tubelet_size=2, temporal dimension is num_frames // 2
        window_size = (
            num_frames // 2,
            input_size // patch_size,
            input_size // patch_size
        )
        
        # Create transform
        self.transform = DataAugmentationForVideoMAE(
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            window_size=window_size,
            mask_type=self.model_config.get('mask_type', 'motion-centric'),
            mask_ratio=self.model_config.get('mask_ratio', 0.9),
            motion_centric_masking_ratio=self.model_config.get('motion_centric_masking_ratio', 0.7),
            frame_size=input_size,
            crop_scale=self.model_config.get('crop_scale', (0.75, 1.0)),
            crop_aspect_ratio=self.model_config.get('crop_aspect_ratio', (0.8, 1.2))
        )
        
        # Get training data directory
        train_data_dir = self.data_config.get('train_optimized_dir')
        
        if train_data_dir is None:
            raise ValueError(
                "train_optimized_dir is not specified. "
                "Please provide path to optimized dataset directory."
            )
        
        if stage == 'fit' or stage is None:
            print("Using optimized litdata datasets")
            
            train_cache = self.data_config.get('cache_dir')
            safe_makedir(train_cache)
            
            # Create training dataset from optimized data
            self.train_dataset = OptimizedVideoDataset(
                data_dir=train_data_dir,
                frames_to_sample=self.training_config.get('frames_to_sample', 16),
                temporal_stride=self.training_config.get('temporal_stride', 1),
                subset_ratio=self.data_config.get('subset_ratio'),
                seed=self.training_config.get('seed', 0),
                transform=self.transform,
                cache_dir=train_cache,
                max_cache_size=self.data_config.get('max_cache_size', '50GB'),
                drop_last=True,
                storage_options=custom_storage_options if self.data_config.get('cloud_type', 's3_public') == 's3_public' else None
            )
            print(f'Created optimized training dataset from {train_data_dir} of length {len(self.train_dataset)}')
    
    def train_dataloader(self):
        """
        Create and return training dataloader.
        
        Returns:
            StreamingDataLoader: Training dataloader
        """
        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        batch_size = self.training_config['batch_size']
        
        # Check if we're in DDP mode (will be True if multiple GPUs are available)
        # In DDP, persistent_workers=True helps maintain worker processes across epochs
        is_ddp = torch.cuda.device_count() > 1
        
        train_loader = StreamingDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
            persistent_workers=False,
        )
        
        return train_loader

