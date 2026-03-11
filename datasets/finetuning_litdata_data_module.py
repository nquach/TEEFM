"""
PyTorch Lightning DataModule for LitData classification finetuning.

Builds train/val/test datasets via build_litdata_finetune_datasets and
returns DataLoaders (Trainer handles DistributedSampler when using DDP).
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .litdata_labeled_dataset import LitDataLabeledDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE
from litdata import train_test_split, StreamingDataLoader
import os

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

class FinetuningLitDataDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for LitData (video, label) finetuning.
    Uses build_litdata_finetune_datasets for train/val/test split or separate dirs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.get("data", {})
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = self.data_config.get('num_classes', 6)

    def setup(self, stage=None):
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
            motion_centric_masking_ratio=self.model_config.get('motion_centric_masking_ratio', 0.4),
            frame_size=input_size,
            crop_scale=self.model_config.get('crop_scale', (0.75, 1.0)),
            crop_aspect_ratio=self.model_config.get('crop_aspect_ratio', (0.8, 1.2))
        )

        # Get training data directory
        if num_frames is None:
            num_frames = self.data_config.get("num_frames", 16)
        train_data_dir = self.data_config.get("train_optimized_dir")
        val_data_dir = self.data_config.get("val_optimized_dir")
        test_data_dir = self.data_config.get("test_optimized_dir")
        if not train_data_dir:
            raise ValueError("data.train_optimized_dir is required.")

        train_cache = self.data_config.get('train_cache_dir')
        val_cache = self.data_config.get('val_cache_dir')
        test_cache = self.data_config.get('test_cache_dir')
        if train_cache:
            safe_makedir(train_cache)
        if val_cache:
            safe_makedir(val_cache)
        if test_cache:
            safe_makedir(test_cache)
        
        # Create training dataset from optimized data
        self.full_dataset = LitDataLabeledDataset(
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
        
        if val_data_dir and test_data_dir:
            #Train dataset
            self.train_dataset = self.full_dataset
            #Val dataset
            self.val_dataset = LitDataLabeledDataset(
            data_dir=val_data_dir,
            frames_to_sample=self.training_config.get('frames_to_sample', 16),
            temporal_stride=self.training_config.get('temporal_stride', 1),
            subset_ratio=None,
            seed=self.training_config.get('seed', 0),
            transform=None,
            cache_dir=val_cache,
            max_cache_size=self.data_config.get('max_cache_size', '50GB'),
            drop_last=False,
            storage_options=custom_storage_options if self.data_config.get('cloud_type', 's3_public') == 's3_public' else None
            )
            #Test dataset
            self.test_dataset = LitDataLabeledDataset(
            data_dir=test_data_dir,
            frames_to_sample=self.training_config.get('frames_to_sample', 16),
            temporal_stride=self.training_config.get('temporal_stride', 1),
            subset_ratio=None,
            seed=self.training_config.get('seed', 0),
            transform=self.transform,
            cache_dir=test_cache,
            max_cache_size=self.data_config.get('max_cache_size', '50GB'),
            drop_last=False,
            storage_options=custom_storage_options if self.data_config.get('cloud_type', 's3_public') == 's3_public' else None
            )
            print(f'Created optimized training dataset from {train_data_dir} of length {len(self.train_dataset)}')
            print(f'Created optimized val dataset from {val_data_dir} of length {len(self.val_dataset)}')
            print(f'Created optimized test dataset from {test_data_dir} of length {len(self.test_dataset)}')
        else:
            ratios = self.data_config.get("train_val_test_split", [0.8, 0.1, 0.1])
            if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-3:
                raise ValueError(
                    "train_val_test_split must be [train_ratio, val_ratio, test_ratio] summing to 1.0"
                )
            self.train_dataset, self.val_dataset, self.test_dataset = train_test_split(self.full_dataset, splits=ratios)
            print(f'Created optimized training dataset from {train_data_dir} of length {len(self.train_dataset)}')
            print(f'Created optimized val dataset from {train_data_dir} of length {len(self.val_dataset)}')
            print(f'Created optimized test dataset from {train_data_dir} of length {len(self.test_dataset)}')

    def train_dataloader(self):
        """
        Create and return training dataloader.
        
        Returns:
            StreamingDataLoader: Training dataloader
        """
        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        batch_size = self.training_config['batch_size']
        
        train_loader = StreamingDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
            persistent_workers=False,
        )
        
        return train_loader

    def val_dataloader(self):
        """
        Create and return training dataloader.
        
        Returns:
            StreamingDataLoader: Val dataloader
        """
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not initialized. Call setup() first.")
        batch_size = self.training_config['batch_size']
        val_loader = StreamingDataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
            persistent_workers=False,
        )
        return val_loader

    def test_dataloader(self):
        """
        Create and return training dataloader.
        
        Returns:
            StreamingDataLoader: Val dataloader
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        batch_size = self.training_config['batch_size']
        test_loader = StreamingDataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
            persistent_workers=False,
        )
        return test_loader




