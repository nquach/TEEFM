"""
PyTorch Lightning DataModule for VideoMAE Finetuning

This module provides a LightningDataModule that encapsulates dataset and dataloader
creation for VideoMAE finetuning with optimized labeled litdata datasets.
"""

import os
import torch
import pytorch_lightning as pl
from litdata import StreamingDataLoader

from .optimized_labeled_video_dataset import OptimizedLabeledVideoDataset
from torchvision.transforms.v2 import Normalize, RandomResizedCrop, Compose, CenterCrop, Resize

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


class _LabeledDatasetTransformWrapper(torch.utils.data.Dataset):
    """Wraps a dataset and applies a transform to the first element (video) of each sample; label unchanged."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        video, label = self.base_dataset[idx]
        return self.transform(video), label


class FinetuningVideoDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VideoMAE finetuning.
    
    This DataModule handles:
    - Transform creation for classification augmentations (no masking)
    - Dataset creation from optimized labeled litdata datasets
    - DataLoader creation for train/val/test splits with proper DDP configuration
    
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
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
    
    def setup(self, stage=None):
        """
        Create datasets and transforms.
        
        Called automatically by Lightning before training starts.
        In DDP mode, this is called once per process.
        
        Args:
            stage (str, optional): Stage of training ('fit', 'validate', 'test', 'predict')
        """
        # Get normalization values
        normalize_mean = self.data_config.get('normalize_mean', [0.485, 0.456, 0.406])  # ImageNet default
        normalize_std = self.data_config.get('normalize_std', [0.229, 0.224, 0.225])  # ImageNet default
        
        input_size = self.model_config.get('input_size', 224)
        crop_scale = self.model_config.get('crop_scale', [0.75, 1.0])
        crop_aspect_ratio = self.model_config.get('crop_aspect_ratio', [0.8, 1.2])
        
        # Create transforms
        # Training: RandomResizedCrop + Normalize
        self.train_transform = Compose([
            RandomResizedCrop(size=input_size, scale=tuple(crop_scale), ratio=tuple(crop_aspect_ratio)),
            Normalize(normalize_mean, normalize_std)
        ])
        
        # Validation/Test: CenterCrop or Resize + Normalize
        # For now, use CenterCrop (can be made configurable)
        self.val_transform = Compose([
            Resize(int(input_size * 1.14)),  # Slightly larger for center crop
            CenterCrop(input_size),
            Normalize(normalize_mean, normalize_std)
        ])
        self.test_transform = self.val_transform  # Same as validation
        
        cache_dir = self.data_config.get('cache_dir')
        safe_makedir(cache_dir)
        
        train_data_dir = self.data_config.get('train_optimized_dir')
        if train_data_dir is None:
            raise ValueError("train_optimized_dir is not specified in config")
        val_data_dir = self.data_config.get('val_optimized_dir')
        test_data_dir = self.data_config.get('test_optimized_dir')
        split_ratios = self.data_config.get('train_val_test_split')
        seed = self.training_config.get('seed', 0)
        storage_opts = custom_storage_options if self.data_config.get('cloud_type', 's3_public') == 's3_public' else None
        
        use_split = val_data_dir is None and test_data_dir is None and split_ratios is not None
        
        if use_split:
            # Split single dataset at train_optimized_dir into train/val/test by user-defined ratios
            if len(split_ratios) != 3:
                raise ValueError("train_val_test_split must be a list of three numbers [train_ratio, val_ratio, test_ratio]")
            total = sum(split_ratios)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"train_val_test_split must sum to 1.0, got {total}")
            print("Splitting train_optimized_dir into train/val/test by train_val_test_split")
            full_dataset = OptimizedLabeledVideoDataset(
                data_dir=train_data_dir,
                frames_to_sample=self.training_config.get('frames_to_sample', 16),
                temporal_stride=self.training_config.get('temporal_stride', 1),
                subset_ratio=None,
                seed=seed,
                transform=None,
                cache_dir=cache_dir,
                max_cache_size=self.data_config.get('max_cache_size', '50GB'),
                drop_last=False,
                storage_options=storage_opts
            )
            n = len(full_dataset)
            train_len = int(n * split_ratios[0])
            val_len = int(n * split_ratios[1])
            test_len = n - train_len - val_len
            train_subset, val_subset, test_subset = torch.utils.data.random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(seed)
            )
            self.train_dataset = _LabeledDatasetTransformWrapper(train_subset, self.train_transform)
            self.val_dataset = _LabeledDatasetTransformWrapper(val_subset, self.val_transform)
            self.test_dataset = _LabeledDatasetTransformWrapper(test_subset, self.test_transform)
            print(f'Split: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}')
        else:
            # Use separate dirs: train from train_optimized_dir, val/test from their dirs if provided
            if stage == 'fit' or stage is None:
                print("Using optimized litdata datasets for training")
                self.train_dataset = OptimizedLabeledVideoDataset(
                    data_dir=train_data_dir,
                    frames_to_sample=self.training_config.get('frames_to_sample', 16),
                    temporal_stride=self.training_config.get('temporal_stride', 1),
                    subset_ratio=self.data_config.get('subset_ratio'),
                    seed=seed,
                    transform=self.train_transform,
                    cache_dir=cache_dir,
                    max_cache_size=self.data_config.get('max_cache_size', '50GB'),
                    drop_last=True,
                    storage_options=storage_opts
                )
                print(f'Created optimized training dataset from {train_data_dir} of length {len(self.train_dataset)}')
                if val_data_dir is not None:
                    print("Using optimized litdata datasets for validation")
                    self.val_dataset = OptimizedLabeledVideoDataset(
                        data_dir=val_data_dir,
                        frames_to_sample=self.training_config.get('frames_to_sample', 16),
                        temporal_stride=self.training_config.get('temporal_stride', 1),
                        subset_ratio=None,
                        seed=seed,
                        transform=self.val_transform,
                        cache_dir=cache_dir,
                        max_cache_size=self.data_config.get('max_cache_size', '50GB'),
                        drop_last=False,
                        storage_options=storage_opts
                    )
                    print(f'Created optimized validation dataset from {val_data_dir} of length {len(self.val_dataset)}')
            if stage == 'test' or stage is None:
                if test_data_dir is not None:
                    print("Using optimized litdata datasets for testing")
                    self.test_dataset = OptimizedLabeledVideoDataset(
                        data_dir=test_data_dir,
                        frames_to_sample=self.training_config.get('frames_to_sample', 16),
                        temporal_stride=self.training_config.get('temporal_stride', 1),
                        subset_ratio=None,
                        seed=seed,
                        transform=self.test_transform,
                        cache_dir=cache_dir,
                        max_cache_size=self.data_config.get('max_cache_size', '50GB'),
                        drop_last=False,
                        storage_options=storage_opts
                    )
                    print(f'Created optimized test dataset from {test_data_dir} of length {len(self.test_dataset)}')
    
    def train_dataloader(self):
        """
        Create and return training dataloader.
        
        Returns:
            StreamingDataLoader: Training dataloader
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup('fit') first.")
        
        batch_size = self.training_config['batch_size']
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
    
    def val_dataloader(self):
        """
        Create and return validation dataloader.
        
        Returns:
            StreamingDataLoader: Validation dataloader or None if no validation dataset
        """
        if self.val_dataset is None:
            return None
        
        batch_size = self.training_config.get('val_batch_size', self.training_config['batch_size'])
        is_ddp = torch.cuda.device_count() > 1
        
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
        Create and return test dataloader.
        
        Returns:
            StreamingDataLoader: Test dataloader or None if no test dataset
        """
        if self.test_dataset is None:
            return None
        
        batch_size = self.training_config.get('test_batch_size', self.training_config['batch_size'])
        is_ddp = torch.cuda.device_count() > 1
        
        test_loader = StreamingDataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
            persistent_workers=False,
        )
        
        return test_loader

