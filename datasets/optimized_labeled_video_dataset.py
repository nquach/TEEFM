"""
Optimized Labeled Video Dataset using LitData StreamingDataset

This module provides a subclass of litdata's StreamingDataset for loading
optimized labeled video datasets created with litdata.optimize(), with custom processing
for frame sampling, temporal downsampling, and classification transforms.
"""

import os
import random
import warnings
import torch
from litdata import StreamingDataset, StreamingDataLoader
import numpy as np
from litdata.streaming.cache import Dir

# Suppress torchvision warnings (in case torchvision is used indirectly)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')

class OptimizedLabeledVideoDataset(StreamingDataset):
    """
    Optimized labeled video dataset using LitData StreamingDataset.
    
    This dataset inherits from StreamingDataset and loads videos with labels from an optimized
    dataset created with litdata.optimize(). It applies frame sampling and temporal downsampling
    similar to OptimizedVideoDataset but returns (video, label) tuples for classification.
    
    Args:
        data_dir (str): Path to the optimized dataset directory
        frames_to_sample (int): Number of consecutive frames to sample (default: 16)
        temporal_stride (int): Stride for temporal downsampling (default: 1)
        subset_ratio (float, optional): Ratio of dataset to use (0 < ratio <= 1).
                                       If None, uses full dataset.
        seed (int, optional): Random seed for subset sampling reproducibility
        transform (callable, optional): Optional transform to apply to video frames
        cache_dir (str, optional): Path to cache directory
        max_cache_size (str): Maximum cache size (default: '50GB')
        drop_last (bool): Whether to drop last incomplete batch (default: False)
        storage_options (dict, optional): Storage options for S3/remote datasets
    """
    
    def __init__(
        self,
        data_dir,
        frames_to_sample=16,
        temporal_stride=1,
        subset_ratio=None,
        seed=None,
        transform=None,
        cache_dir=None,
        max_cache_size='50GB',
        drop_last=False,
        storage_options=None
    ):
        # Initialize parent StreamingDataset class
        try:
            if subset_ratio is not None:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir), 
                    transform=None, 
                    subsample=subset_ratio, 
                    drop_last=drop_last, 
                    max_cache_size=max_cache_size, 
                    storage_options=storage_options
                )
            else:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir), 
                    transform=None, 
                    drop_last=drop_last, 
                    max_cache_size=max_cache_size, 
                    storage_options=storage_options
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize StreamingDataset from {data_dir}. "
                f"Error: {e}"
            )
        
        # Store custom processing parameters
        self.frames_to_sample = frames_to_sample
        self.temporal_stride = temporal_stride
        self.custom_transform = transform
        print(f"Loaded optimized labeled dataset from {data_dir}")
    
    def __getitem__(self, idx):
        """
        Load and process a video sample with label from the optimized dataset.
        
        Args:
            idx (int or ChunkedIndex): Index of the video to load (ChunkedIndex when using StreamingDataLoader)
        
        Returns:
            tuple: (video_tensor, label) where:
                - video_tensor: Tensor of shape [C, T, H, W] for classification/regression
                - label: Integer class label (classification) or float value (regression)
        """
        data = super().__getitem__(idx)
        
        video = data['video']  # TCHW format
        
        # Extract label from data and preserve original type (int for classification, float for regression)
        if 'label' in data:
            label = data['label']
            # Convert to Python scalar while preserving type (int or float)
            if isinstance(label, torch.Tensor):
                label = label.item()  # item() preserves type (int or float)
            elif isinstance(label, np.ndarray):
                label = label.item()  # item() preserves type (int or float)
            elif isinstance(label, (int, float)):
                label = label  # Already a scalar, preserve type
            else:
                # Try to convert, preserving numeric type
                try:
                    label = float(label)
                    # If it's a whole number, convert to int for classification compatibility
                    if label.is_integer():
                        label = int(label)
                except (ValueError, TypeError):
                    raise ValueError(f"Label must be numeric (int or float), got {type(label)}: {label}")
        else:
            raise KeyError(f"Label not found in dataset item. Available keys: {data.keys()}")
        
        num_frames = video.shape[0]
        
        # Randomly sample consecutive frames
        max_start = num_frames - self.frames_to_sample
        if max_start <= 0:
            start_frame = 0
        else:
            start_frame = random.randint(0, max_start)
        
        # Extract consecutive frames
        sampled_frames = video[start_frame:start_frame + self.frames_to_sample]
        
        # Temporal downsampling with stride
        video_tensor = sampled_frames[::self.temporal_stride].float()
        max_val = video_tensor.max().item() if video_tensor.numel() > 0 else 0.0
        if max_val > 1.0:
            video_tensor = video_tensor / 255.0
        
        # Apply transform if provided (for classification augmentations, normalization, etc.)
        if self.custom_transform is not None:
            # Transform expects [C, T, H, W] format
            video_tensor = self.custom_transform(video_tensor.permute(3, 0, 1, 2))  # THWC -> CTHW
            return video_tensor, label
        
        # Return video in [C, T, H, W] format and label
        return video_tensor.permute(3, 0, 1, 2), label  # THWC -> CTHW

