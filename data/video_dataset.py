"""
Video Dataset Module for VideoMAE Training

This module implements a PyTorch Dataset class for loading video data
from MP4 files. It handles frame sampling, temporal downsampling, and
data augmentation for self-supervised video representation learning.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import torchvision.transforms as transforms
from torchvision.io import read_video
import numpy as np


class VideoDataset(Dataset):
    """
    Dataset class for loading videos from MP4 files.
    
    This dataset:
    - Loads video paths from a CSV file
    - Randomly samples 32 consecutive frames from each video
    - Temporally downsamples with stride 2 to get 16 frames
    - Assumes videos are already 224x224x3 (no resizing needed)
    - Returns normalized video tensors for training
    
    Args:
        csv_file (str): Path to CSV file containing video file paths (one per line)
        num_frames (int): Number of frames to sample (default: 32)
        temporal_stride (int): Stride for temporal downsampling (default: 2)
        transform (Optional[transforms.Compose]): Optional transforms to apply
    """
    
    def __init__(
        self,
        csv_file: str,
        num_frames: int = 32,
        temporal_stride: int = 2,
        transform: Optional[transforms.Compose] = None
    ):
        self.csv_file = csv_file
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.transform = transform
        
        # Read video paths from CSV file
        # CSV format: one path per line (no header)
        try:
            df = pd.read_csv(csv_file, header=None, names=['path'])
            self.video_paths = df['path'].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_file}: {e}")
        
        # Filter out non-existent files
        self.video_paths = [p for p in self.video_paths if os.path.exists(p)]
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No valid video files found in {csv_file}")
        
        print(f"Loaded {len(self.video_paths)} video paths from {csv_file}")
    
    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and process a video sample.
        
        Args:
            idx (int): Index of the video to load
            
        Returns:
            torch.Tensor: Video tensor of shape (T, H, W, C) where:
                - T: number of frames after downsampling (16 by default)
                - H, W: height and width (224x224)
                - C: channels (3 for RGB)
        """
        video_path = self.video_paths[idx]
        
        try:
            # Load video using torchvision
            # Returns: (T, H, W, C) tensor with values in [0, 255]
            video, audio, info = read_video(
                video_path,
                pts_unit='sec',
                output_format='TCHW'  # Time, Channels, Height, Width
            )
            
            # Convert to (T, H, W, C) format for easier processing
            video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            
            num_frames_total = video.shape[0]
            
            # Ensure we have enough frames
            if num_frames_total < self.num_frames:
                # If video is shorter than required, pad by repeating last frame
                padding = self.num_frames - num_frames_total
                last_frame = video[-1:].repeat(padding, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
                num_frames_total = video.shape[0]
            
            # Randomly sample starting frame for 32 consecutive frames
            max_start = num_frames_total - self.num_frames
            if max_start < 0:
                start_frame = 0
            else:
                start_frame = torch.randint(0, max_start + 1, (1,)).item()
            
            # Extract 32 consecutive frames
            video = video[start_frame:start_frame + self.num_frames]
            
            # Temporally downsample with stride 2 to get 16 frames
            video = video[::self.temporal_stride]  # Shape: (16, 224, 224, 3)
            
            # Normalize to [0, 1] range
            video = video.float() / 255.0
            
            # Apply transforms if provided (e.g., data augmentation)
            if self.transform is not None:
                # Transforms expect (T, H, W, C) or (H, W, C) format
                video = self.transform(video)
            
            # Ensure output is in (T, H, W, C) format
            # T=16, H=224, W=224, C=3
            return video
            
        except Exception as e:
            # If video loading fails, return a zero tensor
            print(f"Warning: Failed to load video {video_path}: {e}")
            # Return a dummy video with correct shape
            return torch.zeros(
                (self.num_frames // self.temporal_stride, 224, 224, 3),
                dtype=torch.float32
            )


def get_video_transforms(
    mode: str = 'train',
    normalize: bool = True
) -> transforms.Compose:
    """
    Get video transforms for data augmentation.
    
    Args:
        mode (str): 'train' for training augmentations, 'val' for validation
        normalize (bool): Whether to normalize to ImageNet statistics
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    if mode == 'train':
        # Training augmentations
        transform_list = [
            # Random horizontal flip (applied frame-wise)
            # Note: For video, we might want to apply same flip to all frames
            # This is a simple implementation - can be enhanced
        ]
    else:
        # Validation: no augmentation
        transform_list = []
    
    if normalize:
        # ImageNet normalization statistics
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list) if transform_list else None

