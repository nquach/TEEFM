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
import torchvision.transforms.functional as F
from torchvision.io import read_video
import numpy as np
import random


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


class MultiscaleCrop:
    """
    Multiscale cropping augmentation for videos.
    
    Applies the same random crop (location and scale) to all frames,
    preserving temporal coherence. Crops a random region and resizes
    it back to the target size.
    
    Args:
        target_size (int): Target size after cropping and resizing (default: 224)
        scale_range (Tuple[float, float]): Range of scale factors for cropping
            (default: (0.8, 1.0)). Crop size = scale * target_size
    """
    
    def __init__(
        self,
        target_size: int = 224,
        scale_range: Tuple[float, float] = (0.8, 1.0)
    ):
        self.target_size = target_size
        self.scale_range = scale_range
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply multiscale crop to video.
        
        Applies the same random crop (location and scale) to all frames,
        preserving temporal coherence.
        
        Args:
            video: Video tensor of shape (T, H, W, C) with values in [0, 1]
            
        Returns:
            Cropped and resized video tensor of shape (T, target_size, target_size, C)
        """
        T, H, W, C = video.shape
        
        # Randomly select scale factor
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Compute crop size
        crop_size = int(scale * self.target_size)
        crop_size = max(crop_size, self.target_size // 2)  # Ensure minimum size
        crop_size = min(crop_size, min(H, W))  # Don't exceed original size
        
        # Randomly select crop location (top-left corner)
        max_top = max(0, H - crop_size)
        max_left = max(0, W - crop_size)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        
        # Crop all frames with the same parameters
        # video: (T, H, W, C)
        cropped_video = video[:, top:top+crop_size, left:left+crop_size, :]
        
        # Resize all frames to target size
        # Convert to (T, C, H, W) for easier processing
        cropped_video = cropped_video.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Resize all frames at once for efficiency
        # Reshape to (T*C, 1, H, W) for batch resize
        T_orig, C_orig, H_orig, W_orig = cropped_video.shape
        cropped_flat = cropped_video.reshape(T_orig * C_orig, 1, H_orig, W_orig)
        
        # Resize: (T*C, 1, H, W) -> (T*C, 1, target_size, target_size)
        resized_flat = F.resize(
            cropped_flat,
            size=(self.target_size, self.target_size),
            interpolation=F.InterpolationMode.BILINEAR
        )
        
        # Reshape back: (T*C, 1, target_size, target_size) -> (T, C, target_size, target_size)
        resized = resized_flat.reshape(T_orig, C_orig, self.target_size, self.target_size)
        
        # Convert back to (T, H, W, C)
        resized = resized.permute(0, 2, 3, 1)
        
        return resized


def get_video_transforms(
    mode: str = 'train',
    normalize: bool = True,
    use_multiscale_crop: bool = True,
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)
) -> Optional[transforms.Compose]:
    """
    Get video transforms for data augmentation.
    
    Args:
        mode (str): 'train' for training augmentations, 'val' for validation
        normalize (bool): Whether to normalize to ImageNet statistics
        use_multiscale_crop (bool): Whether to use multiscale cropping (default: True)
        crop_scale_range (Tuple[float, float]): Scale range for multiscale cropping
        
    Returns:
        transforms.Compose: Composition of transforms, or None if no transforms
    """
    transform_list = []
    
    if mode == 'train':
        # Training augmentations
        if use_multiscale_crop:
            # Multiscale cropping with temporal coherence
            # Same crop location and scale applied to all frames
            transform_list.append(
                MultiscaleCrop(
                    target_size=224,
                    scale_range=crop_scale_range
                )
            )
    else:
        # Validation: no augmentation
        pass
    
    if normalize:
        # ImageNet normalization statistics
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list) if transform_list else None

