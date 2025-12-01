"""
Video Dataset Module for VideoMAE Training

This module implements a PyTorch Dataset class for loading video data
from MP4 files using torchcodec for efficient video decoding.
It handles frame sampling, temporal downsampling, and data augmentation.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random

try:
    from torchcodec import read_video
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
    from torchvision.io import read_video
    print("Warning: torchcodec not available. Install with: pip install torchcodec. Falling back to torchvision.")


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
        cropped_video = video[:, top:top+crop_size, left:left+crop_size, :]
        
        # Resize all frames to target size
        # Convert to (T, C, H, W) for easier processing
        cropped_video = cropped_video.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Resize all frames at once for efficiency
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


class VideoDataset(Dataset):
    """
    Dataset class for loading videos from MP4 files using torchcodec.
    
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
        cache_video_metadata (bool): Whether to cache video metadata
    """
    
    def __init__(
        self,
        csv_file: str,
        num_frames: int = 32,
        temporal_stride: int = 2,
        transform: Optional[transforms.Compose] = None,
        cache_video_metadata: bool = True
    ):
        self.csv_file = csv_file
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.transform = transform
        self.cache_video_metadata = cache_video_metadata
        
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
        
        # Cache for video metadata (number of frames)
        self._video_metadata_cache: Dict[str, int] = {}
        
        print(f"Loaded {len(self.video_paths)} video paths from {csv_file}")
    
    def _get_video_frame_count(self, video_path: str) -> int:
        """
        Get the number of frames in a video file.
        Uses caching to avoid repeated reads.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Number of frames in the video
        """
        if video_path in self._video_metadata_cache:
            return self._video_metadata_cache[video_path]
        
        try:
            if TORCHCODEC_AVAILABLE:
                # Use torchcodec to get frame count
                video, _, _ = read_video(video_path)
                frame_count = video.shape[0] if video is not None else 0
            else:
                # Fallback to torchvision
                video, _, _ = read_video(video_path, pts_unit='sec', output_format='TCHW')
                frame_count = video.shape[0]
            
            self._video_metadata_cache[video_path] = frame_count
            return frame_count
        except Exception:
            # If we can't read the video, return a default
            print(f"Warning: Could not read video {video_path}, using default frame count")
            return self.num_frames
    
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
            # Get video frame count (cached)
            num_frames_total = self._get_video_frame_count(video_path)
            
            # Randomly sample starting frame for 32 consecutive frames
            max_start = max(0, num_frames_total - self.num_frames)
            if max_start <= 0:
                start_frame = 0
                end_frame = min(num_frames_total, self.num_frames)
            else:
                start_frame = torch.randint(0, max_start + 1, (1,)).item()
                end_frame = start_frame + self.num_frames
            
            # Load video frames using torchcodec
            if TORCHCODEC_AVAILABLE:
                # Use torchcodec read_video
                video, _, _ = read_video(video_path)
                
                if video is None or video.shape[0] == 0:
                    raise ValueError(f"Failed to load video: {video_path}")
                
                # Extract the frame range we need
                if num_frames_total < self.num_frames:
                    # If video is shorter, pad by repeating last frame
                    padding = self.num_frames - num_frames_total
                    last_frame = video[-1:].repeat(padding, 1, 1, 1)
                    video = torch.cat([video, last_frame], dim=0)
                    start_frame = 0
                    end_frame = self.num_frames
                else:
                    video = video[start_frame:end_frame]
                
                # Ensure format is (T, H, W, C)
                if video.dim() == 4 and video.shape[1] == 3:
                    # Convert from (T, C, H, W) to (T, H, W, C)
                    video = video.permute(0, 2, 3, 1)
            else:
                # Fallback to torchvision
                video, _, _ = read_video(
                    video_path,
                    pts_unit='sec',
                    output_format='TCHW'
                )
                
                if num_frames_total < self.num_frames:
                    padding = self.num_frames - num_frames_total
                    last_frame = video[-1:].repeat(padding, 1, 1, 1)
                    video = torch.cat([video, last_frame], dim=0)
                    start_frame = 0
                    end_frame = self.num_frames
                else:
                    video = video[start_frame:end_frame]
                
                video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            
            # Temporally downsample with stride 2 to get 16 frames
            video = video[::self.temporal_stride]  # Shape: (16, 224, 224, 3)
            
            # Normalize to [0, 1] range
            video = video.float() / 255.0
            
            # Apply transforms if provided (e.g., data augmentation)
            if self.transform is not None:
                video = self.transform(video)
            
            # Ensure output is in (T, H, W, C) format
            return video
            
        except Exception as e:
            # If video loading fails, return a zero tensor
            print(f"Warning: Failed to load video {video_path}: {e}")
            return torch.zeros(
                (self.num_frames // self.temporal_stride, 224, 224, 3),
                dtype=torch.float32
            )


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
    
    if mode == 'train' and use_multiscale_crop:
        # Multiscale cropping with temporal coherence
        transform_list.append(
            MultiscaleCrop(
                target_size=224,
                scale_range=crop_scale_range
            )
        )
    
    if normalize:
        # ImageNet normalization statistics
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list) if transform_list else None

