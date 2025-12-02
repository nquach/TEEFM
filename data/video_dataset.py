"""
Video Dataset for VideoMAE training.

This module implements a PyTorch Dataset class for loading video data from CSV files.
Videos are processed to extract 32 consecutive frames, which are then temporally
downsampled with stride 2 to produce 16 frames for model input.
"""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as io


class VideoDataset(Dataset):
    """
    Dataset class for loading videos from CSV files.
    
    Each video is processed to:
    1. Randomly sample 32 consecutive frames
    2. Apply temporal downsampling with stride 2 → 16 frames
    3. Return tensor of shape [16, 3, 224, 224]
    
    Args:
        csv_file (str): Path to CSV file containing video paths (one per line)
        sample_frames (int): Number of consecutive frames to sample (default: 32)
        temporal_stride (int): Stride for temporal downsampling (default: 2)
        use_subset (bool): Whether to use a random subset of the dataset
        subset_ratio (float): Ratio of dataset to use if use_subset is True (0.0 to 1.0)
        seed (int): Random seed for reproducibility (default: None)
    """
    
    def __init__(
        self,
        csv_file: str,
        sample_frames: int = 32,
        temporal_stride: int = 2,
        use_subset: bool = False,
        subset_ratio: float = 0.1,
        seed: int = None
    ):
        """
        Initialize the VideoDataset.
        
        Args:
            csv_file: Path to CSV file with video paths
            sample_frames: Number of frames to sample from each video
            temporal_stride: Stride for temporal downsampling
            use_subset: Whether to use a subset of the dataset
            subset_ratio: Ratio of dataset to use
            seed: Random seed for subset sampling
        """
        # Read video paths from CSV file
        # CSV format: one video path per line (no header expected)
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Read CSV - handle both with and without headers
        try:
            df = pd.read_csv(csv_file, header=None)
            # Assume first column contains paths
            self.video_paths = df[0].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_file}: {e}")
        
        # Filter out non-existent files
        self.video_paths = [path for path in self.video_paths if os.path.exists(path)]
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No valid video paths found in {csv_file}")
        
        # Apply subset sampling if requested
        if use_subset:
            if not (0.0 < subset_ratio <= 1.0):
                raise ValueError(f"subset_ratio must be between 0.0 and 1.0, got {subset_ratio}")
            
            if seed is not None:
                random.seed(seed)
            
            num_samples = int(len(self.video_paths) * subset_ratio)
            num_samples = max(1, num_samples)  # Ensure at least 1 sample
            self.video_paths = random.sample(self.video_paths, num_samples)
        
        self.sample_frames = sample_frames
        self.temporal_stride = temporal_stride
        self.num_frames = sample_frames // temporal_stride  # Final number of frames (16)
        
    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and process a video.
        
        Args:
            idx: Index of the video to load
            
        Returns:
            Tensor of shape [num_frames, 3, 224, 224] where num_frames = sample_frames // temporal_stride
        """
        video_path = self.video_paths[idx]
        
        try:
            # Load video using torchvision
            # read_video returns: (video tensor, audio tensor, info dict)
            # video tensor shape: [T, H, W, C] where T is number of frames
            video, _, info = io.read_video(
                video_path,
                pts_unit='sec',
                output_format='TCHW'  # Output format: [T, C, H, W]
            )
            
            # video shape: [T, C, H, W] where T is total frames
            total_frames = video.shape[0]
            
            # Check if video has enough frames
            required_frames = self.sample_frames * self.temporal_stride
            if total_frames < required_frames:
                # If video is too short, repeat the last frame or pad
                # For now, we'll raise an error (can be handled more gracefully)
                raise ValueError(
                    f"Video {video_path} has {total_frames} frames, "
                    f"but needs at least {required_frames} frames"
                )
            
            # Randomly sample starting frame
            max_start = total_frames - required_frames
            start_frame = random.randint(0, max_start)
            
            # Extract consecutive frames
            end_frame = start_frame + required_frames
            sampled_frames = video[start_frame:end_frame]  # [required_frames, C, H, W]
            
            # Apply temporal downsampling with stride
            # This selects every temporal_stride-th frame
            downsampled_frames = sampled_frames[::self.temporal_stride]  # [num_frames, C, H, W]
            
            # Ensure we have exactly num_frames
            if downsampled_frames.shape[0] != self.num_frames:
                # Truncate or pad if needed
                if downsampled_frames.shape[0] > self.num_frames:
                    downsampled_frames = downsampled_frames[:self.num_frames]
                else:
                    # Pad with last frame
                    padding = self.num_frames - downsampled_frames.shape[0]
                    last_frame = downsampled_frames[-1:].repeat(padding, 1, 1, 1)
                    downsampled_frames = torch.cat([downsampled_frames, last_frame], dim=0)
            
            # Normalize pixel values to [0, 1] if not already
            if downsampled_frames.dtype == torch.uint8:
                downsampled_frames = downsampled_frames.float() / 255.0
            
            # Ensure shape is [num_frames, 3, 224, 224]
            # Videos are already 224x224x3 according to requirements
            assert downsampled_frames.shape == (self.num_frames, 3, 224, 224), \
                f"Unexpected shape: {downsampled_frames.shape}, expected ({self.num_frames}, 3, 224, 224)"
            
            return downsampled_frames
            
        except Exception as e:
            # If there's an error loading this video, raise a more informative error
            raise RuntimeError(f"Error loading video {video_path}: {e}")

