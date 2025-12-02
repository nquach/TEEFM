"""
Video dataset implementation for VideoMAE training.

This module implements a PyTorch Dataset class that loads videos from CSV files,
samples consecutive frames, and applies temporal downsampling.
"""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video


class VideoDataset(Dataset):
    """
    Dataset class for loading videos for VideoMAE training.
    
    Videos are expected to be 224x224x3 with variable number of frames.
    The dataset randomly samples 32 consecutive frames and applies temporal
    downsampling with stride 2 to obtain 16 frames.
    
    Args:
        csv_file (str): Path to CSV file containing video file paths.
        subset_ratio (float, optional): Ratio of dataset to use (0.0-1.0).
            If None, uses entire dataset. Default: None.
        num_sample_frames (int): Number of consecutive frames to sample. Default: 32.
        temporal_stride (int): Temporal stride for downsampling. Default: 2.
        transform (callable, optional): Optional transform to apply to frames.
    """
    
    def __init__(
        self,
        csv_file: str,
        subset_ratio: float = None,
        num_sample_frames: int = 32,
        temporal_stride: int = 2,
        transform=None,
    ):
        # Read video paths from CSV file
        # CSV file should have a column with video paths (assumes first column or 'path' column)
        try:
            # First try reading with header (in case CSV has 'path' column)
            df = pd.read_csv(csv_file)
            if 'path' in df.columns:
                self.video_paths = df['path'].tolist()
            else:
                # No 'path' column, use first column
                self.video_paths = df.iloc[:, 0].tolist()
        except Exception:
            # If that fails, try reading without header
            df = pd.read_csv(csv_file, header=None)
            if len(df.columns) > 0:
                self.video_paths = df.iloc[:, 0].tolist()
            else:
                raise ValueError(f"CSV file {csv_file} appears to be empty or malformed")
        
        # Optionally use a random subset of the dataset
        if subset_ratio is not None and 0.0 < subset_ratio < 1.0:
            num_samples = int(len(self.video_paths) * subset_ratio)
            self.video_paths = random.sample(self.video_paths, num_samples)
            print(f"Using {num_samples} videos ({subset_ratio*100:.1f}% of dataset)")
        
        self.num_sample_frames = num_sample_frames
        self.temporal_stride = temporal_stride
        self.transform = transform
        
        # Filter out videos that don't exist
        self.video_paths = [p for p in self.video_paths if os.path.exists(p)]
        if len(self.video_paths) == 0:
            raise ValueError(f"No valid video paths found in {csv_file}")
    
    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Load and process a video sample.
        
        Args:
            idx (int): Index of the video to load.
            
        Returns:
            torch.Tensor: Processed video frames of shape (T, C, H, W)
                where T is the number of frames after downsampling.
        """
        video_path = self.video_paths[idx]
        
        # Read video using torchvision
        # Returns: video tensor (T, H, W, C), audio, info
        try:
            video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
        except Exception as e:
            raise RuntimeError(f"Error loading video {video_path}: {str(e)}")
        
        # video shape: (T, C, H, W) where T is number of frames
        num_frames = video.shape[0]
        
        # Check if video has enough frames
        required_frames = self.num_sample_frames
        if num_frames < required_frames:
            # If video is too short, pad by repeating the last frame
            padding = required_frames - num_frames
            last_frame = video[-1:].repeat(padding, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
            num_frames = video.shape[0]
        
        # Randomly sample starting frame for consecutive frames
        max_start = num_frames - required_frames
        start_frame = random.randint(0, max_start)
        
        # Extract consecutive frames
        frames = video[start_frame : start_frame + required_frames]
        
        # Apply temporal downsampling with stride
        # This reduces from num_sample_frames to num_sample_frames // temporal_stride
        frames = frames[:: self.temporal_stride]
        
        # Normalize to [0, 1] if not already (assuming uint8 input)
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        
        # Apply optional transform
        if self.transform is not None:
            frames = self.transform(frames)
        
        return frames

