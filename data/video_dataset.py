"""
Video Dataset Module for VideoMAE Training

This module implements a PyTorch Dataset class for loading and processing video files
for VideoMAE training. Videos are expected to be 224x224x3 and are processed by:
1. Randomly sampling 32 consecutive frames
2. Temporally downsampling with stride 2 to get 16 frames
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from typing import Optional, Callable
import random


class VideoDataset(Dataset):
    """
    Dataset class for loading videos from CSV file paths.
    
    Args:
        csv_file (str): Path to CSV file containing video file paths (one per line)
        transform (Optional[Callable]): Optional transform to apply to video frames
        num_frames_to_sample (int): Number of consecutive frames to sample (default: 32)
        temporal_stride (int): Stride for temporal downsampling (default: 2)
        frame_size (tuple): Expected frame size (height, width), default: (224, 224)
    """
    
    def __init__(
        self,
        csv_file: str,
        transform: Optional[Callable] = None,
        num_frames_to_sample: int = 32,
        temporal_stride: int = 2,
        frame_size: tuple = (224, 224)
    ):
        # Read video paths from CSV file
        # Handle both with and without header
        try:
            df = pd.read_csv(csv_file, header=None)
            # Assume first column contains paths
            self.video_paths = df[0].tolist()
        except Exception as e:
            # If CSV has header, try reading with header
            df = pd.read_csv(csv_file)
            if 'path' in df.columns:
                self.video_paths = df['path'].tolist()
            else:
                # Use first column
                self.video_paths = df.iloc[:, 0].tolist()
        
        self.transform = transform
        self.num_frames_to_sample = num_frames_to_sample
        self.temporal_stride = temporal_stride
        self.frame_size = frame_size
        
        # Calculate final number of frames after downsampling
        self.num_frames = num_frames_to_sample // temporal_stride
    
    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and process a video.
        
        Args:
            idx (int): Index of the video to load
            
        Returns:
            torch.Tensor: Video tensor of shape (C, T, H, W) where:
                C = 3 (RGB channels)
                T = 16 (number of frames after downsampling)
                H, W = 224 (frame dimensions)
        """
        video_path = self.video_paths[idx]
        
        # Load video frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Check if video has enough frames
        if len(frames) < self.num_frames_to_sample:
            # If not enough frames, pad by repeating the last frame
            num_padding = self.num_frames_to_sample - len(frames)
            frames.extend([frames[-1]] * num_padding)
        
        # Randomly sample num_frames_to_sample consecutive frames
        max_start_idx = len(frames) - self.num_frames_to_sample
        if max_start_idx < 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start_idx)
        
        sampled_frames = frames[start_idx:start_idx + self.num_frames_to_sample]
        
        # Temporally downsample with stride
        downsampled_frames = sampled_frames[::self.temporal_stride]
        
        # Convert to numpy array and normalize to [0, 1]
        video_array = np.stack(downsampled_frames, axis=0)  # Shape: (T, H, W, C)
        video_array = video_array.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange to (C, T, H, W)
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Apply transform if provided
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor

