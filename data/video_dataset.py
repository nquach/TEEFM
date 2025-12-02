"""
Video dataset implementation using torchcodec for efficient video decoding.

This module implements a PyTorch Dataset class that:
- Reads video file paths from CSV files
- Uses torchcodec VideoDecoder to decode MP4 videos
- Randomly samples consecutive frames
- Applies temporal downsampling with specified stride
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Callable
import os

try:
    from torchcodec import VideoDecoder
except ImportError:
    raise ImportError(
        "torchcodec is required. Please install it with: pip install torchcodec"
    )


class VideoDataset(Dataset):
    """
    PyTorch Dataset for loading video data from MP4 files.
    
    The dataset assumes videos are already preprocessed to 224x224x3 resolution.
    It randomly samples consecutive frames and applies temporal downsampling.
    
    Args:
        csv_file: Path to CSV file containing video file paths (one per line)
        sample_frames: Number of consecutive frames to sample (default: 32)
        temporal_stride: Stride for temporal downsampling (default: 2)
        transform: Optional transform to apply to frames
    """
    
    def __init__(
        self,
        csv_file: str,
        sample_frames: int = 32,
        temporal_stride: int = 2,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the VideoDataset.
        
        Args:
            csv_file: Path to CSV file with video paths
            sample_frames: Number of frames to sample before downsampling
            temporal_stride: Stride for temporal downsampling
            transform: Optional transform function
        """
        self.sample_frames = sample_frames
        self.temporal_stride = temporal_stride
        self.transform = transform
        
        # Read video paths from CSV file
        # CSV format: one path per line (no header expected)
        try:
            df = pd.read_csv(csv_file, header=None, names=['path'])
            self.video_paths = df['path'].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_file}: {e}")
        
        # Filter out non-existent files
        self.video_paths = [
            path for path in self.video_paths 
            if os.path.exists(path.strip())
        ]
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No valid video files found in {csv_file}")
        
        print(f"Loaded {len(self.video_paths)} video paths from {csv_file}")
    
    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a video sample.
        
        Args:
            idx: Index of the video to load
            
        Returns:
            Tensor of shape (T, H, W, C) where:
            - T: Number of frames after downsampling (sample_frames // temporal_stride)
            - H: Height (224)
            - W: Width (224)
            - C: Channels (3)
        """
        video_path = self.video_paths[idx].strip()
        
        try:
            # Initialize video decoder
            decoder = VideoDecoder(video_path)
            
            # Get total number of frames in the video
            total_frames = decoder.frame_count
            
            # Ensure we have enough frames
            if total_frames < self.sample_frames:
                # If video is shorter than required, pad by repeating frames
                # or sample with replacement
                start_frame = 0
                num_frames_to_read = min(self.sample_frames, total_frames)
            else:
                # Randomly sample starting frame
                max_start = total_frames - self.sample_frames
                start_frame = np.random.randint(0, max_start + 1)
                num_frames_to_read = self.sample_frames
            
            # Read frames using torchcodec
            # VideoDecoder.read() returns frames as a tensor
            frames = decoder.read(start_frame, num_frames_to_read)
            
            # Handle case where we got fewer frames than requested
            if frames.shape[0] < self.sample_frames:
                # Pad by repeating the last frame
                last_frame = frames[-1:].repeat(self.sample_frames - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            
            # Apply temporal downsampling with stride
            # frames shape: (T, C, H, W) from torchcodec
            frames = frames[::self.temporal_stride]
            
            # Convert to (T, H, W, C) format if needed
            # torchcodec typically returns (T, C, H, W), but we want (T, H, W, C)
            if frames.dim() == 4 and frames.shape[1] == 3:
                frames = frames.permute(0, 2, 3, 1)
            
            # Normalize to [0, 1] if not already
            if frames.max() > 1.0:
                frames = frames.float() / 255.0
            
            # Apply transform if provided
            if self.transform is not None:
                frames = self.transform(frames)
            
            return frames
            
        except Exception as e:
            # If there's an error loading the video, return a zero tensor
            # This allows training to continue even if some videos are corrupted
            print(f"Warning: Error loading video {video_path}: {e}")
            # Return a dummy tensor with correct shape
            num_output_frames = self.sample_frames // self.temporal_stride
            dummy = torch.zeros(
                num_output_frames, 224, 224, 3,
                dtype=torch.float32
            )
            return dummy
