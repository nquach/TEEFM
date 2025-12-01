"""
LitData Video Dataset Module for VideoMAE Training

This module implements litdata-based streaming dataset for faster data loading.
Videos are preprocessed into optimized chunks that can be streamed efficiently.
"""

import os
import pandas as pd
import torch
from typing import Optional, Tuple, Dict, Any
import torchvision.transforms as transforms
from torchvision.io import read_video, VideoReader
import numpy as np
import random

try:
    import litdata as ld
    from litdata import StreamingDataset, StreamingDataLoader
    LITDATA_AVAILABLE = True
except ImportError:
    LITDATA_AVAILABLE = False
    print("Warning: litdata not available. Install with: pip install litdata")

from .video_dataset import get_video_transforms, MultiscaleCrop


def optimize_video_item(
    item: Dict[str, Any],
    num_frames: int = 32,
    temporal_stride: int = 2,
    img_size: int = 224,
    use_multiscale_crop: bool = True,
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)
) -> Dict[str, Any]:
    """
    Optimization function for litdata.
    
    Processes a single video item:
    1. Loads video from path
    2. Samples frames
    3. Applies augmentations (multiscale crop)
    4. Returns processed video tensor
    
    This function is called during the optimization phase to preprocess videos.
    
    Args:
        item: Dictionary with 'video_path' key
        num_frames: Number of frames to sample before downsampling
        temporal_stride: Stride for temporal downsampling
        img_size: Target image size
        use_multiscale_crop: Whether to apply multiscale cropping
        crop_scale_range: Scale range for multiscale cropping
        
    Returns:
        Dictionary with processed video tensor
    """
    video_path = item['video_path']
    
    try:
        # Get video frame count
        try:
            vr = VideoReader(video_path, "video")
            num_frames_total = len(vr)
        except Exception:
            video, _, _ = read_video(video_path, pts_unit='sec', output_format='TCHW')
            num_frames_total = video.shape[0]
        
        # Randomly sample starting frame for consecutive frames
        max_start = max(0, num_frames_total - num_frames)
        if max_start <= 0:
            start_frame = 0
            end_frame = min(num_frames_total, num_frames)
        else:
            start_frame = torch.randint(0, max_start + 1, (1,)).item()
            end_frame = start_frame + num_frames
        
        # Load video frames
        try:
            vr = VideoReader(video_path, "video")
            vr.seek(start_frame)
            
            frames = []
            for _ in range(min(end_frame - start_frame, num_frames)):
                frame_data = vr.next()
                if frame_data is None:
                    break
                frame_tensor = frame_data['data'].permute(2, 0, 1)  # (C, H, W)
                frames.append(frame_tensor)
            
            if len(frames) < num_frames:
                last_frame = frames[-1] if frames else torch.zeros(3, img_size, img_size)
                while len(frames) < num_frames:
                    frames.append(last_frame)
            
            video = torch.stack(frames[:num_frames], dim=0)  # (num_frames, C, H, W)
            
        except Exception:
            # Fallback: use read_video
            video, _, _ = read_video(
                video_path,
                pts_unit='sec',
                output_format='TCHW'
            )
            
            if num_frames_total < num_frames:
                padding = num_frames - num_frames_total
                last_frame = video[-1:].repeat(padding, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
                start_frame = 0
                end_frame = num_frames
            else:
                video = video[start_frame:end_frame]
        
        # Convert to (T, H, W, C) format
        video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
        
        # Temporally downsample with stride 2
        video = video[::temporal_stride]  # (16, H, W, C)
        
        # Normalize to [0, 1]
        video = video.float() / 255.0
        
        # Apply multiscale crop augmentation if enabled
        if use_multiscale_crop:
            crop_transform = MultiscaleCrop(
                target_size=img_size,
                scale_range=crop_scale_range
            )
            video = crop_transform(video)
        
        # Return processed video
        return {
            'video': video,  # (T, H, W, C) where T=16, H=W=224, C=3
            'video_path': video_path  # Keep path for reference
        }
        
    except Exception as e:
        # Return zero tensor on error
        print(f"Warning: Failed to process video {video_path}: {e}")
        return {
            'video': torch.zeros((num_frames // temporal_stride, img_size, img_size, 3), dtype=torch.float32),
            'video_path': video_path
        }


class LitDataVideoDataset(StreamingDataset):
    """
    LitData StreamingDataset for video data.
    
    This dataset streams preprocessed video chunks for fast data loading.
    Videos must be optimized first using optimize_video_dataset().
    
    Args:
        input_dir: Directory containing optimized data chunks
        num_frames: Number of frames (for validation)
        temporal_stride: Temporal stride (for validation)
        transform: Optional transforms to apply (normalization, etc.)
    """
    
    def __init__(
        self,
        input_dir: str,
        num_frames: int = 32,
        temporal_stride: int = 2,
        transform: Optional[transforms.Compose] = None
    ):
        if not LITDATA_AVAILABLE:
            raise ImportError("litdata is required. Install with: pip install litdata")
        
        super().__init__(input_dir=input_dir)
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.transform = transform
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a video sample from optimized chunks.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Video tensor of shape (T, H, W, C)
        """
        item = super().__getitem__(idx)
        video = item['video']
        
        # Apply transforms if provided (e.g., normalization)
        if self.transform is not None:
            video = self.transform(video)
        
        return video


def optimize_video_dataset(
    csv_file: str,
    output_dir: str,
    num_frames: int = 32,
    temporal_stride: int = 2,
    img_size: int = 224,
    use_multiscale_crop: bool = True,
    crop_scale_range: Tuple[float, float] = (0.8, 1.0),
    subset_ratio: float = 1.0,
    subset_seed: Optional[int] = None,
    num_workers: int = None
) -> None:
    """
    Optimize video dataset using litdata.
    
    Preprocesses all videos and creates optimized chunks for fast streaming.
    This is a one-time preprocessing step that should be run before training.
    
    Args:
        csv_file: Path to CSV file with video paths
        output_dir: Directory to save optimized chunks
        num_frames: Number of frames to sample
        temporal_stride: Temporal downsampling stride
        img_size: Target image size
        use_multiscale_crop: Whether to apply multiscale cropping during optimization
        crop_scale_range: Scale range for multiscale cropping
        subset_ratio: Ratio of dataset to optimize (1.0 = all)
        subset_seed: Seed for subset sampling
        num_workers: Number of workers for optimization (None = auto)
    """
    if not LITDATA_AVAILABLE:
        raise ImportError("litdata is required. Install with: pip install litdata")
    
    # Read video paths
    df = pd.read_csv(csv_file, header=None, names=['path'])
    video_paths = df['path'].tolist()
    video_paths = [p for p in video_paths if os.path.exists(p)]
    
    if len(video_paths) == 0:
        raise ValueError(f"No valid video files found in {csv_file}")
    
    # Apply subset sampling if needed
    if subset_ratio < 1.0:
        if subset_seed is not None:
            random.seed(subset_seed)
            np.random.seed(subset_seed)
        
        num_samples = max(1, int(len(video_paths) * subset_ratio))
        video_paths = random.sample(video_paths, num_samples)
        print(f"Optimizing {num_samples} videos ({subset_ratio*100:.1f}%) from {len(df)} total")
    
    # Create items for optimization
    items = [{'video_path': path} for path in video_paths]
    
    print(f"Optimizing {len(items)} videos to {output_dir}...")
    print("This may take a while. Videos are being preprocessed into optimized chunks.")
    
    # Optimize dataset
    ld.optimize(
        fn=optimize_video_item,
        inputs=items,
        output_dir=output_dir,
        num_workers=num_workers or os.cpu_count() or 4,
        fn_kwargs={
            'num_frames': num_frames,
            'temporal_stride': temporal_stride,
            'img_size': img_size,
            'use_multiscale_crop': use_multiscale_crop,
            'crop_scale_range': crop_scale_range
        }
    )
    
    print(f"Optimization complete! Optimized data saved to {output_dir}")

