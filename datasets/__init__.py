"""
Datasets package for VideoMAE training.
"""
from .optimized_video_dataset import OptimizedVideoDataset
from .video_data_module import VideoDataModule
__all__ = ['OptimizedVideoDataset', 'VideoDataModule']

