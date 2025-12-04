"""
Datasets package for VideoMAE training.
"""

from .custom_video_dataset import CustomVideoDataset

try:
    from .optimized_video_dataset import OptimizedVideoDataset
    __all__ = ['CustomVideoDataset', 'OptimizedVideoDataset']
except ImportError:
    __all__ = ['CustomVideoDataset']

