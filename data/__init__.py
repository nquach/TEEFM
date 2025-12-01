"""
Data package for video dataset loading.
"""

from .video_dataset import VideoDataset, get_video_transforms, MultiscaleCrop

__all__ = ['VideoDataset', 'get_video_transforms', 'MultiscaleCrop']

# LitData imports (optional)
try:
    from .litdata_video_dataset import (
        LitDataVideoDataset,
        optimize_video_dataset,
        StreamingDataLoader,
        LITDATA_AVAILABLE
    )
    __all__.extend([
        'LitDataVideoDataset',
        'optimize_video_dataset',
        'StreamingDataLoader',
        'LITDATA_AVAILABLE'
    ])
except ImportError:
    LITDATA_AVAILABLE = False

