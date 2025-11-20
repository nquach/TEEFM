"""
Models package for VideoMAE with EVEREST masking.
"""

from .vit_backbone import VisionTransformer, get_vit_config
from .videomae import VideoMAE, VideoMAEEncoder, VideoMAEDecoder
from .everest_masking import (
    EVERESTMaskingGenerator,
    MotionEstimator,
    InformationIntensiveFrameSelector,
    random_masking
)

__all__ = [
    'VisionTransformer',
    'get_vit_config',
    'VideoMAE',
    'VideoMAEEncoder',
    'VideoMAEDecoder',
    'EVERESTMaskingGenerator',
    'MotionEstimator',
    'InformationIntensiveFrameSelector',
    'random_masking'
]

