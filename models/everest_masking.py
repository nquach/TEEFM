"""
EVEREST Masking Strategy

EVEREST (Efficient Masked Video Autoencoder by Removing Redundant 
Spatiotemporal Tokens) implements an intelligent masking strategy that
selects informative tokens containing rich motion features and discards
uninformative ones, rather than using random masking.

This significantly reduces computation and memory requirements while
maintaining or improving performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MotionEstimator(nn.Module):
    """
    Simple motion estimator to identify patches with high motion content.
    
    Computes frame differences to identify regions with motion.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate motion intensity for each patch.
        
        Args:
            x: Video tensor of shape (B, T, H, W, C)
            
        Returns:
            Motion scores of shape (B, num_patches) - higher values indicate more motion
        """
        B, T, H, W, C = x.shape
        
        # Convert to grayscale for motion estimation
        # Using luminance formula: 0.299*R + 0.587*G + 0.114*B
        if C == 3:
            gray = 0.299 * x[:, :, :, :, 0] + 0.587 * x[:, :, :, :, 1] + 0.114 * x[:, :, :, :, 2]
        else:
            gray = x[:, :, :, :, 0]
        
        # Compute frame differences (temporal gradient)
        # Shape: (B, T-1, H, W)
        frame_diffs = torch.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
        
        # Sum over temporal dimension to get total motion per spatial location
        # Shape: (B, H, W)
        motion_map = frame_diffs.sum(dim=1)
        
        # Downsample to patch level
        # Assuming patch_size = 16, we need to average over 16x16 regions
        patch_size = 16
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        # Reshape and average to get motion per patch
        motion_map = motion_map.reshape(B, h_patches, patch_size, w_patches, patch_size)
        motion_map = motion_map.mean(dim=(2, 4))  # (B, h_patches, w_patches)
        
        # Flatten to get motion score per patch
        motion_scores = motion_map.reshape(B, h_patches * w_patches)  # (B, num_patches_per_frame)
        
        # Repeat for each frame (simplified - can be enhanced to track motion across frames)
        num_frames = T
        motion_scores = motion_scores.unsqueeze(1).expand(-1, num_frames, -1)
        motion_scores = motion_scores.reshape(B, num_frames * h_patches * w_patches)
        
        return motion_scores


class InformationIntensiveFrameSelector:
    """
    Information-intensive frame selection strategy.
    
    Selects frames that contain the most information with minimal redundancy.
    This is used to focus on informative and causal frames.
    """
    
    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames
    
    def select_frames(
        self,
        video: torch.Tensor,
        num_selected: Optional[int] = None
    ) -> torch.Tensor:
        """
        Select information-intensive frames from video.
        
        Args:
            video: Video tensor of shape (B, T, H, W, C)
            num_selected: Number of frames to select (default: all frames)
            
        Returns:
            Selected video tensor
        """
        if num_selected is None or num_selected >= video.shape[1]:
            return video
        
        # Simple strategy: select frames with highest variance (more information)
        B, T, H, W, C = video.shape
        
        # Compute variance per frame
        frame_vars = video.var(dim=(2, 3, 4))  # (B, T)
        
        # Select top-k frames with highest variance
        _, top_indices = torch.topk(frame_vars, num_selected, dim=1)  # (B, num_selected)
        
        # Sort indices to maintain temporal order
        top_indices, _ = torch.sort(top_indices, dim=1)
        
        # Select frames
        selected_frames = torch.gather(
            video,
            dim=1,
            index=top_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W, C)
        )
        
        return selected_frames


class EVERESTMaskingGenerator:
    """
    EVEREST masking generator that creates intelligent masks based on
    motion and information content rather than random masking.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        mask_ratio: float = 0.9,  # High masking ratio as in VideoMAE
        motion_weight: float = 0.7,  # Weight for motion-based selection
        random_ratio: float = 0.1  # Small random component for diversity
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        self.motion_weight = motion_weight
        self.random_ratio = random_ratio
        
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * num_frames
        
        self.motion_estimator = MotionEstimator()
        self.frame_selector = InformationIntensiveFrameSelector(num_frames)
    
    def __call__(
        self,
        x: torch.Tensor,
        return_indices: bool = False
    ) -> torch.Tensor:
        """
        Generate EVEREST mask for input video.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            return_indices: Whether to return mask indices
            
        Returns:
            Binary mask of shape (B, num_patches) - 1 for visible, 0 for masked
            If return_indices=True, also returns indices of visible tokens
        """
        B = x.shape[0]
        device = x.device
        
        # Estimate motion scores for each patch
        motion_scores = self.motion_estimator(x)  # (B, num_patches)
        
        # Add small random component for diversity
        random_scores = torch.rand(B, self.num_patches, device=device)
        
        # Combine motion and random scores
        combined_scores = (
            self.motion_weight * motion_scores +
            (1 - self.motion_weight) * random_scores
        )
        
        # Select top-k patches to keep visible (low mask_ratio means keep more)
        num_visible = int(self.num_patches * (1 - self.mask_ratio))
        
        # Get indices of top-k patches (highest scores = most informative)
        _, visible_indices = torch.topk(
            combined_scores,
            num_visible,
            dim=1,
            largest=True
        )
        
        # Create binary mask: 1 for visible, 0 for masked
        mask = torch.zeros(B, self.num_patches, device=device, dtype=torch.float32)
        mask.scatter_(1, visible_indices, 1.0)
        
        if return_indices:
            return mask, visible_indices
        return mask
    
    def get_motion_based_mask(
        self,
        x: torch.Tensor,
        top_k_ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Generate mask based purely on motion (for analysis).
        
        Args:
            x: Input video tensor
            top_k_ratio: Ratio of top motion patches to keep visible
            
        Returns:
            Binary mask
        """
        motion_scores = self.motion_estimator(x)
        num_visible = int(self.num_patches * top_k_ratio)
        
        _, visible_indices = torch.topk(
            motion_scores,
            num_visible,
            dim=1,
            largest=True
        )
        
        mask = torch.zeros(
            x.shape[0],
            self.num_patches,
            device=x.device,
            dtype=torch.float32
        )
        mask.scatter_(1, visible_indices, 1.0)
        
        return mask


def random_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.9,
    num_patches: int = 196 * 16
) -> torch.Tensor:
    """
    Random masking strategy (baseline for comparison).
    
    Args:
        x: Input tensor (shape not used, just for batch size)
        mask_ratio: Ratio of patches to mask
        num_patches: Total number of patches
        
    Returns:
        Binary mask of shape (B, num_patches)
    """
    B = x.shape[0]
    device = x.device
    
    num_visible = int(num_patches * (1 - mask_ratio))
    
    # Randomly select visible patches
    mask = torch.zeros(B, num_patches, device=device, dtype=torch.float32)
    
    for i in range(B):
        visible_indices = torch.randperm(num_patches, device=device)[:num_visible]
        mask[i, visible_indices] = 1.0
    
    return mask

