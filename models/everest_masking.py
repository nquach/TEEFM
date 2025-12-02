"""
EVEREST Masking Generator for VideoMAE.

EVEREST (Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens)
identifies informative tokens based on motion/feature density and discards uninformative ones.

This module implements the EVEREST masking strategy that:
1. Computes motion/feature importance for each spatiotemporal token
2. Selects informative tokens to keep (unmasked)
3. Generates binary masks for the remaining tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EverestMaskingGenerator:
    """
    EVEREST masking generator that identifies informative tokens.
    
    The EVEREST approach computes motion/feature importance to determine which
    tokens contain rich motion features and should be kept, while discarding
    uninformative/redundant tokens.
    
    Args:
        input_size (tuple): Input size as (T, H, W) where T is temporal, H and W are spatial
        patch_size (int): Size of each patch (default: 16)
        mask_ratio (float): Ratio of tokens to mask (default: 0.75)
        motion_threshold (float): Threshold for motion importance (default: 0.1)
    """
    
    def __init__(
        self,
        input_size: tuple,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        motion_threshold: float = 0.1
    ):
        """
        Initialize the EVEREST masking generator.
        
        Args:
            input_size: Tuple of (T, H, W) - temporal, height, width
            patch_size: Size of each patch
            mask_ratio: Fraction of tokens to mask
            motion_threshold: Threshold for motion detection
        """
        self.T, self.H, self.W = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.motion_threshold = motion_threshold
        
        # Compute number of patches in spatial dimensions
        self.num_patches_h = self.H // self.patch_size
        self.num_patches_w = self.W // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Total number of spatiotemporal tokens
        self.num_tokens = self.T * self.num_patches
        
        # Number of tokens to keep (unmasked)
        self.num_keep = int(self.num_tokens * (1 - self.mask_ratio))
    
    def compute_motion_importance(self, video: torch.Tensor) -> torch.Tensor:
        """
        Compute motion importance for each spatiotemporal token.
        
        Motion is estimated by computing frame differences and aggregating
        the magnitude of changes across temporal and spatial dimensions.
        
        Args:
            video: Input video tensor of shape [B, T, C, H, W] or [T, C, H, W]
            
        Returns:
            Motion importance scores of shape [B, T, num_patches] or [T, num_patches]
        """
        # Handle batch dimension
        has_batch = len(video.shape) == 5
        if not has_batch:
            video = video.unsqueeze(0)
        
        B, T, C, H, W = video.shape
        
        # Convert to grayscale for motion computation
        # Use luminance: 0.299*R + 0.587*G + 0.114*B
        if C == 3:
            gray = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]
        else:
            gray = video[:, :, 0]
        
        # Compute temporal differences (motion between consecutive frames)
        # Shape: [B, T-1, H, W]
        frame_diffs = torch.abs(gray[:, 1:] - gray[:, :-1])
        
        # Average motion across temporal dimension
        # Shape: [B, T-1, H, W]
        motion = frame_diffs
        
        # Pad to match temporal dimension (use last frame difference for last frame)
        if T > 1:
            motion = F.pad(motion, (0, 0, 0, 0, 0, 1), mode='replicate')
        else:
            # Single frame case - use zero motion
            motion = torch.zeros(B, 1, H, W, device=video.device, dtype=video.dtype)
        
        # Reshape to patches and compute importance per patch
        # Shape: [B, T, num_patches_h, num_patches_w]
        motion_patches = F.avg_pool2d(
            motion.view(B * T, 1, H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        motion_patches = motion_patches.view(B, T, self.num_patches_h, self.num_patches_w)
        
        # Flatten to [B, T, num_patches]
        motion_importance = motion_patches.view(B, T, self.num_patches)
        
        # Normalize importance scores
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        motion_importance = (motion_importance - motion_importance.min()) / (
            motion_importance.max() - motion_importance.min() + eps
        )
        
        if not has_batch:
            motion_importance = motion_importance.squeeze(0)
        
        return motion_importance
    
    def generate_mask(self, video: torch.Tensor) -> torch.Tensor:
        """
        Generate binary mask for EVEREST masking.
        
        Args:
            video: Input video tensor of shape [B, T, C, H, W] or [T, C, H, W]
            
        Returns:
            Binary mask of shape [B, num_tokens] or [num_tokens]
            where 1 indicates keep (unmasked) and 0 indicates mask
        """
        # Compute motion importance
        motion_importance = self.compute_motion_importance(video)
        
        # Handle batch dimension
        has_batch = len(motion_importance.shape) == 3
        if not has_batch:
            motion_importance = motion_importance.unsqueeze(0)
        
        B, T, num_patches = motion_importance.shape
        
        # Flatten to [B, num_tokens]
        importance_flat = motion_importance.view(B, self.num_tokens)
        
        # Select top-k most important tokens to keep
        # Get indices of top-k tokens
        _, top_indices = torch.topk(importance_flat, k=self.num_keep, dim=1)
        
        # Create binary mask: 1 for keep, 0 for mask
        mask = torch.zeros(B, self.num_tokens, device=video.device, dtype=torch.bool)
        for i in range(B):
            mask[i, top_indices[i]] = True
        
        if not has_batch:
            mask = mask.squeeze(0)
        
        return mask
    
    def apply_mask(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple:
        """
        Apply mask to tokens, separating visible and masked tokens.
        
        Args:
            tokens: Token embeddings of shape [B, num_tokens, embed_dim]
            mask: Binary mask of shape [B, num_tokens] where 1 = keep, 0 = mask
            
        Returns:
            Tuple of (visible_tokens, ids_restore)
            - visible_tokens: [B, num_keep, embed_dim]
            - ids_restore: [B, num_tokens] - indices to restore original order
        """
        B, num_tokens, embed_dim = tokens.shape
        
        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        # Get visible token indices for each sample in batch
        visible_tokens_list = []
        ids_restore_list = []
        
        for b in range(B):
            # Get indices where mask is True (visible tokens)
            visible_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_keep]
            # Get indices where mask is False (masked tokens)
            masked_idx = (~mask[b]).nonzero(as_tuple=False).squeeze(-1)  # [num_mask]
            
            # Extract visible tokens
            visible_tokens_b = tokens[b, visible_idx]  # [num_keep, embed_dim]
            visible_tokens_list.append(visible_tokens_b)
            
            # Create ids_restore: concatenate visible and masked indices, then get sort order
            all_indices = torch.cat([visible_idx, masked_idx], dim=0)  # [num_tokens]
            ids_restore_b = torch.argsort(all_indices, dim=0)  # [num_tokens]
            ids_restore_list.append(ids_restore_b)
        
        # Stack to create batch tensors
        visible_tokens = torch.stack(visible_tokens_list, dim=0)  # [B, num_keep, embed_dim]
        ids_restore = torch.stack(ids_restore_list, dim=0)  # [B, num_tokens]
        
        return visible_tokens, ids_restore

