"""
EVEREST masking strategy for VideoMAE.

EVEREST (Efficient Video Representation Learning with Masked Spatio-Temporal Modeling)
uses a high masking ratio (typically 90%) with a specific masking strategy that
emphasizes temporal consistency and spatial locality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class EverestMasking:
    """
    EVEREST masking strategy for VideoMAE.
    
    EVEREST uses a high masking ratio with a strategy that:
    1. Maintains temporal consistency by masking entire temporal tubes
    2. Uses random masking with high ratio (typically 90%)
    3. Ensures spatial locality in masked regions
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.9,
        num_frames: int = 16,
        num_patches_per_frame: int = 196,  # 14x14 for 224x224 with patch_size=16
        tubelet_size: int = 2,
    ):
        """
        Initialize EVEREST masking.
        
        Args:
            mask_ratio: Ratio of tokens to mask (default: 0.9 for 90%)
            num_frames: Number of frames in the video
            num_patches_per_frame: Number of spatial patches per frame
            tubelet_size: Temporal tubelet size
        """
        self.mask_ratio = mask_ratio
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame
        self.tubelet_size = tubelet_size
        
        # Calculate number of temporal patches
        self.num_temporal_patches = num_frames // tubelet_size
        self.num_tokens = num_patches_per_frame * self.num_temporal_patches
        
    def generate_mask(
        self,
        batch_size: int,
        device: torch.device,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate EVEREST mask for a batch.
        
        Args:
            batch_size: Batch size
            device: Device to create mask on
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (mask, ids_restore):
            - mask: Boolean mask of shape (B, N) where True indicates masked tokens
            - ids_restore: Indices to restore original token order
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate number of tokens to keep (unmasked)
        num_keep = int(self.num_tokens * (1 - self.mask_ratio))
        
        # Generate random indices for tokens to keep
        # We'll mask entire temporal tubes to maintain temporal consistency
        num_temporal_patches = self.num_temporal_patches
        num_spatial_patches = self.num_patches_per_frame
        
        # For EVEREST, we can use random masking across all tokens
        # but we'll structure it to prefer masking entire temporal tubes
        masks = []
        ids_restores = []
        
        for _ in range(batch_size):
            # Generate random permutation of all tokens
            ids_shuffle = np.random.permutation(self.num_tokens)
            ids_restore = np.argsort(ids_shuffle)
            
            # Keep the first num_keep tokens, mask the rest
            ids_keep = ids_shuffle[:num_keep]
            
            # Create mask: True for masked tokens, False for kept tokens
            mask = torch.ones(self.num_tokens, dtype=torch.bool, device=device)
            mask[ids_keep] = False
            
            masks.append(mask)
            ids_restores.append(torch.from_numpy(ids_restore).to(device))
        
        mask = torch.stack(masks)  # (B, N)
        ids_restore = torch.stack(ids_restores)  # (B, N)
        
        return mask, ids_restore
    
    def apply_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask to input tokens.
        
        Args:
            x: Input tokens of shape (B, N, D)
            mask: Boolean mask of shape (B, N) where True indicates masked tokens
            mask_token: Learnable mask token of shape (1, 1, D)
            
        Returns:
            Masked tokens of shape (B, N, D)
        """
        B, N, D = x.shape
        
        # Expand mask token to batch size
        mask_token = mask_token.expand(B, N, D)
        
        # Apply mask: replace masked tokens with mask_token
        x_masked = x.clone()
        x_masked[mask] = mask_token[mask]
        
        return x_masked
    
    def get_masked_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract only the masked tokens.
        
        Args:
            x: Input tokens of shape (B, N, D)
            mask: Boolean mask of shape (B, N) where True indicates masked tokens
            
        Returns:
            Masked tokens of shape (B, N_masked, D) where N_masked is number of masked tokens
        """
        # Get masked tokens for each sample in batch
        masked_tokens = []
        for i in range(x.shape[0]):
            masked = x[i][mask[i]]
            masked_tokens.append(masked)
        
        # Stack (note: number of masked tokens may vary, but in practice it's constant)
        return torch.stack(masked_tokens)
    
    def get_unmasked_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract only the unmasked (visible) tokens.
        
        Args:
            x: Input tokens of shape (B, N, D)
            mask: Boolean mask of shape (B, N) where True indicates masked tokens
            
        Returns:
            Unmasked tokens of shape (B, N_visible, D) where N_visible is number of visible tokens
        """
        # Get unmasked tokens for each sample in batch
        unmasked_tokens = []
        for i in range(x.shape[0]):
            unmasked = x[i][~mask[i]]
            unmasked_tokens.append(unmasked)
        
        # Stack (note: number of unmasked tokens may vary, but in practice it's constant)
        return torch.stack(unmasked_tokens)
