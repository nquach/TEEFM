"""
Vision Transformer (ViT) Backbone for VideoMAE

This module implements Vision Transformer architectures (ViT-S, ViT-B, ViT-L)
adapted for video processing. The ViT processes video patches as tokens.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts video frames into patches.
    
    For video, we treat each spatial patch across all frames as a token.
    This creates a 3D patch embedding: (T, H, W) -> (T*H*W, D)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Number of patches per frame
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        # Total number of patches (across all frames)
        self.num_patches = self.num_patches_per_frame * num_frames
        
        # Convolutional patch embedding
        # Input: (B, T, C, H, W) -> Output: (B, T*H*W, D)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch embedding.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, T, H, W, C)
            
        Returns:
            Patch embeddings of shape (B, T*num_patches_per_frame, embed_dim)
        """
        B = x.shape[0]
        
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[2] == 3:  # (B, T, C, H, W)
                pass
            elif x.shape[4] == 3:  # (B, T, H, W, C)
                x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Reshape to process all frames: (B*T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        # Apply patch embedding: (B*T, C, H, W) -> (B*T, D, H//patch_size, W//patch_size)
        x = self.proj(x)  # (B*T, D, H_p, W_p)
        
        # Flatten spatial dimensions: (B*T, D, H_p, W_p) -> (B*T, D, H_p*W_p)
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2)  # (B*T, D, num_patches_per_frame)
        
        # Transpose: (B*T, D, num_patches_per_frame) -> (B*T, num_patches_per_frame, D)
        x = x.transpose(1, 2)  # (B*T, num_patches_per_frame, D)
        
        # Reshape back to separate batch and time: (B, T*num_patches_per_frame, D)
        x = x.reshape(B, T * self.num_patches_per_frame, self.embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            mask: Optional attention mask of shape (B, N) or (B, N, N)
            
        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # (B, N) -> expand to (B, 1, N, N)
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v)  # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop_path = nn.Identity()  # Simplified - can add stochastic depth
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) backbone for video.
    
    Supports ViT-S, ViT-B, and ViT-L configurations.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        num_frames: int = 16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)  # +1 for CLS token
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout for embeddings
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using standard ViT initialization."""
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=0.02)
            nn.init.normal_(block.attn.proj.weight, std=0.02)
            nn.init.normal_(block.mlp[0].weight, std=0.02)
            nn.init.normal_(block.mlp[3].weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ViT.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C) or (B, T, C, H, W)
            mask: Optional mask for tokens (B, num_patches) - 1 for visible, 0 for masked
            
        Returns:
            Output features of shape (B, num_patches+1, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, T, ...) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply mask if provided (mask out tokens)
        if mask is not None:
            # Expand mask to include CLS token (always visible)
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)  # (B, num_patches+1)
            # Set masked tokens to zero (they will be ignored in attention)
            x = x * mask.unsqueeze(-1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final layer norm
        x = self.norm(x)
        
        return x


def get_vit_config(backbone: str = 'ViT-S') -> dict:
    """
    Get ViT configuration for different backbone sizes.
    
    Args:
        backbone: One of 'ViT-S', 'ViT-B', 'ViT-L'
        
    Returns:
        Dictionary with model configuration parameters
    """
    configs = {
        'ViT-S': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
        },
        'ViT-B': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
        },
        'ViT-L': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
        }
    }
    
    if backbone not in configs:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Must be one of {list(configs.keys())}"
        )
    
    return configs[backbone]

