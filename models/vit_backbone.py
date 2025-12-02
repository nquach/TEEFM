"""
Vision Transformer (ViT) backbone implementation for VideoMAE.

This module implements the ViT encoder architecture that serves as the backbone
for the VideoMAE model. Supports ViT-S, ViT-B, and ViT-L variants.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts video patches into tokens.
    
    For video, we use 3D patches (tubelets) that span both spatial and temporal dimensions.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Spatial patch size
            tubelet_size: Temporal tubelet size
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches per frame
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        
        # 3D convolution to extract tubelets and project to embedding dimension
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch embedding.
        
        Args:
            x: Input tensor of shape (B, T, H, W, C) or (B, C, T, H, W)
            
        Returns:
            Embedded patches of shape (B, N, embed_dim) where N is number of tokens
        """
        # Handle different input formats
        if x.dim() == 5 and x.shape[-1] == 3:
            # (B, T, H, W, C) -> (B, C, T, H, W)
            x = x.permute(0, 4, 1, 2, 3)
        
        # x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # Apply 3D convolution
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial and temporal dimensions
        B, embed_dim, T_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        B, N, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Apply mask if provided (masked tokens should have very negative attention)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, embed_dim)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and MLP."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embed_dim
            dropout: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, N, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), mask)
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for VideoMAE.
    
    Supports ViT-S, ViT-B, and ViT-L architectures.
    """
    
    # Architecture configurations
    ARCHITECTURES = {
        'vit_s': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
        },
        'vit_b': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
        },
        'vit_l': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
        },
    }
    
    def __init__(
        self,
        backbone: str = 'vit_s',
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        dropout: float = 0.0,
    ):
        """
        Initialize ViT backbone.
        
        Args:
            backbone: Backbone type ('vit_s', 'vit_b', or 'vit_l')
            img_size: Input image size
            patch_size: Spatial patch size
            tubelet_size: Temporal tubelet size
            num_frames: Number of input frames
            dropout: Dropout probability
        """
        super().__init__()
        
        if backbone not in self.ARCHITECTURES:
            raise ValueError(f"Unknown backbone: {backbone}. Must be one of {list(self.ARCHITECTURES.keys())}")
        
        config = self.ARCHITECTURES[backbone]
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config['mlp_ratio']
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=3,
            embed_dim=self.embed_dim,
        )
        
        # Calculate number of tokens
        num_patches_per_frame = (img_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        self.num_tokens = num_patches_per_frame * num_temporal_patches
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        
        # Class token (for classification, but we use it for reconstruction)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=dropout,
            )
            for _ in range(self.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using standard initialization."""
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        if hasattr(self.patch_embed.proj, 'weight'):
            nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.norm1.weight, 1.0)
            nn.init.constant_(block.norm1.bias, 0.0)
            nn.init.constant_(block.norm2.weight, 1.0)
            nn.init.constant_(block.norm2.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through ViT backbone.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C) or (B, C, T, H, W)
            mask: Optional mask tensor of shape (B, N) indicating which tokens to mask
            return_all_tokens: If True, return all tokens; if False, return only unmasked tokens
            
        Returns:
            Output tokens of shape (B, N, embed_dim) or (B, N_masked, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)
        
        # Add position embedding
        pos_embed = torch.cat([
            torch.zeros(1, 1, self.embed_dim, device=x.device),
            self.pos_embed
        ], dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Return all tokens or only unmasked tokens
        if return_all_tokens:
            return x
        else:
            # Return all tokens (masking is handled in the loss computation)
            return x
