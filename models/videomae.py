"""
VideoMAE model implementation with ViT backbone.

This module implements the VideoMAE (Masked Video Autoencoder) architecture
using Vision Transformer (ViT) backbones. Supports ViT-S, ViT-B, and ViT-L variants.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PatchEmbedding3D(nn.Module):
    """
    3D patch embedding for video data.
    
    Divides video into spatio-temporal patches and projects them to embedding space.
    
    Args:
        img_size (int): Spatial size of input frames (assumed square).
        patch_size (int): Size of spatial patches.
        tubelet_size (int): Size of temporal tubelets.
        in_channels (int): Number of input channels (default: 3 for RGB).
        embed_dim (int): Embedding dimension.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches per frame
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        
        # 3D convolution to extract tubelets and project to embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, N, embed_dim) where N is number of patches
        """
        B, T, C, H, W = x.shape
        
        # Reshape to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolution to extract patches
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial and temporal dimensions
        B, embed_dim, T_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        return x


class PositionalEncoding3D(nn.Module):
    """
    3D positional encoding for video patches.
    
    Adds learnable positional embeddings to patch embeddings.
    """
    
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class VideoMAE(nn.Module):
    """
    VideoMAE model with ViT backbone.
    
    Implements a masked video autoencoder using Vision Transformer architecture.
    Supports ViT-S, ViT-B, and ViT-L backbones.
    
    Args:
        backbone (str): Backbone architecture. Options: "vit_s", "vit_b", "vit_l".
        img_size (int): Spatial size of input frames. Default: 224.
        patch_size (int): Size of spatial patches. Default: 16.
        tubelet_size (int): Size of temporal tubelets. Default: 2.
        num_frames (int): Number of input frames. Default: 16.
        mask_ratio (float): Ratio of patches to mask during training. Default: 0.75.
        pretrained (bool): Whether to use pretrained weights. Default: False.
        pretrained_path (str, optional): Path to pretrained checkpoint.
    """
    
    # ViT architecture configurations
    BACKBONE_CONFIGS = {
        "vit_s": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "vit_b": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "vit_l": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        },
    }
    
    def __init__(
        self,
        backbone: str = "vit_s",
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        mask_ratio: float = 0.75,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Invalid backbone '{backbone}'. "
                f"Must be one of {list(self.BACKBONE_CONFIGS.keys())}"
            )
        
        self.backbone = backbone
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        
        # Get backbone configuration
        config = self.BACKBONE_CONFIGS[backbone]
        self.embed_dim = config["embed_dim"]
        self.depth = config["depth"]
        self.num_heads = config["num_heads"]
        self.mlp_ratio = config["mlp_ratio"]
        
        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // tubelet_size
        self.num_patches = self.num_patches_per_frame * self.num_temporal_patches
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=3,
            embed_dim=self.embed_dim,
        )
        
        # Learnable class token (for compatibility with standard ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )  # +1 for cls token
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Decoder for reconstruction (used in EVEREST training)
        decoder_embed_dim = self.embed_dim
        decoder_depth = 4
        decoder_num_heads = self.num_heads
        
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Reconstruction head: predict pixel values for each patch
        patch_pixels = patch_size * patch_size * tubelet_size * 3
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_pixels)
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained weights if specified
        if pretrained:
            if pretrained_path is not None:
                self.load_pretrained(pretrained_path)
            else:
                print("Warning: pretrained=True but no pretrained_path provided. "
                      "Using random initialization.")
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize decoder
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.norm1.weight, 1.0)
            nn.init.constant_(block.norm1.bias, 0.0)
            nn.init.constant_(block.norm2.weight, 1.0)
            nn.init.constant_(block.norm2.bias, 0.0)
        
        # Initialize decoder blocks
        for block in self.decoder_blocks:
            nn.init.constant_(block.norm1.weight, 1.0)
            nn.init.constant_(block.norm1.bias, 0.0)
            nn.init.constant_(block.norm2.weight, 1.0)
            nn.init.constant_(block.norm2.bias, 0.0)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load state dict (strict=False to allow partial loading)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}")
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches for EVEREST training.
        
        Args:
            x: Input patch embeddings of shape (B, N, embed_dim)
            
        Returns:
            x_masked: Masked embeddings with mask tokens
            mask: Binary mask (1 for visible, 0 for masked)
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Generate random noise and sort
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Generate mask tokens for masked patches
        mask_tokens = self.mask_token.expand(B, N - len_keep, -1)
        x_masked = torch.cat([x_keep, mask_tokens], dim=1)
        
        # Create binary mask (1 for visible, 0 for masked)
        mask = torch.zeros(B, N, device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Restore original order
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore_expanded)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            mask_ratio: Override default mask ratio
            
        Returns:
            latent: Encoded features
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Extract patches
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x[:, 1:])  # Remove cls token for masking
        cls_token = x[:, :1]  # Keep cls token
        x_masked = torch.cat([cls_token, x_masked], dim=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x_masked = block(x_masked)
        
        x_masked = self.norm(x_masked)
        
        return x_masked, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded features from encoder
            ids_restore: Indices to restore original patch order
            
        Returns:
            Reconstructed patch predictions
        """
        # Decoder embedding
        x = self.decoder_embed(x)
        
        # Remove cls token and restore patch order
        cls_token = x[:, :1]
        x_patches = x[:, 1:]
        B, N, D = x_patches.shape
        
        # Restore original order
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_patches = torch.gather(x_patches, dim=1, index=ids_restore_expanded)
        
        # Add cls token back
        x = torch.cat([cls_token, x_patches], dim=1)
        
        # Add decoder positional encoding
        x = x + self.decoder_pos_embed
        
        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        # Prediction head (remove cls token)
        x = x[:, 1:]
        x = self.decoder_pred(x)
        
        return x
    
    def forward(self, x: torch.Tensor, mask_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            mask_ratio: Override default mask ratio
            
        Returns:
            pred: Reconstructed patch predictions
            mask: Binary mask (1 for visible, 0 for masked)
        """
        # Encoder
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # Decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, mask

