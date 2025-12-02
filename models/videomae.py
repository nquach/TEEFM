"""
VideoMAE Model with EVEREST Masking.

This module implements the Video Masked Autoencoder (VideoMAE) architecture
with EVEREST masking strategy for efficient video representation learning.

The model consists of:
1. Patch embedding layer to convert video frames to tokens
2. Temporal and positional embeddings
3. Vision Transformer encoder
4. Decoder for reconstruction
5. EVEREST masking integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from models.everest_masking import EverestMaskingGenerator


class PatchEmbed(nn.Module):
    """
    Patch embedding layer to convert video frames into tokens.
    
    Splits each frame into non-overlapping patches and projects them
    to the embedding dimension.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        """
        Initialize patch embedding layer.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to extract patches and project to embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch embedding.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Patch embeddings of shape [B, T*num_patches, embed_dim]
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames at once
        x = x.view(B * T, C, H, W)
        
        # Extract patches and project
        x = self.proj(x)  # [B*T, embed_dim, H', W'] where H'=W'=img_size//patch_size
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, embed_dim]
        
        # Reshape to separate temporal dimension
        x = x.view(B, T * self.num_patches, -1)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for spatial patches.
    """
    
    def __init__(self, num_patches: int, embed_dim: int):
        """
        Initialize positional encoding.
        
        Args:
            num_patches: Number of spatial patches per frame
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor of shape [B, T*num_patches, embed_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pos_embed.repeat(x.shape[0], x.shape[1] // self.pos_embed.shape[1], 1)


class TemporalEmbedding(nn.Module):
    """
    Learnable temporal embedding for video frames.
    """
    
    def __init__(self, num_frames: int, embed_dim: int):
        """
        Initialize temporal embedding.
        
        Args:
            num_frames: Number of frames in the video
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
    
    def forward(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Add temporal embedding.
        
        Args:
            x: Input tensor of shape [B, T*num_patches, embed_dim]
            num_patches: Number of spatial patches per frame
            
        Returns:
            Tensor with temporal encoding added
        """
        B, _, embed_dim = x.shape
        T = self.temp_embed.shape[1]
        
        # Reshape to separate temporal dimension
        x = x.view(B, T, num_patches, embed_dim)
        
        # Add temporal embedding
        x = x + self.temp_embed.unsqueeze(2)  # [B, T, num_patches, embed_dim]
        
        # Reshape back
        x = x.view(B, T * num_patches, embed_dim)
        
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop_rate,
            bias=qkv_bias,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape [B, N, embed_dim]
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for processing video tokens.
    """
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0
    ):
        """
        Initialize ViT encoder.
        
        Args:
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                drop_rate,
                attn_drop_rate
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tokens of shape [B, N, embed_dim]
            
        Returns:
            Encoded tokens of same shape
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class VideoMAEDecoder(nn.Module):
    """
    Decoder for reconstructing masked patches.
    """
    
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        num_patches: int = 196
    ):
        """
        Initialize decoder.
        
        Args:
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of decoder attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            patch_size: Size of each patch
            num_patches: Number of spatial patches per frame
        """
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: reconstruct pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * 3,  # RGB values for each patch
            bias=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        num_patches: int
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Visible tokens from encoder [B, num_keep, embed_dim]
            ids_restore: Indices to restore original token order [B, num_tokens]
            num_patches: Number of spatial patches per frame
            
        Returns:
            Reconstructed patches [B, num_tokens, patch_size*patch_size*3]
        """
        B = x.shape[0]
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # [B, num_keep, decoder_embed_dim]
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # [B, num_tokens, decoder_embed_dim]
        
        # Restore original token order
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        
        # Add positional embedding
        # Reshape to separate temporal dimension if needed
        T = ids_restore.shape[1] // num_patches
        x = x.view(B, T, num_patches, -1)
        x = x + self.decoder_pos_embed.unsqueeze(1)  # [B, T, num_patches, decoder_embed_dim]
        x = x.view(B, T * num_patches, -1)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)  # [B, num_tokens, patch_size*patch_size*3]
        
        return x


class VideoMAE(nn.Module):
    """
    VideoMAE model with EVEREST masking.
    
    Architecture:
    1. Patch embedding
    2. Temporal and positional embeddings
    3. EVEREST masking
    4. ViT encoder (processes visible tokens only)
    5. Decoder (reconstructs all tokens)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        backbone: str = 'ViT-S',
        mask_ratio: float = 0.75,
        pretrained_weights: Optional[str] = None
    ):
        """
        Initialize VideoMAE model.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch
            num_frames: Number of frames in video sequence
            backbone: Backbone architecture ('ViT-S', 'ViT-B', or 'ViT-L')
            mask_ratio: Ratio of tokens to mask
            pretrained_weights: Path to pretrained weights checkpoint (optional)
        """
        super().__init__()
        
        # Backbone configuration
        backbone_configs = {
            'ViT-S': {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'mlp_ratio': 4.0
            },
            'ViT-B': {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0
            },
            'ViT-L': {
                'embed_dim': 1024,
                'depth': 24,
                'num_heads': 16,
                'mlp_ratio': 4.0
            }
        }
        
        if backbone not in backbone_configs:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(backbone_configs.keys())}")
        
        config = backbone_configs[backbone]
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config['mlp_ratio']
        
        # Model parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        self.num_tokens = num_frames * self.num_patches
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, self.embed_dim)
        
        # Positional and temporal embeddings
        self.pos_embed = PositionalEncoding(self.num_patches, self.embed_dim)
        self.temp_embed = TemporalEmbedding(num_frames, self.embed_dim)
        
        # Class token (optional, for classification tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Encoder
        self.encoder = VisionTransformerEncoder(
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio
        )
        
        # Decoder
        decoder_embed_dim = 512
        decoder_depth = 8
        decoder_num_heads = 16
        self.decoder = VideoMAEDecoder(
            self.embed_dim,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            self.mlp_ratio,
            patch_size,
            self.num_patches
        )
        
        # EVEREST masking generator
        self.masking_generator = EverestMaskingGenerator(
            input_size=(num_frames, img_size, img_size),
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self.load_pretrained(pretrained_weights)
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_embed.temp_embed, std=0.02)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize decoder mask token
        nn.init.trunc_normal_(self.decoder.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder.decoder_pos_embed, std=0.02)
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through VideoMAE.
        
        Args:
            x: Input video tensor of shape [B, T, C, H, W]
            return_mask: Whether to return the mask
            
        Returns:
            Tuple of (reconstructed_patches, mask) if return_mask=True,
            else just reconstructed_patches
        """
        B, T, C, H, W = x.shape
        
        # Generate EVEREST mask from original video (for motion computation)
        mask = self.masking_generator.generate_mask(x)  # [B, num_tokens]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, T*num_patches, embed_dim]
        
        # Add positional and temporal embeddings
        x = self.pos_embed(x)
        x = self.temp_embed(x, self.num_patches)
        
        # Apply mask: separate visible tokens
        visible_tokens, ids_restore = self.masking_generator.apply_mask(x, mask)
        
        # Encoder: process only visible tokens
        encoded_tokens = self.encoder(visible_tokens)  # [B, num_keep, embed_dim]
        
        # Decoder: reconstruct all tokens
        reconstructed = self.decoder(encoded_tokens, ids_restore, self.num_patches)
        # [B, num_tokens, patch_size*patch_size*3]
        
        if return_mask:
            return reconstructed, mask
        return reconstructed

