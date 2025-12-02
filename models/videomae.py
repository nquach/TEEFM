"""
VideoMAE Model Implementation

This module implements the Video Masked Autoencoder (VideoMAE) model with Vision Transformer
backbone. Supports ViT-S, ViT-B, and ViT-L backbones.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from functools import partial


class PatchEmbed(nn.Module):
    """
    3D Patch Embedding module for video inputs.
    Converts video patches into embeddings.
    
    Args:
        img_size (int): Size of input image (assumed square)
        patch_size (int): Size of each patch (assumed square)
        in_chans (int): Number of input channels (default: 3 for RGB)
        embed_dim (int): Embedding dimension
        t_patch_size (int): Temporal patch size (default: 2)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        t_patch_size: int = 2
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        
        # Calculate number of patches per dimension
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_frames = 16  # After temporal downsampling
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(t_patch_size, patch_size, patch_size),
            stride=(t_patch_size, patch_size, patch_size)
        )
        
        # Calculate total number of patches
        self.num_patches = (self.num_frames // t_patch_size) * self.num_patches_per_frame
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Embedded patches of shape (B, num_patches, embed_dim)
        """
        B, C, T, H, W = x.shape
        
        # Apply 3D convolution
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, embed_dim, T_new, H_new, W_new = x.shape
        
        # Flatten spatial and temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for video patches.
    """
    
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_pos_embed()
    
    def _init_pos_embed(self):
        """Initialize positional embeddings."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Use MultiheadAttention with batch_first if available (PyTorch >= 1.9)
        # Otherwise, we'll transpose in forward pass
        try:
            self.attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=attn_drop,
                bias=qkv_bias,
                batch_first=True
            )
            self.batch_first = True
        except TypeError:
            # Fallback for older PyTorch versions
            self.attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=attn_drop,
                bias=qkv_bias
            )
            self.batch_first = False
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual
        x_norm = self.norm1(x)
        if self.batch_first:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        else:
            # Transpose for older PyTorch versions: (B, N, D) -> (N, B, D)
            x_norm_t = x_norm.transpose(0, 1)
            attn_out, _ = self.attn(x_norm_t, x_norm_t, x_norm_t)
            attn_out = attn_out.transpose(0, 1)  # Back to (B, N, D)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone for VideoMAE.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        t_patch_size: int = 2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            t_patch_size=t_patch_size
        )
        
        # Learnable class token (for compatibility, though not used in MAE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        num_patches = self.patch_embed.num_patches + 1  # +1 for cls token
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_fn)
    
    def _init_weights_fn(self, m):
        """Initialize weights for a module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            mask: Optional mask tensor for masked tokens (B, num_patches)
            
        Returns:
            Output features of shape (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional encoding
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        # Apply mask if provided (for masked tokens, set to zero)
        if mask is not None:
            # Expand mask to include cls token (not masked)
            mask_expanded = torch.cat([torch.zeros(B, 1, device=mask.device), mask], dim=1)
            mask_expanded = mask_expanded.unsqueeze(-1)  # (B, num_patches + 1, 1)
            x = x * (1 - mask_expanded)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class VideoMAE(nn.Module):
    """
    Video Masked Autoencoder (VideoMAE) model.
    
    This model implements the VideoMAE architecture with EVEREST training method.
    The model masks a portion of video patches and learns to reconstruct them.
    
    Args:
        backbone (str): Backbone architecture ('vit_s', 'vit_b', or 'vit_l')
        img_size (int): Input image size (default: 224)
        patch_size (int): Patch size (default: 16)
        mask_ratio (float): Ratio of patches to mask (default: 0.75)
        norm_pix_loss (bool): Whether to normalize pixel loss (default: True)
        pretrained (Optional[str]): Path to pretrained weights (default: None)
    """
    
    # Backbone configurations
    BACKBONE_CONFIGS = {
        'vit_s': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0
        },
        'vit_b': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0
        },
        'vit_l': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0
        }
    }
    
    def __init__(
        self,
        backbone: str = 'vit_s',
        img_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        pretrained: Optional[str] = None
    ):
        super().__init__()
        
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(f"Invalid backbone: {backbone}. Choose from {list(self.BACKBONE_CONFIGS.keys())}")
        
        self.backbone_name = backbone
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Get backbone configuration
        config = self.BACKBONE_CONFIGS[backbone]
        
        # Encoder: Vision Transformer
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio']
        )
        
        # Decoder: Lightweight transformer for reconstruction
        decoder_embed_dim = config['embed_dim']
        decoder_depth = 4  # Shallow decoder
        decoder_num_heads = config['num_heads']
        
        # Decoder patch embedding (same as encoder)
        self.decoder_embed = nn.Linear(config['embed_dim'], decoder_embed_dim)
        
        # Decoder positional encoding
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder_pos_embed = PositionalEncoding(num_patches + 1, decoder_embed_dim)
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=config['mlp_ratio'],
                norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: reconstruct pixels
        t_patch_size = 2
        patch_pixels = t_patch_size * patch_size * patch_size * 3
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_pixels)
        
        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights from checkpoint.
        
        Args:
            checkpoint_path (str): Path to pretrained checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        # Load weights, ignoring mismatched keys
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking for EVEREST training.
        
        Args:
            x: Input patches (B, num_patches, embed_dim)
            mask_ratio: Ratio of patches to mask
            
        Returns:
            visible_patches: Visible (unmasked) patches
            mask: Binary mask (1 for masked, 0 for visible)
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        
        # Number of patches to keep (visible)
        len_keep = int(N * (1 - mask_ratio))
        
        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Get visible patches
        visible_patches = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Create mask (1 for masked, 0 for visible)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return visible_patches, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            mask_ratio: Ratio of patches to mask
            
        Returns:
            encoded_patches: Encoded patches
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        # Get patches from encoder (without masking in encoder forward)
        patches = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Apply masking
        visible_patches, mask, ids_restore = self.random_masking(patches, mask_ratio)
        
        # Add class token
        B = visible_patches.shape[0]
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        visible_patches = torch.cat([cls_tokens, visible_patches], dim=1)
        
        # Add positional encoding
        visible_patches = self.encoder.pos_embed(visible_patches)
        visible_patches = self.encoder.pos_drop(visible_patches)
        
        # Apply encoder blocks
        for block in self.encoder.blocks:
            visible_patches = block(visible_patches)
        
        visible_patches = self.encoder.norm(visible_patches)
        
        # Remove class token for decoder
        encoded_patches = visible_patches[:, 1:, :]
        
        return encoded_patches, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded visible patches (B, len_keep, embed_dim)
            ids_restore: Indices to restore original order
            
        Returns:
            Reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Restore all patches (masked tokens are zeros)
        B, len_keep, D = x.shape
        num_patches = self.encoder.patch_embed.num_patches
        
        # Create full sequence with masked tokens
        x_full = torch.zeros(B, num_patches, D, device=x.device, dtype=x.dtype)
        x_full[:, :len_keep] = x
        
        # Restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        
        # Add class token
        cls_token = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([cls_token, x_full], dim=1)
        
        # Add positional encoding
        x_full = self.decoder_pos_embed(x_full)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
        
        x_full = self.decoder_norm(x_full)
        
        # Remove class token
        x_full = x_full[:, 1:, :]
        
        # Predict pixels
        pred = self.decoder_pred(x_full)
        
        return pred
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VideoMAE.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            
        Returns:
            loss: Reconstruction loss
            pred: Predicted patches
            mask: Binary mask
        """
        # Encoder forward with masking
        encoded_patches, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        
        # Decoder forward
        pred = self.forward_decoder(encoded_patches, ids_restore)
        
        # Get target patches for loss computation
        target = self.patchify(x)
        
        # Compute loss on masked patches only
        loss = self.compute_loss(pred, target, mask)
        
        return loss, pred, mask
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert video to patches (same as patch embedding but without projection).
        
        Args:
            x: Input video (B, C, T, H, W)
            
        Returns:
            Patches (B, num_patches, patch_pixels)
        """
        B, C, T, H, W = x.shape
        t_patch_size = 2
        patch_size = self.patch_size
        
        # Reshape to patches
        x = x.reshape(B, C, T // t_patch_size, t_patch_size, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, T', H', W', C, t_p, p, p)
        x = x.reshape(B, -1, C * t_patch_size * patch_size * patch_size)
        
        return x
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.
        
        Args:
            pred: Predicted patches (B, num_patches, patch_pixels)
            target: Target patches (B, num_patches, patch_pixels)
            mask: Binary mask (B, num_patches)
            
        Returns:
            Mean squared error loss on masked patches
        """
        if self.norm_pix_loss:
            # Normalize target patches
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, num_patches)
        
        # Apply mask (1 for masked, 0 for visible)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

