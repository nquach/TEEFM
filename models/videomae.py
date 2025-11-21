"""
VideoMAE Model Implementation

This module implements the Video Masked Autoencoder (VideoMAE) architecture
for self-supervised video representation learning. It uses a Vision Transformer
backbone to encode video patches and reconstruct masked patches.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .vit_backbone import VisionTransformer, get_vit_config


class VideoMAEEncoder(nn.Module):
    """
    VideoMAE Encoder using Vision Transformer.
    
    Encodes visible (unmasked) video patches into latent representations.
    """
    
    def __init__(
        self,
        backbone: str = 'ViT-S',
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Get ViT configuration
        config = get_vit_config(backbone)
        
        # Build ViT encoder
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout=dropout,
            drop_path=drop_path,
            num_frames=num_frames
        )
        
        self.embed_dim = config['embed_dim']
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * num_frames
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode visible patches.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            mask: Binary mask of shape (B, num_patches) - 1 for visible, 0 for masked
            
        Returns:
            Encoded features of shape (B, num_patches+1, embed_dim)
        """
        return self.encoder(x, mask=mask)


class VideoMAEDecoder(nn.Module):
    """
    VideoMAE Decoder for reconstructing masked patches.
    
    Takes encoded visible patches and learnable mask tokens to reconstruct
    the original video patches.
    """
    
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_patches: int = 196 * 16,  # Default for 16 frames, 224x224, patch_size=16
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = num_patches
        
        # Project encoder embeddings to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Learnable mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        
        # Decoder transformer blocks
        from .vit_backbone import TransformerBlock
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=0.0
            )
            for _ in range(decoder_depth)
        ])
        
        # Final layer norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: reconstruct RGB patches
        # Each patch is patch_size x patch_size x 3
        patch_size = 16  # Default
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * 3,
            bias=True
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize prediction head
        nn.init.normal_(self.decoder_pred.weight, std=0.02)
        nn.init.zeros_(self.decoder_pred.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode and reconstruct patches.
        
        Args:
            x: Encoded features from encoder, shape (B, num_patches+1, embed_dim)
            mask: Binary mask of shape (B, num_patches) - 1 for visible, 0 for masked
            
        Returns:
            Reconstructed patches of shape (B, num_patches, patch_size^2 * 3)
        """
        B = x.shape[0]
        
        # Remove CLS token
        x = x[:, 1:, :]  # (B, num_patches, embed_dim)
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, num_patches, decoder_embed_dim)
        
        # Replace masked tokens with learnable mask tokens
        # mask: (B, num_patches) - 1 for visible, 0 for masked
        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
        x = x * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))
        
        # Add CLS token back (for consistency)
        cls_token = x.mean(dim=1, keepdim=True)  # Use mean as CLS
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches+1, decoder_embed_dim)
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final layer norm
        x = self.decoder_norm(x)
        
        # Remove CLS token
        x = x[:, 1:, :]  # (B, num_patches, decoder_embed_dim)
        
        # Predict patches
        x = self.decoder_pred(x)  # (B, num_patches, patch_size^2 * 3)
        
        return x


class VideoMAE(nn.Module):
    """
    Complete VideoMAE model with encoder and decoder.
    
    This model:
    1. Takes video patches as input
    2. Masks a portion of patches (using EVEREST strategy)
    3. Encodes visible patches
    4. Decodes to reconstruct masked patches
    5. Computes reconstruction loss
    """
    
    def __init__(
        self,
        backbone: str = 'ViT-S',
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Encoder
        self.encoder = VideoMAEEncoder(
            backbone=backbone,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            dropout=dropout,
            drop_path=drop_path
        )
        
        # Decoder
        self.decoder = VideoMAEDecoder(
            embed_dim=self.encoder.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            num_patches=self.encoder.num_patches,
            dropout=dropout
        )
        
        # Store configuration
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.num_patches = self.encoder.num_patches
        self.num_patches_per_frame = self.encoder.num_patches_per_frame
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        strict: bool = True,
        **kwargs
    ) -> 'VideoMAE':
        """
        Load VideoMAE model from pretrained checkpoint.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint file (.ckpt, .pth, or .pt)
            strict: If True, requires all keys to match. If False, allows partial loading
            **kwargs: Additional arguments to pass to VideoMAE constructor
                     (backbone, img_size, patch_size, etc.)
        
        Returns:
            VideoMAE model with loaded weights
        """
        # Create model instance
        model = cls(**kwargs)
        
        # Load pretrained weights
        model.load_pretrained(checkpoint_path, strict=strict)
        
        return model
    
    def load_pretrained(
        self,
        checkpoint_path: str,
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> None:
        """
        Load pretrained weights from checkpoint.
        
        Supports multiple checkpoint formats:
        - PyTorch Lightning checkpoint (.ckpt): Extracts 'state_dict' or 'model' key
        - PyTorch checkpoint (.pth/.pt): Direct state_dict or dict with 'state_dict' key
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, requires all keys to match. If False, allows partial loading
            map_location: Device to map weights to (e.g., 'cpu', 'cuda:0')
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading pretrained weights from {checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract state dict from different checkpoint formats
        state_dict = None
        
        # PyTorch Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (Lightning module wrapping)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        # Direct state_dict or 'model' key
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        # Assume it's already a state_dict
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        
        if state_dict is None:
            raise ValueError(
                f"Could not extract state_dict from checkpoint. "
                f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}"
            )
        
        # Load state dict
        try:
            result = self.load_state_dict(state_dict, strict=strict)
            
            # When strict=False, load_state_dict returns a NamedTuple with missing_keys and unexpected_keys
            # When strict=True, it raises RuntimeError if there are missing/unexpected keys
            if not strict and result:
                missing_keys = result.missing_keys
                unexpected_keys = result.unexpected_keys
                
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                    if len(missing_keys) <= 10:
                        print(f"  Missing keys: {missing_keys}")
                    else:
                        print(f"  First 10 missing keys: {missing_keys[:10]}")
                
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    if len(unexpected_keys) <= 10:
                        print(f"  Unexpected keys: {unexpected_keys}")
                    else:
                        print(f"  First 10 unexpected keys: {unexpected_keys[:10]}")
            
            print(f"Successfully loaded pretrained weights from {checkpoint_path}")
            
        except RuntimeError as e:
            if strict:
                raise RuntimeError(
                    f"Error loading pretrained weights with strict=True. "
                    f"Checkpoint keys may not match model architecture. "
                    f"Try setting strict=False for partial loading. "
                    f"Original error: {e}"
                ) from e
            else:
                raise
        except Exception as e:
            raise RuntimeError(
                f"Error loading pretrained weights from {checkpoint_path}: {e}"
            ) from e
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VideoMAE.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            mask: Binary mask of shape (B, num_patches) - 1 for visible, 0 for masked
            
        Returns:
            Tuple of (reconstructed_patches, target_patches)
            - reconstructed_patches: (B, num_patches, patch_size^2 * 3)
            - target_patches: (B, num_patches, patch_size^2 * 3)
        """
        # Encode visible patches
        latent = self.encoder(x, mask=mask)  # (B, num_patches+1, embed_dim)
        
        # Decode to reconstruct all patches
        pred = self.decoder(latent, mask)  # (B, num_patches, patch_size^2 * 3)
        
        # Extract target patches (for loss computation)
        target = self.patchify(x)  # (B, num_patches, patch_size^2 * 3)
        
        return pred, target
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert video frames into patches.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            
        Returns:
            Patches of shape (B, num_patches, patch_size^2 * 3)
        """
        B, T, H, W, C = x.shape
        p = self.patch_size
        
        # Reshape to process frames
        x = x.reshape(B * T, H, W, C)
        
        # Convert to patches: (B*T, H, W, C) -> (B*T, H//p, W//p, p*p*C)
        h = H // p
        w = W // p
        
        x = x.reshape(B * T, h, p, w, p, C)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B*T, h, w, p, p, C)
        x = x.reshape(B * T, h * w, p * p * C)
        
        # Reshape back to separate batch and time
        x = x.reshape(B, T * h * w, p * p * C)
        
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to video frames.
        
        Args:
            x: Patches of shape (B, num_patches, patch_size^2 * 3)
            
        Returns:
            Video tensor of shape (B, T, H, W, C)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        T = self.num_frames
        
        # Reshape patches
        x = x.reshape(B, T, h, w, p, p, 3)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, T, h, p, w, p, 3)
        x = x.reshape(B, T, h * p, w * p, 3)
        
        return x

