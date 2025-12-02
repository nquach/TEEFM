"""
VideoMAE model implementation with EVEREST masking.

VideoMAE (Video Masked Autoencoder) is a self-supervised learning method for
video representation learning. It uses a high masking ratio and reconstructs
the masked video patches.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .vit_backbone import ViTBackbone
from .everest_masking import EverestMasking


class VideoMAEDecoder(nn.Module):
    """
    Decoder for VideoMAE that reconstructs masked patches.
    
    The decoder is a lightweight transformer that takes the encoded visible
    tokens and mask tokens, and reconstructs the original video patches.
    """
    
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Initialize VideoMAE decoder.
        
        Args:
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        # Project encoder tokens to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder position embedding (same size as encoder)
        # We'll set this dynamically based on num_tokens
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            self._make_block(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        
        # Final layer norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: reconstruct original patches
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            embed_dim,
            bias=True
        )
        
        # Initialize mask token
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def _make_block(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> nn.Module:
        """Create a transformer block for the decoder."""
        from .vit_backbone import TransformerBlock
        return TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded tokens of shape (B, N_visible, embed_dim)
            mask: Boolean mask of shape (B, N) indicating masked tokens
            ids_restore: Indices to restore original token order
            num_tokens: Total number of tokens (N)
            
        Returns:
            Reconstructed tokens of shape (B, N, embed_dim)
        """
        B = x.shape[0]
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible, decoder_embed_dim)
        
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(B, num_tokens - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_embed_dim)
        
        # Restore original token order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        )
        
        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
        
        # Final layer norm
        x_full = self.decoder_norm(x_full)
        
        # Prediction head
        x_full = self.decoder_pred(x_full)  # (B, N, embed_dim)
        
        return x_full


class VideoMAE(nn.Module):
    """
    VideoMAE model with EVEREST masking strategy.
    
    The model consists of:
    1. ViT encoder (backbone) that processes visible tokens
    2. EVEREST masking strategy
    3. Lightweight decoder that reconstructs masked patches
    """
    
    def __init__(
        self,
        backbone: str = 'vit_s',
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        mask_ratio: float = 0.9,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        dropout: float = 0.0,
        pretrained_weights: Optional[str] = None,
    ):
        """
        Initialize VideoMAE model.
        
        Args:
            backbone: Backbone type ('vit_s', 'vit_b', or 'vit_l')
            img_size: Input image size
            patch_size: Spatial patch size
            tubelet_size: Temporal tubelet size
            num_frames: Number of input frames
            mask_ratio: Masking ratio for EVEREST
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            dropout: Dropout probability
            pretrained_weights: Path to pretrained weights (None for random init)
        """
        super().__init__()
        
        # Encoder (ViT backbone)
        self.encoder = ViTBackbone(
            backbone=backbone,
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            dropout=dropout,
        )
        
        # Calculate number of patches
        num_patches_per_frame = (img_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        self.num_tokens = num_patches_per_frame * num_temporal_patches
        
        # EVEREST masking
        self.masking = EverestMasking(
            mask_ratio=mask_ratio,
            num_frames=num_frames,
            num_patches_per_frame=num_patches_per_frame,
            tubelet_size=tubelet_size,
        )
        
        # Decoder
        self.decoder = VideoMAEDecoder(
            embed_dim=self.encoder.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
        )
        
        # Normalization for target patches (for loss computation)
        self.norm_pix_loss = True
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)
    
    def load_pretrained_weights(self, weights_path: str):
        """
        Load pretrained weights from checkpoint.
        
        Args:
            weights_path: Path to pretrained weights file
        """
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}")
            
            print(f"Loaded pretrained weights from {weights_path}")
            
        except Exception as e:
            print(f"Error loading pretrained weights from {weights_path}: {e}")
            print("Continuing with random initialization...")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ids_restore: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VideoMAE.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            mask: Optional pre-computed mask (if None, will be generated)
            ids_restore: Optional pre-computed ids_restore (if None, will be generated)
            
        Returns:
            Tuple of (latent, pred, mask):
            - latent: Encoded tokens from encoder
            - pred: Reconstructed tokens from decoder
            - mask: Boolean mask indicating masked tokens
        """
        # Generate mask if not provided
        if mask is None or ids_restore is None:
            mask, ids_restore = self.masking.generate_mask(
                batch_size=x.shape[0],
                device=x.device,
            )
        
        # Encode visible tokens only
        # First, we need to mask the input patches before encoding
        # For efficiency, we'll encode all tokens and then extract visible ones
        
        # Get patch embeddings (without masking at input level)
        # The encoder will process all tokens, but we'll use only visible ones for reconstruction
        latent_all = self.encoder(x, return_all_tokens=True)  # (B, N+1, embed_dim)
        
        # Remove class token for now (we'll add it back if needed)
        latent = latent_all[:, 1:, :]  # (B, N, embed_dim)
        
        # Get visible tokens (unmasked)
        # For each sample, extract tokens that are not masked
        B, N, D = latent.shape
        visible_tokens = []
        for i in range(B):
            visible = latent[i][~mask[i]]  # Extract unmasked tokens
            visible_tokens.append(visible)
        
        # Stack visible tokens (they may have different lengths, but in practice they're constant)
        visible_tokens = torch.stack(visible_tokens)  # (B, N_visible, D)
        
        # Decode to reconstruct all tokens
        pred = self.decoder(visible_tokens, mask, ids_restore, N)  # (B, N, embed_dim)
        
        return latent, pred, mask
    
    def forward_loss(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with loss computation.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            
        Returns:
            Tuple of (loss, pred, target):
            - loss: Reconstruction loss
            - pred: Predicted patches
            - target: Target patches (for visualization)
        """
        # Generate mask
        mask, ids_restore = self.masking.generate_mask(
            batch_size=x.shape[0],
            device=x.device,
        )
        
        # Forward pass
        latent, pred, mask = self.forward(x, mask, ids_restore)
        
        # Get target patches using the encoder's patch embedding
        # We need to get the patches without the class token
        target = self.encoder.patch_embed(x)  # (B, N, embed_dim)
        
        # Normalize target if needed
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss only on masked tokens
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        
        # Apply mask (only compute loss on masked tokens)
        loss = (loss * mask).sum() / mask.sum()  # Mean loss over masked tokens
        
        return loss, pred, target
