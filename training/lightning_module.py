"""
PyTorch Lightning module for VideoMAE training.

This module wraps the VideoMAE model in a LightningModule to enable
easy training with PyTorch Lightning, including multi-GPU support,
automatic checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from models.videomae import VideoMAE
from utils.optimizer import create_optimizer


class VideoMAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE training.
    
    This module handles:
    - Forward pass through VideoMAE model
    - Loss computation (MSE with normalization)
    - Gradient norm tracking (optional)
    - Optimizer configuration (AdamWScheduleFree)
    - Training and validation steps
    """
    
    def __init__(
        self,
        model: VideoMAE,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.05,
        track_gradient_norm: bool = False,
        patch_size: int = 16,
        img_size: int = 224
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model: VideoMAE model instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            track_gradient_norm: Whether to track L2 norm of gradients
            patch_size: Size of patches (for loss computation)
            img_size: Size of input images
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.track_gradient_norm = track_gradient_norm
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Compute number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor of shape [B, T, C, H, W]
            
        Returns:
            Reconstructed patches
        """
        return self.model(x)
    
    def compute_loss(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss with normalization.
        
        The loss is computed only on masked patches (for reconstruction task).
        Target patches are normalized by their mean and std before computing MSE.
        
        Args:
            reconstructed: Reconstructed patches [B, num_tokens, patch_size*patch_size*3]
            target: Target patches [B, num_tokens, patch_size*patch_size*3]
            mask: Binary mask indicating which tokens to compute loss on [B, num_tokens]
                 If None, compute loss on all tokens
            
        Returns:
            Scalar loss value
        """
        # Normalize target patches by mean and std
        # Reshape to [B, num_tokens, patch_size, patch_size, 3]
        B, num_tokens, patch_dim = target.shape
        target_reshaped = target.view(B, num_tokens, self.patch_size, self.patch_size, 3)
        reconstructed_reshaped = reconstructed.view(B, num_tokens, self.patch_size, self.patch_size, 3)
        
        # Normalize target: (x - mean) / std
        target_mean = target_reshaped.mean(dim=(2, 3, 4), keepdim=True)
        target_std = target_reshaped.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        target_normalized = (target_reshaped - target_mean) / target_std
        
        # Normalize reconstructed similarly
        reconstructed_normalized = (reconstructed_reshaped - target_mean) / target_std
        
        # Flatten back
        target_norm = target_normalized.view(B, num_tokens, patch_dim)
        reconstructed_norm = reconstructed_normalized.view(B, num_tokens, patch_dim)
        
        # Compute MSE loss
        loss = F.mse_loss(reconstructed_norm, target_norm, reduction='none')
        # loss shape: [B, num_tokens, patch_dim]
        
        # Apply mask if provided (only compute loss on masked tokens)
        if mask is not None:
            # mask: 1 = keep (visible), 0 = mask (to reconstruct)
            # We want to compute loss only on masked tokens (where mask == 0)
            mask_loss = (~mask.bool()).float()  # Invert: 1 for masked, 0 for visible
            mask_loss = mask_loss.unsqueeze(-1).expand_as(loss)
            loss = loss * mask_loss
        
        # Average over all dimensions
        loss = loss.mean()
        
        return loss
    
    def prepare_target_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare target patches from input video for loss computation.
        
        Args:
            x: Input video tensor [B, T, C, H, W]
            
        Returns:
            Target patches [B, num_tokens, patch_size*patch_size*3]
        """
        B, T, C, H, W = x.shape
        
        # Extract patches to match decoder output format
        # Process each frame separately
        patches_list = []
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            
            # Extract patches using unfold
            # unfold output: [B, C*patch_size*patch_size, num_patches]
            frame_patches = F.unfold(
                frame,
                kernel_size=self.patch_size,
                stride=self.patch_size
            )
            
            # Reshape to [B, num_patches, C, patch_size, patch_size]
            frame_patches = frame_patches.view(B, C, self.patch_size, self.patch_size, self.num_patches)
            frame_patches = frame_patches.permute(0, 4, 1, 2, 3)  # [B, num_patches, C, patch_size, patch_size]
            
            # Flatten to [B, num_patches, C*patch_size*patch_size]
            frame_patches = frame_patches.contiguous().view(B, self.num_patches, C * self.patch_size * self.patch_size)
            patches_list.append(frame_patches)
        
        # Concatenate temporal dimension: [B, T*num_patches, patch_dim]
        patches = torch.cat(patches_list, dim=1)
        
        return patches
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Input video batch [B, T, C, H, W]
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with loss and optional gradient norm
        """
        # Forward pass
        reconstructed, mask = self.model(batch, return_mask=True)
        
        # Prepare target patches
        target = self.prepare_target_patches(batch)
        
        # Compute loss
        loss = self.compute_loss(reconstructed, target, mask)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Track gradient norm if enabled
        if self.track_gradient_norm:
            # Compute L2 norm of all gradients
            total_norm = 0.0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.log('grad_norm', total_norm, on_step=True, on_epoch=False, logger=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Input video batch [B, T, C, H, W]
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with validation loss
        """
        # Forward pass
        reconstructed, mask = self.model(batch, return_mask=True)
        
        # Prepare target patches
        target = self.prepare_target_patches(batch)
        
        # Compute loss
        loss = self.compute_loss(reconstructed, target, mask)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            AdamWScheduleFree optimizer
        """
        optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return optimizer

