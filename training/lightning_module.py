"""
PyTorch Lightning module for VideoMAE training with EVEREST method.

This module implements the training and validation logic using PyTorch Lightning,
including the EVEREST loss computation and gradient norm tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

# Import AdamWScheduleFree optimizer
try:
    from schedule_free import AdamWScheduleFree
except ImportError:
    print("Warning: schedule_free package not found. Install with: pip install schedule-free")
    # Fallback to standard AdamW
    AdamWScheduleFree = None


class VideoMAELightning(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE training with EVEREST method.
    
    Implements masked video autoencoder training where the model learns to
    reconstruct masked video patches.
    
    Args:
        model: VideoMAE model instance.
        learning_rate (float): Learning rate for optimizer. Default: 1e-4.
        weight_decay (float): Weight decay for optimizer. Default: 0.05.
        mask_ratio (float): Ratio of patches to mask. Default: 0.75.
        norm_pix_loss (bool): Whether to normalize pixel values for loss computation. Default: True.
        track_gradient_norm (bool): Whether to track L2 norm of total loss gradient. Default: False.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        track_gradient_norm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.track_gradient_norm = track_gradient_norm
        
        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss(reduction="none")
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            
        Returns:
            pred: Reconstructed patch predictions
            mask: Binary mask (1 for visible, 0 for masked)
        """
        return self.model(x, mask_ratio=self.mask_ratio)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert video frames to patches.
        
        This is the inverse operation of unpatchify, used to prepare
        ground truth patches for loss computation.
        
        Args:
            imgs: Video tensor of shape (B, T, C, H, W)
            
        Returns:
            Patches of shape (B, N, patch_pixels)
        """
        B, T, C, H, W = imgs.shape
        p = self.model.patch_size
        t = self.model.tubelet_size
        
        # Reshape to extract patches
        h = w = H // p
        t_patches = T // t
        
        # Reshape: (B, T, C, H, W) -> (B, t_patches, t, C, h, p, w, p)
        imgs = imgs.reshape(B, t_patches, t, C, h, p, w, p)
        # Permute and reshape: -> (B, t_patches, h, w, t, C, p, p)
        imgs = imgs.permute(0, 1, 4, 6, 2, 3, 5, 7)
        # Reshape: -> (B, N, t*C*p*p) where N = t_patches * h * w
        imgs = imgs.reshape(B, t_patches * h * w, t * C * p * p)
        
        return imgs
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to video frames.
        
        Args:
            x: Patches of shape (B, N, patch_pixels)
            
        Returns:
            Video tensor of shape (B, T, C, H, W)
        """
        B, N, patch_pixels = x.shape
        p = self.model.patch_size
        t = self.model.tubelet_size
        C = 3
        H = W = self.model.img_size
        h = w = H // p
        t_patches = self.model.num_temporal_patches
        
        # Reshape patches: (B, N, t*C*p*p) -> (B, t_patches, h, w, t, C, p, p)
        x = x.reshape(B, t_patches, h, w, t, C, p, p)
        # Permute: -> (B, t_patches, t, C, h, p, w, p)
        x = x.permute(0, 1, 4, 5, 2, 6, 3, 7)
        # Reshape: -> (B, T, C, H, W)
        x = x.reshape(B, t_patches * t, C, H, W)
        
        return x
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute EVEREST reconstruction loss.
        
        The loss is computed only on masked patches (where mask == 0).
        
        Args:
            pred: Predicted patches of shape (B, N, patch_pixels)
            target: Target patches of shape (B, N, patch_pixels)
            mask: Binary mask of shape (B, N) where 1 = visible, 0 = masked
            
        Returns:
            Scalar loss value
        """
        # Normalize target patches if specified
        if self.norm_pix_loss:
            # Normalize by mean and std of each patch
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # Compute loss for each patch
        loss = self.criterion(pred, target)  # (B, N, patch_pixels)
        loss = loss.mean(dim=-1)  # (B, N) - mean over patch pixels
        
        # Apply mask: only compute loss on masked patches (mask == 0)
        mask = mask.bool()
        loss = loss * (1 - mask.float())  # Loss only for masked patches
        
        # Average over all masked patches
        loss = loss.sum() / (1 - mask.float()).sum()
        
        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Video tensor of shape (B, T, C, H, W)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Forward pass
        pred, mask = self.forward(batch)
        
        # Convert target video to patches
        target = self.patchify(batch)
        
        # Compute loss
        loss = self.compute_loss(pred, target, mask)
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Track gradient norm if enabled
        if self.track_gradient_norm:
            # Compute gradient norm after backward pass
            # We'll do this in on_after_backward hook instead
            pass
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Video tensor of shape (B, T, C, H, W)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Forward pass
        pred, mask = self.forward(batch)
        
        # Convert target video to patches
        target = self.patchify(batch)
        
        # Compute loss
        loss = self.compute_loss(pred, target, mask)
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_after_backward(self):
        """
        Hook called after backward pass.
        Used to track gradient norm if enabled.
        """
        if self.track_gradient_norm:
            # Compute L2 norm of total loss gradient
            total_norm = 0.0
            param_count = 0
            
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1.0 / 2)
            
            # Log gradient norm
            self.log("grad_norm", total_norm, on_step=True, on_epoch=False, logger=True)
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses AdamWScheduleFree if available, otherwise falls back to AdamW.
        """
        # Use AdamWScheduleFree if available
        if AdamWScheduleFree is not None:
            optimizer = AdamWScheduleFree(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return optimizer
        else:
            # Fallback to standard AdamW
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return optimizer

