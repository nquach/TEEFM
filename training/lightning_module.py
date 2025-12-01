"""
PyTorch Lightning Module for VideoMAE Training

This module integrates VideoMAE model, EVEREST masking, and training logic
into a PyTorch Lightning module for easy multi-GPU training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
import numpy as np

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.videomae import VideoMAE
from models.everest_masking import EVERESTMaskingGenerator


class VideoMAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training VideoMAE with EVEREST masking.
    
    This module handles:
    - Model initialization
    - Forward pass with EVEREST masking
    - Loss computation (MSE on normalized patches)
    - Optimizer configuration (AdamWScheduleFree)
    - Learning rate scheduling
    - Logging and metrics
    """
    
    def __init__(
        self,
        # Model parameters
        backbone: str = 'ViT-S',
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        # Masking parameters
        mask_ratio: float = 0.9,
        motion_weight: float = 0.7,
        # Training parameters
        learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 40,
        max_epochs: int = 800,
        norm_pix_loss: bool = True,  # Normalize patches before computing loss
        # Optimizer parameters (for AdamWScheduleFree)
        beta1: float = 0.9,
        beta2: float = 0.95,
        warmup_steps: Optional[int] = None,
        # Pretrained weights
        pretrained_checkpoint: Optional[str] = None,  # Path to pretrained checkpoint
        load_pretrained_strict: bool = True,  # Strict loading mode
        # Logging
        log_gradient_norm: bool = False  # Log L2 norm of full loss gradient
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize VideoMAE model
        self.model = VideoMAE(
            backbone=backbone,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            dropout=dropout,
            drop_path=drop_path
        )
        
        # Load pretrained weights if provided
        if pretrained_checkpoint is not None:
            self.model.load_pretrained(
                pretrained_checkpoint,
                strict=load_pretrained_strict
            )
        
        # Initialize EVEREST masking generator
        self.masking_generator = EVERESTMaskingGenerator(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            mask_ratio=mask_ratio,
            motion_weight=motion_weight
        )
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.norm_pix_loss = norm_pix_loss
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_steps = warmup_steps
        self.log_gradient_norm = log_gradient_norm
        
        # Loss function (MSE)
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            mask: Optional pre-computed mask (if None, generates EVEREST mask)
            
        Returns:
            Dictionary with predictions and targets
        """
        # Generate mask if not provided
        if mask is None:
            mask = self.masking_generator(x)
        
        # Forward through model
        pred, target = self.model(x, mask)
        
        return {
            'pred': pred,
            'target': target,
            'mask': mask
        }
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred: Predicted patches of shape (B, num_patches, patch_size^2 * 3)
            target: Target patches of shape (B, num_patches, patch_size^2 * 3)
            mask: Binary mask of shape (B, num_patches) - 1 for visible, 0 for masked
            
        Returns:
            Scalar loss value
        """
        # Only compute loss on masked patches (where mask == 0)
        # In VideoMAE, we predict masked patches, so loss is on masked positions
        loss_per_patch = self.criterion(pred, target)  # (B, num_patches, patch_size^2 * 3)
        loss_per_patch = loss_per_patch.mean(dim=-1)  # (B, num_patches) - mean over patch pixels
        
        # Normalize patches if specified (per-patch normalization)
        if self.norm_pix_loss:
            # Normalize target patches
            target_mean = target.mean(dim=-1, keepdim=True)  # (B, num_patches, 1)
            target_std = target.std(dim=-1, keepdim=True) + 1e-6  # (B, num_patches, 1)
            target_norm = (target - target_mean) / target_std
            
            # Normalize predicted patches
            pred_mean = pred.mean(dim=-1, keepdim=True)
            pred_std = pred.std(dim=-1, keepdim=True) + 1e-6
            pred_norm = (pred - pred_mean) / pred_std
            
            # Recompute loss on normalized patches
            loss_per_patch = self.criterion(pred_norm, target_norm).mean(dim=-1)
        
        # Only compute loss on masked patches (mask == 0 means masked)
        mask_loss = (1 - mask)  # Invert: 1 for masked, 0 for visible
        loss = (loss_per_patch * mask_loss).sum() / mask_loss.sum()
        
        return loss
    
    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Video tensor of shape (B, T, H, W, C)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self.forward(batch)
        pred = outputs['pred']
        target = outputs['target']
        mask = outputs['mask']
        
        # Compute loss
        loss = self.compute_loss(pred, target, mask)
        
        # Logging
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # Log mask ratio (should be close to mask_ratio)
        mask_ratio_actual = 1.0 - mask.mean()
        self.log(
            'train/mask_ratio',
            mask_ratio_actual,
            on_step=False,
            on_epoch=True,
            logger=True
        )
        
        return loss
    
    def on_after_backward(self):
        """Called after backward pass. Log gradient norm if enabled."""
        if self.log_gradient_norm:
            # Compute L2 norm of all gradients
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            self.log(
                'train/gradient_norm',
                total_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True
            )
    
    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step (optional, for monitoring).
        
        Args:
            batch: Video tensor
            batch_idx: Batch index
            
        Returns:
            Dictionary with validation metrics
        """
        # Forward pass
        outputs = self.forward(batch)
        pred = outputs['pred']
        target = outputs['target']
        mask = outputs['mask']
        
        # Compute loss
        loss = self.compute_loss(pred, target, mask)
        
        # Logging
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses AdamWScheduleFree optimizer which doesn't require a scheduler,
        but we can still use warmup if needed.
        """
        try:
            from schedulefree import AdamWScheduleFree
            
            # Create optimizer
            optimizer = AdamWScheduleFree(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )
            
            # AdamWScheduleFree doesn't need a scheduler, but we can add warmup
            # For now, return optimizer directly
            return optimizer
            
        except ImportError:
            # Fallback to standard AdamW if schedulefree is not available
            print("Warning: schedulefree not available, using AdamW instead")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )
            
            # Add cosine annealing scheduler
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            if hasattr(opt, 'lr'):
                self.log('train/lr', opt.lr, on_step=False, on_epoch=True)
            elif hasattr(opt, 'param_groups'):
                self.log('train/lr', opt.param_groups[0]['lr'], on_step=False, on_epoch=True)

