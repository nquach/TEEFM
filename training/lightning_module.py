"""
PyTorch Lightning module for VideoMAE training.

This module implements the training and validation logic using PyTorch Lightning,
which enables easy multi-GPU training and distributed training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, Any

try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    raise ImportError(
        "schedulefree is required. Please install it with: pip install schedulefree"
    )

from ..models.videomae import VideoMAE




class VideoMAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE training.
    
    This module handles:
    - Training and validation steps
    - Optimizer configuration (AdamWScheduleFree)
    - Loss computation
    - Gradient norm tracking (optional)
    - Logging metrics
    """
    
    def __init__(
        self,
        model: VideoMAE,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        loss_type: str = 'mse',
        track_gradient_norm: bool = False,
        gradient_clip_val: Optional[float] = None,
    ):
        """
        Initialize VideoMAE Lightning module.
        
        Args:
            model: VideoMAE model instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_type: Type of loss ('mse' for mean squared error)
            track_gradient_norm: Whether to track L2 norm of gradients
            gradient_clip_val: Gradient clipping value (None to disable)
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.track_gradient_norm = track_gradient_norm
        self.gradient_clip_val = gradient_clip_val
        
        # Store patchify function parameters
        self.patch_size = model.encoder.patch_embed.patch_size
        self.tubelet_size = model.encoder.patch_embed.tubelet_size
        self.embed_dim = model.encoder.embed_dim
        
        # Loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor of shape (B, T, H, W, C)
            
        Returns:
            Dictionary containing model outputs
        """
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of videos of shape (B, T, H, W, C)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass with loss computation
        loss, pred, target = self.model.forward_loss(batch)
        
        # Log training loss
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        # Track gradient norm if enabled
        if self.track_gradient_norm:
            # Compute gradient norm
            total_norm = 0.0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.log(
                    'grad_norm',
                    total_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch of videos of shape (B, T, H, W, C)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass with loss computation
        loss, pred, target = self.model.forward_loss(batch)
        
        # Log validation loss
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer (AdamWScheduleFree).
        
        Returns:
            Optimizer instance
        """
        optimizer = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """
        Hook called before optimizer step.
        Used for gradient clipping if enabled.
        """
        if self.gradient_clip_val is not None:
            # Clip gradients
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm='norm',
            )
