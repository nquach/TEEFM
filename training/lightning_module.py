"""
PyTorch Lightning Module for VideoMAE Training

This module implements the training and validation logic for VideoMAE using
PyTorch Lightning framework. It supports:
- AdamWScheduleFree optimizer
- Optional gradient norm tracking
- Train/validation loss logging
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional
from schedulefree import AdamWScheduleFree


class VideoMAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE training.
    
    Args:
        model (nn.Module): VideoMAE model instance
        learning_rate (float): Learning rate for optimizer (default: 1e-4)
        weight_decay (float): Weight decay for optimizer (default: 0.05)
        warmup_steps (int): Number of warmup steps for optimizer (default: 1000)
        track_grad_norm (bool): Whether to track L2 norm of gradients (default: False)
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        track_grad_norm: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.track_grad_norm = track_grad_norm
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            
        Returns:
            Tuple of (loss, predictions, mask)
        """
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of video tensors (B, C, T, H, W)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        # Forward pass
        loss, pred, mask = self.model(batch)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Track gradient norm if enabled
        if self.track_grad_norm:
            grad_norm = self.compute_grad_norm()
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch of video tensors (B, C, T, H, W)
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        # Forward pass
        loss, pred, mask = self.model(batch)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer using AdamWScheduleFree.
        
        Returns:
            Optimizer instance
        """
        optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps
        )
        
        return optimizer
    
    def compute_grad_norm(self) -> torch.Tensor:
        """
        Compute L2 norm of the total loss gradient.
        
        Returns:
            L2 norm of all gradients
        """
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        return torch.tensor(total_norm, device=self.device)

