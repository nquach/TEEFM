"""
PyTorch Lightning Module for VideoMAE Training

This module provides a PyTorch Lightning wrapper for the VideoMAE model,
enabling easy multi-GPU training, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from modeling.model_factory import create_videomae_model
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer


class VideoMAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE pretraining.
    
    This module wraps the VideoMAE model and provides:
    - Training and validation step implementations
    - MSE loss for reconstruction
    - AdamWScheduleFree optimizer configuration
    - Optional gradient norm tracking
    - Automatic logging of metrics
    
    Args:
        config (dict): Configuration dictionary containing all training parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(config)
        
        # Extract configuration sections
        self.model_config = config['model']
        self.training_config = config['training']
        self.optimizer_config = config['optimizer']
        self.features_config = config.get('features', {})
        
        # Create model using factory function
        self.model = create_videomae_model(
            backbone=self.model_config['backbone'],
            pretrained_path=self.model_config.get('pretrained_path'),
            decoder_depth=self.model_config.get('decoder_depth', 4),
            drop_path=self.model_config.get('drop_path', 0.0),
            mask_type=self.model_config.get('mask_type', 'motion-centric'),
            mask_ratio=self.model_config.get('mask_ratio', 0.9),
            motion_centric_masking_ratio=self.model_config.get('motion_centric_masking_ratio', 0.7),
            use_checkpoint=self.features_config.get('use_checkpoint', False)
        )
        
        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()
        
        # Get patch size from model
        self.patch_size = self.model.encoder.patch_embed.patch_size[0]
        
        # Whether to normalize targets
        self.normalize_target = self.model_config.get('normalize_target', True)
        
        # Mask type
        self.mask_type = self.model_config.get('mask_type', 'motion-centric')
        
        # Whether to track gradient norm
        self.track_grad_norm = self.features_config.get('track_grad_norm', False)
        
        # Window size for masking (will be set in setup)
        self.window_size = None
    
    def setup(self, stage=None):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.
        """
        # Calculate window size based on model configuration
        num_frames = self.training_config.get('num_frames', 16)
        input_size = self.model_config.get('input_size', 224)
        patch_size = self.patch_size
        
        # Window size: (frames, height_patches, width_patches)
        self.window_size = (
            num_frames // 2,  # Temporal dimension (after tubelet_size=2)
            input_size // patch_size,  # Height patches
            input_size // patch_size   # Width patches
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input video tensor [B, C, T, H, W]
            mask (torch.Tensor, optional): Mask tensor for masking patches
        
        Returns:
            tuple: (output, masks) where output is the reconstruction and masks contains mask info
        """
        return self.model(x, mask)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.
        
        Args:
            batch: Batch of data containing (videos, masks)
            batch_idx: Index of the batch
        
        Returns:
            torch.Tensor: Loss value
        """
        videos, bool_masked_pos = batch
        
        # Handle mask based on mask type
        if self.mask_type != 'motion-centric' and bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        
        # Prepare targets (unnormalized video patches)
        with torch.no_grad():
            # Get normalization values from data config
            data_config = self.hparams.get('data', {})
            mean = torch.as_tensor(data_config.get('normalize_mean', [0.117, 0.114, 0.113])).to(videos.device)
            std = torch.as_tensor(data_config.get('normalize_std', [0.208, 0.204, 0.203])).to(videos.device)
            
            # Unnormalize videos to get original pixel values
            unnorm_videos = videos * std[None, :, None, None, None] + mean[None, :, None, None, None]
            
            # Convert to patches
            if self.normalize_target:
                # Normalize patches per patch (spatial normalization)
                videos_squeeze = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,  # tubelet_size
                    p1=self.patch_size,
                    p2=self.patch_size
                )
                videos_norm = (
                    videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,  # tubelet_size
                    p1=self.patch_size,
                    p2=self.patch_size
                )
            
            B, _, C = videos_patch.shape
            
            # Get labels based on mask type
            # For non-motion-centric, labels are extracted here
            # For motion-centric, labels are extracted after forward pass
            if self.mask_type != 'motion-centric' and bool_masked_pos is not None:
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
        
        # Forward pass
        outputs, masks = self.model(videos, bool_masked_pos)
        
        # Get labels for motion-centric masking (mask is generated by model)
        if self.mask_type == 'motion-centric':
            _, mc_target_mask = masks
            labels = videos_patch[~mc_target_mask].reshape(B, -1, C)
        
        # Compute loss
        loss = self.criterion(outputs, labels)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Track gradient norm if enabled
        if self.track_grad_norm:
            # Compute gradient norm after backward pass
            # We'll do this in on_after_backward hook instead
            pass
        
        return loss
    
    def on_after_backward(self):
        """
        Hook called after loss.backward() and before optimizer.step().
        Used to compute gradient norm if tracking is enabled.
        """
        if self.track_grad_norm:
            total_norm = 0.0
            param_count = 0
            
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** 0.5
                self.log('grad_norm', total_norm, on_step=True, on_epoch=False, logger=True)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.
        
        Args:
            batch: Batch of data containing (videos, masks)
            batch_idx: Index of the batch
        
        Returns:
            torch.Tensor: Loss value
        """
        videos, bool_masked_pos = batch
        
        # Handle mask based on mask type
        if self.mask_type != 'motion-centric' and bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        
        # Prepare targets (same as training)
        with torch.no_grad():
            # Get normalization values from data config
            data_config = self.hparams.get('data', {})
            mean = torch.as_tensor(data_config.get('normalize_mean', [0.117, 0.114, 0.113])).to(videos.device)
            std = torch.as_tensor(data_config.get('normalize_std', [0.208, 0.204, 0.203])).to(videos.device)
            
            unnorm_videos = videos * std[None, :, None, None, None] + mean[None, :, None, None, None]
            
            if self.normalize_target:
                videos_squeeze = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=self.patch_size,
                    p2=self.patch_size
                )
                videos_norm = (
                    videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(
                    unnorm_videos,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=self.patch_size,
                    p2=self.patch_size
                )
            
            B, _, C = videos_patch.shape
            
            # Get labels based on mask type
            if self.mask_type != 'motion-centric' and bool_masked_pos is not None:
                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
        
        # Forward pass
        outputs, masks = self.model(videos, bool_masked_pos)
        
        # Get labels for motion-centric masking
        if self.mask_type == 'motion-centric':
            _, mc_target_mask = masks
            labels = videos_patch[~mc_target_mask].reshape(B, -1, C)
        
        # Compute loss
        loss = self.criterion(outputs, labels)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer = create_schedule_free_optimizer(
            self.model,
            lr=self.optimizer_config.get('lr', 1.5e-4),
            weight_decay=self.optimizer_config.get('weight_decay', 0.05),
            betas=tuple(self.optimizer_config.get('betas', [0.9, 0.95])),
            eps=self.optimizer_config.get('eps', 1e-8)
        )
        
        return optimizer

