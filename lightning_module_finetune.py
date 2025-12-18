"""
PyTorch Lightning Module for VideoMAE Finetuning

This module provides a PyTorch Lightning wrapper for the VideoMAE finetuning model,
enabling easy multi-GPU training, checkpointing, and logging for classification tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import sys

from modeling.model_factory import create_videomae_finetune_model
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer
from mixup import Mixup


class VideoMAEFinetuningLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE finetuning (classification).
    
    This module wraps the VideoMAE finetuning model and provides:
    - Training and validation/test step implementations
    - Cross-entropy loss for classification
    - Schedule-free optimizer configuration (AdamWScheduleFree or RAdamScheduleFree)
    - Support for mixup/cutmix augmentation
    - Accuracy metrics (top-1, top-5)
    - Optional gradient norm tracking
    - Automatic logging of metrics
    - Model EMA (Exponential Moving Average) support
    
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
        self.augmentation_config = config.get('augmentation', {})
        
        # Create model using factory function
        self.model = create_videomae_finetune_model(
            backbone=self.model_config['backbone'],
            pretrained_path=self.model_config.get('pretrained_path'),
            num_classes=self.model_config.get('num_classes', 101),
            num_frames=self.training_config.get('num_frames', 16),
            tubelet_size=self.model_config.get('tubelet_size', 2),
            input_size=self.model_config.get('input_size', 224),
            fc_drop_rate=self.model_config.get('fc_drop_rate', 0.0),
            drop_rate=self.model_config.get('drop_rate', 0.0),
            drop_path_rate=self.model_config.get('drop_path', 0.1),
            attn_drop_rate=self.model_config.get('attn_drop_rate', 0.0),
            use_checkpoint=self.features_config.get('use_checkpoint', False),
            use_mean_pooling=self.model_config.get('use_mean_pooling', True),
            init_scale=self.model_config.get('init_scale', 0.001),
            mcm=self.model_config.get('mcm', False),
            mcm_ratio=self.model_config.get('mcm_ratio', 0.4),
            model_key=self.model_config.get('model_key', 'model|module'),
            model_prefix=self.model_config.get('model_prefix', '')
        )
        
        # Loss function (Cross-entropy for classification)
        label_smoothing = self.augmentation_config.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            from timm.loss import LabelSmoothingCrossEntropy
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Accuracy metrics
        num_classes = self.model_config.get('num_classes', 101)
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        
        # Mixup/Cutmix setup
        mixup_alpha = self.augmentation_config.get('mixup', 0.0)
        cutmix_alpha = self.augmentation_config.get('cutmix', 0.0)
        cutmix_minmax = self.augmentation_config.get('cutmix_minmax', None)
        mixup_prob = self.augmentation_config.get('mixup_prob', 1.0)
        mixup_switch_prob = self.augmentation_config.get('mixup_switch_prob', 0.5)
        mixup_mode = self.augmentation_config.get('mixup_mode', 'batch')
        
        self.mixup_fn = None
        if mixup_alpha > 0 or cutmix_alpha > 0 or cutmix_minmax is not None:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=cutmix_minmax,
                prob=mixup_prob,
                switch_prob=mixup_switch_prob,
                mode=mixup_mode,
                label_smoothing=label_smoothing,
                num_classes=num_classes
            )
            print("Mixup/Cutmix is activated!")
        
        # Whether to track gradient norm
        self.track_grad_norm = self.features_config.get('track_grad_norm', False)
        
        # Model EMA (Exponential Moving Average)
        self.model_ema = None
        if self.model_config.get('model_ema', False):
            from timm.utils import ModelEma
            self.model_ema = ModelEma(
                self.model,
                decay=self.model_config.get('model_ema_decay', 0.9999),
                device='cpu' if self.model_config.get('model_ema_force_cpu', False) else ''
            )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input video tensor [B, C, T, H, W]
        
        Returns:
            torch.Tensor: Classification logits [B, num_classes]
        """
        return self.model(x)
    
    def on_before_optimizer_step(self, optimizer):
        """
        Hook called before each optimizer.step() call.
        Required for schedule-free optimizers to ensure they're in train mode.
        
        Args:
            optimizer: The optimizer being stepped (may be wrapped by DeepSpeed)
        """
        # Handle DeepSpeed-wrapped optimizers
        # DeepSpeed wraps the optimizer, so we need to access the underlying optimizer
        actual_optimizer = optimizer
        if hasattr(optimizer, 'optimizer'):
            # DeepSpeed wraps the optimizer in an .optimizer attribute
            actual_optimizer = optimizer.optimizer
        
        # Set optimizer to training mode before each step (required for schedule-free optimizers)
        if hasattr(actual_optimizer, 'train'):
            actual_optimizer.train()
    
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.
        
        Args:
            batch: Batch of data containing (videos, labels)
            batch_idx: Index of the batch
        
        Returns:
            torch.Tensor: Loss value
        """
        videos, targets = batch
        
        # Apply mixup/cutmix if enabled
        if self.mixup_fn is not None:
            videos, targets = self.mixup_fn(videos, targets)
        
        # Forward pass
        outputs = self.model(videos)
        
        # Compute loss
        if self.mixup_fn is not None:
            # For mixup, targets might be soft labels
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(outputs, targets)
        
        if not torch.isfinite(loss).all():
            print("Loss is infinite or NaN, stopping training")
            sys.exit(1)
        
        # Compute accuracy (only if not using mixup, as mixup uses soft labels)
        if self.mixup_fn is None:
            # Update accuracy metrics
            self.train_acc1(outputs, targets.long())
            self.train_acc5(outputs, targets.long())
            
            # Log metrics
            self.log('train_acc1', self.train_acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('train_acc5', self.train_acc5, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update model EMA after each training batch."""
        if self.model_ema is not None:
            self.model_ema.update(self.model)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.
        
        Args:
            batch: Batch of data containing (videos, labels)
            batch_idx: Index of the batch
        """
        videos, targets = batch
        
        # Forward pass
        outputs = self.model(videos)
        
        # Compute loss
        loss = self.criterion(outputs, targets)
        
        # Update accuracy metrics
        self.val_acc1(outputs, targets.long())
        self.val_acc5(outputs, targets.long())
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc1', self.val_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc5', self.val_acc5, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for one batch.
        
        Args:
            batch: Batch of data containing (videos, labels)
            batch_idx: Index of the batch
        """
        videos, targets = batch
        
        # Forward pass (use EMA model if available)
        model_to_use = self.model_ema.module if self.model_ema is not None else self.model
        outputs = model_to_use(videos)
        
        # Compute loss
        loss = self.criterion(outputs, targets)
        
        # Update accuracy metrics
        self.test_acc1(outputs, targets.long())
        self.test_acc5(outputs, targets.long())
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc1', self.test_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc5', self.test_acc5, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
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
    
    def configure_optimizers(self):
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Configured schedule-free optimizer
        """
        # Ensure lr is a float (YAML might parse it as a list in some cases)
        lr = self.optimizer_config.get('lr', 1e-3)
        if isinstance(lr, (list, tuple)):
            lr = float(lr[0]) if len(lr) > 0 else 1e-3
        else:
            lr = float(lr)
        
        # Ensure weight_decay is a float
        weight_decay = self.optimizer_config.get('weight_decay', 0.05)
        if isinstance(weight_decay, (list, tuple)):
            weight_decay = float(weight_decay[0]) if len(weight_decay) > 0 else 0.05
        else:
            weight_decay = float(weight_decay)
        
        # Ensure eps is a float
        eps = self.optimizer_config.get('eps', 1e-8)
        if isinstance(eps, (list, tuple)):
            eps = float(eps[0]) if len(eps) > 0 else 1e-8
        else:
            eps = float(eps)
        
        # Ensure warmup_steps is an int
        warmup_steps = self.optimizer_config.get('warmup_steps', 0)
        if isinstance(warmup_steps, (list, tuple)):
            warmup_steps = int(warmup_steps[0]) if len(warmup_steps) > 0 else 0
        else:
            warmup_steps = int(warmup_steps)
        
        # Ensure betas is a tuple of floats
        betas = self.optimizer_config.get('betas', [0.9, 0.95])
        if isinstance(betas, (list, tuple)):
            betas = tuple(float(b) for b in betas)
        else:
            betas = (0.9, 0.95)
        
        optimizer = create_schedule_free_optimizer(
            self.model,
            optimizer_type=self.optimizer_config.get('type', 'adamw'),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            warmup_steps=warmup_steps
        )
        
        # Store optimizer reference for train/eval mode switching
        self._optimizer = optimizer
        
        return optimizer

