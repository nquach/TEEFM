"""
PyTorch Lightning Module for VideoMAE classification finetuning.

Provides training/validation/test steps, schedule-free optimizer with layer decay,
and weighted AUROC/F1/accuracy metrics with DDP-safe aggregation.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.special import softmax
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from modeling.model_factory import create_videomae_finetune_model
from optim_factory import LayerDecayValueAssigner, get_parameter_groups
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer
from torchmetrics import Accuracy, AUROC, F1Score

class VideoMAEFinetuneLightningModule(pl.LightningModule):
    """
    Lightning module for VideoMAE classification finetuning on LitData (video, label) datasets.
    Uses schedule-free RAdam/AdamW with layerwise LR decay.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        self.training_config = config.get("training", {})
        self.optimizer_config = config.get("optimizer", {})
        self.num_classes = self.data_config.get("num_classes", 101)

        self.model = create_videomae_finetune_model(
            backbone=self.model_config.get("backbone", "vit-b"),
            pretrained_path=self.model_config.get("pretrained_path"),
            num_classes=self.num_classes,
            num_frames=self.model_config.get("num_frames", 16),
            tubelet_size=self.model_config.get("tubelet_size", 2),
            input_size=self.model_config.get("input_size", 224),
            fc_drop_rate=self.model_config.get("fc_drop_rate", 0.0),
            drop_rate=self.model_config.get("drop_rate", 0.0),
            drop_path_rate=self.model_config.get("drop_path", 0.1),
            attn_drop_rate=self.model_config.get("attn_drop_rate", 0.0),
            use_checkpoint=self.model_config.get("use_checkpoint", False),
            use_mean_pooling=self.model_config.get("use_mean_pooling", True),
            init_scale=self.model_config.get("init_scale", 0.001),
            mcm=self.model_config.get("mcm", False),
            mcm_ratio=self.model_config.get("mcm_ratio", 0.4),
            model_key=self.model_config.get("model_key", "model|module"),
            model_prefix=self.model_config.get("model_prefix", ""),
            task_type='classification',
            output_dim=self.num_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.num_classes == 2:
            self.acc = Accuracy(task='binary')
            self.aucroc = AUROC(task='binary')
            self.f1 = F1Score(task='binary')
        if self.num_classes > 2:
            self.top1_acc = Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1)
            self.top3_acc = Accuracy(task='multiclass', num_classes=self.num_classes, top_k=3)
            self.aucroc = AUROC(task='multiclass', num_classes=self.num_classes, average='macro')
            self.top1_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro', top_k=1)
            self.top3_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro', top_k=3)


        self.mask_type = self.model_config.get('mask_type', 'motion-centric')


    def forward(self, x, mask=None):
        return self.model(x, mask)

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def training_step(self, batch, batch_idx):
        videos, targets, _ = batch
        targets = targets.long()
        logits = self.model(videos)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, targets, _ = batch
        targets = targets.long()
        logits = self.model(videos)
        val_loss = self.criterion(logits, targets)
        if self.num_classes == 2:
            acc = self.acc(logits, targets)
            aucroc = self.aucroc(logits, targets)
            f1 = self.f1(logits, targets)
            self.log_dict({'val_loss': val_loss, 'val_acc': acc, 'val_aucroc': aucroc, 'val_f1': f1}, prog_bar=True, logger=True)
        if self.num_classes > 2:
            acc1 = self.top1_acc(logits, targets)
            acc3 = self.top3_acc(logits, targets)
            aucroc = self.aucroc(logits, targets)
            f1_1 = self.top1_f1(logits, targets)
            f1_3 = self.top3_f1(logits, targets)
            self.log_dict({'val_loss': val_loss, 'val_top1_acc': acc1, 'val_top3_acc': acc3,
                'val_aucroc': aucroc,'val_top1_f1': f1_1, 'val_top3_f1': f1_3}, 
                prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        videos, targets, _ = batch
        targets = targets.long()
        logits = self.model(videos)
        test_loss = self.criterion(logits, targets)
        if self.num_classes == 2:
            acc = self.acc(logits, targets)
            aucroc = self.aucroc(logits, targets)
            f1 = self.f1(logits, targets)
            self.log_dict({'test_loss': test_loss, 'test_acc': acc, 'test_aucroc': aucroc, 'test_f1': f1}, prog_bar=True, logger=True)
        if self.num_classes > 2:
            acc1 = self.top1_acc(logits, targets)
            acc3 = self.top3_acc(logits, targets)
            aucroc = self.aucroc(logits, targets)
            f1_1 = self.top1_f1(logits, targets)
            f1_3 = self.top3_f1(logits, targets)
            self.log_dict({'test_loss': test_loss, 'test_top1_acc': acc1, 'test_top3_acc': acc3,
                'test_aucroc': aucroc,'test_top1_f1': f1_1, 'test_top3_f1': f1_3}, 
                prog_bar=True, logger=True)

    def configure_optimizers(self):
        base_lr = self.optimizer_config.get("lr", 1e-3)
        if isinstance(base_lr, (list, tuple)):
            base_lr = float(base_lr[0]) if base_lr else 1e-3
        else:
            base_lr = float(base_lr)
        weight_decay = self.optimizer_config.get("weight_decay", 0.05)
        if isinstance(weight_decay, (list, tuple)):
            weight_decay = float(weight_decay[0]) if weight_decay else 0.05
        else:
            weight_decay = float(weight_decay)
        eps = self.optimizer_config.get("eps", 1e-8)
        if isinstance(eps, (list, tuple)):
            eps = float(eps[0]) if eps else 1e-8
        else:
            eps = float(eps)
        warmup_steps = self.optimizer_config.get("warmup_steps", 0)
        if isinstance(warmup_steps, (list, tuple)):
            warmup_steps = int(warmup_steps[0]) if warmup_steps else 0
        else:
            warmup_steps = int(warmup_steps)
        betas = self.optimizer_config.get("betas", [0.9, 0.95])
        if isinstance(betas, (list, tuple)):
            betas = tuple(float(b) for b in betas)
        else:
            betas = (0.9, 0.95)

        num_layers = self.model.get_num_layers()
        layer_decay = self.optimizer_config.get("layer_decay", 0.75)
        if isinstance(layer_decay, (list, tuple)):
            layer_decay = float(layer_decay[0]) if layer_decay else 0.75
        else:
            layer_decay = float(layer_decay)

        if layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
            )
            param_groups_raw = get_parameter_groups(
                self.model,
                weight_decay,
                self.model.no_weight_decay(),
                assigner.get_layer_id,
                assigner.get_scale,
            )
            param_groups = [
                {"params": g["params"], "lr": base_lr * g["lr_scale"], "weight_decay": g["weight_decay"]}
                for g in param_groups_raw
            ]
            optimizer = create_schedule_free_optimizer(
                optimizer_type=self.optimizer_config.get("type", "radam"),
                param_groups=param_groups,
                betas=betas,
                eps=eps,
                warmup_steps=warmup_steps,
            )
        else:
            optimizer = create_schedule_free_optimizer(
                model=self.model,
                optimizer_type=self.optimizer_config.get("type", "radam"),
                lr=base_lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                warmup_steps=warmup_steps,
            )
        return optimizer
