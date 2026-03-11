"""
PyTorch Lightning Module for VideoMAE classification finetuning.

Encoder-only classification, torchmetrics (Accuracy top-1/3, AUROC, F1),
schedule-free RAdam/AdamW. Single input (video) -> logits; no mask.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score

from modeling.model_factory import create_videomae_finetune_model
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer


class VideoMAEFinetuneLightningModule(pl.LightningModule):
    """
    Lightning module for VideoMAE classification finetuning.
    Uses create_videomae_finetune_model (encoder + head), CrossEntropyLoss,
    torchmetrics (top-1/top-3 accuracy, AUROC, F1), and schedule-free optimizer.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.optimizer_config = config.get("optimizer", {})
        num_classes = self.data_config.get("num_classes", 101)
        self.num_classes = num_classes
        self.model = create_videomae_finetune_model(
            backbone=self.model_config.get("backbone", "vit-b"),
            pretrained_path=self.model_config.get("pretrained_path"),
            num_classes=num_classes,
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
            model_key=self.model_config.get("model_key", "state_dict|model|module"),
            model_prefix=self.model_config.get("model_prefix", ""),
            task_type="classification",
            output_dim=num_classes,
        )
        self.criterion = nn.CrossEntropyLoss()
        if num_classes == 2:
            self.acc = Accuracy(task="binary")
            self.aucroc = AUROC(task="binary")
            self.f1 = F1Score(task="binary")
        else:
            self.top1_acc = Accuracy(
                task="multiclass", num_classes=num_classes, top_k=1
            )
            self.top3_acc = Accuracy(
                task="multiclass", num_classes=num_classes, top_k=3
            )
            self.aucroc = AUROC(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.f1 = F1Score(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            )

    def forward(self, x):
        return self.model(x)

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def training_step(self, batch, batch_idx):
        videos, targets = batch
        targets = targets.long()
        logits = self.model(videos)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, targets = batch
        targets = targets.long()
        logits = self.model(videos)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        num_classes = self.num_classes
        if num_classes == 2:
            self.acc(logits, targets)
            self.aucroc(logits, targets)
            self.f1(logits, targets)
            self.log("val_top1_acc", self.acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_aucroc", self.aucroc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_f1", self.f1, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        else:
            self.top1_acc(logits, targets)
            self.top3_acc(logits, targets)
            self.aucroc(logits, targets)
            self.f1(logits, targets)
            self.log("val_top1_acc", self.top1_acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_top3_acc", self.top3_acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_aucroc", self.aucroc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_f1", self.f1, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        videos, targets = batch
        targets = targets.long()
        logits = self.model(videos)
        loss = self.criterion(logits, targets)
        self.log("test_loss", loss, logger=True, on_epoch=True, on_step=False)
        num_classes = self.num_classes
        if num_classes == 2:
            self.acc(logits, targets)
            self.aucroc(logits, targets)
            self.f1(logits, targets)
            self.log("test_top1_acc", self.acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_aucroc", self.aucroc, logger=True, on_epoch=True, on_step=False)
            self.log("test_f1", self.f1, logger=True, on_epoch=True, on_step=False)
        else:
            self.top1_acc(logits, targets)
            self.top3_acc(logits, targets)
            self.aucroc(logits, targets)
            self.f1(logits, targets)
            self.log("test_top1_acc", self.top1_acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_top3_acc", self.top3_acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_aucroc", self.aucroc, logger=True, on_epoch=True, on_step=False)
            self.log("test_f1", self.f1, logger=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        lr = self.optimizer_config.get("lr", 1e-3)
        if isinstance(lr, (list, tuple)):
            lr = float(lr[0]) if lr else 1e-3
        else:
            lr = float(lr)
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
        optimizer = create_schedule_free_optimizer(
            self.model,
            optimizer_type=self.optimizer_config.get("type", "radam"),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            warmup_steps=warmup_steps,
        )
        return optimizer
