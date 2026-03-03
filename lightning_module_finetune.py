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


def _compute_weighted_metrics(logits, targets, num_classes):
    """Compute weighted AUROC, F1, accuracy, and mean loss (numpy)."""
    probs = softmax(logits, axis=1)
    preds = np.argmax(logits, axis=1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(
        torch.from_numpy(logits).float(),
        torch.from_numpy(targets).long(),
    ).item()
    try:
        if num_classes == 2:
            auroc = roc_auc_score(targets, probs[:, 1], average="weighted")
        else:
            auroc = roc_auc_score(
                targets, probs, multi_class="ovr", average="weighted"
            )
    except ValueError:
        auroc = float("nan")
    f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    weighted_acc = recall_score(
        targets, preds, average="weighted", zero_division=0
    )
    return {"loss": loss, "weighted_auroc": auroc, "weighted_f1": f1, "weighted_acc": weighted_acc}


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
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self._val_logits = []
        self._val_targets = []
        self._val_losses = []
        self._test_logits = []
        self._test_targets = []
        self._test_losses = []

    def forward(self, x):
        return self.model(x)

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        targets = targets.long()
        logits = self(samples)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        targets = targets.long()
        logits = self(samples)
        loss = self.criterion(logits, targets)
        self._val_logits.append(logits.detach())
        self._val_targets.append(targets.detach())
        self._val_losses.append(loss.detach())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        self._val_logits.clear()
        self._val_targets.clear()
        self._val_losses.clear()

        logits, targets = self._gather_logits_targets(logits, targets)
        if logits is None:
            return
        metrics = _compute_weighted_metrics(
            logits.cpu().numpy(), targets.cpu().numpy(), self.num_classes
        )
        self.log("val_loss", metrics["loss"], on_epoch=True, sync_dist=True)
        self.log("val_weighted_auroc", metrics["weighted_auroc"], on_epoch=True, sync_dist=True)
        self.log("val_weighted_f1", metrics["weighted_f1"], on_epoch=True, sync_dist=True)
        self.log("val_weighted_acc", metrics["weighted_acc"], on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        samples, targets = batch
        targets = targets.long()
        logits = self(samples)
        loss = self.criterion(logits, targets)
        self._test_logits.append(logits.detach())
        self._test_targets.append(targets.detach())
        self._test_losses.append(loss.detach())
        return loss

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        logits = torch.cat(self._test_logits, dim=0)
        targets = torch.cat(self._test_targets, dim=0)
        self._test_logits.clear()
        self._test_targets.clear()
        self._test_losses.clear()

        logits, targets = self._gather_logits_targets(logits, targets)
        if logits is None:
            return
        metrics = _compute_weighted_metrics(
            logits.cpu().numpy(), targets.cpu().numpy(), self.num_classes
        )
        self.log("test_loss", metrics["loss"], on_epoch=True, sync_dist=True)
        self.log("test_weighted_auroc", metrics["weighted_auroc"], on_epoch=True, sync_dist=True)
        self.log("test_weighted_f1", metrics["weighted_f1"], on_epoch=True, sync_dist=True)
        self.log("test_weighted_acc", metrics["weighted_acc"], on_epoch=True, sync_dist=True)

    def _gather_logits_targets(self, logits, targets):
        """Gather logits and targets across ranks; return concatenated or None if empty."""
        if self.trainer.world_size <= 1:
            return logits, targets
        world_size = self.trainer.world_size
        local_size = logits.shape[0]
        local_tensor = torch.tensor([local_size], device=logits.device, dtype=torch.long)
        sizes_list = [torch.zeros(1, device=logits.device, dtype=torch.long) for _ in range(world_size)]
        torch.distributed.all_gather(sizes_list, local_tensor)
        max_size = max(s[0].item() for s in sizes_list)
        if max_size == 0:
            return None, None
        if logits.shape[0] < max_size:
            pad_logits = torch.zeros(
                max_size - logits.shape[0], logits.shape[1],
                device=logits.device, dtype=logits.dtype
            )
            logits = torch.cat([logits, pad_logits], dim=0)
            pad_targets = torch.zeros(max_size - targets.shape[0], device=targets.device, dtype=targets.dtype)
            targets = torch.cat([targets, pad_targets], dim=0)
        gathered_logits = [torch.zeros_like(logits) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(targets) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_logits, logits)
        torch.distributed.all_gather(gathered_targets, targets)
        parts_logits = [gathered_logits[i][:sizes_list[i][0].item()] for i in range(world_size)]
        parts_targets = [gathered_targets[i][:sizes_list[i][0].item()] for i in range(world_size)]
        return torch.cat(parts_logits, dim=0), torch.cat(parts_targets, dim=0)

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
