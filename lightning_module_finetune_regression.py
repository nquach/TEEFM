"""
PyTorch Lightning Module for VideoMAE regression finetuning (continuous targets).

Encoder + linear head via create_videomae_finetune_model (task_type='regression'),
MSE or SmoothL1 loss, torchmetrics (MAE, RMSE, R2, Pearson), schedule-free optimizer.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef

from modeling.model_factory import create_videomae_finetune_model
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer


class VideoMAEFinetuneRegressionLightningModule(pl.LightningModule):
    """Lightning module for VideoMAE regression finetuning (e.g. age)."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.optimizer_config = config.get("optimizer", {})
        self.loss_config = config.get("loss", {})

        output_dim = int(self.data_config.get("output_dim", 1))
        self.output_dim = output_dim

        self.model = create_videomae_finetune_model(
            backbone=self.model_config.get("backbone", "vit-b"),
            pretrained_path=self.model_config.get("pretrained_path"),
            num_classes=output_dim,
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
            task_type="regression",
            output_dim=output_dim,
        )

        loss_type = (self.loss_config.get("type") or "mse").lower()
        if loss_type == "smooth_l1":
            beta = float(self.loss_config.get("beta", 1.0))
            self.criterion = nn.SmoothL1Loss(beta=beta)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"loss.type must be 'mse' or 'smooth_l1', got {loss_type!r}"
            )

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.train_r2 = R2Score(num_outputs=output_dim)
        self.val_r2 = R2Score(num_outputs=output_dim)
        self.test_r2 = R2Score(num_outputs=output_dim)
        self.train_pearson = PearsonCorrCoef(num_outputs=output_dim)
        self.val_pearson = PearsonCorrCoef(num_outputs=output_dim)
        self.test_pearson = PearsonCorrCoef(num_outputs=output_dim)

    def forward(self, x):
        return self.model(x)

    def _align_pred_target(self, pred, targets):
        targets = targets.float()
        if pred.dim() == 2 and pred.shape[1] == 1 and targets.dim() == 1:
            pred = pred.squeeze(1)
        elif pred.dim() == 1 and targets.dim() == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if pred.shape != targets.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} != target shape {tuple(targets.shape)}; "
                f"expected output_dim={self.output_dim}."
            )
        return pred, targets

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def training_step(self, batch, batch_idx):
        videos, targets = batch
        pred = self.model(videos)
        pred, targets = self._align_pred_target(pred, targets)
        loss = self.criterion(pred, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_mae(pred, targets)
        self.train_rmse(pred, targets)
        self.train_r2(pred, targets)
        self.train_pearson(pred, targets)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False, logger=True)
        self.log("train_rmse", self.train_rmse, on_epoch=True, on_step=False, logger=True)
        self.log("train_r2", self.train_r2, on_epoch=True, on_step=False, logger=True)
        self.log(
            "train_pearson",
            self.train_pearson,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        videos, targets = batch
        pred = self.model(videos)
        pred, targets = self._align_pred_target(pred, targets)
        loss = self.criterion(pred, targets)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False
        )
        self.val_mae(pred, targets)
        self.val_rmse(pred, targets)
        self.val_r2(pred, targets)
        self.val_pearson(pred, targets)
        self.log("val_mae", self.val_mae, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_rmse", self.val_rmse, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_r2", self.val_r2, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(
            "val_pearson",
            self.val_pearson,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def test_step(self, batch, batch_idx):
        videos, targets = batch
        pred = self.model(videos)
        pred, targets = self._align_pred_target(pred, targets)
        loss = self.criterion(pred, targets)
        self.log("test_loss", loss, logger=True, on_epoch=True, on_step=False)
        self.test_mae(pred, targets)
        self.test_rmse(pred, targets)
        self.test_r2(pred, targets)
        self.test_pearson(pred, targets)
        self.log("test_mae", self.test_mae, logger=True, on_epoch=True, on_step=False)
        self.log("test_rmse", self.test_rmse, logger=True, on_epoch=True, on_step=False)
        self.log("test_r2", self.test_r2, logger=True, on_epoch=True, on_step=False)
        self.log(
            "test_pearson",
            self.test_pearson,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

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
