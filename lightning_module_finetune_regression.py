"""
PyTorch Lightning Module for VideoMAE regression finetuning (continuous targets).

Encoder + linear head via create_videomae_finetune_model (task_type='regression'),
MSE or SmoothL1 loss, torchmetrics (MAE, RMSE, R2, Pearson), schedule-free optimizer.

Optional eval.eval_protocol: multi_clip logs val_video_* / test_video_* (mean prediction per
video_id) at epoch end. See lightning_module_finetune.py for DDP caveats.
"""

from typing import List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling.model_factory import create_videomae_finetune_model
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer


def _video_level_regression_metrics(
    video_ids: List,
    preds_rows: List[torch.Tensor],
    targets_rows: List[torch.Tensor],
):
    by_vid = defaultdict(list)
    tgt_by = {}
    for vid, pr, tg in zip(video_ids, preds_rows, targets_rows):
        if vid is None:
            continue
        by_vid[vid].append(pr.detach().cpu().numpy())
        tgt_by[vid] = tg.detach().cpu().numpy()

    if not by_vid:
        return {}

    mean_preds = []
    mean_tgts = []
    for vid in by_vid:
        stacked = np.stack(by_vid[vid], axis=0)
        mean_preds.append(np.mean(stacked, axis=0))
        mean_tgts.append(tgt_by[vid])

    y_pred = np.stack(mean_preds, axis=0)
    y_true = np.stack(mean_tgts, axis=0)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    if y_true.shape[0] < 2:
        pearson = float("nan")
    elif y_true.shape[1] == 1:
        pearson = float(np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
    else:
        corrs = []
        for j in range(y_true.shape[1]):
            c = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        pearson = float(np.mean(corrs)) if corrs else float("nan")

    return {
        "video_mae": float(mae),
        "video_rmse": float(rmse),
        "video_r2": float(r2),
        "video_pearson": pearson,
    }


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
        eval_cfg = config.get("eval") or {}
        self._eval_multi_clip = (
            (eval_cfg.get("eval_protocol") or "single_clip").lower() == "multi_clip"
        )

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

        self._val_vid_ids: List = []
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._test_vid_ids: List = []
        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

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

    def on_fit_start(self):
        if self._eval_multi_clip and self.trainer.is_global_zero:
            if "ddp" in str(type(self.trainer.strategy)).lower():
                print(
                    "eval_protocol=multi_clip: video-level regression metrics are per-rank from local "
                    "clips; use single-GPU validation for global video-level scores."
                )

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def on_validation_epoch_start(self):
        self._val_vid_ids.clear()
        self._val_preds.clear()
        self._val_targets.clear()

    def on_test_epoch_start(self):
        self._test_vid_ids.clear()
        self._test_preds.clear()
        self._test_targets.clear()

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def _append_video_clip_buffer(self, split: str, vids, pred, targets):
        if not self._eval_multi_clip or vids is None:
            return
        n = pred.shape[0]
        for i in range(n):
            vid = vids[i]
            if vid is None:
                continue
            if split == "val":
                self._val_vid_ids.append(vid)
                self._val_preds.append(pred[i].detach().cpu())
                self._val_targets.append(targets[i].detach().cpu())
            else:
                self._test_vid_ids.append(vid)
                self._test_preds.append(pred[i].detach().cpu())
                self._test_targets.append(targets[i].detach().cpu())

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
        videos, targets, vids = self._unpack_batch(batch)
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
        self._append_video_clip_buffer("val", vids, pred, targets)

    def test_step(self, batch, batch_idx):
        videos, targets, vids = self._unpack_batch(batch)
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
        self._append_video_clip_buffer("test", vids, pred, targets)

    def on_validation_epoch_end(self):
        if not self._eval_multi_clip or not self._val_vid_ids:
            return
        m = _video_level_regression_metrics(
            self._val_vid_ids,
            self._val_preds,
            self._val_targets,
        )
        if not m:
            return
        sync = True
        self.log(
            "val_video_mae",
            m["video_mae"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_rmse",
            m["video_rmse"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_r2",
            m["video_r2"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_pearson",
            m["video_pearson"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )

    def on_test_epoch_end(self):
        if not self._eval_multi_clip or not self._test_vid_ids:
            return
        m = _video_level_regression_metrics(
            self._test_vid_ids,
            self._test_preds,
            self._test_targets,
        )
        if not m:
            return
        sync = True
        self.log(
            "test_video_mae",
            m["video_mae"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_rmse",
            m["video_rmse"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_r2",
            m["video_r2"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_pearson",
            m["video_pearson"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
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
