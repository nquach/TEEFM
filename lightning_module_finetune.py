"""
PyTorch Lightning Module for VideoMAE classification finetuning.

Encoder-only classification, torchmetrics (Accuracy top-1/3, AUROC, F1),
schedule-free RAdam/AdamW. Single input (video) -> logits; no mask.

Optional eval.eval_protocol: multi_clip enables video-level metrics (mean softmax per
video_id) logged as val_video_* / test_video_* via epoch-end buffers and self.log(..., sync_dist=True).

DDP: each rank aggregates only its clips; for correct global video metrics use single-GPU val
or ensure clips per video are not split across ranks (see datasets/finetuning_data_module.py).
"""

from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassAUROC

from metrics.video_eval_metrics import video_level_classification_metrics
from modeling.model_factory import create_videomae_finetune_model
from optim_factory import build_layer_decay_param_groups
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
        eval_cfg = config.get("eval") or {}
        self._eval_multi_clip = (
            (eval_cfg.get("eval_protocol") or "single_clip").lower() == "multi_clip"
        )
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
        self.use_class_weights = bool(self.data_config.get("use_class_weights", False))
        class_weights = self._build_class_weights(self.data_config.get("class_distribution"))
        if class_weights is None:
            class_weights = torch.empty(0, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights, persistent=True)
        if num_classes == 2:
            self.acc = Accuracy(task="binary")
            self.f1 = F1Score(task="binary")
        else:
            self.top1_acc = Accuracy(
                task="multiclass", num_classes=num_classes, top_k=1
            )
            self.top3_acc = Accuracy(
                task="multiclass", num_classes=num_classes, top_k=3
            )
            self.f1 = F1Score(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            )
        self.aucroc_macro = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.aucroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None)
        self._val_vid_ids: List = []
        self._val_logits: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._test_vid_ids: List = []
        self._test_logits: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

    def _build_class_weights(self, distribution: Optional[Dict]) -> Optional[torch.Tensor]:
        if not self.use_class_weights:
            return None
        if distribution is None:
            raise ValueError(
                "data.use_class_weights is true but data.class_distribution is missing."
            )
        if not isinstance(distribution, dict):
            raise ValueError("data.class_distribution must be a mapping of class_id -> count.")

        counts = {}
        for raw_key, raw_value in distribution.items():
            try:
                class_id = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid class id in data.class_distribution: {raw_key}"
                ) from exc
            try:
                count_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid count for class {class_id}: {raw_value}"
                ) from exc
            if count_value <= 0:
                raise ValueError(
                    f"Count for class {class_id} must be > 0, got {count_value}."
                )
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(
                    f"Class id {class_id} out of range [0, {self.num_classes - 1}]."
                )
            counts[class_id] = count_value

        expected_ids = set(range(self.num_classes))
        missing = sorted(expected_ids - set(counts.keys()))
        if missing:
            raise ValueError(
                "data.class_distribution is missing class ids: "
                f"{missing}. Provide counts for all classes 0..{self.num_classes - 1}."
            )

        weights = torch.tensor(
            [1.0 / counts[class_id] for class_id in range(self.num_classes)],
            dtype=torch.float32,
        )
        weights = weights / weights.sum()
        return weights

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_class_weights and self.class_weights.numel() == self.num_classes:
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        return F.cross_entropy(logits, targets)

    def on_fit_start(self):
        if self.use_class_weights and self.class_weights.numel() == self.num_classes:
            if self.trainer.is_global_zero:
                print("Using class-weighted CrossEntropyLoss (inverse frequency, sum-normalized).")
                print(f"Class weights: {self.class_weights.detach().cpu().tolist()}")
        elif self.trainer.is_global_zero:
            print("Using unweighted CrossEntropyLoss.")
        if self._eval_multi_clip and self.trainer.is_global_zero:
            if "ddp" in str(type(self.trainer.strategy)).lower():
                print(
                    "eval_protocol=multi_clip: video-level metrics are computed per rank from local "
                    "val clips only; use a single validation device for globally correct video metrics."
                )

    def forward(self, x):
        return self.model(x)

    def on_before_optimizer_step(self, optimizer):
        actual_optimizer = optimizer
        if hasattr(optimizer, "optimizer"):
            actual_optimizer = optimizer.optimizer
        if hasattr(actual_optimizer, "train"):
            actual_optimizer.train()

    def on_validation_epoch_start(self):
        self._val_vid_ids.clear()
        self._val_logits.clear()
        self._val_targets.clear()

    def on_test_epoch_start(self):
        self._test_vid_ids.clear()
        self._test_logits.clear()
        self._test_targets.clear()

    def training_step(self, batch, batch_idx):
        videos, targets = batch
        targets = targets.long()
        logits = self.model(videos)
        loss = self._compute_loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return batch[0], batch[1].long(), batch[2]
        return batch[0], batch[1].long(), None

    def _log_per_class_auroc(self, split: str):
        scores = self.aucroc_per_class.compute()
        for i in range(scores.shape[0]):
            score = scores[i]
            val = float("nan") if torch.isnan(score) else float(score.item())
            self.log(
                f"{split}_aucroc_class_{i}",
                val,
                logger=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        self.aucroc_per_class.reset()

    def _log_video_per_class_auroc(self, split: str, per_class_scores: List[float]):
        for i, score in enumerate(per_class_scores):
            val = float(score) if score == score else float("nan")
            self.log(
                f"{split}_video_auroc_class_{i}",
                val,
                logger=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def _append_video_clip_buffer(self, split: str, vids, logits, targets):
        if not self._eval_multi_clip or vids is None:
            return
        n = logits.shape[0]
        for i in range(n):
            vid = vids[i]
            if vid is None:
                continue
            if split == "val":
                self._val_vid_ids.append(vid)
                self._val_logits.append(logits[i].detach().cpu())
                self._val_targets.append(targets[i].detach().cpu())
            else:
                self._test_vid_ids.append(vid)
                self._test_logits.append(logits[i].detach().cpu())
                self._test_targets.append(targets[i].detach().cpu())

    def validation_step(self, batch, batch_idx):
        videos, targets, vids = self._unpack_batch(batch)
        logits = self.model(videos)
        loss = self._compute_loss(logits, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        num_classes = self.num_classes
        self.aucroc_macro(logits, targets)
        self.aucroc_per_class(logits, targets)
        if num_classes == 2:
            self.acc(logits, targets)
            self.f1(logits, targets)
            self.log("val_top1_acc", self.acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_f1", self.f1, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        else:
            self.top1_acc(logits, targets)
            self.top3_acc(logits, targets)
            self.f1(logits, targets)
            self.log("val_top1_acc", self.top1_acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_top3_acc", self.top3_acc, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            self.log("val_f1", self.f1, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("val_aucroc", self.aucroc_macro, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self._append_video_clip_buffer("val", vids, logits, targets)

    def test_step(self, batch, batch_idx):
        videos, targets, vids = self._unpack_batch(batch)
        logits = self.model(videos)
        loss = self._compute_loss(logits, targets)
        self.log("test_loss", loss, logger=True, on_epoch=True, on_step=False)
        num_classes = self.num_classes
        self.aucroc_macro(logits, targets)
        self.aucroc_per_class(logits, targets)
        if num_classes == 2:
            self.acc(logits, targets)
            self.f1(logits, targets)
            self.log("test_top1_acc", self.acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_f1", self.f1, logger=True, on_epoch=True, on_step=False)
        else:
            self.top1_acc(logits, targets)
            self.top3_acc(logits, targets)
            self.f1(logits, targets)
            self.log("test_top1_acc", self.top1_acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_top3_acc", self.top3_acc, logger=True, on_epoch=True, on_step=False)
            self.log("test_f1", self.f1, logger=True, on_epoch=True, on_step=False)
        self.log("test_aucroc", self.aucroc_macro, logger=True, on_epoch=True, on_step=False)
        self._append_video_clip_buffer("test", vids, logits, targets)

    def on_validation_epoch_end(self):
        self._log_per_class_auroc("val")
        if not self._eval_multi_clip or not self._val_vid_ids:
            return
        logits_np = [z.numpy() for z in self._val_logits]
        targets_int = [int(t.item()) for t in self._val_targets]
        m = video_level_classification_metrics(
            self._val_vid_ids,
            logits_np,
            targets_int,
            self.num_classes,
        )
        if not m:
            return
        sync = True
        self.log(
            "val_video_top1_acc",
            m["video_top1"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_top3_acc",
            m["video_top3"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_auroc",
            m["video_auroc"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "val_video_f1",
            m["video_f1"],
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        if "video_auroc_per_class" in m:
            self._log_video_per_class_auroc("val", m["video_auroc_per_class"])

    def on_test_epoch_end(self):
        self._log_per_class_auroc("test")
        if not self._eval_multi_clip or not self._test_vid_ids:
            return
        logits_np = [z.numpy() for z in self._test_logits]
        targets_int = [int(t.item()) for t in self._test_targets]
        m = video_level_classification_metrics(
            self._test_vid_ids,
            logits_np,
            targets_int,
            self.num_classes,
        )
        if not m:
            return
        sync = True
        self.log(
            "test_video_top1_acc",
            m["video_top1"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_top3_acc",
            m["video_top3"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_auroc",
            m["video_auroc"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        self.log(
            "test_video_f1",
            m["video_f1"],
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=sync,
        )
        if "video_auroc_per_class" in m:
            self._log_video_per_class_auroc("test", m["video_auroc_per_class"])

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
        layer_decay = self.optimizer_config.get("layer_decay", 1.0)
        if isinstance(layer_decay, (list, tuple)):
            layer_decay = float(layer_decay[0]) if layer_decay else 1.0
        else:
            layer_decay = float(layer_decay)
        skip = (
            self.model.no_weight_decay()
            if hasattr(self.model, "no_weight_decay")
            else ()
        )
        param_groups = build_layer_decay_param_groups(
            self.model, lr, weight_decay, layer_decay, skip_list=skip
        )
        opt_type = self.optimizer_config.get("type", "radam")
        if param_groups is not None and str(opt_type).lower() == "adamw" and warmup_steps > 0:
            print(
                "Note: AdamWScheduleFree uses warmup_steps with layer-wise param groups; "
                "see schedulefree behavior if LR looks unexpected."
            )
        optimizer = create_schedule_free_optimizer(
            model=None if param_groups is not None else self.model,
            optimizer_type=opt_type,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            warmup_steps=warmup_steps,
            param_groups=param_groups,
        )
        return optimizer
