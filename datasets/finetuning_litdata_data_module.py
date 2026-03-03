"""
PyTorch Lightning DataModule for LitData classification finetuning.

Builds train/val/test datasets via build_litdata_finetune_datasets and
returns DataLoaders (Trainer handles DistributedSampler when using DDP).
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .litdata_labeled_dataset import build_litdata_finetune_datasets


class FinetuningLitDataDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for LitData (video, label) finetuning.
    Uses build_litdata_finetune_datasets for train/val/test split or separate dirs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.get("data", {})
        self.training_config = config.get("training", {})
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.num_classes = None

    def setup(self, stage=None):
        seed = self.training_config.get("seed", 0)
        self.train_ds, self.val_ds, self.test_ds, self.num_classes = build_litdata_finetune_datasets(
            self.data_config, seed=seed
        )

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        return DataLoader(
            self.train_ds,
            batch_size=self.training_config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.data_config.get("num_workers", 10),
            pin_memory=self.data_config.get("pin_memory", True),
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.training_config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.data_config.get("num_workers", 10),
            pin_memory=self.data_config.get("pin_memory", True),
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.training_config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.data_config.get("num_workers", 10),
            pin_memory=self.data_config.get("pin_memory", True),
        )
