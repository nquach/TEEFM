"""
PyTorch Lightning DataModule for VideoMAE finetuning (classification or regression).

Uses LitData StreamingDataset. Set data.task to 'classification' (default) or 'regression'
and optional data.target_key for the label field in each sample.

When val_optimized_dir and test_optimized_dir are not set, splits the training directory
into train/val/test via litdata.train_test_split() and split_ratio.
"""

import os
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from litdata import StreamingDataLoader, StreamingDataset, train_test_split
from litdata.streaming.cache import Dir

from .optimized_video_classification_dataset import (
    OptimizedVideoClassificationDataset,
    ClassificationTransform,
)

try:
    import botocore
    custom_storage_options = {
        "config": botocore.config.Config(
            retries={"max_attempts": 1000, "mode": "adaptive"},
            signature_version=botocore.UNSIGNED,
        )
    }
except ImportError:
    custom_storage_options = None


def safe_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class _ClassificationSplitWrapper(torch.utils.data.Dataset):
    """
    Wraps a StreamingDataset (e.g. one split from train_test_split) and applies
    the same (video, target) processing as OptimizedVideoClassificationDataset.
    """

    def __init__(
        self,
        streaming_dataset,
        frames_to_sample,
        temporal_stride,
        transform,
        task='classification',
        target_key='label',
    ):
        self.streaming_dataset = streaming_dataset
        self.frames_to_sample = frames_to_sample
        self.temporal_stride = temporal_stride
        self.transform = transform
        self.task = (task or 'classification').lower()
        self.target_key = target_key or 'label'

    def __len__(self):
        return len(self.streaming_dataset)

    def __getitem__(self, idx):
        data = self.streaming_dataset[idx]
        video = data['video']
        if self.target_key not in data:
            raise KeyError(
                f"Missing key {self.target_key!r} in sample; keys={list(data.keys())}"
            )
        raw_target = data[self.target_key]
        if self.task == 'regression':
            target = torch.as_tensor(raw_target, dtype=torch.float32).reshape(-1)
        else:
            label = raw_target
            if hasattr(label, 'item'):
                label = int(label.item())
            else:
                label = int(label)
            target = label
        num_frames = video.shape[0]
        max_start = max(0, num_frames - self.frames_to_sample)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0
        sampled_frames = video[start_frame : start_frame + self.frames_to_sample]
        video_tensor = sampled_frames[:: self.temporal_stride].float()
        max_val = video_tensor.max().item() if video_tensor.numel() > 0 else 0.0
        if max_val > 1.0:
            video_tensor = video_tensor / 255.0
        if video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)
        return video_tensor, target


class FinetuningDataModule(pl.LightningDataModule):
    """
    DataModule for finetuning with LitData (classification or regression).
    If val_optimized_dir and test_optimized_dir are set, uses three separate dirs.
    Otherwise uses train_test_split(train_optimized_dir, splits=split_ratio) for train/val/test.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.model_config = config['model']
        self.training_config = config['training']
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = None
        self.val_transform = None

    def setup(self, stage=None):
        normalize_mean = self.data_config.get('normalize_mean', [0.117, 0.114, 0.113])
        normalize_std = self.data_config.get('normalize_std', [0.208, 0.204, 0.203])
        input_size = self.model_config.get('input_size', 224)
        crop_scale = self.model_config.get('crop_scale', (0.75, 1.0))
        if isinstance(crop_scale, list):
            crop_scale = tuple(crop_scale)
        crop_aspect = self.model_config.get('crop_aspect_ratio', (0.8, 1.2))
        if isinstance(crop_aspect, list):
            crop_aspect = tuple(crop_aspect)
        self.train_transform = ClassificationTransform(
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            frame_size=input_size,
            crop_scale=crop_scale,
            crop_aspect_ratio=crop_aspect,
            is_train=True,
        )
        self.val_transform = ClassificationTransform(
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            frame_size=input_size,
            is_train=False,
        )
        cache_dir = self.data_config.get('cache_dir', './output/cache_finetune')
        safe_makedir(cache_dir)
        storage_options = None
        if custom_storage_options is not None and self.data_config.get('cloud_type') == 's3_public':
            storage_options = custom_storage_options
        max_cache = self.data_config.get('max_cache_size', '50GB')
        frames_to_sample = self.training_config.get('frames_to_sample', 16)
        temporal_stride = self.training_config.get('temporal_stride', 1)
        train_dir = self.data_config.get('train_optimized_dir')
        val_dir = self.data_config.get('val_optimized_dir')
        test_dir = self.data_config.get('test_optimized_dir')
        split_ratio = self.data_config.get('split_ratio', [0.8, 0.1, 0.1])
        if isinstance(split_ratio, (list, tuple)) and len(split_ratio) >= 3:
            split_ratio = [float(split_ratio[0]), float(split_ratio[1]), float(split_ratio[2])]
        else:
            split_ratio = [0.8, 0.1, 0.1]
        if train_dir is None:
            raise ValueError("data.train_optimized_dir is required.")
        task = self.data_config.get('task', 'classification')
        target_key = self.data_config.get('target_key', 'label')
        use_explicit_splits = val_dir is not None and test_dir is not None
        if stage == 'fit' or stage is None:
            if use_explicit_splits:
                self.train_dataset = OptimizedVideoClassificationDataset(
                    data_dir=train_dir,
                    frames_to_sample=frames_to_sample,
                    temporal_stride=temporal_stride,
                    transform=self.train_transform,
                    cache_dir=cache_dir,
                    max_cache_size=max_cache,
                    drop_last=True,
                    storage_options=storage_options,
                    task=task,
                    target_key=target_key,
                )
                self.val_dataset = OptimizedVideoClassificationDataset(
                    data_dir=val_dir,
                    frames_to_sample=frames_to_sample,
                    temporal_stride=temporal_stride,
                    transform=self.val_transform,
                    cache_dir=cache_dir,
                    max_cache_size=max_cache,
                    drop_last=False,
                    storage_options=storage_options,
                    task=task,
                    target_key=target_key,
                )
                print(f"Train dataset from {train_dir} (len={len(self.train_dataset)})")
                print(f"Val dataset from {val_dir} (len={len(self.val_dataset)})")
            else:
                base_ds = StreamingDataset(
                    input_dir=Dir(path=cache_dir, url=train_dir),
                    transform=None,
                    drop_last=False,
                    max_cache_size=max_cache,
                    storage_options=storage_options,
                )
                train_split, val_split, test_split = train_test_split(
                    streaming_dataset=base_ds, splits=split_ratio
                )
                self.train_dataset = _ClassificationSplitWrapper(
                    train_split,
                    frames_to_sample,
                    temporal_stride,
                    self.train_transform,
                    task=task,
                    target_key=target_key,
                )
                self.val_dataset = _ClassificationSplitWrapper(
                    val_split,
                    frames_to_sample,
                    temporal_stride,
                    self.val_transform,
                    task=task,
                    target_key=target_key,
                )
                self.test_dataset = _ClassificationSplitWrapper(
                    test_split,
                    frames_to_sample,
                    temporal_stride,
                    self.val_transform,
                    task=task,
                    target_key=target_key,
                )
                print(f"Split dataset from {train_dir} with ratio {split_ratio}")
                print(f"Train len={len(self.train_dataset)}, Val len={len(self.val_dataset)}, Test len={len(self.test_dataset)}")
        if stage == 'test' or stage is None:
            if self.test_dataset is None and use_explicit_splits and test_dir is not None:
                self.test_dataset = OptimizedVideoClassificationDataset(
                    data_dir=test_dir,
                    frames_to_sample=frames_to_sample,
                    temporal_stride=temporal_stride,
                    transform=self.val_transform,
                    cache_dir=cache_dir,
                    max_cache_size=max_cache,
                    drop_last=False,
                    storage_options=storage_options,
                    task=task,
                    target_key=target_key,
                )
                print(f"Test dataset from {test_dir} (len={len(self.test_dataset)})")

    def _make_loader(self, dataset, shuffle):
        bs = self.training_config['batch_size']
        kw = dict(
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.training_config.get('num_workers', 10),
            pin_memory=self.training_config.get('pin_memory', True),
        )
        if isinstance(dataset, StreamingDataset):
            return StreamingDataLoader(dataset, **kw, persistent_workers=False)
        return DataLoader(dataset, **kw)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first.")
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return self._make_loader(self.test_dataset, shuffle=False)
