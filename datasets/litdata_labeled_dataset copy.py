"""
LitData labeled dataset for classification finetuning.

StreamingDataset-based loader for optimized (video, label) pairs. Supports
train/val/test from a single directory via ratio-based split or from separate dirs.
"""

import random
import torch
from torch.utils.data import Subset
from litdata import StreamingDataset, StreamingDataLoader
from litdata.streaming.cache import Dir


class LitDataLabeledDataset(StreamingDataset):
    """
    LitData StreamingDataset for (video, label) pairs.
    Each item is expected to have 'video' (tensor) and 'label' (tensor with single int).
    Returns (video CTHW float32, label long) with optional normalization.
    """

    def __init__(
        self,
        data_dir,
        cache_dir=None,
        max_cache_size="50GB",
        drop_last=False,
        storage_options=None,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        subsample=None,
        seed=None,
        num_frames=16,
    ):
        try:
            if subsample is not None:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir),
                    transform=None,
                    subsample=subsample,
                    drop_last=drop_last,
                    max_cache_size=max_cache_size,
                    storage_options=storage_options,
                )
            else:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir),
                    transform=None,
                    drop_last=drop_last,
                    max_cache_size=max_cache_size,
                    storage_options=storage_options,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize StreamingDataset from {data_dir}. Error: {e}"
            )
        # CTHW: (3, 1, 1, 1) for broadcasting with (C, T, H, W)
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1, 1)
        self._seed = seed
        self.num_frames = num_frames

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        video = data["video"]
        label = data["label"]

        # Ensure video is float and in CTHW
        if video.dim() == 4 and video.shape[-1] == 3:
            # THWC -> CTHW
            video = video.permute(3, 0, 1, 2)
        video = video.float()
        if video.max() > 1.0:
            video = video / 255.0
        video = (video - self.normalize_mean) / self.normalize_std

        # Fixed-length temporal sampling: (C, T, H, W) -> (C, num_frames, H, W)
        T = video.shape[1]
        if T >= self.num_frames:
            start = random.randint(0, T - self.num_frames)
            video = video[:, start : start + self.num_frames, :, :]
        else:
            # Pad by repeating the last frame
            if T == 0:
                # Edge case: no frames; pad with zeros
                video = torch.zeros(
                    video.shape[0], self.num_frames, video.shape[2], video.shape[3],
                    dtype=video.dtype, device=video.device,
                )
            else:
                repeat_last = video[:, -1:, :, :].expand(-1, self.num_frames - T, -1, -1)
                video = torch.cat([video, repeat_last], dim=1)

        # Label: 1-element tensor -> int
        if isinstance(label, torch.Tensor):
            label = label.flatten()[0].long().item()
        else:
            label = int(label)

        return video.clone(), label


def build_litdata_finetune_datasets(data_config, seed=0, num_frames=None):
    """
    Build train, val, and test datasets from data_config.

    - If only train_optimized_dir is set: use train_val_test_split ratios to split
      the single dataset into train/val/test (deterministic by seed).
    - If val_optimized_dir and test_optimized_dir are also set: use three separate
      directories; train_val_test_split is ignored.

    Returns:
        train_ds, val_ds, test_ds, num_classes
    """
    if num_frames is None:
        num_frames = data_config.get("num_frames", 16)
    train_dir = data_config.get("train_optimized_dir")
    val_dir = data_config.get("val_optimized_dir")
    test_dir = data_config.get("test_optimized_dir")
    if not train_dir:
        raise ValueError("data.train_optimized_dir is required.")

    cache_dir = data_config.get("cache_dir", "./output/cache")
    max_cache_size = data_config.get("max_cache_size", "50GB")
    storage_options = None
    if data_config.get("cloud_type") == "s3_public":
        try:
            import botocore
            storage_options = {
                "config": botocore.config.Config(
                    retries={"max_attempts": 1000, "mode": "adaptive"},
                    signature_version=botocore.UNSIGNED,
                )
            }
        except ImportError:
            pass

    normalize_mean = tuple(data_config.get("normalize_mean", [0.485, 0.456, 0.406]))
    normalize_std = tuple(data_config.get("normalize_std", [0.229, 0.224, 0.225]))
    seed = data_config.get("seed", seed)

    def make_ds(data_dir, subsample=None):
        return LitDataLabeledDataset(
            data_dir=data_dir,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
            drop_last=False,
            storage_options=storage_options,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            subsample=subsample,
            seed=seed,
            num_frames=num_frames,
        )

    if val_dir and test_dir:
        train_ds = make_ds(train_dir)
        val_ds = make_ds(val_dir)
        test_ds = make_ds(test_dir)
        return train_ds, val_ds, test_ds, data_config["num_classes"]

    # Single dir: split by ratios
    ratios = data_config.get("train_val_test_split", [0.8, 0.1, 0.1])
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            "train_val_test_split must be [train_ratio, val_ratio, test_ratio] summing to 1.0"
        )
    full_ds = make_ds(train_dir)
    n = len(full_ds)
    if n == 0:
        raise ValueError("Dataset has length 0; cannot split.")
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    # Deterministic split by index ranges (no shuffle)
    train_ds = Subset(full_ds, range(0, n_train))
    val_ds = Subset(full_ds, range(n_train, n_train + n_val))
    test_ds = Subset(full_ds, range(n_train + n_val, n))
    return train_ds, val_ds, test_ds, data_config["num_classes"]
