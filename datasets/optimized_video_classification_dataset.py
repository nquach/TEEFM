"""
Optimized Video Classification Dataset using LitData StreamingDataset

StreamingDataset for labeled (video, label) data optimized with litdata.
Returns (video_tensor, label) with classification transform (normalize + optional augmentation).
"""

import random
import warnings
import torch
from litdata import StreamingDataset
from litdata.streaming.cache import Dir
from torchvision.transforms.v2 import Normalize, RandomResizedCrop, Compose

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')


class ClassificationTransform:
    """
    Video transform for classification: normalize and optional RandomResizedCrop (train only).
    No masking. Input/output: [C, T, H, W].
    """

    def __init__(
        self,
        normalize_mean=(0.117, 0.114, 0.113),
        normalize_std=(0.208, 0.204, 0.203),
        frame_size=224,
        crop_scale=(0.75, 1.0),
        crop_aspect_ratio=(0.8, 1.2),
        is_train=True,
    ):
        self.normalizer = Normalize(list(normalize_mean), list(normalize_std))
        self.is_train = is_train
        self.frame_size = frame_size
        if is_train:
            self.augment = RandomResizedCrop(
                size=frame_size,
                scale=crop_scale,
                ratio=crop_aspect_ratio,
            )
            self.transform = Compose([self.augment, self.normalizer])
        else:
            self.transform = self.normalizer

    def __call__(self, video):
        # video: [C, T, H, W]; torchvision expects (N, C, H, W) for batch of images
        video = video.permute(1, 0, 2, 3)   # (C, T, H, W) -> (T, C, H, W)
        video = self.transform(video)
        video = video.permute(1, 0, 2, 3)   # (T, C, H, W) -> (C, T, H, W)
        return video


class OptimizedVideoClassificationDataset(StreamingDataset):
    """
    LitData StreamingDataset for video classification: loads items with 'video' and 'label'.
    Returns (video_tensor [C, T, H, W], label int).
    """

    def __init__(
        self,
        data_dir,
        frames_to_sample=16,
        temporal_stride=1,
        subset_ratio=None,
        seed=None,
        transform=None,
        cache_dir=None,
        max_cache_size='50GB',
        drop_last=False,
        storage_options=None,
    ):
        try:
            if subset_ratio is not None:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir),
                    transform=None,
                    subsample=subset_ratio,
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
        self.frames_to_sample = frames_to_sample
        self.temporal_stride = temporal_stride
        self.custom_transform = transform
        print(f"Loaded optimized classification dataset from {data_dir}")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        video = data['video']
        label = data['label']
        if hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)

        num_frames = video.shape[0]
        max_start = max(0, num_frames - self.frames_to_sample)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0
        sampled_frames = video[start_frame : start_frame + self.frames_to_sample]
        video_tensor = sampled_frames[:: self.temporal_stride].float()
        max_val = video_tensor.max().item() if video_tensor.numel() > 0 else 0.0
        if max_val > 1.0:
            video_tensor = video_tensor / 255.0
        # Raw from LitData is typically T, H, W, C or T, C, H, W
        if video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        # Now [T, C, H, W] -> model expects [C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        if self.custom_transform is not None:
            video_tensor = self.custom_transform(video_tensor)
        return video_tensor, label
