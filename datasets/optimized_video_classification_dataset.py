"""
Optimized Video Classification Dataset using LitData StreamingDataset

StreamingDataset for labeled (video, label) data optimized with litdata.
Returns (video_tensor, label) with classification transform (normalize + optional augmentation).

LitData sample schema (classification / regression):
  - Required: "video" (tensor), target under target_key (default "label").
  - For eval_protocol: multi_clip on val/test: stable string field data.video_id_key
    (default "video_id") is required so clips can be aggregated per logical video.

Train returns a 2-tuple (video, target). Val/test return (video, target, video_id) where
video_id is None if the key is missing (single_clip); multi_clip raises if missing.
"""

import random
import warnings
import torch
import torch.nn.functional as F
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


def _resize_short_side_thwc(thwc: torch.Tensor, short_side: int) -> torch.Tensor:
    """Resize so min(H, W) == short_side (bilinear). Input/output THWC float."""
    t, h, w, c = thwc.shape
    if min(h, w) == short_side:
        return thwc
    if h <= w:
        new_h = short_side
        new_w = max(1, int(round(w * short_side / float(h))))
    else:
        new_w = short_side
        new_h = max(1, int(round(h * short_side / float(w))))
    x = thwc.permute(0, 3, 1, 2).contiguous().float()  # T,C,H,W
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _resize_spatial_cthw_to_square(cthw: torch.Tensor, size: int) -> torch.Tensor:
    """cthw: C,T,H,W -> spatial resized to size x size."""
    c, t, h, w = cthw.shape
    x = cthw.permute(1, 0, 2, 3).contiguous().float()  # T,C,H,W
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3).contiguous()


def _ucf_style_strip_crop_thwc(
    buffer_thwc: torch.Tensor,
    chunk_nb: int,
    split_nb: int,
    temporal_slice_len: int,
    test_num_segment: int,
    test_num_crop: int,
    short_side_size: int,
) -> torch.Tensor:
    """
    UCF-style deterministic temporal index x spatial strip (see ucf.py test mode).
    buffer_thwc: T, H, W, C (already resized short-side = short_side_size).
    Returns a contiguous temporal slice of length <= temporal_slice_len (padded if needed).
    """
    t, h, w, c = buffer_thwc.shape
    clip_len = min(temporal_slice_len, t)
    if test_num_segment <= 1:
        temporal_start = max(0, (t - clip_len) // 2)
    else:
        temporal_step = max(1.0 * (t - clip_len) / (test_num_segment - 1), 0.0)
        temporal_start = int(chunk_nb * temporal_step)
    temporal_start = max(0, min(temporal_start, max(0, t - clip_len)))

    if test_num_crop <= 1:
        long_dim = max(h, w)
        spatial_start = max(0, (long_dim - short_side_size) // 2)
    else:
        spatial_step = 1.0 * (max(h, w) - short_side_size) / (test_num_crop - 1)
        spatial_start = int(split_nb * spatial_step)
        spatial_start = max(0, spatial_start)

    if h >= w:
        out = buffer_thwc[
            temporal_start : temporal_start + clip_len,
            spatial_start : spatial_start + short_side_size,
            :,
            :,
        ]
    else:
        out = buffer_thwc[
            temporal_start : temporal_start + clip_len,
            :,
            spatial_start : spatial_start + short_side_size,
            :,
        ]
    if out.shape[0] < temporal_slice_len:
        pad_n = temporal_slice_len - out.shape[0]
        last = out[-1:].expand(pad_n, *out.shape[1:])
        out = torch.cat([out, last], dim=0)
    elif out.shape[0] > temporal_slice_len:
        out = out[:temporal_slice_len]
    return out.contiguous()


def _video_tensor_to_thwc_float(video: torch.Tensor) -> torch.Tensor:
    """Normalize LitData video to float THWC in [0, 1]. Accepts THWC or TCHW."""
    video_tensor = video.float()
    max_val = video_tensor.max().item() if video_tensor.numel() > 0 else 0.0
    if max_val > 1.0:
        video_tensor = video_tensor / 255.0
    if video_tensor.dim() != 4:
        raise ValueError(f"video must be 4D, got shape {tuple(video_tensor.shape)}")
    if video_tensor.shape[-1] == 3:
        # T, H, W, C
        pass
    elif video_tensor.shape[1] == 3:
        # T, C, H, W -> THWC
        video_tensor = video_tensor.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Cannot infer video layout, shape {tuple(video_tensor.shape)}")
    return video_tensor.contiguous()


def _apply_stride_and_to_cthw(
    thwc: torch.Tensor,
    frames_to_sample: int,
    temporal_stride: int,
) -> torch.Tensor:
    """Take first frames_to_sample frames (already temporal window), stride subsample, THWC->CTHW."""
    sampled = thwc[:frames_to_sample]
    sampled = sampled[::temporal_stride]
    return sampled.permute(3, 0, 1, 2).contiguous()


def _random_train_clip(
    video_thwc: torch.Tensor,
    frames_to_sample: int,
    temporal_stride: int,
) -> torch.Tensor:
    num_frames = video_thwc.shape[0]
    max_start = max(0, num_frames - frames_to_sample)
    start_frame = random.randint(0, max_start) if max_start > 0 else 0
    sampled_frames = video_thwc[start_frame : start_frame + frames_to_sample]
    return _apply_stride_and_to_cthw(sampled_frames, frames_to_sample, temporal_stride)


def build_finetune_video_tensor(
    data: dict,
    *,
    frames_to_sample: int,
    temporal_stride: int,
    transform,
    task: str,
    target_key: str,
    video_id_key: str,
    return_video_id: bool,
    require_video_id: bool,
    eval_multiclip: bool,
    chunk_nb: int = 0,
    split_nb: int = 0,
    test_num_segment: int = 1,
    test_num_crop: int = 1,
    short_side_size: int = 256,
    input_size: int = 224,
    eval_center_crop_only: bool = False,
):
    """
    Core path: LitData dict -> (video CTHW, target) or + video_id.

    eval_multiclip: deterministic UCF-style grid on resized video (val/test).
    eval_center_crop_only: single center temporal/spatial clip (chunk/split ignored).
    """
    if target_key not in data:
        raise KeyError(
            f"Missing key {target_key!r} in sample; keys={list(data.keys())}"
        )
    raw_target = data[target_key]
    task = (task or "classification").lower()
    if task == "regression":
        target = torch.as_tensor(raw_target, dtype=torch.float32).reshape(-1)
    else:
        label = raw_target
        if hasattr(label, "item"):
            label = int(label.item())
        else:
            label = int(label)
        target = label

    video = data["video"]
    video_thwc = _video_tensor_to_thwc_float(video)

    if eval_multiclip:
        if require_video_id and video_id_key not in data:
            raise KeyError(
                f"eval_protocol multi_clip requires LitData field {video_id_key!r} "
                f"per sample; keys={list(data.keys())}"
            )
        resized = _resize_short_side_thwc(video_thwc, short_side_size)
        if eval_center_crop_only:
            seg, crop = 1, 1
            ck, cp = 0, 0
        else:
            seg, crop = test_num_segment, test_num_crop
            ck, cp = chunk_nb, split_nb
        stripped = _ucf_style_strip_crop_thwc(
            resized,
            ck,
            cp,
            frames_to_sample,
            seg,
            crop,
            short_side_size,
        )
        video_cthw = _apply_stride_and_to_cthw(stripped, frames_to_sample, temporal_stride)
        video_cthw = _resize_spatial_cthw_to_square(video_cthw, input_size)
    else:
        video_cthw = _random_train_clip(video_thwc, frames_to_sample, temporal_stride)

    if transform is not None:
        video_cthw = transform(video_cthw)

    if not return_video_id:
        return video_cthw, target

    if video_id_key in data:
        vid = data[video_id_key]
        video_id = vid if isinstance(vid, str) else str(vid)
    else:
        if require_video_id:
            raise KeyError(
                f"Missing {video_id_key!r} for multi_clip evaluation; keys={list(data.keys())}"
            )
        video_id = None
    return video_cthw, target, video_id


class OptimizedVideoClassificationDataset(StreamingDataset):
    """
    LitData StreamingDataset for video classification or regression.
    Loads items with 'video' and target under target_key (default 'label').
    Classification: returns (video [C,T,H,W], int label) when return_video_id is False.
    Regression: returns (video [C,T,H,W], float tensor shape (1,)) when return_video_id is False.

    Val/test can set return_video_id=True for a 3-tuple with video_id (or None).
    eval_multiclip expands __len__ by test_num_segment * test_num_crop (unless eval_center_crop_only).
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
        task='classification',
        target_key='label',
        return_video_id=False,
        video_id_key='video_id',
        eval_multiclip=False,
        test_num_segment=1,
        test_num_crop=1,
        short_side_size=256,
        input_size=224,
        eval_center_crop_only=False,
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
        self.task = (task or 'classification').lower()
        self.target_key = target_key or 'label'
        if self.task not in ('classification', 'regression'):
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        self.return_video_id = bool(return_video_id)
        self.video_id_key = video_id_key or 'video_id'
        self.eval_multiclip = bool(eval_multiclip)
        self.test_num_segment = int(test_num_segment)
        self.test_num_crop = int(test_num_crop)
        self.short_side_size = int(short_side_size)
        self.input_size = int(input_size)
        self.eval_center_crop_only = bool(eval_center_crop_only)
        print(f"Loaded optimized dataset ({self.task}) from {data_dir}")

    def __len__(self):
        n = super().__len__()
        if self.eval_multiclip and not self.eval_center_crop_only:
            return n * self.test_num_segment * self.test_num_crop
        return n

    def __getitem__(self, idx):
        if self.eval_multiclip and not self.eval_center_crop_only:
            per = self.test_num_segment * self.test_num_crop
            row = idx // per
            rem = idx % per
            chunk_nb = rem // self.test_num_crop
            split_nb = rem % self.test_num_crop
        else:
            row = idx
            chunk_nb = 0
            split_nb = 0

        data = super().__getitem__(row)
        return build_finetune_video_tensor(
            data,
            frames_to_sample=self.frames_to_sample,
            temporal_stride=self.temporal_stride,
            transform=self.custom_transform,
            task=self.task,
            target_key=self.target_key,
            video_id_key=self.video_id_key,
            return_video_id=self.return_video_id,
            require_video_id=self.eval_multiclip,
            eval_multiclip=self.eval_multiclip,
            chunk_nb=chunk_nb,
            split_nb=split_nb,
            test_num_segment=self.test_num_segment,
            test_num_crop=self.test_num_crop,
            short_side_size=self.short_side_size,
            input_size=self.input_size,
            eval_center_crop_only=self.eval_center_crop_only,
        )
