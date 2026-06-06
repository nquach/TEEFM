"""Tests for finetune val/test collate (stable batched video_id lists)."""

import torch

from datasets.finetuning_data_module import finetune_video_id_collate_fn


def test_finetune_video_id_collate_fn_batch():
    batch = [
        (torch.zeros(3, 4, 8, 8), torch.tensor(0), "vid_a"),
        (torch.zeros(3, 4, 8, 8), torch.tensor(1), "vid_b"),
    ]
    videos, targets, video_ids = finetune_video_id_collate_fn(batch)
    assert videos.shape[0] == 2
    assert targets.tolist() == [0, 1]
    assert video_ids == ["vid_a", "vid_b"]


def test_finetune_video_id_collate_fn_batch_size_one():
    batch = [(torch.zeros(3, 4, 8, 8), torch.tensor(2), "solo")]
    videos, targets, video_ids = finetune_video_id_collate_fn(batch)
    assert videos.shape[0] == 1
    assert targets.tolist() == [2]
    assert video_ids == ["solo"]
