"""Tests for val/test video_id collation."""

import torch

from datasets.finetuning_data_module import finetune_video_id_collate_fn


def test_collate_batch_size_one_keeps_full_video_id_string():
    video = torch.zeros(3, 16, 224, 224)
    target = 4
    video_id = "study_123_clipA.mp4"
    videos, targets, video_ids = finetune_video_id_collate_fn([(video, target, video_id)])
    assert videos.shape[0] == 1
    assert targets.shape == (1,)
    assert video_ids == ["study_123_clipA.mp4"]


def test_collate_batch_size_two_returns_list_of_strings():
    v = torch.zeros(3, 16, 224, 224)
    batch = [(v, 0, "vid_a"), (v, 1, "vid_b")]
    _, targets, video_ids = finetune_video_id_collate_fn(batch)
    assert targets.tolist() == [0, 1]
    assert video_ids == ["vid_a", "vid_b"]
