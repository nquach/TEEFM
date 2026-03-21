"""Sanity checks for video-level (mean-softmax) aggregation vs engine_for_finetuning.compute_video."""

import numpy as np

from metrics.video_eval_metrics import softmax_np, video_level_classification_metrics


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def compute_video_style(logits_list, label):
    """engine_for_finetuning.compute_video logic on logits (pre-softmax rows)."""
    feat = np.mean([softmax(l) for l in logits_list], axis=0)
    pred = int(np.argmax(feat))
    top1 = 1.0 if pred == int(label) else 0.0
    top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
    return pred, top1, top5


def test_mean_softmax_matches_compute_video_two_clips():
    label = 2
    logits_a = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    logits_b = np.array([0.0, 3.0, 1.0], dtype=np.float32)
    pred_cv, top1_cv, top5_cv = compute_video_style([logits_a, logits_b], label)

    video_ids = ["v1", "v1"]
    logits_rows = [logits_a, logits_b]
    targets = [label, label]
    m = video_level_classification_metrics(video_ids, logits_rows, targets, num_classes=3)

    feat = np.mean([softmax_np(logits_a), softmax_np(logits_b)], axis=0)
    pred_m = int(np.argmax(feat))
    assert pred_m == pred_cv
    assert abs(m["video_top1"] - top1_cv) < 1e-6
    top3 = 1.0 if label in np.argsort(-feat)[:3] else 0.0
    assert abs(m["video_top3"] - top3) < 1e-6
    _ = top5_cv  # parity with engine top5 exists; we assert top1/top3 here


def test_two_videos_independent():
    """Two logical videos with two clips each."""
    ids = ["a", "a", "b", "b"]
    logits = [
        np.array([2.0, 0.0], dtype=np.float32),
        np.array([1.5, 0.0], dtype=np.float32),
        np.array([0.0, 2.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    targets = [0, 0, 1, 1]
    m = video_level_classification_metrics(ids, logits, targets, num_classes=2)
    assert m["video_top1"] == 1.0
