"""Sanity checks for video-level (mean-softmax) aggregation vs engine_for_finetuning.compute_video."""

import numpy as np

from metrics.video_eval_metrics import (
    clip_level_prediction_rows,
    softmax_np,
    video_level_classification_metrics,
)


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
    assert "video_auroc_per_class" in m
    assert len(m["video_auroc_per_class"]) == 2


def test_clip_level_prediction_rows_two_clips():
    video_ids = ["v1", "v2"]
    logits = [
        np.array([1.0, 0.0, 2.0], dtype=np.float32),
        np.array([0.0, 3.0, 1.0], dtype=np.float32),
    ]
    targets = [2, 1]
    rows, skipped = clip_level_prediction_rows(video_ids, logits, targets)
    assert skipped == 0
    assert len(rows) == 2

    probs0 = softmax_np(logits[0])
    pred0 = int(np.argmax(probs0))
    assert rows[0]["video_id"] == "v1"
    assert rows[0]["predicted_label"] == pred0
    assert rows[0]["ground_truth_label"] == 2
    assert abs(rows[0]["predicted_probability"] - probs0[pred0]) < 1e-6

    probs1 = softmax_np(logits[1])
    pred1 = int(np.argmax(probs1))
    assert rows[1]["video_id"] == "v2"
    assert rows[1]["predicted_label"] == pred1
    assert rows[1]["ground_truth_label"] == 1
    assert abs(rows[1]["predicted_probability"] - probs1[pred1]) < 1e-6


def test_clip_level_prediction_rows_skips_missing_video_id():
    video_ids = ["v1", None]
    logits = [
        np.array([2.0, 0.0], dtype=np.float32),
        np.array([0.0, 2.0], dtype=np.float32),
    ]
    targets = [0, 1]
    rows, skipped = clip_level_prediction_rows(video_ids, logits, targets)
    assert skipped == 1
    assert len(rows) == 1
    assert rows[0]["video_id"] == "v1"


def test_video_auroc_per_class_multiclass():
    ids = ["a", "a", "b", "b"]
    logits = [
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.5, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 2.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.5, 0.0], dtype=np.float32),
    ]
    targets = [0, 0, 1, 1]
    m = video_level_classification_metrics(ids, logits, targets, num_classes=3)
    assert len(m["video_auroc_per_class"]) == 3
    assert all(isinstance(x, float) for x in m["video_auroc_per_class"])
