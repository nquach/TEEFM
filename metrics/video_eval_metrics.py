"""
Video-level metric helpers (mean softmax per video_id).

Same spirit as engine_for_finetuning.compute_video. NumPy-only for fast tests; Lightning
converts tensors before calling.
"""

from typing import Any, Dict, List
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _per_class_ovr_auroc(y_true_arr, prob_mat, num_classes: int):
    """Compute per-class OvR AUROC; return (scores, error_message)."""
    nan_per_class = [float("nan")] * num_classes
    try:
        scores = roc_auc_score(
            y_true_arr, prob_mat, multi_class="ovr", average=None
        )
    except ValueError as exc:
        return nan_per_class, str(exc)
    scores_arr = np.atleast_1d(np.asarray(scores, dtype=np.float64))
    if scores_arr.size != num_classes:
        return nan_per_class, (
            f"Expected {num_classes} per-class AUROC scores, got {scores_arr.size} "
            f"(unique classes in y_true={len(set(y_true_arr.tolist()))})."
        )
    return [float(x) for x in scores_arr], None


def video_level_classification_metrics(
    video_ids: List[str],
    logits_rows: List[np.ndarray],
    targets_rows: List[int],
    num_classes: int,
) -> Dict[str, Any]:
    """Mean softmax per video (same spirit as engine_for_finetuning.compute_video)."""
    by_vid = defaultdict(list)
    y_true_by = {}
    for vid, logit, tgt in zip(video_ids, logits_rows, targets_rows):
        if vid is None:
            continue
        by_vid[vid].append(np.asarray(logit, dtype=np.float32))
        y_true_by[vid] = int(tgt)

    if not by_vid:
        return {}

    y_true_arr_preview = np.array(list(y_true_by.values()), dtype=np.int64)
    debug_info = {
        "video_num_clips": len(video_ids),
        "video_num_videos": len(by_vid),
        "video_num_classes_in_y_true": int(len(set(y_true_arr_preview.tolist()))),
    }

    top1_hits = []
    top3_hits = []
    y_true_list = []
    y_prob_rows = []
    y_pred_list = []

    for vid, lst in by_vid.items():
        probs_stack = np.stack([softmax_np(l) for l in lst], axis=0)
        feat = np.mean(probs_stack, axis=0)
        label = y_true_by[vid]
        pred = int(np.argmax(feat))
        y_pred_list.append(pred)
        y_true_list.append(label)
        top1_hits.append(1.0 if pred == label else 0.0)
        if num_classes >= 3:
            top3 = set(np.argsort(-feat)[:3].tolist())
            top3_hits.append(1.0 if label in top3 else 0.0)
        else:
            top3_hits.append(top1_hits[-1])
        if num_classes == 2:
            y_prob_rows.append(feat[1])
        else:
            y_prob_rows.append(feat)

    out = {
        "video_top1": float(np.mean(top1_hits)),
        "video_top3": float(np.mean(top3_hits)),
        **debug_info,
    }
    y_true_arr = np.array(y_true_list, dtype=np.int64)
    nan_per_class = [float("nan")] * num_classes
    if num_classes == 2:
        pos_probs = np.array(y_prob_rows, dtype=np.float64)
        prob_mat = np.stack([1.0 - pos_probs, pos_probs], axis=1)
        try:
            out["video_auroc"] = float(
                roc_auc_score(y_true_arr, pos_probs)
            )
        except ValueError as exc:
            out["video_auroc"] = float("nan")
            out["video_auroc_error"] = str(exc)
        try:
            out["video_auroc_per_class"], err = _per_class_ovr_auroc(
                y_true_arr, prob_mat, num_classes
            )
            if err:
                out["video_auroc_per_class_error"] = err
        except Exception as exc:
            out["video_auroc_per_class"] = list(nan_per_class)
            out["video_auroc_per_class_error"] = str(exc)
        out["video_f1"] = float(
            f1_score(y_true_arr, np.array(y_pred_list), average="binary", zero_division=0)
        )
    else:
        prob_mat = np.stack(y_prob_rows, axis=0)
        try:
            out["video_auroc"] = float(
                roc_auc_score(
                    y_true_arr,
                    prob_mat,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError as exc:
            out["video_auroc"] = float("nan")
            out["video_auroc_error"] = str(exc)
        per_class, err = _per_class_ovr_auroc(y_true_arr, prob_mat, num_classes)
        out["video_auroc_per_class"] = per_class
        if err:
            out["video_auroc_per_class_error"] = err
        out["video_f1"] = float(
            f1_score(y_true_arr, np.array(y_pred_list), average="macro", zero_division=0)
        )
    return out
