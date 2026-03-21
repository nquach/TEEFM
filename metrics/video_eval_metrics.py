"""
Video-level metric helpers (mean softmax per video_id).

Same spirit as engine_for_finetuning.compute_video. NumPy-only for fast tests; Lightning
converts tensors before calling.
"""

from typing import Dict, List
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def video_level_classification_metrics(
    video_ids: List[str],
    logits_rows: List[np.ndarray],
    targets_rows: List[int],
    num_classes: int,
) -> Dict[str, float]:
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
    }
    y_true_arr = np.array(y_true_list, dtype=np.int64)
    if num_classes == 2:
        try:
            out["video_auroc"] = float(
                roc_auc_score(y_true_arr, np.array(y_prob_rows, dtype=np.float64))
            )
        except ValueError:
            out["video_auroc"] = float("nan")
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
        except ValueError:
            out["video_auroc"] = float("nan")
        out["video_f1"] = float(
            f1_score(y_true_arr, np.array(y_pred_list), average="macro", zero_division=0)
        )
    return out
