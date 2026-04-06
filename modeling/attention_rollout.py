"""
Attention rollout for VideoMAE finetuning ViT (no CLS token; mean pooling).

Rollout follows the residual form (I + A) / 2 per layer, composed as
R = A_L @ ... @ A_1. Without a class token, token relevance is taken as
``R.mean(dim=0)`` (mean over query positions).
"""

from typing import List

import torch


def attention_rollout_mean_query(
    layer_attns: List[torch.Tensor],
    batch_idx: int = 0,
    use_residual: bool = True,
) -> torch.Tensor:
    """
    Args:
        layer_attns: Per-layer tensors (B, num_heads, N, N) after softmax, pre-dropout.
        batch_idx: Which batch element to use.
        use_residual: If True, use (I + A) / 2 per layer (head-averaged A).

    Returns:
        Tensor of shape (N,) — relevance per token.
    """
    if not layer_attns:
        raise ValueError("layer_attns must be non-empty")

    attn0 = layer_attns[0]
    N = attn0.shape[-1]
    device = attn0.device
    dtype = attn0.dtype

    R = torch.eye(N, device=device, dtype=dtype)
    for attn in layer_attns:
        A = attn[batch_idx].mean(dim=0)
        if use_residual:
            I = torch.eye(N, device=device, dtype=dtype)
            A = 0.5 * I + 0.5 * A
        R = A @ R

    return R.mean(dim=0)


def relevance_to_frame_heatmap(
    relevance: torch.Tensor,
    *,
    num_frames: int,
    tubelet_size: int,
    grid_h: int,
    grid_w: int,
    frame_index: int,
) -> torch.Tensor:
    """
    Map 1D token relevance to a (grid_h, grid_w) slice for one video frame.

    Token order matches PatchEmbed: time-major over spatial grid
    (t * (H'*W') + h * W' + w).
    """
    t_out = num_frames // tubelet_size
    n = t_out * grid_h * grid_w
    if relevance.numel() != n:
        raise ValueError(
            f"relevance length {relevance.numel()} != T'*H'*W' = {n} "
            f"(num_frames={num_frames}, tubelet_size={tubelet_size}, grid={grid_h}x{grid_w})"
        )

    tt = frame_index // tubelet_size
    if tt < 0 or tt >= t_out:
        raise ValueError(f"frame_index {frame_index} out of range for T'={t_out} tubelets")

    vol = relevance.view(t_out, grid_h, grid_w)
    return vol[tt].contiguous()
