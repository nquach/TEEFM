"""
Attention rollout visualizations for finetuned VideoMAE ViT (classification or regression).

Loads the same YAML as training plus a Lightning finetuning checkpoint, samples random
clips from val or test, picks random video frames per clip, and saves one figure with
three columns: RGB frame, attention heatmap, smoothed heatmap overlay.

This backbone has no CLS token and uses mean pooling. Rollout uses the residual form
(I+A)/2 per layer and relevance R.mean(dim=0) over queries (not ViT-CLS rollout).

MCM (motion-centric masking) must be disabled in config; token layout is undefined when
mcm=True.

Usage:
  python visualize_attention_rollout.py --config configs/finetune_vitb.yaml \\
      --checkpoint path/to.ckpt --split val --num_samples 4 --num_frames 3 \\
      --output rollout.png
"""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")

import matplotlib.pyplot as plt
from matplotlib import cm

from datasets.finetuning_data_module import FinetuningDataModule
from modeling.attention_rollout import attention_rollout_mean_query, relevance_to_frame_heatmap
from modeling.model_factory import create_videomae_finetune_model


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def strip_lightning_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def load_finetune_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")
    state = strip_lightning_prefix(state, "model.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"load_state_dict: {len(missing)} missing keys (e.g. {missing[:3]}...)")
    if unexpected:
        print(f"load_state_dict: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]}...)")


def unpack_sample(item):
    if len(item) == 3:
        video, target, _vid = item
    else:
        video, target = item
    return video, target


def tensor_frame_to_rgb(
    frame_chw: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """frame_chw: (C, H, W) normalized; returns (H, W, 3) float in [0, 1]."""
    x = frame_chw.detach().float().cpu()
    rgb = x * std.view(3, 1, 1) + mean.view(3, 1, 1)
    rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
    return rgb


def normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    lo, hi = m.min(), m.max()
    if hi - lo < 1e-8:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def main():
    parser = argparse.ArgumentParser(description="VideoMAE finetune attention rollout figures")
    parser.add_argument("--config", type=str, required=True, help="Training YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning .ckpt from finetuning")
    parser.add_argument("--split", type=str, choices=("val", "test"), default="val")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of random clips")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Random video-frame indices per clip (without replacement if <= T)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="attention_rollout.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for overlay heatmap")
    parser.add_argument(
        "--no_residual_rollout",
        action="store_true",
        help="Use raw head-averaged A instead of (I+A)/2 per layer",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    if model_cfg.get("mcm", False):
        raise RuntimeError(
            "model.mcm is True: attention rollout token layout is not supported. "
            "Set mcm: false in config for this script."
        )

    task = (data_cfg.get("task") or "classification").lower()
    if task == "regression":
        output_dim = int(data_cfg.get("output_dim", 1))
        num_classes = int(data_cfg.get("num_classes", 1))
        task_type = "regression"
    else:
        num_classes = int(data_cfg.get("num_classes", 101))
        output_dim = num_classes
        task_type = "classification"

    model = create_videomae_finetune_model(
        backbone=model_cfg.get("backbone", "vit-b"),
        pretrained_path=model_cfg.get("pretrained_path"),
        num_classes=num_classes,
        num_frames=model_cfg.get("num_frames", 16),
        tubelet_size=model_cfg.get("tubelet_size", 2),
        input_size=model_cfg.get("input_size", 224),
        fc_drop_rate=model_cfg.get("fc_drop_rate", 0.0),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path", 0.1),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        use_mean_pooling=model_cfg.get("use_mean_pooling", True),
        init_scale=model_cfg.get("init_scale", 0.001),
        mcm=False,
        mcm_ratio=model_cfg.get("mcm_ratio", 0.4),
        model_key=model_cfg.get("model_key", "state_dict|model|module"),
        model_prefix=model_cfg.get("model_prefix", ""),
        task_type=task_type,
        output_dim=output_dim,
    )

    load_finetune_weights(model, args.checkpoint)
    device = torch.device(args.device)
    model.eval()
    model.to(device)

    dm = FinetuningDataModule(config)
    dm.setup("fit")
    ds = dm.val_dataset if args.split == "val" else dm.test_dataset
    if ds is None:
        raise RuntimeError(f"No {args.split} dataset after setup('fit').")

    n_ds = len(ds)
    if n_ds == 0:
        raise RuntimeError(f"{args.split} dataset is empty.")

    rng = np.random.default_rng(args.seed)
    n_draw = min(args.num_samples, n_ds)
    sample_indices = rng.choice(n_ds, size=n_draw, replace=False)

    num_frames = int(model_cfg.get("num_frames", 16))
    tubelet_size = int(model_cfg.get("tubelet_size", 2))
    input_size = int(model_cfg.get("input_size", 224))
    pe = model.patch_embed
    grid_h = pe.img_size[0] // pe.patch_size[0]
    grid_w = pe.img_size[1] // pe.patch_size[1]

    mean = torch.tensor(data_cfg.get("normalize_mean", [0.117, 0.114, 0.113]), dtype=torch.float32)
    std = torch.tensor(data_cfg.get("normalize_std", [0.208, 0.204, 0.203]), dtype=torch.float32)

    n_rows = n_draw * args.num_frames
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows), squeeze=False)
    cmap = cm.get_cmap("viridis")
    use_residual = not args.no_residual_rollout

    row = 0
    for si, idx in enumerate(sample_indices):
        item = ds[int(idx)]
        video, _target = unpack_sample(item)
        if video.dim() != 4:
            raise ValueError(f"Expected video (C,T,H,W), got shape {tuple(video.shape)}")
        _, t_vid, h_vid, w_vid = video.shape
        if t_vid != num_frames:
            warnings.warn(
                f"Video T={t_vid} != model num_frames={num_frames}; rollout assumes they match."
            )

        if args.num_frames <= num_frames:
            frame_ids = rng.choice(num_frames, size=args.num_frames, replace=False)
        else:
            frame_ids = rng.choice(num_frames, size=args.num_frames, replace=True)

        vid = video.unsqueeze(0).to(device)
        attn_layers = []
        with torch.no_grad():
            _ = model.forward_features(vid, attn_weights_out=attn_layers)

        rel = attention_rollout_mean_query(
            attn_layers, batch_idx=0, use_residual=use_residual
        )

        for fi in frame_ids:
            hm_small = relevance_to_frame_heatmap(
                rel.cpu(),
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                grid_h=grid_h,
                grid_w=grid_w,
                frame_index=int(fi),
            )
            hm_up = (
                F.interpolate(
                    hm_small.view(1, 1, grid_h, grid_w),
                    size=(h_vid, w_vid),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            hm_raw = normalize_map(hm_up)
            hm_smooth = gaussian_filter(hm_raw.astype(np.float64), sigma=args.sigma)
            hm_smooth = normalize_map(hm_smooth)

            rgb = tensor_frame_to_rgb(video[:, int(fi)], mean, std)
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"sample {si} idx {idx} frame {int(fi)}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(hm_raw, cmap="viridis", vmin=0, vmax=1)
            axes[row, 1].set_title("attention")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(rgb)
            axes[row, 2].imshow(cmap(hm_smooth), alpha=0.5, interpolation="bilinear")
            axes[row, 2].set_title(f"overlay (sigma={args.sigma})")
            axes[row, 2].axis("off")

            row += 1

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
