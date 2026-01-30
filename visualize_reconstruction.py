"""
Video Reconstruction Visualization Script

This script loads a pretrained VideoMAE model, samples random videos from the
optimized dataset, and generates visualizations showing reconstruction quality.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange

from modeling.model_factory import create_videomae_model
from datasets.optimized_video_dataset import OptimizedVideoDataset
from transforms.custom_transforms import DataAugmentationForVideoMAE
from videomae_utils.reconstruction_utils import (
    patches_to_video,
    create_masked_video,
    denormalize_video,
    create_side_by_side_grid,
    save_visualization_grid
)
import botocore


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model(checkpoint_path, config, device):
    """Load pretrained model from checkpoint."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create model
    model = create_videomae_model(
        backbone=config['model']['backbone'],
        pretrained_path=checkpoint_path,
        decoder_depth=config['model'].get('decoder_depth', 4),
        drop_path=config['model'].get('drop_path', 0.0),
        mask_type=config['model'].get('mask_type', 'motion-centric'),
        mask_ratio=config['model'].get('mask_ratio', 0.9),
        motion_centric_masking_ratio=config['model'].get('motion_centric_masking_ratio', 0.7),
        use_checkpoint=config.get('features', {}).get('use_checkpoint', False)
    )
    
    model = model.to(device)
    model.eval()
    
    return model


def load_dataset(config):
    """Load optimized dataset."""
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    
    # Get dataset directory
    train_data_dir = data_config.get('train_optimized_dir')
    if train_data_dir is None:
        raise ValueError("train_optimized_dir is not specified in config")
    
    # Setup storage options for S3 if needed
    custom_storage_options = None
    if data_config.get('cloud_type', 's3_public') == 's3_public':
        custom_storage_options = {
            "config": botocore.config.Config(
                retries={"max_attempts": 1000, "mode": "adaptive"},
                signature_version=botocore.UNSIGNED,
            )
        }
    
    # Create transform
    normalize_mean = data_config.get('normalize_mean', [0.117, 0.114, 0.113])
    normalize_std = data_config.get('normalize_std', [0.208, 0.204, 0.203])
    
    # Calculate window size
    num_frames = training_config.get('num_frames', 16)
    input_size = model_config.get('input_size', 224)
    patch_size = model_config.get('patch_size', 16)
    
    window_size = (
        num_frames // 2,  # Temporal dimension (after tubelet_size=2)
        input_size // patch_size,  # Height patches
        input_size // patch_size   # Width patches
    )
    
    transform = DataAugmentationForVideoMAE(
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        window_size=window_size,
        mask_type=config['model'].get('mask_type', 'motion-centric'),
        mask_ratio=config['model'].get('mask_ratio', 0.9),
        motion_centric_masking_ratio=config['model'].get('motion_centric_masking_ratio', 0.7),
        frame_size=input_size
    )
    
    # Create dataset
    dataset = OptimizedVideoDataset(
        data_dir=train_data_dir,
        frames_to_sample=num_frames,
        temporal_stride=training_config.get('temporal_stride', 1),
        subset_ratio=None,  # Use full dataset
        seed=None,
        transform=transform,
        cache_dir=data_config.get('cache_dir'),
        max_cache_size=data_config.get('max_cache_size', '50GB'),
        drop_last=False,
        storage_options=custom_storage_options
    )
    
    print(f"Loaded dataset with {len(dataset)} videos")
    return dataset


def compute_reconstruction_loss(model, video, mask, config, device):
    """
    Compute the same MSE loss as training over one video (sanity check for reconstruction).
    Mirrors lightning_module.py training_step: build targets from unnormalized patches,
    forward pass, then MSE(outputs, labels).
    """
    video = video.to(device).unsqueeze(0)  # [1, C, T, H, W]
    if mask is not None and not isinstance(mask, (tuple, int)) and isinstance(mask, torch.Tensor):
        mask = mask.to(device)
        if mask.dim() > 1:
            mask = mask.flatten(0).unsqueeze(0)  # [1, num_patches] for non-MCM

    data_config = config.get('data', {})
    model_config = config.get('model', {})
    mean = torch.tensor(
        data_config.get('normalize_mean', [0.117, 0.114, 0.113]),
        device=device, dtype=video.dtype
    )
    std = torch.tensor(
        data_config.get('normalize_std', [0.208, 0.204, 0.203]),
        device=device, dtype=video.dtype
    )
    patch_size = model_config.get('patch_size', 16)
    tubelet_size = model_config.get('tubelet_size', 2)
    normalize_target = model_config.get('normalize_target', True)
    mask_type = model_config.get('mask_type', 'motion-centric')

    # Unnormalize to get pixel values (same as training)
    unnorm_videos = video * std[None, :, None, None, None] + mean[None, :, None, None, None]

    # Build videos_patch exactly as in lightning_module.py
    if normalize_target:
        videos_squeeze = rearrange(
            unnorm_videos,
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
            p0=tubelet_size,
            p1=patch_size,
            p2=patch_size
        )
        videos_norm = (
            videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
    else:
        videos_patch = rearrange(
            unnorm_videos,
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
            p0=tubelet_size,
            p1=patch_size,
            p2=patch_size
        )

    B, _, C = videos_patch.shape
    if mask_type != 'motion-centric' and mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.to(torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.flatten(1)

    with torch.no_grad():
        outputs, masks = model(video, mask)

    if mask_type == 'motion-centric':
        _, mc_target_mask = masks
        labels = videos_patch[~mc_target_mask].reshape(B, -1, C)
    else:
        if mask is None or not isinstance(mask, torch.Tensor):
            return float('nan')
        labels = videos_patch[mask].reshape(B, -1, C)

    loss = F.mse_loss(outputs, labels)
    return loss.item()


def reconstruct_video(model, video, mask, config, device):
    """Run inference to reconstruct video."""
    # Move to device
    video = video.to(device)
    if mask is not None and not isinstance(mask, (int, tuple)):
        mask = mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs, masks = model(video.unsqueeze(0), mask)
    
    return outputs, masks, video.unsqueeze(0)


def process_video_for_visualization(
    model,
    video,
    mask,
    config,
    device,
    patch_size,
    tubelet_size,
    normalize_target
):
    """Process video through model and prepare for visualization."""
    # Run inference
    reconstructed_patches, masks, normalized_video = reconstruct_video(
        model, video, mask, config, device
    )
    
    # Get original video shape
    B, C, T, H, W = normalized_video.shape
    
    # Get normalization parameters
    data_config = config['data']
    mean = torch.tensor(data_config.get('normalize_mean', [0.117, 0.114, 0.113]), device=device)
    std = torch.tensor(data_config.get('normalize_std', [0.208, 0.204, 0.203]), device=device)
    
    # Get original patches for reconstruction
    # Unnormalize to get original video
    original_video = denormalize_video(normalized_video, mean, std)
    
    # Extract original patches
    original_patches = rearrange(
        original_video,
        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size
    )
    
    # Reconstruct full video from patches
    reconstructed_video = patches_to_video(
        reconstructed_patches,
        masks,
        original_video,
        patch_size,
        tubelet_size,
        normalize_target=normalize_target,
        original_patches=original_patches
    )
    
    # Create masked video visualization
    masked_video = create_masked_video(
        original_video,
        masks,
        patch_size,
        tubelet_size
    )
    
    # Denormalize input-normalized videos for visualization
    original_vis = denormalize_video(original_video, mean, std)
    masked_vis = denormalize_video(masked_video, mean, std)
    # Reconstructed video is already in [0,1] (patches_to_video scales when normalize_target=True)
    reconstructed_vis = reconstructed_video.clamp(0.0, 1.0)
    
    # Option A: per-column min-max stretch so columns 1 and 2 use full [0,1] range (match column 3)
    def stretch_to_01(v):
        v_min, v_max = v.min().item(), v.max().item()
        return ((v - v_min) / (v_max - v_min + 1e-6)).clamp(0.0, 1.0)
    original_vis = stretch_to_01(original_vis)
    masked_vis = stretch_to_01(masked_vis)
    
    return original_vis, masked_vis, reconstructed_vis


def main():
    parser = argparse.ArgumentParser(
        description='Visualize VideoMAE reconstructions from optimized dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (must include dataset config)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to pretrained checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of random videos to sample and visualize'
    )
    parser.add_argument(
        '--frames_to_show',
        type=int,
        default=8,
        help='Number of frames per video to visualize'
    )
    parser.add_argument(
        '--grid_cols',
        type=int,
        default=3,
        help='Number of columns in output grid'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible video sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--compute-loss',
        action='store_true',
        help='Compute training-style MSE over each visualized sample as a sanity check'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Get model parameters
    model_config = config['model']
    patch_size = model_config.get('patch_size', 16)
    tubelet_size = model_config.get('tubelet_size', 2)
    normalize_target = model_config.get('normalize_target', True)
    
    # Randomly sample videos
    dataset_size = len(dataset)
    num_samples = min(args.num_samples, dataset_size)
    sample_indices = random.sample(range(dataset_size), num_samples)
    
    print(f"Sampling {num_samples} videos from dataset...")
    
    loss_values = []  # for aggregate stats when --compute-loss
    # Process each sampled video
    for i, video_idx in enumerate(sample_indices):
        print(f"\nProcessing video {i+1}/{num_samples} (index {video_idx})...")
        
        try:
            # Load video and mask
            video, mask = dataset[video_idx]
            
            if args.compute_loss:
                loss_val = compute_reconstruction_loss(model, video, mask, config, device)
                loss_values.append(loss_val)
                print(f"  Video {video_idx}: reconstruction MSE = {loss_val:.6f}")
            
            # Process video
            original_vis, masked_vis, reconstructed_vis = process_video_for_visualization(
                model,
                video,
                mask,
                config,
                device,
                patch_size,
                tubelet_size,
                normalize_target
            )
            
            # Create visualization grid
            grid_image = create_side_by_side_grid(
                original_vis,
                masked_vis,
                reconstructed_vis,
                num_frames=args.frames_to_show,
                grid_cols=args.grid_cols
            )
            
            # Save visualization
            save_visualization_grid(
                grid_image,
                args.output_dir,
                video_idx,
                frame_indices=None
            )
            
        except Exception as e:
            print(f"Error processing video {video_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if args.compute_loss and loss_values:
        mean_mse = float(np.mean(loss_values))
        n = len(loss_values)
        print(f"\nMean reconstruction MSE over {n} sample(s): {mean_mse:.6f}")
        if n > 1:
            std_mse = float(np.std(loss_values))
            print(f"Std reconstruction MSE: {std_mse:.6f}")
    
    print(f"\nVisualization complete! Saved {num_samples} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()

