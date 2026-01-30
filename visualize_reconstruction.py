"""
Video Reconstruction Visualization Script

This script loads a pretrained VideoMAE model, samples random videos from the
optimized dataset, and generates visualizations showing reconstruction quality.
"""

import os
import argparse
import yaml
import torch
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
    
    # Process each sampled video
    for i, video_idx in enumerate(sample_indices):
        print(f"\nProcessing video {i+1}/{num_samples} (index {video_idx})...")
        
        try:
            # Load video and mask
            video, mask = dataset[video_idx]
            
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
    
    print(f"\nVisualization complete! Saved {num_samples} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()

