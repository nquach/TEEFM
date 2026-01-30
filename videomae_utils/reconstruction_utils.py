"""
Reconstruction Utilities for VideoMAE Visualization

This module provides utility functions for converting reconstructed patches
back to video format, creating masked video visualizations, and generating
side-by-side comparison grids.
"""

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
import os


def patches_to_video(
    reconstructed_patches,
    mask,
    original_video,
    patch_size,
    tubelet_size,
    normalize_target=True,
    original_patches=None
):
    """
    Convert reconstructed patches back to video tensor format.
    
    Args:
        reconstructed_patches (torch.Tensor): Reconstructed patches from decoder
            Shape: [B, num_masked_patches, patch_dim]
        mask (torch.Tensor): Boolean mask indicating which patches were masked
            Shape: [B, num_patches] or tuple for MCM
        original_video (torch.Tensor): Original video tensor [B, C, T, H, W]
        patch_size (int): Size of each patch (e.g., 16)
        tubelet_size (int): Temporal tubelet size (e.g., 2)
        normalize_target (bool): Whether patches were normalized per-patch
        original_patches (torch.Tensor, optional): Original patches for unmasked regions
            Shape: [B, num_patches, patch_dim]
    
    Returns:
        torch.Tensor: Reconstructed video tensor [B, C, T, H, W]
    """
    B, C, T, H, W = original_video.shape
    device = reconstructed_patches.device
    
    # Handle MCM mask format (tuple of masks)
    if isinstance(mask, tuple):
        # For MCM: mask[0] is encoder mask, mask[1] is target mask.
        # Decoder predicts only positions where mask[1] is False (0); use ~mask[1] for placement.
        target_mask = mask[1]  # [B, num_patches]
        bool_masked_pos = ~target_mask.to(torch.bool)
    else:
        bool_masked_pos = mask.to(torch.bool) if mask is not None else None
    
    # Calculate patch dimensions
    num_patches_per_frame = (H // patch_size) * (W // patch_size)
    num_frames_patches = T // tubelet_size
    
    # Patch dimension: 3 * tubelet_size * patch_size^2
    patch_dim = 3 * tubelet_size * patch_size * patch_size
    
    # Get original patches if not provided
    if original_patches is None:
        # Extract original patches from original video
        original_patches = rearrange(
            original_video,
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
            p0=tubelet_size,
            p1=patch_size,
            p2=patch_size
        )
    
    # Create full patch tensor
    full_patches = original_patches.clone()
    
    # Place reconstructed patches back into full patch tensor
    if bool_masked_pos is not None and reconstructed_patches.numel() > 0:
        # reconstructed_patches: [B, num_masked, patch_dim]
        # We need to place them at the masked positions
        # When normalize_target=True, decoder outputs per-patch normalized values; scale to [0,1] for display
        for b in range(B):
            masked_indices = bool_masked_pos[b]
            num_masked = masked_indices.sum().item()
            if num_masked > 0:
                patches = reconstructed_patches[b, :num_masked].float()
                if normalize_target:
                    # Per-patch min-max to [0,1] for display (decoder predicts normalized space)
                    p_min = patches.min(dim=1, keepdim=True)[0]
                    p_max = patches.max(dim=1, keepdim=True)[0]
                    patches = (patches - p_min) / (p_max - p_min + 1e-6)
                full_patches[b, masked_indices] = patches
    
    # Reshape patches back to video format
    # full_patches: [B, num_patches, patch_dim]
    # Reshape to [B, num_frames_patches, h_p, w_p, tubelet_size, patch_size, patch_size, 3] (8D for rearrange)
    h_p = H // patch_size
    w_p = W // patch_size
    patches_reshaped = full_patches.reshape(
        B, num_frames_patches, h_p, w_p, tubelet_size, patch_size, patch_size, 3
    )
    
    # Rearrange to video format: [B, C, T, H, W]
    reconstructed_video = rearrange(
        patches_reshaped,
        'b t_f h_p w_p t_t h_patch w_patch c -> b c (t_f t_t) (h_p h_patch) (w_p w_patch)',
        t_f=num_frames_patches,
        h_p=h_p,
        w_p=w_p,
        t_t=tubelet_size,
        h_patch=patch_size,
        w_patch=patch_size
    )
    
    return reconstructed_video


def create_masked_video(video, mask, patch_size, tubelet_size):
    """
    Create visualization of masked video by replacing masked patches with gray regions.
    
    Args:
        video (torch.Tensor): Original video tensor [B, C, T, H, W]
        mask (torch.Tensor): Boolean mask indicating masked patches [B, num_patches] or tuple
        patch_size (int): Size of each patch
        tubelet_size (int): Temporal tubelet size
    
    Returns:
        torch.Tensor: Masked video tensor [B, C, T, H, W] with gray masked regions
    """
    B, C, T, H, W = video.shape
    device = video.device
    
    # Handle MCM mask format
    if isinstance(mask, tuple):
        # For MCM: gray out target (reconstruction) positions (mask[1] False = where we predict)
        target_mask = mask[1]
        bool_masked_pos = ~target_mask.to(torch.bool)
    else:
        bool_masked_pos = mask.to(torch.bool) if mask is not None else None
    
    if bool_masked_pos is None:
        return video
    
    # Create masked video copy
    masked_video = video.clone()
    
    # Convert to patches
    patches = rearrange(
        masked_video,
        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size
    )
    
    # Set masked patches to gray (0.5 for normalized, or 128 for unnormalized)
    # Check if video is normalized (values typically in [-2, 2] range)
    if video.abs().max() > 1.5:
        # Likely normalized, use gray value that denormalizes to ~128
        gray_value = 0.0  # Will be adjusted after denormalization
    else:
        # Likely in [0, 1] range
        gray_value = 0.5
    
    # Create gray patches
    num_masked = bool_masked_pos.sum(dim=1).max().item()
    gray_patches = torch.full(
        (B, num_masked, patches.shape[-1]),
        gray_value,
        device=device,
        dtype=patches.dtype
    )
    
    # Replace masked patches
    for b in range(B):
        masked_indices = bool_masked_pos[b]
        patches[b, masked_indices] = gray_patches[b, :masked_indices.sum()]
    
    # Reshape back to video (8D for rearrange: split spatial patches into h_p, w_p)
    num_frames_patches = T // tubelet_size
    h_p = H // patch_size
    w_p = W // patch_size
    
    patches_3d = patches.reshape(
        B, num_frames_patches, h_p, w_p, tubelet_size, patch_size, patch_size, 3
    )
    
    masked_video = rearrange(
        patches_3d,
        'b t_f h_p w_p t_t h_patch w_patch c -> b c (t_f t_t) (h_p h_patch) (w_p w_patch)',
        t_f=num_frames_patches,
        h_p=h_p,
        w_p=w_p,
        t_t=tubelet_size,
        h_patch=patch_size,
        w_patch=patch_size
    )
    
    return masked_video


def denormalize_video(video, mean, std):
    """
    Denormalize video from model input format to [0, 1] range.
    
    Args:
        video (torch.Tensor): Normalized video tensor [B, C, T, H, W] or [C, T, H, W]
        mean (list or torch.Tensor): Mean values [R, G, B]
        std (list or torch.Tensor): Standard deviation values [R, G, B]
    
    Returns:
        torch.Tensor: Denormalized video in [0, 1] range
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean, device=video.device, dtype=video.dtype)
    if isinstance(std, list):
        std = torch.tensor(std, device=video.device, dtype=video.dtype)
    
    # Add dimensions for broadcasting: [1, 3, 1, 1, 1] for [B, C, T, H, W]
    if video.dim() == 5:
        mean = mean.view(1, 3, 1, 1, 1)
        std = std.view(1, 3, 1, 1, 1)
    elif video.dim() == 4:
        mean = mean.view(3, 1, 1, 1)
        std = std.view(3, 1, 1, 1)
    
    # Denormalize: x = (x_norm * std) + mean
    denorm_video = video * std + mean
    
    # Clamp to [0, 1] range
    denorm_video = torch.clamp(denorm_video, 0.0, 1.0)
    
    return denorm_video


def video_to_numpy(video):
    """
    Convert video tensor to numpy array for visualization.
    
    Args:
        video (torch.Tensor): Video tensor [B, C, T, H, W] or [C, T, H, W]
    
    Returns:
        numpy.ndarray: Video array in [T, H, W, C] format, values in [0, 255]
    """
    # Handle batch dimension
    if video.dim() == 5:
        video = video[0]  # Take first batch item
    
    # Convert [C, T, H, W] to [T, H, W, C]
    video_np = video.permute(1, 2, 3, 0).cpu().numpy()
    
    # Convert from [0, 1] to [0, 255]
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    return video_np


def create_side_by_side_grid(original, masked, reconstructed, num_frames=8, grid_cols=3):
    """
    Create side-by-side image grid showing original, masked, and reconstructed videos.
    
    Args:
        original (torch.Tensor): Original video [B, C, T, H, W] or [C, T, H, W]
        masked (torch.Tensor): Masked video [B, C, T, H, W] or [C, T, H, W]
        reconstructed (torch.Tensor): Reconstructed video [B, C, T, H, W] or [C, T, H, W]
        num_frames (int): Number of frames to show
        grid_cols (int): Number of columns (default: 3 for original/masked/reconstructed)
    
    Returns:
        PIL.Image: Image grid as PIL Image
    """
    # Convert to numpy
    orig_np = video_to_numpy(original)
    masked_np = video_to_numpy(masked)
    recon_np = video_to_numpy(reconstructed)
    
    T, H, W, C = orig_np.shape
    
    # Sample frames evenly
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Create grid: rows = frames, cols = 3 (original, masked, reconstructed)
    grid_images = []
    
    for frame_idx in frame_indices:
        row_images = []
        
        # Original frame
        orig_frame = orig_np[frame_idx]
        row_images.append(orig_frame)
        
        # Masked frame
        masked_frame = masked_np[frame_idx]
        row_images.append(masked_frame)
        
        # Reconstructed frame
        recon_frame = recon_np[frame_idx]
        row_images.append(recon_frame)
        
        # Concatenate horizontally
        row = np.concatenate(row_images, axis=1)
        grid_images.append(row)
    
    # Concatenate vertically
    grid = np.concatenate(grid_images, axis=0)
    
    # Convert to PIL Image
    grid_image = Image.fromarray(grid)
    
    return grid_image


def save_visualization_grid(grid_image, output_path, video_index, frame_indices=None):
    """
    Save visualization grid as PNG file.
    
    Args:
        grid_image (PIL.Image): Image grid to save
        output_path (str): Directory to save the image
        video_index (int): Index of the video
        frame_indices (list, optional): Frame indices shown in grid
    """
    os.makedirs(output_path, exist_ok=True)
    
    filename = f"video_{video_index}_reconstruction_grid.png"
    filepath = os.path.join(output_path, filename)
    
    grid_image.save(filepath, "PNG")
    print(f"Saved visualization to {filepath}")
    
    return filepath
