"""
Checkpoint Management Utilities

This module provides utility functions for managing and cleaning up checkpoint files
to ensure only the desired number of checkpoints are kept on disk.
"""

import os
import glob
import re
from pathlib import Path
from typing import List, Optional, Tuple


def parse_checkpoint_filename(filename: str, prefix: str) -> Optional[Tuple[int, float]]:
    """
    Parse checkpoint filename to extract epoch number and metric value.
    
    Args:
        filename (str): Checkpoint filename
        prefix (str): Expected prefix for checkpoint files
    
    Returns:
        tuple: (epoch, metric_value) if parsing successful, None otherwise
    """
    # Remove prefix
    if not filename.startswith(prefix):
        return None
    
    suffix = filename[len(prefix):]
    
    # Try to match patterns like: "epoch=02-train_loss_epoch=0.1234.ckpt"
    # or "epoch=02.ckpt" or "-02-0.1234.ckpt"
    patterns = [
        r'epoch=(\d+)-.*?=([\d.]+)\.ckpt',  # epoch=02-train_loss_epoch=0.1234.ckpt
        r'epoch=(\d+)\.ckpt',  # epoch=02.ckpt
        r'-(\d+)-([\d.]+)\.ckpt',  # -02-0.1234.ckpt
        r'-(\d+)\.ckpt',  # -02.ckpt
    ]
    
    for pattern in patterns:
        match = re.search(pattern, suffix)
        if match:
            epoch = int(match.group(1))
            metric = float(match.group(2)) if len(match.groups()) > 1 and match.group(2) else None
            return (epoch, metric)
    
    return None


def get_checkpoint_files(checkpoint_dir: str, prefix: str) -> List[Tuple[str, int, Optional[float]]]:
    """
    Get all checkpoint files matching the prefix, sorted by epoch.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        prefix (str): Prefix for checkpoint files
    
    Returns:
        list: List of tuples (filepath, epoch, metric_value) sorted by epoch
    """
    checkpoint_files = []
    
    # Find all checkpoint files
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.ckpt")
    files = glob.glob(pattern)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        parsed = parse_checkpoint_filename(filename, prefix)
        if parsed:
            epoch, metric = parsed
            checkpoint_files.append((filepath, epoch, metric))
    
    # Sort by epoch (descending - newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    return checkpoint_files


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    prefix: str,
    keep_top_k: int,
    keep_last: bool = False,
    sort_by_metric: bool = False,
    metric_mode: str = 'min'
) -> List[str]:
    """
    Clean up old checkpoint files, keeping only the top k checkpoints.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        prefix (str): Prefix for checkpoint files
        keep_top_k (int): Number of top checkpoints to keep
        keep_last (bool): Whether to also keep the last checkpoint (by epoch)
        sort_by_metric (bool): If True, sort by metric value; if False, sort by epoch
        metric_mode (str): 'min' or 'max' - determines which metric values are "best"
    
    Returns:
        list: List of deleted checkpoint file paths
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoint_files = get_checkpoint_files(checkpoint_dir, prefix)
    
    if len(checkpoint_files) <= keep_top_k + (1 if keep_last else 0):
        return []  # No cleanup needed
    
    deleted_files = []
    
    if sort_by_metric:
        # Sort by metric value (best first based on mode)
        # Filter out files without metric values
        files_with_metric = [(f, e, m) for f, e, m in checkpoint_files if m is not None]
        files_without_metric = [(f, e, m) for f, e, m in checkpoint_files if m is None]
        
        if metric_mode == 'min':
            files_with_metric.sort(key=lambda x: x[2])  # Lower is better
        else:  # max
            files_with_metric.sort(key=lambda x: x[2], reverse=True)  # Higher is better
        
        # Keep top k by metric
        keep_files = set(f[0] for f in files_with_metric[:keep_top_k])
        
        # Also keep last checkpoint if requested
        if keep_last and checkpoint_files:
            last_file = checkpoint_files[-1][0]  # Last by epoch
            keep_files.add(last_file)
        
        # Delete all others
        all_files = set(f[0] for f in checkpoint_files)
        files_to_delete = all_files - keep_files
        
    else:
        # Sort by epoch (newest first)
        # Keep top k by epoch
        keep_files = set(f[0] for f in checkpoint_files[:keep_top_k])
        
        # Also keep last checkpoint if requested (but it's already in top k if it's newest)
        if keep_last and checkpoint_files:
            last_file = checkpoint_files[-1][0]  # Last by epoch (oldest)
            keep_files.add(last_file)
        
        # Delete all others
        all_files = set(f[0] for f in checkpoint_files)
        files_to_delete = all_files - keep_files
    
    # Delete files
    for filepath in files_to_delete:
        try:
            os.remove(filepath)
            deleted_files.append(filepath)
            print(f"Deleted checkpoint: {filepath}")
        except OSError as e:
            print(f"Error deleting checkpoint {filepath}: {e}")
    
    return deleted_files


def get_checkpoint_stats(checkpoint_dir: str, prefix: str) -> dict:
    """
    Get statistics about checkpoint files.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        prefix (str): Prefix for checkpoint files
    
    Returns:
        dict: Statistics including total count, total size, etc.
    """
    checkpoint_files = get_checkpoint_files(checkpoint_dir, prefix)
    
    total_size = 0
    for filepath, _, _ in checkpoint_files:
        if os.path.exists(filepath):
            total_size += os.path.getsize(filepath)
    
    return {
        'count': len(checkpoint_files),
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'total_size_gb': total_size / (1024 * 1024 * 1024),
        'files': [os.path.basename(f[0]) for f in checkpoint_files]
    }
