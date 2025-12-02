"""
Model Factory for VideoMAE Models

This module provides a factory function to create VideoMAE models with different
backbone architectures (ViT-S, ViT-B, ViT-L) and supports loading pretrained weights.
"""

import torch
from timm.models import create_model
import modeling_pretrain


def create_videomae_model(
    backbone='vit-s',
    pretrained_path=None,
    decoder_depth=4,
    drop_path=0.0,
    mask_type='motion-centric',
    mask_ratio=0.9,
    motion_centric_masking_ratio=0.7,
    use_checkpoint=False
):
    """
    Create a VideoMAE model with the specified backbone architecture.
    
    Args:
        backbone (str): Backbone architecture - 'vit-s', 'vit-b', or 'vit-l'
        pretrained_path (str, optional): Path to pretrained checkpoint file
        decoder_depth (int): Depth of the decoder (default: 4)
        drop_path (float): Drop path rate for regularization (default: 0.0)
        mask_type (str): Masking strategy - 'random', 'tube', or 'motion-centric'
        mask_ratio (float): Ratio of patches to mask (default: 0.9)
        motion_centric_masking_ratio (float): Ratio for motion-centric masking (default: 0.7)
        use_checkpoint (bool): Whether to use gradient checkpointing (default: False)
    
    Returns:
        torch.nn.Module: VideoMAE model instance
    
    Raises:
        ValueError: If backbone is not one of 'vit-s', 'vit-b', 'vit-l'
    """
    # Map backbone names to model registration names
    backbone_map = {
        'vit-s': 'pretrain_videoms_small_patch16_224',
        'vit-b': 'pretrain_videoms_base_patch16_224',
        'vit-l': 'pretrain_videoms_large_patch16_224'
    }
    
    if backbone.lower() not in backbone_map:
        raise ValueError(
            f"Unknown backbone: {backbone}. Must be one of {list(backbone_map.keys())}"
        )
    
    model_name = backbone_map[backbone.lower()]
    
    # Create model using timm's create_model
    # The model is registered in modeling_pretrain.py
    model = create_model(
        model_name,
        pretrained=False,  # We handle pretrained loading separately
        drop_path_rate=drop_path,
        drop_block_rate=None,
        decoder_depth=decoder_depth,
        use_checkpoint=use_checkpoint,
        motion_centric_masking=(mask_type == 'motion-centric'),
        motion_centric_masking_ratio=motion_centric_masking_ratio,
        masking_ratio=mask_ratio
    )
    
    # Load pretrained weights if specified
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it's a full checkpoint with 'model' key
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                # Check if it's a state_dict directly
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                # Otherwise assume the dict itself is the state_dict
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False for flexibility
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrained weights: {len(missing_keys)} keys")
                if len(missing_keys) <= 10:  # Only print if not too many
                    print(f"Missing keys: {missing_keys}")
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading pretrained weights: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:  # Only print if not too many
                    print(f"Unexpected keys: {unexpected_keys}")
            
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization")
    else:
        print("Initializing model with random weights")
    
    return model


def get_model_info(backbone):
    """
    Get information about a model architecture.
    
    Args:
        backbone (str): Backbone architecture - 'vit-s', 'vit-b', or 'vit-l'
    
    Returns:
        dict: Dictionary containing model information (embed_dim, depth, num_heads, etc.)
    """
    info_map = {
        'vit-s': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'decoder_embed_dim': 192,
            'decoder_depth': 4,
            'decoder_num_heads': 3
        },
        'vit-b': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'decoder_embed_dim': 384,
            'decoder_depth': 4,
            'decoder_num_heads': 6
        },
        'vit-l': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'decoder_embed_dim': 512,
            'decoder_depth': 4,
            'decoder_num_heads': 8
        }
    }
    
    if backbone.lower() not in info_map:
        raise ValueError(
            f"Unknown backbone: {backbone}. Must be one of {list(info_map.keys())}"
        )
    
    return info_map[backbone.lower()]

