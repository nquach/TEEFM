"""
Model Factory for VideoMAE Models

This module provides a factory function to create VideoMAE models with different
backbone architectures (ViT-S, ViT-B, ViT-L, ViT-H) and supports loading pretrained weights.
"""

import torch
from timm.models import create_model
from collections import OrderedDict
import modeling_pretrain
import modeling_finetune
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import load_state_dict
except ImportError:
    # Fallback if utils is not available
    def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
        """Fallback load_state_dict if utils is not available."""
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)}")


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
        backbone (str): Backbone architecture - 'vit-s', 'vit-b', 'vit-l', or 'vit-h'
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
        ValueError: If backbone is not one of 'vit-s', 'vit-b', 'vit-l', 'vit-h'
    """
    # Map backbone names to model registration names
    backbone_map = {
        'vit-s': 'pretrain_videoms_small_patch16_224',
        'vit-b': 'pretrain_videoms_base_patch16_224',
        'vit-l': 'pretrain_videoms_large_patch16_224',
        'vit-h': 'pretrain_videoms_huge_patch16_224'
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
            
            # Strip DDP/DeepSpeed prefixes from keys (e.g. module., module.module., engine.module.)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                key = k
                while key.startswith('module.'):
                    key = key[7:]
                if key.startswith('engine.module.'):
                    key = key[14:]
                elif key.startswith('model.'):
                    key = key[6:]
                elif key.startswith('engine.'):
                    key = key[7:]
                new_state_dict[key] = v
            state_dict = new_state_dict
            
            # Load state dict with strict=False for flexibility
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrained weights: {len(missing_keys)} keys")
                print(f"  Sample missing: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading pretrained weights: {len(unexpected_keys)} keys")
                print(f"  Sample unexpected: {unexpected_keys[:5]}")
            
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
        backbone (str): Backbone architecture - 'vit-s', 'vit-b', 'vit-l', or 'vit-h'
    
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
        },
        'vit-h': {
            'embed_dim': 1280,
            'depth': 32,
            'num_heads': 16,
            'decoder_embed_dim': 640,
            'decoder_depth': 4,
            'decoder_num_heads': 8
        }
    }
    
    if backbone.lower() not in info_map:
        raise ValueError(
            f"Unknown backbone: {backbone}. Must be one of {list(info_map.keys())}"
        )
    
    return info_map[backbone.lower()]


def create_videomae_finetune_model(
    backbone='vit-b',
    pretrained_path=None,
    num_classes=101,
    num_frames=16,
    tubelet_size=2,
    input_size=224,
    fc_drop_rate=0.0,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    use_checkpoint=False,
    use_mean_pooling=True,
    init_scale=0.001,
    mcm=False,
    mcm_ratio=0.4,
    model_key='model|module',
    model_prefix='',
    task_type='classification',
    output_dim=1
):
    """
    Create a VideoMAE finetuning model with classification or regression head.
    
    This function creates a finetuning model (with classification or regression head) and loads
    pretrained encoder weights from a checkpoint, handling:
    - Key prefix stripping (backbone., encoder.)
    - Head weight removal if shape mismatch
    - Position embedding interpolation for different spatial/temporal sizes
    
    Args:
        backbone (str): Backbone architecture - 'vit-s', 'vit-b', 'vit-l', or 'vit-h'
        pretrained_path (str, optional): Path to pretrained checkpoint file (from pretraining)
        num_classes (int): Number of classification classes (default: 101). 
                         Used when task_type='classification'
        num_frames (int): Number of frames (default: 16)
        tubelet_size (int): Tubelet size for patch embedding (default: 2)
        input_size (int): Input image size (default: 224)
        fc_drop_rate (float): Dropout rate for FC layer (default: 0.0)
        drop_rate (float): Dropout rate (default: 0.0)
        drop_path_rate (float): Drop path rate (default: 0.1)
        attn_drop_rate (float): Attention dropout rate (default: 0.0)
        use_checkpoint (bool): Whether to use gradient checkpointing (default: False)
        use_mean_pooling (bool): Whether to use mean pooling (default: True)
        init_scale (float): Initial scale for head weights (default: 0.001)
        mcm (bool): Whether to use motion-centric masking (default: False)
        mcm_ratio (float): Motion-centric masking ratio (default: 0.4)
        model_key (str): Keys to try in checkpoint dict, separated by '|' (default: 'model|module')
        model_prefix (str): Prefix to add when loading state dict (default: '')
        task_type (str): Task type - 'classification' or 'regression' (default: 'classification')
        output_dim (int): Output dimension for regression (default: 1). 
                         Used when task_type='regression'
    
    Returns:
        torch.nn.Module: VideoMAE finetuning model instance
    
    Raises:
        ValueError: If backbone is not one of 'vit-s', 'vit-b', 'vit-l', 'vit-h'
        ValueError: If task_type is not 'classification' or 'regression'
    """
    # Validate task_type
    task_type = task_type.lower()
    if task_type not in ['classification', 'regression']:
        raise ValueError(
            f"Unknown task_type: {task_type}. Must be one of 'classification' or 'regression'"
        )
    
    # Map backbone names to finetuning model registration names
    backbone_map = {
        'vit-s': 'vit_small_patch16_224',
        'vit-b': 'vit_base_patch16_224',
        'vit-l': 'vit_large_patch16_224',
        'vit-h': 'vit_huge_patch16_224'
    }
    
    if backbone.lower() not in backbone_map:
        raise ValueError(
            f"Unknown backbone: {backbone}. Must be one of {list(backbone_map.keys())}"
        )
    
    model_name = backbone_map[backbone.lower()]
    
    # Determine output dimension based on task type
    if task_type == 'regression':
        head_output_dim = output_dim
        print(f"Creating regression model with output_dim={output_dim}")
    else:
        head_output_dim = num_classes
        print(f"Creating classification model with num_classes={num_classes}")
    
    # Create finetuning model using timm's create_model
    # The model is registered in modeling_finetune.py
    model = create_model(
        model_name,
        pretrained=False,  # We handle pretrained loading separately
        num_classes=head_output_dim,
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        fc_drop_rate=fc_drop_rate,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=use_checkpoint,
        use_mean_pooling=use_mean_pooling,
        init_scale=init_scale,
        mcm=mcm,
        mcm_ratio=mcm_ratio
    )
    
    # Load pretrained weights if specified
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        try:
            # Load checkpoint
            if pretrained_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    pretrained_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            checkpoint_model = None
            for key in model_key.split('|'):
                if key in checkpoint:
                    checkpoint_model = checkpoint[key]
                    print(f"Load state_dict by model_key = {key}")
                    break
            
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            
            # Remove head weights if shape mismatch (pretraining has no head)
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
                    del checkpoint_model[k]
            
            # Strip key prefixes (backbone., encoder.)
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]  # Remove 'backbone.' prefix
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]  # Remove 'encoder.' prefix
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict
            
            # Interpolate position embedding if spatial size differs
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1
                
                # Calculate original and new spatial sizes
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // 
                                (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
                new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
                
                # Interpolate if sizes differ
                if orig_size != new_size:
                    print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    
                    # Reshape for interpolation: B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(
                        -1, num_frames // model.patch_embed.tubelet_size, 
                        orig_size, orig_size, embedding_size
                    )
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    
                    # Interpolate
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
                    )
                    
                    # Reshape back: BT, C, H, W -> BT, H, W, C -> B, T, H, W, C -> B, L, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                        -1, num_frames // model.patch_embed.tubelet_size, 
                        new_size, new_size, embedding_size
                    )
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed
            
            # Load state dict using flexible loading function
            load_state_dict(model, checkpoint_model, prefix=model_prefix)
            print("Successfully loaded pretrained weights")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with random initialization")
    else:
        print("Initializing model with random weights")
    
    return model

