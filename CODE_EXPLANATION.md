# Codebase Explanation

This document provides a detailed explanation of all components in the VideoMAE EVEREST implementation.

## Overview

The codebase implements VideoMAE (Video Masked Autoencoder) training with EVEREST masking strategy. It uses PyTorch Lightning for training infrastructure and supports multiple Vision Transformer backbones (ViT-S, ViT-B, ViT-L).

## Architecture Overview

```
Input Video [B, T, C, H, W]
    ↓
Patch Embedding → Tokens [B, T*num_patches, embed_dim]
    ↓
Positional + Temporal Embeddings
    ↓
EVEREST Masking → Select Informative Tokens
    ↓
ViT Encoder (processes visible tokens only)
    ↓
Decoder (reconstructs all tokens)
    ↓
Reconstructed Patches [B, num_tokens, patch_size²*3]
```

## Component Details

### 1. Data Loading (`data/video_dataset.py`)

**Purpose**: Loads videos from CSV files and prepares them for training.

**Key Features**:
- Reads video paths from CSV (one path per line)
- Uses `torchvision.io.read_video` for efficient video loading
- Randomly samples 32 consecutive frames from each video
- Applies temporal downsampling with stride 2 → 16 frames
- Optional random subset sampling for faster experimentation

**Key Methods**:
- `__init__()`: Initializes dataset, reads CSV, optionally applies subset sampling
- `__getitem__()`: Loads a single video, samples frames, applies temporal downsampling
- Returns tensor of shape `[16, 3, 224, 224]` (16 frames, RGB, 224x224)

**Important Notes**:
- Videos must already be preprocessed to 224x224x3
- Handles videos with insufficient frames gracefully
- Normalizes pixel values to [0, 1] range

### 2. EVEREST Masking (`models/everest_masking.py`)

**Purpose**: Implements the EVEREST masking strategy that identifies informative tokens based on motion.

**Key Concepts**:
- **Motion Importance**: Computes frame differences to estimate motion
- **Token Selection**: Selects top-k most informative tokens to keep (unmasked)
- **Efficiency**: Only processes visible tokens through encoder, reducing computation

**Key Classes**:
- `EverestMaskingGenerator`: Main masking generator class

**Key Methods**:
- `compute_motion_importance()`: 
  - Converts video to grayscale
  - Computes temporal differences between consecutive frames
  - Aggregates motion per patch
  - Returns importance scores for each spatiotemporal token
  
- `generate_mask()`:
  - Computes motion importance
  - Selects top-k tokens to keep
  - Returns binary mask (1 = keep, 0 = mask)
  
- `apply_mask()`:
  - Separates visible tokens from all tokens
  - Creates `ids_restore` to reconstruct original token order
  - Returns visible tokens and restoration indices

**Algorithm**:
1. Compute motion between consecutive frames
2. Average motion within each patch
3. Rank tokens by motion importance
4. Keep top (1 - mask_ratio) tokens
5. Mask remaining tokens for reconstruction

### 3. VideoMAE Model (`models/videomae.py`)

**Purpose**: Implements the VideoMAE architecture with EVEREST masking integration.

**Architecture Components**:

#### 3.1 Patch Embedding (`PatchEmbed`)
- Converts video frames to tokens using 2D convolution
- Splits each 224x224 frame into 14x14 patches (patch_size=16)
- Projects each patch to embedding dimension
- Output: `[B, T*num_patches, embed_dim]`

#### 3.2 Positional Encoding (`PositionalEncoding`)
- Learnable positional embeddings for spatial patches
- Adds spatial position information to tokens
- Shared across all frames

#### 3.3 Temporal Embedding (`TemporalEmbedding`)
- Learnable temporal embeddings for video frames
- Adds temporal position information
- Different embedding for each frame

#### 3.4 Vision Transformer Encoder (`VisionTransformerEncoder`)
- Stack of Transformer blocks
- Processes only visible (unmasked) tokens
- Reduces computation compared to processing all tokens

**Backbone Configurations**:
- **ViT-S**: embed_dim=384, depth=12, num_heads=6
- **ViT-B**: embed_dim=768, depth=12, num_heads=12
- **ViT-L**: embed_dim=1024, depth=24, num_heads=16

#### 3.5 Decoder (`VideoMAEDecoder`)
- Lightweight decoder (8 layers, 512 dim)
- Reconstructs all tokens (visible + masked)
- Uses mask tokens for masked positions
- Outputs pixel values for each patch

**Forward Pass Flow**:
1. Generate EVEREST mask from original video
2. Patch embedding + positional/temporal embeddings
3. Apply mask → separate visible tokens
4. Encode visible tokens only
5. Decode all tokens (visible + mask tokens)
6. Return reconstructed patches

### 4. PyTorch Lightning Module (`training/lightning_module.py`)

**Purpose**: Wraps VideoMAE model for PyTorch Lightning training.

**Key Features**:
- Automatic multi-GPU support
- Built-in checkpointing and logging
- Gradient norm tracking (optional)
- MSE loss with normalization

**Key Methods**:

#### `training_step()`
- Forward pass through model
- Computes loss on masked tokens only
- Logs training loss
- Optionally tracks gradient norm

#### `validation_step()`
- Forward pass on validation data
- Computes validation loss
- Logs validation metrics

#### `compute_loss()`
- MSE loss with normalization
- Normalizes target patches by mean/std
- Applies mask to compute loss only on masked tokens
- Returns scalar loss value

#### `prepare_target_patches()`
- Extracts patches from input video
- Matches format of decoder output
- Uses `F.unfold` to extract non-overlapping patches

**Loss Function Details**:
- Target patches normalized: `(x - mean) / std`
- Loss computed only on masked tokens (reconstruction task)
- Helps with training stability

### 5. Optimizer Factory (`utils/optimizer.py`)

**Purpose**: Creates AdamWScheduleFree optimizer.

**Key Features**:
- Schedule-free optimizer (no learning rate schedule needed)
- Automatically adapts learning rate during training
- Configurable learning rate and weight decay

**Usage**:
```python
optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.05)
```

### 6. Training Script (`training/train.py`)

**Purpose**: Main entry point for training.

**Workflow**:
1. Load configuration from YAML
2. Create datasets (train/val)
3. Create data loaders
4. Initialize VideoMAE model
5. Create Lightning module
6. Setup PyTorch Lightning Trainer
7. Start training

**Key Functions**:
- `load_config()`: Loads YAML configuration
- `create_datasets()`: Creates train/val datasets
- `create_data_loaders()`: Creates PyTorch DataLoaders
- `create_model()`: Initializes VideoMAE model
- `create_lightning_module()`: Wraps model in Lightning module
- `create_trainer()`: Sets up PyTorch Lightning Trainer with callbacks

**Trainer Configuration**:
- ModelCheckpoint callback for saving best models
- TensorBoard logger for experiment tracking
- Automatic multi-GPU support
- Gradient accumulation support
- Mixed precision training support

### 7. Configuration (`configs/config.yaml`)

**Purpose**: Centralized configuration for all hyperparameters.

**Sections**:
- **model**: Backbone, pretrained weights, masking ratio
- **data**: CSV paths, frame sampling, subset settings
- **training**: Batch size, learning rate, epochs, GPU settings
- **checkpoint**: Directory, prefix, save settings
- **logging**: Log directory, experiment name

**Usage**:
All hyperparameters can be modified in this single file without changing code.

## Training Flow

1. **Data Loading**:
   - Videos loaded from CSV files
   - 32 consecutive frames sampled randomly
   - Temporal downsampling (stride 2) → 16 frames
   - Batch shape: `[B, 16, 3, 224, 224]`

2. **Forward Pass**:
   - Patch embedding: `[B, 16, 3, 224, 224]` → `[B, 16*196, 384]`
   - EVEREST masking: Select informative tokens
   - Encoder: Process visible tokens only
   - Decoder: Reconstruct all tokens
   - Output: `[B, 16*196, 16*16*3]` (reconstructed patches)

3. **Loss Computation**:
   - Extract target patches from input video
   - Normalize target patches
   - Compute MSE loss on masked tokens only
   - Backpropagate

4. **Optimization**:
   - AdamWScheduleFree optimizer
   - Automatic learning rate adaptation
   - Gradient clipping (optional)

## Key Design Decisions

1. **EVEREST Masking**: Reduces computation by processing only informative tokens
2. **PyTorch Lightning**: Simplifies multi-GPU training and checkpointing
3. **YAML Configuration**: Easy hyperparameter tuning without code changes
4. **Normalized Loss**: Improves training stability
5. **Modular Design**: Each component is independent and testable

## Extension Points

1. **Data Augmentation**: Add transforms in `utils/transforms.py`
2. **Additional Backbones**: Add to `backbone_configs` in `VideoMAE.__init__()`
3. **Custom Loss Functions**: Modify `compute_loss()` in Lightning module
4. **Additional Metrics**: Add to `training_step()` and `validation_step()`

## Performance Considerations

1. **Memory**: Batch size should be adjusted based on GPU memory
2. **Data Loading**: `num_workers` can be increased for faster loading
3. **Mixed Precision**: Set `precision: 16` in config for faster training
4. **Gradient Accumulation**: Use for larger effective batch sizes

## Troubleshooting

- **Out of Memory**: Reduce batch size or enable gradient accumulation
- **Slow Training**: Increase `num_workers`, use mixed precision
- **Poor Convergence**: Adjust learning rate, check data preprocessing
- **Masking Issues**: Verify mask ratio and motion computation

