# VideoMAE Codebase Explanation

This document provides a comprehensive explanation of all components in the VideoMAE training codebase.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Module](#data-module)
4. [Model Architecture](#model-architecture)
5. [Training Module](#training-module)
6. [Training Script](#training-script)
7. [Configuration](#configuration)
8. [Data Flow](#data-flow)
9. [Training Process](#training-process)

---

## Overview

This codebase implements a **Video Masked Autoencoder (VideoMAE)** using the **EVEREST** training method. The model learns representations from unlabeled video data by:

1. Randomly masking 75% of video patches
2. Encoding only the visible (unmasked) patches
3. Reconstructing the masked patches using a decoder
4. Computing reconstruction loss only on masked patches

The architecture uses a **Vision Transformer (ViT)** backbone with options for:
- **ViT-S** (Small): 384 embedding dim, 12 layers, 6 heads
- **ViT-B** (Base): 768 embedding dim, 12 layers, 12 heads
- **ViT-L** (Large): 1024 embedding dim, 24 layers, 16 heads

---

## Project Structure

```
TEEFM/
├── data/
│   ├── __init__.py
│   └── video_dataset.py          # Video dataset and data loading
├── models/
│   ├── __init__.py
│   └── videomae.py                # VideoMAE model implementation
├── training/
│   ├── __init__.py
│   ├── lightning_module.py        # PyTorch Lightning wrapper
│   └── train.py                   # Main training script
├── config/
│   └── config.yaml                # Configuration file
├── mp4_paths.csv                   # Training video paths
├── val500_2023-2024.csv           # Validation video paths
├── requirements.txt                # Python dependencies
├── README.md                       # Usage instructions
└── CODE_EXPLANATION.md             # This file
```

---

## Data Module

### File: `data/video_dataset.py`

#### `VideoDataset` Class

**Purpose**: Loads and processes video files for training/validation.

**Key Components**:

1. **Initialization** (`__init__`):
   - Reads video paths from CSV file (handles both with/without headers)
   - Configures frame sampling parameters:
     - `num_frames_to_sample`: Number of consecutive frames to sample (default: 32)
     - `temporal_stride`: Stride for temporal downsampling (default: 2)
     - `frame_size`: Expected frame dimensions (default: 224x224)

2. **Video Loading** (`__getitem__`):
   - Uses OpenCV to read video frames
   - Converts BGR to RGB color space
   - Handles videos with insufficient frames by padding with last frame
   - **Random Sampling**: Randomly selects 32 consecutive frames from the video
   - **Temporal Downsampling**: Applies stride of 2 to get 16 frames
   - **Normalization**: Converts pixel values from [0, 255] to [0, 1]
   - **Tensor Format**: Returns tensor of shape `(C, T, H, W)` = `(3, 16, 224, 224)`

**Data Processing Pipeline**:
```
Video File → Load Frames → Sample 32 Frames → Downsample (stride=2) → 16 Frames → Normalize → Tensor
```

**Important Notes**:
- Videos are expected to already be 224x224x3 (no resizing needed)
- Random sampling ensures data augmentation
- Temporal downsampling reduces computational cost while maintaining temporal information

---

## Model Architecture

### File: `models/videomae.py`

The VideoMAE model consists of several components:

### 1. `PatchEmbed` Class

**Purpose**: Converts video into patch embeddings.

**How it works**:
- Uses 3D convolution to extract spatio-temporal patches
- Input: `(B, C, T, H, W)` = `(batch, 3, 16, 224, 224)`
- Kernel: `(t_patch_size, patch_size, patch_size)` = `(2, 16, 16)`
- Stride: `(2, 16, 16)` (matches kernel size)
- Output: `(B, num_patches, embed_dim)`

**Patch Calculation**:
- Spatial patches per frame: `(224 / 16)² = 14² = 196`
- Temporal patches: `16 / 2 = 8`
- Total patches: `8 × 196 = 1,568`

### 2. `PositionalEncoding` Class

**Purpose**: Adds learnable positional embeddings to patches.

**Implementation**:
- Learnable parameter tensor: `(1, num_patches, embed_dim)`
- Initialized with truncated normal distribution (std=0.02)
- Added element-wise to patch embeddings

### 3. `TransformerBlock` Class

**Purpose**: Standard transformer block with self-attention and MLP.

**Architecture**:
```
Input → LayerNorm → MultiheadAttention → Residual → LayerNorm → MLP → Residual → Output
```

**Components**:
- **Self-Attention**: Multi-head attention mechanism
  - Compatible with both old and new PyTorch versions
  - Uses `batch_first=True` if available, otherwise transposes
- **MLP**: Two-layer feedforward network
  - Hidden dimension: `embed_dim × mlp_ratio` (typically 4x)
  - Activation: GELU
  - Dropout for regularization

### 4. `VisionTransformer` Class

**Purpose**: Vision Transformer encoder backbone.

**Architecture Flow**:
1. **Patch Embedding**: Convert video to patches
2. **Add Class Token**: Prepend learnable class token (for compatibility)
3. **Positional Encoding**: Add positional embeddings
4. **Transformer Blocks**: Apply N transformer blocks (12 for ViT-S/B, 24 for ViT-L)
5. **Layer Normalization**: Final normalization

**Key Features**:
- Supports different backbone sizes (S, B, L)
- Configurable depth, attention heads, and embedding dimensions
- Proper weight initialization

### 5. `VideoMAE` Class

**Purpose**: Main VideoMAE model implementing masked autoencoding.

#### Architecture Components:

**Encoder**:
- Full Vision Transformer
- Processes only visible (unmasked) patches
- Outputs encoded representations

**Decoder**:
- Lightweight transformer (4 layers vs 12/24 in encoder)
- Takes encoded visible patches + masked token placeholders
- Reconstructs pixel values for all patches

#### Key Methods:

##### `random_masking(x, mask_ratio)`
**Purpose**: Randomly masks patches for EVEREST training.

**Process**:
1. Generate random noise for each patch
2. Sort patches by noise (random shuffle)
3. Keep first `(1 - mask_ratio) × num_patches` patches (visible)
4. Mask remaining patches
5. Return visible patches, binary mask, and restore indices

**Example** (mask_ratio=0.75):
- Total patches: 1,568
- Visible patches: 392 (25%)
- Masked patches: 1,176 (75%)

##### `forward_encoder(x, mask_ratio)`
**Purpose**: Encodes visible patches only.

**Process**:
1. Extract patches from video
2. Apply random masking
3. Add class token to visible patches
4. Add positional encoding
5. Pass through encoder transformer blocks
6. Return encoded patches, mask, and restore indices

##### `forward_decoder(x, ids_restore)`
**Purpose**: Reconstructs all patches (visible + masked).

**Process**:
1. Project encoded patches to decoder dimension
2. Create full sequence with zeros for masked patches
3. Restore original patch order using `ids_restore`
4. Add class token and positional encoding
5. Pass through decoder transformer blocks
6. Predict pixel values for all patches

##### `patchify(x)`
**Purpose**: Converts video to patches (for loss computation).

**Process**:
- Reshapes video tensor into patches
- Same patch structure as encoder embedding
- Used to compute target values for reconstruction loss

##### `compute_loss(pred, target, mask)`
**Purpose**: Computes reconstruction loss on masked patches only.

**Process**:
1. Optionally normalize target patches (if `norm_pix_loss=True`)
2. Compute MSE between predicted and target patches
3. Apply mask (only compute loss on masked patches)
4. Return mean loss over masked patches

**Loss Formula**:
```
loss = mean((pred - target)²) for masked patches only
```

##### `forward(x)`
**Purpose**: Complete forward pass through VideoMAE.

**Process**:
1. Encoder: Encode visible patches
2. Decoder: Reconstruct all patches
3. Compute loss on masked patches
4. Return loss, predictions, and mask

#### Backbone Configurations:

```python
BACKBONE_CONFIGS = {
    'vit_s': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0
    },
    'vit_b': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0
    },
    'vit_l': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0
    }
}
```

#### Pretrained Weights:

The `load_pretrained()` method handles loading pretrained weights:
- Supports different checkpoint formats (`state_dict`, `model`, or direct dict)
- Removes `model.` prefix if present
- Uses `strict=False` to allow partial loading

---

## Training Module

### File: `training/lightning_module.py`

#### `VideoMAELightningModule` Class

**Purpose**: PyTorch Lightning wrapper for VideoMAE training.

**Inherits from**: `pl.LightningModule`

#### Key Components:

##### Initialization (`__init__`)
- Wraps VideoMAE model
- Stores hyperparameters (learning rate, weight decay, warmup steps)
- Configures gradient norm tracking (optional, default: False)
- Saves hyperparameters for logging/reproducibility

##### `forward(x)`
- Simple wrapper around model's forward pass
- Returns `(loss, predictions, mask)`

##### `training_step(batch, batch_idx)`
**Purpose**: Defines training step logic.

**Process**:
1. Forward pass through model
2. Extract loss
3. Log training loss (on step and epoch)
4. Optionally compute and log gradient norm
5. Return loss for backpropagation

**Logging**:
- `train_loss`: Reconstruction loss
- `grad_norm`: L2 norm of gradients (if enabled)

##### `validation_step(batch, batch_idx)`
**Purpose**: Defines validation step logic.

**Process**:
1. Forward pass through model (no gradients)
2. Extract loss
3. Log validation loss (on epoch only)
4. Return loss for monitoring

**Logging**:
- `val_loss`: Validation reconstruction loss

##### `configure_optimizers()`
**Purpose**: Configures optimizer.

**Returns**: `AdamWScheduleFree` optimizer

**AdamWScheduleFree Features**:
- Schedule-free optimizer (no learning rate scheduling needed)
- Combines benefits of AdamW with schedule-free learning
- Parameters:
  - `lr`: Learning rate (default: 1e-4)
  - `weight_decay`: Weight decay (default: 0.05)
  - `warmup_steps`: Number of warmup steps (default: 1000)

##### `compute_grad_norm()`
**Purpose**: Computes L2 norm of all gradients.

**Process**:
1. Iterate through all model parameters
2. Compute L2 norm of each parameter's gradient
3. Sum squared norms
4. Return square root (total gradient norm)

**Use Case**: Monitoring gradient flow and detecting vanishing/exploding gradients

---

## Training Script

### File: `training/train.py`

**Purpose**: Main entry point for training VideoMAE models.

#### Key Functions:

##### `parse_args()`
**Purpose**: Parses command-line arguments.

**Argument Categories**:
- **Data**: CSV paths, batch size, workers, frame sampling
- **Model**: Backbone, image size, patch size, mask ratio, pretrained weights
- **Training**: Learning rate, weight decay, warmup, epochs, gradient tracking
- **Checkpointing**: Directory, prefix, resume checkpoint
- **Logging**: TensorBoard directory, experiment name
- **Hardware**: Number of GPUs, precision (16/32 bit)

##### `create_data_loaders(args)`
**Purpose**: Creates training and validation data loaders.

**Process**:
1. Create `VideoDataset` instances for train/val
2. Create `DataLoader` instances with:
   - Batch size from args
   - Shuffling (train only)
   - Number of workers
   - Pin memory (if CUDA available)
   - Drop last batch (train only)

**Returns**: `(train_loader, val_loader)`

##### `create_model(args)`
**Purpose**: Creates VideoMAE model instance.

**Process**:
1. Initialize VideoMAE with specified backbone
2. Load pretrained weights if provided
3. Return model

##### `create_lightning_module(args, model)`
**Purpose**: Creates PyTorch Lightning module.

**Process**:
1. Wrap model in `VideoMAELightningModule`
2. Configure hyperparameters
3. Return lightning module

##### `create_callbacks(args)`
**Purpose**: Creates training callbacks.

**Callbacks**:
1. **ModelCheckpoint**:
   - Saves top 3 models (by validation loss)
   - Saves last checkpoint
   - Filename format: `{prefix}-{epoch:02d}-{val_loss:.4f}.ckpt`
   - Monitors `val_loss` (minimize)

2. **LearningRateMonitor**:
   - Logs learning rate (for monitoring, even with schedule-free optimizer)

##### `main()`
**Purpose**: Main training function.

**Process**:
1. Parse arguments
2. Set random seed for reproducibility
3. Create data loaders
4. Create model and count parameters
5. Create lightning module
6. Create callbacks and logger
7. Create PyTorch Lightning trainer
8. Start training with `trainer.fit()`

**Trainer Configuration**:
- Maximum epochs
- GPU configuration (single or multi-GPU)
- Precision (16 or 32 bit)
- Callbacks and logger
- Gradient clipping (value: 1.0)
- Validation frequency (twice per epoch)
- Logging frequency (every 10 steps)

**Multi-GPU Support**:
- Uses `ddp` (Distributed Data Parallel) strategy for multi-GPU
- Automatically handles data distribution across GPUs

---

## Configuration

### File: `config/config.yaml`

**Purpose**: YAML configuration file for hyperparameter tuning.

**Structure**:
```yaml
data:           # Data loading parameters
model:          # Model architecture parameters
training:       # Training hyperparameters
checkpointing:  # Checkpoint configuration
logging:        # Logging configuration
hardware:       # Hardware configuration
```

**Note**: Currently, the training script uses command-line arguments. The config file can be used for future enhancements or external hyperparameter tuning tools.

---

## Data Flow

### Training Data Flow:

```
CSV File (video paths)
    ↓
VideoDataset.__getitem__()
    ↓
Load video with OpenCV
    ↓
Sample 32 consecutive frames
    ↓
Temporal downsampling (stride=2) → 16 frames
    ↓
Normalize to [0, 1]
    ↓
Tensor: (3, 16, 224, 224)
    ↓
DataLoader batches: (B, 3, 16, 224, 224)
    ↓
VideoMAE.forward()
    ↓
Patch Embedding → (B, 1568, embed_dim)
    ↓
Random Masking → Visible: (B, 392, embed_dim), Mask: (B, 1568)
    ↓
Encoder → Encoded: (B, 392, embed_dim)
    ↓
Decoder → Predictions: (B, 1568, patch_pixels)
    ↓
Loss Computation (masked patches only)
    ↓
Backpropagation
```

### Forward Pass Details:

1. **Input**: Video tensor `(B, 3, 16, 224, 224)`

2. **Patch Embedding**:
   - 3D convolution extracts patches
   - Output: `(B, 1568, embed_dim)`

3. **Masking**:
   - Random shuffle and keep 25% (392 patches)
   - Mask: `(B, 1568)` with 1s for masked, 0s for visible

4. **Encoder**:
   - Process only visible patches: `(B, 392, embed_dim)`
   - Add class token and positional encoding
   - Pass through transformer blocks
   - Output: `(B, 392, embed_dim)`

5. **Decoder**:
   - Project to decoder dimension
   - Restore full sequence with zeros for masked patches
   - Add class token and positional encoding
   - Pass through decoder transformer blocks
   - Predict pixels: `(B, 1568, patch_pixels)`

6. **Loss**:
   - Compute target patches from original video
   - MSE loss only on masked patches
   - Return scalar loss value

---

## Training Process

### EVEREST Training Method:

The EVEREST (Efficient Video Representation Learning) method is implemented through:

1. **High Masking Ratio**: 75% of patches are masked (higher than image MAE)
2. **Temporal Masking**: Patches are masked across both spatial and temporal dimensions
3. **Reconstruction Target**: Model learns to reconstruct pixel values of masked patches
4. **Asymmetric Architecture**: Heavy encoder (processes visible patches) + light decoder (reconstructs all patches)

### Training Loop:

```
For each epoch:
    For each training batch:
        1. Load video batch
        2. Forward pass (masking + encoding + decoding)
        3. Compute loss (masked patches only)
        4. Backward pass
        5. Optimizer step (AdamWScheduleFree)
        6. Log metrics
    
    For each validation batch:
        1. Load video batch
        2. Forward pass (no gradients)
        3. Compute loss
        4. Log metrics
    
    Save checkpoint (if best or last)
```

### Key Training Features:

1. **Schedule-Free Optimizer**: No learning rate scheduling needed
2. **Gradient Clipping**: Prevents exploding gradients (value: 1.0)
3. **Mixed Precision**: Optional 16-bit training for faster training
4. **Multi-GPU**: Automatic data parallelization across GPUs
5. **Checkpointing**: Saves best models and last checkpoint
6. **TensorBoard Logging**: Real-time monitoring of training metrics

### Monitoring Metrics:

- **train_loss**: Training reconstruction loss (logged every step and epoch)
- **val_loss**: Validation reconstruction loss (logged every epoch)
- **grad_norm**: L2 norm of gradients (optional, logged every step if enabled)
- **learning_rate**: Current learning rate (logged every step)

### Checkpointing:

- **Best Models**: Top 3 models by validation loss
- **Last Checkpoint**: Always saved for resuming training
- **Format**: `{prefix}-{epoch:02d}-{val_loss:.4f}.ckpt`
- **Resume**: Use `--resume_from_checkpoint` to continue training

---

## Key Design Decisions

### 1. **3D Patch Embedding**
- Uses 3D convolution to extract spatio-temporal patches
- Maintains temporal relationships in patches

### 2. **Asymmetric Encoder-Decoder**
- Heavy encoder (12/24 layers) processes visible patches
- Light decoder (4 layers) reconstructs all patches
- Reduces computational cost while maintaining performance

### 3. **High Masking Ratio (75%)**
- Forces model to learn strong temporal and spatial representations
- Higher than image MAE (typically 75% vs 75% for images, but more challenging for video)

### 4. **Normalized Pixel Loss**
- Optional normalization of target patches
- Helps with training stability

### 5. **PyTorch Lightning Framework**
- Simplifies multi-GPU training
- Handles distributed training automatically
- Clean separation of model and training logic

### 6. **AdamWScheduleFree Optimizer**
- Schedule-free learning eliminates need for LR scheduling
- Combines benefits of AdamW with schedule-free approach
- Reduces hyperparameter tuning

---

## Usage Examples

### Basic Training:
```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --batch_size 8 \
    --max_epochs 100
```

### With Pretrained Weights:
```bash
python training/train.py \
    --backbone vit_b \
    --pretrained path/to/weights.pth \
    --batch_size 4
```

### Multi-GPU Training:
```bash
python training/train.py \
    --gpus 4 \
    --batch_size 8 \
    --backbone vit_s
```

### With Gradient Tracking:
```bash
python training/train.py \
    --track_grad_norm \
    --backbone vit_s
```

---

## Summary

This codebase provides a complete implementation of VideoMAE with:

1. **Flexible Data Loading**: Handles variable-length videos with proper sampling
2. **Modular Architecture**: Separate components for easy modification
3. **Multiple Backbones**: Support for ViT-S, ViT-B, and ViT-L
4. **EVEREST Training**: Implements masked autoencoding for video
5. **Production-Ready**: PyTorch Lightning for easy scaling and deployment
6. **Well-Documented**: Comprehensive comments and documentation
7. **Configurable**: Easy hyperparameter tuning via command-line arguments

The implementation follows best practices for deep learning research code, with proper separation of concerns, comprehensive error handling, and extensive documentation.

