# EVEREST VideoMAE Refactoring Explanation

This document explains the refactored codebase for training EVEREST VideoMAE models on custom datasets using PyTorch Lightning.

## Overview

The codebase has been refactored to:
- Use PyTorch Lightning for easy multi-GPU training
- Support custom video datasets from CSV files
- Use YAML configuration for hyperparameters
- Integrate AdamWScheduleFree optimizer
- Support ViT-S/B/L backbone architectures
- Enable pretrained weight loading
- Provide optional features (gradient norm tracking, dataset subset sampling)

## Directory Structure

```
TEEFM/
├── configs/
│   └── train_config.yaml          # YAML configuration file
├── datasets/
│   ├── __init__.py
│   └── custom_video_dataset.py    # Custom dataset class for CSV-based video loading
├── transforms/
│   ├── __init__.py
│   └── custom_transforms.py       # Transform pipeline with custom normalization
├── modeling/
│   ├── __init__.py
│   └── model_factory.py           # Factory function for creating VideoMAE models
├── optimizers/
│   ├── __init__.py
│   └── schedule_free_optimizer.py # AdamWScheduleFree optimizer integration
├── lightning_module.py            # PyTorch Lightning module wrapper
├── train.py                       # Main training script
└── requirements.txt               # Updated dependencies
```

## Key Components

### 1. Configuration System (`configs/train_config.yaml`)

The YAML configuration file contains all hyperparameters and settings:

- **Data Configuration**: Paths to train/val CSV files, normalization values, subset ratio
- **Model Configuration**: Backbone architecture (ViT-S/B/L), pretrained weights path, masking settings
- **Training Configuration**: Batch size, epochs, number of workers, frame sampling parameters
- **Optimizer Configuration**: Learning rate, weight decay, beta parameters
- **Checkpoint Configuration**: Directory, prefix, save frequency
- **Optional Features**: Gradient norm tracking, gradient checkpointing

**Key Feature**: Custom normalization values (mean=[0.117, 0.114, 0.113], std=[0.208, 0.204, 0.203]) are used instead of ImageNet defaults.

### 2. Custom Video Dataset (`datasets/custom_video_dataset.py`)

The `CustomVideoDataset` class:

- **Loads videos from CSV files**: Reads video paths from CSV (one path per line)
- **Uses torchvision.io.read_video**: For loading video files
- **Frame sampling**: Randomly samples 32 consecutive frames from each video
- **Temporal downsampling**: Downsamples with stride 2 to get 16 frames
- **No resizing**: Assumes videos are already 224x224x3
- **Subset sampling**: Optional random subset of dataset based on ratio parameter

**Key Implementation Details**:
- Videos are loaded as [T, H, W, C] and converted to [C, T, H, W]
- If a video has fewer than 32 frames, the last frame is repeated
- Returns normalized video tensor and mask (if applicable)

### 3. Transform Pipeline (`transforms/custom_transforms.py`)

The `DataAugmentationForVideoMAE` class:

- **Custom normalization**: Uses dataset-specific mean/std values
- **Masking generation**: Supports three masking strategies:
  - `random`: Random patch masking
  - `tube`: Tube masking (same patches across frames)
  - `motion-centric`: Motion-centric masking (handled by model)
- **Integration**: Works with existing masking generators from the original codebase

**Key Feature**: Normalization values are configurable via the YAML file, allowing easy adaptation to different datasets.

### 4. Model Factory (`modeling/model_factory.py`)

The `create_videomae_model` function:

- **Backbone selection**: Maps 'vit-s', 'vit-b', 'vit-l' to model registrations:
  - ViT-S → `pretrain_videoms_small_patch16_224`
  - ViT-B → `pretrain_videoms_base_patch16_224`
  - ViT-L → `pretrain_videoms_large_patch16_224`
- **Pretrained weight loading**: Handles multiple checkpoint formats:
  - Full checkpoint with 'model' key
  - State dict with 'state_dict' key
  - Direct state dict
- **Flexible loading**: Uses `strict=False` to handle partial weight loading

**Key Feature**: Automatically handles different checkpoint formats and provides informative warnings for missing/unexpected keys.

### 5. Optimizer Integration (`optimizers/schedule_free_optimizer.py`)

The `create_schedule_free_optimizer` function:

- **AdamWScheduleFree**: Wraps the schedule-free optimizer
- **No learning rate scheduling**: The optimizer eliminates the need for LR schedules
- **Error handling**: Provides clear error messages if package is not installed

**Key Feature**: Simplifies training by removing the need for learning rate scheduling, as the optimizer handles this internally.

### 6. PyTorch Lightning Module (`lightning_module.py`)

The `VideoMAELightningModule` class:

- **Wraps VideoMAE model**: Provides Lightning interface for the model
- **Training step**: Implements forward pass, loss computation, and logging
- **Validation step**: Similar to training but without gradient computation
- **Gradient norm tracking**: Optional L2 norm tracking of total loss gradient
- **Automatic logging**: Logs train/val loss and gradient norm (if enabled)

**Key Implementation Details**:
- Handles both motion-centric and non-motion-centric masking
- Computes targets by unnormalizing videos and converting to patches
- Supports normalized and non-normalized target patches
- Uses `on_after_backward` hook for gradient norm tracking

**Loss Computation**:
1. Unnormalize input videos to get original pixel values
2. Convert to patches using einops rearrange
3. Optionally normalize patches per patch (spatial normalization)
4. Extract labels based on mask type
5. Compute MSE loss between predictions and labels

### 7. Main Training Script (`train.py`)

The main script:

- **Configuration loading**: Loads YAML config file
- **Dataset creation**: Creates train and validation datasets
- **Data loader setup**: Configures DataLoaders with proper settings
- **Model initialization**: Creates Lightning module with config
- **Checkpoint callback**: Sets up ModelCheckpoint with user-defined directory/prefix
- **TensorBoard logging**: Configures logging for monitoring
- **Trainer setup**: Configures PyTorch Lightning Trainer with:
  - Multi-GPU support (automatic via DDP)
  - Mixed precision training (16-bit on GPU)
  - Gradient clipping (if specified)
  - Checkpoint resuming

**Key Features**:
- Automatic multi-GPU detection and setup
- Mixed precision training for faster training and lower memory usage
- Comprehensive logging and checkpointing

## Training Workflow

1. **Load Configuration**: Read YAML config file
2. **Create Datasets**: Initialize train/val datasets with transforms
3. **Create Data Loaders**: Set up DataLoaders with batch size, workers, etc.
4. **Initialize Model**: Create VideoMAE model with specified backbone
5. **Setup Trainer**: Configure PyTorch Lightning Trainer
6. **Train**: Call `trainer.fit()` to start training

## Frame Sampling Process

1. Load video using `torchvision.io.read_video` → [T, H, W, C]
2. Randomly select start frame from [0, total_frames - 32]
3. Extract 32 consecutive frames: `frames[start:start+32]`
4. Temporal downsampling with stride 2: `frames[::2]` → 16 frames
5. Convert to [C, T, H, W] format: `frames.permute(3, 0, 1, 2)`
6. Normalize to [0, 1] if needed: `frames / 255.0`
7. Apply transforms (normalization, masking)

## Masking Strategies

### Random Masking
- Randomly masks patches across the entire video
- Uses `RandomMaskingGenerator` from original codebase

### Tube Masking
- Masks the same patches across all frames (temporal consistency)
- Uses `TubeMaskingGenerator` from original codebase

### Motion-Centric Masking (EVEREST)
- Identifies informative patches based on motion
- Handled internally by the model's encoder
- More efficient than random masking

## Usage Example

```bash
# Train with default config
python train.py --config configs/train_config.yaml

# Resume from checkpoint
python train.py --config configs/train_config.yaml --resume output/checkpoints/videomae-epoch=50-val_loss=0.1234.ckpt
```

## Configuration Options

### Data Options
- `train_csv`: Path to training CSV file
- `val_csv`: Path to validation CSV file
- `subset_ratio`: Optional ratio for random subset sampling (null to use full dataset)
- `normalize_mean/std`: Custom normalization values

### Model Options
- `backbone`: 'vit-s', 'vit-b', or 'vit-l'
- `pretrained_path`: Path to pretrained checkpoint (null for random init)
- `mask_type`: 'random', 'tube', or 'motion-centric'
- `mask_ratio`: Fraction of patches to mask
- `decoder_depth`: Depth of decoder

### Training Options
- `batch_size`: Batch size per GPU
- `max_epochs`: Number of training epochs
- `num_workers`: Data loading workers
- `frames_to_sample`: Number of frames to sample (32)
- `temporal_stride`: Stride for downsampling (2)

### Feature Flags
- `track_grad_norm`: Enable gradient norm tracking (default: false)
- `use_checkpoint`: Enable gradient checkpointing for memory savings (default: false)

## Dependencies

New dependencies added:
- `pytorch-lightning>=2.0.0`: For Lightning framework
- `pyyaml`: For YAML configuration parsing
- `schedule-free`: For AdamWScheduleFree optimizer
- `torchvision`: For video loading (already used, but explicitly listed)

## Key Improvements

1. **Multi-GPU Training**: Automatic via PyTorch Lightning DDP
2. **Configuration Management**: All hyperparameters in YAML file
3. **Custom Normalization**: Dataset-specific normalization values
4. **Flexible Backbone**: Easy switching between ViT-S/B/L
5. **Pretrained Weights**: Generic mechanism for loading checkpoints
6. **Optional Features**: Gradient tracking and subset sampling can be toggled
7. **Better Logging**: TensorBoard integration with automatic logging
8. **Checkpoint Management**: User-defined directory and prefix

## Notes

- Videos must be preprocessed to 224x224x3 before training
- The dataset is unlabeled (self-supervised learning)
- Motion-centric masking is the default (EVEREST method)
- AdamWScheduleFree eliminates the need for learning rate scheduling
- Mixed precision training is enabled by default on GPU for faster training

