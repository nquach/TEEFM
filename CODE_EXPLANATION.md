# Code Explanation

This document provides a comprehensive explanation of all components in the VideoMAE training codebase.

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Data Loading](#data-loading)
3. [Model Architecture](#model-architecture)
4. [Training Infrastructure](#training-infrastructure)
5. [Main Training Script](#main-training-script)

---

## Configuration System

### `config.py`

The configuration system uses Python dataclasses to organize all hyperparameters into logical groups:

- **ModelConfig**: Model architecture parameters (backbone type, patch size, masking ratio)
- **DataConfig**: Data loading parameters (CSV paths, batch size, frame sampling)
- **TrainingConfig**: Training hyperparameters (learning rate, epochs, loss type)
- **CheckpointConfig**: Checkpoint saving strategy (directory, prefix, top-k)
- **TrainerConfig**: PyTorch Lightning trainer settings (GPUs, logging, precision)

**Key Features:**
- All parameters have sensible defaults
- Easy to modify via command-line arguments or direct code changes
- Type hints for better IDE support and error checking

---

## Data Loading

### `data/video_dataset.py`

The `VideoDataset` class implements efficient video loading using torchcodec:

**Key Components:**

1. **CSV Reading**: Reads video file paths from CSV files (one path per line)
2. **Video Decoding**: Uses `torchcodec.VideoDecoder` for hardware-accelerated decoding
3. **Frame Sampling**: 
   - Randomly samples 32 consecutive frames from each video
   - Handles videos shorter than 32 frames by padding
4. **Temporal Downsampling**: Applies stride of 2 to reduce 32 frames → 16 frames
5. **Error Handling**: Returns zero tensors for corrupted videos to allow training to continue

**Data Flow:**
```
MP4 Video → VideoDecoder → 32 Frames → Temporal Stride 2 → 16 Frames → (B, T, H, W, C)
```

**Important Notes:**
- Videos are assumed to be preprocessed to 224x224x3
- No resizing is performed (as per requirements)
- Frames are normalized to [0, 1] range if needed

---

## Model Architecture

### `models/vit_backbone.py`

Implements the Vision Transformer encoder backbone:

**Components:**

1. **PatchEmbedding**: 
   - Converts video to 3D patches (tubelets)
   - Uses 3D convolution: `(tubelet_size, patch_size, patch_size)`
   - Projects to embedding dimension

2. **MultiHeadAttention**:
   - Standard self-attention mechanism
   - Supports optional attention masking
   - Scaled dot-product attention

3. **TransformerBlock**:
   - Self-attention + MLP with residual connections
   - Layer normalization before each sub-layer
   - GELU activation in MLP

4. **ViTBackbone**:
   - Supports ViT-S (384 dim, 12 layers, 6 heads)
   - Supports ViT-B (768 dim, 12 layers, 12 heads)
   - Supports ViT-L (1024 dim, 24 layers, 16 heads)
   - Learnable position embeddings
   - Class token (used for reconstruction in VideoMAE)

**Architecture Details:**
- Input: `(B, T, H, W, C)` video tensor
- Output: `(B, N+1, embed_dim)` tokens (N patch tokens + 1 class token)
- Number of patches: `(H/patch_size)² × (T/tubelet_size)`

### `models/everest_masking.py`

Implements the EVEREST masking strategy:

**Key Features:**

1. **High Masking Ratio**: Default 90% (configurable)
2. **Random Masking**: Randomly selects tokens to mask
3. **Temporal Consistency**: Can be extended to mask entire temporal tubes
4. **Mask Generation**: Creates boolean mask and restoration indices

**Masking Process:**
1. Generate random permutation of all tokens
2. Keep first `(1 - mask_ratio) × N` tokens visible
3. Mask remaining tokens
4. Create restoration indices to restore original order

**Output:**
- `mask`: Boolean tensor `(B, N)` where `True` = masked
- `ids_restore`: Indices to restore original token order

### `models/videomae.py`

Implements the complete VideoMAE model:

**Architecture:**

1. **Encoder (ViT Backbone)**:
   - Processes all video patches
   - Outputs encoded representations

2. **Masking (EVEREST)**:
   - Generates high-ratio mask
   - Separates visible and masked tokens

3. **Decoder (Lightweight Transformer)**:
   - Takes visible tokens + mask tokens
   - Reconstructs all patches
   - Much smaller than encoder (8 layers vs 12/24)

**Forward Pass:**

```
Input Video (B, T, H, W, C)
    ↓
Patch Embedding → (B, N, embed_dim)
    ↓
Encoder (all tokens) → (B, N, embed_dim)
    ↓
Extract Visible Tokens → (B, N_visible, embed_dim)
    ↓
Decoder (visible + mask tokens) → (B, N, embed_dim)
    ↓
Reconstruction Loss (only on masked tokens)
```

**Loss Computation:**
- MSE loss between predicted and target patches
- Only computed on masked tokens (90% of tokens)
- Target patches are normalized (mean=0, std=1) if `norm_pix_loss=True`

**Pretrained Weights:**
- Supports loading pretrained weights from checkpoint files
- Handles different checkpoint formats (state_dict, model, raw dict)
- Removes 'module.' prefix for DataParallel compatibility

---

## Training Infrastructure

### `training/lightning_module.py`

PyTorch Lightning wrapper for VideoMAE training:

**Key Components:**

1. **VideoMAELightningModule**:
   - Wraps VideoMAE model
   - Implements training and validation steps
   - Configures optimizer (AdamWScheduleFree)
   - Handles gradient clipping

2. **Training Step**:
   - Calls `model.forward_loss()` to get loss
   - Logs training loss
   - Optionally tracks gradient L2 norm

3. **Validation Step**:
   - Same as training but without gradient computation
   - Logs validation loss

4. **Optimizer Configuration**:
   - Uses `AdamWScheduleFree` from schedulefree package
   - No learning rate scheduling needed (schedule-free optimizer)
   - Configurable learning rate and weight decay

**Gradient Norm Tracking:**
- Optional feature (off by default)
- Computes L2 norm of all gradients
- Logged to TensorBoard for monitoring
- Useful for debugging training stability

**Gradient Clipping:**
- Configurable gradient clipping value
- Applied before optimizer step
- Helps prevent gradient explosion

---

## Main Training Script

### `train.py`

Main entry point for training:

**Functionality:**

1. **Argument Parsing**:
   - Command-line interface for all hyperparameters
   - Overrides default configuration values

2. **Data Loader Creation**:
   - Creates training and validation datasets
   - Sets up DataLoaders with proper batching and workers

3. **Model Creation**:
   - Instantiates VideoMAE model
   - Loads pretrained weights if provided
   - Prints model parameter count

4. **Lightning Module Creation**:
   - Wraps model in Lightning module
   - Configures optimizer and loss

5. **Trainer Setup**:
   - Creates PyTorch Lightning Trainer
   - Sets up checkpoint callbacks
   - Configures logging (TensorBoard)
   - Handles multi-GPU training

6. **Training Execution**:
   - Calls `trainer.fit()` to start training
   - Automatically handles:
     - Checkpoint saving
     - Validation
     - Logging
     - Multi-GPU synchronization

**Checkpointing:**
- Saves top-k checkpoints based on validation loss
- Saves last checkpoint
- Customizable directory and prefix
- Format: `{prefix}-{epoch:02d}-{val_loss:.4f}.ckpt`

**Multi-GPU Support:**
- Automatic data parallelism via PyTorch Lightning
- Specify number of GPUs with `--gpus N`
- Handles gradient synchronization automatically

---

## Training Workflow

### Complete Training Pipeline

1. **Data Loading**:
   ```
   CSV File → VideoDataset → DataLoader → Batched Videos
   ```

2. **Forward Pass**:
   ```
   Video Batch → Patch Embedding → Encoder → Masking → Decoder → Loss
   ```

3. **Backward Pass**:
   ```
   Loss → Backward → Gradient Clipping → Optimizer Step
   ```

4. **Checkpointing**:
   ```
   Validation Loss → Checkpoint Callback → Save Top-K Models
   ```

### Key Design Decisions

1. **Why encode all tokens then extract visible ones?**
   - More efficient than masking before encoding
   - Allows encoder to see full context
   - Matches original VideoMAE implementation

2. **Why use EVEREST masking?**
   - High masking ratio (90%) forces strong representation learning
   - Temporal consistency improves video understanding
   - Proven effective for self-supervised learning

3. **Why AdamWScheduleFree?**
   - Eliminates need for learning rate scheduling
   - More stable training
   - Simpler hyperparameter tuning

4. **Why PyTorch Lightning?**
   - Easy multi-GPU training
   - Built-in checkpointing and logging
   - Cleaner code organization
   - Production-ready features

---

## Hyperparameter Tuning Guide

### Important Hyperparameters

1. **Learning Rate**: Start with 1e-4, adjust based on loss curve
2. **Batch Size**: Larger batches = more stable training, but requires more memory
3. **Mask Ratio**: 0.9 (90%) is standard for EVEREST
4. **Gradient Clipping**: 1.0 is a good default
5. **Weight Decay**: 0.05 works well with AdamWScheduleFree

### Backbone Selection

- **ViT-S**: Fastest, good for initial experiments
- **ViT-B**: Balanced performance and speed
- **ViT-L**: Best performance, requires more memory and time

### Monitoring Training

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should track training loss
3. **Gradient Norm** (if enabled): Should be stable, not exploding
4. **Learning Rate**: Constant with AdamWScheduleFree (no scheduling)

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Use smaller backbone (ViT-S instead of ViT-L)

2. **Loss Not Decreasing**:
   - Check learning rate (might be too high/low)
   - Verify data loading (check if videos are loading correctly)
   - Check gradient norm (should not be zero)

3. **Slow Training**:
   - Increase number of workers
   - Use mixed precision (FP16)
   - Use multiple GPUs

4. **Checkpoint Loading Errors**:
   - Verify checkpoint format
   - Check if model architecture matches
   - Ensure all required keys are present

---

## Extension Points

The codebase is designed to be easily extensible:

1. **Custom Masking Strategies**: Modify `EverestMasking` class
2. **Different Loss Functions**: Modify `VideoMAELightningModule.compute_loss()`
3. **Additional Metrics**: Add to training/validation steps
4. **Custom Augmentations**: Add transforms to `VideoDataset`
5. **Different Optimizers**: Modify `configure_optimizers()` in Lightning module

---

## Summary

This codebase provides a complete, production-ready implementation of VideoMAE training with EVEREST masking. It follows best practices for:

- **Modularity**: Clear separation of concerns
- **Configurability**: Easy hyperparameter tuning
- **Scalability**: Multi-GPU support out of the box
- **Maintainability**: Well-commented, type-hinted code
- **Extensibility**: Easy to modify and extend

The implementation is ready for both research experiments and production training pipelines.

