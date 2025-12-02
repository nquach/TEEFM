# VideoMAE Training with EVEREST Method

This codebase implements and trains a Video Vision Transformer (VideoMAE) model using the EVEREST (Masked Video Autoencoder) training method. The implementation supports ViT-S, ViT-B, and ViT-L backbones and uses PyTorch Lightning for easy multi-GPU training.

## Project Structure

```
TEEFM/
├── config/
│   └── config.yaml          # Configuration file for hyperparameters
├── data/
│   ├── __init__.py
│   └── video_dataset.py     # Video dataset implementation
├── models/
│   ├── __init__.py
│   └── videomae.py          # VideoMAE model implementation
├── training/
│   ├── __init__.py
│   ├── lightning_module.py  # PyTorch Lightning module
│   └── train.py             # Main training script
├── mp4_paths.csv            # Training video paths
├── val500_2023-2024.csv     # Validation video paths
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CUDA-enabled PyTorch if using GPU training.

## Quick Start

1. Configure training parameters in `config/config.yaml`
2. Run training:
```bash
python training/train.py --config config/config.yaml
```

3. Resume from checkpoint:
```bash
python training/train.py --config config/config.yaml --resume checkpoints/videomae-epoch=10-val_loss=0.1234.ckpt
```

## Code Explanation

### 1. Configuration System (`config/config.yaml`)

The configuration file uses YAML format and organizes all hyperparameters into logical sections:

- **Model configuration**: Backbone type (vit_s/vit_b/vit_l), pretrained weights
- **Data configuration**: CSV file paths, frame sampling parameters, subset ratio
- **Training configuration**: Batch size, learning rate, number of epochs
- **EVEREST configuration**: Mask ratio, patch size, tubelet size
- **Checkpoint configuration**: Save directory, file prefix, monitoring metric
- **Logging configuration**: Gradient norm tracking, log directory
- **Hardware configuration**: GPU settings, mixed precision training

### 2. Data Loading (`data/video_dataset.py`)

The `VideoDataset` class handles video loading and preprocessing:

**Key Features:**
- Reads video paths from CSV files (assumes first column or 'path' column)
- Randomly samples 32 consecutive frames from each video
- Applies temporal downsampling with stride 2 to get 16 frames
- Handles videos with insufficient frames by padding
- Supports random subset selection via `subset_ratio` parameter

**Frame Processing Pipeline:**
1. Load video using `torchvision.io.read_video()` → Returns (T, C, H, W) tensor
2. Check if video has ≥32 frames, pad if necessary
3. Randomly select starting frame for 32 consecutive frames
4. Apply temporal stride of 2 → Results in 16 frames
5. Normalize pixel values from [0, 255] to [0, 1]

**Important Notes:**
- Videos are expected to be 224×224×3 (no resizing performed)
- The dataset filters out non-existent video paths
- Subset ratio feature is off by default (set to `null` in config)

### 3. VideoMAE Model (`models/videomae.py`)

The `VideoMAE` class implements a masked video autoencoder using Vision Transformer architecture.

#### Architecture Components:

**a) PatchEmbedding3D:**
- Converts video frames into spatio-temporal patches
- Uses 3D convolution to extract tubelets (temporal patches)
- Projects patches to embedding dimension
- Input: (B, T, C, H, W) → Output: (B, N, embed_dim) where N = number of patches

**b) PositionalEncoding3D:**
- Adds learnable positional embeddings to patch embeddings
- Helps model understand spatial and temporal relationships

**c) TransformerBlock:**
- Standard transformer block with:
  - Layer normalization
  - Multi-head self-attention
  - Feed-forward MLP with GELU activation
  - Residual connections

**d) Encoder:**
- Stack of transformer blocks
- Processes visible and masked patches
- Outputs encoded representations

**e) Decoder:**
- Lightweight decoder (4 transformer blocks)
- Reconstructs masked patches
- Predicts pixel values for each patch

#### Backbone Configurations:

- **ViT-S**: 384 dim, 12 layers, 6 heads
- **ViT-B**: 768 dim, 12 layers, 12 heads  
- **ViT-L**: 1024 dim, 24 layers, 16 heads

#### Key Methods:

**`random_masking()`:**
- Implements EVEREST masking strategy
- Randomly masks 75% of patches (configurable)
- Returns masked embeddings, binary mask, and restore indices

**`forward_encoder()`:**
- Extracts patches from input video
- Applies random masking
- Processes through transformer encoder
- Returns encoded features, mask, and restore indices

**`forward_decoder()`:**
- Takes encoded features and restore indices
- Reconstructs full sequence with mask tokens
- Predicts pixel values for all patches
- Returns reconstructed patch predictions

**`load_pretrained()`:**
- Loads pretrained weights from checkpoint
- Handles different checkpoint formats
- Supports partial loading (strict=False)

### 4. Training Module (`training/lightning_module.py`)

The `VideoMAELightning` class wraps the VideoMAE model for PyTorch Lightning training.

#### EVEREST Loss Computation:

**`patchify()`:**
- Converts video frames to patches (inverse of unpatchify)
- Used to prepare ground truth for loss computation
- Input: (B, T, C, H, W) → Output: (B, N, patch_pixels)

**`unpatchify()`:**
- Converts patches back to video frames
- Used for visualization (not in training)
- Input: (B, N, patch_pixels) → Output: (B, T, C, H, W)

**`compute_loss()`:**
- Computes MSE loss between predicted and target patches
- Loss is computed **only on masked patches** (where mask == 0)
- Supports normalized pixel loss (normalize by patch mean/std)
- Returns scalar loss value

#### Training/Validation Steps:

**`training_step()`:**
1. Forward pass through model → Get predictions and mask
2. Convert target video to patches
3. Compute reconstruction loss on masked patches
4. Log training loss
5. Return loss for backpropagation

**`validation_step()`:**
- Same as training step but without gradient computation
- Logs validation loss for monitoring

#### Gradient Norm Tracking:

**`on_after_backward()`:**
- Hook called after backward pass
- Computes L2 norm of total loss gradient
- Logs gradient norm if `track_gradient_norm=True`
- Useful for debugging training stability

#### Optimizer Configuration:

**`configure_optimizers()`:**
- Uses `AdamWScheduleFree` if available (from schedule-free package)
- Falls back to standard `AdamW` if package not installed
- Configurable learning rate and weight decay

### 5. Training Script (`training/train.py`)

The main training script orchestrates the entire training process.

#### Key Functions:

**`load_config()`:**
- Loads YAML configuration file
- Returns configuration dictionary

**`create_datasets()`:**
- Creates training and validation datasets
- Applies subset ratio to training set if specified
- Validation set always uses full dataset

**`create_model()`:**
- Instantiates VideoMAE model with configuration
- Handles pretrained weight loading
- Returns model instance

**`create_lightning_module()`:**
- Wraps model in PyTorch Lightning module
- Configures loss, optimizer, and logging settings

**`main()`:**
- Sets up data loaders with proper batch size and workers
- Creates checkpoint callback for model saving
- Sets up TensorBoard logger
- Configures PyTorch Lightning trainer with:
  - Multi-GPU support
  - Mixed precision training (16-bit)
  - Gradient clipping
  - Checkpoint saving
- Runs training loop

#### Checkpoint Management:

- Saves top-k models based on validation loss
- Saves last checkpoint every epoch
- Checkpoint filename format: `{prefix}-{epoch:02d}-{val_loss:.4f}.ckpt`
- Supports resuming from checkpoint via `--resume` argument

## Configuration Options

### Model Options:
- `backbone`: "vit_s", "vit_b", or "vit_l"
- `pretrained`: Load pretrained weights (boolean)
- `pretrained_path`: Path to pretrained checkpoint

### Data Options:
- `subset_ratio`: Use random subset of training data (0.0-1.0, null = use all)
- `num_frames`: Number of frames after downsampling (default: 16)
- `sample_frames`: Frames to sample before downsampling (default: 32)
- `temporal_stride`: Stride for temporal downsampling (default: 2)

### Training Options:
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimizer
- `max_epochs`: Number of training epochs
- `weight_decay`: Weight decay for optimizer

### EVEREST Options:
- `mask_ratio`: Ratio of patches to mask (default: 0.75)
- `patch_size`: Spatial patch size (default: 16)
- `tubelet_size`: Temporal tubelet size (default: 2)

### Logging Options:
- `track_gradient_norm`: Track L2 norm of gradients (default: false)

### Checkpoint Options:
- `dir`: Directory to save checkpoints
- `prefix`: Prefix for checkpoint filenames
- `save_top_k`: Number of best models to keep

## Features

### ✅ Implemented Features:

1. **Multiple Backbone Support**: ViT-S, ViT-B, ViT-L
2. **Pretrained Weight Loading**: Initialize from pretrained checkpoints
3. **EVEREST Training**: Masked video autoencoder with 75% masking
4. **AdamWScheduleFree Optimizer**: Schedule-free optimizer for better convergence
5. **PyTorch Lightning**: Easy multi-GPU training
6. **Gradient Norm Tracking**: Optional L2 norm tracking (off by default)
7. **Dataset Subsampling**: Random subset selection (off by default)
8. **Flexible Checkpointing**: User-defined directory and prefix
9. **Mixed Precision Training**: 16-bit training for faster training
10. **Comprehensive Logging**: TensorBoard integration

### 📝 Usage Examples:

**Train with ViT-S backbone:**
```yaml
# config/config.yaml
model:
  backbone: "vit_s"
  pretrained: false
```

**Train with subset of data (10%):**
```yaml
data:
  subset_ratio: 0.1
```

**Enable gradient norm tracking:**
```yaml
logging:
  track_gradient_norm: true
```

**Use pretrained weights:**
```yaml
model:
  pretrained: true
  pretrained_path: "path/to/checkpoint.ckpt"
```

**Multi-GPU training:**
```yaml
hardware:
  gpus: -1  # Use all available GPUs
```

## Training Process

1. **Data Loading**: Videos are loaded, 32 consecutive frames are sampled, then downsampled to 16 frames
2. **Patch Extraction**: Frames are divided into spatio-temporal patches
3. **Masking**: 75% of patches are randomly masked
4. **Encoding**: Visible patches are encoded through transformer encoder
5. **Decoding**: Encoded features + mask tokens are decoded to reconstruct all patches
6. **Loss Computation**: MSE loss computed only on masked patches
7. **Backpropagation**: Gradients computed and model updated

## Monitoring Training

Training progress can be monitored via TensorBoard:
```bash
tensorboard --logdir logs/
```

Metrics logged:
- `train_loss`: Training reconstruction loss
- `val_loss`: Validation reconstruction loss
- `grad_norm`: L2 norm of gradients (if enabled)

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config
- Reduce `num_workers` in config
- Use gradient accumulation (add to trainer config)

**Slow Training:**
- Increase `num_workers` for data loading
- Enable mixed precision (set `precision: 16`)
- Use multiple GPUs

**Poor Convergence:**
- Adjust learning rate
- Check gradient norms (enable `track_gradient_norm`)
- Verify data loading (check video paths in CSV)

## Citation

If you use this codebase, please cite the original VideoMAE and EVEREST papers.

