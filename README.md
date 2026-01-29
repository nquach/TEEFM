# EVEREST: Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens [ICML2024]

This repository is an official Pytorch implementation of [EVEREST: Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens](https://arxiv.org/abs/2211.10636).

**The new version of EVEREST will be updated soon!!!** 🚨

<p align="center">
  <img align="middle" width="1000" src="assets/EVEREST_concept.PNG">
</p>

## Table of Contents

- [Abstract](#abstract)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Pretraining with PyTorch Lightning](#pretraining-with-pytorch-lightning)
- [Finetuning](#finetuning)
- [Visualization](#visualization)
- [Configuration Guide](#configuration-guide)
- [Advanced Features](#advanced-features)
- [Examples and Tutorials](#examples-and-tutorials)
- [Troubleshooting](#troubleshooting)
- [Training Logs and Checkpoints](#training-logs-and-checkpoints)
- [Contact](#contact)
- [Acknowledgment](#acknowledgment)
- [Citations](#citations)

## Abstract

Masked Video Autoencoder (MVA) approaches have demonstrated their potential by significantly outperforming previous video representation learning methods. However, they waste an excessive amount of computations and memory in predicting uninformative tokens/frames due to random masking strategies. (e.g., over 16 nodes with 128 NVIDIA A100 GPUs). To resolve this issue, we exploit the unequal information density among the patches in videos and propose EVEREST, a surprisingly efficient MVA approach for video representation learning that finds tokens containing rich motion features and discards uninformative ones during both pre-training and fine-tuning. We further present an information-intensive frame selection strategy that allows the model to focus on informative and causal frames with minimal redundancy. Our method significantly reduces the computation and memory requirements of MVA, enabling the pre-training and fine-tuning on a single machine with 8 GPUs while achieving comparable performance to computation- and memory-heavy baselines on multiple benchmarks and the uncurated Ego4D dataset. We hope that our work contributes to reducing the barrier to further research on video understanding.

## Results

<p align="center">
  <img align="middle" width="750" src="assets/EVEREST_plot.PNG">
</p>

## Installation and Setup

### Prerequisites

- **Python**: 3.7.12 or higher
- **PyTorch**: 1.8.0 or higher (with CUDA support for GPU training)
- **PyTorch Lightning**: >= 2.0.0
- **CUDA**: Compatible version for GPU training (recommended)
- **GPU**: NVIDIA GPU with sufficient VRAM (8GB+ recommended for base models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TEEFM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pytorch-lightning>=2.0.0` - Multi-GPU training framework
- `litdata` - Optimized dataset streaming
- `einops` - Tensor operations
- `timm` - Vision transformer models
- `torchmetrics` - Evaluation metrics
- `tensorboardX` - Logging and visualization

### Verify Installation

Test that PyTorch and CUDA are working:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

Here's a quick workflow to get started:

1. **Prepare your dataset** (see [Dataset Preparation](#dataset-preparation))
2. **Optimize dataset** with litdata for faster loading
3. **Configure training** by editing YAML config files
4. **Run pretraining**: `python train.py --config configs/base_test.yaml`
5. **Run finetuning**: `python finetune.py --config configs/finetune_example.yaml`
6. **Visualize results**: `python visualize_reconstruction.py --config ... --checkpoint ...`

## Dataset Preparation

### Overview

EVEREST uses optimized datasets created with `litdata` for efficient streaming and caching. Videos are preprocessed and stored in an optimized format that enables fast data loading during training.

### Creating Optimized Datasets

1. **Prepare your video files**: Organize videos in a directory structure
2. **Create annotation files** (for labeled datasets):
   - Classification: CSV format with `path/to/video label`
   - Regression: CSV format with `path/to/video continuous_value`
3. **Optimize dataset** using `litdata.optimize()`:

```python
from litdata import optimize
from pathlib import Path

# Define your data processing function
def process_video(item):
    # Load and process video
    video_path, label = item
    video = load_video(video_path)  # Your video loading logic
    return {"video": video, "label": label}

# Optimize dataset
optimize(
    fn=process_video,
    inputs=list_of_video_paths_and_labels,
    output_dir="path/to/optimized/dataset",
    num_workers=os.cpu_count()
)
```

See `optimize/optimize_dataset.py` for a complete example.

### Dataset Format Requirements

- **Videos**: Common formats (MP4, AVI, etc.) supported by `decord` or `av`
- **Resolution**: Will be resized to model input size (default 224x224)
- **Frames**: Videos should have sufficient frames (minimum 16+ recommended)
- **Labels**: 
  - Classification: Integer class indices (0 to num_classes-1)
  - Regression: Float values for continuous variables

### Configuration for Local vs. Cloud Datasets

**Local datasets:**
```yaml
data:
  train_optimized_dir: "/path/to/local/optimized/dataset"
  cache_dir: "./cache"
  cloud_type: null  # or omit
```

**S3/Cloud datasets:**
```yaml
data:
  train_optimized_dir: "s3://bucket-name/path/to/dataset"
  cache_dir: "./cache"
  cloud_type: 's3_public'
  max_cache_size: '50GB'
```

## Pretraining with PyTorch Lightning

### Overview

Pretraining learns video representations by reconstructing masked video patches. The model learns to predict masked patches from visible ones, learning rich spatiotemporal features.

### Configuration

Create or edit a YAML configuration file (see `configs/base_test.yaml` for an example):

```yaml
data:
  train_optimized_dir: "path/to/optimized/dataset"
  cache_dir: "./cache"
  normalize_mean: [0.117, 0.114, 0.113]
  normalize_std: [0.208, 0.204, 0.203]

model:
  backbone: "vit-s"  # or "vit-b", "vit-l", "vit-h"
  mask_type: "motion-centric"  # or "random", "tube"
  mask_ratio: 0.9
  motion_centric_masking_ratio: 0.7

training:
  batch_size: 16
  num_frames: 16
  max_epochs: 100
  num_workers: 4  # CPU workers per GPU for data loading
```

### Running Pretraining

**Basic usage:**
```bash
python train.py --config configs/base_test.yaml
```

**Resume from checkpoint:**
```bash
python train.py --config configs/base_test.yaml --resume checkpoints/checkpoint.ckpt
```

### Key Parameters Explained

- **`backbone`**: Model architecture size (vit-s, vit-b, vit-l, vit-h)
- **`mask_type`**: Masking strategy
  - `"motion-centric"`: EVEREST's motion-centric masking (recommended)
  - `"random"`: Random patch masking
  - `"tube"`: Temporal tube masking
- **`mask_ratio`**: Fraction of patches to mask (0.9 = 90% masked)
- **`num_workers`**: CPU worker processes per GPU for data loading (4-16 recommended)
- **`batch_size`**: Batch size per GPU (adjust based on GPU memory)

### Multi-GPU Training

PyTorch Lightning automatically detects and uses all available GPUs. For explicit control:

```yaml
training:
  strategy: "ddp"  # Distributed Data Parallel
  # or "deepspeed" for DeepSpeed ZeRO optimization
```

### Monitoring Training

Training logs are saved to TensorBoard. View logs:
```bash
tensorboard --logdir output/logs
```

Key metrics logged:
- `train_loss`: Reconstruction loss
- `train_loss_epoch`: Epoch-averaged loss
- `grad_norm`: Gradient norm (if enabled)

### Resuming from Checkpoint

To resume training from a checkpoint:
```bash
python train.py --config configs/base_test.yaml --resume checkpoints/videomae-epoch=50.ckpt
```

## Finetuning

### Overview

Finetuning adapts the pretrained model for downstream tasks: classification (discrete classes) or regression (continuous variables).

### Classification Finetuning

**Configuration:**
```yaml
model:
  task_type: "classification"  # or omit (default)
  num_classes: 101  # Number of classes
  pretrained_path: "path/to/pretrained/checkpoint.pth"

data:
  train_optimized_dir: "path/to/train/optimized/dataset"
  val_optimized_dir: "path/to/val/optimized/dataset"  # Optional

checkpoint:
  monitor: "val_acc1"  # Monitor validation top-1 accuracy
```

**Running:**
```bash
python finetune.py --config configs/finetune_example.yaml
```

**Metrics:**
- `val_acc1`: Top-1 accuracy
- `val_acc5`: Top-5 accuracy
- `train_acc1`, `train_acc5`: Training accuracies

### Regression Finetuning

**Configuration:**
```yaml
model:
  task_type: "regression"
  output_dim: 1  # Output dimension (1 for single-value regression)
  regression_loss: "mse"  # or "l1"
  pretrained_path: "path/to/pretrained/checkpoint.pth"

data:
  train_optimized_dir: "path/to/train/optimized/dataset"
  # Labels should be float values for regression

checkpoint:
  monitor: "val_mse"  # Monitor validation MSE (minimize)
```

**Running:**
```bash
python finetune.py --config configs/finetune_example.yaml
```

**Metrics:**
- `val_mse`: Mean Squared Error
- `val_mae`: Mean Absolute Error
- `val_r2`: R² score
- Training equivalents: `train_mse`, `train_mae`, `train_r2`

### Key Differences: Classification vs. Regression

| Feature | Classification | Regression |
|---------|---------------|------------|
| Task Type | `task_type: "classification"` | `task_type: "regression"` |
| Labels | Integer class indices | Float continuous values |
| Loss Function | CrossEntropyLoss | MSELoss or L1Loss |
| Metrics | Accuracy (top-1, top-5) | MSE, MAE, R² |
| Model Head | `num_classes` outputs | `output_dim` outputs (default: 1) |
| Checkpoint Monitor | `val_acc1` (maximize) | `val_mse` (minimize) |
| Mixup/Cutmix | Supported | Disabled |

### Resuming Finetuning

```bash
python finetune.py --config configs/finetune_example.yaml --resume checkpoints/finetune-epoch=50.ckpt
```

## Visualization

### Overview

The visualization script generates side-by-side comparisons showing original videos, masked videos, and reconstructed videos to assess reconstruction quality.

### Running Visualization

**Basic usage:**
```bash
python visualize_reconstruction.py \
    --config configs/base_test.yaml \
    --checkpoint checkpoints/pretrain/checkpoint.pth \
    --output_dir visualizations/ \
    --num_samples 10 \
    --frames_to_show 8
```

**Arguments:**
- `--config`: Path to YAML config (must include dataset config)
- `--checkpoint`: Path to pretrained checkpoint
- `--output_dir`: Directory to save visualizations (default: `visualizations/`)
- `--num_samples`: Number of random videos to visualize (default: 5)
- `--frames_to_show`: Number of frames per video to show (default: 8)
- `--seed`: Random seed for reproducible sampling (optional)

### Understanding Output

Visualizations are saved as PNG image grids with:
- **Column 1**: Original video frames
- **Column 2**: Masked video frames (gray regions show masked patches)
- **Column 3**: Reconstructed video frames

Filename format: `video_{index}_reconstruction_grid.png`

### Configuration Requirements

The config file must include:
- Dataset configuration (`data.train_optimized_dir`, etc.)
- Model configuration (backbone, mask_type, etc.)
- Training configuration (num_frames, etc.)

See `configs/visualize_example.yaml` for a complete example.

## Configuration Guide

### Configuration File Structure

Configuration files use YAML format with the following main sections:

```yaml
data:          # Dataset configuration
model:         # Model architecture and parameters
training:      # Training hyperparameters
optimizer:     # Optimizer configuration
augmentation:  # Data augmentation (finetuning)
features:      # Advanced features
checkpoint:    # Checkpoint management
logging:       # Logging configuration
```

### Data Configuration

```yaml
data:
  # Optimized dataset directory (required)
  train_optimized_dir: "path/to/optimized/dataset"
  val_optimized_dir: "path/to/val/dataset"  # Optional
  test_optimized_dir: "path/to/test/dataset"  # Optional
  
  # Cache directory for litdata
  cache_dir: "./cache"
  max_cache_size: '50GB'
  
  # Subset ratio (0 < ratio <= 1, or null for full dataset)
  subset_ratio: null
  
  # Normalization values [R, G, B]
  normalize_mean: [0.117, 0.114, 0.113]
  normalize_std: [0.208, 0.204, 0.203]
  
  # Cloud storage (null for local, 's3_public' for S3)
  cloud_type: null
```

### Model Configuration

**Pretraining:**
```yaml
model:
  backbone: "vit-s"  # "vit-s", "vit-b", "vit-l", "vit-h"
  decoder_depth: 4
  mask_type: "motion-centric"  # "random", "tube", "motion-centric"
  mask_ratio: 0.9
  motion_centric_masking_ratio: 0.7
  normalize_target: true
  input_size: 224
  patch_size: 16
  tubelet_size: 2
```

**Finetuning:**
```yaml
model:
  backbone: "vit-b"
  pretrained_path: "path/to/pretrained/checkpoint.pth"
  task_type: "classification"  # or "regression"
  num_classes: 101  # For classification
  output_dim: 1  # For regression
  regression_loss: "mse"  # For regression: "mse" or "l1"
  use_mean_pooling: true
  mcm: true  # Motion-centric masking for finetuning
  mcm_ratio: 0.4
```

### Training Configuration

```yaml
training:
  # Batch size per GPU
  batch_size: 16
  val_batch_size: 16  # Optional, defaults to batch_size
  
  # Number of frames
  num_frames: 16
  frames_to_sample: 16
  temporal_stride: 1
  
  # Training duration
  max_epochs: 100
  
  # Data loading
  num_workers: 10  # CPU workers per GPU (4-16 recommended)
  pin_memory: true
  
  # Optimization
  gradient_clip_val: 0  # 0 to disable
  accumulate_grad_batches: 1  # Gradient accumulation steps
  
  # Distributed training
  strategy: "ddp"  # "ddp", "deepspeed", or "auto"
  seed: 0
```

**Key Parameters:**
- **`num_workers`**: Number of CPU worker processes per GPU for data loading. Higher values (8-16) improve data loading speed but use more CPU/memory. Lower values (2-4) use less resources.
- **`batch_size`**: Adjust based on GPU memory. Use gradient accumulation for effective larger batches.
- **`accumulate_grad_batches`**: Simulates larger batch size. Effective batch = `batch_size × accumulate_grad_batches × num_gpus`

### Optimizer Configuration

```yaml
optimizer:
  type: "adamw"  # "adamw" or "radam"
  lr: 1e-3
  weight_decay: 0.05
  betas: [0.9, 0.95]
  eps: 1e-8
  warmup_steps: 0  # Only for adamw
```

### Checkpoint Configuration

```yaml
checkpoint:
  enable: true
  dir: "./output/checkpoints"
  prefix: "videomae_"
  save_top_k: 3  # Number of best checkpoints to keep
  save_last: false  # Save last checkpoint (default: false to save space)
  strict_top_k: false  # If true, ensures ONLY top_k checkpoints exist
  monitor: "train_loss_epoch"  # Metric to monitor
  save_ckpt_freq: null  # Save every N epochs (null = only best)
```

**Checkpoint Management:**
- **`save_top_k`**: Keeps the top K checkpoints based on monitored metric
- **`save_last`**: Saves the last checkpoint in addition to top_k (uses extra disk space)
- **`strict_top_k`**: When `true`, ensures exactly `save_top_k` checkpoints exist (disables `save_last`)

### Logging Configuration

```yaml
logging:
  log_dir: "./output/logs"
  log_freq: 10  # Log every N steps
```

## Advanced Features

### Checkpoint Management

Control disk space usage with checkpoint options:

**Strict mode (only top_k checkpoints):**
```yaml
checkpoint:
  save_top_k: 3
  save_last: false
  strict_top_k: true  # Ensures ONLY 3 checkpoints exist
```

**Normal mode (top_k + last):**
```yaml
checkpoint:
  save_top_k: 3
  save_last: true  # Also saves last checkpoint
  strict_top_k: false
```

### Multi-GPU Strategies

**DDP (Distributed Data Parallel):**
```yaml
training:
  strategy: "ddp"  # Standard multi-GPU training
```

**DeepSpeed (for very large models):**
```yaml
training:
  strategy: "deepspeed"  # DeepSpeed ZeRO optimization
```

### Mixed Precision Training

Automatically enabled on GPU:
- **16-bit mixed precision** on GPU (faster, less memory)
- **32-bit precision** on CPU

### Gradient Accumulation

Simulate larger batch sizes:
```yaml
training:
  batch_size: 8
  accumulate_grad_batches: 4  # Effective batch = 8 × 4 = 32
```

### Model EMA (Exponential Moving Average)

For finetuning, use EMA for better test performance:
```yaml
features:
  model_ema: true
  model_ema_decay: 0.9999
  model_ema_force_cpu: false
```

### Mixup/Cutmix Augmentation

For classification finetuning:
```yaml
augmentation:
  mixup: 0.8
  cutmix: 1.0
  label_smoothing: 0.1
```

**Note**: Mixup/Cutmix are disabled for regression tasks.

### Motion-Centric Masking (MCM)

EVEREST's key feature - selectively masks informative patches:
```yaml
model:
  mask_type: "motion-centric"
  motion_centric_masking_ratio: 0.7  # For pretraining
  # or
  mcm: true
  mcm_ratio: 0.4  # For finetuning
```

## Examples and Tutorials

### Quick Start: End-to-End Workflow

1. **Prepare dataset:**
```bash
# Optimize your video dataset
python optimize/optimize_dataset.py --input_dir videos/ --output_dir optimized/
```

2. **Configure pretraining:**
Edit `configs/base_test.yaml` with your dataset paths.

3. **Run pretraining:**
```bash
python train.py --config configs/base_test.yaml
```

4. **Configure finetuning:**
Edit `configs/finetune_example.yaml`:
- Set `pretrained_path` to your pretrained checkpoint
- Set `num_classes` for your task
- Configure dataset paths

5. **Run finetuning:**
```bash
python finetune.py --config configs/finetune_example.yaml
```

6. **Visualize reconstructions:**
```bash
python visualize_reconstruction.py \
    --config configs/base_test.yaml \
    --checkpoint checkpoints/pretrain/checkpoint.pth \
    --output_dir visualizations/
```

### Custom Dataset Example

**For classification:**
1. Organize videos by class or use CSV annotations
2. Optimize dataset with labels
3. Configure finetuning with correct `num_classes`

**For regression:**
1. Prepare videos with continuous labels (e.g., video quality scores)
2. Optimize dataset with float labels
3. Configure finetuning with `task_type: "regression"` and `output_dim: 1`

### Memory-Optimized Configuration

For limited GPU memory:
```yaml
training:
  batch_size: 4
  accumulate_grad_batches: 4  # Effective batch = 16
  num_workers: 4  # Reduce CPU workers

features:
  use_checkpoint: true  # Gradient checkpointing (slower but saves memory)
```

### Multi-GPU Example

With 4 GPUs:
```yaml
training:
  batch_size: 8  # Per GPU
  strategy: "ddp"
  # Effective batch size = 8 × 4 GPUs = 32
```

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors:**
- Reduce `batch_size`
- Enable gradient checkpointing: `features.use_checkpoint: true`
- Reduce `num_workers`
- Use gradient accumulation instead of larger batches

**Slow Data Loading:**
- Increase `num_workers` (but not more than CPU cores)
- Ensure dataset is optimized with litdata
- Check disk I/O speed (use SSD if possible)
- For cloud datasets, increase `max_cache_size`

**Checkpoint Loading Errors:**
- Verify checkpoint path is correct
- Check that model architecture matches (backbone, num_classes, etc.)
- For finetuning, ensure `pretrained_path` points to pretraining checkpoint

**Data Loading Errors:**
- Verify optimized dataset path is correct
- Check that videos are in supported format
- Ensure cache directory has write permissions
- For S3 datasets, verify `cloud_type: 's3_public'` is set

**Training Not Converging:**
- Check learning rate (try lower values: 1e-4, 5e-4)
- Verify normalization values match your dataset
- Ensure sufficient training data
- Check that labels are correct (0-indexed for classification)

### Performance Optimization Tips

1. **Data Loading:**
   - Use optimized datasets (litdata)
   - Set `num_workers` to 4-16 based on CPU cores
   - Enable `pin_memory: true` for faster GPU transfer

2. **Training Speed:**
   - Use mixed precision (automatic on GPU)
   - Increase batch size if memory allows
   - Use multiple GPUs with DDP

3. **Memory Usage:**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable `strict_top_k: true` for checkpoints
   - Set `save_last: false`

## Training Logs and Checkpoints

### UCF101

| Backbone | \#Frame |                          Pre-train (3,200 epochs)                           |                          Fine-tune (100 epochs)                           | Top-1 | Top-5 |
| :------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
|  ViT-B   |  16x5x3  | [log](https://drive.google.com/file/d/1dupg3ultdh1qsijUAYSZm8-hW2SAspLT/view?usp=share_link) / [checkpoint](https://drive.google.com/file/d/1liGNGprKdfiOCArK-WMqIcfeOJ-AZKzr/view?usp=share_link) | [log](https://drive.google.com/file/d/1EMlHBPqTC1_QURiCiaOdwPeoXdL67Gql/view?usp=share_link) / [checkpoint](https://drive.google.com/file/d/1iGFUxYpzjb7zaajB0O0j1MzS6PzyzrQF/view?usp=share_link) | 93.7  | 98.9  |

## Contact

Sunil Hwang: sunilhoho@kaist.ac.kr   
Jaehong Yoon: jaehong.yoon@kaist.ac.kr

## Acknowledgment

The code is built upon [VidoeMAE](https://github.com/MCG-NJU/VideoMAE).

## Citations

```
@inproceedings{hwang2024everest,
    title={EVEREST: Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens},
    author={Hwang, Sunil and Yoon, Jaehong and Lee, Youngwan and Hwang, Sung Ju},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```
