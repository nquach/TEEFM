# VideoMAE Training with EVEREST Method

This codebase implements and trains a Video Masked Autoencoder (VideoMAE) model using the EVEREST training method. The model uses a Vision Transformer (ViT) backbone with options for ViT-S, ViT-B, and ViT-L architectures.

## Features

- **VideoMAE Model**: Implements masked autoencoding for video data
- **EVEREST Training**: Self-supervised learning method for video understanding
- **Multiple Backbones**: Support for ViT-S, ViT-B, and ViT-L
- **PyTorch Lightning**: Easy multi-GPU training support
- **AdamWScheduleFree**: Schedule-free optimizer for training
- **Configurable Checkpointing**: Custom checkpoint directory and prefix
- **Optional Gradient Tracking**: Track L2 norm of gradients (off by default)
- **Pretrained Weights**: Support for loading pretrained model weights

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The codebase expects:
- Training videos: CSV file with video paths (one per line)
- Validation videos: CSV file with video paths (one per line)
- Videos should be 224x224x3 (already resized)
- Variable number of frames per video

## Usage

### Basic Training

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --batch_size 8 \
    --max_epochs 100 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_prefix videomae
```

### Training with ViT-B Backbone

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_b \
    --batch_size 4 \
    --max_epochs 100
```

### Training with Pretrained Weights

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --pretrained path/to/pretrained/weights.pth \
    --batch_size 8
```

### Training with Gradient Norm Tracking

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --track_grad_norm \
    --batch_size 8
```

### Multi-GPU Training

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --gpus 4 \
    --batch_size 8
```

## Command Line Arguments

### Data Arguments
- `--train_csv`: Path to training CSV file (default: `mp4_paths.csv`)
- `--val_csv`: Path to validation CSV file (default: `val500_2023-2024.csv`)
- `--batch_size`: Batch size for training (default: 8)
- `--num_workers`: Number of data loading workers (default: 4)
- `--num_frames_to_sample`: Number of consecutive frames to sample (default: 32)
- `--temporal_stride`: Temporal stride for downsampling (default: 2)

### Model Arguments
- `--backbone`: Vision Transformer backbone (`vit_s`, `vit_b`, `vit_l`) (default: `vit_s`)
- `--img_size`: Input image size (default: 224)
- `--patch_size`: Patch size (default: 16)
- `--mask_ratio`: Ratio of patches to mask (default: 0.75)
- `--norm_pix_loss`: Normalize pixel loss (flag)
- `--pretrained`: Path to pretrained weights (default: None)

### Training Arguments
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.05)
- `--warmup_steps`: Number of warmup steps (default: 1000)
- `--max_epochs`: Maximum number of epochs (default: 100)
- `--track_grad_norm`: Track L2 norm of gradients (flag)

### Checkpointing Arguments
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--checkpoint_prefix`: Prefix for checkpoint filenames (default: `videomae`)
- `--resume_from_checkpoint`: Path to checkpoint to resume from (default: None)

### Hardware Arguments
- `--gpus`: Number of GPUs to use (default: 1)
- `--precision`: Training precision (16 or 32) (default: 32)

## Project Structure

```
TEEFM/
├── data/
│   └── video_dataset.py          # Dataset class for video loading
├── models/
│   └── videomae.py                # VideoMAE model implementation
├── training/
│   ├── lightning_module.py        # PyTorch Lightning module
│   └── train.py                   # Main training script
├── config/
│   └── config.yaml                # Configuration file
├── mp4_paths.csv                  # Training video paths
├── val500_2023-2024.csv           # Validation video paths
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Model Architecture

The VideoMAE model consists of:
1. **Encoder**: Vision Transformer that processes visible (unmasked) patches
2. **Decoder**: Lightweight transformer that reconstructs masked patches
3. **Masking**: Random masking of 75% of patches (configurable)

## Training Process

1. **Data Loading**: Videos are loaded and processed:
   - Randomly sample 32 consecutive frames
   - Temporally downsample with stride 2 to get 16 frames
   - No resizing (videos already 224x224)

2. **Masking**: 75% of patches are randomly masked

3. **Reconstruction**: Model learns to reconstruct masked patches

4. **Loss**: Mean squared error loss on masked patches only

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir ./logs
```

Metrics logged:
- `train_loss`: Training reconstruction loss
- `val_loss`: Validation reconstruction loss
- `grad_norm`: L2 norm of gradients (if enabled)

## Checkpoints

Checkpoints are saved in the specified checkpoint directory with the format:
```
{checkpoint_prefix}-{epoch:02d}-{val_loss:.4f}.ckpt
```

The best 3 models (based on validation loss) are kept, along with the last checkpoint.

## License

This codebase is provided for research purposes.

