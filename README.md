# VideoMAE Training with EVEREST Masking

This codebase implements and trains a Video Vision Transformer (VideoMAE) model using the EVEREST (Efficient Video Representation Learning with Masked Spatio-Temporal Modeling) training method.

## Features

- **VideoMAE Architecture**: Self-supervised video representation learning with masked autoencoding
- **EVEREST Masking**: High masking ratio (90%) with temporal consistency
- **Multiple Backbones**: Support for ViT-S, ViT-B, and ViT-L
- **PyTorch Lightning**: Easy multi-GPU training and distributed training
- **AdamWScheduleFree Optimizer**: Schedule-free optimizer for stable training
- **Efficient Video Loading**: Uses torchcodec for fast video decoding
- **Flexible Configuration**: Easy hyperparameter tuning via config module
- **Checkpointing**: Customizable checkpoint directory and prefix
- **Gradient Tracking**: Optional L2 norm tracking of gradients

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TEEFM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The codebase expects:
- Training videos listed in `mp4_paths.csv` (one path per line)
- Validation videos listed in `val500_2023-2024.csv` (one path per line)
- Videos should be preprocessed to 224x224x3 resolution
- Videos can have variable number of frames

## Usage

### Basic Training

Train with default settings (ViT-S backbone):
```bash
python train.py
```

### Training with Custom Configuration

```bash
python train.py \
    --backbone vit_b \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --gpus 2 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_prefix videomae_vitb
```

### Training with Pretrained Weights

```bash
python train.py \
    --pretrained_weights /path/to/pretrained/weights.pth \
    --backbone vit_s
```

### Enable Gradient Norm Tracking

```bash
python train.py --track_gradient_norm
```

## Configuration

The codebase uses a flexible configuration system in `config.py`. You can modify:

- **Model Configuration**: Backbone type, patch size, masking ratio
- **Data Configuration**: Batch size, number of workers, frame sampling
- **Training Configuration**: Learning rate, weight decay, epochs
- **Checkpoint Configuration**: Directory, prefix, saving strategy

## Architecture

### VideoMAE Model

The model consists of:
1. **ViT Encoder**: Processes visible video patches
2. **EVEREST Masking**: High-ratio masking with temporal consistency
3. **Lightweight Decoder**: Reconstructs masked patches

### Data Processing Pipeline

1. Load video using torchcodec VideoDecoder
2. Randomly sample 32 consecutive frames
3. Apply temporal downsampling with stride 2 → 16 frames
4. Extract 3D patches (tubelets) of size (2, 16, 16)
5. Apply EVEREST masking (90% masking ratio)

## Project Structure

```
TEEFM/
├── config.py                 # Configuration module
├── train.py                  # Main training script
├── data/
│   ├── __init__.py
│   └── video_dataset.py      # Video dataset implementation
├── models/
│   ├── __init__.py
│   ├── vit_backbone.py       # ViT backbone implementation
│   ├── everest_masking.py    # EVEREST masking strategy
│   └── videomae.py           # VideoMAE model
├── training/
│   ├── __init__.py
│   └── lightning_module.py    # PyTorch Lightning module
├── requirements.txt
└── README.md
```

## Key Components

### 1. VideoDataset (`data/video_dataset.py`)

- Loads videos from CSV files
- Uses torchcodec for efficient decoding
- Randomly samples consecutive frames
- Applies temporal downsampling

### 2. ViTBackbone (`models/vit_backbone.py`)

- Implements Vision Transformer encoder
- Supports ViT-S, ViT-B, and ViT-L
- 3D patch embedding for video
- Multi-head self-attention

### 3. EverestMasking (`models/everest_masking.py`)

- High masking ratio (90%)
- Maintains temporal consistency
- Random masking strategy

### 4. VideoMAE (`models/videomae.py`)

- Encoder-decoder architecture
- Masked autoencoding objective
- Reconstruction loss on masked tokens

### 5. VideoMAELightningModule (`training/lightning_module.py`)

- PyTorch Lightning wrapper
- Training and validation steps
- AdamWScheduleFree optimizer
- Optional gradient norm tracking

## Training Details

- **Loss Function**: Mean Squared Error (MSE) on masked patches
- **Optimizer**: AdamWScheduleFree (no learning rate scheduling needed)
- **Mixed Precision**: Enabled by default (FP16)
- **Gradient Clipping**: Default value of 1.0

## Checkpointing

Checkpoints are saved in the specified directory with the format:
```
{checkpoint_prefix}-{epoch:02d}-{val_loss:.4f}.ckpt
```

The top-k checkpoints (based on validation loss) are saved, along with the last checkpoint.

## Multi-GPU Training

The codebase automatically supports multi-GPU training when using PyTorch Lightning. Simply specify the number of GPUs:

```bash
python train.py --gpus 4
```

PyTorch Lightning will handle data parallelism and gradient synchronization.

## Monitoring

Training progress is logged to TensorBoard. View logs with:

```bash
tensorboard --logdir ./checkpoints/logs
```

## Hyperparameter Tuning

All hyperparameters can be easily tuned by:
1. Modifying `config.py` directly
2. Using command-line arguments in `train.py`
3. Creating custom configuration objects

## Citation

If you use this codebase, please cite the original papers:
- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
- EVEREST: Efficient Video Representation Learning with Masked Spatio-Temporal Modeling

## License

[Add your license here]

## Contact

[Add contact information here]
