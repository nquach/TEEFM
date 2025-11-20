# TEEFM: VideoMAE Training with EVEREST Masking

This repository implements a Video Vision Transformer (VideoMAE) model trained using the EVEREST (Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens) method. The implementation uses PyTorch Lightning for easy multi-GPU training and supports ViT-S, ViT-B, and ViT-L backbones.

## Features

- **VideoMAE Architecture**: Masked autoencoder for self-supervised video representation learning
- **EVEREST Masking**: Intelligent masking strategy that selects informative tokens based on motion features
- **Multiple Backbones**: Support for ViT-S, ViT-B, and ViT-L architectures
- **PyTorch Lightning**: Easy multi-GPU training with automatic distributed data parallel support
- **AdamWScheduleFree Optimizer**: State-of-the-art optimizer that doesn't require learning rate scheduling
- **Comprehensive Configuration**: Easy hyperparameter tuning through configuration files

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

The dataset should be a CSV file (`mp4_paths.csv`) with one video path per line (no header):

```
/path/to/video1.mp4
/path/to/video2.mp4
...
```

**Video Requirements:**
- Videos should already be resized to 224x224x3
- Variable number of frames (will be sampled during training)
- MP4 format

## Usage

### Basic Training

Train with default ViT-S backbone:
```bash
python train.py --csv_file mp4_paths.csv
```

### Training with Different Backbones

**ViT-S (Small, fastest):**
```bash
python train.py --backbone ViT-S --batch_size 16
```

**ViT-B (Base, balanced):**
```bash
python train.py --backbone ViT-B --batch_size 8
```

**ViT-L (Large, best performance):**
```bash
python train.py --backbone ViT-L --batch_size 4
```

### Custom Hyperparameters

```bash
python train.py \
    --backbone ViT-B \
    --batch_size 8 \
    --learning_rate 1.5e-4 \
    --mask_ratio 0.9 \
    --max_epochs 800 \
    --devices 4 \
    --precision 16-mixed
```

### Multi-GPU Training

PyTorch Lightning automatically handles multi-GPU training. Simply specify the number of devices:

```bash
python train.py --backbone ViT-S --devices 4
```

Or use all available GPUs:
```bash
python train.py --backbone ViT-S --devices -1
```

### Resume Training

```bash
python train.py --resume checkpoints/last.ckpt
```

## Configuration

All hyperparameters can be configured in `config.py`. The `VideoMAEConfig` class contains:

- **Model parameters**: backbone, patch_size, decoder settings
- **Masking parameters**: mask_ratio, motion_weight
- **Training parameters**: learning_rate, batch_size, max_epochs
- **Hardware settings**: devices, precision, gradient accumulation

You can also use predefined configurations:
- `get_vit_s_config()`: Optimized for ViT-S
- `get_vit_b_config()`: Optimized for ViT-B
- `get_vit_l_config()`: Optimized for ViT-L

## Project Structure

```
TEEFM/
├── data/
│   ├── __init__.py
│   └── video_dataset.py          # Video dataset and data loading
├── models/
│   ├── __init__.py
│   ├── vit_backbone.py           # Vision Transformer backbone
│   ├── videomae.py               # VideoMAE encoder-decoder
│   └── everest_masking.py        # EVEREST masking strategy
├── training/
│   ├── __init__.py
│   └── lightning_module.py       # PyTorch Lightning module
├── config.py                     # Configuration and hyperparameters
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Components

### 1. Video Dataset (`data/video_dataset.py`)
- Loads videos from CSV file
- Randomly samples 32 consecutive frames
- Temporally downsamples with stride 2 to get 16 frames
- Handles videos already at 224x224 resolution

### 2. Vision Transformer Backbone (`models/vit_backbone.py`)
- Implements ViT-S, ViT-B, and ViT-L architectures
- Adapted for video processing with temporal patches
- Supports masking for self-supervised learning

### 3. VideoMAE Model (`models/videomae.py`)
- Encoder: ViT backbone for encoding visible patches
- Decoder: Lightweight transformer for reconstructing masked patches
- Patchify/Unpatchify: Converts between video frames and patches

### 4. EVEREST Masking (`models/everest_masking.py`)
- Motion-based token selection: Identifies patches with high motion content
- Information-intensive frame selection: Focuses on informative frames
- Combines motion scores with small random component for diversity

### 5. Lightning Module (`training/lightning_module.py`)
- Integrates model, masking, and training logic
- Handles loss computation (MSE on normalized patches)
- Configures AdamWScheduleFree optimizer
- Logs metrics to TensorBoard

## Training Details

### Loss Function
The model uses Mean Squared Error (MSE) loss on normalized patches. Only masked patches contribute to the loss (where mask == 0).

### Masking Strategy
EVEREST uses a combination of:
- **Motion estimation**: Computes frame differences to identify motion-rich regions
- **Information selection**: Selects top-k most informative patches
- **Random component**: Small random component for diversity

### Optimizer
Uses `AdamWScheduleFree` which doesn't require learning rate scheduling. Falls back to AdamW with cosine annealing if schedulefree is not available.

## Monitoring Training

Training logs are saved to `logs/` directory. View with TensorBoard:

```bash
tensorboard --logdir logs
```

Key metrics:
- `train/loss`: Reconstruction loss
- `train/mask_ratio`: Actual masking ratio
- `train/lr`: Learning rate

## Checkpoints

Checkpoints are saved to `checkpoints/` directory:
- `last.ckpt`: Latest checkpoint
- `videomae-{epoch}-{loss}.ckpt`: Top checkpoints based on validation loss

## Hyperparameter Tuning

Easy hyperparameter tuning through:
1. **Configuration file**: Modify `config.py`
2. **Command-line arguments**: Override specific parameters
3. **Predefined configs**: Use `get_vit_*_config()` functions

Example:
```python
from config import VideoMAEConfig

config = VideoMAEConfig(
    backbone='ViT-B',
    batch_size=16,
    learning_rate=2e-4,
    mask_ratio=0.85
)
```

## Citation

If you use this code, please cite the EVEREST paper:

```bibtex
@inproceedings{hwang2024everest,
    title={EVEREST: Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens},
    author={Hwang, Sunil and Yoon, Jaehong and Lee, Youngwan and Hwang, Sung Ju},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```

## License

[Add your license here]

## Acknowledgments

- EVEREST: Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens
- VideoMAE: Masked Autoencoders are Scalable Vision Learners for Video
- PyTorch Lightning for the training framework
