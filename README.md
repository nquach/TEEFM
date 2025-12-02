# VideoMAE EVEREST Training

This repository implements VideoMAE (Video Masked Autoencoder) training with EVEREST (Efficient Masked Video Autoencoder by Removing Redundant Spatiotemporal Tokens) masking strategy. The implementation uses PyTorch Lightning for easy multi-GPU training and supports ViT-S, ViT-B, and ViT-L backbones.

## Features

- **EVEREST Masking**: Efficient token selection based on motion/feature importance
- **Multiple Backbones**: Support for ViT-S (default), ViT-B, and ViT-L
- **PyTorch Lightning**: Easy multi-GPU training with automatic distributed training
- **AdamWScheduleFree Optimizer**: Schedule-free optimizer that adapts automatically
- **Configurable Training**: YAML-based configuration for easy hyperparameter tuning
- **Optional Features**:
  - Pretrained weight loading
  - Gradient norm tracking
  - Random dataset subset sampling
  - Configurable checkpoint directory and prefix

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

The dataset should be organized as CSV files with one video path per line:

- **Training CSV** (`mp4_paths.csv`): Contains full paths to training videos
- **Validation CSV** (`val500_2023-2024.csv`): Contains full paths to validation videos

Example CSV format:
```
/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.mp4
```

**Note**: Videos should already be preprocessed to 224x224x3 resolution. The dataloader will:
- Randomly sample 32 consecutive frames from each video
- Apply temporal downsampling with stride 2 → 16 frames
- No resizing is performed (assumes videos are already 224x224)

## Configuration

All training parameters are configured via `configs/config.yaml`. Key settings include:

### Model Configuration
- `backbone`: Choose from 'ViT-S', 'ViT-B', or 'ViT-L'
- `pretrained_weights`: Path to pretrained checkpoint (optional, set to `null` for random initialization)
- `mask_ratio`: Fraction of tokens to mask (default: 0.75)

### Data Configuration
- `train_csv`: Path to training CSV file
- `val_csv`: Path to validation CSV file
- `use_subset`: Enable random subset sampling (default: false)
- `subset_ratio`: Ratio of dataset to use if `use_subset` is true (default: 0.1)

### Training Configuration
- `batch_size`: Batch size per GPU
- `learning_rate`: Learning rate for optimizer
- `max_epochs`: Maximum number of training epochs
- `gpus`: Number of GPUs to use (0 for CPU)
- `track_gradient_norm`: Enable L2 norm tracking of gradients (default: false)

### Checkpoint Configuration
- `dir`: Directory to save checkpoints
- `prefix`: Prefix for checkpoint filenames

## Usage

### Basic Training

Train with default configuration:
```bash
python training/train.py --config configs/config.yaml
```

### Training with Custom Configuration

Create a custom configuration file and specify it:
```bash
python training/train.py --config configs/my_config.yaml
```

### Multi-GPU Training

PyTorch Lightning automatically handles multi-GPU training. Set the `gpus` parameter in the config file:
```yaml
training:
  gpus: 4  # Use 4 GPUs
```

## Codebase Structure

```
TEEFM/
├── configs/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── __init__.py
│   └── video_dataset.py          # Video dataset loader
├── models/
│   ├── __init__.py
│   ├── videomae.py               # VideoMAE model architecture
│   └── everest_masking.py        # EVEREST masking generator
├── training/
│   ├── __init__.py
│   ├── lightning_module.py       # PyTorch Lightning module
│   └── train.py                  # Main training script
├── utils/
│   ├── __init__.py
│   └── optimizer.py              # Optimizer factory
├── requirements.txt
└── README.md
```

## Key Components

### VideoMAE Model (`models/videomae.py`)

The VideoMAE model implements:
- **Patch Embedding**: Converts video frames to tokens
- **Temporal Embeddings**: Learnable embeddings for temporal dimension
- **Vision Transformer Encoder**: Processes visible tokens only
- **Decoder**: Reconstructs all tokens from encoded visible tokens
- **EVEREST Masking**: Efficient token selection based on motion importance

### EVEREST Masking (`models/everest_masking.py`)

The EVEREST masking strategy:
1. Computes motion importance for each spatiotemporal token
2. Selects top-k most informative tokens to keep (unmasked)
3. Masks the remaining tokens for reconstruction

### Lightning Module (`training/lightning_module.py`)

The PyTorch Lightning module handles:
- Forward pass through VideoMAE
- Loss computation (MSE with normalization)
- Optional gradient norm tracking
- Optimizer configuration (AdamWScheduleFree)
- Training and validation steps

### Training Script (`training/train.py`)

The main training script:
- Loads configuration from YAML
- Creates datasets and data loaders
- Initializes model and Lightning module
- Sets up PyTorch Lightning Trainer with checkpointing
- Starts training

## Loss Function

The model uses MSE loss with normalization:
- Target patches are normalized by their mean and std
- Loss is computed only on masked tokens (reconstruction task)
- Normalization helps with training stability

## Checkpointing

Checkpoints are automatically saved to the directory specified in the config:
- Best model based on validation loss
- Last checkpoint
- Top-k checkpoints (configurable)

Checkpoint format: `{prefix}-{epoch:02d}-{val_loss:.4f}.ckpt`

## Loading Pretrained Weights

To initialize the model with pretrained weights, set the `pretrained_weights` path in the config:

```yaml
model:
  pretrained_weights: './checkpoints/videomae_everest-epoch=50-val_loss=0.1234.ckpt'
```

## Optional Features

### Gradient Norm Tracking

Enable gradient norm tracking to monitor training stability:
```yaml
training:
  track_gradient_norm: true
```

This will log the L2 norm of the total loss gradient at each training step.

### Dataset Subset Sampling

Train on a random subset of the dataset for faster experimentation:
```yaml
data:
  use_subset: true
  subset_ratio: 0.1  # Use 10% of the dataset
```

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in the config
- Reduce `num_workers` for data loading
- Enable gradient accumulation: `accumulate_grad_batches: 2`

### Slow Data Loading

- Increase `num_workers` in the config
- Ensure videos are stored on fast storage (SSD)
- Consider using `decord` for faster video I/O (uncomment in requirements.txt)

### CUDA Errors

- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if GPU memory is limited
- Set `gpus: 0` in config to use CPU (slow, for testing only)

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

## Contact

[Add contact information here]

