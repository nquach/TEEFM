# How to Use config.yaml

The training script now supports loading configuration from a YAML file. This makes it easy to manage hyperparameters and experiment configurations.

## Basic Usage

### 1. Using Default Config File

If you have a `config/config.yaml` file in your project root, the script will automatically load it:

```bash
python training/train.py
```

The script will automatically look for `config/config.yaml` and load all settings from there.

### 2. Using a Custom Config File

You can specify a custom config file path:

```bash
python training/train.py --config path/to/your/config.yaml
```

### 3. Overriding Config Values with Command-Line Arguments

Command-line arguments **always override** config file values. This allows you to quickly test different hyperparameters without editing the config file:

```bash
# Use config.yaml but override batch size and learning rate
python training/train.py --batch_size 16 --learning_rate 2e-4

# Use config.yaml but change backbone
python training/train.py --backbone vit_b
```

### 4. Using Only Command-Line Arguments (No Config File)

If you don't have a config file or don't want to use one, you can specify all parameters via command-line:

```bash
python training/train.py \
    --train_csv mp4_paths.csv \
    --val_csv val500_2023-2024.csv \
    --backbone vit_s \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 100
```

## Config File Structure

The `config/config.yaml` file is organized into sections:

```yaml
# Data configuration
data:
  train_csv: "mp4_paths.csv"
  val_csv: "val500_2023-2024.csv"
  batch_size: 8
  num_workers: 4
  num_frames_to_sample: 32
  temporal_stride: 2

# Model configuration
model:
  backbone: "vit_s"  # Options: vit_s, vit_b, vit_l
  img_size: 224
  patch_size: 16
  mask_ratio: 0.75
  norm_pix_loss: true
  pretrained: null  # Path to pretrained weights, or null

# Training configuration
training:
  learning_rate: 1.0e-4
  weight_decay: 0.05
  warmup_steps: 1000
  max_epochs: 100
  track_grad_norm: false

# Checkpointing configuration
checkpointing:
  checkpoint_dir: "./checkpoints"
  checkpoint_prefix: "videomae"
  resume_from_checkpoint: null

# Logging configuration
logging:
  log_dir: "./logs"
  experiment_name: "videomae_experiment"

# Hardware configuration
hardware:
  gpus: 1
  precision: 32  # Options: 16, 32
```

## Examples

### Example 1: Training with ViT-B Backbone

Edit `config/config.yaml`:
```yaml
model:
  backbone: "vit_b"
  # ... other settings
```

Then run:
```bash
python training/train.py
```

### Example 2: Quick Hyperparameter Tuning

Keep your base config in `config/config.yaml`, then override specific values:

```bash
# Test different learning rates
python training/train.py --learning_rate 5e-5
python training/train.py --learning_rate 2e-4
python training/train.py --learning_rate 1e-3

# Test different batch sizes
python training/train.py --batch_size 4
python training/train.py --batch_size 8
python training/train.py --batch_size 16
```

### Example 3: Multiple Experiment Configs

Create different config files for different experiments:

**config/vit_s_experiment.yaml:**
```yaml
model:
  backbone: "vit_s"
training:
  learning_rate: 1e-4
  max_epochs: 100
```

**config/vit_b_experiment.yaml:**
```yaml
model:
  backbone: "vit_b"
training:
  learning_rate: 5e-5
  max_epochs: 150
```

Then run:
```bash
python training/train.py --config config/vit_s_experiment.yaml
python training/train.py --config config/vit_b_experiment.yaml
```

### Example 4: Resuming Training

Edit `config/config.yaml`:
```yaml
checkpointing:
  resume_from_checkpoint: "./checkpoints/videomae-epoch=50-val_loss=0.1234.ckpt"
```

Or override via command-line:
```bash
python training/train.py --resume_from_checkpoint ./checkpoints/videomae-epoch=50-val_loss=0.1234.ckpt
```

## Priority Order

When both config file and command-line arguments are provided, the priority is:

1. **Command-line arguments** (highest priority)
2. **Config file values**
3. **Default values** (lowest priority)

This means command-line arguments always win, making it easy to override config values for quick experiments.

## Tips

1. **Use config files for stable configurations**: Store your main experiment settings in `config/config.yaml`

2. **Use command-line for quick tests**: Override specific hyperparameters without editing files

3. **Create multiple config files**: For different experiments, backbones, or datasets

4. **Track your configs**: Consider version controlling your config files to track experiment settings

5. **Use null for optional values**: Set `pretrained: null` or `resume_from_checkpoint: null` in YAML to indicate no value

## Troubleshooting

**Config file not found?**
- The script will print: "No config file found, using command-line arguments and defaults"
- This is fine - you can still use command-line arguments

**Config value not being used?**
- Check if you're overriding it with a command-line argument
- Make sure the YAML syntax is correct (proper indentation, no tabs)

**YAML syntax errors?**
- Use spaces, not tabs for indentation
- Make sure strings are quoted if they contain special characters
- Use `null` (not `None` or `NULL`) for empty values

