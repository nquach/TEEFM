# Learning Rate Sweep Guide

This guide explains how to use the Optuna-based learning rate sweep feature to find the optimal learning rate for training EVEREST ViT-S VideoMAE models.

## Overview

The learning rate sweep uses [Optuna](https://optuna.org/), a hyperparameter optimization framework, to automatically search for the best learning rate. It runs multiple training trials with different learning rates and selects the one that minimizes validation loss.

### Key Features

- **Intelligent Search**: Uses Tree-structured Parzen Estimator (TPE) sampler for efficient hyperparameter exploration
- **Early Stopping**: Pruning mechanism stops unpromising trials early to save computational resources
- **Resumable**: Studies can be saved and resumed if interrupted
- **Visualization**: Generates plots to visualize the optimization process
- **Automatic Config Generation**: Creates a configuration file with the optimal learning rate for full training

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install optuna plotly kaleido
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Configuration Files

You need two configuration files:

1. **Base Training Config** (`configs/train_config.yaml`): Your standard training configuration
2. **LR Sweep Config** (`configs/lr_sweep_config.yaml`): Configuration for the learning rate sweep

The LR sweep config should reference your base config:

```yaml
base_config_path: "configs/train_config.yaml"

lr_sweep:
  n_trials: 10
  epochs_per_trial: 15
  lr_min: 1e-6
  lr_max: 1e-2
  # ... other settings
```

### 2. Run the Learning Rate Sweep

```bash
python lr_sweep.py --sweep_config configs/lr_sweep_config.yaml
```

Or with explicit base config:

```bash
python lr_sweep.py --sweep_config configs/lr_sweep_config.yaml --base_config configs/train_config.yaml
```

### 3. Review Results

After completion, check the output directory (default: `output/lr_sweep/`) for:
- `best_params.json`: Best learning rate and validation loss
- `best_config.yaml`: Configuration file with optimal learning rate
- `trials_results.csv`: All trial results
- `optimization_history.png`: Plot showing optimization progress
- `param_importance.png`: Parameter importance visualization

## Configuration Options

### LR Sweep Configuration (`configs/lr_sweep_config.yaml`)

#### Study Configuration

```yaml
lr_sweep:
  # Study name (used for storage and identification)
  study_name: "videomae_lr_sweep"
  
  # Number of trials to run
  n_trials: 10
  
  # Optimization direction: "minimize" for validation loss
  direction: "minimize"
  
  # Study storage (SQLite database for persistence)
  # Set to null for in-memory storage (not resumable)
  storage: "sqlite:///output/lr_sweep/lr_sweep_study.db"
```

#### Learning Rate Search Space

```yaml
  # Learning rate search space
  lr_min: 1e-6      # Minimum learning rate to test
  lr_max: 1e-2      # Maximum learning rate to test
  lr_log_scale: true  # Use log-uniform distribution (recommended)
```

**Recommendations:**
- For initial exploration: `lr_min: 1e-6, lr_max: 1e-2`
- For focused search: `lr_min: 1e-5, lr_max: 1e-3`
- Always use `lr_log_scale: true` for learning rates

#### Training Configuration

```yaml
  # Number of epochs to train each trial
  epochs_per_trial: 15  # Moderate for good estimates
  
  # Pruning configuration
  enable_pruning: true   # Enable early stopping
  pruning_patience: 3   # Epochs without improvement before pruning
```

**Epochs per Trial Guidelines:**
- **Quick sweep (5-10 epochs)**: Fast but less accurate
- **Moderate (15-20 epochs)**: Good balance (recommended)
- **Thorough (20+ epochs)**: More accurate but slower

#### Output Configuration

```yaml
  # Output directory for results
  output_dir: "output/lr_sweep"
  
  # Save best config with optimal LR
  save_best_config: true
  
  # Generate visualization plots
  generate_plots: true
```

## Understanding the Results

### Best Parameters (`best_params.json`)

```json
{
  "best_learning_rate": 1.5e-4,
  "best_value": 0.2345,
  "best_trial_number": 7,
  "n_trials": 10,
  "n_complete": 8,
  "n_pruned": 2,
  "n_failed": 0
}
```

- `best_learning_rate`: The optimal learning rate found
- `best_value`: Best validation loss achieved
- `best_trial_number`: Trial number that achieved best result
- `n_complete`: Number of completed trials
- `n_pruned`: Number of trials stopped early
- `n_failed`: Number of failed trials

### Trials Results (`trials_results.csv`)

CSV file containing all trial information:
- Trial number
- Learning rate tested
- Validation loss achieved
- Trial state (COMPLETE, PRUNED, FAIL)
- Timestamps

### Best Configuration (`best_config.yaml`)

A copy of your base config with the learning rate updated to the optimal value. Use this for full training:

```bash
python train.py --config output/lr_sweep/best_config.yaml
```

## Advanced Usage

### Resuming a Study

If a study is interrupted, you can resume it:

```bash
python lr_sweep.py --sweep_config configs/lr_sweep_config.yaml --resume
```

The study will continue from where it left off, using the SQLite database specified in the config.

### Customizing the Search Space

You can narrow the search space based on initial results:

```yaml
lr_sweep:
  # If initial sweep suggests LR around 1e-4, narrow the range
  lr_min: 5e-5
  lr_max: 5e-4
  lr_log_scale: true
```

### Disabling Pruning

For more thorough exploration (at the cost of time):

```yaml
lr_sweep:
  enable_pruning: false
```

### Adjusting Pruning Sensitivity

Make pruning more or less aggressive:

```yaml
lr_sweep:
  enable_pruning: true
  pruning_patience: 5  # More patience = less aggressive pruning
```

## Best Practices

### 1. Start with a Wide Range

For the first sweep, use a wide learning rate range (1e-6 to 1e-2) to explore the full space.

### 2. Use Moderate Epochs per Trial

15-20 epochs per trial provides a good balance between accuracy and speed. Too few epochs may not capture the true performance, while too many wastes time.

### 3. Enable Pruning

Pruning saves significant time by stopping unpromising trials early. The default settings work well for most cases.

### 4. Run Multiple Sweeps

If time permits:
1. Run initial wide sweep (10-20 trials)
2. Analyze results and narrow the range
3. Run focused sweep (10-15 trials) in the promising region

### 5. Use Study Storage

Always use SQLite storage for important sweeps to enable resuming and result persistence:

```yaml
storage: "sqlite:///output/lr_sweep/lr_sweep_study.db"
```

### 6. Validate Results

After finding the optimal LR, validate it by:
1. Training for a few epochs with the best LR
2. Comparing with nearby LRs to ensure it's not a fluke
3. Checking that the validation loss is stable

## Troubleshooting

### Issue: "Could not find val_loss metric"

**Solution**: Ensure your Lightning module logs `val_loss` in `validation_step()`:

```python
def validation_step(self, batch, batch_idx):
    # ... your validation code ...
    self.log('val_loss', loss)
```

### Issue: All trials are being pruned

**Possible causes:**
1. Learning rate range is too high (model diverges)
2. Pruning is too aggressive
3. Epochs per trial is too low

**Solutions:**
- Lower `lr_max` in the config
- Increase `pruning_patience`
- Increase `epochs_per_trial`

### Issue: Study takes too long

**Solutions:**
- Reduce `n_trials`
- Reduce `epochs_per_trial`
- Enable pruning (if disabled)
- Use a narrower LR range

### Issue: No improvement in validation loss

**Possible causes:**
1. Learning rate range doesn't include optimal value
2. Model architecture or other hyperparameters need tuning
3. Dataset issues

**Solutions:**
- Expand the LR range
- Check other hyperparameters (batch size, model size, etc.)
- Verify dataset and data loading

## Integration with Full Training

After finding the optimal learning rate:

1. **Use the generated best config:**
   ```bash
   python train.py --config output/lr_sweep/best_config.yaml
   ```

2. **Or manually update your config:**
   ```yaml
   optimizer:
     lr: 1.5e-4  # From best_params.json
   ```

3. **Run full training:**
   ```bash
   python train.py --config configs/train_config.yaml
   ```

## Example Workflow

### Step 1: Initial Wide Sweep

```yaml
# lr_sweep_config.yaml
lr_sweep:
  n_trials: 20
  epochs_per_trial: 15
  lr_min: 1e-6
  lr_max: 1e-2
  lr_log_scale: true
```

```bash
python lr_sweep.py --sweep_config configs/lr_sweep_config.yaml
```

### Step 2: Analyze Results

Check `output/lr_sweep/optimization_history.png` to see where the best LRs cluster.

### Step 3: Focused Sweep (Optional)

If a promising region is found, narrow the search:

```yaml
lr_sweep:
  n_trials: 15
  epochs_per_trial: 20
  lr_min: 1e-4
  lr_max: 1e-3
  lr_log_scale: true
```

### Step 4: Use Best LR for Full Training

```bash
python train.py --config output/lr_sweep/best_config.yaml
```

## Visualization

The sweep generates several visualization plots:

### Optimization History

Shows how validation loss changes across trials. Look for:
- Steep downward slopes (good learning rates)
- Flat or upward trends (poor learning rates)
- Clustering of good results

### Parameter Importance

Shows the importance of different hyperparameters (in this case, just learning rate).

### Parallel Coordinate Plot

Shows the relationship between learning rate and validation loss across all trials.

## Technical Details

### How It Works

1. **Study Creation**: Optuna creates a study with TPE sampler and MedianPruner
2. **Trial Execution**: For each trial:
   - Optuna suggests a learning rate
   - Model is trained with that LR for specified epochs
   - Validation loss is reported after each epoch
   - Pruner decides if trial should continue
3. **Result Collection**: Best trial is identified based on validation loss
4. **Output Generation**: Results are saved and visualized

### Pruning Mechanism

The MedianPruner compares each trial's intermediate results to the median of all completed trials. If a trial is significantly worse, it's pruned (stopped early) to save time.

### TPE Sampler

Tree-structured Parzen Estimator is a Bayesian optimization algorithm that:
- Models the distribution of good hyperparameters
- Suggests promising regions to explore
- Balances exploration and exploitation

## Additional Resources

- [Optuna Documentation](https://optuna.org/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/)
- [Learning Rate Finder in PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#learning-rate-finder)

## Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the generated logs in `output/logs/lr_sweep_trial_*/`
3. Examine the trial results CSV for patterns
4. Verify your base training config works independently

