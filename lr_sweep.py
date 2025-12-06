"""
Learning Rate Sweep Script using Optuna

This script performs hyperparameter optimization to find the optimal learning rate
for training EVEREST ViT-S VideoMAE models. It uses Optuna to search the learning
rate space and selects the best based on validation loss.
"""

import os
import yaml
import json
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from train import load_config, create_datasets, create_data_loaders
from lightning_module import VideoMAELightningModule
from callbacks.optuna_callback import OptunaPruningCallback
from utils.optuna_utils import (
    save_study_results as save_study_results_util,
    save_trials_csv,
    save_best_config,
    generate_plots,
    print_study_summary
)


def objective(trial, base_config, sweep_config):
    """
    Optuna objective function for learning rate optimization.
    
    This function is called for each trial. It:
    1. Suggests a learning rate from the search space
    2. Creates a model with that learning rate
    3. Trains for a specified number of epochs
    4. Returns the best validation loss
    
    Args:
        trial (optuna.Trial): The Optuna trial object
        base_config (dict): Base training configuration
        sweep_config (dict): LR sweep specific configuration
    
    Returns:
        float: Best validation loss achieved during training
    """
    # Suggest learning rate from search space
    lr_min = sweep_config.get('lr_min', 1e-6)
    lr_max = sweep_config.get('lr_max', 1e-2)
    lr_log_scale = sweep_config.get('lr_log_scale', True)
    
    if lr_log_scale:
        lr = trial.suggest_float('learning_rate', lr_min, lr_max, log=True)
    else:
        lr = trial.suggest_float('learning_rate', lr_min, lr_max)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing learning rate = {lr:.6e}")
    print(f"{'='*60}")
    
    # Update config with suggested learning rate
    config = base_config.copy()
    config['optimizer']['lr'] = lr
    
    # Set epochs per trial
    epochs_per_trial = sweep_config.get('epochs_per_trial', 15)
    config['training']['max_epochs'] = epochs_per_trial
    
    # Disable checkpointing during sweep (to save disk space)
    config['checkpoint']['enable'] = False
    
    # Set random seed for reproducibility (can vary by trial if desired)
    seed = base_config['training'].get('seed', 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create datasets
    train_dataset, val_dataset, use_optimized = create_datasets(config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, config, use_optimized
    )
    
    # Create Lightning module
    model = VideoMAELightningModule(config)
    
    # Setup logging (separate log directory for each trial)
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'output/logs')
    trial_log_dir = os.path.join(log_dir, f"lr_sweep_trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=trial_log_dir,
        name=f"trial_{trial.number}"
    )
    
    # Create Optuna callback for pruning
    enable_pruning = sweep_config.get('enable_pruning', True)
    pruning_patience = sweep_config.get('pruning_patience', 3)
    callbacks = []
    
    if enable_pruning:
        optuna_callback = OptunaPruningCallback(
            trial=trial,
            monitor='val_loss',
            patience=pruning_patience
        )
        callbacks.append(optuna_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs_per_trial,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks if callbacks else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable to reduce output
        precision='16-mixed' if torch.cuda.is_available() else '32',
        gradient_clip_val=config.get('training', {}).get('gradient_clip_val', None),
        enable_checkpointing=False  # Disable checkpointing during sweep
    )
    
    # Train the model
    try:
        trainer.fit(model, train_loader, val_loader)
        
        # Get best validation loss from trainer
        best_val_loss = trainer.callback_metrics.get('val_loss')
        if best_val_loss is None:
            # Try to get from logged metrics
            best_val_loss = trainer.logged_metrics.get('val_loss')
        
        if best_val_loss is not None:
            if hasattr(best_val_loss, 'item'):
                best_val_loss = best_val_loss.item()
            else:
                best_val_loss = float(best_val_loss)
        else:
            # Fallback: use a high value if we can't find the metric
            print("Warning: Could not find val_loss metric. Using default value.")
            best_val_loss = float('inf')
        
        print(f"Trial {trial.number} completed. Best validation loss: {best_val_loss:.6f}")
        return best_val_loss
        
    except optuna.exceptions.TrialPruned:
        # Trial was pruned - return current best or a high value
        print(f"Trial {trial.number} was pruned.")
        raise  # Re-raise to let Optuna handle it properly


def create_study(sweep_config, storage=None):
    """
    Create an Optuna study for learning rate optimization.
    
    Args:
        sweep_config (dict): LR sweep configuration
        storage (str, optional): Storage URL for study persistence (e.g., 'sqlite:///study.db')
    
    Returns:
        optuna.Study: Created study object
    """
    study_name = sweep_config.get('study_name', 'videomae_lr_sweep')
    direction = sweep_config.get('direction', 'minimize')
    
    # Create sampler (TPE is good for continuous hyperparameters)
    sampler = TPESampler(seed=sweep_config.get('seed', 42))
    
    # Create pruner (MedianPruner stops unpromising trials)
    enable_pruning = sweep_config.get('enable_pruning', True)
    if enable_pruning:
        pruner = MedianPruner(
            n_startup_trials=2,  # Don't prune first 2 trials
            n_warmup_steps=2,  # Don't prune first 2 steps
            interval_steps=1  # Check pruning every step
        )
    else:
        pruner = None
    
    # Create study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True  # Resume if study already exists
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
    
    return study


def save_study_results(study, output_dir, sweep_config):
    """
    Save Optuna study results to files.
    
    Args:
        study (optuna.Study): The completed study
        output_dir (str): Directory to save results
        sweep_config (dict): LR sweep configuration
    """
    # Save best parameters
    best_params = save_study_results_util(study, output_dir, sweep_config)
    
    print(f"\nBest learning rate: {best_params['best_learning_rate']:.6e}")
    print(f"Best validation loss: {best_params['best_value']:.6f}")
    print(f"Best trial number: {best_params['best_trial_number']}")
    
    # Save all trial results to CSV
    save_trials_csv(study, output_dir)
    
    # Save best config for use in full training
    if sweep_config.get('save_best_config', True):
        base_config_path = sweep_config.get('base_config_path')
        if base_config_path and os.path.exists(base_config_path):
            save_best_config(study, base_config_path, output_dir)
    
    # Generate plots if requested
    if sweep_config.get('generate_plots', True):
        try:
            generate_plots(study, output_dir)
        except ImportError:
            print("Warning: Plotting requires plotly and kaleido. Install with: pip install plotly kaleido")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


def main():
    """Main function for learning rate sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Learning Rate Sweep with Optuna')
    parser.add_argument(
        '--sweep_config',
        type=str,
        required=True,
        help='Path to LR sweep configuration YAML file'
    )
    parser.add_argument(
        '--base_config',
        type=str,
        default=None,
        help='Path to base training configuration YAML file (optional, can be in sweep_config)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume existing study if available'
    )
    
    args = parser.parse_args()
    
    # Load sweep configuration
    if not os.path.exists(args.sweep_config):
        raise FileNotFoundError(f"Sweep configuration file not found: {args.sweep_config}")
    
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)['lr_sweep']
    
    # Load base configuration
    base_config_path = args.base_config or sweep_config.get('base_config_path')
    if not base_config_path:
        raise ValueError("base_config_path must be specified in sweep_config or via --base_config")
    
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
    
    base_config = load_config(base_config_path)
    sweep_config['base_config_path'] = base_config_path
    
    # Create output directory
    output_dir = sweep_config.get('output_dir', 'output/lr_sweep')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup study storage
    storage = sweep_config.get('storage')
    if storage and args.resume:
        print(f"Resuming study from storage: {storage}")
    elif storage:
        print(f"Using study storage: {storage}")
    
    # Create study
    study = create_study(sweep_config, storage=storage if storage else None)
    
    print(f"\n{'='*60}")
    print(f"Starting Learning Rate Sweep")
    print(f"{'='*60}")
    print(f"Study name: {study.study_name}")
    print(f"Number of trials: {sweep_config.get('n_trials', 10)}")
    print(f"Epochs per trial: {sweep_config.get('epochs_per_trial', 15)}")
    print(f"LR range: [{sweep_config.get('lr_min', 1e-6):.2e}, {sweep_config.get('lr_max', 1e-2):.2e}]")
    print(f"Pruning enabled: {sweep_config.get('enable_pruning', True)}")
    print(f"{'='*60}\n")
    
    # Optimize
    n_trials = sweep_config.get('n_trials', 10)
    
    try:
        study.optimize(
            lambda trial: objective(trial, base_config, sweep_config),
            n_trials=n_trials,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Save results
    print(f"\n{'='*60}")
    print("Optimization completed!")
    print(f"{'='*60}")
    save_study_results(study, output_dir, sweep_config)
    
    # Print summary
    print_study_summary(study)


if __name__ == '__main__':
    main()

