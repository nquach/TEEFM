"""
Optuna Utility Functions

This module provides utility functions for working with Optuna studies,
including saving/loading results, generating visualizations, and extracting
best hyperparameters.
"""

import os
import json
import yaml
import optuna
from typing import Optional, Dict, Any
import pandas as pd


def save_study_results(study: optuna.Study, output_dir: str, config: Dict[str, Any]) -> None:
    """
    Save Optuna study results to files.
    
    Args:
        study: The completed Optuna study
        output_dir: Directory to save results
        config: Configuration dictionary with save options
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params = {
        'best_learning_rate': study.best_params.get('learning_rate'),
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    }
    
    best_params_path = os.path.join(output_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters saved to: {best_params_path}")
    
    return best_params


def save_trials_csv(study: optuna.Study, output_dir: str) -> str:
    """
    Save all trial results to a CSV file.
    
    Args:
        study: The Optuna study
        output_dir: Directory to save CSV file
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        
        # Add all parameters
        for param_name, param_value in trial.params.items():
            trial_data[f'param_{param_name}'] = param_value
        
        # Add all user attributes
        for attr_name, attr_value in trial.user_attrs.items():
            trial_data[f'attr_{attr_name}'] = attr_value
        
        trials_data.append(trial_data)
    
    df = pd.DataFrame(trials_data)
    csv_path = os.path.join(output_dir, 'trials_results.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"All trial results saved to: {csv_path}")
    return csv_path


def save_best_config(study: optuna.Study, base_config_path: str, output_dir: str) -> str:
    """
    Save best configuration with optimal hyperparameters.
    
    Args:
        study: The Optuna study
        base_config_path: Path to base configuration file
        output_dir: Directory to save best config
    
    Returns:
        str: Path to saved best config file
    """
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    # Load base config
    with open(base_config_path, 'r') as f:
        best_config = yaml.safe_load(f)
    
    # Update with best hyperparameters
    best_config['optimizer']['lr'] = study.best_params.get('learning_rate')
    
    # Save best config
    os.makedirs(output_dir, exist_ok=True)
    best_config_path_out = os.path.join(output_dir, 'best_config.yaml')
    with open(best_config_path_out, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Best configuration saved to: {best_config_path_out}")
    return best_config_path_out


def generate_plots(study: optuna.Study, output_dir: str) -> Dict[str, str]:
    """
    Generate visualization plots for the Optuna study.
    
    Args:
        study: The Optuna study
        output_dir: Directory to save plots
    
    Returns:
        dict: Dictionary mapping plot names to file paths
    """
    try:
        import optuna.visualization as vis
    except ImportError:
        raise ImportError(
            "Plotting requires optuna[visualization]. "
            "Install with: pip install 'optuna[visualization]'"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    # Optimization history plot
    try:
        fig = vis.plot_optimization_history(study)
        history_path = os.path.join(output_dir, 'optimization_history.png')
        fig.write_image(history_path)
        plot_paths['optimization_history'] = history_path
        print(f"Optimization history plot saved to: {history_path}")
    except Exception as e:
        print(f"Warning: Could not generate optimization history plot: {e}")
    
    # Parameter importance plot
    try:
        if len(study.trials) > 1:  # Need at least 2 trials
            fig = vis.plot_param_importances(study)
            importance_path = os.path.join(output_dir, 'param_importance.png')
            fig.write_image(importance_path)
            plot_paths['param_importance'] = importance_path
            print(f"Parameter importance plot saved to: {importance_path}")
    except Exception as e:
        print(f"Warning: Could not generate parameter importance plot: {e}")
    
    # Learning rate vs validation loss scatter plot
    try:
        if len(study.trials) > 0:
            fig = vis.plot_parallel_coordinate(study)
            parallel_path = os.path.join(output_dir, 'parallel_coordinate.png')
            fig.write_image(parallel_path)
            plot_paths['parallel_coordinate'] = parallel_path
            print(f"Parallel coordinate plot saved to: {parallel_path}")
    except Exception as e:
        print(f"Warning: Could not generate parallel coordinate plot: {e}")
    
    return plot_paths


def load_study_from_storage(storage: str, study_name: str) -> optuna.Study:
    """
    Load an existing Optuna study from storage.
    
    Args:
        storage: Storage URL (e.g., 'sqlite:///study.db')
        study_name: Name of the study
    
    Returns:
        optuna.Study: Loaded study
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study


def print_study_summary(study: optuna.Study) -> None:
    """
    Print a summary of the Optuna study.
    
    Args:
        study: The Optuna study
    """
    print(f"\n{'='*60}")
    print("Study Summary")
    print(f"{'='*60}")
    print(f"Study name: {study.study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"  - Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  - Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  - Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if study.best_trial:
        print(f"\nBest trial:")
        print(f"  Trial number: {study.best_trial.number}")
        print(f"  Value: {study.best_value:.6f}")
        print(f"  Params:")
        for param_name, param_value in study.best_params.items():
            print(f"    {param_name}: {param_value}")
    print(f"{'='*60}\n")

