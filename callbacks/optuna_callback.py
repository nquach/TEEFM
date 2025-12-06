"""
Optuna Callback for PyTorch Lightning

This module provides a PyTorch Lightning callback for integrating Optuna
hyperparameter optimization with training. It reports metrics and handles pruning.
"""

import optuna
from optuna.exceptions import TrialPruned
import pytorch_lightning as pl


class OptunaPruningCallback(pl.Callback):
    """
    PyTorch Lightning callback for Optuna integration.
    
    This callback reports validation metrics to Optuna and handles trial pruning.
    It should be used with Optuna's study.optimize() method.
    
    Args:
        trial (optuna.Trial): The Optuna trial object
        monitor (str): Metric to monitor for pruning (default: 'val_loss')
        patience (int): Number of epochs to wait before pruning (default: 3)
    """
    
    def __init__(self, trial, monitor='val_loss', patience=3):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.best_score = None
        self.no_improvement_count = 0
        self.current_epoch = 0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch.
        Reports metrics to Optuna and handles pruning.
        """
        # Get the monitored metric value
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            # If metric not found, try with 'val_' prefix
            metric_key = f'val_{self.monitor}' if not self.monitor.startswith('val_') else self.monitor
            if metric_key not in metrics:
                print(f"Warning: Metric '{self.monitor}' not found in callback_metrics. Available: {list(metrics.keys())}")
                return
        
        # Get the metric value
        metric_key = self.monitor if self.monitor in metrics else f'val_{self.monitor}'
        score = float(metrics[metric_key].item() if hasattr(metrics[metric_key], 'item') else metrics[metric_key])
        
        # Report to Optuna
        self.trial.report(score, step=self.current_epoch)
        
        # Check for improvement
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            print(f"Trial {self.trial.number} pruned at epoch {self.current_epoch} (score: {score:.4f})")
            raise TrialPruned()
        
        # Increment epoch counter
        self.current_epoch += 1
    
    def on_train_start(self, trainer, pl_module):
        """Called at the beginning of training."""
        self.current_epoch = 0
        self.best_score = None
        self.no_improvement_count = 0

