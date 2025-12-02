"""
Optimizer factory for creating optimizers with configurable parameters.

This module provides a factory function to create the AdamWScheduleFree optimizer
used for training the VideoMAE model.
"""

from typing import Optional
import torch.nn as nn

try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    raise ImportError(
        "schedule-free package is required. Install it with: pip install schedule-free"
    )


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 0.05,
    **kwargs
) -> AdamWScheduleFree:
    """
    Create an AdamWScheduleFree optimizer for the model.
    
    AdamWScheduleFree is a schedule-free optimizer that combines the benefits
    of AdamW with automatic learning rate scheduling. It doesn't require
    a learning rate schedule and adapts automatically during training.
    
    Args:
        model: PyTorch model whose parameters will be optimized
        learning_rate: Initial learning rate (default: 1.0e-4)
        weight_decay: Weight decay coefficient (default: 0.05)
        **kwargs: Additional arguments to pass to AdamWScheduleFree
        
    Returns:
        Configured AdamWScheduleFree optimizer
        
    Example:
        >>> model = VideoMAE(...)
        >>> optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.05)
    """
    # Get all trainable parameters
    parameters = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer with specified parameters
    optimizer = AdamWScheduleFree(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        **kwargs
    )
    
    return optimizer

