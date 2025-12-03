"""
Schedule-Free Optimizer Integration

This module provides integration for schedule-free optimizers (AdamWScheduleFree and
RAdamScheduleFree) from the schedule-free package for use with PyTorch Lightning.
"""

try:
    from schedulefree import AdamWScheduleFree, RAdamScheduleFree
    SCHEDULE_FREE_AVAILABLE = True
    # Export for use in other modules
    __all__ = ['AdamWScheduleFree', 'RAdamScheduleFree', 'SCHEDULE_FREE_AVAILABLE', 'create_schedule_free_optimizer']
except ImportError:
    SCHEDULE_FREE_AVAILABLE = False
    AdamWScheduleFree = None
    RAdamScheduleFree = None
    print("Warning: schedule-free package not found. Install with: pip install schedule-free")
    __all__ = ['SCHEDULE_FREE_AVAILABLE', 'create_schedule_free_optimizer']


def create_schedule_free_optimizer(
    model,
    optimizer_type='adamw',
    lr=1.5e-4,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    eps=1e-8
):
    """
    Create a schedule-free optimizer for the model.
    
    The schedule-free optimizers eliminate the need for learning rate scheduling
    by using a schedule-free approach. This simplifies training configuration.
    
    Supported optimizers:
    - 'adamw': AdamWScheduleFree (default)
    - 'radam': RAdamScheduleFree
    
    Args:
        model (torch.nn.Module): Model to optimize
        optimizer_type (str): Type of optimizer - 'adamw' or 'radam' (default: 'adamw')
        lr (float): Learning rate (default: 1.5e-4)
        weight_decay (float): Weight decay coefficient (default: 0.05)
        betas (tuple): Beta parameters for Adam-based optimizers (default: (0.9, 0.95))
                      Note: RAdam may use different default betas
        eps (float): Epsilon for numerical stability (default: 1e-8)
    
    Returns:
        torch.optim.Optimizer: Schedule-free optimizer instance
    
    Raises:
        ImportError: If schedule-free package is not installed
        ValueError: If optimizer_type is not supported
    """
    if not SCHEDULE_FREE_AVAILABLE:
        raise ImportError(
            "schedule-free package is required but not installed. "
            "Install with: pip install schedule-free"
        )
    
    # Get model parameters
    parameters = model.parameters()
    
    # Normalize optimizer type to lowercase
    optimizer_type = optimizer_type.lower()
    
    # Create optimizer based on type
    if optimizer_type == 'adamw':
        optimizer = AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == 'radam':
        optimizer = RAdamScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    else:
        raise ValueError(
            f"Unsupported optimizer_type: {optimizer_type}. "
            "Must be 'adamw' or 'radam'"
        )
    
    return optimizer

