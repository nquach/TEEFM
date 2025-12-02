"""
Schedule-Free Optimizer Integration

This module provides integration for the AdamWScheduleFree optimizer from the
schedule-free package for use with PyTorch Lightning.
"""

try:
    from schedulefree import AdamWScheduleFree
    SCHEDULE_FREE_AVAILABLE = True
except ImportError:
    SCHEDULE_FREE_AVAILABLE = False
    print("Warning: schedule-free package not found. Install with: pip install schedule-free")


def create_schedule_free_optimizer(model, lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8):
    """
    Create an AdamWScheduleFree optimizer for the model.
    
    The AdamWScheduleFree optimizer eliminates the need for learning rate scheduling
    by using a schedule-free approach. This simplifies training configuration.
    
    Args:
        model (torch.nn.Module): Model to optimize
        lr (float): Learning rate (default: 1.5e-4)
        weight_decay (float): Weight decay coefficient (default: 0.05)
        betas (tuple): Beta parameters for Adam optimizer (default: (0.9, 0.95))
        eps (float): Epsilon for numerical stability (default: 1e-8)
    
    Returns:
        torch.optim.Optimizer: AdamWScheduleFree optimizer instance
    
    Raises:
        ImportError: If schedule-free package is not installed
    """
    if not SCHEDULE_FREE_AVAILABLE:
        raise ImportError(
            "schedule-free package is required but not installed. "
            "Install with: pip install schedule-free"
        )
    
    # Get model parameters
    parameters = model.parameters()
    
    # Create optimizer
    optimizer = AdamWScheduleFree(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )
    
    return optimizer

