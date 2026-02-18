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
    model=None,
    optimizer_type='adamw',
    lr=1.5e-4,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup_steps=0,
    param_groups=None
):
    """
    Create a schedule-free optimizer for the model or for given parameter groups.
    
    The schedule-free optimizers eliminate the need for learning rate scheduling
    by using a schedule-free approach. This simplifies training configuration.
    
    When param_groups is provided (e.g. for layer-wise LR decay), each group must
    have 'params', 'lr', and 'weight_decay'; model/lr/weight_decay are ignored.
    When param_groups is None, model is required and a single lr/weight_decay apply.
    
    Supported optimizers:
    - 'adamw': AdamWScheduleFree (default) - supports warmup_steps
    - 'radam': RAdamScheduleFree
    
    Args:
        model (torch.nn.Module, optional): Model to optimize. Required if param_groups is None.
        optimizer_type (str): Type of optimizer - 'adamw' or 'radam' (default: 'adamw')
        lr (float): Learning rate (default: 1.5e-4). Ignored if param_groups is not None.
        weight_decay (float): Weight decay coefficient (default: 0.05). Ignored if param_groups is not None.
        betas (tuple): Beta parameters for Adam-based optimizers (default: (0.9, 0.95))
        eps (float): Epsilon for numerical stability (default: 1e-8)
        warmup_steps (int): Number of warmup steps for AdamWScheduleFree (default: 0).
                           Only applies to 'adamw' optimizer type. Set to 0 to disable warmup.
        param_groups (list of dict, optional): Parameter groups, each with 'params', 'lr', 'weight_decay'.
                                                If set, model/lr/weight_decay are ignored.
    
    Returns:
        torch.optim.Optimizer: Schedule-free optimizer instance
    
    Raises:
        ImportError: If schedule-free package is not installed
        ValueError: If optimizer_type is not supported, warmup_steps is invalid, or neither model nor param_groups provided
    """
    if not SCHEDULE_FREE_AVAILABLE:
        raise ImportError(
            "schedule-free package is required but not installed. "
            "Install with: pip install schedule-free"
        )
    
    if param_groups is None and model is None:
        raise ValueError("Either model or param_groups must be provided")
    
    # Validate warmup_steps
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    
    if param_groups is not None:
        parameters = param_groups
        use_param_groups = True
    else:
        parameters = model.parameters()
        use_param_groups = False
    
    # Normalize optimizer type to lowercase
    optimizer_type = optimizer_type.lower()
    
    # Create optimizer based on type
    if optimizer_type == 'adamw':
        if use_param_groups:
            optimizer = AdamWScheduleFree(
                parameters,
                betas=betas,
                eps=eps,
                warmup_steps=warmup_steps
            )
        else:
            optimizer = AdamWScheduleFree(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                warmup_steps=warmup_steps
            )
        if warmup_steps > 0 and not use_param_groups:
            print(f"AdamWScheduleFree initialized with {warmup_steps} warmup steps")
    elif optimizer_type == 'radam':
        if warmup_steps > 0:
            print(f"Warning: warmup_steps={warmup_steps} specified but RAdamScheduleFree "
                  "does not support warmup. Ignoring warmup_steps parameter.")
        if use_param_groups:
            optimizer = RAdamScheduleFree(
                parameters,
                betas=betas,
                eps=eps
            )
        else:
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

