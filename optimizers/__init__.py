"""
Optimizers package for VideoMAE training.
"""

from .schedule_free_optimizer import (
    create_schedule_free_optimizer,
    SCHEDULE_FREE_AVAILABLE
)

# Try to import optimizer classes if available
try:
    from .schedule_free_optimizer import AdamWScheduleFree, RAdamScheduleFree
    __all__ = [
        'create_schedule_free_optimizer',
        'SCHEDULE_FREE_AVAILABLE',
        'AdamWScheduleFree',
        'RAdamScheduleFree'
    ]
except ImportError:
    __all__ = [
        'create_schedule_free_optimizer',
        'SCHEDULE_FREE_AVAILABLE'
    ]

