"""DEPRECATED: Use timesim.training.trainer instead.

This module is kept for backward compatibility.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "timesim.engine.trainer is deprecated. Use timesim.training.trainer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..training.trainer import Trainer

__all__ = ["Trainer"] 