"""DEPRECATED: Use timesim.training.rollout instead.

This module is kept for backward compatibility.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "timesim.engine.rollout is deprecated. Use timesim.training.rollout instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..training.rollout import rollout_autoregressive

__all__ = ["rollout_autoregressive"] 