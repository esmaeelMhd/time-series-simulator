"""Legacy/backup model namespace.

These models remain available for ablations and backward compatibility while
RSSM (`latent_ssm`) is the primary architecture.
"""

from __future__ import annotations

from ..dlinear import DLinearWorldModel
from ..lstm import LSTMWorldModel
from ..nlinear import NLinearWorldModel
from ..tft import TFTWorldModel
from ..transformer import TransformerWorldModel

try:
    from ..xgboost_model import XGBoostForecaster
except Exception:  # pragma: no cover - optional dependency
    XGBoostForecaster = None

__all__ = [
    "LSTMWorldModel",
    "DLinearWorldModel",
    "NLinearWorldModel",
    "TFTWorldModel",
    "TransformerWorldModel",
    "XGBoostForecaster",
]

