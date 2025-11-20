"""World models for time-series simulation and control."""

from .base import WorldModelBase
from .lstm import LSTMWorldModel, SimpleLSTM
from .transformer import SimpleTransformer

MODEL_REGISTRY = {
    "lstm": LSTMWorldModel,
    "transformer": SimpleTransformer,
}


def get_model(name: str):
    """Get a model class by name.
    
    Parameters
    ----------
    name : str
        Model name (case-insensitive).
    
    Returns
    -------
    type
        Model class.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model {name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


__all__ = [
    "WorldModelBase",
    "LSTMWorldModel",
    "SimpleLSTM",
    "SimpleTransformer",
    "get_model",
    "MODEL_REGISTRY",
]
