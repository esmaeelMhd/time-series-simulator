"""World models for time-series simulation and control.

Model Families
--------------
Primary:
   - LatentSSMWorldModel (RSSM): intervention-aware probabilistic world model

Legacy/backup:
1. Recurrent (RNN-based):
   - LSTMWorldModel: LSTM-based world model with hidden state
2. Attention-based:
   - SimpleTransformer: Transformer encoder for sequence modeling
   - TemporalFusionTransformer: Interpretable attention with variable selection

3. Linear:
   - DLinear: Decomposition-based linear (trend + seasonal)
   - NLinear: Normalization-based linear (simple but effective)

4. Tree-based:
   - XGBoostForecaster: Gradient boosting (sklearn-style fit/predict)

Usage
-----
Neural models (can use .forward() or world model interface):
>>> from timesim.models import get_model
>>> model = get_model("lstm")(input_dim=10, output_dim=3, hidden_dim=64)
>>> pred = model(x)  # or model.rollout(...)

XGBoost (sklearn-style):
>>> from timesim.models import XGBoostForecaster
>>> model = XGBoostForecaster(input_dim=10, seq_len=24, pred_len=12)
>>> model.fit(X_train, y_train)
>>> pred = model.predict(X_test)
"""

from .base import WorldModelBase
from .decoders import AuxiliaryDecoder, ObjectiveDecoder
from .dlinear import DLinear, DLinearWorldModel
from .encoders import (
    ControlEncoder,
    ExogenousEncoder,
    ObservationEncoder,
    UniversalSharedEncoder,
    assert_no_shared_encoder_params,
)
from .factory import (
    LEGACY_MODEL_TYPES,
    MODEL_TYPE_CONSTANTS,
    NEURAL_MODELS,
    PRIMARY_MODEL_TYPES,
    build_model,
    count_parameters,
    get_model_param_names,
)
from .latent_ssm import LatentSSMWorldModel
from .lstm import LSTMWorldModel, SimpleLSTM
from .nlinear import NLinear, NLinearWorldModel
from .rssm import RSSMCell, RSSMOutput, RSSMState
from .tft import TemporalFusionTransformer, TFTWorldModel
from .transformer import SimpleTransformer, TransformerWorldModel

# Optional XGBoost (may not be installed)
try:
    from .xgboost_model import XGBoostEnsemble, XGBoostForecaster
    _HAS_XGBOOST = True
except ImportError:
    XGBoostForecaster = None
    XGBoostEnsemble = None
    _HAS_XGBOOST = False


# Registry for neural models that support the WorldModelBase interface
MODEL_REGISTRY = {
    # Legacy recurrent
    "lstm": LSTMWorldModel,

    # Legacy attention
    "transformer": TransformerWorldModel,
    "tft": TFTWorldModel,

    # Legacy linear
    "dlinear": DLinearWorldModel,
    "nlinear": NLinearWorldModel,

    # Primary probabilistic architecture
    "latent_ssm": LatentSSMWorldModel,
}

# Standalone models (different interface)
STANDALONE_REGISTRY = {
    "dlinear_simple": DLinear,
    "nlinear_simple": NLinear,
    "tft_simple": TemporalFusionTransformer,
}

if _HAS_XGBOOST:
    STANDALONE_REGISTRY["xgboost"] = XGBoostForecaster
    STANDALONE_REGISTRY["xgboost_ensemble"] = XGBoostEnsemble


def get_model(name: str, world_model: bool = True):
    """Get a model class by name.
    
    Parameters
    ----------
    name : str
        Model name (case-insensitive).
    world_model : bool, default True
        If True, return WorldModelBase-compatible model.
        If False, return standalone model (may have different interface).
    
    Returns
    -------
    type
        Model class.
    
    Examples
    --------
    >>> LSTMModel = get_model("lstm")
    >>> model = LSTMModel(input_dim=10, output_dim=3)
    
    >>> DLinearSimple = get_model("dlinear_simple", world_model=False)
    >>> model = DLinearSimple(input_dim=10, seq_len=24, pred_len=12)
    """
    key = name.lower()

    if world_model:
        if key not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise KeyError(
                f"World model '{name}' not found. Available: {available}. "
                f"Use world_model=False for standalone models."
            )
        return MODEL_REGISTRY[key]
    else:
        all_models = {**MODEL_REGISTRY, **STANDALONE_REGISTRY}
        if key not in all_models:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(all_models.keys())}"
            )
        return all_models[key]


def list_models() -> dict:
    """List all available models.
    
    Returns
    -------
    dict
        Dictionary with 'world_models' and 'standalone' keys.
    """
    return {
        "world_models": list(MODEL_REGISTRY.keys()),
        "standalone": list(STANDALONE_REGISTRY.keys()),
    }


__all__ = [
    # Base
    "WorldModelBase",

    # LSTM
    "LSTMWorldModel",
    "SimpleLSTM",

    # Transformer
    "SimpleTransformer",
    "TransformerWorldModel",

    # Linear
    "DLinear",
    "DLinearWorldModel",
    "NLinear",
    "NLinearWorldModel",

    # TFT
    "TemporalFusionTransformer",
    "TFTWorldModel",

    # Probabilistic
    "LatentSSMWorldModel",
    "RSSMState",
    "RSSMOutput",
    "RSSMCell",
    "ControlEncoder",
    "ExogenousEncoder",
    "ObservationEncoder",
    "UniversalSharedEncoder",
    "assert_no_shared_encoder_params",
    "ObjectiveDecoder",
    "AuxiliaryDecoder",

    # XGBoost (optional)
    "XGBoostForecaster",
    "XGBoostEnsemble",

    # Factory
    "build_model",
    "count_parameters",
    "get_model_param_names",
    "NEURAL_MODELS",
    "PRIMARY_MODEL_TYPES",
    "LEGACY_MODEL_TYPES",
    "MODEL_TYPE_CONSTANTS",

    # Utilities
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
    "STANDALONE_REGISTRY",
]
