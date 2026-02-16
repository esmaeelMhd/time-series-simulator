"""Model factory -- single place to instantiate any supported model.

Used by training and comparison scripts to avoid duplicating the
model-construction logic.

The factory merges three sources of parameters (highest priority last):

1. ``model_defaults`` from config (global fallbacks)
2. Per-model entry in ``config["models"]``
3. Explicit ``overrides`` dict passed at call site
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Lazy imports to avoid circular dependency with __init__.py
# (factory is imported by __init__ which is still loading MODEL_REGISTRY)
_HAS_XGBOOST: bool | None = None


def _check_xgboost() -> bool:
    global _HAS_XGBOOST
    if _HAS_XGBOOST is None:
        try:
            from .xgboost_model import XGBoostForecaster  # noqa: F401
            _HAS_XGBOOST = True
        except ImportError:
            _HAS_XGBOOST = False
    return _HAS_XGBOOST

# Which model types are neural (PyTorch) vs tree-based
NEURAL_MODELS = {"lstm", "dlinear", "nlinear", "tft", "transformer", "latent_ssm"}

# Hard-coded fallbacks used when *neither* config nor caller supplies a value.
# These mirror the constructor defaults of each model class.
_BUILTIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.0,
    },
    "dlinear": {
        "kernel_size": 25,
        "individual": False,
    },
    "nlinear": {
        "individual": False,
    },
    "tft": {
        "hidden_dim": 64,
        "n_heads": 4,
        "num_lstm_layers": 2,
        "dropout": 0.1,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "latent_ssm": {
        "hidden_dim": 64,
        "latent_dim": 16,
        "num_layers": 1,
        "dropout": 0.1,
        "min_scale": 1e-4,
        "min_df": 2.1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "early_stopping_rounds": 10,
        "objective": "reg:squarederror",
    },
}


def _merge_params(
    model_type: str,
    model_defaults_cfg: Dict[str, Dict[str, Any]],
    per_model_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge parameter sources: builtin < config model_defaults < per-model < overrides."""
    merged: Dict[str, Any] = {}
    merged.update(_BUILTIN_DEFAULTS.get(model_type, {}))
    merged.update(model_defaults_cfg.get(model_type, {}))
    # per_model_cfg may contain "type" key -- skip it
    merged.update({k: v for k, v in per_model_cfg.items() if k != "type"})
    if overrides:
        merged.update(overrides)
    return merged


def build_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    seq_len: int,
    pred_len: int,
    per_model_cfg: Optional[Dict[str, Any]] = None,
    model_defaults_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Instantiate a model by type name.

    Parameters
    ----------
    model_type : str
        One of ``lstm``, ``dlinear``, ``nlinear``, ``tft``,
        ``transformer``, ``latent_ssm``, ``xgboost``.
    input_dim, output_dim, seq_len, pred_len : int
        Dimensions derived from the dataset.
    per_model_cfg : dict, optional
        The per-model entry from ``config["models"]`` (may include ``type``).
    model_defaults_cfg : dict, optional
        The ``config["model_defaults"]`` section (keyed by model type).
    overrides : dict, optional
        Any last-minute overrides (e.g. from CLI).
    """
    per_model_cfg = per_model_cfg or {}
    model_defaults_cfg = model_defaults_cfg or {}

    p = _merge_params(model_type, model_defaults_cfg, per_model_cfg, overrides)

    if model_type == "lstm":
        from .lstm import LSTMWorldModel
        return LSTMWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=p.get("hidden_dim", 64),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.0),
            pred_len=pred_len,
        )

    elif model_type == "dlinear":
        from .dlinear import DLinearWorldModel
        return DLinearWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=p.get("kernel_size", 25),
            individual=p.get("individual", False),
        )

    elif model_type == "nlinear":
        from .nlinear import NLinearWorldModel
        return NLinearWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=p.get("individual", False),
        )

    elif model_type == "tft":
        from .tft import TFTWorldModel
        return TFTWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=p.get("hidden_dim", 64),
            n_heads=p.get("n_heads", 4),
            num_lstm_layers=p.get("num_lstm_layers", 2),
            dropout=p.get("dropout", 0.1),
        )

    elif model_type == "transformer":
        from .transformer import TransformerWorldModel
        return TransformerWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=p.get("d_model", 64),
            nhead=p.get("nhead", 4),
            num_layers=p.get("num_layers", 2),
            dim_feedforward=p.get("dim_feedforward", 128),
            dropout=p.get("dropout", 0.1),
        )

    elif model_type == "latent_ssm":
        from .latent_ssm import LatentSSMWorldModel
        return LatentSSMWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=p.get("hidden_dim", 64),
            latent_dim=p.get("latent_dim", 16),
            num_layers=p.get("num_layers", 1),
            dropout=p.get("dropout", 0.1),
            pred_len=pred_len,
            min_scale=p.get("min_scale", 1e-4),
            min_df=p.get("min_df", 2.1),
        )

    elif model_type == "xgboost":
        if not _check_xgboost():
            raise ImportError("xgboost is not installed.  pip install xgboost")
        from .xgboost_model import XGBoostForecaster
        return XGBoostForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            output_dim=output_dim,
            strategy=p.get("strategy", "recursive"),
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'.  "
            f"Available neural: {sorted(NEURAL_MODELS)}, tree: ['xgboost']"
        )


def count_parameters(model) -> int:
    """Count trainable parameters (returns 0 for non-PyTorch models)."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0
