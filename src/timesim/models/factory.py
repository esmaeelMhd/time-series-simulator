"""Model factory -- single place to instantiate any supported model.

Used by training and comparison scripts to avoid duplicating the
model-construction logic.

The factory merges three sources of parameters (highest priority last):

1. ``model_defaults`` from config (global fallbacks)
2. Per-model entry in ``config["models"]``
3. Explicit ``overrides`` dict passed at call site
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
import warnings

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
PRIMARY_MODEL_TYPES = {"latent_ssm"}
LEGACY_MODEL_TYPES = {"lstm", "dlinear", "nlinear", "tft", "transformer", "xgboost"}
_WARNED_LEGACY_TYPES: set[str] = set()

# Single source of truth for model type classification
# Use these constants in configuration files to ensure consistency
MODEL_TYPE_CONSTANTS = {
    "neural_models": sorted(NEURAL_MODELS),
    "primary_model_types": sorted(PRIMARY_MODEL_TYPES),
    "legacy_model_types": sorted(LEGACY_MODEL_TYPES),
}


def get_model_param_names(model_type: str) -> set[str]:
    """Return constructor parameter names for a model type.

    This is used by CLI scripts to split model params from training params
    (e.g. when applying Optuna best params).
    """
    mt = str(model_type).lower().strip()
    cls = None
    if mt == "lstm":
        from .lstm import LSTMWorldModel

        cls = LSTMWorldModel
    elif mt == "dlinear":
        from .dlinear import DLinearWorldModel

        cls = DLinearWorldModel
    elif mt == "nlinear":
        from .nlinear import NLinearWorldModel

        cls = NLinearWorldModel
    elif mt == "tft":
        from .tft import TFTWorldModel

        cls = TFTWorldModel
    elif mt == "transformer":
        from .transformer import TransformerWorldModel

        cls = TransformerWorldModel
    elif mt == "latent_ssm":
        from .latent_ssm import LatentSSMWorldModel

        cls = LatentSSMWorldModel
    elif mt == "xgboost":
        if not _check_xgboost():
            return set()
        from .xgboost_model import XGBoostForecaster

        cls = XGBoostForecaster
    else:
        return set()

    sig = inspect.signature(cls.__init__)
    names = {
        str(name)
        for name, param in sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return names

def _merge_params(
    model_type: str,
    model_defaults_cfg: Dict[str, Dict[str, Any]],
    per_model_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge parameter sources: config model_defaults < per-model < overrides.

    Constructor defaults remain the final fallback source.
    """
    merged: Dict[str, Any] = {}
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
    control_dim: Optional[int] = None,
    exogenous_dim: Optional[int] = None,
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
    model_type = str(model_type).lower().strip()

    if model_type in LEGACY_MODEL_TYPES and model_type not in _WARNED_LEGACY_TYPES:
        warnings.warn(
            (
                f"Model '{model_type}' is in legacy/backup mode. "
                "RSSM ('latent_ssm') is the primary architecture."
            ),
            UserWarning,
            stacklevel=2,
        )
        _WARNED_LEGACY_TYPES.add(model_type)

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
            min_scale=p.get("min_scale", 0.5),
            min_std=p.get("min_std", p.get("min_scale", 0.5)),
            max_std=p.get("max_std", p.get("decoder_max_std", 2.0)),
            decoder_min_std=p.get("decoder_min_std", p.get("min_std", p.get("min_scale", 0.5))),
            decoder_max_std=p.get("decoder_max_std", p.get("max_std", 2.0)),
            prior_min_std=p.get("prior_min_std", 0.1),
            prior_max_std=p.get("prior_max_std", 1.5),
            posterior_min_std=p.get("posterior_min_std", 0.1),
            posterior_max_std=p.get("posterior_max_std", 1.5),
            prior_constant_std=p.get("prior_constant_std"),
            posterior_constant_std=p.get("posterior_constant_std"),
            min_df=p.get("min_df", 2.1),
            control_dim=p.get("control_dim", control_dim),
            exogenous_dim=p.get("exogenous_dim", exogenous_dim),
            encoder_dim=p.get("encoder_dim", 64),
            decoder_layers=p.get("decoder_layers", 2),
            use_symlog=p.get("use_symlog", False),
            use_aux_decoder=p.get("use_aux_decoder", True),
            predict_exogenous=p.get("predict_exogenous", True),
            use_dual_path=p.get("use_dual_path", True),
            use_stochastic_path=p.get("use_stochastic_path", True),
            share_encoder_weights=p.get("share_encoder_weights", False),
            leak_objective_to_transition=p.get("leak_objective_to_transition", False),
            h_dropout=p.get("h_dropout", 0.0),
            decoder_hidden=p.get("decoder_hidden"),
            allow_objective_leak_for_ablation=p.get("allow_objective_leak_for_ablation", False),
            allow_disable_aux_decoder_for_ablation=p.get("allow_disable_aux_decoder_for_ablation", False),
            allow_shared_encoder_for_ablation=p.get("allow_shared_encoder_for_ablation", False),
            allow_disable_stochastic_for_ablation=p.get("allow_disable_stochastic_for_ablation", False),
            latent_distribution=p.get("latent_distribution", "gaussian"),
            stochastic_groups=p.get("stochastic_groups", 32),
            stochastic_classes=p.get("stochastic_classes", 32),
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
        try:
            from torch.nn.parameter import UninitializedParameter  # type: ignore
        except Exception:  # pragma: no cover - older torch fallback
            UninitializedParameter = tuple()  # type: ignore
        total = 0
        for p in model.parameters():
            if not getattr(p, "requires_grad", False):
                continue
            if isinstance(p, UninitializedParameter):
                continue
            total += int(p.numel())
        return total
    return 0
