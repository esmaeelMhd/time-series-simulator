"""Structured config schema + validation for Hydra-first training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from timesim.data.validation import validate_variable_groups


@dataclass
class VariableGroupsConfig:
    control: List[str] = field(default_factory=list)
    exogenous: List[str] = field(default_factory=list)
    objective: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str = ""
    csv: str = ""
    seq_len: int = 24
    pred_len: int = 1
    batch_size: int = 32
    index_col: str = "date"
    variables: VariableGroupsConfig = field(default_factory=VariableGroupsConfig)


@dataclass
class ModelConfig:
    type: str = "latent_ssm"
    hidden_dim: int = 96
    latent_dim: int = 24
    dim_h: Optional[int] = None
    dim_z: Optional[int] = None
    min_scale: float = 0.02
    use_aux_decoder: bool = True
    use_dual_path: bool = True
    use_stochastic_path: bool = True
    share_encoder_weights: bool = False
    leak_objective_to_transition: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 50
    steps_per_epoch: Optional[int] = None
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 1e-6
    grad_clip_norm: float = 100.0
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.01
    probabilistic: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfigSchema:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data: Dict[str, Any] = field(default_factory=dict)
    model_io: Dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    training_rounds: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    serving: Dict[str, Any] = field(default_factory=dict)
    tracking: Dict[str, Any] = field(default_factory=dict)
    plotting: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    model_defaults: Dict[str, Any] = field(default_factory=dict)
    misc: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)

    # Runtime override fields (Hydra CLI)
    models: Optional[List[str]] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    device: Optional[str] = None
    use_optuna_best_params: Optional[bool] = None
    optuna_summary: Optional[str] = None


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required config section/key: '{key}'")
    return cfg[key]


def _as_groups(groups: Dict[str, Any]) -> VariableGroupsConfig:
    if not isinstance(groups, dict):
        raise ValueError("dataset.variables must be a mapping")
    data = {"control": [], "exogenous": [], "objective": []}
    data.update(groups)
    for role in ("control", "exogenous", "objective"):
        if not isinstance(data[role], list):
            raise ValueError(f"dataset.variables.{role} must be a list")
    return VariableGroupsConfig(**data)


def _as_dataset(dataset: Dict[str, Any]) -> DatasetConfig:
    groups = _as_groups(dataset.get("variables", {}))
    return DatasetConfig(
        name=str(dataset.get("name", "")),
        csv=str(dataset.get("csv", "")),
        seq_len=int(dataset.get("seq_len", 0)),
        pred_len=int(dataset.get("pred_len", 0)),
        batch_size=int(dataset.get("batch_size", 0)),
        index_col=str(dataset.get("index_col", "date")),
        variables=groups,
    )


def _as_model(model: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        type=str(model.get("type", "latent_ssm")),
        hidden_dim=int(model.get("hidden_dim", 96)),
        latent_dim=int(model.get("latent_dim", 24)),
        dim_h=int(model["dim_h"]) if model.get("dim_h") is not None else None,
        dim_z=int(model["dim_z"]) if model.get("dim_z") is not None else None,
        min_scale=float(model.get("min_scale", 0.02)),
        use_aux_decoder=bool(model.get("use_aux_decoder", True)),
        use_dual_path=bool(model.get("use_dual_path", True)),
        use_stochastic_path=bool(model.get("use_stochastic_path", True)),
        share_encoder_weights=bool(model.get("share_encoder_weights", False)),
        leak_objective_to_transition=bool(model.get("leak_objective_to_transition", False)),
    )


def _as_training(training: Dict[str, Any]) -> TrainingConfig:
    steps_raw = training.get("steps_per_epoch")
    steps = int(steps_raw) if steps_raw is not None else None
    return TrainingConfig(
        epochs=int(training.get("epochs", 0)),
        steps_per_epoch=steps,
        learning_rate=float(training.get("learning_rate", 3e-4)),
        optimizer=str(training.get("optimizer", "adamw")),
        weight_decay=float(training.get("weight_decay", 1e-6)),
        grad_clip_norm=float(training.get("grad_clip_norm", 100.0)),
        lr_warmup_steps=int(training.get("lr_warmup_steps", 1000)),
        lr_min_ratio=float(training.get("lr_min_ratio", 0.01)),
        probabilistic=dict(training.get("probabilistic", {}) or {}),
    )


def coerce_and_validate_train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate minimally required invariants for Hydra-composed configs."""
    dataset_raw = _require(cfg, "dataset")
    training_raw = _require(cfg, "training")
    model_raw = _require(cfg, "model")

    dataset = _as_dataset(dataset_raw)
    training = _as_training(training_raw)
    model = _as_model(model_raw)

    if not dataset.name:
        raise ValueError("dataset.name is required")
    if not dataset.csv:
        raise ValueError("dataset.csv is required")
    if dataset.seq_len <= 0 or dataset.pred_len <= 0:
        raise ValueError("dataset.seq_len and dataset.pred_len must be > 0")
    if dataset.batch_size <= 0:
        raise ValueError("dataset.batch_size must be > 0")

    validate_variable_groups(
        {
            "control": dataset.variables.control,
            "exogenous": dataset.variables.exogenous,
            "objective": dataset.variables.objective,
        }
    )

    if not model.type:
        raise ValueError("model.type is required")
    if training.epochs <= 0:
        raise ValueError("training.epochs must be > 0")

    # Default fallbacks expected by the trainer.
    cfg.setdefault("data", {})
    cfg.setdefault(
        "model_io",
        {"input_groups": ["control", "exogenous", "objective"], "output_groups": ["objective"]},
    )
    cfg.setdefault("training_rounds", [])
    cfg.setdefault("evaluation", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("output", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("plotting", {})
    cfg.setdefault("optimization", {})
    cfg.setdefault("model_defaults", {})
    cfg.setdefault("misc", {})
    cfg.setdefault("architecture", {})

    return cfg


__all__ = [
    "VariableGroupsConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "TrainConfigSchema",
    "coerce_and_validate_train_cfg",
]
