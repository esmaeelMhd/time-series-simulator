"""Hydra structured config helpers."""

from .structured import (
    DatasetConfig,
    ModelConfig,
    TrainConfigSchema,
    TrainingConfig,
    VariableGroupsConfig,
    coerce_and_validate_train_cfg,
)

__all__ = [
    "VariableGroupsConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "TrainConfigSchema",
    "coerce_and_validate_train_cfg",
]
