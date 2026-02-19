"""Training infrastructure for world models."""

from .losses import (
    OneStepLoss,
    MultiStepLoss,
    CombinedLoss,
    ProbabilisticRolloutLoss,
    dilate_loss,
)
from .rollout import (
    batch_rollout,
    batch_rollout_padded,
    rollout_autoregressive,
)
from .trainer import (
    WorldModelTrainer,
    Trainer,
)
try:
    from .lightning_module import WorldModelLightningModule
except ImportError:  # optional dependency: pytorch-lightning
    WorldModelLightningModule = None  # type: ignore[assignment]
from .callbacks import get_git_metadata
from .safety import (
    merged_latent_ssm_params,
    merged_probabilistic_cfg,
    validate_latent_ssm_do_not,
)

__all__ = [
    "OneStepLoss",
    "MultiStepLoss",
    "CombinedLoss",
    "ProbabilisticRolloutLoss",
    "dilate_loss",
    "batch_rollout",
    "batch_rollout_padded",
    "rollout_autoregressive",
    "WorldModelTrainer",
    "Trainer",
    "get_git_metadata",
    "merged_latent_ssm_params",
    "merged_probabilistic_cfg",
    "validate_latent_ssm_do_not",
]

if WorldModelLightningModule is not None:
    __all__.append("WorldModelLightningModule")
