#!/usr/bin/env python3
"""Train world models for time-series simulation (digital-twin).

Trains all models defined in the config using the training_rounds approach.
Supports train-only, retrain-only (with checkpoint_dir), or both.

Outputs are saved to ``<runs_dir>/<dataset>[/<run_name>]/<model>/`` with round-prefixed
artifact names (``train_checkpoint.pth``, ``retrain_loss.png``, etc.).
The scaler is saved at ``<runs_dir>/<dataset>[/<run_name>]/scaler.pkl`` for reuse by
``compare.py`` or any evaluation script.

Usage:
    python scripts/train.py --config configs/wastewater.yaml
    python scripts/train.py --config configs/wastewater.small.yaml
    python scripts/train.py --config configs/wastewater.small.yaml --models lstm dlinear
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import logging
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from timesim.utils.misc import configure_torch_defaults
configure_torch_defaults()

import matplotlib
matplotlib.use("Agg")

# Ensure Unicode-safe console output on Windows terminals (cp1252 by default).
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from timesim.utils.config import compose_config
from timesim.data.loader import load_csv_dataset, build_dataloaders_from_config
from timesim.data.schema import VariableSchema, VariableRole
from timesim.data.stamps import get_time_feature_columns
from timesim.data.sampling import (
    RandomStartFixedHorizon,
    RandomStartRandomHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
    StrideBasedSampling,
)
from timesim.training import WorldModelTrainer
from timesim.training.safety import (
    merged_latent_ssm_params,
    merged_probabilistic_cfg,
    validate_latent_ssm_do_not,
)
from timesim.utils.plotting import save_loss_plot, save_forecast_plot
from timesim.models.factory import (
    build_model,
    count_parameters,
    get_model_param_names,
    NEURAL_MODELS,
)
from timesim.utils.tracking import ExperimentTracker
from timesim.utils.misc import seed_everything, resolve_device

# Shared eval / simulation utilities  (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (
    evaluate_neural_model,
    evaluate_xgboost_model,
    simulate_recursive_neural,
    simulate_recursive_xgboost,
    save_per_model_simulation_plot,
    save_per_model_simulation_csv,
)

# Try importing XGBoost
try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


TRAINING_PARAM_KEYS = {
    "learning_rate",
    "optimizer",
    "weight_decay",
    "loss_type",
    "loss_weighting",
    "loss_weight_scale",
    "mode",
    "feedback",
    "teacher_forcing_ratio",
    "one_step_weight",
    "recon_weight",
    "elbo_weight",
    "kl_weight",
    "aux_weight",
    "rollout_mse_weight",
    "rollout_weight",
    "rollout_dtw_weight",
    "rollout_dtw_gamma",
    "rollout_warmup_fraction",
    "rollout_max_horizon",
    "min_context",
    "kl_free_bits",
    "kl_balance",
    "use_kl_balancing",
    "use_free_bits",
    "use_symlog",
    "grad_clip_norm",
    "lr_warmup_steps",
    "lr_min_ratio",
    "checkpoint_top_k",
    "early_stopping_monitor",
    "objective",
    "kl_warmup_enabled",
    "kl_beta_end",
    "checkpoint_metric",
    "checkpoint_open_loop_horizon",
    "checkpoint_open_loop_windows",
    "checkpoint_open_loop_samples",
    "engine",
    "allow_engine_fallback",
}


# ─────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────

def build_sampling_strategy(config, pred_len):
    """Build sampling strategy from training config."""
    tcfg = config["training"]
    dcfg = config["dataset"]
    scfg = tcfg.get("sampling", {})
    if "sampling_strategy" in tcfg or "sampling_horizon" in tcfg:
        raise ValueError(
            "Deprecated sampling keys detected in training config. "
            "Use `training.sampling.strategy` and `training.sampling.horizon`."
        )

    strategy_name = str(scfg.get("strategy", "random_fixed")).lower()
    default_horizon = int(scfg.get("horizon", pred_len))

    if strategy_name in {"random_fixed", "fixed"}:
        horizon = int(scfg.get("horizon", default_horizon))
        return RandomStartFixedHorizon(horizon=horizon), f"random_fixed(horizon={horizon})"

    if strategy_name in {"random_random", "random"}:
        h_min = int(scfg.get("h_min", 1))
        h_max = int(scfg.get("h_max", default_horizon))
        return RandomStartRandomHorizon(h_min=h_min, h_max=h_max), (
            f"random_random(h_min={h_min}, h_max={h_max})"
        )

    if strategy_name in {"geometric", "geometric_horizon"}:
        h_max = int(scfg.get("h_max", default_horizon))
        return GeometricHorizonSampling(pred_len=pred_len, h_max=h_max), (
            f"geometric(pred_len={pred_len}, h_max={h_max})"
        )

    if strategy_name in {"daily_fixed", "daily"}:
        start_hour = int(scfg.get("start_hour", 0))
        horizon = int(scfg.get("horizon", default_horizon))
        samples_per_hour = int(scfg.get("samples_per_hour", dcfg.get("samples_per_hour", 1)))
        return DailyFixedHorizon(
            start_hour=start_hour,
            horizon=horizon,
            samples_per_hour=samples_per_hour,
        ), (
            "daily_fixed("
            f"start_hour={start_hour}, horizon={horizon}, "
            f"samples_per_hour={samples_per_hour})"
        )

    if strategy_name in {"stride", "stride_based"}:
        stride = int(scfg.get("stride", 12))
        h_max = int(scfg.get("h_max", default_horizon))
        return StrideBasedSampling(stride=stride, h_max=h_max), (
            f"stride(stride={stride}, h_max={h_max})"
        )

    raise ValueError(
        f"Unknown sampling strategy '{strategy_name}'. "
        "Use one of: random_fixed, random_random, geometric, daily_fixed, stride"
    )


class _RoleWindowDatasetFromGrouped(Dataset):
    """Role-batch dataset adapter for Lightning from GroupedTimeSeriesDataset values."""

    def __init__(
        self,
        grouped_dataset,
        *,
        seq_len: int,
        stride: int = 1,
    ):
        self.values = np.asarray(grouped_dataset.values, dtype=np.float32)
        self.seq_len = int(max(1, seq_len))
        self.stride = int(max(1, stride))
        schema = getattr(grouped_dataset, "variable_schema", None)
        feat_cols = list(getattr(grouped_dataset, "feature_cols", []))
        if schema is None or not feat_cols:
            raise ValueError("Grouped dataset missing variable schema / feature columns for Lightning adapter.")

        control_cols = list(schema.columns_for_role(VariableRole.CONTROL))
        exo_cols = list(schema.columns_for_role(VariableRole.EXOGENOUS))
        obj_cols = list(schema.columns_for_role(VariableRole.OBJECTIVE))
        self.control_idx = [feat_cols.index(c) for c in control_cols if c in feat_cols]
        self.exo_idx = [feat_cols.index(c) for c in exo_cols if c in feat_cols]
        self.obj_idx = [feat_cols.index(c) for c in obj_cols if c in feat_cols]
        if not self.obj_idx:
            raise ValueError("No objective columns found for Lightning role-batch adapter.")

    def __len__(self) -> int:
        max_start = len(self.values) - self.seq_len
        if max_start < 0:
            return 0
        return (max_start // self.stride) + 1

    def __getitem__(self, idx: int):
        start = int(idx) * self.stride
        w = self.values[start : start + self.seq_len]
        return {
            "control": torch.tensor(w[:, self.control_idx], dtype=torch.float32),
            "exogenous": torch.tensor(w[:, self.exo_idx], dtype=torch.float32),
            "objective": torch.tensor(w[:, self.obj_idx], dtype=torch.float32),
        }



def train_neural_model(model, train_dataset, val_dataset, config, device,
                       model_dir, lr_override=None, epochs_override=None,
                       steps_per_epoch_override=None,
                       optimizer_override=None,
                       training_overrides=None,
                       checkpoint_path=None,
                       checkpoint_callback: Optional[Callable[[int, float], None]] = None):
    """Train a neural WorldModel. Returns (train_losses, val_losses)."""
    model_dir.mkdir(parents=True, exist_ok=True)

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    tcfg = config["training"]
    training_overrides = training_overrides or {}
    epochs = epochs_override or tcfg["epochs"]
    steps_per_epoch = (
        steps_per_epoch_override
        if steps_per_epoch_override is not None
        else tcfg.get("steps_per_epoch", None)
    )
    batch_size = config["dataset"]["batch_size"]
    lr = lr_override if lr_override is not None else tcfg.get("learning_rate", 1e-3)
    warmup_len = tcfg.get("window_len", tcfg.get("warmup_len", seq_len))
    sampling, sampling_desc = build_sampling_strategy(config, pred_len)
    loss_type = training_overrides.get("loss_type", tcfg.get("loss_type", "mse"))
    shape_loss_cfg = tcfg.get("shape_loss", None)
    loss_weighting = training_overrides.get("loss_weighting", tcfg.get("loss_weighting", "uniform"))
    loss_weight_scale = training_overrides.get("loss_weight_scale", tcfg.get("loss_weight_scale", 1.0))
    training_mode = training_overrides.get("mode", tcfg.get("mode", "multi_step"))
    feedback = training_overrides.get("feedback", tcfg.get("feedback", "model"))
    teacher_forcing_ratio = training_overrides.get(
        "teacher_forcing_ratio", tcfg.get("teacher_forcing_ratio", 0.0)
    )
    one_step_weight = training_overrides.get("one_step_weight", tcfg.get("one_step_weight", 0.5))
    use_amp = bool(tcfg.get("use_amp", False))
    prob_cfg = dict(tcfg.get("probabilistic", {}) or {})
    if "recon_weight" in training_overrides:
        prob_cfg["recon_weight"] = training_overrides["recon_weight"]
    if "elbo_weight" in training_overrides:
        prob_cfg["elbo_weight"] = training_overrides["elbo_weight"]
    if "kl_weight" in training_overrides:
        prob_cfg["kl_weight"] = training_overrides["kl_weight"]
    if "aux_weight" in training_overrides:
        prob_cfg["aux_weight"] = training_overrides["aux_weight"]
    if "rollout_mse_weight" in training_overrides:
        prob_cfg["rollout_mse_weight"] = training_overrides["rollout_mse_weight"]
    for key in [
        "rollout_weight",
        "rollout_dtw_weight",
        "rollout_dtw_gamma",
        "rollout_warmup_fraction",
        "rollout_max_horizon",
        "min_context",
        "kl_free_bits",
        "kl_balance",
        "use_kl_balancing",
        "use_free_bits",
        "use_symlog",
        "use_aux_decoder",
        "use_dual_path",
        "leak_objective_to_transition",
        "grad_clip_norm",
        "lr_warmup_steps",
        "lr_min_ratio",
        "checkpoint_top_k",
        "early_stopping_monitor",
        "checkpoint_metric",
        "checkpoint_open_loop_horizon",
        "checkpoint_open_loop_windows",
        "checkpoint_open_loop_samples",
    ]:
        if key in training_overrides:
            prob_cfg[key] = training_overrides[key]
    if "objective" in training_overrides:
        prob_cfg["objective"] = training_overrides["objective"]
    if "kl_warmup_enabled" in training_overrides:
        prob_cfg["kl_warmup_enabled"] = training_overrides["kl_warmup_enabled"]
    if "kl_beta_start" in training_overrides:
        prob_cfg["kl_beta_start"] = training_overrides["kl_beta_start"]
    if "kl_beta_end" in training_overrides:
        prob_cfg["kl_beta_end"] = training_overrides["kl_beta_end"]
    if "kl_warmup_epochs" in training_overrides:
        prob_cfg["kl_warmup_epochs"] = training_overrides["kl_warmup_epochs"]

    # Optimizer
    opt_name = str(
        optimizer_override or training_overrides.get("optimizer", tcfg.get("optimizer", "adam"))
    ).lower()
    weight_decay = float(
        training_overrides.get("weight_decay", tcfg.get("weight_decay", 0.0))
    )
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  Sampling strategy: {sampling_desc}")

    variable_schema_payload: Dict[str, Any] = {}
    schema_obj = getattr(train_dataset, "variable_schema", None)
    if schema_obj is not None and hasattr(schema_obj, "to_groups"):
        try:
            variable_schema_payload = dict(schema_obj.to_groups())
        except Exception:
            variable_schema_payload = {}

    normalization_stats_payload: Dict[str, Any] = {}
    scaler_obj = getattr(train_dataset, "scaler", None)
    if scaler_obj is not None:
        normalization_stats_payload["scaler_class"] = scaler_obj.__class__.__name__
        for attr in (
            "feature_range",
            "data_min_",
            "data_max_",
            "data_range_",
            "scale_",
            "min_",
            "n_features_in_",
        ):
            if not hasattr(scaler_obj, attr):
                continue
            val = getattr(scaler_obj, attr)
            if isinstance(val, np.ndarray):
                normalization_stats_payload[attr] = val.tolist()
            elif isinstance(val, (list, tuple)):
                normalization_stats_payload[attr] = list(val)
            elif isinstance(val, (int, float, str, bool)):
                normalization_stats_payload[attr] = val

    checkpoint_metadata = {
        "normalization_stats": normalization_stats_payload,
        "variable_schema": variable_schema_payload,
        "config": config,
    }

    engine = str(training_overrides.get("engine", tcfg.get("engine", "custom"))).lower().strip()
    if engine == "lightning":
        allow_engine_fallback = bool(
            training_overrides.get(
                "allow_engine_fallback",
                tcfg.get("allow_engine_fallback", False),
            )
        )
        try:
            import pytorch_lightning as pl  # type: ignore
            from pytorch_lightning.callbacks import (  # type: ignore
                Callback,
                EarlyStopping as PLEarlyStopping,
                ModelCheckpoint,
            )
            from pytorch_lightning.loggers import CSVLogger  # type: ignore
            from timesim.training.lightning_module import WorldModelLightningModule
            from timesim.training.health_check import TrainingHealthCheck
        except Exception as exc:
            if allow_engine_fallback:
                print(
                    "  Warning: Lightning engine requested but unavailable "
                    f"({exc}); falling back to custom."
                )
                engine = "custom"
            else:
                raise RuntimeError(
                    "Lightning engine requested but unavailable and fallback is disabled. "
                    "Install pytorch-lightning or set training.allow_engine_fallback=true."
                ) from exc
        else:
            warnings.filterwarnings(
                "ignore",
                message=r".*isinstance\(treespec, LeafSpec\).*deprecated.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*Checkpoint directory .* exists and is not empty.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*does not support `log_graph`.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*try installing \[litlogger\].*",
            )
            class _MetricCollectorCallback(Callback):  # type: ignore[misc]
                _CSV_COLUMNS = [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "loss_std",
                    "loss_total",
                    "recon_nll",
                    "kl",
                    "kl_mean",
                    "kl_raw",
                    "kl_per_dim_active",
                    "decoder_std_mean",
                    "decoder_std_min",
                    "prior_std_mean",
                    "prior_std_max",
                    "posterior_std_mean",
                    "posterior_std_max",
                    "aux_nll",
                    "horizon",
                    "context_len",
                    "rollout_nll",
                    "rollout_dtw",
                    "rollout_total",
                    "rollout_weight_eff",
                    "rollout_ramp",
                    "horizon_schedule",
                    "grad_norm",
                    "grad_norm_pre",
                    "lr",
                    "open_loop_crps",
                    "val_loss_std",
                    "val_loss_total",
                    "val_recon_nll",
                    "val_kl",
                    "val_kl_mean",
                    "val_kl_raw",
                    "val_kl_per_dim_active",
                    "val_decoder_std_mean",
                    "val_decoder_std_min",
                    "val_prior_std_mean",
                    "val_prior_std_max",
                    "val_posterior_std_mean",
                    "val_posterior_std_max",
                    "val_aux_nll",
                    "val_horizon",
                    "val_context_len",
                    "val_rollout_nll",
                    "val_rollout_dtw",
                    "val_rollout_total",
                    "val_rollout_weight_eff",
                    "val_rollout_ramp",
                    "val_horizon_schedule",
                ]

                def __init__(self, metrics_path: Path):
                    self.train_losses: list[float] = []
                    self.val_losses: list[float] = []
                    self.metrics_path = Path(metrics_path)
                    self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)
                        writer.writeheader()

                @staticmethod
                def _as_float(x) -> float:
                    try:
                        return float(x.item())
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return float("nan")

                @staticmethod
                def _metric(trainer, key: str) -> float:
                    # Lightning can expose train/val aggregates across different stores
                    # depending on hook timing and logger backend.
                    stores = [
                        getattr(trainer, "callback_metrics", {}) or {},
                        getattr(trainer, "logged_metrics", {}) or {},
                        getattr(trainer, "progress_bar_metrics", {}) or {},
                    ]
                    for metrics in stores:
                        for k in (f"{key}_epoch", key, f"{key}_step"):
                            val = metrics.get(k)
                            if val is None:
                                continue
                            out = _MetricCollectorCallback._as_float(val)
                            if np.isfinite(out):
                                return out
                    return float("nan")

                def _append_metrics_row(self, trainer) -> None:
                    row: Dict[str, Any] = {k: np.nan for k in self._CSV_COLUMNS}
                    row["epoch"] = int(trainer.current_epoch) + 1

                    # Backward-compatible columns (train-focused, plus val_loss/open_loop_crps).
                    row["train_loss"] = self._metric(trainer, "train/loss")
                    row["val_loss"] = self._metric(trainer, "val/loss")
                    row["loss_std"] = self._metric(trainer, "train/loss_std")
                    row["loss_total"] = self._metric(trainer, "train/loss_total")
                    row["recon_nll"] = self._metric(trainer, "train/recon_nll")
                    row["kl"] = self._metric(trainer, "train/kl")
                    row["kl_mean"] = self._metric(trainer, "train/kl_mean")
                    row["kl_raw"] = self._metric(trainer, "train/kl_raw")
                    row["kl_per_dim_active"] = self._metric(trainer, "train/kl_active")
                    row["decoder_std_mean"] = self._metric(trainer, "train/decoder_std_mean")
                    row["decoder_std_min"] = self._metric(trainer, "train/dec_std_min")
                    row["prior_std_mean"] = self._metric(trainer, "train/prior_std_mean")
                    row["prior_std_max"] = self._metric(trainer, "train/prior_std_max")
                    row["posterior_std_mean"] = self._metric(trainer, "train/posterior_std_mean")
                    row["posterior_std_max"] = self._metric(trainer, "train/posterior_std_max")
                    row["aux_nll"] = self._metric(trainer, "train/aux_nll")
                    row["horizon"] = self._metric(trainer, "train/horizon")
                    row["context_len"] = self._metric(trainer, "train/context_len")
                    row["rollout_nll"] = self._metric(trainer, "train/rollout_nll")
                    row["rollout_dtw"] = self._metric(trainer, "train/rollout_dtw")
                    row["rollout_total"] = self._metric(trainer, "train/rollout_total")
                    row["rollout_weight_eff"] = self._metric(trainer, "train/rollout_weight_eff")
                    row["rollout_ramp"] = self._metric(trainer, "train/rollout_ramp")
                    row["horizon_schedule"] = self._metric(trainer, "train/horizon_schedule")
                    row["grad_norm"] = self._metric(trainer, "train/grad_norm")
                    row["grad_norm_pre"] = self._metric(trainer, "train/grad_norm_pre")
                    row["lr"] = self._metric(trainer, "train/lr")
                    row["open_loop_crps"] = self._metric(trainer, "val/open_loop_crps")

                    # Validation metrics (explicit val_* columns).
                    row["val_loss_std"] = self._metric(trainer, "val/loss_std")
                    row["val_loss_total"] = self._metric(trainer, "val/loss_total")
                    row["val_recon_nll"] = self._metric(trainer, "val/recon_nll")
                    row["val_kl"] = self._metric(trainer, "val/kl")
                    row["val_kl_mean"] = self._metric(trainer, "val/kl_mean")
                    row["val_kl_raw"] = self._metric(trainer, "val/kl_raw")
                    row["val_kl_per_dim_active"] = self._metric(trainer, "val/kl_active")
                    row["val_decoder_std_mean"] = self._metric(trainer, "val/decoder_std_mean")
                    row["val_decoder_std_min"] = self._metric(trainer, "val/dec_std_min")
                    row["val_prior_std_mean"] = self._metric(trainer, "val/prior_std_mean")
                    row["val_prior_std_max"] = self._metric(trainer, "val/prior_std_max")
                    row["val_posterior_std_mean"] = self._metric(trainer, "val/posterior_std_mean")
                    row["val_posterior_std_max"] = self._metric(trainer, "val/posterior_std_max")
                    row["val_aux_nll"] = self._metric(trainer, "val/aux_nll")
                    row["val_horizon"] = self._metric(trainer, "val/horizon")
                    row["val_context_len"] = self._metric(trainer, "val/context_len")
                    row["val_rollout_nll"] = self._metric(trainer, "val/rollout_nll")
                    row["val_rollout_dtw"] = self._metric(trainer, "val/rollout_dtw")
                    row["val_rollout_total"] = self._metric(trainer, "val/rollout_total")
                    row["val_rollout_weight_eff"] = self._metric(trainer, "val/rollout_weight_eff")
                    row["val_rollout_ramp"] = self._metric(trainer, "val/rollout_ramp")
                    row["val_horizon_schedule"] = self._metric(trainer, "val/horizon_schedule")
                    if not np.isfinite(row["val_loss"]):
                        row["val_loss"] = row["val_loss_total"]

                    with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)
                        writer.writerow(row)

                def on_train_epoch_end(self, trainer, pl_module):
                    tr = self._metric(trainer, "train/loss")
                    if np.isfinite(tr):
                        self.train_losses.append(float(tr))
                
                def on_validation_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    vl = metrics.get("val/loss")
                    if vl is not None:
                        try:
                            self.val_losses.append(float(vl.item()))
                        except Exception:
                            pass
                    self._append_metrics_row(trainer)

            class _EarlyStoppingWithMinEpoch(PLEarlyStopping):  # type: ignore[misc]
                def __init__(self, min_epoch: int = 0, **kwargs):
                    super().__init__(**kwargs)
                    self.min_epoch = int(max(0, min_epoch))

                @staticmethod
                def _as_float(x) -> float:
                    try:
                        return float(x.item())
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return float("nan")

                @staticmethod
                def _metric_from_trainer(trainer, key: str) -> float:
                    stores = [
                        getattr(trainer, "callback_metrics", {}) or {},
                        getattr(trainer, "logged_metrics", {}) or {},
                        getattr(trainer, "progress_bar_metrics", {}) or {},
                    ]
                    for metrics in stores:
                        for k in (f"{key}_epoch", key, f"{key}_step"):
                            val = metrics.get(k)
                            if val is None:
                                continue
                            out = _EarlyStoppingWithMinEpoch._as_float(val)
                            if np.isfinite(out):
                                return out
                    return float("nan")

                def on_validation_end(self, trainer, pl_module):
                    if int(trainer.current_epoch) < self.min_epoch:
                        return
                    # Make sure early stopping sees the monitored metric key even when
                    # Lightning places it in logged/progress stores instead of callback_metrics.
                    monitor_key = str(getattr(self, "monitor", ""))
                    if monitor_key:
                        monitor_score = self._metric_from_trainer(trainer, monitor_key)
                        if np.isfinite(monitor_score):
                            trainer.callback_metrics[monitor_key] = torch.tensor(
                                monitor_score, device=pl_module.device
                            )
                    super().on_validation_end(trainer, pl_module)

            class _CheckpointEventBridge(Callback):  # type: ignore[misc]
                """Forward Lightning best-checkpoint events to legacy callback hook."""

                def __init__(
                    self,
                    checkpoint_cb: Optional[Callable[[int, float], None]],
                    model_ckpt_cb,
                    monitor_key: str,
                    monitor_mode: str,
                ):
                    self.checkpoint_cb = checkpoint_cb
                    self.model_ckpt_cb = model_ckpt_cb
                    self.monitor_key = str(monitor_key)
                    self.monitor_mode = str(monitor_mode).lower()
                    self._last_best_path: str = ""
                    self._best_metric = (
                        float("inf") if self.monitor_mode == "min" else float("-inf")
                    )

                @staticmethod
                def _as_float(x) -> float:
                    try:
                        return float(x.item())
                    except Exception:
                        try:
                            return float(x)
                        except Exception:
                            return float("nan")

                @staticmethod
                def _metric_from_trainer(trainer, key: str) -> float:
                    stores = [
                        getattr(trainer, "callback_metrics", {}) or {},
                        getattr(trainer, "logged_metrics", {}) or {},
                        getattr(trainer, "progress_bar_metrics", {}) or {},
                    ]
                    for metrics in stores:
                        for k in (f"{key}_epoch", key, f"{key}_step"):
                            val = metrics.get(k)
                            if val is None:
                                continue
                            out = _CheckpointEventBridge._as_float(val)
                            if np.isfinite(out):
                                return out
                    return float("nan")

                def on_validation_end(self, trainer, pl_module):
                    if self.checkpoint_cb is None:
                        return
                    # Trigger only when monitored metric actually improves.
                    score = self._metric_from_trainer(trainer, self.monitor_key)
                    improved = False
                    if np.isfinite(score):
                        if self.monitor_mode == "min":
                            improved = score < self._best_metric
                        else:
                            improved = score > self._best_metric
                        if improved:
                            self._best_metric = score
                            self.checkpoint_cb(int(trainer.current_epoch) + 1, float(score))
                            return

                    # Fallback: path changed and best_model_score improved.
                    best_path = str(getattr(self.model_ckpt_cb, "best_model_path", "") or "")
                    if not best_path or best_path == self._last_best_path:
                        return
                    best_score_obj = getattr(self.model_ckpt_cb, "best_model_score", None)
                    if best_score_obj is None:
                        return
                    try:
                        best_score = float(best_score_obj.item())
                    except Exception:
                        return
                    if not np.isfinite(best_score):
                        return
                    if self.monitor_mode == "min":
                        improved = best_score < self._best_metric
                    else:
                        improved = best_score > self._best_metric
                    if not improved:
                        self._last_best_path = best_path
                        return
                    self._best_metric = best_score
                    self._last_best_path = best_path
                    self.checkpoint_cb(int(trainer.current_epoch) + 1, best_score)

            class _LightningEpochPrinter(Callback):  # type: ignore[misc]
                """Console epoch summary aligned with custom trainer diagnostics."""

                def __init__(self):
                    self._last_epoch = -1
                    self._last_ts: Optional[float] = None

                @staticmethod
                def _metric(trainer, key: str) -> float:
                    stores = [
                        getattr(trainer, "callback_metrics", {}) or {},
                        getattr(trainer, "logged_metrics", {}) or {},
                        getattr(trainer, "progress_bar_metrics", {}) or {},
                    ]
                    for metrics in stores:
                        for k in (f"{key}_epoch", key, f"{key}_step"):
                            val = metrics.get(k)
                            if val is None:
                                continue
                            try:
                                return float(val.item())
                            except Exception:
                                try:
                                    return float(val)
                                except Exception:
                                    continue
                    return float("nan")

                def on_validation_epoch_end(self, trainer, pl_module):
                    epoch = int(trainer.current_epoch) + 1
                    if epoch == self._last_epoch:
                        return
                    self._last_epoch = epoch
                    now = time.time()
                    epoch_time = float("nan") if self._last_ts is None else (now - self._last_ts)
                    self._last_ts = now

                    max_epochs = int(getattr(trainer, "max_epochs", epoch) or epoch)

                    train_loss = self._metric(trainer, "train/loss")
                    val_loss = self._metric(trainer, "val/loss")
                    open_loop = self._metric(trainer, "val/open_loop_crps")
                    loss_std = self._metric(trainer, "train/loss_std")
                    recon = self._metric(trainer, "train/recon")
                    aux = self._metric(trainer, "train/aux_nll")
                    kl_mean = self._metric(trainer, "train/kl_mean")
                    kl_raw = self._metric(trainer, "train/kl_raw")
                    kl_active = self._metric(trainer, "train/kl_active")
                    dec_std_mean = self._metric(trainer, "train/decoder_std_mean")
                    dec_std_min = self._metric(trainer, "train/dec_std_min")
                    prior_std_mean = self._metric(trainer, "train/prior_std_mean")
                    prior_std_max = self._metric(trainer, "train/prior_std_max")
                    post_std_mean = self._metric(trainer, "train/posterior_std_mean")
                    post_std_max = self._metric(trainer, "train/posterior_std_max")
                    rollout_nll = self._metric(trainer, "train/rollout_nll")
                    horizon = self._metric(trainer, "train/horizon")
                    ramp = self._metric(trainer, "train/rollout_ramp")

                    parts = [f"[Epoch {epoch}/{max_epochs}]"]
                    if np.isfinite(train_loss):
                        parts.append(f"train={train_loss:.6f}")
                    if np.isfinite(val_loss):
                        parts.append(f"val={val_loss:.6f}")
                    if np.isfinite(open_loop):
                        parts.append(f"open_loop_crps={open_loop:.6f}")
                    if np.isfinite(recon):
                        parts.append(f"recon={recon:.4f}")
                    if np.isfinite(kl_mean):
                        parts.append(f"kl={kl_mean:.4f}")
                    if np.isfinite(kl_active):
                        parts.append(f"kl_active={int(round(kl_active))}")
                    if np.isfinite(dec_std_mean):
                        parts.append(f"dec_std={dec_std_mean:.4f}")
                    if np.isfinite(dec_std_min):
                        parts.append(f"dec_min={dec_std_min:.4f}")
                    if np.isfinite(prior_std_mean):
                        parts.append(f"prior_std={prior_std_mean:.4f}")
                    if np.isfinite(post_std_mean):
                        parts.append(f"post_std={post_std_mean:.4f}")
                    if np.isfinite(aux):
                        parts.append(f"aux={aux:.4f}")
                    if np.isfinite(rollout_nll):
                        parts.append(f"rollout_nll={rollout_nll:.4f}")
                    if np.isfinite(horizon):
                        parts.append(f"h={horizon:.0f}")
                    if np.isfinite(ramp):
                        parts.append(f"ramp={ramp:.2f}")
                    if np.isfinite(epoch_time):
                        parts.append(f"time={epoch_time:.2f}s")
                    print(" | ".join(parts))

            def _build_lightning_logger(run_dir: Path, experiment_name: str):
                """Prefer LitLogger when installed; otherwise use CSVLogger."""
                # Prefer Lightning-specific integration to avoid launcher-style
                # top-level classes that can cause duplicate process output.
                for module_name, class_names in (
                    ("litlogger.pytorch_lightning", ("LitLogger", "LightningLogger")),
                    ("litlogger", ("LitLogger", "LightningLogger", "Logger")),
                ):
                    try:
                        module = importlib.import_module(module_name)
                    except Exception:
                        continue
                    for cls_name in class_names:
                        logger_cls = getattr(module, cls_name, None)
                        if logger_cls is None:
                            continue
                        for kwargs in (
                            # litlogger: prevent internal script re-run via PTY capture
                            {"root_dir": str(run_dir / "lightning_logs"), "name": experiment_name, "save_logs": False},
                            {"root_dir": str(run_dir / "lightning_logs"), "save_logs": False},
                            {"save_logs": False},
                            # some logger variants may accept save_dir + log_graph
                            {"save_dir": str(run_dir), "name": experiment_name, "log_graph": False},
                            {"log_graph": False},
                            {},
                        ):
                            try:
                                logger = logger_cls(**kwargs)
                                print(
                                    f"  Logger: {module_name}.{cls_name}"
                                    f"{' (with save_dir)' if kwargs else ''}"
                                )
                                print(f"  Logger run name: {experiment_name}")
                                return logger
                            except TypeError:
                                continue
                            except Exception:
                                break
                print("  Logger: CSVLogger (litlogger not available)")
                return CSVLogger(save_dir=str(run_dir), name=experiment_name)

            def _sanitize_lightning_logger_metrics(logger_obj):
                """Guard logger backend from non-finite step metrics (NaN/Inf)."""
                if logger_obj is None:
                    return logger_obj
                orig = getattr(logger_obj, "log_metrics", None)
                if not callable(orig):
                    return logger_obj

                def _safe_log_metrics(metrics, step=None, **kwargs):
                    cleaned: Dict[str, float] = {}
                    if isinstance(metrics, dict):
                        for key, val in metrics.items():
                            try:
                                if hasattr(val, "item"):
                                    val = val.item()
                                fval = float(val)
                            except Exception:
                                continue
                            if np.isfinite(fval):
                                cleaned[str(key)] = fval
                    if not cleaned:
                        return None
                    safe_step = None
                    if step is not None:
                        try:
                            step_f = float(step)
                            if np.isfinite(step_f):
                                safe_step = int(step_f)
                        except Exception:
                            safe_step = None
                    return orig(cleaned, step=safe_step, **kwargs)

                setattr(logger_obj, "log_metrics", _safe_log_metrics)
                return logger_obj

            def _log_lightning_training_config(
                logger_obj,
                *,
                cfg: Dict[str, Any],
                model_name: str,
                round_name: str,
                lr_value: float,
                wd_value: float,
                overrides_model: Dict[str, Any],
                overrides_train: Dict[str, Any],
            ) -> None:
                if logger_obj is None:
                    return

                def _as_scalar(v: Any) -> Any:
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        return v
                    return yaml.safe_dump(v, sort_keys=True)

                payload: Dict[str, Any] = {
                    "timesim/model_type": str(model_name),
                    "timesim/round": str(round_name),
                    "timesim/engine": "lightning",
                    "timesim/device": str(device),
                    "timesim/dataset_name": str(cfg.get("dataset", {}).get("name", "")),
                    "timesim/seq_len": int(seq_len),
                    "timesim/pred_len": int(pred_len),
                    "timesim/batch_size": int(batch_size),
                    "timesim/learning_rate": float(lr_value),
                    "timesim/weight_decay": float(wd_value),
                    "timesim/config_yaml": yaml.safe_dump(cfg, sort_keys=False),
                    "timesim/model_overrides_yaml": yaml.safe_dump(overrides_model, sort_keys=True),
                    "timesim/training_overrides_yaml": yaml.safe_dump(overrides_train, sort_keys=True),
                }
                payload = {k: _as_scalar(v) for k, v in payload.items()}

                try:
                    if hasattr(logger_obj, "log_hyperparams"):
                        logger_obj.log_hyperparams(payload)
                        return
                except Exception as exc:
                    print(f"  Warning: logger.log_hyperparams failed: {exc}")
                try:
                    exp = getattr(logger_obj, "experiment", None)
                    if exp is not None and hasattr(exp, "log_hyperparams"):
                        exp.log_hyperparams(payload)
                except Exception as exc:
                    print(f"  Warning: logger.experiment.log_hyperparams failed: {exc}")

            print("  Engine: lightning (experimental parity mode)")
            stride = int(config.get("data", {}).get("window_stride", 1))
            pl_train_ds = _RoleWindowDatasetFromGrouped(train_dataset, seq_len=warmup_len, stride=stride)
            pl_val_ds = _RoleWindowDatasetFromGrouped(val_dataset, seq_len=warmup_len, stride=stride)
            data_cfg = config.get("data", {})
            num_workers = int(data_cfg.get("num_workers", 0))
            pin_memory = bool(data_cfg.get("pin_memory", False))
            train_loader = DataLoader(
                pl_train_ds,
                batch_size=batch_size,
                shuffle=bool(data_cfg.get("shuffle_train", True)),
                drop_last=bool(data_cfg.get("drop_last", True)),
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=bool(num_workers > 0),
            )
            val_loader = DataLoader(
                pl_val_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=bool(data_cfg.get("drop_last", True)),
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=bool(num_workers > 0),
            )

            lightning_model = WorldModelLightningModule(
                model=model,
                learning_rate=float(lr),
                weight_decay=float(weight_decay),
                scheduler_warmup_steps=int(tcfg.get("lr_warmup_steps", 1000)),
                scheduler_min_ratio=float(tcfg.get("lr_min_ratio", 0.01)),
                grad_clip_norm=float(tcfg.get("grad_clip_norm", 0.0)),
                probabilistic_cfg=prob_cfg,
            )
            checkpoint_metric = str(prob_cfg.get("checkpoint_metric", "open_loop_crps")).lower()
            monitor_key = "val/open_loop_crps" if checkpoint_metric == "open_loop_crps" else "val/loss"
            monitor_mode = "min"
            metric_collector = _MetricCollectorCallback(metrics_path=model_dir / "metrics.csv")
            ckpt_dir = model_dir / "checkpoints"
            model_checkpoint_cb = ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="epoch{epoch:03d}",
                monitor=monitor_key,
                mode=monitor_mode,
                save_top_k=int(max(1, prob_cfg.get("checkpoint_top_k", 1))),
                save_last=True,
                auto_insert_metric_name=False,
            )
            callbacks: list[Callback] = [
                metric_collector,
                TrainingHealthCheck(check_every_n_epochs=1),
                _LightningEpochPrinter(),
                model_checkpoint_cb,
                _CheckpointEventBridge(
                    checkpoint_callback,
                    model_checkpoint_cb,
                    monitor_key=monitor_key,
                    monitor_mode=monitor_mode,
                ),
            ]
            dataset_name = str(config.get("dataset", {}).get("name", "dataset"))
            run_name = str((config.get("output", {}) or {}).get("run_name", "default") or "default")
            model_name = str(model.__class__.__name__).lower()
            round_name = "train"
            if checkpoint_path is not None:
                stem = checkpoint_path.stem
                if stem.endswith("_checkpoint"):
                    round_name = stem[:-len("_checkpoint")] or "train"
                else:
                    round_name = stem or "train"
            try:
                ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            except AttributeError:
                ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{dataset_name}-{run_name}-{model_name}-{round_name}-{ts}"
            experiment_name = experiment_name.replace(" ", "_")
            # litlogger backend enforces experiment-name length <= 64.
            if len(experiment_name) > 64:
                digest = hashlib.sha1(experiment_name.encode("utf-8")).hexdigest()[:8]
                keep = max(16, 64 - (len(ts) + len(digest) + 2))
                base = experiment_name[:keep].rstrip("-_")
                experiment_name = f"{base}-{ts}-{digest}"
                if len(experiment_name) > 64:
                    experiment_name = experiment_name[:64]
            pl_logger = _build_lightning_logger(model_dir, experiment_name)
            pl_logger = _sanitize_lightning_logger_metrics(pl_logger)
            _log_lightning_training_config(
                pl_logger,
                cfg=config,
                model_name=model.__class__.__name__,
                round_name=round_name,
                lr_value=float(lr),
                wd_value=float(weight_decay),
                overrides_model={},
                overrides_train=dict(training_overrides),
            )
            if bool(tcfg.get("early_stopping", False)):
                callbacks.append(
                    _EarlyStoppingWithMinEpoch(
                        min_epoch=int(
                            tcfg.get("early_stopping_min_epoch", tcfg.get("early_stopping_start_epoch", 25))
                        ),
                        monitor=monitor_key,
                        mode=monitor_mode,
                        patience=int(tcfg.get("patience", 5)),
                        min_delta=float(tcfg.get("min_delta", 0.0)),
                    )
                )

            trainer = pl.Trainer(
                accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                devices=1,
                max_epochs=int(epochs),
                callbacks=callbacks,
                logger=pl_logger,
                enable_checkpointing=True,
                precision="16-mixed" if use_amp and str(device).startswith("cuda") else "32-true",
                gradient_clip_val=float(tcfg.get("grad_clip_norm", 0.0)),
                log_every_n_steps=10,
                enable_progress_bar=bool(tcfg.get("lightning_progress_bar", True)),
                num_sanity_val_steps=0,
                enable_model_summary=False,
            )

            trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            train_losses = metric_collector.train_losses
            val_losses = metric_collector.val_losses

            best_path = ""
            best_score = float("nan")
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    best_path = cb.best_model_path or ""
                    if cb.best_model_score is not None:
                        try:
                            best_score = float(cb.best_model_score.item())
                        except Exception:
                            best_score = float("nan")
                    break
            if checkpoint_path is not None:
                if best_path:
                    try:
                        ckpt = torch.load(best_path, map_location=device, weights_only=True)
                    except Exception:
                        # Lightning checkpoints can include objects disallowed by
                        # strict weights-only unpickling in newer PyTorch.
                        ckpt = torch.load(best_path, map_location=device, weights_only=False)
                    state_dict = ckpt.get("state_dict", ckpt)
                    model_state = {}
                    for k, v in state_dict.items():
                        model_state[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
                    torch.save({"model_state_dict": model_state, "metadata": checkpoint_metadata}, checkpoint_path)
                else:
                    torch.save({"model_state_dict": model.state_dict(), "metadata": checkpoint_metadata}, checkpoint_path)
            if checkpoint_callback is not None and np.isfinite(best_score):
                checkpoint_callback(int(len(train_losses)), float(best_score))
            return train_losses, val_losses

    trainer = WorldModelTrainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        sampling_strategy=sampling,
        warmup_len=warmup_len,
        batch_size=batch_size,
        loss_type=loss_type,
        loss_weighting=loss_weighting,
        loss_weight_scale=loss_weight_scale,
        shape_loss_cfg=shape_loss_cfg,
        training_mode=training_mode,
        feedback=feedback,
        teacher_forcing_ratio=teacher_forcing_ratio,
        one_step_weight=one_step_weight,
        optimizer=optimizer,
        device=device,
        use_amp=use_amp,
        early_stopping=tcfg.get("early_stopping", False),
        patience=tcfg.get("patience", 5),
        min_delta=tcfg.get("min_delta", 0.0),
        early_stopping_min_epoch=tcfg.get(
            "early_stopping_min_epoch",
            tcfg.get("early_stopping_start_epoch", 25),
        ),
        run_dir=model_dir,
        probabilistic_cfg=prob_cfg,
        sequence_curriculum_cfg=tcfg.get("sequence_curriculum", None),
        checkpoint_metadata=checkpoint_metadata,
        early_stopping_start_epoch=tcfg.get("early_stopping_start_epoch", 25),
        seed=int(config.get("misc", {}).get("seed", 42)),
    )

    train_losses, val_losses = trainer.fit(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=True,
        checkpoint_path=checkpoint_path,
        on_checkpoint_saved=checkpoint_callback,
    )
    return train_losses, val_losses


def _resolve_optuna_summary_path(
    config: Dict[str, Any], out_dir: Path, cli_path: Optional[str] = None
) -> Path:
    """Resolve Optuna summary path from CLI, config, or default run location."""
    if cli_path:
        return Path(cli_path)
    cfg_path = config.get("training", {}).get("optuna_summary_path", None)
    if cfg_path:
        return Path(cfg_path)
    return out_dir / "optuna" / "summary.yaml"


def _load_optuna_best_params(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load best_params per model from an Optuna summary file."""
    if not summary_path.exists():
        print(f"  Optuna summary not found: {summary_path}")
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        print(f"  Warning: Optuna summary is not a mapping: {summary_path}")
        return {}

    best_by_model: Dict[str, Dict[str, Any]] = {}
    for model_type, entry in raw.items():
        if isinstance(entry, dict):
            best_params = entry.get("best_params", {})
            if isinstance(best_params, dict):
                best_by_model[str(model_type)] = dict(best_params)
    return best_by_model


def _split_optuna_params(
    model_type: str, best_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split Optuna params into model-constructor params and training params."""
    model_keys = get_model_param_names(model_type)
    model_overrides = {k: v for k, v in best_params.items() if k in model_keys}
    training_overrides = {k: v for k, v in best_params.items() if k in TRAINING_PARAM_KEYS}
    # Keep KL warmup schedule fixed from training config, even for legacy summaries.
    training_overrides.pop("kl_beta_start", None)
    training_overrides.pop("kl_warmup_epochs", None)
    if "weight_decay" not in training_overrides:
        optimizer_name = str(training_overrides.get("optimizer", best_params.get("optimizer", ""))).lower()
        if optimizer_name == "adamw" and "weight_decay_adamw" in best_params:
            training_overrides["weight_decay"] = best_params["weight_decay_adamw"]
        elif optimizer_name == "adam" and "weight_decay_adam" in best_params:
            training_overrides["weight_decay"] = best_params["weight_decay_adam"]
        elif "weight_decay_adamw" in best_params:
            training_overrides["weight_decay"] = best_params["weight_decay_adamw"]
        elif "weight_decay_adam" in best_params:
            training_overrides["weight_decay"] = best_params["weight_decay_adam"]
    return model_overrides, training_overrides


def _maybe_compile_model(model, config: Dict[str, Any]):
    """Optionally compile inner RSSM cell components with torch.compile.

    The outer observe/imagine methods contain Python for-loops over time steps
    which are too expensive for the compiler to trace.  Instead we compile the
    per-step RSSM cell operations and the encoder/decoder MLPs so the Inductor
    backend can fuse their small kernels into larger ones.
    """
    if getattr(model, "_timesim_compiled", False):
        return model
    tcfg = config.get("training", {})
    if not bool(tcfg.get("use_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        print("  Warning: torch.compile not available in this torch build.")
        return model
    compile_mode = str(tcfg.get("compile_mode", "default"))
    compile_backend = str(tcfg.get("compile_backend", "inductor"))
    if sys.platform.startswith("win") and shutil.which("cl") is None:
        if compile_backend == "inductor":
            print("  Warning: MSVC 'cl' not found — falling back to 'aot_eager' backend for torch.compile.")
            print("  Tip: Install 'Build Tools for Visual Studio' for full Inductor performance.")
            compile_backend = "aot_eager"

    compiled_any = False
    rssm_cell = getattr(model, "rssm_cell", None)
    if rssm_cell is not None:
        for method_name in ("observe", "imagine", "transition"):
            orig = getattr(rssm_cell, method_name, None)
            if orig is None:
                continue
            try:
                compiled_fn = torch.compile(orig, mode=compile_mode, backend=compile_backend)
                setattr(rssm_cell, method_name, compiled_fn)
                compiled_any = True
            except Exception as exc:
                print(f"  Warning: torch.compile failed for rssm_cell.{method_name}: {exc}")

    if compiled_any:
        model._timesim_compiled = True
        print(f"  torch.compile enabled on RSSM cell (backend={compile_backend}, mode={compile_mode})")
    else:
        print("  Warning: torch.compile failed on all components, using eager mode.")
    return model


def prepare_xgboost_data(dataset, seq_len):
    """Prepare (X, y) arrays for XGBoost from a GroupedTimeSeriesDataset."""
    values = dataset.values
    out_idx = dataset.out_idx
    n_samples = len(values) - seq_len
    n_features = values.shape[1]
    output_dim = len(out_idx)

    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.zeros((n_samples, 1, output_dim), dtype=np.float32)
    for i in range(n_samples):
        X[i] = values[i : i + seq_len]
        y[i, 0] = values[i + seq_len, out_idx]
    return X, y


def train_xgboost_model(model, train_dataset, val_dataset, config, model_dir):
    """Train XGBoost model. Returns (train_losses, val_losses)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    seq_len = config["dataset"]["seq_len"]

    print("  Preparing XGBoost training data...")
    X_train, y_train = prepare_xgboost_data(train_dataset, seq_len)
    X_val, y_val = prepare_xgboost_data(val_dataset, seq_len)
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  X_val  ={X_val.shape},  y_val  ={y_val.shape}")

    print("  Fitting XGBoost...")
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    elapsed = time.time() - t0
    print(f"  XGBoost training time: {elapsed:.2f}s")

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_mse = float(np.mean((train_pred - y_train) ** 2))
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    print(f"  train MSE={train_mse:.6f}  val MSE={val_mse:.6f}")

    return [train_mse], [val_mse]


def _save_model_artifacts(model_dir: Path, dataset, scaler) -> None:
    """Save schema and normalization stats next to model checkpoints."""
    model_dir.mkdir(parents=True, exist_ok=True)
    schema = getattr(dataset, "variable_schema", None)
    if schema is not None:
        with open(model_dir / "variable_schema.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(schema.to_groups(), f, sort_keys=False)

    from joblib import dump
    dump(scaler, model_dir / "scaler.pkl")
    dump(scaler, model_dir / "normalization_stats.pkl")


def _log_effective_model_config(
    *,
    model_type: str,
    model: Any,
    model_dir: Path,
    merged_model_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Log and persist effective model configuration for reproducibility."""
    payload: Dict[str, Any] = {
        "model_type": str(model_type),
        "resolved_config": dict(merged_model_cfg or {}),
    }

    runtime_keys = [
        "hidden_dim",
        "latent_dim",
        "encoder_dim",
        "decoder_hidden",
        "decoder_layers",
        "h_dropout",
        "predict_exogenous",
        "use_aux_decoder",
        "decode_exogenous",
        "prior_constant_std",
        "posterior_constant_std",
        "min_std",
        "max_std",
    ]
    runtime_values: Dict[str, Any] = {}
    for key in runtime_keys:
        if hasattr(model, key):
            val = getattr(model, key)
            if hasattr(val, "p"):  # nn.Dropout
                try:
                    val = float(val.p)
                except Exception:
                    val = str(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.integer,)):
                val = int(val)
            runtime_values[key] = val

    if runtime_values:
        payload["runtime"] = runtime_values

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "effective_model_config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    if runtime_values:
        preview_keys = ["hidden_dim", "latent_dim", "encoder_dim", "decoder_hidden", "h_dropout", "predict_exogenous"]
        preview = ", ".join(
            f"{k}={runtime_values[k]}"
            for k in preview_keys
            if k in runtime_values
        )
        if preview:
            print(f"  Effective model config: {preview}")
    print(f"  Saved effective model config -> {out_path}")


def _log_trainer_metrics_csv(
    tracker: ExperimentTracker,
    metrics_path: Path,
    metric_prefix: str,
    start_row: int = 0,
) -> int:
    """Log detailed trainer metrics (loss terms, LR, grad norm, schedules) to tracker."""
    if not metrics_path.exists():
        return int(start_row)
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return int(start_row)
    if start_row >= len(df):
        return int(len(df))

    new_rows = df.iloc[int(start_row):]
    for idx, row in new_rows.iterrows():
        metrics: Dict[str, float] = {}
        for col, val in row.items():
            if col == "epoch":
                continue
            if isinstance(val, (int, float)) and np.isfinite(val):
                metrics[f"{metric_prefix}/{col}"] = float(val)
        if not metrics:
            continue
        epoch_val = row.get("epoch", np.nan)
        if isinstance(epoch_val, (int, float)) and np.isfinite(epoch_val):
            step = int(epoch_val)
        else:
            step = int(idx + 1)
        tracker.log_metrics(metrics, step=step)
    return int(len(df))


def _build_simulation_start_idx_schedule(
    total_len: int,
    seq_len: int,
    sim_horizon: int,
    n_rounds: int,
    fixed_start_idx: int,
    seed_base: int,
    round_name: str,
    model_type: str,
) -> list[int]:
    """Build start-index schedule: first fixed index, remaining random indices."""
    if n_rounds <= 0:
        return []
    max_start_allowed = max(0, total_len - seq_len - sim_horizon)
    fixed = int(np.clip(fixed_start_idx, 0, max_start_allowed))
    schedule = [fixed]
    if n_rounds == 1:
        return schedule

    model_round_salt = sum(ord(ch) for ch in f"{round_name}:{model_type}")
    rng = np.random.default_rng(seed_base + model_round_salt)
    candidate_starts = [i for i in range(max_start_allowed + 1) if i != fixed]
    remaining = n_rounds - 1
    if len(candidate_starts) >= remaining:
        sampled = rng.choice(
            np.asarray(candidate_starts, dtype=np.int64),
            size=remaining,
            replace=False,
        ).tolist()
    elif len(candidate_starts) > 0:
        sampled = list(candidate_starts)
        sampled.extend(
            rng.choice(
                np.asarray(candidate_starts, dtype=np.int64),
                size=remaining - len(candidate_starts),
                replace=True,
            ).tolist()
        )
    else:
        sampled = [fixed] * remaining
    schedule.extend(int(s) for s in sampled)
    return schedule


def _git_hash() -> str:
    """Return current git hash, or 'unknown' when unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Train world models for time-series simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "--config-name", type=str, required=True,
                        help="Hydra config name or path to YAML config")
    parser.add_argument("--models", nargs="*",
                        help="Override: train only these model types")
    parser.add_argument("--epochs", type=int,
                        help="Override epochs for all rounds")
    parser.add_argument("--steps-per-epoch", type=int,
                        help="Override steps per epoch for all rounds")
    parser.add_argument("--device", type=str,
                        help="Override device (auto / cpu / cuda)")
    parser.add_argument(
        "--use-optuna-best-params",
        dest="use_optuna_best_params",
        action="store_true",
        help="Use per-model best params from Optuna summary if available",
    )
    parser.add_argument(
        "--no-use-optuna-best-params",
        dest="use_optuna_best_params",
        action="store_false",
        help="Disable Optuna best params even if enabled in config",
    )
    parser.set_defaults(use_optuna_best_params=None)
    parser.add_argument(
        "--optuna-summary",
        type=str,
        help="Path to Optuna summary.yaml (default: <run_dir>/optuna/summary.yaml)",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(config: Optional[Dict[str, Any]] = None) -> None:
    """Entry point for training.

    When *config* is ``None`` (direct CLI invocation), the config is loaded
    from ``--config`` with any extra ``key=value`` args forwarded as Hydra
    overrides.  When *config* is provided (Hydra path via ``train_hydra.py``),
    it is used as-is.
    """
    config_source = "<hydra>"
    cli_models: Optional[list] = None
    cli_epochs: Optional[int] = None
    cli_steps_per_epoch: Optional[int] = None
    cli_optuna_summary: Optional[str] = None

    if config is None:
        parser = _build_cli_parser()
        args, hydra_overrides = parser.parse_known_args()
        config = compose_config(args.config, overrides=hydra_overrides)
        config_source = args.config
        cli_models = args.models
        cli_epochs = args.epochs
        cli_steps_per_epoch = args.steps_per_epoch
        cli_optuna_summary = args.optuna_summary
        if args.device:
            config["misc"]["device"] = args.device
        if args.use_optuna_best_params is not None:
            config.setdefault("training", {})
            config["training"]["use_optuna_best_params"] = args.use_optuna_best_params
        if cli_optuna_summary:
            config.setdefault("training", {})
            config["training"]["optuna_summary_path"] = cli_optuna_summary
    else:
        config = dict(config)

    # Extract top-level override fields that Hydra may have composed.
    _popped_models = cli_models or config.pop("models", None) or None
    if isinstance(_popped_models, list) and _popped_models:
        cli_models = [
            m["type"] if isinstance(m, dict) else str(m) for m in _popped_models
        ]
    else:
        cli_models = _popped_models
    cli_epochs = cli_epochs or config.pop("epochs", None) or None
    cli_steps_per_epoch = (
        cli_steps_per_epoch
        if cli_steps_per_epoch is not None
        else config.pop("steps_per_epoch", None)
    )
    cli_optuna_summary = cli_optuna_summary or config.pop("optuna_summary", None) or None

    device_override = config.pop("device", None)
    if device_override:
        config.setdefault("misc", {})
        config["misc"]["device"] = device_override

    use_optuna_override = config.pop("use_optuna_best_params", None)
    if use_optuna_override is not None:
        config.setdefault("training", {})
        config["training"]["use_optuna_best_params"] = use_optuna_override

    if cli_epochs:
        config["training"]["epochs"] = cli_epochs

    seed = int(config["misc"].get("seed", 42))
    deterministic = bool(config.get("misc", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(config.get("misc", {}).get("device", "auto"))
    config.setdefault("misc", {})
    config["misc"]["device"] = device

    # ── Config sub-dicts ──────────────────────────────────────────────
    tcfg = config.get("training", {})
    data_cfg = config.get("data", {})
    add_time_features = bool(data_cfg.get("add_time_features", False))
    time_features_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(time_features_cfg, dict) and "enabled" in time_features_cfg:
        add_time_features = bool(time_features_cfg.get("enabled")) or add_time_features
    model_defaults_cfg = config.get("model_defaults", {})
    plot_cfg = config.get("plotting", {})
    output_cfg = config.get("output", {})
    tracking_cfg = config.get("tracking", {}) or {}
    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None

    # ── Determine models ──────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}
    single_model_cfg = dict(config.get("model", {}) or {})

    def _resolve_model_cfg(model_type: str) -> Dict[str, Any]:
        cfg = models_cfg_map.get(model_type)
        if cfg is not None:
            return cfg
        if single_model_cfg and str(single_model_cfg.get("type", "")).lower() == str(model_type).lower():
            return dict(single_model_cfg)
        return {"type": model_type}

    if cli_models:
        model_names = cli_models
    elif models_cfg_list:
        model_names = [m["type"] for m in models_cfg_list]
    else:
        model_names = [config.get("model", {}).get("type", "lstm")]

    # Enforce RSSM "DO NOT" policy upfront for selected models.
    for model_name in model_names:
        if model_name != "latent_ssm":
            continue
        per_model_cfg = _resolve_model_cfg(model_name)
        effective_model = merged_latent_ssm_params(
            model_defaults_cfg=model_defaults_cfg,
            per_model_cfg=per_model_cfg,
            model_overrides=None,
        )
        effective_prob = merged_probabilistic_cfg(
            training_cfg=tcfg,
            training_overrides=None,
        )
        validate_latent_ssm_do_not(
            model_name=model_name,
            model_params=effective_model,
            prob_cfg=effective_prob,
            data_cfg=data_cfg,
            context="base_config",
        )

    # ── Dimension setup ───────────────────────────────────────────────
    groups = config["dataset"]["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    input_cols = schema.columns_for_group_names(input_groups)
    output_cols = schema.columns_for_group_names(output_groups)

    dataset_seq_len = int(config["dataset"]["seq_len"])
    pred_len = config["dataset"]["pred_len"]
    batch_size = config["dataset"]["batch_size"]
    # Use union to avoid double-counting when output_cols are in input_groups
    all_input_features = set(input_cols) | set(output_cols)
    input_dim = len(all_input_features)
    if add_time_features:
        input_dim += len(
            get_time_feature_columns(
                features=time_features_cfg.get("features"),
                encoding=time_features_cfg.get("encoding", "cyclical"),
            )
        )
    output_dim = len(output_cols)

    # ── Training rounds ───────────────────────────────────────────────
    training_rounds = config.get("training_rounds", None)
    if training_rounds is None:
        training_rounds = [{
            "name": "train",
            "epochs": config["training"]["epochs"],
            "models": model_names,
        }]
    else:
        for rc in training_rounds:
            if "models" not in rc:
                rc["models"] = model_names
        # CLI override: --models applies to every round
        if cli_models:
            for rc in training_rounds:
                rc["models"] = model_names

    # Effective sequence length migration:
    # - RSSM-only runs use training.window_len (fallback: warmup_len, then dataset.seq_len)
    # - mixed/legacy runs keep dataset.seq_len
    all_round_models = {
        str(mt).lower()
        for rc in training_rounds
        for mt in rc.get("models", model_names)
    }
    latent_only_pipeline = bool(all_round_models) and all(mt == "latent_ssm" for mt in all_round_models)
    rssm_window_len = int(
        config.get("training", {}).get(
            "window_len",
            config.get("training", {}).get("warmup_len", dataset_seq_len),
        )
    )
    seq_len = int(rssm_window_len if latent_only_pipeline else dataset_seq_len)

    # ── Print banner ──────────────────────────────────────────────────
    round_desc = " -> ".join(rc.get("name", "?") for rc in training_rounds)
    print("\n" + "=" * 70)
    print("  MULTI-MODEL TRAINING")
    print("=" * 70)
    print(f"  Config       : {config_source}")
    print(f"  Rounds       : {round_desc}")
    print(f"  Models       : {model_names}")
    print(f"  Device       : {device}")
    print(f"  Input cols   : {input_cols}")
    print(f"  Output cols  : {output_cols}")
    print(f"  input_dim={input_dim}  output_dim={output_dim}")
    if add_time_features:
        print(f"  Time features: enabled ({time_features_cfg})")
    if seq_len != dataset_seq_len:
        print(f"  seq_len={seq_len} (dataset.seq_len={dataset_seq_len})  pred_len={pred_len}")
    else:
        print(f"  seq_len={seq_len}  pred_len={pred_len}")
    print(f"  Run name     : {run_name if run_name is not None else '(default)'}")
    print("=" * 70 + "\n")

    # ── Eval / simulation dimensions ──────────────────────────────────
    control_dim = 0
    exo_dim = 0

    warmup_len = config["training"].get(
        "window_len",
        config["training"].get("warmup_len", seq_len),
    )
    eval_cfg = config.get("evaluation", {}) or {}
    eval_horizon = eval_cfg.get("horizon", max(pred_len, 12))
    n_windows = eval_cfg.get("n_windows", 4)
    prob_eval_cfg = eval_cfg.get("probabilistic", {}) or {}
    prob_train_cfg = dict(prob_eval_cfg)
    prob_training_cfg = config.get("training", {}).get("probabilistic", {}) or {}
    if "mc_train_samples" in prob_training_cfg:
        prob_train_cfg["mc_samples"] = int(prob_training_cfg["mc_train_samples"])

    sim_cfg = config.get("simulation", {})
    sim_start = sim_cfg.get("start_idx", 0)
    checkpoint_sim_cfg = config.get("training", {}).get("checkpoint_simulation", {}) or {}
    checkpoint_sim_enabled = bool(checkpoint_sim_cfg.get("enabled", False))
    checkpoint_sim_rounds = int(checkpoint_sim_cfg.get("n_rounds", 0) or 0)
    checkpoint_sim_rounds = max(0, checkpoint_sim_rounds)
    checkpoint_sim_horizon_cfg = checkpoint_sim_cfg.get("horizon", None)
    if checkpoint_sim_horizon_cfg is None:
        checkpoint_sim_horizon_cfg = sim_cfg.get("horizon", None)
    checkpoint_sim_start = int(checkpoint_sim_cfg.get("start_idx", sim_start))

    # ── Output directory ──────────────────────────────────────────────
    dataset_name = config["dataset"]["name"]
    out_dir = runs_dir / dataset_name
    if run_name is not None:
        out_dir = out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    use_optuna_best_params = bool(
        config.get("training", {}).get("use_optuna_best_params", False)
    )
    optuna_summary_path = _resolve_optuna_summary_path(config, out_dir, cli_optuna_summary)
    optuna_best_by_model: Dict[str, Dict[str, Any]] = {}
    if use_optuna_best_params:
        print("Loading Optuna best params...")
        print(f"  Summary path: {optuna_summary_path}")
        optuna_best_by_model = _load_optuna_best_params(optuna_summary_path)
        print(f"  Models with best params: {sorted(optuna_best_by_model.keys())}")

    # Save resolved config
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    run_meta = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_hash": _git_hash(),
        "config_path": config_source,
    }
    with open(out_dir / "run_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_meta, f, sort_keys=False)

    tracker_run_name = tracking_cfg.get("run_name", None) or run_name or "run"
    tracker = ExperimentTracker(
        backend=str(tracking_cfg.get("backend", "none")),
        project=str(
            tracking_cfg.get("project", f"timesim-{config['dataset']['name']}")
        ),
        run_name=str(tracker_run_name),
        run_dir=out_dir,
        config=config if bool(tracking_cfg.get("log_config", True)) else None,
        tags=list(tracking_cfg.get("tags", [])),
    )
    tracker.log_params(
        {
            "git_hash": run_meta.get("git_hash", "unknown"),
            "config_path": run_meta.get("config_path", ""),
            "dataset.name": config["dataset"]["name"],
            "output.run_name": run_name or "",
        }
    )

    # Pre-create per-model directories
    for rc in training_rounds:
        for mt in rc.get("models", model_names):
            (out_dir / mt).mkdir(exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    index_col = config["dataset"].get("index_col",
                                       data_cfg.get("index_col", "date"))

    print("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=config["dataset"].get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    train_loader, val_loader, scaler = build_dataloaders_from_config(
        config=config,
        df=df,
        seed=seed,
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    control_dim = len(getattr(train_dataset, "control_positions", []))
    exo_dim = len(getattr(train_dataset, "known_exo_positions", []))
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Save scaler ───────────────────────────────────────────────────
    from joblib import dump
    scaler_path = out_dir / "scaler.pkl"
    dump(scaler, scaler_path)
    dump(scaler, out_dir / "normalization_stats.pkl")
    print(f"  Saved scaler -> {scaler_path}")

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING ROUNDS
    # ══════════════════════════════════════════════════════════════════
    trained_models = {}
    model_n_params = {}
    cumul_train_losses = {}
    cumul_val_losses = {}
    cumul_time = {}
    model_last_round = {}
    trainer_metric_row_offsets: Dict[str, int] = {}

    for round_cfg in training_rounds:
        round_name = round_cfg.get("name", "train")
        round_epochs = round_cfg.get("epochs", config["training"]["epochs"])
        round_lr = round_cfg.get("learning_rate",
                                 config["training"].get("learning_rate", 1e-3))
        round_steps_per_epoch = round_cfg.get(
            "steps_per_epoch",
            config["training"].get("steps_per_epoch", None)
        )
        round_model_types = round_cfg.get("models", model_names)
        checkpoint_dir = round_cfg.get("checkpoint_dir", None)

        # Apply --epochs CLI override to every round
        if cli_epochs:
            round_epochs = cli_epochs
        if cli_steps_per_epoch is not None:
            round_steps_per_epoch = cli_steps_per_epoch

        print(f"\n{'═' * 70}")
        print(f"  ROUND: {round_name.upper()} "
              f"({round_epochs} epochs, LR={round_lr})")
        print(f"  Steps/epoch: {round_steps_per_epoch if round_steps_per_epoch is not None else 'auto'}")
        print(f"  Models: {round_model_types}")
        if checkpoint_dir:
            print(f"  Loading checkpoints from: {checkpoint_dir}")
        print(f"{'═' * 70}")

        for model_type in round_model_types:
            mc = _resolve_model_cfg(model_type)
            model_dir = out_dir / model_type
            model_lr = round_lr
            effective_model: Optional[Dict[str, Any]] = None
            if model_type in NEURAL_MODELS and "learning_rate" in mc:
                model_lr = mc["learning_rate"]
            model_overrides: Dict[str, Any] = {}
            train_overrides: Dict[str, Any] = {}
            if use_optuna_best_params and model_type in optuna_best_by_model:
                model_overrides, train_overrides = _split_optuna_params(
                    model_type, optuna_best_by_model[model_type]
                )
                model_lr = train_overrides.get("learning_rate", model_lr)
                print(
                    f"  Optuna overrides for {model_type}: "
                    f"model={model_overrides} training={train_overrides}"
                )

            if model_type == "latent_ssm":
                effective_model = merged_latent_ssm_params(
                    model_defaults_cfg=model_defaults_cfg,
                    per_model_cfg=mc,
                    model_overrides=model_overrides,
                )
                effective_prob = merged_probabilistic_cfg(
                    training_cfg=tcfg,
                    training_overrides=train_overrides,
                )
                validate_latent_ssm_do_not(
                    model_name=model_type,
                    model_params=effective_model,
                    prob_cfg=effective_prob,
                    data_cfg=data_cfg,
                    context=f"round={round_name}",
                )

            # ── Decide: new model, resume in-memory, or load checkpoint ──
            is_retrain = False

            if model_type in trained_models and model_type in NEURAL_MODELS:
                model = trained_models[model_type]
                is_retrain = True
            elif checkpoint_dir and model_type in NEURAL_MODELS:
                model = build_model(
                    model_type, input_dim, output_dim, seq_len, pred_len,
                    per_model_cfg=mc,
                    model_defaults_cfg=model_defaults_cfg,
                    overrides=model_overrides)
                ckpt = Path(checkpoint_dir) / model_type / "train_checkpoint.pth"
                if ckpt.exists():
                    try:
                        state = torch.load(ckpt, map_location=device, weights_only=True)
                    except Exception:
                        state = torch.load(ckpt, map_location=device, weights_only=False)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        state = state["model_state_dict"]
                    model.load_state_dict(state)
                    model.to(device)
                    is_retrain = True
                    print(f"  Loaded checkpoint: {ckpt}")
                else:
                    print(f"  Warning: {ckpt} not found -> training from scratch")
            elif checkpoint_dir and model_type == "xgboost" and HAS_XGBOOST:
                pkl = Path(checkpoint_dir) / model_type / "train_model.pkl"
                if pkl.exists():
                    model = XGBoostForecaster.load(str(pkl))
                    is_retrain = True
                    print(f"  Loaded model: {pkl}")
                else:
                    model = build_model(
                        model_type, input_dim, output_dim, seq_len, pred_len,
                        per_model_cfg=mc,
                        model_defaults_cfg=model_defaults_cfg,
                        overrides=model_overrides)
            else:
                model = build_model(
                    model_type, input_dim, output_dim, seq_len, pred_len,
                    per_model_cfg=mc,
                    model_defaults_cfg=model_defaults_cfg,
                    overrides=model_overrides)
            if model_type in NEURAL_MODELS:
                model = _maybe_compile_model(model, config)

            tag = "RETRAIN" if is_retrain else "TRAIN"
            print(f"\n{'─' * 70}")
            print(f"  [{tag}] {model_type.upper()}")
            print(f"{'─' * 70}")

            if model_type == "xgboost" and not HAS_XGBOOST:
                print("  SKIPPED – xgboost not installed")
                continue

            n_params = count_parameters(model)
            model_n_params[model_type] = n_params
            print(f"  Parameters: {n_params:,}" if n_params
                  else "  Parameters: N/A (tree-based)")
            if model_type in NEURAL_MODELS:
                _log_effective_model_config(
                    model_type=model_type,
                    model=model,
                    model_dir=model_dir,
                    merged_model_cfg=effective_model,
                )

            t0 = time.time()
            try:
                if model_type in NEURAL_MODELS:
                    effective_lr = model_lr
                    effective_optimizer = train_overrides.get("optimizer", None)
                    checkpoint_callback = None
                    if checkpoint_sim_enabled and checkpoint_sim_rounds > 0:
                        monitor_state = {"plot_counter": 0}
                        if checkpoint_sim_horizon_cfg is None:
                            checkpoint_sim_horizon = len(val_dataset.values) - seq_len
                        else:
                            checkpoint_sim_horizon = int(checkpoint_sim_horizon_cfg)
                        checkpoint_sim_horizon = max(0, checkpoint_sim_horizon)
                        seed_base = int(config.get("misc", {}).get("seed", 42))
                        start_idx_schedule = _build_simulation_start_idx_schedule(
                            total_len=len(val_dataset.values),
                            seq_len=seq_len,
                            sim_horizon=checkpoint_sim_horizon,
                            n_rounds=checkpoint_sim_rounds,
                            fixed_start_idx=checkpoint_sim_start,
                            seed_base=seed_base,
                            round_name=round_name,
                            model_type=model_type,
                        )

                        round_name_local = round_name
                        model_type_local = model_type
                        model_dir_local = model_dir
                        simulation_plot_dir = model_dir_local / "simulations"
                        print(
                            "  Checkpoint simulation start_idx schedule: "
                            f"{start_idx_schedule[:checkpoint_sim_rounds]}"
                        )

                        def checkpoint_callback(epoch_idx: int, improved_score: float):
                            if checkpoint_sim_horizon <= 0:
                                return
                            if isinstance(model, torch.nn.Module):
                                model.to(device)
                                model.eval()
                            saved_this_epoch = 0
                            # Save all configured simulation rounds for each
                            # improved checkpoint epoch.
                            for sim_start_idx in start_idx_schedule[:checkpoint_sim_rounds]:
                                sim_result = simulate_recursive_neural(
                                    model, val_dataset, seq_len,
                                    checkpoint_sim_horizon, device,
                                    start_idx=sim_start_idx,
                                    probabilistic_cfg=prob_train_cfg,
                                )
                                if sim_result["n_steps"] <= 0:
                                    continue
                                monitor_state["plot_counter"] += 1
                                sim_idx = monitor_state["plot_counter"]
                                sim_plot_path = simulation_plot_dir / f"{round_name_local}_simulation_{sim_idx}.png"
                                save_per_model_simulation_plot(
                                    sim_result, output_cols, model_type_local,
                                    sim_plot_path,
                                    plot_cfg=plot_cfg,
                                )
                                saved_this_epoch += 1
                            if saved_this_epoch > 0:
                                print(
                                    f"  Saved {saved_this_epoch} checkpoint simulation plots "
                                    f"(epoch={epoch_idx}, score={improved_score:.6f})"
                                )

                    train_losses, val_losses = train_neural_model(
                        model, train_dataset, val_dataset, config, device,
                        model_dir,
                        lr_override=effective_lr,
                        epochs_override=round_epochs,
                        steps_per_epoch_override=round_steps_per_epoch,
                        optimizer_override=effective_optimizer,
                        training_overrides=train_overrides,
                        checkpoint_path=model_dir / f"{round_name}_checkpoint.pth",
                        checkpoint_callback=checkpoint_callback,
                    )
                    trainer_metric_row_offsets[model_type] = _log_trainer_metrics_csv(
                        tracker=tracker,
                        metrics_path=model_dir / "metrics.csv",
                        metric_prefix=f"{model_type}/{round_name}",
                        start_row=trainer_metric_row_offsets.get(model_type, 0),
                    )
                else:
                    train_losses, val_losses = train_xgboost_model(
                        model, train_dataset, val_dataset, config, model_dir,
                    )
                    model.save(str(model_dir / f"{round_name}_model.pkl"))
                    if checkpoint_sim_enabled and checkpoint_sim_rounds > 0:
                        if checkpoint_sim_horizon_cfg is None:
                            checkpoint_sim_horizon = len(val_dataset.values) - seq_len
                        else:
                            checkpoint_sim_horizon = int(checkpoint_sim_horizon_cfg)
                        checkpoint_sim_horizon = max(0, checkpoint_sim_horizon)
                        if checkpoint_sim_horizon > 0:
                            seed_base = int(config.get("misc", {}).get("seed", 42))
                            start_idx_schedule = _build_simulation_start_idx_schedule(
                                total_len=len(val_dataset.values),
                                seq_len=seq_len,
                                sim_horizon=checkpoint_sim_horizon,
                                n_rounds=checkpoint_sim_rounds,
                                fixed_start_idx=checkpoint_sim_start,
                                seed_base=seed_base,
                                round_name=round_name,
                                model_type=model_type,
                            )
                            simulation_plot_dir = model_dir / "simulations"
                            print(
                                "  Checkpoint simulation start_idx schedule: "
                                f"{start_idx_schedule[:checkpoint_sim_rounds]}"
                            )
                            saved_xgb_sims = 0
                            for sim_start_idx in start_idx_schedule[:checkpoint_sim_rounds]:
                                sim_result = simulate_recursive_xgboost(
                                    model, val_dataset, seq_len,
                                    checkpoint_sim_horizon,
                                    start_idx=sim_start_idx,
                                )
                                if sim_result["n_steps"] <= 0:
                                    continue
                                saved_xgb_sims += 1
                                sim_plot_path = simulation_plot_dir / f"{round_name}_simulation_{saved_xgb_sims}.png"
                                save_per_model_simulation_plot(
                                    sim_result, output_cols, model_type,
                                    sim_plot_path,
                                    plot_cfg=plot_cfg,
                                )
                            if saved_xgb_sims > 0:
                                print(
                                    f"  Saved {saved_xgb_sims} checkpoint simulation plots "
                                    "(post-fit xgboost)"
                                )

                _save_model_artifacts(model_dir, train_dataset, scaler)
                elapsed = time.time() - t0

                # Round-specific loss plot
                save_loss_plot(
                    train_losses, val_losses,
                    model_dir / f"{round_name}_loss.png",
                    title=f"{model_type.upper()} – {round_name} "
                          f"({round_epochs} epochs, LR={model_lr})",
                )

                cumul_train_losses.setdefault(model_type, []).extend(train_losses)
                cumul_val_losses.setdefault(model_type, []).extend(val_losses)
                cumul_time[model_type] = cumul_time.get(model_type, 0.0) + elapsed
                trained_models[model_type] = model
                model_last_round[model_type] = round_name

                # Save / update model config
                model_params = {k: v for k, v in mc.items() if k != "type"}
                resolved_model_params = dict(model_params)
                resolved_model_params.update(model_overrides)
                total_epochs = len(cumul_train_losses[model_type])
                resolved_prob_cfg = dict(tcfg.get("probabilistic", {}) or {})
                for k in [
                    "recon_weight",
                    "elbo_weight",
                    "kl_weight",
                    "aux_weight",
                    "rollout_mse_weight",
                    "rollout_weight",
                    "rollout_dtw_weight",
                    "rollout_dtw_gamma",
                    "rollout_warmup_fraction",
                    "rollout_max_horizon",
                    "min_context",
                    "kl_free_bits",
                    "kl_balance",
                    "use_kl_balancing",
                    "use_free_bits",
                    "use_symlog",
                    "use_aux_decoder",
                    "use_dual_path",
                    "leak_objective_to_transition",
                    "grad_clip_norm",
                    "lr_warmup_steps",
                    "lr_min_ratio",
                    "checkpoint_top_k",
                    "early_stopping_monitor",
                    "objective",
                    "kl_warmup_enabled",
                    "kl_beta_start",
                    "kl_beta_end",
                    "kl_warmup_epochs",
                    "checkpoint_metric",
                    "checkpoint_open_loop_horizon",
                    "checkpoint_open_loop_windows",
                    "checkpoint_open_loop_samples",
                ]:
                    if k in train_overrides:
                        resolved_prob_cfg[k] = train_overrides[k]
                model_cfg_data = {
                    "type": model_type,
                    **resolved_model_params,
                    "last_round": round_name,
                    "total_epochs": total_epochs,
                    "resolved": {
                        "model_params": resolved_model_params,
                        "model_overrides": dict(model_overrides),
                        "training_overrides": dict(train_overrides),
                        "training": {
                            "epochs": round_epochs,
                            "steps_per_epoch": round_steps_per_epoch,
                            "learning_rate": model_lr,
                            "optimizer": train_overrides.get(
                                "optimizer", tcfg.get("optimizer", "adam")
                            ),
                            "weight_decay": float(
                                train_overrides.get(
                                    "weight_decay", tcfg.get("weight_decay", 0.0)
                                )
                            ),
                            "mode": train_overrides.get("mode", tcfg.get("mode", "multi_step")),
                            "feedback": train_overrides.get("feedback", tcfg.get("feedback", "model")),
                            "teacher_forcing_ratio": train_overrides.get(
                                "teacher_forcing_ratio",
                                tcfg.get("teacher_forcing_ratio", 0.0),
                            ),
                            "one_step_weight": train_overrides.get(
                                "one_step_weight", tcfg.get("one_step_weight", 0.5)
                            ),
                            "sequence_curriculum": dict(
                                tcfg.get("sequence_curriculum", {}) or {}
                            ),
                            "probabilistic": resolved_prob_cfg,
                        },
                        "data": {
                            "seq_len": seq_len,
                            "pred_len": pred_len,
                            "batch_size": batch_size,
                            "input_dim": input_dim,
                            "output_dim": output_dim,
                            "add_time_features": add_time_features,
                            "time_features_cfg": time_features_cfg,
                            "require_full_role_mapping": bool(
                                data_cfg.get("require_full_role_mapping", True)
                            ),
                            "variable_schema": train_dataset.variable_schema.to_groups(),
                        },
                        "optuna": {
                            "enabled": bool(
                                use_optuna_best_params and model_type in optuna_best_by_model
                            ),
                            "summary_path": str(optuna_summary_path)
                            if use_optuna_best_params
                            else None,
                            "best_params": dict(optuna_best_by_model.get(model_type, {}))
                            if use_optuna_best_params
                            else {},
                        },
                    },
                }
                with open(model_dir / "model_config.yaml", "w") as f:
                    yaml.safe_dump(model_cfg_data, f)

                tl = train_losses[-1] if train_losses else float("nan")
                vl = val_losses[-1] if val_losses else float("nan")
                print(f"  => train_loss={tl:.6f}  val_loss={vl:.6f}  "
                      f"time={elapsed:.1f}s")
                tracker.log_metrics(
                    {
                        f"{model_type}/{round_name}/train_loss": float(tl),
                        f"{model_type}/{round_name}/val_loss": float(vl),
                        f"{model_type}/{round_name}/train_time_sec": float(elapsed),
                    },
                    step=int(total_epochs),
                )
                tracker.log_artifact(model_dir / "model_config.yaml", artifact_path=model_type)

                # ── Post-round evaluation & simulation ────────────
                try:
                    print(f"  Running evaluation & simulation [{round_name}]...")

                    # Evaluation (forecast rollout)
                    if model_type in NEURAL_MODELS:
                        gt_list, pred_list, eval_info = evaluate_neural_model(
                            model, val_dataset, warmup_len, eval_horizon,
                            control_dim, exo_dim, device, n_windows,
                            probabilistic_cfg=prob_train_cfg,
                            return_info=True,
                        )
                    else:
                        gt_list, pred_list = evaluate_xgboost_model(
                            model, val_dataset, seq_len, eval_horizon,
                            n_windows,
                        )
                        eval_info = {
                            "is_probabilistic": False,
                            "rollout_nll": float("nan"),
                            "coverage_90": float("nan"),
                            "interval_width_90": float("nan"),
                        }

                    if gt_list and pred_list:
                        mean_mse = float(np.mean(
                            [np.mean((g - p) ** 2)
                             for g, p in zip(gt_list, pred_list)]))
                        mean_mae = float(np.mean(
                            [np.mean(np.abs(g - p))
                             for g, p in zip(gt_list, pred_list)]))
                        print(f"    Eval  MSE={mean_mse:.6f}  "
                              f"MAE={mean_mae:.6f}")
                        tracker.log_metrics(
                            {
                                f"{model_type}/{round_name}/eval_mse": float(mean_mse),
                                f"{model_type}/{round_name}/eval_mae": float(mean_mae),
                            },
                            step=int(total_epochs),
                        )
                        if bool(eval_info.get("is_probabilistic", False)):
                            nll = eval_info.get("rollout_nll", float("nan"))
                            cov = eval_info.get("coverage_90", float("nan"))
                            wid = eval_info.get("interval_width_90", float("nan"))
                            print(
                                f"    Eval  NLL={nll:.6f}  "
                                f"Coverage@90={cov:.6f}  Width@90={wid:.6f}"
                            )
                            tracker.log_metrics(
                                {
                                    f"{model_type}/{round_name}/eval_nll": float(nll),
                                    f"{model_type}/{round_name}/eval_coverage90": float(cov),
                                    f"{model_type}/{round_name}/eval_width90": float(wid),
                                },
                                step=int(total_epochs),
                            )

                        save_forecast_plot(
                            gt_list[0], pred_list[0], output_cols,
                            model_dir / f"{round_name}_forecast.png",
                            title=f"{model_type.upper()} [{round_name}] "
                                  f"(horizon={eval_horizon})",
                            show_metrics=True,
                        )

                    # Recursive simulation
                    sim_horizon_val = sim_cfg.get("horizon", None)
                    if sim_horizon_val is None:
                        sim_horizon_val = (
                            len(val_dataset.values) - seq_len
                        )

                    if model_type in NEURAL_MODELS:
                        sim_result = simulate_recursive_neural(
                            model, val_dataset, seq_len,
                            sim_horizon_val, device,
                            start_idx=sim_start,
                            probabilistic_cfg=prob_train_cfg,
                        )
                    else:
                        sim_result = simulate_recursive_xgboost(
                            model, val_dataset, seq_len,
                            sim_horizon_val,
                            start_idx=sim_start,
                        )

                    if sim_result["n_steps"] > 0:
                        gt_s = sim_result["ground_truths"]
                        pr_s = sim_result["predictions"]
                        sim_mse = float(np.mean((gt_s - pr_s) ** 2))
                        sim_mae = float(np.mean(np.abs(gt_s - pr_s)))
                        print(f"    Sim   MSE={sim_mse:.6f}  "
                              f"MAE={sim_mae:.6f}  "
                              f"({sim_result['n_steps']} steps)")
                        tracker.log_metrics(
                            {
                                f"{model_type}/{round_name}/sim_mse": float(sim_mse),
                                f"{model_type}/{round_name}/sim_mae": float(sim_mae),
                            },
                            step=int(total_epochs),
                        )

                        save_per_model_simulation_plot(
                            sim_result, output_cols, model_type,
                            model_dir / f"{round_name}_simulation.png",
                            plot_cfg=plot_cfg,
                        )
                        save_per_model_simulation_csv(
                            sim_result, output_cols,
                            model_dir, round_name,
                        )

                except Exception as eval_exc:
                    print(f"  Eval/Sim warning: {eval_exc}")

            except Exception as exc:
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()

    # ── Cumulative loss plots ─────────────────────────────────────────
    for model_type in trained_models:
        if model_type in cumul_train_losses:
            model_dir = out_dir / model_type
            total_epochs = len(cumul_train_losses[model_type])
            save_loss_plot(
                cumul_train_losses[model_type],
                cumul_val_losses.get(model_type, []),
                model_dir / "loss_full.png",
                title=f"{model_type.upper()} – Full Training History "
                      f"({total_epochs} epochs)",
            )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    for mt, model in trained_models.items():
        n_p = model_n_params.get(mt, 0)
        rnd = model_last_round.get(mt, "train")
        train_hist = cumul_train_losses.get(mt, [])
        val_hist = cumul_val_losses.get(mt, [])
        te = len(train_hist)
        tl = train_hist[-1] if train_hist else None
        vl = val_hist[-1] if val_hist else None
        tt = cumul_time.get(mt, 0.0)
        tl_s = f"{tl:.6f}" if tl is not None else "n/a"
        vl_s = f"{vl:.6f}" if vl is not None else "n/a"
        print(f"  {mt:15s}  last_round={rnd:10s}  epochs={te:3d}  "
              f"train_loss={tl_s}  val_loss={vl_s}  "
              f"time={tt:.1f}s  params={n_p:,}")
    print(f"\n  Output directory : {out_dir}")
    print(f"  Scaler           : {scaler_path}")
    print(f"  Config           : {out_dir / 'config.yaml'}")
    print(f"{'=' * 70}\n")
    tracker.log_artifact(out_dir / "run_metadata.yaml")
    tracker.log_artifact(out_dir / "config.yaml")
    tracker.finish()


if __name__ == "__main__":
    main()
