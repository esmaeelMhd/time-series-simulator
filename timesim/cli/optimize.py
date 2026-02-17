#!/usr/bin/env python3
"""Hyperparameter optimization for TimeSim models using Optuna (Bayesian/TPE).

Optimizes one or more models for a dataset/config and saves:
- best params YAML
- trials CSV
- Optuna SQLite study DB (resume-capable)

Output directory:
  <runs_dir>/<dataset>[/<run_name>]/optuna/<model_type>/
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

from timesim.utils.config import load_config
from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.data.stamps import get_time_feature_columns
from timesim.data.sampling import (
    RandomStartFixedHorizon,
    RandomStartRandomHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
    StrideBasedSampling,
)
from timesim.models.factory import build_model, NEURAL_MODELS
from timesim.training import WorldModelTrainer

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is required for optimize.py. Install with: pip install optuna"
    ) from exc

try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna (Bayesian/TPE).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (supports _base chain)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional subset of model types to optimize")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override number of Optuna trials")
    p.add_argument("--timeout", type=int, default=None,
                   help="Override timeout per model in seconds")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs per trial")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Override train steps per epoch per trial")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cpu / cuda)")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed")
    p.add_argument(
        "--fast-mode",
        dest="fast_mode",
        action="store_true",
        help="Use fast-mode defaults for cheaper Optuna trials",
    )
    p.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable fast-mode defaults",
    )
    p.set_defaults(fast_mode=None)
    return p.parse_args()


def _build_sampling_strategy(
    config: Dict[str, Any], pred_len: int, horizon_override: Optional[int] = None
):
    tcfg = config["training"]
    dcfg = config["dataset"]
    scfg = tcfg.get("sampling", {})

    strategy_name = str(
        scfg.get("strategy", tcfg.get("sampling_strategy", "random_fixed"))
    ).lower()
    legacy_horizon = int(tcfg.get("sampling_horizon", pred_len))

    if strategy_name in {"random_fixed", "fixed"}:
        horizon = int(
            horizon_override if horizon_override is not None
            else scfg.get("horizon", legacy_horizon)
        )
        return RandomStartFixedHorizon(horizon=horizon)
    if strategy_name in {"random_random", "random"}:
        h_min = int(scfg.get("h_min", 1))
        h_max = int(scfg.get("h_max", legacy_horizon))
        return RandomStartRandomHorizon(h_min=h_min, h_max=h_max)
    if strategy_name in {"geometric", "geometric_horizon"}:
        h_max = int(scfg.get("h_max", legacy_horizon))
        return GeometricHorizonSampling(pred_len=pred_len, h_max=h_max)
    if strategy_name in {"daily_fixed", "daily"}:
        return DailyFixedHorizon(
            start_hour=int(scfg.get("start_hour", 0)),
            horizon=int(
                horizon_override if horizon_override is not None
                else scfg.get("horizon", legacy_horizon)
            ),
            samples_per_hour=int(scfg.get("samples_per_hour", dcfg.get("samples_per_hour", 1))),
        )
    if strategy_name in {"stride", "stride_based"}:
        return StrideBasedSampling(
            stride=int(scfg.get("stride", 12)),
            h_max=int(scfg.get("h_max", legacy_horizon)),
        )
    raise ValueError(f"Unknown sampling strategy '{strategy_name}'")


def _prepare_xgboost_data(dataset, seq_len: int):
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


def _suggest_overrides(
    trial: "optuna.trial.Trial",
    model_type: str,
    profile: str = "fast_gpu",
) -> Dict[str, Any]:
    """Model-specific hyperparameter search spaces.

    Profiles:
    - fast_gpu: narrowed ranges for practical throughput on consumer GPUs.
    - wide: broader exploratory ranges.
    """
    if profile not in {"fast_gpu", "wide"}:
        raise ValueError(f"Unknown search profile '{profile}'")

    if model_type == "lstm":
        if profile == "fast_gpu":
            hidden_choices = [32, 64, 96, 128]
            max_layers = 2
            max_dropout = 0.3
        else:
            hidden_choices = [32, 64, 96, 128, 160, 192, 224, 256]
            max_layers = 3
            max_dropout = 0.4
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", hidden_choices),
            "num_layers": trial.suggest_int("num_layers", 1, max_layers),
            "dropout": trial.suggest_float("dropout", 0.0, max_dropout, step=0.05),
        }
    if model_type == "dlinear":
        kernel_choices = [5, 9, 13, 17, 25] if profile == "fast_gpu" else list(range(3, 50, 2))
        return {
            "kernel_size": (
                trial.suggest_categorical("kernel_size", kernel_choices)
                if profile == "fast_gpu"
                else trial.suggest_int("kernel_size", 3, 49, step=2)
            ),
            "individual": trial.suggest_categorical("individual", [False, True]),
        }
    if model_type == "nlinear":
        return {
            "individual": trial.suggest_categorical("individual", [False, True]),
        }
    if model_type == "tft":
        hidden_choices = [32, 64, 96] if profile == "fast_gpu" else [32, 64, 96, 128]
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_choices)
        possible_heads = [h for h in [2, 4, 8] if hidden_dim % h == 0]
        if not possible_heads:
            possible_heads = [2]
        max_layers = 2 if profile == "fast_gpu" else 3
        max_dropout = 0.3 if profile == "fast_gpu" else 0.4
        return {
            "hidden_dim": hidden_dim,
            "n_heads": trial.suggest_categorical("n_heads", possible_heads),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, max_layers),
            "dropout": trial.suggest_float("dropout", 0.0, max_dropout, step=0.05),
        }
    if model_type == "transformer":
        d_model_choices = [32, 64, 96, 128] if profile == "fast_gpu" else [32, 64, 96, 128, 160, 192, 256]
        d_model = trial.suggest_categorical("d_model", d_model_choices)
        possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        if not possible_heads:
            possible_heads = [2]
        max_layers = 3 if profile == "fast_gpu" else 4
        ff_choices = [64, 128, 256] if profile == "fast_gpu" else [64, 128, 256, 512]
        max_dropout = 0.3 if profile == "fast_gpu" else 0.4
        return {
            "d_model": d_model,
            "nhead": trial.suggest_categorical("nhead", possible_heads),
            "num_layers": trial.suggest_int("num_layers", 1, max_layers),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", ff_choices),
            "dropout": trial.suggest_float("dropout", 0.0, max_dropout, step=0.05),
        }
    if model_type == "latent_ssm":
        hidden_choices = [32, 64, 96, 128] if profile == "fast_gpu" else [32, 64, 96, 128, 160, 192]
        latent_choices = [8, 16, 24, 32] if profile == "fast_gpu" else [8, 16, 24, 32, 48, 64]
        max_layers = 2 if profile == "fast_gpu" else 3
        max_dropout = 0.3 if profile == "fast_gpu" else 0.4
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", hidden_choices),
            "latent_dim": trial.suggest_categorical("latent_dim", latent_choices),
            "num_layers": trial.suggest_int("num_layers", 1, max_layers),
            "dropout": trial.suggest_float("dropout", 0.0, max_dropout, step=0.05),
            "min_scale": trial.suggest_float("min_scale", 1e-5, 1e-3, log=True),
            "min_df": trial.suggest_float("min_df", 2.01, 5.0),
        }
    if model_type == "xgboost":
        if profile == "fast_gpu":
            n_estimators_low, n_estimators_high = 50, 300
            max_depth_high = 8
            lr_low, lr_high = 1e-2, 2e-1
        else:
            n_estimators_low, n_estimators_high = 50, 500
            max_depth_high = 12
            lr_low, lr_high = 1e-3, 3e-1
        return {
            "n_estimators": trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, max_depth_high),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
        }
    return {}


def _sanitize_for_yaml(obj: Any):
    """Recursively convert NumPy types and non-finite floats for YAML safety."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def _normalize_best_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize best params for downstream config compatibility."""
    normalized = dict(best_params)
    optimizer_name = str(normalized.get("optimizer", "")).lower()

    if "weight_decay" not in normalized:
        if optimizer_name == "adamw" and "weight_decay_adamw" in normalized:
            normalized["weight_decay"] = normalized["weight_decay_adamw"]
        elif optimizer_name == "adam" and "weight_decay_adam" in normalized:
            normalized["weight_decay"] = normalized["weight_decay_adam"]
        elif "weight_decay_adamw" in normalized:
            normalized["weight_decay"] = normalized["weight_decay_adamw"]
        elif "weight_decay_adam" in normalized:
            normalized["weight_decay"] = normalized["weight_decay_adam"]

    normalized.pop("weight_decay_adamw", None)
    normalized.pop("weight_decay_adam", None)
    # KL warmup schedule is intentionally kept under training config control.
    normalized.pop("kl_beta_start", None)
    normalized.pop("kl_warmup_epochs", None)
    return normalized


def _maybe_compile_model(model, config: Dict[str, Any]):
    """Optionally compile a model with torch.compile."""
    if getattr(model, "_timesim_compiled", False):
        return model
    tcfg = config.get("training", {})
    if not bool(tcfg.get("use_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    # On Windows, Inductor requires MSVC compiler toolchain (`cl`).
    # If unavailable, avoid runtime compile failures and use eager mode.
    if sys.platform.startswith("win") and shutil.which("cl") is None:
        return model
    compile_mode = str(tcfg.get("compile_mode", "default"))
    try:
        compiled = torch.compile(model, mode=compile_mode)
        setattr(compiled, "_timesim_compiled", True)
        return compiled
    except Exception:
        return model


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device:
        config["misc"]["device"] = args.device

    seed = args.seed if args.seed is not None else config.get("misc", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = config.get("training", {})
    ocfg = config.get("optimization", {})
    dataset_cfg = config["dataset"]
    model_defaults_cfg = config.get("model_defaults", {})
    output_cfg = config.get("output", {})

    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None
    dataset_name = dataset_cfg["name"]

    run_dir = runs_dir / dataset_name
    if run_name is not None:
        run_dir = run_dir / run_name
    optuna_root = run_dir / "optuna"
    optuna_root.mkdir(parents=True, exist_ok=True)

    device = config.get("misc", {}).get("device", "cpu")
    seq_len = int(dataset_cfg["seq_len"])
    pred_len = int(dataset_cfg["pred_len"])
    batch_size = int(dataset_cfg["batch_size"])

    groups = dataset_cfg["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]
    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])

    all_input_features = set(input_cols) | set(output_cols)
    input_dim = len(all_input_features)
    add_time_features = bool(config.get("data", {}).get("add_time_features", False))
    time_features_cfg = config.get("data", {}).get("time_features", {}) or {}
    if isinstance(time_features_cfg, dict) and "enabled" in time_features_cfg:
        add_time_features = bool(time_features_cfg.get("enabled")) or add_time_features
    if add_time_features:
        input_dim += len(
            get_time_feature_columns(
                features=time_features_cfg.get("features"),
                encoding=time_features_cfg.get("encoding", "cyclical"),
            )
        )
    output_dim = len(output_cols)

    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}
    if args.models:
        model_names = args.models
    elif ocfg.get("models"):
        model_names = list(ocfg["models"])
    elif models_cfg_list:
        model_names = [m["type"] for m in models_cfg_list]
    else:
        model_names = [config.get("model", {}).get("type", "lstm")]

    n_trials = int(args.n_trials if args.n_trials is not None else ocfg.get("n_trials", 30))
    timeout = args.timeout if args.timeout is not None else ocfg.get("timeout", None)
    fast_mode = (
        args.fast_mode
        if args.fast_mode is not None
        else bool(ocfg.get("fast_mode", False))
    )
    if fast_mode:
        trial_epochs = int(
            args.epochs
            if args.epochs is not None
            else ocfg.get("fast_epochs", ocfg.get("epochs", 1))
        )
        trial_steps = (
            args.steps_per_epoch if args.steps_per_epoch is not None
            else ocfg.get("fast_steps_per_epoch", ocfg.get("steps_per_epoch", 40))
        )
        sampling_horizon_override = int(ocfg.get("fast_sampling_horizon", 24))
        training_mode = str(ocfg.get("fast_training_mode", "one_step"))
    else:
        trial_epochs = int(args.epochs if args.epochs is not None else ocfg.get("epochs", 5))
        trial_steps = (
            args.steps_per_epoch if args.steps_per_epoch is not None
            else ocfg.get("steps_per_epoch", tcfg.get("steps_per_epoch", None))
        )
        sampling_horizon_override = None
        training_mode = str(tcfg.get("mode", "multi_step"))
    sampler_seed = int(ocfg.get("seed", seed))
    direction = str(ocfg.get("direction", "minimize"))
    search_space_profile = str(ocfg.get("search_space_profile", "fast_gpu"))
    enable_pruning = bool(ocfg.get("enable_pruning", True))
    pruner_startup_trials = int(ocfg.get("pruner_startup_trials", 5))
    pruner_warmup_steps = int(ocfg.get("pruner_warmup_steps", 1))
    pruner_min_epochs_default = 2 if fast_mode else 1
    pruner_min_epochs = int(ocfg.get("pruner_min_epochs", pruner_min_epochs_default))
    pruner_min_epochs = max(1, min(pruner_min_epochs, trial_epochs))

    print("\n" + "=" * 70)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"  Config          : {args.config}")
    print(f"  Dataset         : {dataset_name}")
    print(f"  Run name        : {run_name if run_name is not None else '(default)'}")
    print(f"  Models          : {model_names}")
    print(f"  Trials/model    : {n_trials}")
    print(f"  Epochs/trial    : {trial_epochs}")
    print(f"  Steps/epoch     : {trial_steps if trial_steps is not None else 'auto'}")
    print(f"  Timeout/model   : {timeout if timeout is not None else 'none'}")
    print(f"  Device          : {device}")
    print(f"  Search profile  : {search_space_profile}")
    print(f"  Fast mode       : {fast_mode}")
    print(f"  Pruning         : {enable_pruning}")
    if enable_pruning:
        print(
            "  Pruner          : "
            f"startup={pruner_startup_trials}, "
            f"warmup_steps={pruner_warmup_steps}, "
            f"min_epochs={pruner_min_epochs}"
        )
    print("=" * 70 + "\n")

    # Build data once and reuse across all trials/models.
    index_col = dataset_cfg.get("index_col", config.get("data", {}).get("index_col", "date"))
    train_split = dataset_cfg.get("train_split", config.get("data", {}).get("train_split", 0.8))
    df = load_csv_dataset(
        dataset_cfg["csv"],
        index_col=index_col,
        slice_cfg=dataset_cfg.get("slice"),
    )
    train_loader, val_loader, scaler = build_grouped_dataloaders(
        df,
        groups,
        input_groups,
        output_groups,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        train_split=train_split,
        add_time=add_time_features,
        time_features_cfg=time_features_cfg,
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    sampling_strategy = _build_sampling_strategy(
        config, pred_len, horizon_override=sampling_horizon_override
    )

    def make_objective(model_type: str):
        per_model_cfg = models_cfg_map.get(model_type, {"type": model_type})

        if model_type in NEURAL_MODELS:
            def objective(trial: "optuna.trial.Trial") -> float:
                model_overrides = _suggest_overrides(trial, model_type, profile=search_space_profile)
                if search_space_profile == "fast_gpu":
                    lr = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
                else:
                    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
                prob_cfg_default = tcfg.get("probabilistic", {}) or {}
                if model_type == "latent_ssm":
                    # Keep rollout MSE contribution meaningful for trajectory quality.
                    elbo_weight = trial.suggest_float("elbo_weight", 0.5, 2.0)
                    kl_weight = trial.suggest_float("kl_weight", 0.05, 1.5)
                    rollout_mse_weight = trial.suggest_float("rollout_mse_weight", 0.5, 3.0)
                    # Keep KL warmup schedule fixed from training config.
                    kl_beta_start = float(prob_cfg_default.get("kl_beta_start", 1.0))
                    kl_beta_end = float(prob_cfg_default.get("kl_beta_end", 1.0))
                    kl_warmup_epochs = int(prob_cfg_default.get("kl_warmup_epochs", 1))
                else:
                    elbo_weight = float(prob_cfg_default.get("elbo_weight", 1.0))
                    kl_weight = float(prob_cfg_default.get("kl_weight", 1.0))
                    rollout_mse_weight = float(prob_cfg_default.get("rollout_mse_weight", 1.0))
                    kl_beta_start = float(prob_cfg_default.get("kl_beta_start", 1.0))
                    kl_beta_end = float(prob_cfg_default.get("kl_beta_end", 1.0))
                    kl_warmup_epochs = int(prob_cfg_default.get("kl_warmup_epochs", 1))

                one_step_weight = tcfg.get("one_step_weight", 0.5)
                if tcfg.get("mode", "multi_step") == "combined":
                    one_step_weight = trial.suggest_float("one_step_weight", 0.1, 0.9)

                teacher_forcing_ratio = tcfg.get("teacher_forcing_ratio", 0.0)
                if tcfg.get("feedback", "model") == "mixed":
                    teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.0, 0.6)

                optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
                if optimizer_name == "adamw":
                    # Keep per-optimizer parameter names so Optuna distributions
                    # remain compatible when resuming an existing study.
                    weight_decay = trial.suggest_float(
                        "weight_decay_adamw", 1e-6, 1e-2, log=True
                    )
                else:
                    # Keep Adam regularization mild to avoid over-penalizing dynamics.
                    weight_decay = trial.suggest_float("weight_decay_adam", 0.0, 1e-4)
                model = build_model(
                    model_type,
                    input_dim,
                    output_dim,
                    seq_len,
                    pred_len,
                    per_model_cfg=per_model_cfg,
                    model_defaults_cfg=model_defaults_cfg,
                    overrides=model_overrides,
                )
                model = _maybe_compile_model(model, config)

                if optimizer_name == "adamw":
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=lr, weight_decay=weight_decay
                    )
                else:
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=lr, weight_decay=weight_decay
                    )

                trainer = WorldModelTrainer(
                    model=model,
                    dataset=train_dataset,
                    val_dataset=val_dataset,
                    sampling_strategy=sampling_strategy,
                    warmup_len=tcfg.get("warmup_len", seq_len),
                    batch_size=batch_size,
                    loss_type=tcfg.get("loss_type", "mse"),
                    loss_weighting=tcfg.get("loss_weighting", "uniform"),
                    loss_weight_scale=tcfg.get("loss_weight_scale", 1.0),
                    shape_loss_cfg=tcfg.get("shape_loss", None),
                    training_mode=training_mode,
                    feedback=tcfg.get("feedback", "model"),
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    one_step_weight=one_step_weight,
                    optimizer=optimizer,
                    device=device,
                    use_amp=bool(tcfg.get("use_amp", False)),
                    early_stopping=tcfg.get("early_stopping", False),
                    patience=tcfg.get("patience", 5),
                    min_delta=tcfg.get("min_delta", 0.0),
                    run_dir=None,  # avoid trial artifact overhead
                    probabilistic_cfg={
                        "objective": prob_cfg_default.get("objective", "elbo_plus_rollout_mse"),
                        "elbo_weight": elbo_weight,
                        "kl_weight": kl_weight,
                        "rollout_mse_weight": rollout_mse_weight,
                        "kl_warmup_enabled": bool(prob_cfg_default.get("kl_warmup_enabled", False)),
                        "kl_beta_start": kl_beta_start,
                        "kl_beta_end": kl_beta_end,
                        "kl_warmup_epochs": kl_warmup_epochs,
                    },
                )

                best_score = float("inf")
                final_val = None
                for ep in range(trial_epochs):
                    _, val_losses = trainer.fit(
                        epochs=1,
                        steps_per_epoch=trial_steps,
                        verbose=False,
                    )
                    val_finite = [v for v in val_losses if v is not None and np.isfinite(v)]
                    if val_finite:
                        final_val = float(val_finite[-1])
                        best_score = min(best_score, final_val)
                        # Report best-so-far to reduce noisy prune decisions.
                        trial.report(best_score, step=ep)
                        if (
                            enable_pruning
                            and (ep + 1) >= pruner_min_epochs
                            and (ep + 1) < trial_epochs
                            and trial.should_prune()
                        ):
                            raise optuna.TrialPruned()
                if not np.isfinite(best_score):
                    return float("inf")

                score = float(best_score)
                if final_val is not None:
                    trial.set_user_attr("final_val_loss", final_val)
                trial.set_user_attr("best_val_loss", score)
                return score
            return objective

        def objective_xgb(trial: "optuna.trial.Trial") -> float:
            if not HAS_XGBOOST:
                raise RuntimeError("xgboost is not installed")
            model_overrides = _suggest_overrides(trial, model_type, profile=search_space_profile)
            model = build_model(
                model_type,
                input_dim,
                output_dim,
                seq_len,
                pred_len,
                per_model_cfg=per_model_cfg,
                model_defaults_cfg=model_defaults_cfg,
                overrides=model_overrides,
            )

            X_train, y_train = _prepare_xgboost_data(train_dataset, seq_len)
            X_val, y_val = _prepare_xgboost_data(val_dataset, seq_len)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            val_pred = model.predict(X_val)
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            return val_mse

        return objective_xgb

    summary_path = optuna_root / "summary.yaml"
    all_summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded_summary = yaml.safe_load(f) or {}
            if isinstance(loaded_summary, dict):
                all_summary = loaded_summary
            else:
                print(
                    f"Warning: existing summary is not a mapping ({summary_path}); "
                    "starting from empty summary."
                )
        except Exception as exc:
            print(f"Warning: failed to read existing summary ({summary_path}): {exc}")
            all_summary = {}

    for model_type in model_names:
        print(f"\n--- Optimizing {model_type} ---")
        if model_type == "xgboost" and not HAS_XGBOOST:
            print("Skipped: xgboost not installed")
            continue

        model_opt_dir = optuna_root / model_type
        model_opt_dir.mkdir(parents=True, exist_ok=True)
        db_path = model_opt_dir / "study.db"
        storage = f"sqlite:///{db_path.as_posix()}"
        study_name = f"{dataset_name}_{run_name or 'default'}_{model_type}"

        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        if enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruner_startup_trials,
                n_warmup_steps=pruner_warmup_steps,
            )
        else:
            pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        objective = make_objective(model_type)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = {
            "study_name": study_name,
            "model_type": model_type,
            "direction": direction,
            "n_trials_total": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": float(study.best_value),
            "best_params": _sanitize_for_yaml(_normalize_best_params(study.best_params)),
        }

        with open(model_opt_dir / "best_params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best, f, sort_keys=False)

        try:
            df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
            df_trials.to_csv(model_opt_dir / "trials.csv", index=False)
        except Exception as exc:
            print(f"Warning: failed to export trials CSV for {model_type}: {exc}")

        all_summary[model_type] = best
        print(f"Best value: {best['best_value']:.6f}")
        print(f"Best params: {best['best_params']}")

    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_sanitize_for_yaml(all_summary), f, sort_keys=False)

    print("\n" + "=" * 70)
    print("  OPTUNA COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {optuna_root}")
    print(f"  Summary         : {summary_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
