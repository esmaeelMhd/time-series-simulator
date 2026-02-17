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
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

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
from timesim.training import WorldModelTrainer
from timesim.utils.plotting import save_loss_plot, save_forecast_plot
from timesim.models.factory import build_model, count_parameters, NEURAL_MODELS

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


MODEL_PARAM_KEYS_BY_TYPE = {
    "lstm": {"hidden_dim", "num_layers", "dropout"},
    "dlinear": {"kernel_size", "individual"},
    "nlinear": {"individual"},
    "tft": {"hidden_dim", "n_heads", "num_lstm_layers", "dropout"},
    "transformer": {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout"},
    "latent_ssm": {"hidden_dim", "latent_dim", "num_layers", "dropout", "min_scale", "min_df"},
    "xgboost": {"strategy", "n_estimators", "max_depth", "learning_rate"},
}

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
    "elbo_weight",
    "kl_weight",
    "rollout_mse_weight",
    "objective",
    "kl_warmup_enabled",
    "kl_beta_end",
}


# ─────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────

def build_sampling_strategy(config, pred_len):
    """Build sampling strategy from training config."""
    tcfg = config["training"]
    dcfg = config["dataset"]
    scfg = tcfg.get("sampling", {})

    strategy_name = str(
        scfg.get("strategy", tcfg.get("sampling_strategy", "random_fixed"))
    ).lower()

    # Backward-compatible path for existing configs
    legacy_horizon = int(tcfg.get("sampling_horizon", pred_len))

    if strategy_name in {"random_fixed", "fixed"}:
        horizon = int(scfg.get("horizon", legacy_horizon))
        return RandomStartFixedHorizon(horizon=horizon), f"random_fixed(horizon={horizon})"

    if strategy_name in {"random_random", "random"}:
        h_min = int(scfg.get("h_min", 1))
        h_max = int(scfg.get("h_max", legacy_horizon))
        return RandomStartRandomHorizon(h_min=h_min, h_max=h_max), (
            f"random_random(h_min={h_min}, h_max={h_max})"
        )

    if strategy_name in {"geometric", "geometric_horizon"}:
        h_max = int(scfg.get("h_max", legacy_horizon))
        return GeometricHorizonSampling(pred_len=pred_len, h_max=h_max), (
            f"geometric(pred_len={pred_len}, h_max={h_max})"
        )

    if strategy_name in {"daily_fixed", "daily"}:
        start_hour = int(scfg.get("start_hour", 0))
        horizon = int(scfg.get("horizon", legacy_horizon))
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
        h_max = int(scfg.get("h_max", legacy_horizon))
        return StrideBasedSampling(stride=stride, h_max=h_max), (
            f"stride(stride={stride}, h_max={h_max})"
        )

    raise ValueError(
        f"Unknown sampling strategy '{strategy_name}'. "
        "Use one of: random_fixed, random_random, geometric, daily_fixed, stride"
    )


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
    warmup_len = tcfg.get("warmup_len", seq_len)
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
    if "elbo_weight" in training_overrides:
        prob_cfg["elbo_weight"] = training_overrides["elbo_weight"]
    if "kl_weight" in training_overrides:
        prob_cfg["kl_weight"] = training_overrides["kl_weight"]
    if "rollout_mse_weight" in training_overrides:
        prob_cfg["rollout_mse_weight"] = training_overrides["rollout_mse_weight"]
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
        run_dir=model_dir,
        probabilistic_cfg=prob_cfg,
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
    model_keys = MODEL_PARAM_KEYS_BY_TYPE.get(model_type, set())
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
    """Optionally compile a model with torch.compile when configured."""
    if getattr(model, "_timesim_compiled", False):
        return model
    tcfg = config.get("training", {})
    if not bool(tcfg.get("use_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        print("  Warning: torch.compile not available in this torch build.")
        return model
    # On Windows, Inductor requires MSVC compiler toolchain (`cl`).
    # If unavailable, avoid runtime compile failures and use eager mode.
    if sys.platform.startswith("win") and shutil.which("cl") is None:
        print("  Warning: torch.compile requested but MSVC 'cl' was not found; using eager mode.")
        return model
    compile_mode = str(tcfg.get("compile_mode", "default"))
    try:
        compiled = torch.compile(model, mode=compile_mode)
        setattr(compiled, "_timesim_compiled", True)
        return compiled
    except Exception as exc:
        print(f"  Warning: torch.compile failed, using eager mode ({exc})")
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


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train world models for time-series simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (supports _base chain)")
    parser.add_argument("--models", nargs="*",
                        help="Override: train only these model types")
    parser.add_argument("--epochs", type=int,
                        help="Override epochs for all rounds")
    parser.add_argument("--steps-per-epoch", type=int,
                        help="Override steps per epoch for all rounds")
    parser.add_argument("--device", type=str,
                        help="Override device (cpu / cuda)")
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
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config (with _base chain) ────────────────────────────────
    config = load_config(args.config)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["misc"]["device"] = args.device
    if args.use_optuna_best_params is not None:
        config.setdefault("training", {})
        config["training"]["use_optuna_best_params"] = args.use_optuna_best_params
    if args.optuna_summary:
        config.setdefault("training", {})
        config["training"]["optuna_summary_path"] = args.optuna_summary

    seed = config["misc"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config["misc"].get("device", "cpu")

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
    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None

    # ── Determine models ──────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}

    if args.models:
        model_names = args.models
    elif models_cfg_list:
        model_names = [m["type"] for m in models_cfg_list]
    else:
        model_names = [config.get("model", {}).get("type", "lstm")]

    # ── Dimension setup ───────────────────────────────────────────────
    groups = config["dataset"]["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])

    seq_len = config["dataset"]["seq_len"]
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
        if args.models:
            for rc in training_rounds:
                rc["models"] = model_names

    # ── Print banner ──────────────────────────────────────────────────
    round_desc = " -> ".join(rc.get("name", "?") for rc in training_rounds)
    print("\n" + "=" * 70)
    print("  MULTI-MODEL TRAINING")
    print("=" * 70)
    print(f"  Config       : {args.config}")
    print(f"  Rounds       : {round_desc}")
    print(f"  Models       : {model_names}")
    print(f"  Device       : {device}")
    print(f"  Input cols   : {input_cols}")
    print(f"  Output cols  : {output_cols}")
    print(f"  input_dim={input_dim}  output_dim={output_dim}")
    if add_time_features:
        print(f"  Time features: enabled ({time_features_cfg})")
    print(f"  seq_len={seq_len}  pred_len={pred_len}")
    print(f"  Run name     : {run_name if run_name is not None else '(default)'}")
    print("=" * 70 + "\n")

    # ── Eval / simulation dimensions ──────────────────────────────────
    control_cols = groups.get("control", [])
    exo_cols_list = groups.get("exogenous", [])
    control_dim = len([c for c in input_cols if c in control_cols])
    exo_dim = len([c for c in input_cols if c in exo_cols_list])

    warmup_len = config["training"].get("warmup_len", seq_len)
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
    optuna_summary_path = _resolve_optuna_summary_path(config, out_dir, args.optuna_summary)
    optuna_best_by_model: Dict[str, Dict[str, Any]] = {}
    if use_optuna_best_params:
        print("Loading Optuna best params...")
        print(f"  Summary path: {optuna_summary_path}")
        optuna_best_by_model = _load_optuna_best_params(optuna_summary_path)
        print(f"  Models with best params: {sorted(optuna_best_by_model.keys())}")

    # Save resolved config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Pre-create per-model directories
    for rc in training_rounds:
        for mt in rc.get("models", model_names):
            (out_dir / mt).mkdir(exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    index_col = config["dataset"].get("index_col",
                                       data_cfg.get("index_col", "date"))
    train_split = config["dataset"].get("train_split",
                                         data_cfg.get("train_split", 0.8))

    print("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        slice_cfg=config["dataset"].get("slice"),
    )
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    train_loader, val_loader, scaler = build_grouped_dataloaders(
        df, groups, input_groups, output_groups,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=config["dataset"]["batch_size"],
        train_split=train_split,
        add_time=add_time_features,
        time_features_cfg=time_features_cfg,
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Save scaler ───────────────────────────────────────────────────
    from joblib import dump
    scaler_path = out_dir / "scaler.pkl"
    dump(scaler, scaler_path)
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
        if args.epochs:
            round_epochs = args.epochs
        if args.steps_per_epoch is not None:
            round_steps_per_epoch = args.steps_per_epoch

        print(f"\n{'═' * 70}")
        print(f"  ROUND: {round_name.upper()} "
              f"({round_epochs} epochs, LR={round_lr})")
        print(f"  Steps/epoch: {round_steps_per_epoch if round_steps_per_epoch is not None else 'auto'}")
        print(f"  Models: {round_model_types}")
        if checkpoint_dir:
            print(f"  Loading checkpoints from: {checkpoint_dir}")
        print(f"{'═' * 70}")

        for model_type in round_model_types:
            mc = models_cfg_map.get(model_type, {"type": model_type})
            model_dir = out_dir / model_type
            model_lr = round_lr
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
                    model.load_state_dict(torch.load(ckpt, map_location=device))
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

                        def checkpoint_callback(epoch_idx: int, improved_val_loss: float):
                            if checkpoint_sim_horizon <= 0:
                                return
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
                                    f"(epoch={epoch_idx}, val_loss={improved_val_loss:.6f})"
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
                    "elbo_weight",
                    "kl_weight",
                    "rollout_mse_weight",
                    "objective",
                    "kl_warmup_enabled",
                    "kl_beta_start",
                    "kl_beta_end",
                    "kl_warmup_epochs",
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
                        if bool(eval_info.get("is_probabilistic", False)):
                            nll = eval_info.get("rollout_nll", float("nan"))
                            cov = eval_info.get("coverage_90", float("nan"))
                            wid = eval_info.get("interval_width_90", float("nan"))
                            print(
                                f"    Eval  NLL={nll:.6f}  "
                                f"Coverage@90={cov:.6f}  Width@90={wid:.6f}"
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
        te = len(cumul_train_losses.get(mt, []))
        tl = cumul_train_losses.get(mt, [None])[-1]
        vl = cumul_val_losses.get(mt, [None])[-1]
        tt = cumul_time.get(mt, 0.0)
        print(f"  {mt:15s}  last_round={rnd:10s}  epochs={te:3d}  "
              f"train_loss={tl:.6f}  val_loss={vl:.6f}  "
              f"time={tt:.1f}s  params={n_p:,}")
    print(f"\n  Output directory : {out_dir}")
    print(f"  Scaler           : {scaler_path}")
    print(f"  Config           : {out_dir / 'config.yaml'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
