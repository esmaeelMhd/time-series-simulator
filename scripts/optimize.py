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
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

from timesim.utils.config import load_config
from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
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
    return p.parse_args()


def _build_sampling_strategy(config: Dict[str, Any], pred_len: int):
    tcfg = config["training"]
    dcfg = config["dataset"]
    scfg = tcfg.get("sampling", {})

    strategy_name = str(
        scfg.get("strategy", tcfg.get("sampling_strategy", "random_fixed"))
    ).lower()
    legacy_horizon = int(tcfg.get("sampling_horizon", pred_len))

    if strategy_name in {"random_fixed", "fixed"}:
        horizon = int(scfg.get("horizon", legacy_horizon))
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
            horizon=int(scfg.get("horizon", legacy_horizon)),
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


def _suggest_overrides(trial: "optuna.trial.Trial", model_type: str) -> Dict[str, Any]:
    """Model-specific hyperparameter search spaces."""
    if model_type == "lstm":
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        }
    if model_type == "dlinear":
        return {
            "kernel_size": trial.suggest_int("kernel_size", 3, 49, step=2),
            "individual": trial.suggest_categorical("individual", [False, True]),
        }
    if model_type == "nlinear":
        return {
            "individual": trial.suggest_categorical("individual", [False, True]),
        }
    if model_type == "tft":
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
        possible_heads = [h for h in [2, 4, 8] if hidden_dim % h == 0]
        if not possible_heads:
            possible_heads = [2]
        return {
            "hidden_dim": hidden_dim,
            "n_heads": trial.suggest_categorical("n_heads", possible_heads),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        }
    if model_type == "transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128, 160, 192, 256])
        possible_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        if not possible_heads:
            possible_heads = [2]
        return {
            "d_model": d_model,
            "nhead": trial.suggest_categorical("nhead", possible_heads),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        }
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
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
    trial_epochs = int(args.epochs if args.epochs is not None else ocfg.get("epochs", 5))
    trial_steps = (
        args.steps_per_epoch if args.steps_per_epoch is not None
        else ocfg.get("steps_per_epoch", tcfg.get("steps_per_epoch", None))
    )
    sampler_seed = int(ocfg.get("seed", seed))
    direction = str(ocfg.get("direction", "minimize"))

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
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    sampling_strategy = _build_sampling_strategy(config, pred_len)

    def make_objective(model_type: str):
        per_model_cfg = models_cfg_map.get(model_type, {"type": model_type})

        if model_type in NEURAL_MODELS:
            def objective(trial: "optuna.trial.Trial") -> float:
                model_overrides = _suggest_overrides(trial, model_type)
                lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

                one_step_weight = tcfg.get("one_step_weight", 0.5)
                if tcfg.get("mode", "multi_step") == "combined":
                    one_step_weight = trial.suggest_float("one_step_weight", 0.1, 0.9)

                teacher_forcing_ratio = tcfg.get("teacher_forcing_ratio", 0.0)
                if tcfg.get("feedback", "model") == "mixed":
                    teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.0, 0.6)

                optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
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

                if optimizer_name == "adamw":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                    training_mode=tcfg.get("mode", "multi_step"),
                    feedback=tcfg.get("feedback", "model"),
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    one_step_weight=one_step_weight,
                    optimizer=optimizer,
                    device=device,
                    early_stopping=tcfg.get("early_stopping", False),
                    patience=tcfg.get("patience", 5),
                    min_delta=tcfg.get("min_delta", 0.0),
                    run_dir=None,  # avoid trial artifact overhead
                )

                _, val_losses = trainer.fit(
                    epochs=trial_epochs,
                    steps_per_epoch=trial_steps,
                    verbose=False,
                )
                val_finite = [v for v in val_losses if v is not None and np.isfinite(v)]
                if not val_finite:
                    return float("inf")

                score = float(min(val_finite))
                trial.set_user_attr("final_val_loss", float(val_finite[-1]))
                trial.set_user_attr("best_val_loss", score)
                return score
            return objective

        def objective_xgb(trial: "optuna.trial.Trial") -> float:
            if not HAS_XGBOOST:
                raise RuntimeError("xgboost is not installed")
            model_overrides = _suggest_overrides(trial, model_type)
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

    all_summary: Dict[str, Any] = {}
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
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
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
            "best_params": _sanitize_for_yaml(study.best_params),
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

    summary_path = optuna_root / "summary.yaml"
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
