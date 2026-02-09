#!/usr/bin/env python3
"""Compare multiple time series models on the same dataset.

Train and evaluate different model architectures (LSTM, DLinear, NLinear,
TFT, Transformer, XGBoost) using the same seq-to-seq world model framework.

All models use:
- Lookback window (seq_len) as input
- Prediction window (pred_len, normally 1) as output
- Variable categories: control, exogenous, objective
- Autoregressive rollout for multi-step evaluation

Produces:
- Per-model training loss plots and forecast plots
- Comparative figures showing all models side by side
- CSV file with metrics for all models

Usage:
    python examples/wastewater/compare_models.py --config configs/test_small.yml
    python examples/wastewater/compare_models.py --config configs/test_small.yml --models lstm dlinear
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch

from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.training import WorldModelTrainer
from timesim.utils.plotting import save_loss_plot, save_forecast_plot

# Try importing XGBoost
try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Try importing matplotlib for comparison plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────

NEURAL_MODELS = {"lstm", "dlinear", "nlinear", "tft", "transformer"}


def build_model(model_type, model_params, input_dim, output_dim, seq_len, pred_len):
    """Create a model instance from type name and parameters.

    Returns a WorldModelBase (neural) or XGBoostForecaster (tree-based).
    """
    if model_type == "lstm":
        from timesim.models.lstm import LSTMWorldModel
        return LSTMWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=model_params.get("hidden_dim", 64),
            num_layers=model_params.get("num_layers", 2),
            dropout=model_params.get("dropout", 0.0),
            pred_len=pred_len,
        )
    elif model_type == "dlinear":
        from timesim.models.dlinear import DLinearWorldModel
        return DLinearWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=model_params.get("kernel_size", 25),
            individual=model_params.get("individual", False),
        )
    elif model_type == "nlinear":
        from timesim.models.nlinear import NLinearWorldModel
        return NLinearWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=model_params.get("individual", False),
        )
    elif model_type == "tft":
        from timesim.models.tft import TFTWorldModel
        return TFTWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=model_params.get("hidden_dim", 64),
            n_heads=model_params.get("n_heads", 4),
            num_lstm_layers=model_params.get("num_lstm_layers", 2),
            dropout=model_params.get("dropout", 0.1),
        )
    elif model_type == "transformer":
        from timesim.models.transformer import TransformerWorldModel
        return TransformerWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=model_params.get("d_model", 64),
            nhead=model_params.get("nhead", 4),
            num_layers=model_params.get("num_layers", 2),
            dim_feedforward=model_params.get("dim_feedforward", 128),
            dropout=model_params.get("dropout", 0.1),
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. pip install xgboost")
        return XGBoostForecaster(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            output_dim=output_dim,
            strategy="recursive",
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 6),
            learning_rate=model_params.get("learning_rate", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters for neural models."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0  # non-neural models


# ─────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────

def train_neural_model(model, train_dataset, val_dataset, config, device, model_dir):
    """Train a neural WorldModel using WorldModelTrainer. Returns (train_losses, val_losses)."""
    model_dir.mkdir(parents=True, exist_ok=True)

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    epochs = config["training"]["epochs"]
    batch_size = config["dataset"]["batch_size"]
    lr = config["training"].get("learning_rate", 1e-3)
    warmup_len = config["training"].get("warmup_len", seq_len)

    sampling_horizon = config["training"].get("sampling_horizon", pred_len)
    sampling = RandomStartFixedHorizon(horizon=sampling_horizon)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = WorldModelTrainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        sampling_strategy=sampling,
        warmup_len=warmup_len,
        batch_size=batch_size,
        loss_type=config["training"].get("loss_type", "mse"),
        training_mode=config["training"].get("mode", "multi_step"),
        feedback=config["training"].get("feedback", "model"),
        teacher_forcing_ratio=config["training"].get("teacher_forcing_ratio", 0.0),
        optimizer=optimizer,
        device=device,
        run_dir=model_dir,
    )

    train_losses, val_losses = trainer.fit(epochs=epochs, verbose=True)

    # Save checkpoint
    trainer.save(model_dir / "checkpoint.pth")

    return train_losses, val_losses


def prepare_xgboost_data(dataset, seq_len):
    """Prepare (X, y) arrays for XGBoost from a GroupedTimeSeriesDataset.

    X : (N, seq_len, all_features)   – lookback window
    y : (N, 1, output_dim)           – next-step target
    """
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

    # Compute MSE on train / val
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_mse = float(np.mean((train_pred - y_train) ** 2))
    val_mse = float(np.mean((val_pred - y_val) ** 2))

    print(f"  train MSE={train_mse:.6f}  val MSE={val_mse:.6f}")

    model.save(str(model_dir / "model.pkl"))

    # Return as single-element lists (one "epoch")
    return [train_mse], [val_mse]


# ─────────────────────────────────────────────────────────────────────
# Evaluation (rollout)
# ─────────────────────────────────────────────────────────────────────

def evaluate_neural_model(
    model, val_dataset, warmup_len, eval_horizon,
    control_dim, exo_dim, device, n_windows=4,
):
    """Evaluate a neural model on multiple validation windows via rollout.

    During rollout:
    - Real values are used for control / exogenous variables
    - Model predictions are fed back for output variables

    Returns (gt_list, pred_list) – each a list of (eval_horizon, output_dim) arrays.
    """
    model.eval()
    val_data = val_dataset.values
    val_len = len(val_data)

    min_required = warmup_len + eval_horizon
    if val_len < min_required:
        print(f"  Warning: val data too short ({val_len} < {min_required})")
        return [], []

    max_start = val_len - min_required
    step = max(1, max_start // n_windows)
    start_indices = list(range(0, max_start, step))[:n_windows]

    gt_list, pred_list = [], []
    with torch.no_grad():
        for start_idx in start_indices:
            warmup_end = start_idx + warmup_len
            horizon_end = warmup_end + eval_horizon

            warmup_data = val_data[start_idx:warmup_end]
            horizon_data = val_data[warmup_end:horizon_end]

            warmup_inputs = warmup_data[:, val_dataset.in_idx]
            warmup_outputs = warmup_data[:, val_dataset.out_idx]
            horizon_inputs = horizon_data[:, val_dataset.in_idx]
            horizon_outputs = horizon_data[:, val_dataset.out_idx]

            # Build warmup tensor (input_cols + output_cols concatenated)
            warmup_full = np.concatenate([warmup_inputs, warmup_outputs], axis=-1)
            warmup_tensor = torch.from_numpy(warmup_full).float().unsqueeze(0).to(device)

            # Split horizon inputs into controls and exogenous
            controls_np = horizon_inputs[:, :control_dim]
            exo_np = (
                horizon_inputs[:, control_dim : control_dim + exo_dim]
                if exo_dim > 0
                else np.zeros((eval_horizon, 0), dtype=np.float32)
            )

            controls_t = torch.from_numpy(controls_np).float().unsqueeze(0).to(device)
            exo_t = torch.from_numpy(exo_np).float().unsqueeze(0).to(device)

            result = model.rollout(
                warmup_seq={"inputs": warmup_tensor},
                rollout_inputs={"controls": controls_t, "exogenous": exo_t},
                horizon=eval_horizon,
            )

            predictions = result["predictions"].squeeze(0).cpu().numpy()
            gt_list.append(horizon_outputs)
            pred_list.append(predictions)

    return gt_list, pred_list


def evaluate_xgboost_model(
    model, val_dataset, seq_len, eval_horizon, n_windows=4,
):
    """Evaluate XGBoost model via recursive rollout on validation data.

    During rollout:
    - Real values for non-output features at each step
    - Predicted values for output features (autoregressive)
    """
    val_data = val_dataset.values
    out_idx = val_dataset.out_idx
    val_len = len(val_data)

    min_required = seq_len + eval_horizon
    if val_len < min_required:
        print(f"  Warning: val data too short ({val_len} < {min_required})")
        return [], []

    max_start = val_len - min_required
    step = max(1, max_start // n_windows)
    start_indices = list(range(0, max_start, step))[:n_windows]

    gt_list, pred_list = [], []

    for start_idx in start_indices:
        lookback = val_data[start_idx : start_idx + seq_len]
        horizon_data = val_data[start_idx + seq_len : start_idx + seq_len + eval_horizon]
        gt = horizon_data[:, out_idx]

        current_input = lookback.copy()[np.newaxis, :, :]  # (1, seq_len, F)
        preds = []

        for h in range(eval_horizon):
            X_flat = current_input.reshape(1, -1)
            step_pred = np.zeros((1, model.output_dim))
            for out_i, m in enumerate(model.models_):
                step_pred[:, out_i] = m.predict(X_flat)
            preds.append(step_pred[0])

            # Slide window for next step
            if h < eval_horizon - 1:
                real_idx = start_idx + seq_len + h
                if real_idx < val_len:
                    new_step = val_data[real_idx].copy()
                else:
                    new_step = current_input[0, -1, :].copy()
                # Override output features with predictions
                for oi, idx in enumerate(out_idx):
                    new_step[idx] = step_pred[0, oi]
                current_input = np.concatenate(
                    [current_input[:, 1:, :], new_step[np.newaxis, np.newaxis, :]], axis=1
                )

        pred_list.append(np.array(preds))
        gt_list.append(gt)

    return gt_list, pred_list


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette for up to 8 models
MODEL_COLORS = [
    "#2E86AB",  # blue
    "#E94F37",  # red
    "#2CA58D",  # teal
    "#F18F01",  # orange
    "#A23B72",  # purple
    "#84BC9C",  # sage
    "#BC5D2E",  # brown
    "#6F2DBD",  # violet
]


def save_comparison_forecast_plot(all_results, output_cols, out_path, eval_horizon):
    """Overlay forecasts from all models on the same axes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())
    n_vars = len(output_cols)

    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3.5 * n_vars), sharex=True, squeeze=False)
    axes = axes.flatten()

    for v in range(n_vars):
        ax = axes[v]

        # Ground truth from first model that has data
        gt = None
        for mn in model_names:
            if all_results[mn]["gt_list"]:
                gt = all_results[mn]["gt_list"][0]
                break
        if gt is None:
            continue

        steps = np.arange(gt.shape[0])
        ax.plot(steps, gt[:, v], color="#1B1B1E", linewidth=2.5, label="Ground Truth", zorder=10)

        for i, mn in enumerate(model_names):
            if not all_results[mn]["pred_list"]:
                continue
            pred = all_results[mn]["pred_list"][0]
            mse = np.mean((gt[:, v] - pred[:, v]) ** 2)
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(
                steps, pred[:, v], color=color, linewidth=1.5,
                linestyle="--", alpha=0.85, label=f"{mn} (MSE={mse:.4f})",
            )

        ax.set_ylabel(output_cols[v], fontweight="medium")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Time Step", fontweight="medium")
    fig.suptitle(
        f"Model Comparison – Recursive Rollout (horizon={eval_horizon})",
        fontweight="bold", fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def save_comparison_loss_plot(all_results, out_path):
    """Side-by-side train / val loss curves for all models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, mn in enumerate(model_names):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        tl = all_results[mn]["train_losses"]
        vl = all_results[mn]["val_losses"]
        epochs = np.arange(1, len(tl) + 1)

        ax1.plot(
            epochs, tl, color=color, linewidth=2, label=mn,
            marker="o" if len(epochs) <= 20 else None, markersize=4,
        )

        valid_vl = [v for v in vl if v is not None]
        if valid_vl:
            ve = [j + 1 for j, v in enumerate(vl) if v is not None]
            ax2.plot(
                ve, valid_vl, color=color, linewidth=2, label=mn,
                marker="s" if len(ve) <= 20 else None, markersize=4,
            )

    for ax, title in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Epoch", fontweight="medium")
        ax.set_ylabel("Loss", fontweight="medium")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def save_metrics_bar_chart(all_results, out_path):
    """Bar chart comparing rollout MSE / MAE across models."""
    plt.style.use("seaborn-v0_8-whitegrid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())
    mse_vals = [all_results[mn]["mean_mse"] for mn in model_names]
    mae_vals = [all_results[mn]["mean_mae"] for mn in model_names]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(model_names))]

    x = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, mse_vals, width, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontweight="medium")
    ax1.set_ylabel("MSE", fontweight="medium")
    ax1.set_title("Rollout MSE by Model", fontweight="bold")
    for bar, val in zip(bars1, mse_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=8,
        )

    bars2 = ax2.bar(x, mae_vals, width, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontweight="medium")
    ax2.set_ylabel("MAE", fontweight="medium")
    ax2.set_title("Rollout MAE by Model", fontweight="bold")
    for bar, val in zip(bars2, mae_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple time series models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/test_small.yml",
                        help="Path to YAML config file")
    parser.add_argument("--models", nargs="*",
                        help="Override: list of model types to train (e.g. lstm dlinear)")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--device", type=str, help="Override device (cpu / cuda)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["misc"]["device"] = args.device

    seed = config["misc"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config["misc"].get("device", "cpu")

    # ── Determine models to train ────────────────────────────────────
    # Build lookup from config models list
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}

    if args.models:
        selected_types = args.models
    else:
        if models_cfg_list:
            selected_types = [m["type"] for m in models_cfg_list]
        else:
            selected_types = [config.get("model", {}).get("type", "lstm")]

    # Build final model configs (merge per-model params)
    model_configs = []
    for mt in selected_types:
        params = dict(models_cfg_map.get(mt, {}))
        params["type"] = mt
        model_configs.append(params)

    model_names = [mc["type"] for mc in model_configs]

    # ── Dimension / data setup ───────────────────────────────────────
    groups = config["dataset"]["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    input_dim = len(input_cols) + len(output_cols)
    output_dim = len(output_cols)

    control_cols = groups.get("control", [])
    exo_cols_list = groups.get("exogenous", [])
    control_dim = len([c for c in input_cols if c in control_cols])
    exo_dim = len([c for c in input_cols if c in exo_cols_list])

    warmup_len = config["training"].get("warmup_len", seq_len)
    eval_horizon = config.get("evaluation", {}).get("horizon", max(pred_len, 12))
    n_windows = config.get("evaluation", {}).get("n_windows", 4)

    # ── Print banner ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MULTI-MODEL TIME SERIES COMPARISON")
    print("=" * 70)
    print(f"  Models       : {model_names}")
    print(f"  Device       : {device}")
    print(f"  Input cols   : {input_cols}")
    print(f"  Output cols  : {output_cols}")
    print(f"  input_dim={input_dim}  output_dim={output_dim}")
    print(f"  control_dim={control_dim}  exo_dim={exo_dim}")
    print(f"  seq_len={seq_len}  pred_len={pred_len}  eval_horizon={eval_horizon}")
    print("=" * 70 + "\n")

    # ── Create output directory ──────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / config["dataset"]["name"] / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # ── Load dataset ─────────────────────────────────────────────────
    print("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=config["dataset"]["index_col"],
        slice_cfg=config["dataset"].get("slice"),
    )
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    train_loader, val_loader, scaler = build_grouped_dataloaders(
        df, groups, input_groups, output_groups,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=config["dataset"]["batch_size"],
        train_split=config["dataset"].get("train_split", 0.8),
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Train & evaluate each model ──────────────────────────────────
    all_results = {}

    for mc in model_configs:
        model_type = mc["type"]
        model_params = {k: v for k, v in mc.items() if k != "type"}
        model_dir = out_dir / model_type

        print(f"\n{'─' * 70}")
        print(f"  MODEL: {model_type.upper()}")
        print(f"{'─' * 70}")

        # Skip XGBoost if not installed
        if model_type == "xgboost" and not HAS_XGBOOST:
            print("  SKIPPED – xgboost not installed")
            continue

        t0 = time.time()

        try:
            model = build_model(model_type, model_params, input_dim, output_dim, seq_len, pred_len)
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}" if n_params else "  Parameters: N/A (tree-based)")

            # ── Train ────────────────────────────────────────────────
            if model_type in NEURAL_MODELS:
                train_losses, val_losses = train_neural_model(
                    model, train_dataset, val_dataset, config, device, model_dir,
                )
            else:
                train_losses, val_losses = train_xgboost_model(
                    model, train_dataset, val_dataset, config, model_dir,
                )

            # ── Evaluate ─────────────────────────────────────────────
            print(f"\n  Evaluating {model_type} (rollout horizon={eval_horizon})...")
            if model_type in NEURAL_MODELS:
                gt_list, pred_list = evaluate_neural_model(
                    model, val_dataset, warmup_len, eval_horizon,
                    control_dim, exo_dim, device, n_windows,
                )
            else:
                gt_list, pred_list = evaluate_xgboost_model(
                    model, val_dataset, seq_len, eval_horizon, n_windows,
                )

            # Metrics
            if gt_list and pred_list:
                all_mse = [float(np.mean((g - p) ** 2)) for g, p in zip(gt_list, pred_list)]
                all_mae = [float(np.mean(np.abs(g - p))) for g, p in zip(gt_list, pred_list)]
                mean_mse = float(np.mean(all_mse))
                mean_mae = float(np.mean(all_mae))
            else:
                mean_mse = mean_mae = float("nan")

            elapsed = time.time() - t0

            all_results[model_type] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "gt_list": gt_list,
                "pred_list": pred_list,
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "time": elapsed,
                "n_params": n_params,
            }

            print(f"  => MSE={mean_mse:.6f}  MAE={mean_mae:.6f}  Time={elapsed:.1f}s")

            # Individual model plots
            if gt_list and pred_list:
                save_loss_plot(
                    train_losses, val_losses, model_dir / "loss.png",
                    title=f"{model_type.upper()} Training Progress",
                )
                save_forecast_plot(
                    gt_list[0], pred_list[0], output_cols,
                    model_dir / "forecast.png",
                    title=f"{model_type.upper()} Rollout (horizon={eval_horizon})",
                    show_metrics=True,
                )

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR: {exc}")
            all_results[model_type] = {
                "train_losses": [], "val_losses": [],
                "gt_list": [], "pred_list": [],
                "mean_mse": float("nan"), "mean_mae": float("nan"),
                "time": elapsed, "n_params": 0,
            }

    # ── Comparative outputs ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  GENERATING COMPARATIVE RESULTS")
    print(f"{'=' * 70}")

    if len(all_results) >= 2:
        save_comparison_forecast_plot(all_results, output_cols, figs_dir / "comparison_forecast.png", eval_horizon)
        save_comparison_loss_plot(all_results, figs_dir / "comparison_losses.png")
        save_metrics_bar_chart(all_results, figs_dir / "comparison_metrics.png")
        print("  Saved comparison plots.")

    # Results CSV
    rows = []
    for mn, res in all_results.items():
        rows.append({
            "model": mn,
            "parameters": res["n_params"],
            "final_train_loss": res["train_losses"][-1] if res["train_losses"] else None,
            "final_val_loss": res["val_losses"][-1] if res["val_losses"] else None,
            "rollout_mse": res["mean_mse"],
            "rollout_mae": res["mean_mae"],
            "training_time_s": round(res["time"], 2),
        })

    results_df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison_results.csv"
    results_df.to_csv(csv_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(results_df.to_string(index=False))
    print(f"\n  Output directory : {out_dir}")
    print(f"  Figures          : {figs_dir}")
    print(f"  Results CSV      : {csv_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
