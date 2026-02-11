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

def train_neural_model(model, train_dataset, val_dataset, config, device, model_dir,
                       lr_override=None, epochs_override=None):
    """Train a neural WorldModel using WorldModelTrainer. Returns (train_losses, val_losses).

    Parameters
    ----------
    lr_override : float, optional
        Override learning rate (used for retraining rounds).
    epochs_override : int, optional
        Override number of epochs.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    epochs = epochs_override or config["training"]["epochs"]
    batch_size = config["dataset"]["batch_size"]
    lr = lr_override or config["training"].get("learning_rate", 1e-3)
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

    # NOTE: checkpoint saving is handled by the caller (round-specific names)
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

    # NOTE: model saving is handled by the caller (round-specific names)
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
# Simulation (recursive step-by-step rollout for DRL environment)
# ─────────────────────────────────────────────────────────────────────

def simulate_recursive_neural(
    model, val_dataset, seq_len, sim_horizon,
    device, start_idx=0,
):
    """Run recursive (environment-style) simulation for a neural model.

    Simulates a DRL environment interaction loop:
    - At each step, the full lookback window (seq_len) is fed to model.forward()
    - ONLY the first row of the prediction is used (pred[:, 0, :])
      even if pred_len > 1 — only the immediate next-step matters
    - For output variables: predicted values are fed back into the window
    - For non-output variables: real values from the dataset are used
      (these represent actions/controls from the DRL agent + exogenous)

    This ensures the model behaves as a single-step environment:
    each timestamp receives new actions and responds accordingly.

    Parameters
    ----------
    model : WorldModelBase
        Trained neural model with forward(x) -> (B, pred_len, output_dim)
    val_dataset : GroupedTimeSeriesDataset
        Validation dataset (scaled values, in_idx, out_idx)
    seq_len : int
        Lookback window size
    sim_horizon : int
        Number of simulation steps to run
    device : str or torch.device
    start_idx : int
        Starting index in the validation data

    Returns
    -------
    dict with keys:
        predictions   : (n_steps, output_dim) ndarray
        ground_truths : (n_steps, output_dim) ndarray
        n_steps       : int
    """
    model.eval()
    val_data = val_dataset.values
    in_idx = val_dataset.in_idx
    out_idx = val_dataset.out_idx
    output_dim = len(out_idx)

    max_horizon = len(val_data) - start_idx - seq_len
    sim_horizon = min(sim_horizon, max_horizon)
    if sim_horizon <= 0:
        return {
            "predictions": np.empty((0, output_dim)),
            "ground_truths": np.empty((0, output_dim)),
            "n_steps": 0,
        }

    # Sliding window over all features (initialised with real data)
    window = val_data[start_idx : start_idx + seq_len].copy()

    predictions = np.zeros((sim_horizon, output_dim), dtype=np.float32)
    ground_truths = np.zeros((sim_horizon, output_dim), dtype=np.float32)

    with torch.no_grad():
        for t in range(sim_horizon):
            real_idx = start_idx + seq_len + t

            # Build model input: [input_cols, output_cols]
            input_feats = window[:, in_idx]   # (seq_len, len(in_idx))
            output_feats = window[:, out_idx]  # (seq_len, output_dim)
            full_input = np.concatenate([input_feats, output_feats], axis=-1)

            x = torch.from_numpy(full_input).float().unsqueeze(0).to(device)
            pred = model.forward(x)  # (1, pred_len, output_dim)

            # ── CRITICAL: take ONLY the first prediction row ──
            pred_step = pred[0, 0, :].cpu().numpy()  # (output_dim,)

            predictions[t] = pred_step
            ground_truths[t] = val_data[real_idx, out_idx]

            # Next row: real controls/exo + predicted outputs
            new_row = val_data[real_idx].copy()
            for oi, idx in enumerate(out_idx):
                new_row[idx] = pred_step[oi]

            # Slide window forward by 1
            window = np.vstack([window[1:], new_row[np.newaxis, :]])

    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "n_steps": sim_horizon,
    }


def simulate_recursive_xgboost(
    model, val_dataset, seq_len, sim_horizon, start_idx=0,
):
    """Run recursive (environment-style) simulation for XGBoost.

    Same step-by-step logic as neural simulation.
    At each step a single prediction is produced and fed back.
    """
    val_data = val_dataset.values
    out_idx = val_dataset.out_idx
    output_dim = len(out_idx)

    max_horizon = len(val_data) - start_idx - seq_len
    sim_horizon = min(sim_horizon, max_horizon)
    if sim_horizon <= 0:
        return {
            "predictions": np.empty((0, output_dim)),
            "ground_truths": np.empty((0, output_dim)),
            "n_steps": 0,
        }

    window = val_data[start_idx : start_idx + seq_len].copy()

    predictions = np.zeros((sim_horizon, output_dim), dtype=np.float32)
    ground_truths = np.zeros((sim_horizon, output_dim), dtype=np.float32)

    for t in range(sim_horizon):
        real_idx = start_idx + seq_len + t

        X_flat = window.reshape(1, -1)
        pred_step = np.zeros(output_dim, dtype=np.float32)
        for oi, m in enumerate(model.models_):
            pred_step[oi] = m.predict(X_flat)[0]

        predictions[t] = pred_step
        ground_truths[t] = val_data[real_idx, out_idx]

        new_row = val_data[real_idx].copy()
        for oi, idx in enumerate(out_idx):
            new_row[idx] = pred_step[oi]

        window = np.vstack([window[1:], new_row[np.newaxis, :]])

    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "n_steps": sim_horizon,
    }


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


def save_simulation_trajectory_plot(sim_results_all, output_cols, out_path, n_steps=None):
    """Plot step-by-step simulation trajectories for all models vs ground truth.

    The bottom panel shows cumulative MSE to visualise how errors accumulate
    during recursive rollout — the key quality metric for a DRL environment.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [mn for mn in sim_results_all if sim_results_all[mn]["n_steps"] > 0]
    if not model_names:
        return

    n_vars = len(output_cols)
    gt = sim_results_all[model_names[0]]["ground_truths"]
    total_steps = gt.shape[0]
    if n_steps:
        total_steps = min(total_steps, n_steps)
    steps = np.arange(total_steps)

    n_panels = n_vars + 1  # one per output + error accumulation
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3] * n_vars + [2]},
    )
    axes = axes.flatten()

    # ── Per-output-variable panels ─────────────────────────────────
    for v in range(n_vars):
        ax = axes[v]
        ax.plot(steps, gt[:total_steps, v], color="#1B1B1E", linewidth=2,
                label="Ground Truth", zorder=10)

        for i, mn in enumerate(model_names):
            pred = sim_results_all[mn]["predictions"][:total_steps]
            mse = float(np.mean((gt[:total_steps, v] - pred[:, v]) ** 2))
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(steps, pred[:, v], color=color, linewidth=1.3,
                    linestyle="--", alpha=0.85, label=f"{mn} (MSE={mse:.4f})")

        ax.set_ylabel(output_cols[v], fontweight="medium", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(f"Output: {output_cols[v]}", fontsize=10,
                     fontweight="medium", loc="left")

    # ── Error-accumulation panel ───────────────────────────────────
    ax_err = axes[n_vars]
    for i, mn in enumerate(model_names):
        pred = sim_results_all[mn]["predictions"][:total_steps]
        per_step_mse = np.mean((gt[:total_steps] - pred) ** 2, axis=-1)
        cum_mse = np.cumsum(per_step_mse) / (steps + 1)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax_err.plot(steps, cum_mse, color=color, linewidth=1.5, label=mn)

    ax_err.set_ylabel("Cumulative MSE", fontweight="medium")
    ax_err.set_xlabel("Simulation Step", fontweight="medium")
    ax_err.set_title("Error Accumulation Over Simulation Steps",
                     fontsize=10, fontweight="medium", loc="left")
    ax_err.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Recursive Simulation – Environment Rollout ({total_steps} steps)\n"
        f"At each step: predict → take first row only → feed back → next step",
        fontweight="bold", fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def save_simulation_csv(sim_results_all, output_cols, out_path):
    """Save step-by-step simulation results to CSV.

    Columns: step, gt_{var}, {model}_{var}, {model}_err_{var}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [mn for mn in sim_results_all if sim_results_all[mn]["n_steps"] > 0]
    if not model_names:
        return

    gt = sim_results_all[model_names[0]]["ground_truths"]
    n_steps = gt.shape[0]

    data = {"step": np.arange(n_steps)}
    for v, col in enumerate(output_cols):
        data[f"gt_{col}"] = gt[:, v]

    for mn in model_names:
        pred = sim_results_all[mn]["predictions"]
        for v, col in enumerate(output_cols):
            data[f"{mn}_{col}"] = pred[:, v]
            data[f"{mn}_abserr_{col}"] = np.abs(pred[:, v] - gt[:, v])

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False, float_format="%.6f")


def save_per_model_simulation_plot(sim_result, output_cols, model_name, out_path):
    """Save a detailed simulation plot for a single model.

    Shows: prediction vs ground truth, per-step absolute error.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt = sim_result["ground_truths"]
    pred = sim_result["predictions"]
    n_steps = sim_result["n_steps"]
    n_vars = len(output_cols)
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 3.5 * n_vars), squeeze=False)

    for v in range(n_vars):
        # Left: trajectory
        ax = axes[v, 0]
        ax.plot(steps, gt[:, v], color="#1B1B1E", linewidth=2, label="Ground Truth")
        ax.plot(steps, pred[:, v], color="#2E86AB", linewidth=1.3,
                linestyle="--", alpha=0.85, label=f"{model_name}")
        ax.set_ylabel(output_cols[v], fontweight="medium")
        ax.legend(fontsize=8)
        ax.set_title(f"{output_cols[v]} – Trajectory", fontsize=10,
                     fontweight="medium", loc="left")

        # Right: absolute error
        ax_e = axes[v, 1]
        err = np.abs(gt[:, v] - pred[:, v])
        ax_e.fill_between(steps, 0, err, color="#E94F37", alpha=0.4)
        ax_e.plot(steps, err, color="#E94F37", linewidth=1)
        ax_e.set_ylabel("Abs Error", fontweight="medium")
        ax_e.set_title(f"{output_cols[v]} – Step Error", fontsize=10,
                       fontweight="medium", loc="left")

    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation Step", fontweight="medium")

    fig.suptitle(
        f"{model_name.upper()} – Recursive Simulation ({n_steps} steps)",
        fontweight="bold", fontsize=13, y=1.02,
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

    # ── Determine models ─────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}

    if args.models:
        selected_types = args.models
    elif models_cfg_list:
        selected_types = [m["type"] for m in models_cfg_list]
    else:
        selected_types = [config.get("model", {}).get("type", "lstm")]

    model_names = selected_types

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

    # ── Training rounds ──────────────────────────────────────────────
    # Supports: train-only, retrain-only (with checkpoint_dir), or both.
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

    # ── Print banner ─────────────────────────────────────────────────
    round_desc = " → ".join(rc.get("name", "?") for rc in training_rounds)
    print("\n" + "=" * 70)
    print("  MULTI-MODEL TIME SERIES COMPARISON")
    print("=" * 70)
    print(f"  Rounds       : {round_desc}")
    print(f"  Models       : {model_names}")
    print(f"  Device       : {device}")
    print(f"  Input cols   : {input_cols}")
    print(f"  Output cols  : {output_cols}")
    print(f"  input_dim={input_dim}  output_dim={output_dim}")
    print(f"  control_dim={control_dim}  exo_dim={exo_dim}")
    print(f"  seq_len={seq_len}  pred_len={pred_len}  eval_horizon={eval_horizon}")
    print("=" * 70 + "\n")

    # ── Create output directory ──────────────────────────────────────
    #
    # Layout:
    #   runs/<name>/compare_<ts>/
    #     config.yaml
    #     comparison_results.csv
    #     simulation_results.csv, simulation_metrics.csv
    #     figures/                          ← cross-model comparison plots
    #     <model_type>/                    ← one folder per model
    #       model_config.yaml
    #       train_checkpoint.pth           ← from "train" round
    #       retrain_checkpoint.pth         ← from "retrain" round (if any)
    #       train_loss.png, retrain_loss.png
    #       loss_full.png                  ← cumulative across all rounds
    #       metrics.csv                    ← epoch-level metrics from trainer
    #       train_forecast.png             ← eval after train round
    #       retrain_forecast.png           ← eval after retrain round (if any)
    #       train_simulation.png/.csv      ← sim after train round
    #       retrain_simulation.png/.csv    ← sim after retrain round (if any)
    #
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / config["dataset"]["name"] / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # Pre-create per-model directories
    all_model_types_in_rounds = []
    for rc in training_rounds:
        all_model_types_in_rounds.extend(rc.get("models", model_names))
    for mt in set(all_model_types_in_rounds):
        (out_dir / mt).mkdir(exist_ok=True)

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

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING ROUNDS
    # ══════════════════════════════════════════════════════════════════
    trained_models = {}        # model objects (latest version)
    model_n_params = {}
    cumul_train_losses = {}
    cumul_val_losses = {}
    cumul_time = {}
    model_last_round = {}      # model_type → round name

    for round_cfg in training_rounds:
        round_name = round_cfg.get("name", "train")
        round_epochs = round_cfg.get("epochs", config["training"]["epochs"])
        round_lr = round_cfg.get("learning_rate",
                                 config["training"].get("learning_rate", 1e-3))
        round_model_types = round_cfg.get("models", model_names)
        checkpoint_dir = round_cfg.get("checkpoint_dir", None)

        print(f"\n{'═' * 70}")
        print(f"  ROUND: {round_name.upper()} "
              f"({round_epochs} epochs, LR={round_lr})")
        print(f"  Models: {round_model_types}")
        if checkpoint_dir:
            print(f"  Loading checkpoints from: {checkpoint_dir}")
        print(f"{'═' * 70}")

        for model_type in round_model_types:
            mc = models_cfg_map.get(model_type, {"type": model_type})
            model_params = {k: v for k, v in mc.items() if k != "type"}
            model_dir = out_dir / model_type      # flat – always same dir

            # ── Decide: new model, resume in-memory, or load checkpoint ──
            is_retrain = False

            if model_type in trained_models and model_type in NEURAL_MODELS:
                # Continue from in-memory weights (same run, previous round)
                model = trained_models[model_type]
                is_retrain = True
            elif checkpoint_dir and model_type in NEURAL_MODELS:
                # Load checkpoint from a previous run
                model = build_model(model_type, model_params,
                                    input_dim, output_dim, seq_len, pred_len)
                ckpt = Path(checkpoint_dir) / model_type / "train_checkpoint.pth"
                if ckpt.exists():
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                    model.to(device)
                    is_retrain = True
                    print(f"  Loaded checkpoint: {ckpt}")
                else:
                    print(f"  Warning: {ckpt} not found → training from scratch")
            elif checkpoint_dir and model_type == "xgboost" and HAS_XGBOOST:
                pkl = Path(checkpoint_dir) / model_type / "train_model.pkl"
                if pkl.exists():
                    model = XGBoostForecaster.load(str(pkl))
                    is_retrain = True
                    print(f"  Loaded model: {pkl}")
                else:
                    model = build_model(model_type, model_params,
                                        input_dim, output_dim, seq_len, pred_len)
            else:
                model = build_model(model_type, model_params,
                                    input_dim, output_dim, seq_len, pred_len)

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
                    train_losses, val_losses = train_neural_model(
                        model, train_dataset, val_dataset, config, device,
                        model_dir,
                        lr_override=round_lr,
                        epochs_override=round_epochs,
                    )
                    # Save round-specific checkpoint
                    torch.save(model.state_dict(),
                               model_dir / f"{round_name}_checkpoint.pth")
                else:
                    train_losses, val_losses = train_xgboost_model(
                        model, train_dataset, val_dataset, config, model_dir,
                    )
                    model.save(str(model_dir / f"{round_name}_model.pkl"))

                elapsed = time.time() - t0

                # Round-specific loss plot
                save_loss_plot(
                    train_losses, val_losses,
                    model_dir / f"{round_name}_loss.png",
                    title=f"{model_type.upper()} – {round_name} "
                          f"({round_epochs} epochs, LR={round_lr})",
                )

                cumul_train_losses.setdefault(model_type, []).extend(train_losses)
                cumul_val_losses.setdefault(model_type, []).extend(val_losses)
                cumul_time[model_type] = cumul_time.get(model_type, 0.0) + elapsed
                trained_models[model_type] = model
                model_last_round[model_type] = round_name

                # Save / update model config
                total_epochs = len(cumul_train_losses[model_type])
                model_cfg_data = {
                    "type": model_type, **model_params,
                    "last_round": round_name,
                    "total_epochs": total_epochs,
                }
                with open(model_dir / "model_config.yaml", "w") as f:
                    yaml.safe_dump(model_cfg_data, f)

                tl = train_losses[-1] if train_losses else float("nan")
                vl = val_losses[-1] if val_losses else float("nan")
                print(f"  => train_loss={tl:.6f}  val_loss={vl:.6f}  "
                      f"time={elapsed:.1f}s")

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATE & SIMULATE EVERY CHECKPOINT
    # ══════════════════════════════════════════════════════════════════
    # For each model, for each training round it participated in, load
    # that round's checkpoint and produce:
    #   <round>_forecast.png   – rollout evaluation
    #   <round>_simulation.png – recursive step-by-step simulation
    #   <round>_simulation.csv – simulation data
    # The comparison plots (figures/) use the *latest* checkpoint only.

    sim_cfg = config.get("simulation", {})
    sim_horizon = sim_cfg.get("horizon", None)
    if sim_horizon is None:
        sim_horizon = len(val_dataset.values) - seq_len
    sim_start = sim_cfg.get("start_idx", 0)

    all_results = {}           # latest checkpoint eval  → for comparison plots
    sim_results_latest = {}    # latest checkpoint sim   → for comparison plots

    print(f"\n{'═' * 70}")
    print("  EVALUATING & SIMULATING PER CHECKPOINT")
    print(f"{'═' * 70}")
    print(f"  Eval horizon : {eval_horizon}  |  Sim horizon : {sim_horizon}")
    print(f"{'═' * 70}")

    for model_type in trained_models:
        mc = models_cfg_map.get(model_type, {"type": model_type})
        model_params = {k: v for k, v in mc.items() if k != "type"}
        model_dir = out_dir / model_type
        n_params = model_n_params.get(model_type, 0)

        # Which rounds trained this model?
        rounds_for_model = [
            rc for rc in training_rounds
            if model_type in rc.get("models", model_names)
        ]

        for round_cfg in rounds_for_model:
            round_name = round_cfg.get("name", "train")

            # ── Load checkpoint for this round ───────────────────────
            if model_type in NEURAL_MODELS:
                ckpt_path = model_dir / f"{round_name}_checkpoint.pth"
                if not ckpt_path.exists():
                    print(f"\n  {model_type}/{round_name}: checkpoint not found, skipping")
                    continue
                model = build_model(model_type, model_params,
                                    input_dim, output_dim, seq_len, pred_len)
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True))
                model.to(device)
                model.eval()
            else:
                pkl_path = model_dir / f"{round_name}_model.pkl"
                if not pkl_path.exists():
                    print(f"\n  {model_type}/{round_name}: model file not found, skipping")
                    continue
                model = XGBoostForecaster.load(str(pkl_path))

            print(f"\n{'─' * 70}")
            print(f"  {model_type.upper()} [{round_name}]")
            print(f"{'─' * 70}")

            # ── Evaluate (multi-window rollout) ──────────────────────
            try:
                if model_type in NEURAL_MODELS:
                    gt_list, pred_list = evaluate_neural_model(
                        model, val_dataset, warmup_len, eval_horizon,
                        control_dim, exo_dim, device, n_windows,
                    )
                else:
                    gt_list, pred_list = evaluate_xgboost_model(
                        model, val_dataset, seq_len, eval_horizon, n_windows,
                    )

                if gt_list and pred_list:
                    mean_mse = float(np.mean(
                        [np.mean((g - p) ** 2) for g, p in zip(gt_list, pred_list)]))
                    mean_mae = float(np.mean(
                        [np.mean(np.abs(g - p)) for g, p in zip(gt_list, pred_list)]))
                else:
                    mean_mse = mean_mae = float("nan")

                print(f"    Eval  MSE={mean_mse:.6f}  MAE={mean_mae:.6f}")

                if gt_list and pred_list:
                    save_forecast_plot(
                        gt_list[0], pred_list[0], output_cols,
                        model_dir / f"{round_name}_forecast.png",
                        title=f"{model_type.upper()} [{round_name}] "
                              f"(horizon={eval_horizon})",
                        show_metrics=True,
                    )
            except Exception as exc:
                print(f"    Eval ERROR: {exc}")
                gt_list, pred_list = [], []
                mean_mse = mean_mae = float("nan")

            # ── Recursive simulation ─────────────────────────────────
            try:
                if model_type in NEURAL_MODELS:
                    sim_result = simulate_recursive_neural(
                        model, val_dataset, seq_len, sim_horizon,
                        device, start_idx=sim_start,
                    )
                else:
                    sim_result = simulate_recursive_xgboost(
                        model, val_dataset, seq_len, sim_horizon,
                        start_idx=sim_start,
                    )

                n = sim_result["n_steps"]
                if n > 0:
                    gt = sim_result["ground_truths"]
                    pr = sim_result["predictions"]
                    sim_mse = float(np.mean((gt - pr) ** 2))
                    sim_mae = float(np.mean(np.abs(gt - pr)))
                    print(f"    Sim   MSE={sim_mse:.6f}  MAE={sim_mae:.6f}  "
                          f"({n} steps)")

                    save_per_model_simulation_plot(
                        sim_result, output_cols, model_type,
                        model_dir / f"{round_name}_simulation.png",
                    )
                    sim_data = {"step": np.arange(n)}
                    for v, col in enumerate(output_cols):
                        sim_data[f"gt_{col}"] = gt[:, v]
                        sim_data[f"pred_{col}"] = pr[:, v]
                        sim_data[f"abserr_{col}"] = np.abs(gt[:, v] - pr[:, v])
                    sim_data["step_mse"] = np.mean((gt - pr) ** 2, axis=-1)
                    pd.DataFrame(sim_data).to_csv(
                        model_dir / f"{round_name}_simulation.csv",
                        index=False, float_format="%.6f",
                    )
                else:
                    print(f"    Sim   no steps (insufficient data)")
                    sim_result = None
            except Exception as exc:
                print(f"    Sim ERROR: {exc}")
                sim_result = None

            # ── Store latest round results for comparison ────────────
            # This dict is overwritten each round, so the last round
            # processed becomes the "latest" used in comparison plots.
            total_epochs = len(cumul_train_losses.get(model_type, []))
            all_results[model_type] = {
                "train_losses": cumul_train_losses.get(model_type, []),
                "val_losses": cumul_val_losses.get(model_type, []),
                "gt_list": gt_list,
                "pred_list": pred_list,
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "time": cumul_time.get(model_type, 0.0),
                "n_params": n_params,
                "last_round_name": model_last_round.get(model_type, "train"),
                "total_epochs": total_epochs,
            }
            if sim_result and sim_result["n_steps"] > 0:
                sim_results_latest[model_type] = sim_result

        # ── Cumulative loss plot (across all rounds) ─────────────────
        if model_type in cumul_train_losses:
            total_epochs = len(cumul_train_losses[model_type])
            save_loss_plot(
                cumul_train_losses[model_type],
                cumul_val_losses.get(model_type, []),
                model_dir / "loss_full.png",
                title=f"{model_type.upper()} – Full Training History "
                      f"({total_epochs} epochs)",
            )

    # ══════════════════════════════════════════════════════════════════
    #  COMPARATIVE OUTPUTS  →  figures/   (latest checkpoint per model)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  GENERATING COMPARATIVE RESULTS")
    print(f"{'=' * 70}")

    if len(all_results) >= 2:
        save_comparison_forecast_plot(
            all_results, output_cols,
            figs_dir / "comparison_forecast.png", eval_horizon)
        save_comparison_loss_plot(all_results, figs_dir / "comparison_losses.png")
        save_metrics_bar_chart(all_results, figs_dir / "comparison_metrics.png")
        print("  Saved comparison plots → figures/")

    rows = []
    for mn, res in all_results.items():
        rows.append({
            "model": mn,
            "last_round": res.get("last_round_name", "train"),
            "total_epochs": res.get("total_epochs", 0),
            "parameters": res["n_params"],
            "final_train_loss": (res["train_losses"][-1]
                                 if res["train_losses"] else None),
            "final_val_loss": (res["val_losses"][-1]
                               if res["val_losses"] else None),
            "rollout_mse": res["mean_mse"],
            "rollout_mae": res["mean_mae"],
            "training_time_s": round(res["time"], 2),
        })
    results_df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(results_df.to_string(index=False))
    print(f"\n  Output directory : {out_dir}")
    print(f"  Figures          : {figs_dir}")
    print(f"  Results CSV      : {csv_path}")
    print(f"{'=' * 70}")

    # ── Comparison simulation outputs (latest checkpoint) ─────────
    if sim_results_latest:
        save_simulation_trajectory_plot(
            sim_results_latest, output_cols,
            figs_dir / "simulation_trajectory.png",
        )
        save_simulation_csv(
            sim_results_latest, output_cols,
            out_dir / "simulation_results.csv",
        )

        sim_rows = []
        for mn, sr in sim_results_latest.items():
            if sr["n_steps"] > 0:
                gt_s = sr["ground_truths"]
                pr_s = sr["predictions"]
                sim_rows.append({
                    "model": mn,
                    "last_round": model_last_round.get(mn, "train"),
                    "sim_steps": sr["n_steps"],
                    "sim_mse": float(np.mean((gt_s - pr_s) ** 2)),
                    "sim_mae": float(np.mean(np.abs(gt_s - pr_s))),
                })
        if sim_rows:
            sim_df = pd.DataFrame(sim_rows)
            sim_csv_path = out_dir / "simulation_metrics.csv"
            sim_df.to_csv(sim_csv_path, index=False)

            print(f"\n{'=' * 70}")
            print("  SIMULATION SUMMARY")
            print(f"{'=' * 70}")
            print(sim_df.to_string(index=False))
            print(f"\n  Simulation plot      : {figs_dir / 'simulation_trajectory.png'}")
            print(f"  Simulation data CSV  : {out_dir / 'simulation_results.csv'}")
            print(f"  Simulation metrics   : {sim_csv_path}")
            print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
