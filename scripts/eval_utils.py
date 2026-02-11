"""Shared evaluation, simulation, and plotting utilities.

Used by both ``train.py`` and ``compare.py`` to avoid code duplication.
All plotting functions read style/dpi/colors from a ``plot_cfg`` dict
(typically ``config["plotting"]``) so behaviour is fully config-driven.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timesim.models.factory import NEURAL_MODELS  # canonical source


# ─────────────────────────────────────────────────────────────────────
# Default plotting config (used when no config dict is provided)
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_PLOT_CFG = {
    "style": "seaborn-v0_8-whitegrid",
    "dpi": 150,
    "font_size": 10,
    "legend_font_size": 9,
    "model_colors": [
        "#2E86AB", "#E94F37", "#2CA58D", "#F18F01",
        "#A23B72", "#84BC9C", "#BC5D2E", "#6F2DBD",
    ],
}


def _pcfg(plot_cfg: Optional[Dict] = None) -> Dict:
    """Return a plotting config dict, falling back to defaults."""
    if plot_cfg is None:
        return _DEFAULT_PLOT_CFG
    merged = {**_DEFAULT_PLOT_CFG, **plot_cfg}
    return merged


def _colors(plot_cfg: Optional[Dict] = None) -> List[str]:
    return _pcfg(plot_cfg).get("model_colors", _DEFAULT_PLOT_CFG["model_colors"])


def _apply_style(plot_cfg: Optional[Dict] = None):
    cfg = _pcfg(plot_cfg)
    plt.style.use(cfg["style"])


def _save(fig, out_path, plot_cfg: Optional[Dict] = None):
    cfg = _pcfg(plot_cfg)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Evaluation (rollout)
# ─────────────────────────────────────────────────────────────────────

def evaluate_neural_model(
    model, val_dataset, warmup_len, eval_horizon,
    control_dim, exo_dim, device, n_windows=4,
):
    """Evaluate a neural model on multiple validation windows via rollout."""
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
            horizon_inputs = horizon_data[:, val_dataset.in_idx]
            horizon_outputs = horizon_data[:, val_dataset.out_idx]

            # Only append output columns not already in input columns
            in_idx_set = set(val_dataset.in_idx)
            extra_out_idx = [i for i in val_dataset.out_idx if i not in in_idx_set]
            if extra_out_idx:
                extra_out = warmup_data[:, extra_out_idx]
                warmup_full = np.concatenate([warmup_inputs, extra_out], axis=-1)
            else:
                warmup_full = warmup_inputs
            warmup_tensor = torch.from_numpy(warmup_full).float().unsqueeze(0).to(device)

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
    """Evaluate XGBoost model via recursive rollout on validation data."""
    val_data = val_dataset.values
    out_idx = val_dataset.out_idx
    val_len = len(val_data)

    min_required = seq_len + eval_horizon
    if val_len < min_required:
        return [], []

    max_start = val_len - min_required
    step = max(1, max_start // n_windows)
    start_indices = list(range(0, max_start, step))[:n_windows]

    gt_list, pred_list = [], []
    for start_idx in start_indices:
        lookback = val_data[start_idx : start_idx + seq_len]
        horizon_data = val_data[start_idx + seq_len : start_idx + seq_len + eval_horizon]
        gt = horizon_data[:, out_idx]

        current_input = lookback.copy()[np.newaxis, :, :]
        preds = []
        for h in range(eval_horizon):
            X_flat = current_input.reshape(1, -1)
            step_pred = np.zeros((1, model.output_dim))
            for out_i, m in enumerate(model.models_):
                step_pred[:, out_i] = m.predict(X_flat)
            preds.append(step_pred[0])

            if h < eval_horizon - 1:
                real_idx = start_idx + seq_len + h
                if real_idx < val_len:
                    new_step = val_data[real_idx].copy()
                else:
                    new_step = current_input[0, -1, :].copy()
                for oi, idx in enumerate(out_idx):
                    new_step[idx] = step_pred[0, oi]
                current_input = np.concatenate(
                    [current_input[:, 1:, :], new_step[np.newaxis, np.newaxis, :]], axis=1
                )

        pred_list.append(np.array(preds))
        gt_list.append(gt)

    return gt_list, pred_list


# ─────────────────────────────────────────────────────────────────────
# Simulation (recursive step-by-step)
# ─────────────────────────────────────────────────────────────────────

def simulate_recursive_neural(
    model, val_dataset, seq_len, sim_horizon, device, start_idx=0,
):
    """Environment-style recursive simulation for a neural model."""
    model.eval()
    val_data = val_dataset.values
    in_idx = val_dataset.in_idx
    out_idx = val_dataset.out_idx
    output_dim = len(out_idx)

    max_horizon = len(val_data) - start_idx - seq_len
    sim_horizon = min(sim_horizon, max_horizon)
    if sim_horizon <= 0:
        return {"predictions": np.empty((0, output_dim)),
                "ground_truths": np.empty((0, output_dim)), "n_steps": 0}

    window = val_data[start_idx : start_idx + seq_len].copy()
    predictions = np.zeros((sim_horizon, output_dim), dtype=np.float32)
    ground_truths = np.zeros((sim_horizon, output_dim), dtype=np.float32)

    # Only append output columns not already in input columns
    in_idx_set = set(in_idx)
    extra_out_idx = [i for i in out_idx if i not in in_idx_set]

    with torch.no_grad():
        for t in range(sim_horizon):
            real_idx = start_idx + seq_len + t
            input_feats = window[:, in_idx]
            if extra_out_idx:
                extra_out = window[:, extra_out_idx]
                full_input = np.concatenate([input_feats, extra_out], axis=-1)
            else:
                full_input = input_feats

            x = torch.from_numpy(full_input).float().unsqueeze(0).to(device)
            pred = model.forward(x)
            pred_step = pred[0, 0, :].cpu().numpy()

            predictions[t] = pred_step
            ground_truths[t] = val_data[real_idx, out_idx]

            new_row = val_data[real_idx].copy()
            for oi, idx in enumerate(out_idx):
                new_row[idx] = pred_step[oi]
            window = np.vstack([window[1:], new_row[np.newaxis, :]])

    return {"predictions": predictions, "ground_truths": ground_truths,
            "n_steps": sim_horizon}


def simulate_recursive_xgboost(
    model, val_dataset, seq_len, sim_horizon, start_idx=0,
):
    """Environment-style recursive simulation for XGBoost."""
    val_data = val_dataset.values
    out_idx = val_dataset.out_idx
    output_dim = len(out_idx)

    max_horizon = len(val_data) - start_idx - seq_len
    sim_horizon = min(sim_horizon, max_horizon)
    if sim_horizon <= 0:
        return {"predictions": np.empty((0, output_dim)),
                "ground_truths": np.empty((0, output_dim)), "n_steps": 0}

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

    return {"predictions": predictions, "ground_truths": ground_truths,
            "n_steps": sim_horizon}


# ─────────────────────────────────────────────────────────────────────
# Per-model plotting helpers
# ─────────────────────────────────────────────────────────────────────

def save_per_model_simulation_plot(sim_result, output_cols, model_name, out_path,
                                   plot_cfg=None):
    """Save a per-model simulation trajectory + error plot."""
    _apply_style(plot_cfg)
    cfg = _pcfg(plot_cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt = sim_result["ground_truths"]
    pred = sim_result["predictions"]
    n_steps = sim_result["n_steps"]
    n_vars = len(output_cols)
    steps = np.arange(n_steps)
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 3.5 * n_vars), squeeze=False)
    for v in range(n_vars):
        ax = axes[v, 0]
        ax.plot(steps, gt[:, v], color="#1B1B1E", linewidth=2, label="Ground Truth")
        ax.plot(steps, pred[:, v], color="#2E86AB", linewidth=1.3, linestyle="--",
                alpha=0.85, label=model_name)
        ax.set_ylabel(output_cols[v])
        ax.legend(fontsize=cfg["legend_font_size"])
        ax.set_title(f"{output_cols[v]} – Trajectory", fontsize=10,
                     fontweight="medium", loc="left")
        ax_e = axes[v, 1]
        err = np.abs(gt[:, v] - pred[:, v])
        ax_e.fill_between(steps, 0, err, color="#E94F37", alpha=0.4)
        ax_e.plot(steps, err, color="#E94F37", linewidth=1)
        ax_e.set_ylabel("Abs Error")
        ax_e.set_title(f"{output_cols[v]} – Step Error", fontsize=10,
                       fontweight="medium", loc="left")
    for ax in axes[-1, :]:
        ax.set_xlabel("Simulation Step")
    fig.suptitle(f"{model_name.upper()} – Recursive Simulation ({n_steps} steps)",
                 fontweight="bold", fontsize=13, y=1.02)
    _save(fig, out_path, plot_cfg)


def save_per_model_simulation_csv(sim_result, output_cols, model_dir, round_name):
    """Save per-model simulation results to CSV."""
    n = sim_result["n_steps"]
    if n <= 0:
        return
    gt_s = sim_result["ground_truths"]
    pr_s = sim_result["predictions"]
    sim_data = {"step": np.arange(n)}
    for v, col in enumerate(output_cols):
        sim_data[f"gt_{col}"] = gt_s[:, v]
        sim_data[f"pred_{col}"] = pr_s[:, v]
        sim_data[f"abserr_{col}"] = np.abs(gt_s[:, v] - pr_s[:, v])
    sim_data["step_mse"] = np.mean((gt_s - pr_s) ** 2, axis=-1)
    pd.DataFrame(sim_data).to_csv(
        model_dir / f"{round_name}_simulation.csv",
        index=False, float_format="%.6f",
    )


# ─────────────────────────────────────────────────────────────────────
# Cross-model comparison plots
# ─────────────────────────────────────────────────────────────────────

def save_comparison_forecast_plot(all_results, output_cols, out_path, eval_horizon,
                                  plot_cfg=None):
    _apply_style(plot_cfg)
    colors = _colors(plot_cfg)
    cfg = _pcfg(plot_cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    n_vars = len(output_cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3.5 * n_vars),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for v in range(n_vars):
        ax = axes[v]
        gt = None
        for mn in model_names:
            if all_results[mn]["gt_list"]:
                gt = all_results[mn]["gt_list"][0]; break
        if gt is None:
            continue
        steps = np.arange(gt.shape[0])
        ax.plot(steps, gt[:, v], color="#1B1B1E", linewidth=2.5,
                label="Ground Truth", zorder=10)
        for i, mn in enumerate(model_names):
            if not all_results[mn]["pred_list"]:
                continue
            pred = all_results[mn]["pred_list"][0]
            mse = np.mean((gt[:, v] - pred[:, v]) ** 2)
            color = colors[i % len(colors)]
            ax.plot(steps, pred[:, v], color=color, linewidth=1.5,
                    linestyle="--", alpha=0.85, label=f"{mn} (MSE={mse:.4f})")
        ax.set_ylabel(output_cols[v], fontweight="medium")
        ax.legend(loc="upper right", fontsize=cfg["legend_font_size"], framealpha=0.9)
    axes[-1].set_xlabel("Time Step", fontweight="medium")
    fig.suptitle(f"Model Comparison – Recursive Rollout (horizon={eval_horizon})",
                 fontweight="bold", fontsize=13, y=1.02)
    _save(fig, out_path, plot_cfg)


def save_comparison_loss_plot(all_results, out_path, plot_cfg=None):
    _apply_style(plot_cfg)
    colors = _colors(plot_cfg)
    cfg = _pcfg(plot_cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, mn in enumerate(model_names):
        color = colors[i % len(colors)]
        tl = all_results[mn]["train_losses"]
        vl = all_results[mn]["val_losses"]
        epochs = np.arange(1, len(tl) + 1)
        ax1.plot(epochs, tl, color=color, linewidth=2, label=mn,
                 marker="o" if len(epochs) <= 20 else None, markersize=4)
        valid_vl = [v for v in vl if v is not None]
        if valid_vl:
            ve = [j + 1 for j, v in enumerate(vl) if v is not None]
            ax2.plot(ve, valid_vl, color=color, linewidth=2, label=mn,
                     marker="s" if len(ve) <= 20 else None, markersize=4)
    for ax, title in [(ax1, "Training Loss"), (ax2, "Validation Loss")]:
        ax.set_xlabel("Epoch", fontweight="medium")
        ax.set_ylabel("Loss", fontweight="medium")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=cfg["legend_font_size"])
    _save(fig, out_path, plot_cfg)


def save_metrics_bar_chart(all_results, out_path, plot_cfg=None):
    _apply_style(plot_cfg)
    colors = _colors(plot_cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    mse_vals = [all_results[mn]["mean_mse"] for mn in model_names]
    mae_vals = [all_results[mn]["mean_mae"] for mn in model_names]
    bar_colors = [colors[i % len(colors)] for i in range(len(model_names))]
    x = np.arange(len(model_names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.bar(x, mse_vals, 0.35, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(model_names, fontweight="medium")
    ax1.set_ylabel("MSE"); ax1.set_title("Rollout MSE by Model", fontweight="bold")
    for bar, val in zip(bars1, mse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    bars2 = ax2.bar(x, mae_vals, 0.35, color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(model_names, fontweight="medium")
    ax2.set_ylabel("MAE"); ax2.set_title("Rollout MAE by Model", fontweight="bold")
    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    _save(fig, out_path, plot_cfg)


def save_simulation_trajectory_plot(sim_results_all, output_cols, out_path,
                                    plot_cfg=None):
    _apply_style(plot_cfg)
    colors = _colors(plot_cfg)
    cfg = _pcfg(plot_cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = [mn for mn in sim_results_all if sim_results_all[mn]["n_steps"] > 0]
    if not model_names:
        return
    n_vars = len(output_cols)
    gt = sim_results_all[model_names[0]]["ground_truths"]
    total_steps = gt.shape[0]
    steps = np.arange(total_steps)
    n_panels = n_vars + 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True,
        squeeze=False, gridspec_kw={"height_ratios": [3]*n_vars + [2]})
    axes = axes.flatten()
    for v in range(n_vars):
        ax = axes[v]
        ax.plot(steps, gt[:total_steps, v], color="#1B1B1E", linewidth=2,
                label="Ground Truth", zorder=10)
        for i, mn in enumerate(model_names):
            pred = sim_results_all[mn]["predictions"][:total_steps]
            mse = float(np.mean((gt[:total_steps, v] - pred[:, v]) ** 2))
            color = colors[i % len(colors)]
            ax.plot(steps, pred[:, v], color=color, linewidth=1.3,
                    linestyle="--", alpha=0.85, label=f"{mn} (MSE={mse:.4f})")
        ax.set_ylabel(output_cols[v], fontweight="medium", fontsize=11)
        ax.legend(loc="upper right", fontsize=cfg["legend_font_size"], framealpha=0.9)
        ax.set_title(f"Output: {output_cols[v]}", fontsize=10,
                     fontweight="medium", loc="left")
    ax_err = axes[n_vars]
    for i, mn in enumerate(model_names):
        pred = sim_results_all[mn]["predictions"][:total_steps]
        per_step_mse = np.mean((gt[:total_steps] - pred) ** 2, axis=-1)
        cum_mse = np.cumsum(per_step_mse) / (steps + 1)
        color = colors[i % len(colors)]
        ax_err.plot(steps, cum_mse, color=color, linewidth=1.5, label=mn)
    ax_err.set_ylabel("Cumulative MSE"); ax_err.set_xlabel("Simulation Step")
    ax_err.set_title("Error Accumulation", fontsize=10, fontweight="medium", loc="left")
    ax_err.legend(fontsize=cfg["legend_font_size"], framealpha=0.9)
    fig.suptitle(f"Recursive Simulation ({total_steps} steps)",
                 fontweight="bold", fontsize=12, y=1.02)
    _save(fig, out_path, plot_cfg)


def save_simulation_csv(sim_results_all, output_cols, out_path):
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
    pd.DataFrame(data).to_csv(out_path, index=False, float_format="%.6f")
