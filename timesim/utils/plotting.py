"""Visualization utilities for time series forecasting.

This module provides professional-quality plotting functions for:
- Training loss curves with statistics
- Recursive forecast comparisons
- Multi-horizon prediction analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import ceil

# Use a clean, professional style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette (colorblind-friendly)
COLORS = {
    'train': '#2E86AB',      # Blue
    'val': '#E94F37',        # Red/Orange
    'pred': '#2E86AB',       # Blue
    'gt': '#1B1B1E',         # Near black
    'before': '#A23B72',     # Purple
    'after': '#F18F01',      # Orange
    'fill': '#E8E8E8',       # Light gray
    'grid': '#CCCCCC',       # Grid gray
}


def _setup_figure_style():
    """Apply consistent styling to figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def save_loss_plot(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    out_path: str | Path,
    title: str = "Training Progress",
    log_scale: bool = False,
    show_stats: bool = True,
):
    """Save a publication-quality loss curve figure.
    
    Parameters
    ----------
    train_losses : list of float
        Training loss per epoch.
    val_losses : list of float or None
        Validation loss per epoch.
    out_path : str or Path
        Output path for the figure.
    title : str, default "Training Progress"
        Figure title.
    log_scale : bool, default False
        Use logarithmic scale for y-axis.
    show_stats : bool, default True
        Show statistics annotation (min loss, final loss).
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Epochs are 1-indexed for display
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot training loss
    ax.plot(
        epochs, train_losses,
        color=COLORS['train'],
        linewidth=2,
        label='Training Loss',
        marker='o' if len(epochs) <= 30 else None,
        markersize=4,
        alpha=0.9,
    )
    
    # Plot validation loss if available
    has_val = val_losses is not None and any(v is not None for v in val_losses)
    if has_val:
        # Filter out None values
        val_epochs = [i+1 for i, v in enumerate(val_losses) if v is not None]
        val_values = [v for v in val_losses if v is not None]
        
        ax.plot(
            val_epochs, val_values,
            color=COLORS['val'],
            linewidth=2,
            label='Validation Loss',
            marker='s' if len(val_epochs) <= 30 else None,
            markersize=4,
            alpha=0.9,
        )
    
    # Log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Epoch', fontweight='medium')
    ax.set_ylabel('Loss', fontweight='medium')
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    
    # Integer epochs on x-axis
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    
    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
    )
    
    # Statistics annotation
    if show_stats:
        stats_text = []
        
        # Training stats
        min_train = min(train_losses)
        min_train_epoch = train_losses.index(min_train) + 1
        final_train = train_losses[-1]
        stats_text.append(f"Train: min={min_train:.2e} (ep.{min_train_epoch}), final={final_train:.2e}")
        
        # Validation stats
        if has_val:
            valid_vals = [v for v in val_losses if v is not None]
            if valid_vals:
                min_val = min(valid_vals)
                min_val_idx = [v for v in val_losses if v is not None].index(min_val)
                min_val_epoch = [i+1 for i, v in enumerate(val_losses) if v is not None][min_val_idx]
                final_val = valid_vals[-1]
                stats_text.append(f"Val: min={min_val:.2e} (ep.{min_val_epoch}), final={final_val:.2e}")
        
        # Add text box
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#CCCCCC')
        ax.text(
            0.02, 0.02, textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            fontfamily='monospace',
            bbox=props,
        )
    
    # Mark best validation point if available
    if has_val:
        valid_vals = [v for v in val_losses if v is not None]
        if valid_vals:
            min_val = min(valid_vals)
            min_val_idx = valid_vals.index(min_val)
            min_val_epoch = [i+1 for i, v in enumerate(val_losses) if v is not None][min_val_idx]
            ax.axvline(
                x=min_val_epoch, color=COLORS['val'],
                linestyle='--', alpha=0.5, linewidth=1
            )
            ax.scatter(
                [min_val_epoch], [min_val],
                color=COLORS['val'], s=80, zorder=5,
                marker='*', edgecolors='white', linewidths=1
            )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def save_forecast_plot(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    column_names: List[str],
    out_path: str | Path,
    title: str = "Recursive Forecast",
    warmup_len: int = 0,
    time_axis: Optional[np.ndarray] = None,
    show_metrics: bool = True,
):
    """Save a forecast comparison plot (predictions vs ground truth).
    
    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth values, shape (horizon, n_features).
    predictions : np.ndarray
        Predicted values, shape (horizon, n_features).
    column_names : list of str
        Names of the output features.
    out_path : str or Path
        Output path for the figure.
    title : str, default "Recursive Forecast"
        Figure title.
    warmup_len : int, default 0
        If > 0, first warmup_len points are shown as context (shaded).
    time_axis : np.ndarray, optional
        Custom time axis values. If None, uses step indices.
    show_metrics : bool, default True
        Show error metrics (MSE, MAE) in the plot.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_steps, n_vars = ground_truth.shape
    
    # Time axis
    if time_axis is None:
        time_axis = np.arange(n_steps)
    
    # Create subplots
    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(10, 3 * n_vars),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()
    
    for i, (ax, col_name) in enumerate(zip(axes, column_names)):
        gt = ground_truth[:, i]
        pred = predictions[:, i]
        
        # Plot ground truth
        ax.plot(
            time_axis, gt,
            color=COLORS['gt'],
            linewidth=2,
            label='Ground Truth',
            zorder=3,
        )
        
        # Plot predictions
        ax.plot(
            time_axis, pred,
            color=COLORS['pred'],
            linewidth=2,
            label='Prediction',
            linestyle='--',
            alpha=0.9,
            zorder=2,
        )
        
        # Fill between for error visualization
        ax.fill_between(
            time_axis, gt, pred,
            color=COLORS['pred'],
            alpha=0.15,
            zorder=1,
        )
        
        # Warmup region shading
        if warmup_len > 0 and warmup_len < n_steps:
            ax.axvspan(
                time_axis[0], time_axis[warmup_len-1],
                color='#F0F0F0', alpha=0.5, zorder=0,
                label='Warmup',
            )
            ax.axvline(
                time_axis[warmup_len-1], color='#888888',
                linestyle=':', linewidth=1, alpha=0.7,
            )
        
        # Metrics
        if show_metrics:
            # Calculate metrics on forecast region (after warmup)
            start_idx = warmup_len if warmup_len > 0 else 0
            gt_eval = gt[start_idx:]
            pred_eval = pred[start_idx:]
            
            mse = np.mean((gt_eval - pred_eval) ** 2)
            mae = np.mean(np.abs(gt_eval - pred_eval))
            
            metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}"
            props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCCCCC')
            ax.text(
                0.98, 0.95, metrics_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                fontfamily='monospace',
                bbox=props,
            )
        
        ax.set_ylabel(col_name, fontweight='medium')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    axes[-1].set_xlabel('Time Step', fontweight='medium')
    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.02)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def save_multi_forecast_plot(
    ground_truth_list: List[np.ndarray],
    predictions_list: List[np.ndarray],
    column_names: List[str],
    out_path: str | Path,
    start_indices: Optional[List[int]] = None,
    title: str = "Multi-Window Forecast Evaluation",
    max_cols: int = 3,
):
    """Save a multi-window forecast comparison plot.
    
    Shows multiple forecast windows from different starting points,
    useful for evaluating model performance across the dataset.
    
    Parameters
    ----------
    ground_truth_list : list of np.ndarray
        Ground truth for each window, each shape (horizon, n_features).
    predictions_list : list of np.ndarray
        Predictions for each window.
    column_names : list of str
        Names of output features.
    out_path : str or Path
        Output path.
    start_indices : list of int, optional
        Starting indices for labeling.
    title : str
        Figure title.
    max_cols : int, default 3
        Maximum columns in subplot grid.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_windows = len(ground_truth_list)
    if n_windows == 0:
        raise ValueError("No forecast windows provided")
    
    n_vars = ground_truth_list[0].shape[1]
    
    # Grid layout
    n_cols = min(max_cols, n_windows)
    n_rows = ceil(n_windows / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    
    # Collect metrics for summary
    all_mse = []
    all_mae = []
    
    for idx in range(n_windows):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        gt = ground_truth_list[idx]
        pred = predictions_list[idx]
        horizon = gt.shape[0]
        steps = np.arange(horizon)
        
        # Plot each variable
        for v in range(n_vars):
            gt_v = gt[:, v]
            pred_v = pred[:, v]
            
            # Ground truth
            ax.plot(
                steps, gt_v,
                color=COLORS['gt'],
                linewidth=1.5,
                label='GT' if v == 0 else None,
            )
            
            # Prediction
            ax.plot(
                steps, pred_v,
                color=COLORS['pred'],
                linewidth=1.5,
                linestyle='--',
                alpha=0.8,
                label='Pred' if v == 0 else None,
            )
        
        # Metrics
        mse = np.mean((gt - pred) ** 2)
        mae = np.mean(np.abs(gt - pred))
        all_mse.append(mse)
        all_mae.append(mae)
        
        # Title with start index
        start_label = f"Start: {start_indices[idx]}" if start_indices else f"Window {idx+1}"
        ax.set_title(f"{start_label}\nMSE: {mse:.4f}", fontsize=9)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)
        
        if row == n_rows - 1:
            ax.set_xlabel('Step')
    
    # Turn off unused subplots
    for idx in range(n_windows, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    
    # Summary statistics in suptitle
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    fig.suptitle(
        f"{title}\nMean MSE: {mean_mse:.4f} | Mean MAE: {mean_mae:.4f}",
        fontweight='bold',
        fontsize=12,
        y=1.02,
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'mean_mse': mean_mse, 'mean_mae': mean_mae, 'mse_list': all_mse, 'mae_list': all_mae}


def save_simulation_plot(
    real: np.ndarray,
    pred: np.ndarray,
    columns: List[str],
    out_path: str | Path,
    title: str = "Simulation Results",
):
    """Plot each output variable for simulation horizon.
    
    Parameters
    ----------
    real : np.ndarray
        Ground truth, shape (horizon, n_features).
    pred : np.ndarray
        Predictions, shape (horizon, n_features).
    columns : list of str
        Feature names.
    out_path : str or Path
        Output path.
    title : str
        Figure title.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    steps = np.arange(real.shape[0])
    n_vars = real.shape[1]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True, squeeze=False)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.plot(steps, real[:, i], color=COLORS['gt'], linewidth=2, label="Ground Truth")
        ax.plot(steps, pred[:, i], color=COLORS['pred'], linewidth=2, linestyle='--', label="Prediction")
        ax.fill_between(steps, real[:, i], pred[:, i], color=COLORS['pred'], alpha=0.15)
        ax.set_ylabel(columns[i], fontweight='medium')
        ax.legend(loc='upper right', fontsize=8)
        
        # Metrics
        mse = np.mean((real[:, i] - pred[:, i]) ** 2)
        mae = np.mean(np.abs(real[:, i] - pred[:, i]))
        ax.text(
            0.02, 0.95, f"MSE: {mse:.4f} | MAE: {mae:.4f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )
    
    axes[-1].set_xlabel("Time Step", fontweight='medium')
    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def compare_simulation_plot(
    real: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    columns: List[str],
    out_path: str | Path,
    title_prefix: str = "Simulation",
):
    """Overlay ground-truth vs predictions before/after retrain.
    
    Parameters
    ----------
    real : np.ndarray
        Ground truth, shape (horizon, n_features).
    pred_before : np.ndarray
        Predictions before retraining.
    pred_after : np.ndarray
        Predictions after retraining.
    columns : list of str
        Feature names.
    out_path : str or Path
        Output path.
    title_prefix : str
        Title prefix.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    steps = np.arange(real.shape[0])
    n_vars = real.shape[1]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True, squeeze=False)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.plot(steps, real[:, i], color=COLORS['gt'], linewidth=2, label="Ground Truth")
        ax.plot(steps, pred_before[:, i], color=COLORS['before'], linewidth=1.5, 
                linestyle='--', alpha=0.8, label="Before")
        ax.plot(steps, pred_after[:, i], color=COLORS['after'], linewidth=2, label="After")
        ax.set_ylabel(columns[i], fontweight='medium')
        ax.legend(loc='upper right', fontsize=8)
        
        # Improvement metrics
        mse_before = np.mean((real[:, i] - pred_before[:, i]) ** 2)
        mse_after = np.mean((real[:, i] - pred_after[:, i]) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        color = '#2E7D32' if improvement > 0 else '#C62828'
        ax.text(
            0.02, 0.95, f"MSE: {mse_before:.4f} → {mse_after:.4f} ({improvement:+.1f}%)",
            transform=ax.transAxes, fontsize=8, color=color,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )
    
    axes[-1].set_xlabel("Time Step", fontweight='medium')
    fig.suptitle(f"{title_prefix} (horizon={real.shape[0]})", fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def multi_compare_simulation_plot(
    real_list: List[np.ndarray],
    before_list: List[np.ndarray],
    after_list: List[np.ndarray],
    columns: List[str],
    out_path: str | Path,
    max_cols: int = 3,
    title: str = "Multi-Start Comparison",
):
    """Plot comparison for multiple start points.

    Parameters
    ----------
    real_list : list of np.ndarray
        Ground truth for each starting point.
    before_list : list of np.ndarray
        Predictions before retraining.
    after_list : list of np.ndarray
        Predictions after retraining.
    columns : list of str
        Feature names.
    out_path : str or Path
        Output path.
    max_cols : int, default 3
        Maximum columns in grid.
    title : str
        Figure title.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_points = len(real_list)
    if n_points == 0:
        raise ValueError("real_list is empty – nothing to plot")

    # Grid layout
    n_cols = min(max_cols, n_points)
    if n_cols == 3 and n_points < 5:
        n_cols = 2
    n_rows = ceil(n_points / n_cols)

    n_vars = real_list[0].shape[1]
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes = axes.reshape(n_rows, n_cols)
    
    # Collect metrics
    improvements = []

    for idx in range(n_points):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        steps = np.arange(real_list[idx].shape[0])
        
        for v in range(n_vars):
            ax.plot(steps, real_list[idx][:, v], color=COLORS['gt'], 
                   linewidth=1.5, label="GT" if v == 0 else "")
            ax.plot(steps, before_list[idx][:, v], color=COLORS['before'],
                   linewidth=1.5, linestyle='--', alpha=0.8, label="Before" if v == 0 else "")
            ax.plot(steps, after_list[idx][:, v], color=COLORS['after'],
                   linewidth=1.5, label="After" if v == 0 else "")
        
        # Calculate improvement
        mse_before = np.mean((real_list[idx] - before_list[idx]) ** 2)
        mse_after = np.mean((real_list[idx] - after_list[idx]) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        improvements.append(improvement)
        
        color = '#2E7D32' if improvement > 0 else '#C62828'
        ax.set_title(f"Start {idx+1}: {improvement:+.1f}%", fontsize=10, color=color)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Turn off unused subplots
    for idx in range(n_points, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis('off')

    # Summary
    mean_improvement = np.mean(improvements)
    color = '#2E7D32' if mean_improvement > 0 else '#C62828'
    fig.suptitle(
        f"{title}\nMean Improvement: {mean_improvement:+.1f}%",
        fontweight='bold', fontsize=12, color=color, y=1.02,
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def save_training_summary(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    column_names: List[str],
    out_path: str | Path,
    model_name: str = "Model",
):
    """Create a comprehensive training summary figure.
    
    Combines loss curve and forecast visualization in a single figure.
    
    Parameters
    ----------
    train_losses : list of float
        Training losses.
    val_losses : list of float or None
        Validation losses.
    ground_truth : np.ndarray
        Ground truth for forecast, shape (horizon, n_features).
    predictions : np.ndarray
        Predictions, shape (horizon, n_features).
    column_names : list of str
        Feature names.
    out_path : str or Path
        Output path.
    model_name : str
        Model name for title.
    """
    _setup_figure_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_vars = ground_truth.shape[1]
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 4 + 2.5 * n_vars))
    
    # Top row: Loss curve (spans full width)
    ax_loss = fig.add_subplot(n_vars + 1, 1, 1)
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, color=COLORS['train'], linewidth=2, 
                 marker='o' if len(epochs) <= 20 else None, markersize=4, label='Train')
    
    if val_losses and any(v is not None for v in val_losses):
        val_epochs = [i+1 for i, v in enumerate(val_losses) if v is not None]
        val_values = [v for v in val_losses if v is not None]
        ax_loss.plot(val_epochs, val_values, color=COLORS['val'], linewidth=2,
                    marker='s' if len(val_epochs) <= 20 else None, markersize=4, label='Val')
    
    ax_loss.set_xlabel('Epoch', fontweight='medium')
    ax_loss.set_ylabel('Loss', fontweight='medium')
    ax_loss.set_title('Training Progress', fontweight='bold')
    ax_loss.legend(loc='upper right')
    
    # Add final loss annotation
    final_train = train_losses[-1]
    ax_loss.text(0.02, 0.95, f"Final: {final_train:.2e}", transform=ax_loss.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom rows: Forecast for each variable
    horizon = ground_truth.shape[0]
    steps = np.arange(horizon)
    
    for i in range(n_vars):
        ax = fig.add_subplot(n_vars + 1, 1, i + 2)
        
        gt = ground_truth[:, i]
        pred = predictions[:, i]
        
        ax.plot(steps, gt, color=COLORS['gt'], linewidth=2, label='Ground Truth')
        ax.plot(steps, pred, color=COLORS['pred'], linewidth=2, linestyle='--', label='Prediction')
        ax.fill_between(steps, gt, pred, color=COLORS['pred'], alpha=0.15)
        
        mse = np.mean((gt - pred) ** 2)
        mae = np.mean(np.abs(gt - pred))
        
        ax.set_ylabel(column_names[i], fontweight='medium')
        ax.legend(loc='upper right', fontsize=8)
        ax.text(0.02, 0.95, f"MSE: {mse:.4f} | MAE: {mae:.4f}",
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Time Step', fontweight='medium')
    
    fig.suptitle(f'{model_name} Training Summary', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close() 