from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def save_loss_plot(train_losses: List[float],
                   val_losses: List[float] | None,
                   out_path: str | Path,
                   title: str = "Loss curve"):
    """Save a PNG figure of train/val loss vs epoch."""
    out_path = Path(out_path)
    plt.figure(figsize=(5, 3))
    plt.plot(train_losses, label="train")
    if val_losses is not None and any(v is not None for v in val_losses):
        plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_simulation_plot(real: np.ndarray,
                         pred: np.ndarray,
                         columns: List[str],
                         out_path: str | Path):
    """Plot each output variable for simulation horizon."""
    out_path = Path(out_path)
    steps = np.arange(real.shape[0])
    n_vars = real.shape[1]
    fig, axes = plt.subplots(n_vars, 1, figsize=(6, 2 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(steps, real[:, i], label="real")
        ax.plot(steps, pred[:, i], label="pred")
        ax.set_ylabel(columns[i])
        ax.legend()
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compare_simulation_plot(real: np.ndarray,
                             pred_before: np.ndarray,
                             pred_after: np.ndarray,
                             columns: List[str],
                             out_path: str | Path,
                             title_prefix: str = "Simulation"):
    """Overlay ground-truth vs predictions before/after retrain."""
    out_path = Path(out_path)
    steps = np.arange(real.shape[0])
    n_vars = real.shape[1]
    fig, axes = plt.subplots(n_vars, 1, figsize=(6, 2 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(steps, real[:, i], label="GT", color="black")
        ax.plot(steps, pred_before[:, i], label="before", linestyle="--")
        ax.plot(steps, pred_after[:, i], label="after")
        ax.set_ylabel(columns[i])
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{title_prefix} (horizon={real.shape[0]})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def multi_compare_simulation_plot(real_list: List[np.ndarray],
                                  before_list: List[np.ndarray],
                                  after_list: List[np.ndarray],
                                  columns: List[str],
                                  out_path: str | Path,
                                  max_cols: int = 3):
    """Plot comparison for multiple start points.

    Layout adapts automatically:
    - up to *max_cols* columns (default 3)
    - rows = ceil(n_points / n_cols)
    """
    out_path = Path(out_path)
    n_points = len(real_list)
    if n_points == 0:
        raise ValueError("real_list is empty – nothing to plot")

    # Choose number of columns (1..max_cols) trying to get a square-ish grid
    n_cols = min(max_cols, n_points)
    # Prefer 2 columns instead of 3 for e.g. 2-4 plots so each subplot is larger
    if n_cols == 3 and n_points < 5:
        n_cols = 2
    n_rows = ceil(n_points / n_cols)

    n_vars = real_list[0].shape[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.5 * n_rows), sharex=False)
    # axes can be 2-D or 1-D depending on rows/cols
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(n_points):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        steps = np.arange(real_list[idx].shape[0])
        for v in range(n_vars):
            ax.plot(steps, real_list[idx][:, v], color="black", label="GT" if v == 0 else "")
            ax.plot(steps, before_list[idx][:, v], linestyle="--", label="before" if v == 0 else "")
            ax.plot(steps, after_list[idx][:, v], label="after" if v == 0 else "")
        ax.set_title(f"Start {idx}")
        if idx == 0:
            ax.legend(fontsize=8)
    # Turn off any unused subplots
    for idx in range(n_points, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    axes_flat = axes.flatten()
    axes_flat[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close() 