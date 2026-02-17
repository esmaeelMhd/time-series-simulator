from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import Module

from ..data.dataset import GroupedTimeSeriesDataset
from .plotting import save_simulation_plot


def simulate_autoregressive(model: Module,
                             df: pd.DataFrame,
                             groups: Dict[str, List[str]],
                             input_groups: List[str],
                             output_groups: List[str],
                             seq_len: int,
                             horizon: int,
                             device: str | torch.device = "cpu",
                             out_fig: str | Path | None = None,
                             run_dir: str | Path | None = None,
                             start_idx: int | None = None):
    """Run an autoregressive simulation and optionally save a plot.

    Model gets *input_groups* as input, predicts *output_groups*; any missing
    input feature (i.e. in input_groups but not in output_groups) is filled
    using ground-truth.
    """
    device = torch.device(device)
    model.eval()

    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])

    # Determine start index
    if start_idx is None:
        start_max = len(df) - (seq_len + horizon) - 1
        start_idx = np.random.randint(0, start_max)

    window_df = df.iloc[start_idx : start_idx + seq_len + horizon].copy()

    # Collect results
    preds = []

    # Use scaler from dataset for consistency
    dataset = GroupedTimeSeriesDataset(window_df, groups, input_groups, output_groups, seq_len, horizon, scale=True)
    scaler = dataset.scaler

    # Work on numpy copy to allow in-place updates
    needed_cols = list(dict.fromkeys(input_cols + output_cols))
    values_full = window_df[needed_cols].values
    values = scaler.transform(values_full)
    # keep only input columns for rolling buffer
    values = values[:, [needed_cols.index(c) for c in input_cols]]

    for step in range(horizon):
        # slice last seq_len rows as input
        x_window = values[step : step + seq_len]
        x_tensor = torch.tensor(x_window, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()[0, -1]  # (F_out,)
        preds.append(pred)

        # Insert pred into values if its columns overlap inputs
        for j, col in enumerate(output_cols):
            if col in input_cols:
                col_idx = input_cols.index(col)
                values[step + seq_len, col_idx] = pred[j]
        # otherwise keep real value for that input feature

    preds = np.stack(preds)  # (horizon, F_out)
    # Build array of real outputs in original units
    real_full = window_df[needed_cols].values[seq_len:seq_len + horizon]
    output_positions = [needed_cols.index(c) for c in output_cols]
    real = real_full[:, output_positions]

    # Inverse-transform predictions to original output units
    pred_full_scaled = np.zeros((horizon, len(needed_cols)), dtype=np.float32)
    pred_full_scaled[:, output_positions] = preds
    pred_full = scaler.inverse_transform(pred_full_scaled)
    pred = pred_full[:, output_positions]

    if out_fig is not None:
        save_simulation_plot(real, pred, output_cols, out_fig)

    if run_dir is not None:
        np.save(Path(run_dir)/"simulation_real.npy", real)
        np.save(Path(run_dir)/"simulation_pred.npy", pred)

    return real, pred, start_idx 
