"""Multi-environment rollout utilities for world model training."""

from __future__ import annotations

from typing import Dict, List, Optional, Literal

import numpy as np
import torch

from ..data.dataset import GroupedTimeSeriesDataset
from ..data.sampling import SamplingStrategy
from ..models.base import WorldModelBase


def batch_rollout(
    model: WorldModelBase,
    dataset: GroupedTimeSeriesDataset,
    start_indices: np.ndarray,
    horizons: np.ndarray,
    warmup_len: int,
    feedback: Literal["model", "teacher", "mixed"] = "model",
    teacher_forcing_ratio: float = 0.0,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Perform batched multi-environment rollouts.
    
    This function samples multiple starting points and horizons, then performs
    rollouts in parallel. This is the core of the "see every possible path"
    training approach.
    
    Parameters
    ----------
    model : WorldModelBase
        World model to roll out.
    dataset : GroupedTimeSeriesDataset
        Dataset containing the time-series data.
    start_indices : np.ndarray
        Starting indices for each rollout, shape (batch_size,).
    horizons : np.ndarray
        Horizon lengths for each rollout, shape (batch_size,).
    warmup_len : int
        Length of warmup sequence for state initialization.
    feedback : {"model", "teacher", "mixed"}
        Feedback mode for rollout.
    teacher_forcing_ratio : float
        Ratio for mixed feedback mode.
    device : torch.device or str
        Device for computation.
    
    Returns
    -------
    dict
        Dictionary containing:
        - "predictions": List of prediction tensors (variable horizon lengths)
        - "targets": List of target tensors
        - "horizons": Tensor of horizon lengths
        - "start_indices": Tensor of start indices
    
    Notes
    -----
    Since horizons can vary, we return lists of tensors rather than a single
    padded tensor. The caller can pad if needed for loss computation.
    """
    device = torch.device(device)
    batch_size = len(start_indices)
    
    predictions_list = []
    targets_list = []
    
    for i in range(batch_size):
        start_idx = int(start_indices[i])
        horizon = int(horizons[i])
        
        # Get warmup window
        warmup_data = dataset.get_warmup_window(start_idx, warmup_len)
        # For world models, concatenate inputs (control+exo) and outputs for full sequence
        warmup_full = np.concatenate([
            warmup_data["inputs"],
            warmup_data["outputs"]
        ], axis=-1)
        warmup_inputs = torch.tensor(
            warmup_full, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, warmup_len, control_dim+exo_dim+output_dim)
        
        # Get rollout data
        rollout_data = dataset.get_rollout_slice(start_idx, horizon)
        rollout_inputs_np = rollout_data["inputs"]  # (horizon, input_dim)
        targets_np = rollout_data["targets"]  # (horizon, output_dim)
        
        # Split inputs into controls and exogenous
        # For now, assume all inputs are controls+exogenous
        # TODO: Make this more flexible based on dataset structure
        control_dim = len([c for c in dataset.input_cols if c in dataset.groups.get("control", [])])
        exo_dim = len([c for c in dataset.input_cols if c in dataset.groups.get("exogenous", [])])
        
        if control_dim + exo_dim == 0:
            # Fallback: split evenly or use all as controls
            control_dim = rollout_inputs_np.shape[-1]
            exo_dim = 0
        
        controls = torch.tensor(
            rollout_inputs_np[:, :control_dim], dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, horizon, control_dim)
        
        if exo_dim > 0:
            exogenous = torch.tensor(
                rollout_inputs_np[:, control_dim:control_dim+exo_dim],
                dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, horizon, exo_dim)
        else:
            exogenous = torch.zeros(1, horizon, 0, device=device)
        
        targets = torch.tensor(
            targets_np, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, horizon, output_dim)
        
        # Perform rollout
        rollout_result = model.rollout(
            warmup_seq={"inputs": warmup_inputs},
            rollout_inputs={"controls": controls, "exogenous": exogenous},
            horizon=horizon,
            feedback=feedback,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets if feedback in ["teacher", "mixed"] else None,
        )
        
        predictions = rollout_result["predictions"].squeeze(0)  # (horizon, output_dim)
        predictions_list.append(predictions)
        targets_list.append(targets.squeeze(0))
    
    return {
        "predictions": predictions_list,
        "targets": targets_list,
        "horizons": torch.tensor(horizons, dtype=torch.long),
        "start_indices": torch.tensor(start_indices, dtype=torch.long),
    }


def batch_rollout_padded(
    model: WorldModelBase,
    dataset: GroupedTimeSeriesDataset,
    start_indices: np.ndarray,
    horizons: np.ndarray,
    warmup_len: int,
    feedback: Literal["model", "teacher", "mixed"] = "model",
    teacher_forcing_ratio: float = 0.0,
    device: torch.device | str = "cpu",
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Perform batched rollouts with padding to uniform horizon.
    
    This is a convenience wrapper around batch_rollout that pads all
    rollouts to the maximum horizon length. This makes it easier to
    compute batched losses.
    
    Parameters
    ----------
    model : WorldModelBase
        World model to roll out.
    dataset : GroupedTimeSeriesDataset
        Dataset containing the time-series data.
    start_indices : np.ndarray
        Starting indices for each rollout, shape (batch_size,).
    horizons : np.ndarray
        Horizon lengths for each rollout, shape (batch_size,).
    warmup_len : int
        Length of warmup sequence.
    feedback : {"model", "teacher", "mixed"}
        Feedback mode.
    teacher_forcing_ratio : float
        Ratio for mixed feedback.
    device : torch.device or str
        Device for computation.
    pad_value : float
        Value to use for padding.
    
    Returns
    -------
    dict
        Dictionary containing:
        - "predictions": (batch_size, max_horizon, output_dim)
        - "targets": (batch_size, max_horizon, output_dim)
        - "mask": (batch_size, max_horizon) - 1 for valid, 0 for padded
        - "horizons": (batch_size,)
    """
    result = batch_rollout(
        model, dataset, start_indices, horizons, warmup_len,
        feedback, teacher_forcing_ratio, device
    )
    
    predictions_list = result["predictions"]
    targets_list = result["targets"]
    horizons_tensor = result["horizons"]
    
    batch_size = len(predictions_list)
    max_horizon = int(horizons.max())
    output_dim = predictions_list[0].shape[-1]
    device = predictions_list[0].device
    
    # Create padded tensors
    predictions_padded = torch.full(
        (batch_size, max_horizon, output_dim),
        pad_value, dtype=torch.float32, device=device
    )
    targets_padded = torch.full(
        (batch_size, max_horizon, output_dim),
        pad_value, dtype=torch.float32, device=device
    )
    mask = torch.zeros(batch_size, max_horizon, dtype=torch.bool, device=device)
    
    # Fill in actual values
    for i in range(batch_size):
        h = int(horizons[i])
        predictions_padded[i, :h] = predictions_list[i]
        targets_padded[i, :h] = targets_list[i]
        mask[i, :h] = True
    
    return {
        "predictions": predictions_padded,
        "targets": targets_padded,
        "mask": mask,
        "horizons": horizons_tensor,
    }


def rollout_autoregressive(
    model: torch.nn.Module,
    x0: torch.Tensor,
    h_max: int,
    device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Legacy autoregressive rollout for backward compatibility.
    
    This function maintains compatibility with the old rollout interface.
    For new code, use WorldModelBase.rollout() instead.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model with forward() method.
    x0 : torch.Tensor
        Initial sequence, shape (batch_size, seq_len, input_dim).
    h_max : int
        Number of steps to roll out.
    device : torch.device or str
        Device for computation.
    
    Returns
    -------
    torch.Tensor
        Predictions of shape (batch_size, h_max, output_dim).
    """
    device = torch.device(device)
    model.eval()
    preds = []
    x = x0.to(device)
    
    with torch.no_grad():
        for _ in range(h_max):
            y = model(x)  # (B, pred_len, F_out)
            step = y[:, -1:, :]  # last step, keep time dim
            preds.append(step.cpu())
            x = torch.cat([x, step], dim=1)[:, 1:, :]  # slide window
    
    return torch.cat(preds, dim=1)  # (B, h_max, F_out)

