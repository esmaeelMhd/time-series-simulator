"""Multi-environment rollout utilities for world model training.

HOT PATH: This module contains performance-critical rollout code.
Key optimizations applied:
- Batched tensor operations instead of Python loops
- Preallocated buffers for predictions
- Single data conversion at batch boundaries
- Vectorized padding operations
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal

import numpy as np
import torch

from ..data.dataset import GroupedTimeSeriesDataset
from ..data.sampling import SamplingStrategy
from ..models.base import WorldModelBase


def _prepare_batch_data(
    dataset: GroupedTimeSeriesDataset,
    start_indices: np.ndarray,
    horizons: np.ndarray,
    warmup_len: int,
    max_horizon: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare batched data for rollout with a single conversion to tensors.
    
    HOT PATH: This function extracts data for all batch items at once,
    avoiding per-item Python loops and tensor conversions.
    
    Parameters
    ----------
    dataset : GroupedTimeSeriesDataset
        Dataset containing the time-series data.
    start_indices : np.ndarray
        Starting indices, shape (batch_size,).
    horizons : np.ndarray
        Horizon lengths, shape (batch_size,).
    warmup_len : int
        Length of warmup sequence.
    max_horizon : int
        Maximum horizon for padding.
    device : torch.device
        Target device for tensors.
    
    Returns
    -------
    dict
        Dictionary with batched tensors ready for rollout.
    """
    batch_size = len(start_indices)
    input_dim = len(dataset.in_idx)
    output_dim = len(dataset.out_idx)
    
    # Determine which output indices are NOT already covered by input indices.
    # When output_groups overlap with input_groups (e.g. "objective" in both),
    # out_idx entries already appear in in_idx.  We must NOT duplicate them in
    # warmup_full, otherwise the tensor dimension won't match the model's
    # input_dim (= len(union(in_idx, out_idx))).
    in_idx_set = set(dataset.in_idx)
    extra_out_idx = [idx for idx in dataset.out_idx if idx not in in_idx_set]
    extra_out_dim = len(extra_out_idx)
    
    # Preallocate numpy arrays for batch data (avoid list appends)
    warmup_inputs = np.zeros((batch_size, warmup_len, input_dim), dtype=np.float32)
    if extra_out_dim > 0:
        warmup_extra_out = np.zeros((batch_size, warmup_len, extra_out_dim), dtype=np.float32)
    rollout_inputs = np.zeros((batch_size, max_horizon, input_dim), dtype=np.float32)
    rollout_targets = np.zeros((batch_size, max_horizon, output_dim), dtype=np.float32)
    
    # Extract data for each batch item (unavoidable due to variable positions)
    # But we do all numpy operations first, then a single tensor conversion
    values = dataset.values
    in_idx = dataset.in_idx
    out_idx = dataset.out_idx
    
    for i in range(batch_size):
        start_idx = int(start_indices[i])
        horizon = int(horizons[i])
        
        # Warmup window: [start_idx - warmup_len : start_idx]
        warmup_slice = values[start_idx - warmup_len : start_idx]
        warmup_inputs[i] = warmup_slice[:, in_idx]
        if extra_out_dim > 0:
            warmup_extra_out[i] = warmup_slice[:, extra_out_idx]
        
        # Rollout window: [start_idx : start_idx + horizon]
        rollout_slice = values[start_idx : start_idx + horizon]
        rollout_inputs[i, :horizon] = rollout_slice[:, in_idx]
        rollout_targets[i, :horizon] = rollout_slice[:, out_idx]
    
    # Single batch conversion to tensors (Rule 7: avoid repeated conversions)
    # Build warmup_full: input columns + any output columns not already in inputs.
    # This ensures warmup_full dim == model's input_dim == len(union(in_idx, out_idx))
    if extra_out_dim > 0:
        warmup_full = np.concatenate([warmup_inputs, warmup_extra_out], axis=-1)
    else:
        warmup_full = warmup_inputs
    
    return {
        "warmup_full": torch.from_numpy(warmup_full).to(device),
        "rollout_inputs": torch.from_numpy(rollout_inputs).to(device),
        "rollout_targets": torch.from_numpy(rollout_targets).to(device),
    }


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
    
    HOT PATH: This function is called every training step.
    Optimizations:
    - Batch data preparation with single numpy->tensor conversion
    - True batched rollout when horizons are uniform
    - Minimized Python loop overhead
    
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
    max_horizon = int(horizons.max())
    
    # Compute dimension info once (Rule 4: minimize repeated calls)
    control_cols = dataset.groups.get("control", [])
    exo_cols = dataset.groups.get("exogenous", [])
    control_dim = len([c for c in dataset.input_cols if c in control_cols])
    exo_dim = len([c for c in dataset.input_cols if c in exo_cols])
    
    if control_dim + exo_dim == 0:
        control_dim = len(dataset.in_idx)
        exo_dim = 0
    
    # Prepare batch data with single conversion (HOT PATH optimization)
    batch_data = _prepare_batch_data(
        dataset, start_indices, horizons, warmup_len, max_horizon, device
    )
    
    warmup_full = batch_data["warmup_full"]  # (B, warmup_len, model_input_dim)
    rollout_inputs_all = batch_data["rollout_inputs"]  # (B, max_horizon, input)
    rollout_targets_all = batch_data["rollout_targets"]  # (B, max_horizon, output)
    
    # Split rollout inputs into controls and exogenous
    controls = rollout_inputs_all[:, :, :control_dim]  # (B, max_horizon, C)
    if exo_dim > 0:
        exogenous = rollout_inputs_all[:, :, control_dim:control_dim+exo_dim]
    else:
        exogenous = torch.zeros(batch_size, max_horizon, 0, device=device)
    
    # Check if all horizons are the same (enables true batched rollout)
    uniform_horizon = (horizons == horizons[0]).all()
    
    if uniform_horizon:
        # HOT PATH: True batched rollout - single model.rollout call
        horizon = int(horizons[0])
        targets_for_feedback = rollout_targets_all[:, :horizon] if feedback in ["teacher", "mixed"] else None
        
        rollout_result = model.rollout(
            warmup_seq={"inputs": warmup_full},
            rollout_inputs={"controls": controls[:, :horizon], "exogenous": exogenous[:, :horizon]},
            horizon=horizon,
            feedback=feedback,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets_for_feedback,
        )
        
        predictions = rollout_result["predictions"]  # (B, horizon, output)
        targets = rollout_targets_all[:, :horizon]
        
        # Convert to list format for compatibility (no copy, just views)
        predictions_list = [predictions[i] for i in range(batch_size)]
        targets_list = [targets[i] for i in range(batch_size)]
    else:
        # Variable horizons: need per-item rollouts (less common case)
        predictions_list = []
        targets_list = []
        
        for i in range(batch_size):
            horizon = int(horizons[i])
            
            # Slice pre-prepared tensors (no conversion, just indexing)
            warmup_i = warmup_full[i:i+1]  # (1, warmup_len, F)
            controls_i = controls[i:i+1, :horizon]  # (1, horizon, C)
            exogenous_i = exogenous[i:i+1, :horizon]  # (1, horizon, E)
            targets_i = rollout_targets_all[i:i+1, :horizon]  # (1, horizon, O)
            
            rollout_result = model.rollout(
                warmup_seq={"inputs": warmup_i},
                rollout_inputs={"controls": controls_i, "exogenous": exogenous_i},
                horizon=horizon,
                feedback=feedback,
                teacher_forcing_ratio=teacher_forcing_ratio,
                targets=targets_i if feedback in ["teacher", "mixed"] else None,
            )
            
            predictions_list.append(rollout_result["predictions"].squeeze(0))
            targets_list.append(targets_i.squeeze(0))
    
    return {
        "predictions": predictions_list,
        "targets": targets_list,
        "horizons": torch.as_tensor(horizons, dtype=torch.long, device=device),
        "start_indices": torch.as_tensor(start_indices, dtype=torch.long, device=device),
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
    
    HOT PATH: This is called every training step.
    Optimizations:
    - Vectorized mask creation using torch.arange broadcasting
    - Efficient padding using pre-allocated tensors
    - Fast path for uniform horizons (no per-item loop)
    
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
    dev = predictions_list[0].device
    
    # Check if all horizons are uniform (fast path)
    uniform_horizon = (horizons == horizons[0]).all()
    
    if uniform_horizon:
        # HOT PATH: Fast path - just stack tensors (no padding needed)
        predictions_padded = torch.stack(predictions_list, dim=0)
        targets_padded = torch.stack(targets_list, dim=0)
        mask = torch.ones(batch_size, max_horizon, dtype=torch.bool, device=dev)
    else:
        # Variable horizons: need padding
        # Preallocate output tensors (Rule 5: no allocations in loop)
        predictions_padded = torch.full(
            (batch_size, max_horizon, output_dim),
            pad_value, dtype=torch.float32, device=dev
        )
        targets_padded = torch.full(
            (batch_size, max_horizon, output_dim),
            pad_value, dtype=torch.float32, device=dev
        )
        
        # Vectorized mask creation (Rule 2: no Python loops for data ops)
        # mask[i, j] = True if j < horizons[i]
        time_indices = torch.arange(max_horizon, device=dev).unsqueeze(0)  # (1, max_horizon)
        horizon_expanded = horizons_tensor.unsqueeze(1)  # (batch_size, 1)
        mask = time_indices < horizon_expanded  # (batch_size, max_horizon)
        
        # Fill in actual values using indexing
        # This loop is unavoidable due to variable lengths, but we avoid
        # creating new tensors inside - just assignment
        for i in range(batch_size):
            h = int(horizons[i])
            predictions_padded[i, :h] = predictions_list[i]
            targets_padded[i, :h] = targets_list[i]
    
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

