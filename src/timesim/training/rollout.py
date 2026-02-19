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
    
    control_positions = list(getattr(dataset, "control_positions", []))
    known_exo_positions = list(getattr(dataset, "known_exo_positions", []))
    control_dim = len(control_positions)
    exo_dim = len(known_exo_positions)
    
    # Preallocate numpy arrays for batch data (avoid list appends)
    warmup_controls = np.zeros((batch_size, warmup_len, control_dim), dtype=np.float32)
    warmup_exogenous = np.zeros((batch_size, warmup_len, exo_dim), dtype=np.float32)
    warmup_outputs = np.zeros((batch_size, warmup_len, output_dim), dtype=np.float32)
    rollout_inputs = np.zeros((batch_size, max_horizon, input_dim), dtype=np.float32)
    rollout_targets = np.zeros((batch_size, max_horizon, output_dim), dtype=np.float32)
    
    # Extract data for each batch item (unavoidable due to variable positions)
    # But we do all numpy operations first, then a single tensor conversion
    values = dataset.values
    in_idx = dataset.in_idx
    out_idx = dataset.out_idx
    control_idx = [in_idx[pos] for pos in control_positions]
    exo_idx = [in_idx[pos] for pos in known_exo_positions]
    
    for i in range(batch_size):
        start_idx = int(start_indices[i])
        horizon = int(horizons[i])
        
        # Warmup window: [start_idx - warmup_len : start_idx]
        warmup_slice = values[start_idx - warmup_len : start_idx]
        if control_dim > 0:
            warmup_controls[i] = warmup_slice[:, control_idx]
        if exo_dim > 0:
            warmup_exogenous[i] = warmup_slice[:, exo_idx]
        warmup_outputs[i] = warmup_slice[:, out_idx]
        
        # Rollout window: [start_idx : start_idx + horizon]
        rollout_slice = values[start_idx : start_idx + horizon]
        rollout_inputs[i, :horizon] = rollout_slice[:, in_idx]
        rollout_targets[i, :horizon] = rollout_slice[:, out_idx]
    
    # Build warmup_full in the same semantic order used during rollout steps:
    # [controls, known_exogenous(+time), previous_outputs].
    warmup_full = np.concatenate([warmup_controls, warmup_exogenous, warmup_outputs], axis=-1)
    
    return {
        "warmup_controls": torch.from_numpy(warmup_controls).to(device),
        "warmup_exogenous": torch.from_numpy(warmup_exogenous).to(device),
        "warmup_outputs": torch.from_numpy(warmup_outputs).to(device),
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
    
    # Build feature positions from centralized dataset taxonomy.
    control_positions = list(getattr(dataset, "control_positions", []))
    known_exo_positions = list(getattr(dataset, "known_exo_positions", []))
    control_dim = len(control_positions)
    exo_dim = len(known_exo_positions)
    
    # Prepare batch data with single conversion (HOT PATH optimization)
    batch_data = _prepare_batch_data(
        dataset,
        start_indices,
        horizons,
        warmup_len,
        max_horizon,
        device,
    )

    warmup_controls = batch_data["warmup_controls"]
    warmup_exogenous = batch_data["warmup_exogenous"]
    warmup_outputs = batch_data["warmup_outputs"]
    warmup_full = batch_data["warmup_full"]  # (B, warmup_len, model_input_dim)
    rollout_inputs_all = batch_data["rollout_inputs"]  # (B, max_horizon, input)
    rollout_targets_all = batch_data["rollout_targets"]  # (B, max_horizon, output)
    
    # Split rollout inputs into controls and exogenous
    if control_dim > 0:
        control_idx = torch.as_tensor(control_positions, dtype=torch.long, device=device)
        controls = torch.index_select(rollout_inputs_all, dim=2, index=control_idx)
    else:
        controls = torch.zeros(batch_size, max_horizon, 0, device=device)

    if exo_dim > 0:
        exo_idx = torch.as_tensor(known_exo_positions, dtype=torch.long, device=device)
        exogenous = torch.index_select(rollout_inputs_all, dim=2, index=exo_idx)
    else:
        exogenous = torch.zeros(batch_size, max_horizon, 0, device=device)
    
    # Check if all horizons are the same (enables true batched rollout)
    uniform_horizon = (horizons == horizons[0]).all()
    
    if uniform_horizon:
        # HOT PATH: True batched rollout - single model.rollout call
        horizon = int(horizons[0])
        targets_for_rollout = rollout_targets_all[:, :horizon]

        rollout_result = model.rollout(
            warmup_seq={
                "inputs": warmup_full,
                "controls": warmup_controls,
                "exogenous": warmup_exogenous,
                "outputs": warmup_outputs,
            },
            rollout_inputs={"controls": controls[:, :horizon], "exogenous": exogenous[:, :horizon]},
            horizon=horizon,
            feedback=feedback,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets_for_rollout,
        )

        predictions = rollout_result["predictions"]  # (B, horizon, output)
        targets = rollout_targets_all[:, :horizon]
        exogenous_h = exogenous[:, :horizon]

        # Convert to list format for compatibility (no copy, just views)
        predictions_list = [predictions[i] for i in range(batch_size)]
        targets_list = [targets[i] for i in range(batch_size)]
        exogenous_list = [exogenous_h[i] for i in range(batch_size)]
        extra_lists: Dict[str, List[torch.Tensor]] = {}
        for key, val in rollout_result.items():
            if key in {"predictions", "states"}:
                continue
            if torch.is_tensor(val) and val.shape[0] == batch_size:
                extra_lists[key] = [val[i] for i in range(batch_size)]
    else:
        # Variable horizons: need per-item rollouts (less common case)
        predictions_list = []
        targets_list = []
        exogenous_list = []
        extra_lists: Dict[str, List[torch.Tensor]] = {}

        for i in range(batch_size):
            horizon = int(horizons[i])
            
            # Slice pre-prepared tensors (no conversion, just indexing)
            warmup_i = warmup_full[i:i+1]  # (1, warmup_len, F)
            controls_i = controls[i:i+1, :horizon]  # (1, horizon, C)
            exogenous_i = exogenous[i:i+1, :horizon]  # (1, horizon, E)
            targets_i = rollout_targets_all[i:i+1, :horizon]  # (1, horizon, O)
            
            rollout_result = model.rollout(
                warmup_seq={
                    "inputs": warmup_i,
                    "controls": warmup_controls[i:i+1],
                    "exogenous": warmup_exogenous[i:i+1],
                    "outputs": warmup_outputs[i:i+1],
                },
                rollout_inputs={"controls": controls_i, "exogenous": exogenous_i},
                horizon=horizon,
                feedback=feedback,
                teacher_forcing_ratio=teacher_forcing_ratio,
                targets=targets_i,
            )

            predictions_list.append(rollout_result["predictions"].squeeze(0))
            targets_list.append(targets_i.squeeze(0))
            exogenous_list.append(exogenous_i.squeeze(0))
            for key, val in rollout_result.items():
                if key in {"predictions", "states"}:
                    continue
                if not torch.is_tensor(val):
                    continue
                val_squeezed = val.squeeze(0) if val.dim() > 0 else val
                extra_lists.setdefault(key, []).append(val_squeezed)

    out = {
        "predictions": predictions_list,
        "targets": targets_list,
        "exogenous": exogenous_list,
        "horizons": torch.as_tensor(horizons, dtype=torch.long, device=device),
        "start_indices": torch.as_tensor(start_indices, dtype=torch.long, device=device),
    }
    out.update(extra_lists)
    return out


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
    extra_keys = [
        k for k in result.keys()
        if k not in {"predictions", "targets", "horizons", "start_indices"}
    ]
    
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
        extras_padded: Dict[str, torch.Tensor] = {}
        for key in extra_keys:
            extra_list = result.get(key, [])
            if extra_list and torch.is_tensor(extra_list[0]):
                extras_padded[key] = torch.stack(extra_list, dim=0)
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

        extras_padded = {}
        for key in extra_keys:
            extra_list = result.get(key, [])
            if not extra_list or not torch.is_tensor(extra_list[0]):
                continue
            v0 = extra_list[0]
            if v0.dim() == 1:
                padded = torch.full(
                    (batch_size, max_horizon),
                    pad_value,
                    dtype=v0.dtype,
                    device=v0.device,
                )
                for i in range(batch_size):
                    h = int(horizons[i])
                    padded[i, :h] = extra_list[i]
                extras_padded[key] = padded
            elif v0.dim() == 2:
                padded = torch.full(
                    (batch_size, max_horizon, v0.shape[-1]),
                    pad_value,
                    dtype=v0.dtype,
                    device=v0.device,
                )
                for i in range(batch_size):
                    h = int(horizons[i])
                    padded[i, :h, :] = extra_list[i]
                extras_padded[key] = padded

    out = {
        "predictions": predictions_padded,
        "targets": targets_padded,
        "mask": mask,
        "horizons": horizons_tensor,
    }
    out.update(extras_padded)
    return out


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
