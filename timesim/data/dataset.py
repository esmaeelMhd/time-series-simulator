from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .stamps import add_time_features
from ..utils.scaler import MinMaxScaler


class TimeSeriesDataset(Dataset):
    """Windowed dataset from raw numpy array or DataFrame."""

    def __init__(self,
                 series: pd.DataFrame | np.ndarray,
                 seq_len: int,
                 pred_len: int,
                 scale: bool = True,
                 add_time: bool = False,
                 time_features_cfg: Optional[Dict[str, Any]] = None):
        if isinstance(series, np.ndarray):
            series = pd.DataFrame(series)
        self.series = series.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len

        if add_time:
            if not isinstance(self.series.index, pd.DatetimeIndex):
                raise ValueError("Time features requested but index is not DateTime")
            tf_cfg = dict(time_features_cfg or {})
            tf_cfg.pop("enabled", None)
            self.series = add_time_features(self.series, **tf_cfg)

        self.scaler = None
        if scale:
            self.scaler = MinMaxScaler().fit(self.series.values)
            scaled = self.scaler.transform(self.series.values)
            self.series.iloc[:, :] = scaled

        self.values = self.series.values.astype(np.float32)

        # Sanity check – any NaNs at this stage will break MSE and yield NaN
        if np.isnan(self.values).any():
            raise ValueError("NaNs remain in TimeSeriesDataset after preprocessing. Please inspect the raw data and preprocessing pipeline.")

    def __len__(self):
        return len(self.values) - (self.seq_len + self.pred_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.values[idx : idx + self.seq_len]
        y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)


class GroupedTimeSeriesDataset(Dataset):
    """Dataset that allows selecting columns by semantic group (C/E/O).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe *already time sorted*.
    groups : dict[str, list[str]]
        Mapping {"control": [...], "exogenous": [...], "objective": [...]}.
    input_groups : list[str]
        Names of groups whose columns will form the **input** features.
    output_groups : list[str]
        Names of groups whose columns will form the **output**/target.
    seq_len, pred_len : int
        Window and horizon.
    scale : bool, default True
        Whether to apply MinMax scaling jointly over *all* columns.
    add_time : bool, default False
        Whether to append date/time features (hour, day-of-week, …).
    scaler : MinMaxScaler | None, default None
        Optional MinMaxScaler to use for scaling.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 groups: dict[str, list[str]],
                 input_groups: list[str],
                 output_groups: list[str],
                 seq_len: int,
                 pred_len: int,
                 scale: bool = True,
                 add_time: bool = False,
                 time_features_cfg: Optional[Dict[str, Any]] = None,
                 scaler: 'MinMaxScaler | None' = None):

        # Validate columns
        all_cols = sum(groups.values(), [])
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing in DataFrame: {missing}")

        self.groups = groups
        self.input_groups = input_groups
        self.output_groups = output_groups
        self.input_cols = sum((groups[g] for g in input_groups), [])
        self.output_cols = sum((groups[g] for g in output_groups), [])
        self.time_feature_cols: list[str] = []

        # Reorder DF so that input/output order is preserved for later plotting
        needed_cols = list(dict.fromkeys(self.input_cols + self.output_cols))
        df = df[needed_cols].copy()

        if add_time:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Time features requested but index is not DateTime")
            cols_before = list(df.columns)
            tf_cfg = dict(time_features_cfg or {})
            tf_cfg.pop("enabled", None)
            df = add_time_features(df, **tf_cfg)
            self.time_feature_cols = [c for c in df.columns if c not in cols_before]
            self.input_cols = list(dict.fromkeys(self.input_cols + self.time_feature_cols))

        self.seq_len = seq_len
        self.pred_len = pred_len

        # scaling
        if scale:
            if scaler is not None:
                # Use provided scaler
                self.scaler = scaler
                df.iloc[:, :] = self.scaler.transform(df.values)
            else:
                self.scaler = MinMaxScaler().fit(df.values)
                df.iloc[:, :] = self.scaler.transform(df.values)
        else:
            self.scaler = None

        self.values = df.values.astype(np.float32)
        self.in_idx = [df.columns.get_loc(c) for c in self.input_cols]
        self.out_idx = [df.columns.get_loc(c) for c in self.output_cols]

        # Sanity check – any NaNs at this stage will break MSE and yield NaN
        if np.isnan(self.values).any():
            raise ValueError("NaNs remain in GroupedTimeSeriesDataset after preprocessing. Please inspect the raw data and preprocessing pipeline.")

    def __len__(self):
        return len(self.values) - (self.seq_len + self.pred_len)

    def __getitem__(self, idx: int):
        window = self.values[idx : idx + self.seq_len]
        horizon = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        x = window[:, self.in_idx]
        y = horizon[:, self.out_idx]
        return torch.tensor(x), torch.tensor(y)
    
    def get_warmup_window(self, start_idx: int, warmup_len: int) -> Dict[str, np.ndarray]:
        """Get warmup window for world model initialization.
        
        Parameters
        ----------
        start_idx : int
            Starting index (must be >= warmup_len).
        warmup_len : int
            Length of warmup sequence.
        
        Returns
        -------
        dict
            Dictionary containing:
            - "controls": (warmup_len, control_dim)
            - "exogenous": (warmup_len, exo_dim)
            - "outputs": (warmup_len, output_dim)
            - "inputs": (warmup_len, input_dim) - concatenated inputs
        """
        if start_idx < warmup_len:
            raise ValueError(f"start_idx ({start_idx}) must be >= warmup_len ({warmup_len})")
        
        warmup = self.values[start_idx - warmup_len : start_idx]
        
        return {
            "inputs": warmup[:, self.in_idx],
            "outputs": warmup[:, self.out_idx],
        }
    
    def get_rollout_slice(
        self,
        start_idx: int,
        horizon: int,
    ) -> Dict[str, np.ndarray]:
        """Get data slice for rollout (controls, exogenous, targets).
        
        Parameters
        ----------
        start_idx : int
            Starting index for the rollout.
        horizon : int
            Number of steps to roll out.
        
        Returns
        -------
        dict
            Dictionary containing:
            - "controls": (horizon, control_dim) - if controls in input
            - "exogenous": (horizon, exo_dim) - if exogenous in input
            - "targets": (horizon, output_dim)
            - "inputs": (horizon, input_dim) - full inputs for convenience
        """
        if start_idx + horizon > len(self.values):
            raise ValueError(
                f"Rollout extends beyond dataset: start={start_idx}, "
                f"horizon={horizon}, length={len(self.values)}"
            )
        
        rollout = self.values[start_idx : start_idx + horizon]
        
        return {
            "inputs": rollout[:, self.in_idx],
            "targets": rollout[:, self.out_idx],
        } 
