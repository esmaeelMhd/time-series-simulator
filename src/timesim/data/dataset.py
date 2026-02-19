from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .stamps import add_time_features
from .schema import VariableRole, VariableSchema
from .validation import validate_variable_groups
from .preprocessing import fit_normalization_stats
from ..utils.scaler import MinMaxScaler, NormalizationStats


class TimeSeriesDataset(Dataset):
    """Windowed dataset from raw numpy array or DataFrame."""

    def __init__(
        self,
        series: pd.DataFrame | np.ndarray,
        seq_len: int,
        pred_len: int,
        scale: bool = True,
        add_time: bool = False,
        time_features_cfg: Optional[Dict[str, Any]] = None,
        stride: int = 1,
    ):
        if isinstance(series, np.ndarray):
            series = pd.DataFrame(series)
        self.series = series.copy()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = max(1, int(stride))

        if add_time:
            if not isinstance(self.series.index, pd.DatetimeIndex):
                raise ValueError("Time features requested but index is not DateTime")
            tf_cfg = dict(time_features_cfg or {})
            tf_cfg.pop("enabled", None)
            self.series = add_time_features(self.series, **tf_cfg)

        self.scaler: MinMaxScaler | NormalizationStats | None = None
        if scale:
            self.scaler = MinMaxScaler().fit(self.series.values)
            scaled = self.scaler.transform(self.series.values).astype(np.float32, copy=False)
            self.series = pd.DataFrame(
                scaled,
                index=self.series.index,
                columns=self.series.columns,
            )

        self.values = self.series.values.astype(np.float32)
        if np.isnan(self.values).any():
            raise ValueError("NaNs remain in TimeSeriesDataset after preprocessing.")

    def __len__(self):
        base = len(self.values) - (self.seq_len + self.pred_len)
        if base <= 0:
            return 0
        return base // self.stride

    def __getitem__(self, idx: int):
        start = int(idx) * self.stride
        x = self.values[start : start + self.seq_len]
        y = self.values[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)


class GroupedTimeSeriesDataset(Dataset):
    """Legacy grouped dataset returning (x, y) tensors for trainer compatibility."""

    def __init__(
        self,
        df: pd.DataFrame,
        groups: dict[str, list[str]],
        input_groups: list[str],
        output_groups: list[str],
        seq_len: int,
        pred_len: int,
        scale: bool = True,
        add_time: bool = False,
        time_features_cfg: Optional[Dict[str, Any]] = None,
        scaler: MinMaxScaler | NormalizationStats | None = None,
        require_full_role_mapping: bool = True,
        stride: int = 1,
        use_symlog: bool = False,
        symlog_columns: Optional[list[str]] = None,
    ):
        validate_variable_groups(groups)
        self.variable_schema = VariableSchema.from_groups(groups)
        self.variable_schema.validate_columns(
            list(df.columns),
            require_exact_match=bool(require_full_role_mapping),
        )
        self.groups = self.variable_schema.to_groups()

        self.input_groups = input_groups
        self.output_groups = output_groups
        self.input_cols = self.variable_schema.columns_for_group_names(input_groups)
        self.output_cols = self.variable_schema.columns_for_group_names(output_groups)
        self.time_feature_cols: list[str] = []
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = max(1, int(stride))

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

        if scale:
            if scaler is None:
                scaler = fit_normalization_stats(
                    df.values,
                    feature_names=list(df.columns),
                    use_symlog=bool(use_symlog),
                    symlog_columns=symlog_columns,
                )
            self.scaler = scaler
            scaled = self.scaler.transform(df.values).astype(np.float32, copy=False)
            df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        else:
            self.scaler = None

        self.feature_cols = list(df.columns)
        self.values = df.values.astype(np.float32, copy=False)
        self.in_idx = [df.columns.get_loc(c) for c in self.input_cols]
        self.out_idx = [df.columns.get_loc(c) for c in self.output_cols]

        control_cols = set(self.variable_schema.columns_for_role(VariableRole.CONTROL))
        output_cols_set = set(self.output_cols)
        self.control_positions = [i for i, col in enumerate(self.input_cols) if col in control_cols]
        self.known_exo_positions = [
            i for i, col in enumerate(self.input_cols) if (col not in control_cols and col not in output_cols_set)
        ]
        if np.isnan(self.values).any():
            raise ValueError("NaNs remain in GroupedTimeSeriesDataset after preprocessing.")

    def __len__(self):
        base = len(self.values) - (self.seq_len + self.pred_len)
        if base <= 0:
            return 0
        return base // self.stride

    def __getitem__(self, idx: int):
        start = int(idx) * self.stride
        window = self.values[start : start + self.seq_len]
        horizon = self.values[start + self.seq_len : start + self.seq_len + self.pred_len]
        x = window[:, self.in_idx]
        y = horizon[:, self.out_idx]
        return torch.tensor(x), torch.tensor(y)

    def get_warmup_window(self, start_idx: int, warmup_len: int) -> Dict[str, np.ndarray]:
        if start_idx < warmup_len:
            raise ValueError(f"start_idx ({start_idx}) must be >= warmup_len ({warmup_len})")
        warmup = self.values[start_idx - warmup_len : start_idx]
        return {"inputs": warmup[:, self.in_idx], "outputs": warmup[:, self.out_idx]}

    def get_rollout_slice(self, start_idx: int, horizon: int) -> Dict[str, np.ndarray]:
        if start_idx + horizon > len(self.values):
            raise ValueError(
                f"Rollout extends beyond dataset: start={start_idx}, horizon={horizon}, length={len(self.values)}"
            )
        rollout = self.values[start_idx : start_idx + horizon]
        return {"inputs": rollout[:, self.in_idx], "targets": rollout[:, self.out_idx]}


class SlidingWindowRoleDataset(Dataset):
    """Role-based sliding-window dataset returning dicts per sample.

    Returns:
      {
        "control": (T, dim_c),
        "exogenous": (T, dim_x),
        "objective": (T, dim_y),
        "target_objective": (pred_len, dim_y),  # when pred_len > 0
      }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        groups: dict[str, list[str]],
        seq_len: int,
        pred_len: int = 0,
        stride: int = 1,
        normalization_stats: Optional[NormalizationStats] = None,
        fit_stats: bool = False,
        use_symlog: bool = False,
        symlog_columns: Optional[list[str]] = None,
        require_full_role_mapping: bool = True,
    ):
        validate_variable_groups(groups)
        self.variable_schema = VariableSchema.from_groups(groups)
        self.variable_schema.validate_columns(
            list(df.columns),
            require_exact_match=bool(require_full_role_mapping),
        )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = max(1, int(stride))

        ordered_cols = self.variable_schema.ordered_columns
        frame = df[ordered_cols].copy()
        if np.isnan(frame.values).any():
            nan_cols = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"NaNs detected in sliding-window dataset columns: {nan_cols}")

        if normalization_stats is None and fit_stats:
            normalization_stats = fit_normalization_stats(
                frame.values,
                feature_names=ordered_cols,
                use_symlog=bool(use_symlog),
                symlog_columns=symlog_columns,
            )
        self.normalization_stats = normalization_stats
        if self.normalization_stats is not None:
            vals = self.normalization_stats.transform(frame.values)
            frame = pd.DataFrame(vals, index=frame.index, columns=frame.columns)

        self.columns = ordered_cols
        self.values = frame.values.astype(np.float32, copy=False)
        self.control_idx = [self.columns.index(c) for c in self.variable_schema.columns_for_role(VariableRole.CONTROL)]
        self.exo_idx = [self.columns.index(c) for c in self.variable_schema.columns_for_role(VariableRole.EXOGENOUS)]
        self.obj_idx = [self.columns.index(c) for c in self.variable_schema.columns_for_role(VariableRole.OBJECTIVE)]

    def __len__(self) -> int:
        max_start = len(self.values) - (self.seq_len + self.pred_len)
        if max_start < 0:
            return 0
        return (max_start // self.stride) + 1

    def __getitem__(self, idx: int):
        start = int(idx) * self.stride
        w = self.values[start : start + self.seq_len]
        sample = {
            "control": torch.tensor(w[:, self.control_idx], dtype=torch.float32),
            "exogenous": torch.tensor(w[:, self.exo_idx], dtype=torch.float32),
            "objective": torch.tensor(w[:, self.obj_idx], dtype=torch.float32),
        }
        if self.pred_len > 0:
            t = self.values[start + self.seq_len : start + self.seq_len + self.pred_len]
            sample["target_objective"] = torch.tensor(t[:, self.obj_idx], dtype=torch.float32)
        return sample
