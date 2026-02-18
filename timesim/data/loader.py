from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from .dataset import TimeSeriesDataset
import pandas as pd
from .validation import validate_time_series_dataframe


def generate_sine_dataset(length: int = 1000,
                           n_features: int = 1,
                           freq: float = 0.01,
                           noise: float = 0.1) -> np.ndarray:
    """Generate a simple sine-wave dataset for quick experiments."""
    x = np.arange(length)
    data = np.sin(2 * np.pi * freq * x)  # base sine
    data += noise * np.random.randn(length)  # add noise
    data = data.reshape(-1, 1)  # (time, feature)
    if n_features > 1:
        data = np.hstack([data for _ in range(n_features)])
    return data.astype(np.float32)


def build_dataloaders(series: np.ndarray,
                      seq_len: int,
                      pred_len: int,
                      batch_size: int = 32,
                      train_split: float = 0.8,
                      device: torch.device | str = "cpu") -> Tuple[DataLoader, DataLoader]:
    """Split a univariate/multivariate series into sliding windows and wrap DataLoaders."""
    device = torch.device(device)

    n_total = len(series)
    n_train = int(n_total * train_split)

    train_ds = TimeSeriesDataset(series[:n_train], seq_len, pred_len, scale=False)
    val_ds = TimeSeriesDataset(series[n_train - seq_len:], seq_len, pred_len, scale=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


def load_csv_dataset(path: str | Path,
                     index_col: str = "date",
                     parse_dates: bool = True,
                     slice_cfg: Dict | None = None,
                     engine: str = "pandas",
                     validation_cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Read a CSV file and return a time-indexed DataFrame."""
    engine_name = str(engine or "pandas").lower()
    if engine_name == "polars":
        try:
            import polars as pl  # type: ignore
        except Exception as exc:
            raise ImportError("CSV engine 'polars' requested but polars is not installed.") from exc
        pl_df = pl.read_csv(path)
        if parse_dates:
            pl_df = pl_df.with_columns(pl.col(index_col).str.to_datetime(strict=False))
        if index_col not in pl_df.columns:
            raise KeyError(f"Index column '{index_col}' not found in CSV")
        df = pl_df.to_pandas()
    else:
        df = pd.read_csv(path)

    if parse_dates:
        df[index_col] = pd.to_datetime(df[index_col])
    df = df.set_index(index_col)
    df = df.sort_index()

    # ------------------------------------------------------------------
    # Handle missing values early to avoid NaNs propagating into training
    # and causing NaN losses.  We first forward-fill, then backward-fill as
    # a fallback.  Finally, if any column is still entirely NaN we drop it,
    # raising a warning for visibility.
    # ------------------------------------------------------------------
    if df.isna().any().any():
        df = df.fillna(method="ffill").fillna(method="bfill")

        # After the double fill there can still be columns that were all-NaN
        # (e.g. sensors offline for whole period).  Drop them as they carry
        # no information and would break scaling → loss = NaN.
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            import warnings
            warnings.warn(f"Dropping constant NaN columns from dataset: {all_nan_cols}")
            df = df.drop(columns=all_nan_cols)

    # optional slicing for quick tests
    if slice_cfg:
        if "first_n" in slice_cfg:
            df = df.iloc[: slice_cfg["first_n"]]
        elif "start" in slice_cfg:
            start = slice_cfg["start"]
            end = slice_cfg.get("end")
            df = df.loc[start:end]

    vcfg = validation_cfg or {}
    if bool(vcfg.get("enabled", False)):
        df = validate_time_series_dataframe(
            df,
            required_columns=vcfg.get("required_columns"),
            strict=bool(vcfg.get("strict", False)),
            require_datetime_index=bool(vcfg.get("require_datetime_index", True)),
        )
    return df


def build_grouped_dataloaders(df: pd.DataFrame,
                              groups: Dict[str, List[str]],
                              input_groups: List[str],
                              output_groups: List[str],
                              seq_len: int,
                              pred_len: int,
                              batch_size: int = 32,
                              train_split: float = 0.8,
                              device: str | torch.device = "cpu",
                              add_time: bool = False,
                              time_features_cfg: Optional[Dict[str, Any]] = None,
                              existing_scaler = None,
                              require_full_role_mapping: bool = True) -> Tuple[DataLoader, DataLoader, object]:
    from .dataset import GroupedTimeSeriesDataset
    n_total = len(df)
    n_train = int(n_total * train_split)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train - seq_len:]

    train_ds = GroupedTimeSeriesDataset(
        train_df, groups, input_groups, output_groups, seq_len, pred_len,
        add_time=add_time, time_features_cfg=time_features_cfg, scaler=existing_scaler,
        require_full_role_mapping=require_full_role_mapping,
    )
    val_ds = GroupedTimeSeriesDataset(
        val_df, groups, input_groups, output_groups, seq_len, pred_len,
        add_time=add_time, time_features_cfg=time_features_cfg, scaler=train_ds.scaler,
        require_full_role_mapping=require_full_role_mapping,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, train_ds.scaler 
