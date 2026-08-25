from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import GroupedTimeSeriesDataset, TimeSeriesDataset
from .validation import validate_time_series_dataframe


def generate_sine_dataset(
    length: int = 1000,
    n_features: int = 1,
    freq: float = 0.01,
    noise: float = 0.1,
) -> np.ndarray:
    x = np.arange(length)
    data = np.sin(2 * np.pi * freq * x)
    data += noise * np.random.randn(length)
    data = data.reshape(-1, 1)
    if n_features > 1:
        data = np.hstack([data for _ in range(n_features)])
    return data.astype(np.float32)


def _seed_worker_factory(seed: Optional[int]):
    def _seed_worker(worker_id: int):
        if seed is None:
            return
        worker_seed = int(seed) + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _seed_worker


def _build_generator(seed: Optional[int]):
    if seed is None:
        return None
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def resolve_split_ratios(
    split_cfg: Optional[Dict[str, Any]] = None,
    *,
    train_split: Optional[float] = None,
    default: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[float, float, float]:
    if split_cfg:
        tr = float(split_cfg.get("train", default[0]))
        va = float(split_cfg.get("val", default[1]))
        te = float(split_cfg.get("test", default[2]))
    elif train_split is not None:
        tr = float(train_split)
        rem = float(max(0.0, 1.0 - tr))
        tail = float(default[1] + default[2])
        if tail <= 0.0:
            va = rem
            te = 0.0
        else:
            va = rem * float(default[1]) / tail
            te = rem * float(default[2]) / tail
    else:
        tr, va, te = default
    s = tr + va + te
    if s <= 0:
        raise ValueError("Split ratios sum to zero.")
    tr, va, te = tr / s, va / s, te / s
    return tr, va, te


def chronological_split_dataframe(
    df: pd.DataFrame,
    *,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 rows to build train/val/test splits.")
    tr, va, te = split_ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    # keep at least one sample in each split when possible
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def build_dataloaders(
    series: np.ndarray,
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
    train_split: float = 0.7,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    _ = torch.device(device)
    n_total = len(series)
    n_train = int(n_total * float(train_split))
    train_ds = TimeSeriesDataset(series[:n_train], seq_len, pred_len, scale=False)
    val_ds = TimeSeriesDataset(series[n_train - seq_len :], seq_len, pred_len, scale=False)

    generator = _build_generator(seed)
    worker_init_fn = _seed_worker_factory(seed) if seed is not None else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(shuffle_train),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


def load_csv_dataset(
    path: str | Path,
    index_col: str = "date",
    parse_dates: bool = True,
    slice_cfg: Dict | None = None,
    engine: str = "pandas",
    validation_cfg: Dict[str, Any] | None = None,
) -> pd.DataFrame:
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

    if slice_cfg:
        if "first_n" in slice_cfg:
            df = df.iloc[: int(slice_cfg["first_n"])]
        elif "start" in slice_cfg:
            start = slice_cfg["start"]
            end = slice_cfg.get("end")
            df = df.loc[start:end]

    vcfg = validation_cfg or {}
    df = validate_time_series_dataframe(
        df,
        required_columns=vcfg.get("required_columns"),
        strict=bool(vcfg.get("strict", False)),
        require_datetime_index=bool(vcfg.get("require_datetime_index", True)),
        require_monotonic_index=bool(vcfg.get("require_monotonic_index", True)),
        require_numeric_dtypes=bool(vcfg.get("require_numeric_dtypes", True)),
        allow_nan=bool(vcfg.get("allow_nan", False)),
    )
    return df


def build_grouped_dataloaders(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    input_groups: List[str],
    output_groups: List[str],
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
    train_split: float = 0.7,
    device: str | torch.device = "cpu",
    add_time: bool = False,
    time_features_cfg: Optional[Dict[str, Any]] = None,
    existing_scaler=None,
    require_full_role_mapping: bool = True,
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    stride: int = 1,
    use_symlog: bool = False,
    symlog_columns: Optional[list[str]] = None,
    split_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    train_loader, val_loader, test_loader, scaler = build_grouped_triplet_dataloaders(
        df=df,
        groups=groups,
        input_groups=input_groups,
        output_groups=output_groups,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        train_split=train_split,
        device=device,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        existing_scaler=existing_scaler,
        require_full_role_mapping=require_full_role_mapping,
        seed=seed,
        shuffle_train=shuffle_train,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        stride=stride,
        use_symlog=use_symlog,
        symlog_columns=symlog_columns,
        split_cfg=split_cfg,
    )
    return train_loader, val_loader, test_loader, scaler


def build_dataloaders_from_config(
    config: Dict[str, Any],
    df: pd.DataFrame,
    seed: Optional[int],
    scaler=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """Build grouped train/val/test dataloaders from a composed runtime config."""
    data_cfg = config.get("data", {}) or {}
    dataset_cfg = config["dataset"]
    model_io_cfg = config["model_io"]

    add_time = bool(data_cfg.get("add_time_features", False))
    time_features_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(time_features_cfg, dict) and "enabled" in time_features_cfg:
        add_time = bool(time_features_cfg.get("enabled")) or add_time

    split_cfg = data_cfg.get("splits", None)
    train_split = float((split_cfg or {}).get("train", 0.7))

    return build_grouped_dataloaders(
        df=df,
        groups=dataset_cfg["variables"],
        input_groups=model_io_cfg["input_groups"],
        output_groups=model_io_cfg["output_groups"],
        seq_len=int(dataset_cfg["seq_len"]),
        pred_len=int(dataset_cfg["pred_len"]),
        batch_size=int(dataset_cfg["batch_size"]),
        train_split=train_split,
        split_cfg=split_cfg,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        existing_scaler=scaler,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
        seed=seed,
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        drop_last=bool(data_cfg.get("drop_last", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        stride=int(data_cfg.get("window_stride", 1)),
        use_symlog=bool((data_cfg.get("symlog", {}) or {}).get("enabled", False)),
        symlog_columns=(data_cfg.get("symlog", {}) or {}).get("columns", None),
    )


def build_grouped_triplet_dataloaders(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    input_groups: List[str],
    output_groups: List[str],
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
    train_split: float = 0.7,
    device: str | torch.device = "cpu",
    add_time: bool = False,
    time_features_cfg: Optional[Dict[str, Any]] = None,
    existing_scaler=None,
    require_full_role_mapping: bool = True,
    seed: Optional[int] = None,
    shuffle_train: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    stride: int = 1,
    use_symlog: bool = False,
    symlog_columns: Optional[list[str]] = None,
    split_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    _ = torch.device(device)
    ratios = resolve_split_ratios(
        split_cfg=split_cfg,
        train_split=train_split,
        default=(0.70, 0.15, 0.15),
    )
    train_df, val_df, test_df = chronological_split_dataframe(df, split_ratios=ratios)

    train_ds = GroupedTimeSeriesDataset(
        train_df,
        groups,
        input_groups,
        output_groups,
        seq_len,
        pred_len,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        scaler=existing_scaler,
        require_full_role_mapping=require_full_role_mapping,
        stride=stride,
        use_symlog=use_symlog,
        symlog_columns=symlog_columns,
    )
    val_ds = GroupedTimeSeriesDataset(
        val_df,
        groups,
        input_groups,
        output_groups,
        seq_len,
        pred_len,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        scaler=train_ds.scaler,
        require_full_role_mapping=require_full_role_mapping,
        stride=stride,
        use_symlog=use_symlog,
        symlog_columns=symlog_columns,
    )
    test_ds = GroupedTimeSeriesDataset(
        test_df,
        groups,
        input_groups,
        output_groups,
        seq_len,
        pred_len,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        scaler=train_ds.scaler,
        require_full_role_mapping=require_full_role_mapping,
        stride=stride,
        use_symlog=use_symlog,
        symlog_columns=symlog_columns,
    )

    generator = _build_generator(seed)
    worker_init_fn = _seed_worker_factory(seed) if seed is not None else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(shuffle_train),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader, train_ds.scaler
