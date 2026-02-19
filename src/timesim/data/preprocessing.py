"""Data preprocessing helpers."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.scaler import MinMaxScaler, NormalizationStats

__all__ = [
    "NormalizationStats",
    "fit_scaler",
    "fit_normalization_stats",
    "apply_scaler",
    "normalize_array",
    "denormalize_array",
    "select_columns",
]


def fit_scaler(values: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(values)
    return scaler


def fit_normalization_stats(
    values: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    use_symlog: bool = False,
    symlog_columns: Optional[Iterable[str]] = None,
) -> NormalizationStats:
    stats = NormalizationStats(use_symlog=bool(use_symlog))
    stats.fit(values, feature_names=feature_names, symlog_columns=symlog_columns)
    return stats


def apply_scaler(values: np.ndarray, scaler: NormalizationStats) -> np.ndarray:
    return scaler.transform(values)


def normalize_array(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return stats.transform(values)


def denormalize_array(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return stats.inverse_transform(values)


def select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns)
    return df.loc[:, cols].copy()
