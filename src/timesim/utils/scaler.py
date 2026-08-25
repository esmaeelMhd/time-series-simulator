from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


def _symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


@dataclass
class NormalizationStats:
    """Train-fit normalization stats with optional symlog pre-transform."""

    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    feature_names: Optional[list[str]] = None
    use_symlog: bool = False
    symlog_indices: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(
        self,
        data: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        symlog_columns: Optional[Iterable[str]] = None,
    ) -> "NormalizationStats":
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for normalization fit, got shape {arr.shape}")
        if np.isnan(arr).any():
            raise ValueError("NaNs detected while fitting normalization stats.")

        self.feature_names = list(feature_names) if feature_names is not None else None
        if self.use_symlog and self.feature_names is not None and symlog_columns is not None:
            col_set = set(str(c) for c in symlog_columns)
            self.symlog_indices = np.array(
                [i for i, c in enumerate(self.feature_names) if c in col_set],
                dtype=np.int64,
            )
        elif self.use_symlog:
            self.symlog_indices = np.arange(arr.shape[1], dtype=np.int64)
        else:
            self.symlog_indices = np.array([], dtype=np.int64)

        fit_arr = self._apply_symlog(arr)
        self.min = fit_arr.min(axis=0)
        self.max = fit_arr.max(axis=0)
        span = self.max - self.min
        tiny = span < 1e-6
        if np.any(tiny):
            names = self.feature_names or [str(i) for i in range(arr.shape[1])]
            frozen = [names[i] for i, flag in enumerate(tiny) if flag]
            warnings.warn(
                "Near-constant train features detected; using unit scale instead of "
                f"dividing by eps. Columns: {frozen}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.max = np.where(tiny, self.min + 1.0, self.max)
        return self

    def _require_fit(self) -> None:
        if self.min is None or self.max is None:
            raise RuntimeError("NormalizationStats is not fit yet.")

    def _apply_symlog(self, data: np.ndarray) -> np.ndarray:
        if not self.use_symlog:
            return data
        out = data.copy()
        idx = self.symlog_indices if self.symlog_indices is not None else np.arange(out.shape[1], dtype=np.int64)
        if idx.size > 0:
            out[:, idx] = _symlog(out[:, idx])
        return out

    def _apply_symexp(self, data: np.ndarray) -> np.ndarray:
        if not self.use_symlog:
            return data
        out = data.copy()
        idx = self.symlog_indices if self.symlog_indices is not None else np.arange(out.shape[1], dtype=np.int64)
        if idx.size > 0:
            out[:, idx] = _symexp(out[:, idx])
        return out

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fit()
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for normalization transform, got shape {arr.shape}")
        arr = self._apply_symlog(arr)
        return (arr - self.min) / (self.max - self.min + float(self.eps))

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fit()
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for normalization inverse_transform, got shape {arr.shape}")
        out = arr * (self.max - self.min + float(self.eps)) + self.min
        return self._apply_symexp(out)


class MinMaxScaler(NormalizationStats):
    """Backward-compatible alias used across legacy pipeline code."""

    def __init__(self):
        super().__init__(use_symlog=False)
