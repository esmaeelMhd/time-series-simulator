from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

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
                 add_time: bool = False):
        if isinstance(series, np.ndarray):
            series = pd.DataFrame(series)
        self.series = series.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len

        if add_time:
            if not isinstance(self.series.index, pd.DatetimeIndex):
                raise ValueError("Time features requested but index is not DateTime")
            self.series = add_time_features(self.series)

        self.scaler = None
        if scale:
            self.scaler = MinMaxScaler().fit(self.series.values)
            scaled = self.scaler.transform(self.series.values)
            self.series.iloc[:, :] = scaled

        self.values = self.series.values.astype(np.float32)

    def __len__(self):
        return len(self.values) - (self.seq_len + self.pred_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.values[idx : idx + self.seq_len]
        y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y) 