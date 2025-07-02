from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import GroupedTimeSeriesDataset


class SEPPWindowDataset(Dataset):
    """Return windows of length seq_len+H_max starting every *stride* samples.

    It reuses the scaling logic of GroupedTimeSeriesDataset so the caller gets
    tensors that can be fed directly to the model.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 groups: Dict[str, List[str]],
                 input_groups: List[str],
                 output_groups: List[str],
                 seq_len: int,
                 h_max: int,
                 stride: int = 1,
                 scaler: 'MinMaxScaler | None' = None):
        self.seq_len = seq_len
        self.h_max = h_max
        self.stride = stride
        self.df = df
        self.groups = groups
        self.input_groups = input_groups
        self.output_groups = output_groups
        self.dataset = GroupedTimeSeriesDataset(df,
                                                groups,
                                                input_groups,
                                                output_groups,
                                                seq_len=seq_len,
                                                pred_len=h_max,
                                                scaler=scaler)
        self.in_idx = self.dataset.in_idx
        self.out_idx = self.dataset.out_idx

    def __len__(self):
        return (len(self.df) - (self.seq_len + self.h_max)) // self.stride

    def __getitem__(self, idx: int):
        start = idx * self.stride
        window = self.dataset.values[start : start + self.seq_len + self.h_max]
        return torch.tensor(window) 