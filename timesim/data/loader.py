import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Tuple
from .dataset import TimeSeriesDataset


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