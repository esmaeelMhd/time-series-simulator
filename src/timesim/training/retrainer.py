"""Checkpoint retraining / fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import Trainer


class EpisodeSampler:
    """Slide a fixed window over a time-series tensor yielding (src, tgt) episodes."""

    def __init__(self, series: torch.Tensor, seq_len: int, pred_len: int):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.series) - (self.seq_len + self.pred_len) - 1
        idx = torch.randint(0, max_start, (1,)).item()
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x.unsqueeze(0), y.unsqueeze(0)


class Retrainer:
    def __init__(
        self,
        model_cls: type[nn.Module],
        checkpoint: str | Path,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model_cls.to(self.device) if isinstance(model_cls, nn.Module) else model_cls()
        self.model.to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 3,
        lr: float = 1e-4,
    ):
        trainer = Trainer(
            self.model,
            loss="mse",
            optimizer=torch.optim.Adam(self.model.parameters(), lr=lr),
            device=self.device,
        )
        return trainer.fit(train_loader, val_loader, epochs=epochs)
