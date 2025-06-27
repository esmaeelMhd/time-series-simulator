from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.early_stop import EarlyStopping
from ..utils.dilate import dilate_loss


class Trainer:
    """Lightweight training loop used for both first training and retraining."""

    def __init__(self,
                 model: nn.Module,
                 loss: str = "mse",  # mse or dilate
                 optimizer: torch.optim.Optimizer | None = None,
                 device: torch.device | str = "cpu",
                 early_stopping: bool = False,
                 patience: int = 5):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "dilate":
            # we will compute dilate in _step
            self.loss_fn = None
        else:
            raise ValueError("loss must be mse or dilate")
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_type = loss
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        if self.loss_type == "mse":
            loss = self.loss_fn(pred, y)
        else:
            loss, _, _ = dilate_loss(y, pred, device=self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _validate(self, loader: DataLoader):
        self.model.eval()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            if self.loss_type == "mse":
                total += self.loss_fn(pred, y).item() * len(x)
            n += len(x)
        self.model.train()
        return total / max(n, 1)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader | None = None,
            epochs: int = 10,
            verbose: bool = True):
        train_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch in train_loader:
                batch_loss = self._step(batch)
                epoch_losses.append(batch_loss)
            train_loss = sum(epoch_losses) / len(epoch_losses)
            val_loss = self._validate(val_loader) if val_loader else None
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if verbose and (epoch == 1 or epoch == epochs or epoch % 5 == 0):
                msg = f"[Epoch {epoch}/{epochs}] train={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val={val_loss:.4f}"
                print(msg)
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    if verbose:
                        print("Early stopping triggered.")
                    break
        return train_losses, val_losses

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 