from __future__ import annotations

from typing import Callable, Tuple
from pathlib import Path

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
                 patience: int = 5,
                 run_dir: str | None = None,
                 writer: "SummaryWriter" | None = None):
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
        # Logging / outputs
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Lazy import – avoids dependency if not requested
            if writer is not None:
                self.writer = writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter  # type: ignore
                    self.writer = SummaryWriter(log_dir=self.run_dir)
                except ModuleNotFoundError:
                    self.writer = None
            # Prepare csv file
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,train_loss,val_loss\n")
        else:
            self.writer = writer  # could be None
            self.metrics_path = None

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
            # After computing train_loss and val_loss
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
            if self.metrics_path is not None:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{train_loss},{val_loss if val_loss is not None else ''}\n")
        return train_losses, val_losses

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 