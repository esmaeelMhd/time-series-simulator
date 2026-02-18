"""Lightweight LightningModule wrapper for world models.

This module provides a low-boilerplate training entry for users who prefer
PyTorch Lightning orchestration (DDP/FSDP/callback ecosystem).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "timesim.lightning requires pytorch-lightning. "
        "Install with: pip install pytorch-lightning"
    ) from exc


class WorldModelLightningModule(pl.LightningModule):  # type: ignore[misc]
    """Generic one-step wrapper for supervised world model training.

    Notes
    -----
    - This wrapper intentionally keeps a conservative default loss (MSE) so it
      can wrap any model that returns tensor predictions.
    - Advanced RSSM training remains available through the native trainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-6,
        scheduler_warmup_steps: int = 1000,
        scheduler_min_ratio: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler_warmup_steps = int(max(0, scheduler_warmup_steps))
        self.scheduler_min_ratio = float(max(0.0, min(1.0, scheduler_min_ratio)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            if "predictions" in out:
                return out["predictions"]
            if "mean" in out:
                return out["mean"]
        if torch.is_tensor(out):
            return out
        raise TypeError("Wrapped model output is not compatible with Lightning wrapper.")

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        if pred.shape != y.shape:
            # Common case: model predicts full sequence but target is horizon-only.
            pred = pred[:, -y.shape[1]:, : y.shape[-1]]
        loss = F.mse_loss(pred, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if self.scheduler_warmup_steps > 0 and step < self.scheduler_warmup_steps:
                return float(step + 1) / float(self.scheduler_warmup_steps)
            # cosine decay towards min_ratio
            # Lightning provides global step progression.
            # Fallback to stable decay even when max_steps is unknown.
            s = max(0, step - self.scheduler_warmup_steps)
            decay = 0.5 * (1.0 + torch.cos(torch.tensor(min(s, 100000) / 100000.0 * torch.pi)).item())
            return self.scheduler_min_ratio + (1.0 - self.scheduler_min_ratio) * decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
