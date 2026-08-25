from __future__ import annotations

import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        # Ignore invalid validations; do not mutate state.
        if val_loss is None or not np.isfinite(val_loss):
            return

        if self.best_loss is None:
            self.best_loss = float(val_loss)
            self.counter = 0
            return

        # Significant improvement: update best and reset patience.
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.counter = 0
            return

        # Small improvement (< min_delta): still update best, but don't consume patience.
        if val_loss < self.best_loss:
            self.best_loss = float(val_loss)
            return

        # No improvement.
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
