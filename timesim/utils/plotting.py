from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def save_loss_plot(train_losses: List[float],
                   val_losses: List[float] | None,
                   out_path: str | Path,
                   title: str = "Loss curve"):
    """Save a PNG figure of train/val loss vs epoch."""
    out_path = Path(out_path)
    plt.figure(figsize=(5, 3))
    plt.plot(train_losses, label="train")
    if val_losses is not None and any(v is not None for v in val_losses):
        plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close() 