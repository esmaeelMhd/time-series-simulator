"""Small utility helpers."""

from __future__ import annotations

import random
import os
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        d = str(device).strip().lower()
        if d not in {"", "auto"}:
            return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["seed_everything", "resolve_device"]
