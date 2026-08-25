"""Small utility helpers."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def configure_torch_defaults() -> None:
    """Set recommended torch defaults for performance and logging.

    Call once at the top of any entry-point script (train, eval, simulate, etc.).
    """
    # TF32: use Tensor Cores for float32 matmul (~2-3x faster, 10-bit mantissa)
    torch.set_float32_matmul_precision("high")
    # Silence noisy Inductor warnings (e.g. "Not enough SMs", TF32 hint)
    for _name in ("torch._inductor.utils", "torch._inductor.compile_fx"):
        logging.getLogger(_name).setLevel(logging.ERROR)


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
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        d = str(device).strip().lower()
        if d not in {"", "auto"}:
            return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["configure_torch_defaults", "seed_everything", "resolve_device"]
