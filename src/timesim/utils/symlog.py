"""Symlog/symexp transforms used in Dreamer-style training."""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


__all__ = ["symlog", "symexp"]
