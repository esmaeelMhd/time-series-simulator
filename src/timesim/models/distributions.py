"""Distribution helpers for decoder heads."""

from __future__ import annotations

import torch


def diagonal_normal(loc: torch.Tensor, scale: torch.Tensor) -> torch.distributions.Normal:
    return torch.distributions.Normal(loc=loc, scale=scale)


def diagonal_independent_normal(
    loc: torch.Tensor, scale: torch.Tensor
) -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Normal(loc=loc, scale=scale),
        1,
    )


__all__ = ["diagonal_normal", "diagonal_independent_normal"]
