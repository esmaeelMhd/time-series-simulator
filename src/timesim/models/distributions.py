"""Distribution helpers for decoder heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def diagonal_normal(loc: torch.Tensor, scale: torch.Tensor) -> torch.distributions.Normal:
    return torch.distributions.Normal(loc=loc, scale=scale)


def diagonal_independent_normal(
    loc: torch.Tensor, scale: torch.Tensor
) -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Normal(loc=loc, scale=scale),
        1,
    )


class FastCategoricalHead(nn.Module):
    """Fast grouped categorical head that returns raw logits."""

    def __init__(
        self,
        dim_input: int,
        num_groups: int = 32,
        num_classes: int = 32,
        hidden: int = 256,
        layers: int = 2,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_classes = int(num_classes)

        dims = [int(dim_input)] + [int(hidden)] * max(0, int(layers) - 1)
        mlp_layers = []
        for i in range(len(dims) - 1):
            mlp_layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ELU()])
        self.shared = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.logits_head = nn.Linear(dims[-1], self.num_groups * self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        logits = self.logits_head(h)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        return logits.view(*logits.shape[:-1], self.num_groups, self.num_classes)


def fast_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Straight-through categorical sample using optimized gumbel-softmax kernel."""
    z = F.gumbel_softmax(logits, tau=float(temperature), hard=True, dim=-1)
    return z.reshape(*z.shape[:-2], -1)


__all__ = [
    "diagonal_normal",
    "diagonal_independent_normal",
    "FastCategoricalHead",
    "fast_sample",
]
