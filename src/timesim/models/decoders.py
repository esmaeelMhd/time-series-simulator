"""Decoder modules for world-model outputs."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """Diagonal Gaussian decoder head producing a distribution object."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(int(in_dim), int(hidden_dim)), nn.ELU()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ELU()])
        layers.append(nn.Linear(int(hidden_dim), 2 * int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.out_dim = int(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        min_scale: float = 1e-2,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        loc, raw_scale = torch.chunk(raw, 2, dim=-1)
        scale = F.softplus(raw_scale) + float(max(1e-2, min_scale))
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=loc, scale=scale),
            1,
        )
        return dist, loc, scale


class ObjectiveDecoder(GaussianHead):
    """Objective decoder: p(y_t | h_t, z_t)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, num_layers=num_layers)


class AuxiliaryDecoder(GaussianHead):
    """Auxiliary exogenous decoder: p(x_t | h_t, z_t)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, num_layers=num_layers)


__all__ = [
    "GaussianHead",
    "ObjectiveDecoder",
    "AuxiliaryDecoder",
]
