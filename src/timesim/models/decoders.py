"""Decoder modules for world-model outputs."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """Diagonal Gaussian decoder head producing a distribution object."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        min_std: float = 0.5,
        max_std: float = 2.0,
        constant_std: float | None = None,
    ):
        super().__init__()
        layers = [nn.Linear(int(in_dim), int(hidden_dim)), nn.ELU()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ELU()])
        layers.append(nn.Linear(int(hidden_dim), 2 * int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.out_dim = int(out_dim)
        self.min_std = float(max(0.0, float(min_std)))
        self.max_std = float(max(self.min_std, float(max_std)))
        self.constant_std = None if constant_std is None else float(max(1e-6, float(constant_std)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        min_scale: float | None = None,
        max_scale: float | None = None,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        loc, raw_scale = torch.chunk(raw, 2, dim=-1)
        if self.constant_std is not None:
            scale = torch.full_like(loc, fill_value=self.constant_std)
        else:
            min_std = self.min_std if min_scale is None else float(max(self.min_std, float(min_scale)))
            max_std = self.max_std if max_scale is None else float(min(self.max_std, float(max_scale)))
            max_std = max(min_std, max_std)
            scale = F.softplus(raw_scale) + min_std
            scale = torch.clamp(scale, max=max_std)
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=loc, scale=scale),
            1,
        )
        return dist, loc, scale


class ObjectiveDecoder(GaussianHead):
    """Objective decoder: p(y_t | h_t, z_t)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        min_std: float = 0.5,
        max_std: float = 2.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            min_std=min_std,
            max_std=max_std,
        )


class AuxiliaryDecoder(GaussianHead):
    """Auxiliary exogenous decoder: p(x_t | h_t, z_t)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        min_std: float = 0.5,
        max_std: float = 2.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            min_std=min_std,
            max_std=max_std,
        )


__all__ = [
    "GaussianHead",
    "ObjectiveDecoder",
    "AuxiliaryDecoder",
]
