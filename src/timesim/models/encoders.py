"""Typed encoder modules used by RSSM-style models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TypedEncoder(nn.Module):
    """Small MLP encoder for a specific variable type.

    Args:
      input_dim: role-specific input width. If ``None`` a lazy first layer is used.
      hidden_dim: hidden width of the MLP.
      embed_dim: role-specific embedding width.
    """

    def __init__(
        self,
        input_dim: Optional[int],
        hidden_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.input_dim = None if input_dim is None else int(input_dim)

        if self.input_dim is not None and self.input_dim == 0:
            self.net = None
        elif self.input_dim is None:
            self.net = nn.Sequential(
                nn.LazyLinear(int(hidden_dim)),
                nn.ELU(),
                nn.Linear(int(hidden_dim), self.embed_dim),
                nn.ELU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, int(hidden_dim)),
                nn.ELU(),
                nn.Linear(int(hidden_dim), self.embed_dim),
                nn.ELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0 or self.net is None:
            return torch.zeros(x.shape[0], self.embed_dim, dtype=x.dtype, device=x.device)
        if self.input_dim is not None and x.shape[-1] != self.input_dim:
            raise ValueError(f"Encoder input mismatch: expected {self.input_dim}, got {x.shape[-1]}")
        return self.net(x)


class ControlEncoder(TypedEncoder):
    """Control-role encoder MLP: dim_c -> hidden -> dim_control_embed."""

    def __init__(self, input_dim: Optional[int], hidden_dim: int, embed_dim: int):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)


class ExogenousEncoder(TypedEncoder):
    """Exogenous-role encoder MLP: dim_x -> hidden -> dim_exogenous_embed."""

    def __init__(self, input_dim: Optional[int], hidden_dim: int, embed_dim: int):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)


class ObservationEncoder(TypedEncoder):
    """Objective/observation-role encoder MLP: dim_y -> hidden -> dim_obs_embed."""

    def __init__(self, input_dim: Optional[int], hidden_dim: int, embed_dim: int):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)


def assert_no_shared_encoder_params(
    control_encoder: nn.Module,
    exogenous_encoder: nn.Module,
    observation_encoder: nn.Module,
) -> None:
    """Raise if any encoder modules share parameter objects."""
    encoders = [control_encoder, exogenous_encoder, observation_encoder]
    if len({id(enc) for enc in encoders}) != 3:
        raise RuntimeError("Control, exogenous, and observation encoders must be separate modules.")

    ctrl_params = set(id(p) for p in control_encoder.parameters())
    exog_params = set(id(p) for p in exogenous_encoder.parameters())
    obs_params = set(id(p) for p in observation_encoder.parameters())
    if (ctrl_params & exog_params) or (ctrl_params & obs_params) or (exog_params & obs_params):
        raise RuntimeError("Encoder parameter sharing detected; variable-type encoders must not share weights.")


__all__ = [
    "TypedEncoder",
    "ControlEncoder",
    "ExogenousEncoder",
    "ObservationEncoder",
    "assert_no_shared_encoder_params",
]
