"""Core recurrent state-space model (RSSM) cell primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    """Compact recurrent RSSM state."""

    h: torch.Tensor
    z: torch.Tensor


@dataclass
class RSSMOutput:
    """Single-step RSSM output bundle."""

    state: RSSMState
    h: torch.Tensor
    prior: torch.distributions.Distribution
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    prior_logvar: torch.Tensor
    posterior: Optional[torch.distributions.Distribution] = None
    posterior_mean: Optional[torch.Tensor] = None
    posterior_std: Optional[torch.Tensor] = None
    posterior_logvar: Optional[torch.Tensor] = None


class RSSMCell(nn.Module):
    """RSSM cell with deterministic GRU path and stochastic latent path.

    Standard transition:
      h_t = GRU( MLP([z_{t-1}, c_embed_t, x_embed_t]), h_{t-1} )
    """

    def __init__(
        self,
        dim_h: int,
        dim_z: int,
        dim_control_embed: int,
        dim_exogenous_embed: int,
        dim_obs_embed: int,
        *,
        transition_hidden_dim: Optional[int] = None,
        min_std: float = 0.01,
        use_dual_path: bool = True,
        leak_objective_to_transition: bool = False,
    ):
        super().__init__()
        self.dim_h = int(dim_h)
        self.dim_z = int(dim_z)
        self.dim_control_embed = int(dim_control_embed)
        self.dim_exogenous_embed = int(dim_exogenous_embed)
        self.dim_obs_embed = int(dim_obs_embed)
        self.use_dual_path = bool(use_dual_path)
        self.leak_objective_to_transition = bool(leak_objective_to_transition)
        self.min_std = float(max(0.01, float(min_std)))

        trans_hidden = int(transition_hidden_dim or self.dim_h)
        self.transition_input_dim = self.dim_z + self.dim_control_embed + self.dim_exogenous_embed
        if self.leak_objective_to_transition:
            self.transition_input_dim += self.dim_obs_embed

        self.transition_mlp = nn.Sequential(
            nn.Linear(self.transition_input_dim, trans_hidden),
            nn.ELU(),
            nn.Linear(trans_hidden, self.dim_h),
            nn.ELU(),
        )
        self.transition_gru = nn.GRUCell(input_size=self.dim_h, hidden_size=self.dim_h)

        self.prior_head = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h),
            nn.ELU(),
            nn.Linear(self.dim_h, 2 * self.dim_z),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(self.dim_h + self.dim_obs_embed, self.dim_h),
            nn.ELU(),
            nn.Linear(self.dim_h, 2 * self.dim_z),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RSSMState:
        return RSSMState(
            h=torch.zeros(int(batch_size), self.dim_h, device=device, dtype=dtype),
            z=torch.zeros(int(batch_size), self.dim_z, device=device, dtype=dtype),
        )

    def _diag_gaussian(
        self, params: torch.Tensor
    ) -> tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std),
            1,
        )
        logvar = 2.0 * torch.log(std)
        return dist, mean, std, logvar

    def _transition_input(
        self,
        prev_state: RSSMState,
        control_embed: torch.Tensor,
        exogenous_embed: torch.Tensor,
        observation_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [prev_state.z, control_embed, exogenous_embed]
        if self.leak_objective_to_transition:
            if observation_embed is None:
                observation_embed = torch.zeros(
                    control_embed.shape[0],
                    self.dim_obs_embed,
                    device=control_embed.device,
                    dtype=control_embed.dtype,
                )
            pieces.append(observation_embed)
        return torch.cat(pieces, dim=-1)

    def transition(
        self,
        prev_state: RSSMState,
        control_embed: torch.Tensor,
        exogenous_embed: torch.Tensor,
        observation_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trans_raw = self._transition_input(
            prev_state=prev_state,
            control_embed=control_embed,
            exogenous_embed=exogenous_embed,
            observation_embed=observation_embed,
        )
        pre_gru = self.transition_mlp(trans_raw)
        prev_h = prev_state.h if self.use_dual_path else torch.zeros_like(prev_state.h)
        return self.transition_gru(pre_gru, prev_h)

    def prior_from_h(
        self, h_t: torch.Tensor
    ) -> tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._diag_gaussian(self.prior_head(h_t))

    def posterior_from_h(
        self, h_t: torch.Tensor, observation_embed: torch.Tensor
    ) -> tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        post_in = torch.cat([h_t, observation_embed], dim=-1)
        return self._diag_gaussian(self.posterior_head(post_in))

    def observe(
        self,
        prev_state: RSSMState,
        control_embed: torch.Tensor,
        exogenous_embed: torch.Tensor,
        observation_embed: torch.Tensor,
        *,
        sample: bool = True,
    ) -> RSSMOutput:
        h_t = self.transition(
            prev_state=prev_state,
            control_embed=control_embed,
            exogenous_embed=exogenous_embed,
            observation_embed=observation_embed if self.leak_objective_to_transition else None,
        )
        prior_dist, prior_mean, prior_std, prior_logvar = self.prior_from_h(h_t)
        post_dist, post_mean, post_std, post_logvar = self.posterior_from_h(h_t, observation_embed)
        z_t = post_dist.rsample() if sample else post_mean
        state = RSSMState(h=h_t, z=z_t)
        return RSSMOutput(
            state=state,
            h=h_t,
            prior=prior_dist,
            prior_mean=prior_mean,
            prior_std=prior_std,
            prior_logvar=prior_logvar,
            posterior=post_dist,
            posterior_mean=post_mean,
            posterior_std=post_std,
            posterior_logvar=post_logvar,
        )

    def imagine(
        self,
        prev_state: RSSMState,
        control_embed: torch.Tensor,
        exogenous_embed: torch.Tensor,
        *,
        sample: bool = True,
    ) -> RSSMOutput:
        h_t = self.transition(
            prev_state=prev_state,
            control_embed=control_embed,
            exogenous_embed=exogenous_embed,
            observation_embed=None,
        )
        prior_dist, prior_mean, prior_std, prior_logvar = self.prior_from_h(h_t)
        z_t = prior_dist.rsample() if sample else prior_mean
        state = RSSMState(h=h_t, z=z_t)
        return RSSMOutput(
            state=state,
            h=h_t,
            prior=prior_dist,
            prior_mean=prior_mean,
            prior_std=prior_std,
            prior_logvar=prior_logvar,
        )


__all__ = ["RSSMState", "RSSMOutput", "RSSMCell"]
