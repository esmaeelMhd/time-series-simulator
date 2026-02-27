"""Core recurrent state-space model (RSSM) cell primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import FastCategoricalHead, fast_sample


class RSSMGaussianHead(nn.Module):
    """Diagonal Gaussian head for RSSM latent prior/posterior."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        *,
        min_std: float = 0.1,
        max_std: float = 1.5,
        constant_std: float | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), 2 * int(out_dim)),
        )
        self.min_std = float(max(0.0, float(min_std)))
        self.max_std = float(max(self.min_std, float(max_std)))
        self.constant_std = None if constant_std is None else float(max(1e-6, float(constant_std)))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.net(x)
        params = torch.nan_to_num(params, nan=0.0, posinf=1e4, neginf=-1e4)
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        mean = torch.clamp(mean, min=-1e4, max=1e4)

        if self.constant_std is not None:
            std = torch.full_like(mean, fill_value=self.constant_std)
        else:
            raw_std = torch.clamp(raw_std, min=-20.0, max=20.0)
            std = F.softplus(raw_std) + self.min_std
            std = torch.clamp(std, min=self.min_std, max=self.max_std)

        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std),
            1,
        )
        logvar = 2.0 * torch.log(std)
        return dist, mean, std, logvar


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
    prior: Any
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    prior_logvar: torch.Tensor
    posterior: Optional[Any] = None
    posterior_mean: Optional[torch.Tensor] = None
    posterior_std: Optional[torch.Tensor] = None
    posterior_logvar: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None


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
        min_std: float = 0.1,
        max_std: float = 1.5,
        prior_constant_std: float | None = None,
        posterior_constant_std: float | None = None,
        use_dual_path: bool = True,
        leak_objective_to_transition: bool = False,
        use_stochastic_path: bool = True,
        latent_distribution: str = "gaussian",
        stochastic_groups: int = 32,
        stochastic_classes: int = 32,
    ):
        super().__init__()
        self.dim_h = int(dim_h)
        self.latent_distribution = str(latent_distribution).lower().strip()
        self.stochastic_groups = int(stochastic_groups)
        self.stochastic_classes = int(stochastic_classes)
        if self.latent_distribution == "categorical":
            expected = self.stochastic_groups * self.stochastic_classes
            if int(dim_z) != expected:
                raise ValueError(
                    f"For categorical latents, dim_z must equal groups*classes ({expected}), got {dim_z}."
                )
        elif self.latent_distribution != "gaussian":
            raise ValueError(f"Unsupported latent_distribution: {latent_distribution}")
        self.dim_z = int(dim_z)
        self.dim_control_embed = int(dim_control_embed)
        self.dim_exogenous_embed = int(dim_exogenous_embed)
        self.dim_obs_embed = int(dim_obs_embed)
        self.use_dual_path = bool(use_dual_path)
        if bool(leak_objective_to_transition):
            raise ValueError(
                "Objective/observation leakage into transition is not allowed. "
                "Transition must not depend on y/obs embedding."
            )
        self.leak_objective_to_transition = False
        self.use_stochastic_path = bool(use_stochastic_path)
        self.min_std = float(max(0.0, float(min_std)))
        self.max_std = float(max(self.min_std, float(max_std)))
        self.prior_constant_std = None if prior_constant_std is None else float(max(1e-6, float(prior_constant_std)))
        self.posterior_constant_std = (
            None if posterior_constant_std is None else float(max(1e-6, float(posterior_constant_std)))
        )

        trans_hidden = int(transition_hidden_dim or self.dim_h)
        self.transition_input_dim = self.dim_z + self.dim_control_embed + self.dim_exogenous_embed
        self.pre_gru_norm = nn.LayerNorm(self.transition_input_dim)

        self.transition_mlp = nn.Sequential(
            nn.Linear(self.transition_input_dim, trans_hidden),
            nn.ELU(),
            nn.Linear(trans_hidden, self.dim_h),
            nn.ELU(),
        )
        self.transition_gru = nn.GRUCell(input_size=self.dim_h, hidden_size=self.dim_h)

        if self.latent_distribution == "categorical":
            self.prior_head = FastCategoricalHead(
                dim_input=self.dim_h,
                num_groups=self.stochastic_groups,
                num_classes=self.stochastic_classes,
                hidden=self.dim_h,
                layers=2,
            )
            self.posterior_head = FastCategoricalHead(
                dim_input=self.dim_h + self.dim_obs_embed,
                num_groups=self.stochastic_groups,
                num_classes=self.stochastic_classes,
                hidden=self.dim_h,
                layers=2,
            )
        else:
            self.prior_head = RSSMGaussianHead(
                in_dim=self.dim_h,
                out_dim=self.dim_z,
                hidden_dim=self.dim_h,
                min_std=self.min_std,
                max_std=self.max_std,
                constant_std=0.1 if self.prior_constant_std is None else self.prior_constant_std,
            )
            self.posterior_head = RSSMGaussianHead(
                in_dim=self.dim_h + self.dim_obs_embed,
                out_dim=self.dim_z,
                hidden_dim=self.dim_h,
                min_std=self.min_std,
                max_std=self.max_std,
                constant_std=0.1 if self.posterior_constant_std is None else self.posterior_constant_std,
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
        self,
        params: torch.Tensor,
        *,
        constant_std: float | None = None,
    ) -> tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mixed-precision training can occasionally produce NaN/Inf logits.
        # Clamp/sanitize before constructing Normal distributions.
        params = torch.nan_to_num(params, nan=0.0, posinf=1e4, neginf=-1e4)
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        mean = torch.clamp(mean, min=-1e4, max=1e4)
        if constant_std is not None:
            std = torch.full_like(mean, fill_value=float(max(1e-6, constant_std)))
        else:
            raw_std = torch.clamp(raw_std, min=-20.0, max=20.0)
            std = F.softplus(raw_std) + self.min_std
            std = torch.clamp(std, min=self.min_std, max=self.max_std)
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
    ) -> torch.Tensor:
        prev_z = prev_state.z if self.use_stochastic_path else torch.zeros_like(prev_state.z)
        return torch.cat([prev_z, control_embed, exogenous_embed], dim=-1)

    def transition(
        self,
        prev_state: RSSMState,
        control_embed: torch.Tensor,
        exogenous_embed: torch.Tensor,
    ) -> torch.Tensor:
        # CAUTION: y/obs must NOT enter here. Only in posterior.
        trans_raw = self._transition_input(
            prev_state=prev_state,
            control_embed=control_embed,
            exogenous_embed=exogenous_embed,
        )
        trans_raw = self.pre_gru_norm(trans_raw)
        pre_gru = self.transition_mlp(trans_raw)
        prev_h = prev_state.h if self.use_dual_path else torch.zeros_like(prev_state.h)
        return self.transition_gru(pre_gru, prev_h)

    def prior_from_h(
        self, h_t: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_stochastic_path:
            mean = torch.zeros(h_t.shape[0], self.dim_z, device=h_t.device, dtype=h_t.dtype)
            std = torch.full_like(mean, fill_value=self.min_std)
            dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=mean, scale=std),
                1,
            )
            logvar = 2.0 * torch.log(std)
            return dist, mean, std, logvar, None
        if self.latent_distribution == "categorical":
            logits = self.prior_head(h_t)
            probs = torch.softmax(logits, dim=-1)
            mean = probs.reshape(*probs.shape[:-2], -1)
            var = (probs * (1.0 - probs)).clamp_min(1e-6).reshape(*probs.shape[:-2], -1)
            std = torch.sqrt(var)
            logvar = torch.log(var)
            return logits, mean, std, logvar, logits.reshape(*logits.shape[:-2], -1)
        dist, mean, std, logvar = self.prior_head(h_t)
        return dist, mean, std, logvar, None

    def posterior_from_h(
        self, h_t: torch.Tensor, observation_embed: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_stochastic_path:
            mean = torch.zeros(h_t.shape[0], self.dim_z, device=h_t.device, dtype=h_t.dtype)
            std = torch.full_like(mean, fill_value=self.min_std)
            dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=mean, scale=std),
                1,
            )
            logvar = 2.0 * torch.log(std)
            return dist, mean, std, logvar, None
        post_in = torch.cat([h_t, observation_embed], dim=-1)
        if self.latent_distribution == "categorical":
            logits = self.posterior_head(post_in)
            probs = torch.softmax(logits, dim=-1)
            mean = probs.reshape(*probs.shape[:-2], -1)
            var = (probs * (1.0 - probs)).clamp_min(1e-6).reshape(*probs.shape[:-2], -1)
            std = torch.sqrt(var)
            logvar = torch.log(var)
            return logits, mean, std, logvar, logits.reshape(*logits.shape[:-2], -1)
        dist, mean, std, logvar = self.posterior_head(post_in)
        return dist, mean, std, logvar, None

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
        )
        prior_dist, prior_mean, prior_std, prior_logvar, prior_logits = self.prior_from_h(h_t)
        post_dist, post_mean, post_std, post_logvar, post_logits = self.posterior_from_h(h_t, observation_embed)
        if self.latent_distribution == "categorical":
            if post_logits is None:
                raise RuntimeError("Categorical latent mode requires posterior logits.")
            logits_shaped = post_logits.view(*post_logits.shape[:-1], self.stochastic_groups, self.stochastic_classes)
            z_t = fast_sample(logits_shaped) if sample else post_mean
        else:
            z_t = post_dist.rsample() if sample else post_mean
        if not self.use_stochastic_path:
            z_t = torch.zeros_like(z_t)
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
            prior_logits=prior_logits,
            posterior_logits=post_logits,
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
        )
        prior_dist, prior_mean, prior_std, prior_logvar, prior_logits = self.prior_from_h(h_t)
        if self.latent_distribution == "categorical":
            if prior_logits is None:
                raise RuntimeError("Categorical latent mode requires prior logits.")
            logits_shaped = prior_logits.view(*prior_logits.shape[:-1], self.stochastic_groups, self.stochastic_classes)
            z_t = fast_sample(logits_shaped) if sample else prior_mean
        else:
            z_t = prior_dist.rsample() if sample else prior_mean
        if not self.use_stochastic_path:
            z_t = torch.zeros_like(z_t)
        state = RSSMState(h=h_t, z=z_t)
        return RSSMOutput(
            state=state,
            h=h_t,
            prior=prior_dist,
            prior_mean=prior_mean,
            prior_std=prior_std,
            prior_logvar=prior_logvar,
            prior_logits=prior_logits,
        )


__all__ = ["RSSMState", "RSSMOutput", "RSSMCell"]
