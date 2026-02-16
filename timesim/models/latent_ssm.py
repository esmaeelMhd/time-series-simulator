"""Probabilistic latent state-space world model with Student-t emissions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WorldModelBase


class LatentSSMWorldModel(WorldModelBase):
    """Latent probabilistic world model (DKF/VRNN style)."""

    is_probabilistic = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        pred_len: int = 1,
        min_scale: float = 1e-4,
        min_df: float = 2.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.pred_len = int(pred_len)
        self.min_scale = float(min_scale)
        self.min_df = float(min_df)

        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.obs_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3 * self.output_dim),
        )

    def init_state(self, warmup_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, h = self.gru(warmup_seq)
        return {"h": h}

    @staticmethod
    def _kl_diag_normal(
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        post_var = torch.exp(post_logvar)
        prior_var = torch.exp(prior_logvar)
        kl = 0.5 * (
            prior_logvar - post_logvar
            + (post_var + (post_mu - prior_mu) ** 2) / (prior_var + 1e-12)
            - 1.0
        )
        return kl.sum(dim=-1)

    def _split_mu_logvar(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, raw_logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(raw_logvar, min=-10.0, max=10.0)
        return mu, logvar

    def _obs_params(self, h_t: torch.Tensor, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.obs_net(torch.cat([h_t, z_t], dim=-1))
        loc, raw_scale, raw_df = torch.chunk(raw, 3, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        df = F.softplus(raw_df) + self.min_df
        return loc, scale, df

    def _assemble_step_input(
        self,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prev_output_t is None:
            prev_output_t = torch.zeros(
                control_t.shape[0], self.output_dim, device=control_t.device, dtype=control_t.dtype
            )
        x_t = torch.cat([control_t, exo_t, prev_output_t], dim=-1)
        if x_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x_t.shape[-1]}. "
                "Make sure input_dim = control_dim + exo_dim + output_dim"
            )
        return x_t

    def step(
        self,
        state: Dict[str, torch.Tensor],
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x_t = self._assemble_step_input(control_t, exo_t, prev_output_t)
        out_t, h_new = self.gru(x_t.unsqueeze(1), state["h"])
        h_t = out_t.squeeze(1)

        prior_mu, _ = self._split_mu_logvar(self.prior_net(h_t))
        loc, _, _ = self._obs_params(h_t, prior_mu)
        return {"h": h_new}, loc

    def _rollout_impl(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
        sample_latent: bool = False,
        sample_observation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if feedback in {"teacher", "mixed"} and targets is None:
            raise ValueError(f"targets required when feedback='{feedback}'")

        if "inputs" in warmup_seq:
            warmup_inputs = warmup_seq["inputs"]
        else:
            warmup_inputs = torch.cat(
                [warmup_seq["controls"], warmup_seq["exogenous"], warmup_seq["outputs"]],
                dim=-1,
            )
        state = self.init_state(warmup_inputs)

        controls = rollout_inputs["controls"]
        exogenous = rollout_inputs["exogenous"]
        batch_size = controls.shape[0]
        device = controls.device

        if "outputs" in warmup_seq:
            prev_output = warmup_seq["outputs"][:, -1, :]
        else:
            prev_output = warmup_inputs[:, -1, -self.output_dim:]

        predictions = torch.empty(batch_size, horizon, self.output_dim, dtype=torch.float32, device=device)
        dist_loc = torch.empty_like(predictions)
        dist_scale = torch.empty_like(predictions)
        dist_df = torch.empty_like(predictions)
        kl_terms = torch.zeros(batch_size, horizon, dtype=torch.float32, device=device)

        states = []
        for t in range(horizon):
            control_t = controls[:, t, :]
            exo_t = exogenous[:, t, :]
            x_t = self._assemble_step_input(control_t, exo_t, prev_output)

            out_t, h_new = self.gru(x_t.unsqueeze(1), state["h"])
            h_t = out_t.squeeze(1)
            prior_mu, prior_logvar = self._split_mu_logvar(self.prior_net(h_t))

            # IMPORTANT: avoid target leakage in autoregressive rollouts.
            # Only use posterior q(z_t|h_t,y_t) when explicit teacher forcing is active.
            use_posterior = (feedback == "teacher") and (targets is not None)
            if use_posterior:
                post_in = torch.cat([h_t, targets[:, t, :]], dim=-1)
                post_mu, post_logvar = self._split_mu_logvar(self.posterior_net(post_in))
                if sample_latent:
                    eps = torch.randn_like(post_mu)
                    z_t = post_mu + eps * torch.exp(0.5 * post_logvar)
                else:
                    z_t = post_mu
                kl_terms[:, t] = self._kl_diag_normal(post_mu, post_logvar, prior_mu, prior_logvar)
            else:
                if sample_latent:
                    eps = torch.randn_like(prior_mu)
                    z_t = prior_mu + eps * torch.exp(0.5 * prior_logvar)
                else:
                    z_t = prior_mu

            loc_t, scale_t, df_t = self._obs_params(h_t, z_t)
            dist_loc[:, t, :] = loc_t
            dist_scale[:, t, :] = scale_t
            dist_df[:, t, :] = df_t

            if sample_observation:
                dist = torch.distributions.StudentT(df_t, loc=loc_t, scale=scale_t)
                pred_t = dist.sample()
            else:
                pred_t = loc_t
            predictions[:, t, :] = pred_t
            state = {"h": h_new}
            states.append(state)

            if feedback == "model":
                prev_output = pred_t
            elif feedback == "teacher":
                prev_output = targets[:, t, :]
            else:
                use_teacher = torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio
                prev_output = torch.where(use_teacher, targets[:, t, :], pred_t)

        return {
            "predictions": predictions,
            "dist_loc": dist_loc,
            "dist_scale": dist_scale,
            "dist_df": dist_df,
            "kl_terms": kl_terms,
            "states": states,
        }

    def rollout(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._rollout_impl(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            feedback=feedback,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets,
            sample_latent=False,
            sample_observation=False,
        )

    @torch.no_grad()
    def rollout_mc(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        n_samples: int = 256,
        interval_level: float = 0.90,
    ) -> Dict[str, torch.Tensor]:
        n_samples = max(1, int(n_samples))
        alpha = max(0.0, min(1.0, float(interval_level)))
        lo_q = (1.0 - alpha) / 2.0
        hi_q = 1.0 - lo_q

        if "inputs" in warmup_seq:
            warmup_inputs = warmup_seq["inputs"]
        else:
            warmup_inputs = torch.cat(
                [warmup_seq["controls"], warmup_seq["exogenous"], warmup_seq["outputs"]],
                dim=-1,
            )
        controls = rollout_inputs["controls"]
        exogenous = rollout_inputs["exogenous"]
        batch_size = controls.shape[0]

        # Vectorized MC: run all samples in one batched rollout call.
        warmup_mc = (
            warmup_inputs.unsqueeze(0)
            .repeat(n_samples, 1, 1, 1)
            .reshape(n_samples * batch_size, warmup_inputs.shape[1], warmup_inputs.shape[2])
        )
        controls_mc = (
            controls.unsqueeze(0)
            .repeat(n_samples, 1, 1, 1)
            .reshape(n_samples * batch_size, controls.shape[1], controls.shape[2])
        )
        exogenous_mc = (
            exogenous.unsqueeze(0)
            .repeat(n_samples, 1, 1, 1)
            .reshape(n_samples * batch_size, exogenous.shape[1], exogenous.shape[2])
        )
        out = self._rollout_impl(
            warmup_seq={"inputs": warmup_mc},
            rollout_inputs={"controls": controls_mc, "exogenous": exogenous_mc},
            horizon=horizon,
            feedback="model",
            sample_latent=True,
            sample_observation=True,
        )
        samples_t = out["predictions"].reshape(n_samples, batch_size, horizon, self.output_dim)
        mean_t = samples_t.mean(dim=0)
        lower_t = torch.quantile(samples_t, q=lo_q, dim=0)
        upper_t = torch.quantile(samples_t, q=hi_q, dim=0)
        return {
            "samples": samples_t,
            "mean": mean_t,
            "lower": lower_t,
            "upper": upper_t,
            "interval_level": alpha,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h_t = out[:, -1, :]
        prior_mu, _ = self._split_mu_logvar(self.prior_net(h_t))
        loc, _, _ = self._obs_params(h_t, prior_mu)
        return loc.unsqueeze(1).repeat(1, self.pred_len, 1)
