"""Recurrent state-space world model (RSSM) with typed encoders.

Design constraints enforced (default configuration):
- Transition depends on control + exogenous only (never objective/target).
- Separate, non-shared encoders for control/exogenous/objective variables.
- Dual latent path: deterministic h_t + stochastic z_t.
- Prior p(z_t|h_t) and posterior q(z_t|h_t,y_t) heads at every step.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .base import WorldModelBase
from .decoders import AuxiliaryDecoder, ObjectiveDecoder
from .distributions import diagonal_independent_normal
from .encoders import (
    ControlEncoder,
    ExogenousEncoder,
    ObservationEncoder,
    assert_no_shared_encoder_params,
)
from .rssm import RSSMCell, RSSMState


class LatentSSMWorldModel(WorldModelBase):
    """RSSM world model for intervention-aware simulation."""

    is_probabilistic = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        pred_len: int = 1,
        min_scale: float = 1e-4,
        min_df: float = 2.1,
        control_dim: Optional[int] = None,
        exogenous_dim: Optional[int] = None,
        encoder_dim: int = 64,
        decoder_layers: int = 2,
        use_symlog: bool = False,
        use_aux_decoder: bool = True,
        use_dual_path: bool = True,
        leak_objective_to_transition: bool = False,
        allow_objective_leak_for_ablation: bool = False,
        allow_disable_aux_decoder_for_ablation: bool = False,
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
        self.control_dim = None if control_dim is None else int(control_dim)
        self.exogenous_dim = None if exogenous_dim is None else int(exogenous_dim)
        self.encoder_dim = int(encoder_dim)
        self.decoder_layers = max(1, int(decoder_layers))
        self.use_symlog = bool(use_symlog)
        self.use_aux_decoder = bool(use_aux_decoder)
        self.use_dual_path = bool(use_dual_path)
        self.leak_objective_to_transition = bool(leak_objective_to_transition)
        self.min_latent_std = float(max(0.01, float(min_scale)))
        self.allow_objective_leak_for_ablation = bool(allow_objective_leak_for_ablation)
        self.allow_disable_aux_decoder_for_ablation = bool(allow_disable_aux_decoder_for_ablation)

        if self.leak_objective_to_transition and not self.allow_objective_leak_for_ablation:
            raise ValueError(
                "Objective leakage into transition is blocked by default. "
                "Set `allow_objective_leak_for_ablation=True` only for explicit ablations."
            )
        if (not self.use_aux_decoder) and (not self.allow_disable_aux_decoder_for_ablation):
            raise ValueError(
                "Auxiliary exogenous decoder disable is blocked by default. "
                "Set `allow_disable_aux_decoder_for_ablation=True` only for explicit ablations."
            )

        # Explicitly independent encoders per variable type (no weight sharing).
        self.control_encoder = ControlEncoder(
            input_dim=self.control_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.encoder_dim,
        )
        self.exogenous_encoder = ExogenousEncoder(
            input_dim=self.exogenous_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.encoder_dim,
        )
        self.observation_encoder = ObservationEncoder(
            input_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.encoder_dim,
        )
        self._assert_encoder_independence()
        self.rssm_cell = RSSMCell(
            dim_h=self.hidden_dim,
            dim_z=self.latent_dim,
            dim_control_embed=self.encoder_dim,
            dim_exogenous_embed=self.encoder_dim,
            dim_obs_embed=self.encoder_dim,
            transition_hidden_dim=self.hidden_dim,
            min_std=self.min_latent_std,
            use_dual_path=self.use_dual_path,
            leak_objective_to_transition=self.leak_objective_to_transition,
        )

        # Decoders from full latent state [h_t, z_t].
        self.obs_decoder = ObjectiveDecoder(
            in_dim=self.hidden_dim + self.latent_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_layers,
        )
        self._aux_decoder_hidden = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._aux_decoder_heads = nn.ModuleDict()

    def _assert_encoder_independence(self) -> None:
        assert_no_shared_encoder_params(
            self.control_encoder,
            self.exogenous_encoder,
            self.observation_encoder,
        )

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.expm1(torch.abs(x)))

    @staticmethod
    def _stack_or_empty(buffers, batch: int, time: int, dim: int, device: torch.device, dtype: torch.dtype):
        if buffers:
            return torch.stack(buffers, dim=1)
        return torch.zeros(batch, time, dim, device=device, dtype=dtype)

    def _get_aux_head(
        self,
        exogenous_dim: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[AuxiliaryDecoder]:
        if exogenous_dim <= 0:
            return None
        key = str(int(exogenous_dim))
        if key not in self._aux_decoder_heads:
            head = AuxiliaryDecoder(
                in_dim=self.hidden_dim,
                out_dim=int(exogenous_dim),
                hidden_dim=self.hidden_dim,
                num_layers=max(1, self.decoder_layers),
            )
            if device is not None or dtype is not None:
                head = head.to(device=device, dtype=dtype)
            self._aux_decoder_heads[key] = head
        head = self._aux_decoder_heads[key]
        if device is not None and head.net[0].weight.device != device:
            head = head.to(device=device)
            self._aux_decoder_heads[key] = head
        if dtype is not None and head.net[0].weight.dtype != dtype:
            head = head.to(dtype=dtype)
            self._aux_decoder_heads[key] = head
        return head

    def _decode_obs(
        self, h_t: torch.Tensor, z_t: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.cat([h_t, z_t], dim=-1)
        dist_latent, loc_latent, scale = self.obs_decoder(
            latent,
            min_scale=max(0.01, self.min_scale),
        )
        loc = self.symexp(loc_latent) if self.use_symlog else loc_latent
        dist = diagonal_independent_normal(loc=loc, scale=scale)
        return dist, dist_latent, loc, scale, loc_latent

    def _decode_exogenous(
        self,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        exogenous_dim: int,
    ) -> Tuple[Optional[torch.distributions.Distribution], torch.Tensor, torch.Tensor]:
        if exogenous_dim <= 0:
            bsz = h_t.shape[0]
            empty = torch.zeros(bsz, 0, device=h_t.device, dtype=h_t.dtype)
            return None, empty, empty
        if not self.use_aux_decoder:
            bsz = h_t.shape[0]
            loc = torch.zeros(bsz, exogenous_dim, device=h_t.device, dtype=h_t.dtype)
            scale = torch.full_like(loc, fill_value=max(0.01, self.min_scale))
            dist = diagonal_independent_normal(loc=loc, scale=scale)
            return dist, loc, scale
        hidden = self._aux_decoder_hidden(torch.cat([h_t, z_t], dim=-1))
        head = self._get_aux_head(
            exogenous_dim,
            device=h_t.device,
            dtype=h_t.dtype,
        )
        assert head is not None
        dist, loc, scale = head(hidden, min_scale=max(0.01, self.min_scale))
        return dist, loc, scale

    def _transition_step(
        self,
        state: RSSMState,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        objective_encoded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c_enc = self.control_encoder(control_t)
        x_enc = self.exogenous_encoder(exo_t)
        h_t = self.rssm_cell.transition(
            prev_state=state,
            control_embed=c_enc,
            exogenous_embed=x_enc,
            observation_embed=objective_encoded,
        )
        return h_t

    def _zero_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RSSMState:
        return self.rssm_cell.initial_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _infer_split_dims(
        self,
        feature_dim: int,
        control_dim_hint: Optional[int] = None,
        exogenous_dim_hint: Optional[int] = None,
    ) -> Tuple[int, int]:
        if control_dim_hint is not None and exogenous_dim_hint is not None:
            return int(control_dim_hint), int(exogenous_dim_hint)
        if self.control_dim is not None and self.exogenous_dim is not None:
            return int(self.control_dim), int(self.exogenous_dim)

        # Backward-compat fallback for callers that only provide concatenated warmup inputs.
        if control_dim_hint is not None:
            cdim = int(control_dim_hint)
            return cdim, max(0, feature_dim - cdim)
        if exogenous_dim_hint is not None:
            xdim = int(exogenous_dim_hint)
            return max(0, feature_dim - xdim), xdim
        return 0, int(feature_dim)

    def init_state(self, warmup_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        if warmup_seq.dim() != 3:
            raise ValueError(
                f"warmup_seq must be rank-3 (B,T,F), got shape {tuple(warmup_seq.shape)}"
            )
        batch_size, _, feat_dim = warmup_seq.shape
        device, dtype = warmup_seq.device, warmup_seq.dtype

        if feat_dim >= self.output_dim:
            dyn_dim = feat_dim - self.output_dim
            cdim, xdim = self._infer_split_dims(dyn_dim)
            controls = warmup_seq[:, :, :cdim] if cdim > 0 else warmup_seq.new_zeros(batch_size, warmup_seq.shape[1], 0)
            exogenous = (
                warmup_seq[:, :, cdim:cdim + xdim]
                if xdim > 0
                else warmup_seq.new_zeros(batch_size, warmup_seq.shape[1], 0)
            )
            outputs = warmup_seq[:, :, -self.output_dim:]
            conditioned = self.observe(
                controls=controls,
                exogenous=exogenous,
                observations=outputs,
                initial_state=None,
                sample_posterior=False,
            )
            return {
                "h": conditioned["state"].h,
                "z": conditioned["state"].z,
            }

        state = self._zero_state(batch_size, device=device, dtype=dtype)
        return {"h": state.h, "z": state.z}

    def step(
        self,
        state: Dict[str, torch.Tensor],
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
        prev_output_t: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        prev = RSSMState(h=state["h"], z=state["z"])
        c_enc = self.control_encoder(control_t)
        x_enc = self.exogenous_encoder(exo_t)
        step_out = self.rssm_cell.imagine(
            prev_state=prev,
            control_embed=c_enc,
            exogenous_embed=x_enc,
            sample=False,
        )
        h_t = step_out.state.h
        z_t = step_out.state.z
        _, _, pred_t, _, _ = self._decode_obs(h_t, z_t)
        return {
            "h": h_t,
            "z": z_t,
            "prior_mu": step_out.prior_mean,
            "prior_logvar": step_out.prior_logvar,
        }, pred_t

    def observe(
        self,
        controls: torch.Tensor,
        exogenous: torch.Tensor,
        observations: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
        sample_posterior: bool = False,
    ) -> Dict[str, object]:
        """Run posterior filtering over a sequence."""
        if controls.shape[:2] != exogenous.shape[:2] or controls.shape[:2] != observations.shape[:2]:
            raise ValueError("controls/exogenous/observations must have matching (batch,time) dimensions")

        batch_size, horizon, _ = controls.shape
        device, dtype = controls.device, controls.dtype
        state = initial_state if initial_state is not None else self._zero_state(batch_size, device, dtype)

        deter_states = []
        stoch_states = []
        prior_mus = []
        prior_logvars = []
        post_mus = []
        post_logvars = []
        kl_terms = []
        recon_means = []
        recon_scales = []
        recon_means_latent = []
        recon_dists = []
        recon_dists_latent = []
        aux_means = []
        aux_scales = []
        aux_dists = []

        for t in range(horizon):
            c_t = controls[:, t, :]
            x_t = exogenous[:, t, :]
            y_t_raw = observations[:, t, :]
            y_t = self.symlog(y_t_raw) if self.use_symlog else y_t_raw
            y_enc = self.observation_encoder(y_t)
            c_enc = self.control_encoder(c_t)
            x_enc = self.exogenous_encoder(x_t)
            step_out = self.rssm_cell.observe(
                prev_state=state,
                control_embed=c_enc,
                exogenous_embed=x_enc,
                observation_embed=y_enc,
                sample=sample_posterior,
            )
            h_t = step_out.state.h
            z_t = step_out.state.z
            prior_mu = step_out.prior_mean
            prior_logvar = step_out.prior_logvar
            post_mu = step_out.posterior_mean
            post_logvar = step_out.posterior_logvar
            if post_mu is None or post_logvar is None or step_out.posterior is None:
                raise RuntimeError("Posterior outputs are required in observe().")

            y_dist, y_dist_latent, y_loc, y_scale, y_loc_latent = self._decode_obs(h_t, z_t)
            x_dist, x_loc, x_scale = self._decode_exogenous(h_t, z_t, exogenous.shape[-1])
            kl_t = torch.distributions.kl_divergence(step_out.posterior, step_out.prior)

            deter_states.append(h_t)
            stoch_states.append(z_t)
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
            post_mus.append(post_mu)
            post_logvars.append(post_logvar)
            kl_terms.append(kl_t)
            recon_means.append(y_loc)
            recon_scales.append(y_scale)
            recon_means_latent.append(y_loc_latent)
            recon_dists.append(y_dist)
            recon_dists_latent.append(y_dist_latent)
            aux_means.append(x_loc)
            aux_scales.append(x_scale)
            aux_dists.append(x_dist)

            state = step_out.state

        return {
            "deter": torch.stack(deter_states, dim=1),
            "stoch": torch.stack(stoch_states, dim=1),
            "prior_mu": torch.stack(prior_mus, dim=1),
            "prior_logvar": torch.stack(prior_logvars, dim=1),
            "posterior_mu": torch.stack(post_mus, dim=1),
            "posterior_logvar": torch.stack(post_logvars, dim=1),
            "kl_terms": torch.stack(kl_terms, dim=1),
            "dist_loc": torch.stack(recon_means, dim=1),
            "dist_loc_latent": torch.stack(recon_means_latent, dim=1),
            "dist_scale": torch.stack(recon_scales, dim=1),
            "aux_loc": torch.stack(aux_means, dim=1),
            "aux_scale": torch.stack(aux_scales, dim=1),
            "objective_dists": recon_dists,
            "objective_dists_latent": recon_dists_latent,
            "aux_dists": aux_dists,
            "state": state,
            "predictions": torch.stack(recon_means, dim=1),
        }

    def imagine(
        self,
        initial_state: RSSMState,
        future_controls: torch.Tensor,
        future_exogenous: torch.Tensor,
        n_steps: Optional[int] = None,
        n_samples: int = 1,
        sample_latent: bool = True,
    ) -> Dict[str, object]:
        """Roll forward using prior dynamics only (no target leakage)."""
        if future_controls.shape[:2] != future_exogenous.shape[:2]:
            raise ValueError("future_controls and future_exogenous must have matching (batch,time)")

        batch_size, horizon, _ = future_controls.shape
        if n_steps is None:
            n_steps = horizon
        n_steps = int(n_steps)
        horizon = min(horizon, n_steps)
        n_samples = max(1, int(n_samples))

        controls = future_controls[:, :horizon, :]
        exogenous = future_exogenous[:, :horizon, :]

        # Vectorized ensemble by expanding batch axis.
        if n_samples > 1:
            controls = controls.unsqueeze(0).repeat(n_samples, 1, 1, 1)
            exogenous = exogenous.unsqueeze(0).repeat(n_samples, 1, 1, 1)
            controls = controls.reshape(n_samples * batch_size, horizon, controls.shape[-1])
            exogenous = exogenous.reshape(n_samples * batch_size, horizon, exogenous.shape[-1])
            state = RSSMState(
                h=initial_state.h.unsqueeze(0).repeat(n_samples, 1, 1).reshape(n_samples * batch_size, self.hidden_dim),
                z=initial_state.z.unsqueeze(0).repeat(n_samples, 1, 1).reshape(n_samples * batch_size, self.latent_dim),
            )
        else:
            state = RSSMState(h=initial_state.h, z=initial_state.z)

        deter_states = []
        stoch_states = []
        prior_mus = []
        prior_logvars = []
        pred_means = []
        pred_means_latent = []
        pred_scales = []
        pred_dists = []
        pred_dists_latent = []
        aux_means = []
        aux_scales = []
        aux_dists = []

        for t in range(horizon):
            c_t = controls[:, t, :]
            x_t = exogenous[:, t, :]
            c_enc = self.control_encoder(c_t)
            x_enc = self.exogenous_encoder(x_t)
            step_out = self.rssm_cell.imagine(
                prev_state=state,
                control_embed=c_enc,
                exogenous_embed=x_enc,
                sample=sample_latent,
            )
            h_t = step_out.state.h
            z_t = step_out.state.z
            prior_mu = step_out.prior_mean
            prior_logvar = step_out.prior_logvar

            y_dist, y_dist_latent, y_loc, y_scale, y_loc_latent = self._decode_obs(h_t, z_t)
            x_dist, x_loc, x_scale = self._decode_exogenous(h_t, z_t, exogenous.shape[-1])

            deter_states.append(h_t)
            stoch_states.append(z_t)
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
            pred_means.append(y_loc)
            pred_means_latent.append(y_loc_latent)
            pred_scales.append(y_scale)
            pred_dists.append(y_dist)
            pred_dists_latent.append(y_dist_latent)
            aux_means.append(x_loc)
            aux_scales.append(x_scale)
            aux_dists.append(x_dist)

            state = step_out.state

        pred = torch.stack(pred_means, dim=1)
        pred_scale = torch.stack(pred_scales, dim=1)
        pred_latent = torch.stack(pred_means_latent, dim=1)
        aux_loc = torch.stack(aux_means, dim=1)
        aux_scale = torch.stack(aux_scales, dim=1)

        if n_samples > 1:
            pred_samples = pred.reshape(n_samples, batch_size, horizon, self.output_dim)
            pred_scale_samples = pred_scale.reshape(n_samples, batch_size, horizon, self.output_dim)
            pred_latent_samples = pred_latent.reshape(n_samples, batch_size, horizon, self.output_dim)
            aux_loc_samples = aux_loc.reshape(n_samples, batch_size, horizon, aux_loc.shape[-1])
            aux_scale_samples = aux_scale.reshape(n_samples, batch_size, horizon, aux_scale.shape[-1])
            prior_mu_samples = torch.stack(prior_mus, dim=1).reshape(n_samples, batch_size, horizon, self.latent_dim)
            prior_logvar_samples = torch.stack(prior_logvars, dim=1).reshape(n_samples, batch_size, horizon, self.latent_dim)
            deter_samples = torch.stack(deter_states, dim=1).reshape(n_samples, batch_size, horizon, self.hidden_dim)
            stoch_samples = torch.stack(stoch_states, dim=1).reshape(n_samples, batch_size, horizon, self.latent_dim)

            return {
                "samples": pred_samples,
                "dist_scale_samples": pred_scale_samples,
                "dist_loc_latent_samples": pred_latent_samples,
                "aux_loc_samples": aux_loc_samples,
                "aux_scale_samples": aux_scale_samples,
                "mean": pred_samples.mean(dim=0),
                "std": pred_samples.std(dim=0, unbiased=False),
                "dist_scale": pred_scale_samples.mean(dim=0),
                "dist_loc_latent": pred_latent_samples.mean(dim=0),
                "aux_loc": aux_loc_samples.mean(dim=0),
                "aux_scale": aux_scale_samples.mean(dim=0),
                "prior_mu": prior_mu_samples.mean(dim=0),
                "prior_logvar": prior_logvar_samples.mean(dim=0),
                "deter": deter_samples.mean(dim=0),
                "stoch": stoch_samples.mean(dim=0),
                "objective_dists": pred_dists,
                "objective_dists_latent": pred_dists_latent,
                "aux_dists": aux_dists,
                "state": RSSMState(
                    h=state.h.reshape(n_samples, batch_size, self.hidden_dim).mean(dim=0),
                    z=state.z.reshape(n_samples, batch_size, self.latent_dim).mean(dim=0),
                ),
            }

        return {
            "predictions": pred,
            "dist_scale": pred_scale,
            "dist_loc_latent": pred_latent,
            "aux_loc": aux_loc,
            "aux_scale": aux_scale,
            "prior_mu": torch.stack(prior_mus, dim=1),
            "prior_logvar": torch.stack(prior_logvars, dim=1),
            "deter": torch.stack(deter_states, dim=1),
            "stoch": torch.stack(stoch_states, dim=1),
            "objective_dists": pred_dists,
            "objective_dists_latent": pred_dists_latent,
            "aux_dists": aux_dists,
            "state": state,
        }

    def condition_then_simulate(
        self,
        history_controls: torch.Tensor,
        history_exogenous: torch.Tensor,
        history_objectives: torch.Tensor,
        future_controls: torch.Tensor,
        future_exogenous: torch.Tensor,
        n_steps: Optional[int] = None,
        n_samples: int = 50,
    ) -> Dict[str, object]:
        observed = self.observe(
            controls=history_controls,
            exogenous=history_exogenous,
            observations=history_objectives,
            initial_state=None,
            sample_posterior=False,
        )
        initial_state = observed["state"]
        assert isinstance(initial_state, RSSMState)
        return self.imagine(
            initial_state=initial_state,
            future_controls=future_controls,
            future_exogenous=future_exogenous,
            n_steps=n_steps,
            n_samples=n_samples,
            sample_latent=True,
        )

    def imagine_forward(
        self,
        initial_state: RSSMState,
        future_controls: torch.Tensor,
        future_exogenous: torch.Tensor,
        n_steps: Optional[int] = None,
        n_samples: int = 1,
        sample_latent: bool = True,
    ) -> Dict[str, object]:
        """Simulation mode entrypoint.

        Loops prior dynamics only (no observations) and returns predicted
        objective distributions/summary statistics.
        """
        return self.imagine(
            initial_state=initial_state,
            future_controls=future_controls,
            future_exogenous=future_exogenous,
            n_steps=n_steps,
            n_samples=n_samples,
            sample_latent=sample_latent,
        )

    def imagine_rollout_with_loss(
        self,
        batch: Dict[str, torch.Tensor],
        context_len: int,
        horizon: int,
        *,
        sample_posterior: bool = True,
        sample_prior: bool = True,
        compute_rollout_dtw: bool = False,
        rollout_dtw_gamma: float = 0.1,
    ) -> Dict[str, object]:
        """Hybrid mode: observe context then imagine horizon with loss terms."""
        controls = batch["control"]
        exogenous = batch["exogenous"]
        objectives = batch["objective"]
        if controls.shape[:2] != exogenous.shape[:2] or controls.shape[:2] != objectives.shape[:2]:
            raise ValueError("Batch tensors must share (batch,time) dimensions.")

        total_steps = controls.shape[1]
        context_len = int(max(0, context_len))
        horizon = int(max(0, horizon))
        if context_len > total_steps:
            raise ValueError(f"context_len ({context_len}) > sequence length ({total_steps})")
        if context_len + horizon > total_steps:
            horizon = total_steps - context_len

        context_controls = controls[:, :context_len, :]
        context_exogenous = exogenous[:, :context_len, :]
        context_objectives = objectives[:, :context_len, :]
        future_controls = controls[:, context_len:context_len + horizon, :]
        future_exogenous = exogenous[:, context_len:context_len + horizon, :]
        future_objectives = objectives[:, context_len:context_len + horizon, :]

        observed = self.observe(
            controls=context_controls,
            exogenous=context_exogenous,
            observations=context_objectives,
            initial_state=None,
            sample_posterior=sample_posterior,
        )
        obs_recon_nll = self._sequence_nll_from_dist_list(
            observed.get("objective_dists_latent", []),  # type: ignore[arg-type]
            self.symlog(context_objectives) if self.use_symlog else context_objectives,
        )
        obs_kl = self._kl_mean_from_params(
            prior_mu=observed["prior_mu"],  # type: ignore[index]
            prior_logvar=observed["prior_logvar"],  # type: ignore[index]
            post_mu=observed["posterior_mu"],  # type: ignore[index]
            post_logvar=observed["posterior_logvar"],  # type: ignore[index]
            min_std=self.min_latent_std,
        )
        obs_aux_nll = self._sequence_nll_from_dist_list(
            observed.get("aux_dists", []),  # type: ignore[arg-type]
            context_exogenous,
            allow_none=True,
        )

        rollout_nll = torch.zeros((), dtype=controls.dtype, device=controls.device)
        rollout_aux_nll = torch.zeros((), dtype=controls.dtype, device=controls.device)
        rollout_dtw = torch.zeros((), dtype=controls.dtype, device=controls.device)
        imagined: Optional[Dict[str, object]] = None

        if horizon > 0:
            init_state = observed["state"]
            assert isinstance(init_state, RSSMState)
            imagined = self.imagine_forward(
                initial_state=init_state,
                future_controls=future_controls,
                future_exogenous=future_exogenous,
                n_steps=horizon,
                n_samples=1,
                sample_latent=sample_prior,
            )
            rollout_nll = self._sequence_nll_from_dist_list(
                imagined.get("objective_dists_latent", []),  # type: ignore[arg-type]
                self.symlog(future_objectives) if self.use_symlog else future_objectives,
            )
            rollout_aux_nll = self._sequence_nll_from_dist_list(
                imagined.get("aux_dists", []),  # type: ignore[arg-type]
                future_exogenous,
                allow_none=True,
            )
            if compute_rollout_dtw:
                from ..training.losses import soft_dtw_distance

                pred_means = imagined.get("predictions", imagined.get("mean"))
                if not torch.is_tensor(pred_means):
                    raise RuntimeError("imagine_forward output missing tensor predictions/mean.")
                rollout_dtw = soft_dtw_distance(
                    pred_means,
                    future_objectives,
                    gamma=float(rollout_dtw_gamma),
                )

        return {
            "observed": observed,
            "imagined": imagined,
            "context_len": int(context_len),
            "horizon": int(horizon),
            "obs_recon_nll": obs_recon_nll,
            "obs_kl": obs_kl,
            "obs_aux_nll": obs_aux_nll,
            "rollout_nll": rollout_nll,
            "rollout_aux_nll": rollout_aux_nll,
            "rollout_dtw": rollout_dtw,
            "kl_raw_mean": obs_kl.detach(),
        }

    def _warmup_components(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "controls" in warmup_seq and "exogenous" in warmup_seq and "outputs" in warmup_seq:
            return warmup_seq["controls"], warmup_seq["exogenous"], warmup_seq["outputs"]

        if "inputs" not in warmup_seq:
            raise ValueError("warmup_seq must provide either inputs or controls/exogenous/outputs")

        inputs = warmup_seq["inputs"]
        batch, _, feat_dim = inputs.shape

        control_dim_hint = None
        exogenous_dim_hint = None
        if rollout_inputs is not None:
            if "controls" in rollout_inputs:
                control_dim_hint = int(rollout_inputs["controls"].shape[-1])
            if "exogenous" in rollout_inputs:
                exogenous_dim_hint = int(rollout_inputs["exogenous"].shape[-1])

        if feat_dim < self.output_dim:
            controls = inputs.new_zeros(batch, inputs.shape[1], 0)
            exogenous = inputs
            outputs = inputs.new_zeros(batch, inputs.shape[1], self.output_dim)
            return controls, exogenous, outputs

        dyn_dim = feat_dim - self.output_dim
        cdim, xdim = self._infer_split_dims(
            feature_dim=dyn_dim,
            control_dim_hint=control_dim_hint,
            exogenous_dim_hint=exogenous_dim_hint,
        )
        if cdim + xdim > dyn_dim:
            raise ValueError(
                f"Warmup split mismatch: control_dim({cdim}) + exogenous_dim({xdim}) > {dyn_dim}"
            )

        controls = inputs[:, :, :cdim] if cdim > 0 else inputs.new_zeros(batch, inputs.shape[1], 0)
        exogenous = (
            inputs[:, :, cdim:cdim + xdim]
            if xdim > 0
            else inputs.new_zeros(batch, inputs.shape[1], 0)
        )
        outputs = inputs[:, :, -self.output_dim:]
        return controls, exogenous, outputs

    @staticmethod
    def _sequence_nll_from_dist_list(
        dist_list: list[Optional[torch.distributions.Distribution]],
        targets: torch.Tensor,
        *,
        allow_none: bool = False,
    ) -> torch.Tensor:
        if targets.dim() != 3:
            raise ValueError(f"targets must be rank-3 (B,T,D), got {tuple(targets.shape)}")
        batch_size, horizon, _ = targets.shape
        if len(dist_list) != horizon:
            raise ValueError(
                f"Distribution list length ({len(dist_list)}) does not match horizon ({horizon})"
            )
        losses = []
        for t, dist in enumerate(dist_list):
            if dist is None:
                if allow_none:
                    continue
                raise ValueError(f"Missing distribution at timestep {t}")
            lp = dist.log_prob(targets[:, t, :])
            losses.append(-lp)
        if not losses:
            return torch.zeros((), dtype=targets.dtype, device=targets.device)
        stacked = torch.stack(losses, dim=1)  # (B,T_valid)
        return stacked.mean()

    def _split_observation_inputs(
        self, seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq.dim() != 3:
            raise ValueError(f"Expected rank-3 sequence (B,T,F), got {tuple(seq.shape)}")
        batch_size, horizon, feat_dim = seq.shape
        if feat_dim < self.output_dim:
            raise ValueError(
                f"Observation mode requires at least output_dim={self.output_dim} features, got {feat_dim}"
            )
        dyn_dim = feat_dim - self.output_dim
        cdim, xdim = self._infer_split_dims(dyn_dim)
        controls = seq[:, :, :cdim] if cdim > 0 else seq.new_zeros(batch_size, horizon, 0)
        exogenous = (
            seq[:, :, cdim:cdim + xdim]
            if xdim > 0
            else seq.new_zeros(batch_size, horizon, 0)
        )
        objectives = seq[:, :, -self.output_dim:]
        return controls, exogenous, objectives

    @staticmethod
    def _kl_mean_from_params(
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        min_std: float,
    ) -> torch.Tensor:
        prior_std = torch.exp(0.5 * prior_logvar).clamp_min(float(min_std))
        post_std = torch.exp(0.5 * post_logvar).clamp_min(float(min_std))
        prior = torch.distributions.Independent(
            torch.distributions.Normal(prior_mu, prior_std),
            1,
        )
        post = torch.distributions.Independent(
            torch.distributions.Normal(post_mu, post_std),
            1,
        )
        kl = torch.distributions.kl_divergence(post, prior)  # (B,T)
        return kl.mean()

    def _rollout_mixed(
        self,
        initial_state: RSSMState,
        controls: torch.Tensor,
        exogenous: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> Dict[str, object]:
        batch_size, horizon, _ = controls.shape
        state = initial_state

        predictions = torch.empty(batch_size, horizon, self.output_dim, device=controls.device, dtype=controls.dtype)
        dist_loc = torch.empty_like(predictions)
        dist_loc_latent = torch.empty_like(predictions)
        dist_scale = torch.empty_like(predictions)
        kl_terms = torch.empty(batch_size, horizon, device=controls.device, dtype=controls.dtype)
        prior_mu_all = torch.empty(batch_size, horizon, self.latent_dim, device=controls.device, dtype=controls.dtype)
        prior_logvar_all = torch.empty_like(prior_mu_all)
        post_mu_all = torch.empty_like(prior_mu_all)
        post_logvar_all = torch.empty_like(prior_mu_all)
        aux_loc = torch.empty(batch_size, horizon, exogenous.shape[-1], device=controls.device, dtype=controls.dtype)
        aux_scale = torch.empty_like(aux_loc)
        objective_dists = []
        objective_dists_latent = []
        aux_dists = []

        for t in range(horizon):
            c_t = controls[:, t, :]
            x_t = exogenous[:, t, :]
            y_t_raw = targets[:, t, :]
            y_t = self.symlog(y_t_raw) if self.use_symlog else y_t_raw

            y_enc = self.observation_encoder(y_t)
            c_enc = self.control_encoder(c_t)
            x_enc = self.exogenous_encoder(x_t)
            step_out = self.rssm_cell.observe(
                prev_state=state,
                control_embed=c_enc,
                exogenous_embed=x_enc,
                observation_embed=y_enc,
                sample=False,
            )
            h_t = step_out.h
            prior_mu = step_out.prior_mean
            prior_logvar = step_out.prior_logvar
            post_mu = step_out.posterior_mean
            post_logvar = step_out.posterior_logvar
            if post_mu is None or post_logvar is None or step_out.posterior is None:
                raise RuntimeError("Posterior outputs are required in mixed rollout.")

            teacher_mask = (
                torch.rand(batch_size, 1, device=controls.device) < float(teacher_forcing_ratio)
            ).to(dtype=controls.dtype)
            z_prior = prior_mu
            z_post = post_mu
            z_t = teacher_mask * z_post + (1.0 - teacher_mask) * z_prior

            y_dist, y_dist_latent, y_loc, y_scale, y_loc_latent = self._decode_obs(h_t, z_t)
            x_dist, x_loc, x_scale = self._decode_exogenous(h_t, z_t, exogenous.shape[-1])
            kl_t = torch.distributions.kl_divergence(step_out.posterior, step_out.prior)

            predictions[:, t, :] = y_loc
            dist_loc[:, t, :] = y_loc
            dist_loc_latent[:, t, :] = y_loc_latent
            dist_scale[:, t, :] = y_scale
            kl_terms[:, t] = kl_t
            prior_mu_all[:, t, :] = prior_mu
            prior_logvar_all[:, t, :] = prior_logvar
            post_mu_all[:, t, :] = post_mu
            post_logvar_all[:, t, :] = post_logvar
            objective_dists.append(y_dist)
            objective_dists_latent.append(y_dist_latent)
            aux_dists.append(x_dist)
            if x_loc.shape[-1] > 0:
                aux_loc[:, t, :] = x_loc
                aux_scale[:, t, :] = x_scale

            state = RSSMState(h=h_t, z=z_t)

        return {
            "predictions": predictions,
            "dist_loc": dist_loc,
            "dist_loc_latent": dist_loc_latent,
            "dist_scale": dist_scale,
            "dist_df": torch.full_like(dist_scale, fill_value=self.min_df + 0.5),
            "kl_terms": kl_terms,
            "prior_mu": prior_mu_all,
            "prior_logvar": prior_logvar_all,
            "posterior_mu": post_mu_all,
            "posterior_logvar": post_logvar_all,
            "aux_loc": aux_loc,
            "aux_scale": aux_scale,
            "objective_dists": objective_dists,
            "objective_dists_latent": objective_dists_latent,
            "aux_dists": aux_dists,
            "states": [
                {"h": state.h, "z": state.z},
            ],
        }

    def rollout(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if feedback in {"teacher", "mixed"} and targets is None:
            raise ValueError(f"targets required when feedback='{feedback}'")

        controls_w, exo_w, outputs_w = self._warmup_components(warmup_seq, rollout_inputs=rollout_inputs)
        observed_warmup = self.observe(
            controls=controls_w,
            exogenous=exo_w,
            observations=outputs_w,
            initial_state=None,
            sample_posterior=False,
        )
        state0 = observed_warmup["state"]
        assert isinstance(state0, RSSMState)

        controls = rollout_inputs["controls"][:, :horizon, :]
        exogenous = rollout_inputs["exogenous"][:, :horizon, :]

        if feedback == "teacher":
            assert targets is not None
            posterior = self.observe(
                controls=controls,
                exogenous=exogenous,
                observations=targets[:, :horizon, :],
                initial_state=state0,
                sample_posterior=False,
            )
            return {
                "predictions": posterior["predictions"],
                "dist_loc": posterior["dist_loc"],
                "dist_loc_latent": posterior["dist_loc_latent"],
                "dist_scale": posterior["dist_scale"],
                "dist_df": torch.full_like(posterior["dist_scale"], fill_value=self.min_df + 0.5),
                "kl_terms": posterior["kl_terms"],
                "prior_mu": posterior["prior_mu"],
                "prior_logvar": posterior["prior_logvar"],
                "posterior_mu": posterior["posterior_mu"],
                "posterior_logvar": posterior["posterior_logvar"],
                "aux_loc": posterior["aux_loc"],
                "aux_scale": posterior["aux_scale"],
                "objective_dists": posterior.get("objective_dists", []),
                "objective_dists_latent": posterior.get("objective_dists_latent", []),
                "aux_dists": posterior.get("aux_dists", []),
                "states": [{"h": posterior["state"].h, "z": posterior["state"].z}],
            }

        if feedback == "mixed":
            assert targets is not None
            return self._rollout_mixed(
                initial_state=state0,
                controls=controls,
                exogenous=exogenous,
                targets=targets[:, :horizon, :],
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

        imagined = self.imagine(
            initial_state=state0,
            future_controls=controls,
            future_exogenous=exogenous,
            n_steps=horizon,
            n_samples=1,
            sample_latent=False,
        )
        preds = imagined["predictions"]
        assert isinstance(preds, torch.Tensor)
        dist_scale = imagined["dist_scale"]
        dist_loc_latent = imagined["dist_loc_latent"]
        prior_mu = imagined["prior_mu"]
        prior_logvar = imagined["prior_logvar"]
        aux_loc = imagined["aux_loc"]
        aux_scale = imagined["aux_scale"]
        return {
            "predictions": preds,
            "dist_loc": preds,
            "dist_loc_latent": dist_loc_latent,
            "dist_scale": dist_scale,
            "dist_df": torch.full_like(dist_scale, fill_value=self.min_df + 0.5),
            "kl_terms": torch.zeros(preds.shape[0], preds.shape[1], device=preds.device, dtype=preds.dtype),
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "posterior_mu": torch.zeros_like(prior_mu),
            "posterior_logvar": torch.zeros_like(prior_logvar),
            "aux_loc": aux_loc,
            "aux_scale": aux_scale,
            "objective_dists": imagined.get("objective_dists", []),
            "objective_dists_latent": imagined.get("objective_dists_latent", []),
            "aux_dists": imagined.get("aux_dists", []),
            "states": [{"h": imagined["state"].h, "z": imagined["state"].z}],
        }

    @torch.no_grad()
    def rollout_mc(
        self,
        warmup_seq: Dict[str, torch.Tensor],
        rollout_inputs: Dict[str, torch.Tensor],
        horizon: int,
        n_samples: int = 50,
        interval_level: float = 0.90,
    ) -> Dict[str, torch.Tensor]:
        controls_w, exo_w, outputs_w = self._warmup_components(warmup_seq, rollout_inputs=rollout_inputs)
        observed_warmup = self.observe(
            controls=controls_w,
            exogenous=exo_w,
            observations=outputs_w,
            initial_state=None,
            sample_posterior=False,
        )
        state0 = observed_warmup["state"]
        assert isinstance(state0, RSSMState)

        imagined = self.imagine(
            initial_state=state0,
            future_controls=rollout_inputs["controls"][:, :horizon, :],
            future_exogenous=rollout_inputs["exogenous"][:, :horizon, :],
            n_steps=horizon,
            n_samples=n_samples,
            sample_latent=True,
        )
        samples = imagined["samples"]
        assert isinstance(samples, torch.Tensor)

        alpha = max(0.0, min(1.0, float(interval_level)))
        lo_q = (1.0 - alpha) / 2.0
        hi_q = 1.0 - lo_q

        return {
            "samples": samples,
            "mean": samples.mean(dim=0),
            "std": samples.std(dim=0, unbiased=False),
            "lower": torch.quantile(samples, q=lo_q, dim=0),
            "upper": torch.quantile(samples, q=hi_q, dim=0),
            "interval_level": alpha,
        }

    def forward(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        mode: Literal["auto", "observe", "predict"] = "auto",
    ) -> object:
        """Forward entrypoint.

        Modes
        -----
        - ``observe``: run observation/filtering pass and return priors,
          posteriors, decoded distributions and state.
        - ``predict``: keep backward-compatible prediction behavior using
          imagined rollout from initialized state.
        - ``auto``: observe when input is role-mapped dict; otherwise predict.
        """
        if mode not in {"auto", "observe", "predict"}:
            raise ValueError(f"Unknown forward mode: {mode}")

        do_observe = (mode == "observe") or (mode == "auto" and isinstance(x, dict))
        if do_observe:
            if isinstance(x, dict):
                controls = x["control"]
                exogenous = x["exogenous"]
                objectives = x["objective"]
            else:
                controls, exogenous, objectives = self._split_observation_inputs(x)
            return self.observe(
                controls=controls,
                exogenous=exogenous,
                observations=objectives,
                initial_state=None,
                sample_posterior=False,
            )

        if isinstance(x, dict):
            raise ValueError("Predict mode expects tensor input, got dict.")

        state = self.init_state(x)
        batch = x.shape[0]
        cdim = 0 if self.control_dim is None else int(self.control_dim)
        xdim = 0 if self.exogenous_dim is None else int(self.exogenous_dim)
        controls = torch.zeros(batch, self.pred_len, cdim, device=x.device, dtype=x.dtype)
        exogenous = torch.zeros(batch, self.pred_len, xdim, device=x.device, dtype=x.dtype)
        imagined = self.imagine(
            initial_state=RSSMState(h=state["h"], z=state["z"]),
            future_controls=controls,
            future_exogenous=exogenous,
            n_steps=self.pred_len,
            n_samples=1,
            sample_latent=False,
        )
        preds = imagined["predictions"]
        assert isinstance(preds, torch.Tensor)
        return preds
