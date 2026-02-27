"""Recurrent state-space world model (RSSM) with typed encoders.

Design constraints enforced (default configuration):
- Transition depends on control + exogenous only (never objective/target).
- Separate, non-shared encoders for control/exogenous/objective variables.
- Dual latent path: deterministic h_t + stochastic z_t.
- Prior p(z_t|h_t) and posterior q(z_t|h_t,y_t) heads at every step.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
import logging

import torch
import torch.nn as nn

from .base import WorldModelBase
from .decoders import AuxiliaryDecoder, ObjectiveDecoder
from .distributions import diagonal_independent_normal, fast_sample
from .encoders import (
    ControlEncoder,
    ExogenousEncoder,
    ObservationEncoder,
    UniversalSharedEncoder,
    assert_no_shared_encoder_params,
)
from .rssm import RSSMCell, RSSMState

logger = logging.getLogger(__name__)


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
        min_scale: float = 0.5,
        min_std: Optional[float] = None,
        max_std: float = 2.0,
        decoder_min_std: Optional[float] = None,
        decoder_max_std: Optional[float] = None,
        prior_min_std: float = 0.1,
        prior_max_std: float = 1.5,
        posterior_min_std: float = 0.1,
        posterior_max_std: float = 1.5,
        prior_constant_std: Optional[float] = None,
        posterior_constant_std: Optional[float] = None,
        min_df: float = 2.1,
        control_dim: Optional[int] = None,
        exogenous_dim: Optional[int] = None,
        encoder_dim: int = 64,
        decoder_layers: int = 2,
        use_symlog: bool = False,
        use_aux_decoder: bool = True,
        predict_exogenous: bool = True,
        use_dual_path: bool = True,
        use_stochastic_path: bool = True,
        share_encoder_weights: bool = False,
        leak_objective_to_transition: bool = False,
        h_dropout: float = 0.0,
        decoder_hidden: Optional[int] = None,
        allow_objective_leak_for_ablation: bool = False,
        allow_disable_aux_decoder_for_ablation: bool = False,
        allow_shared_encoder_for_ablation: bool = False,
        allow_disable_stochastic_for_ablation: bool = False,
        latent_distribution: str = "gaussian",
        stochastic_groups: int = 32,
        stochastic_classes: int = 32,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_distribution = str(latent_distribution).lower().strip()
        self.stochastic_groups = int(stochastic_groups)
        self.stochastic_classes = int(stochastic_classes)
        if self.latent_distribution == "categorical":
            self.latent_dim = self.stochastic_groups * self.stochastic_classes
        else:
            self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.pred_len = int(pred_len)
        self.min_scale = float(min_scale)
        if decoder_min_std is None:
            raw_min_std = self.min_scale if min_std is None else float(min_std)
        else:
            raw_min_std = float(decoder_min_std)
        dec_max_std_cfg = float(max_std if decoder_max_std is None else decoder_max_std)
        # Decoder std bounds.
        self.min_std = float(max(0.5, raw_min_std))
        self.max_std = float(max(self.min_std, dec_max_std_cfg))
        # Latent prior/posterior std bounds.
        self.prior_min_std = float(max(0.0, float(prior_min_std)))
        self.prior_max_std = float(max(self.prior_min_std, float(prior_max_std)))
        self.posterior_min_std = float(max(0.0, float(posterior_min_std)))
        self.posterior_max_std = float(max(self.posterior_min_std, float(posterior_max_std)))
        self.prior_constant_std = (
            None if prior_constant_std is None else float(max(1e-6, float(prior_constant_std)))
        )
        self.posterior_constant_std = (
            None if posterior_constant_std is None else float(max(1e-6, float(posterior_constant_std)))
        )
        if self.prior_min_std != self.posterior_min_std or self.prior_max_std != self.posterior_max_std:
            raise ValueError(
                "RSSMCell currently shares prior/posterior std bounds. "
                "Use matching prior_* and posterior_* values."
            )
        self.min_latent_std = self.prior_min_std
        self.max_latent_std = self.prior_max_std
        self.min_df = float(min_df)
        self.control_dim = None if control_dim is None else int(control_dim)
        self.exogenous_dim = None if exogenous_dim is None else int(exogenous_dim)
        self.encoder_dim = int(encoder_dim)
        self.decoder_layers = max(1, int(decoder_layers))
        self.use_symlog = bool(use_symlog)
        self.use_aux_decoder = bool(use_aux_decoder)
        self.predict_exogenous = bool(predict_exogenous)
        self.decode_exogenous = bool(self.use_aux_decoder and self.predict_exogenous)
        self.use_dual_path = bool(use_dual_path)
        self.use_stochastic_path = bool(use_stochastic_path)
        self.share_encoder_weights = bool(share_encoder_weights)
        self.leak_objective_to_transition = bool(leak_objective_to_transition)
        self.h_dropout = nn.Dropout(float(max(0.0, min(1.0, h_dropout))))
        self.decoder_hidden = int(decoder_hidden) if decoder_hidden is not None else self.hidden_dim
        self.allow_objective_leak_for_ablation = bool(allow_objective_leak_for_ablation)
        self.allow_disable_aux_decoder_for_ablation = bool(allow_disable_aux_decoder_for_ablation)
        self.allow_shared_encoder_for_ablation = bool(allow_shared_encoder_for_ablation)
        self.allow_disable_stochastic_for_ablation = bool(allow_disable_stochastic_for_ablation)

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
        if self.share_encoder_weights and not self.allow_shared_encoder_for_ablation:
            raise ValueError(
                "Shared encoder weights across variable roles are blocked by default. "
                "Set `allow_shared_encoder_for_ablation=True` only for explicit ablations."
            )
        if (not self.use_stochastic_path) and (not self.allow_disable_stochastic_for_ablation):
            raise ValueError(
                "Disabling stochastic latent path is blocked by default. "
                "Set `allow_disable_stochastic_for_ablation=True` only for explicit ablations."
            )

        # Explicitly independent encoders per variable type (no weight sharing).
        self._shared_encoder = None
        if self.share_encoder_weights:
            shared = UniversalSharedEncoder(
                hidden_dim=self.hidden_dim,
                embed_dim=self.encoder_dim,
            )
            self.control_encoder = shared
            self.exogenous_encoder = shared
            self.observation_encoder = shared
            self._shared_encoder = shared
        else:
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
            max_std=self.max_latent_std,
            prior_constant_std=self.prior_constant_std,
            posterior_constant_std=self.posterior_constant_std,
            use_dual_path=self.use_dual_path,
            use_stochastic_path=self.use_stochastic_path,
            leak_objective_to_transition=self.leak_objective_to_transition,
            latent_distribution=self.latent_distribution,
            stochastic_groups=self.stochastic_groups,
            stochastic_classes=self.stochastic_classes,
        )
        logger.info("Decoder min_std: %s", self.min_std)

        # Decoders from full latent state [h_t, z_t]. Dropout on h before decoder to weaken deterministic path.
        self.obs_decoder = ObjectiveDecoder(
            in_dim=self.hidden_dim + self.latent_dim,
            out_dim=self.output_dim,
            hidden_dim=self.decoder_hidden,
            num_layers=self.decoder_layers,
            min_std=self.min_std,
            max_std=self.max_std,
        )
        self._aux_decoder_hidden: Optional[nn.Sequential]
        if self.decode_exogenous:
            self._aux_decoder_hidden = nn.Sequential(
                nn.Linear(self.hidden_dim + self.latent_dim, self.decoder_hidden),
                nn.ELU(),
            )
        else:
            self._aux_decoder_hidden = None
        self._aux_decoder_heads = nn.ModuleDict()

    def _assert_encoder_independence(self) -> None:
        assert_no_shared_encoder_params(
            self.control_encoder,
            self.exogenous_encoder,
            self.observation_encoder,
        )

    def _encode_control(self, x: torch.Tensor) -> torch.Tensor:
        return self.control_encoder(x)

    def _encode_exogenous(self, x: torch.Tensor) -> torch.Tensor:
        return self.exogenous_encoder(x)

    def _encode_observation(self, x: torch.Tensor) -> torch.Tensor:
        return self.observation_encoder(x)

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
        if exogenous_dim <= 0 or not self.decode_exogenous:
            return None
        key = str(int(exogenous_dim))
        if key not in self._aux_decoder_heads:
            head = AuxiliaryDecoder(
                in_dim=self.decoder_hidden,
                out_dim=int(exogenous_dim),
                hidden_dim=self.decoder_hidden,
                num_layers=max(1, self.decoder_layers),
                min_std=self.min_std,
                max_std=self.max_std,
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

    def _decoder_latent(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Build decoder latent with dropout on h only, then concatenate with z."""
        # Information bottleneck: apply dropout only to deterministic state h.
        # Stochastic state z is never dropped here.
        return torch.cat([self.h_dropout(deter), stoch], dim=-1)

    def _decode_obs(
        self, h_t: torch.Tensor, z_t: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.distributions.Distribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._decoder_latent(h_t, z_t)
        dist_latent, loc_latent, scale = self.obs_decoder(
            latent,
            min_scale=self.min_std,
            max_scale=self.max_std,
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
        if not self.decode_exogenous:
            bsz = h_t.shape[0]
            empty = torch.zeros(bsz, 0, device=h_t.device, dtype=h_t.dtype)
            return None, empty, empty
        if self._aux_decoder_hidden is None:
            raise RuntimeError("Aux decoder hidden stack is missing while exogenous decoding is enabled.")
        hidden = self._aux_decoder_hidden(self._decoder_latent(h_t, z_t))
        head = self._get_aux_head(
            exogenous_dim,
            device=h_t.device,
            dtype=h_t.dtype,
        )
        assert head is not None
        dist, loc, scale = head(hidden, min_scale=self.min_std, max_scale=self.max_std)
        return dist, loc, scale

    def _transition_step(
        self,
        state: RSSMState,
        control_t: torch.Tensor,
        exo_t: torch.Tensor,
    ) -> torch.Tensor:
        c_enc = self._encode_control(control_t)
        x_enc = self._encode_exogenous(exo_t)
        h_t = self.rssm_cell.transition(
            prev_state=state,
            control_embed=c_enc,
            exogenous_embed=x_enc,
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
        c_enc = self._encode_control(control_t)
        x_enc = self._encode_exogenous(exo_t)
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

    @staticmethod
    def _kl_diagonal_normal(
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p) for diagonal Normal, computed directly from parameters.

        Avoids constructing torch.distributions objects in the inner loop,
        which eliminates significant Python/CUDA overhead per time step.
        """
        var_ratio = (post_logvar - prior_logvar).exp()
        delta = prior_mu - post_mu
        return 0.5 * (prior_logvar - post_logvar + var_ratio + delta.pow(2) / prior_logvar.exp() - 1.0)

    @staticmethod
    def _kl_categorical_logits(
        post_logits: torch.Tensor,
        prior_logits: torch.Tensor,
        groups: int,
        classes: int,
    ) -> torch.Tensor:
        post_l = post_logits.view(*post_logits.shape[:-1], int(groups), int(classes))
        prior_l = prior_logits.view(*prior_logits.shape[:-1], int(groups), int(classes))
        post_log_probs = torch.log_softmax(post_l, dim=-1)
        prior_log_probs = torch.log_softmax(prior_l, dim=-1)
        post_probs = torch.exp(post_log_probs)
        return torch.sum(post_probs * (post_log_probs - prior_log_probs), dim=-1)

    def _batch_decode_obs(
        self, deter: torch.Tensor, stoch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-decode observation outputs from pre-computed (B,T,H) and (B,T,Z).

        Returns (loc, loc_latent, scale) each with shape (B,T,output_dim).
        Processes all time steps in a single forward pass through the decoder MLP.
        """
        B, T, _ = deter.shape
        latent_flat = self._decoder_latent(deter, stoch).reshape(B * T, -1)
        _, loc_latent_flat, scale_flat = self.obs_decoder(
            latent_flat, min_scale=self.min_std, max_scale=self.max_std,
        )
        loc_latent = loc_latent_flat.reshape(B, T, self.output_dim)
        scale = scale_flat.reshape(B, T, self.output_dim)
        loc = self.symexp(loc_latent) if self.use_symlog else loc_latent
        return loc, loc_latent, scale

    def _batch_decode_exogenous(
        self, deter: torch.Tensor, stoch: torch.Tensor, exo_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-decode exogenous outputs. Returns (loc, scale) each (B,T,exo_dim)."""
        B, T, _ = deter.shape
        device, dtype = deter.device, deter.dtype
        if exo_dim <= 0 or not self.decode_exogenous:
            loc = torch.zeros(B, T, 0, device=device, dtype=dtype)
            scale = torch.zeros_like(loc)
            return loc, scale
        latent_flat = self._decoder_latent(deter, stoch).reshape(B * T, -1)
        if self._aux_decoder_hidden is None:
            raise RuntimeError("Aux decoder hidden stack is missing while exogenous decoding is enabled.")
        hidden_flat = self._aux_decoder_hidden(latent_flat)
        head = self._get_aux_head(exo_dim, device=device, dtype=dtype)
        assert head is not None
        _, loc_flat, scale_flat = head(hidden_flat, min_scale=self.min_std, max_scale=self.max_std)
        return loc_flat.reshape(B, T, exo_dim), scale_flat.reshape(B, T, exo_dim)

    @staticmethod
    def _build_dist_list(
        loc: torch.Tensor, scale: torch.Tensor
    ) -> list:
        """Build per-step distribution list from batched (B,T,D) loc/scale."""
        T = loc.shape[1]
        return [diagonal_independent_normal(loc=loc[:, t], scale=scale[:, t]) for t in range(T)]

    def observe(
        self,
        controls: torch.Tensor,
        exogenous: torch.Tensor,
        observations: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
        sample_posterior: bool = False,
    ) -> Dict[str, object]:
        """Run posterior filtering over a sequence.

        Encoding and decoding are batched over the full time dimension to
        minimize CUDA kernel launches.  Only the sequential RSSM cell
        (GRU transition + prior/posterior heads) remains in the per-step loop.
        """
        if controls.shape[:2] != exogenous.shape[:2] or controls.shape[:2] != observations.shape[:2]:
            raise ValueError("controls/exogenous/observations must have matching (batch,time) dimensions")

        batch_size, horizon, _ = controls.shape
        device, dtype = controls.device, controls.dtype
        state = initial_state if initial_state is not None else self._zero_state(batch_size, device, dtype)
        exo_dim = exogenous.shape[-1]
        BT = batch_size * horizon

        # --- PRE-ENCODE: 3 batched forward passes instead of 3*T per-step calls ---
        obs_for_enc = self.symlog(observations) if self.use_symlog else observations
        c_enc_all = self._encode_control(controls.reshape(BT, -1)).reshape(batch_size, horizon, -1)
        x_enc_all = self._encode_exogenous(exogenous.reshape(BT, -1)).reshape(batch_size, horizon, -1)
        y_enc_all = self._encode_observation(obs_for_enc.reshape(BT, -1)).reshape(batch_size, horizon, -1)

        # Pre-allocate outputs for the sequential core
        out_deter = torch.empty(batch_size, horizon, self.hidden_dim, device=device, dtype=dtype)
        out_stoch = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_mu = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_lv = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        out_post_mu = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        out_post_lv = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_logits = (
            torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
            if self.latent_distribution == "categorical"
            else None
        )
        out_post_logits = (
            torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
            if self.latent_distribution == "categorical"
            else None
        )

        # --- SEQUENTIAL CORE: only RSSM cell (transition + prior + posterior + sample) ---
        for t in range(horizon):
            step_out = self.rssm_cell.observe(
                prev_state=state,
                control_embed=c_enc_all[:, t],
                exogenous_embed=x_enc_all[:, t],
                observation_embed=y_enc_all[:, t],
                sample=sample_posterior,
            )
            post_mu = step_out.posterior_mean
            post_logvar = step_out.posterior_logvar
            if post_mu is None or post_logvar is None or step_out.posterior is None:
                raise RuntimeError("Posterior outputs are required in observe().")

            out_deter[:, t] = step_out.state.h
            out_stoch[:, t] = step_out.state.z
            out_prior_mu[:, t] = step_out.prior_mean
            out_prior_lv[:, t] = step_out.prior_logvar
            out_post_mu[:, t] = post_mu
            out_post_lv[:, t] = post_logvar
            if out_prior_logits is not None and out_post_logits is not None:
                if step_out.prior_logits is None or step_out.posterior_logits is None:
                    raise RuntimeError("Categorical latent mode requires prior/posterior logits.")
                out_prior_logits[:, t] = step_out.prior_logits
                out_post_logits[:, t] = step_out.posterior_logits
            state = step_out.state

        # --- POST-DECODE: 2 batched forward passes instead of 2*T per-step calls ---
        out_recon_loc, out_recon_loc_lat, out_recon_scale = self._batch_decode_obs(out_deter, out_stoch)
        out_aux_loc, out_aux_scale = self._batch_decode_exogenous(out_deter, out_stoch, exo_dim)

        # Vectorized KL over all time steps at once
        if out_prior_logits is not None and out_post_logits is not None:
            out_kl = self._kl_categorical_logits(
                out_post_logits,
                out_prior_logits,
                self.stochastic_groups,
                self.stochastic_classes,
            ).sum(dim=-1)
        else:
            out_kl = self._kl_diagonal_normal(out_post_mu, out_post_lv, out_prior_mu, out_prior_lv).sum(dim=-1)

        # Build per-step distribution lists from batched tensors (cheap Python objects)
        recon_dists = self._build_dist_list(out_recon_loc, out_recon_scale)
        recon_dists_latent = self._build_dist_list(out_recon_loc_lat, out_recon_scale)
        if exo_dim > 0 and self.decode_exogenous:
            aux_dists = self._build_dist_list(out_aux_loc, out_aux_scale)
        else:
            aux_dists: list = []

        return {
            "deter": out_deter,
            "stoch": out_stoch,
            "prior_mu": out_prior_mu,
            "prior_logvar": out_prior_lv,
            "posterior_mu": out_post_mu,
            "posterior_logvar": out_post_lv,
            "prior_logits": out_prior_logits,
            "posterior_logits": out_post_logits,
            "kl_terms": out_kl,
            "dist_loc": out_recon_loc,
            "dist_loc_latent": out_recon_loc_lat,
            "dist_scale": out_recon_scale,
            "aux_loc": out_aux_loc,
            "aux_scale": out_aux_scale,
            "objective_dists": recon_dists,
            "objective_dists_latent": recon_dists_latent,
            "aux_dists": aux_dists,
            "state": state,
            "predictions": out_recon_loc,
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
        """Roll forward using prior dynamics only (no target leakage).

        Encoding and decoding are batched over the full time dimension.
        """
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
        exo_dim = exogenous.shape[-1]

        if n_samples > 1:
            controls = controls.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * batch_size, horizon, -1)
            exogenous = exogenous.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * batch_size, horizon, -1)
            state = RSSMState(
                h=initial_state.h.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * batch_size, self.hidden_dim),
                z=initial_state.z.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * batch_size, self.latent_dim),
            )
            eff_batch = n_samples * batch_size
        else:
            state = RSSMState(h=initial_state.h, z=initial_state.z)
            eff_batch = batch_size

        device, dtype = controls.device, controls.dtype
        BT = eff_batch * horizon

        # --- PRE-ENCODE: 2 batched forward passes instead of 2*T ---
        c_enc_all = self._encode_control(controls.reshape(BT, -1)).reshape(eff_batch, horizon, -1)
        x_enc_all = self._encode_exogenous(exogenous.reshape(BT, -1)).reshape(eff_batch, horizon, -1)

        out_deter = torch.empty(eff_batch, horizon, self.hidden_dim, device=device, dtype=dtype)
        out_stoch = torch.empty(eff_batch, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_mu = torch.empty(eff_batch, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_lv = torch.empty(eff_batch, horizon, self.latent_dim, device=device, dtype=dtype)
        out_prior_logits = (
            torch.empty(eff_batch, horizon, self.latent_dim, device=device, dtype=dtype)
            if self.latent_distribution == "categorical"
            else None
        )

        # --- SEQUENTIAL CORE: only RSSM cell ---
        for t in range(horizon):
            step_out = self.rssm_cell.imagine(
                prev_state=state,
                control_embed=c_enc_all[:, t],
                exogenous_embed=x_enc_all[:, t],
                sample=sample_latent,
            )
            out_deter[:, t] = step_out.state.h
            out_stoch[:, t] = step_out.state.z
            out_prior_mu[:, t] = step_out.prior_mean
            out_prior_lv[:, t] = step_out.prior_logvar
            if out_prior_logits is not None:
                if step_out.prior_logits is None:
                    raise RuntimeError("Categorical latent mode requires prior logits.")
                out_prior_logits[:, t] = step_out.prior_logits
            state = step_out.state

        # --- POST-DECODE: 2 batched forward passes instead of 2*T ---
        out_pred_loc, out_pred_loc_lat, out_pred_scale = self._batch_decode_obs(out_deter, out_stoch)
        out_aux_loc, out_aux_scale = self._batch_decode_exogenous(out_deter, out_stoch, exo_dim)

        pred_dists = self._build_dist_list(out_pred_loc, out_pred_scale)
        pred_dists_latent = self._build_dist_list(out_pred_loc_lat, out_pred_scale)
        if exo_dim > 0 and self.decode_exogenous:
            aux_dists = self._build_dist_list(out_aux_loc, out_aux_scale)
        else:
            aux_dists: list = []

        if n_samples > 1:
            pred_samples = out_pred_loc.reshape(n_samples, batch_size, horizon, self.output_dim)
            pred_scale_samples = out_pred_scale.reshape(n_samples, batch_size, horizon, self.output_dim)
            pred_latent_samples = out_pred_loc_lat.reshape(n_samples, batch_size, horizon, self.output_dim)
            aux_out_dim = out_aux_loc.shape[-1]
            aux_loc_samples = out_aux_loc.reshape(n_samples, batch_size, horizon, aux_out_dim)
            aux_scale_samples = out_aux_scale.reshape(n_samples, batch_size, horizon, aux_out_dim)
            prior_mu_samples = out_prior_mu.reshape(n_samples, batch_size, horizon, self.latent_dim)
            prior_logvar_samples = out_prior_lv.reshape(n_samples, batch_size, horizon, self.latent_dim)
            prior_logits_samples = (
                out_prior_logits.reshape(n_samples, batch_size, horizon, self.latent_dim)
                if out_prior_logits is not None
                else None
            )
            deter_samples = out_deter.reshape(n_samples, batch_size, horizon, self.hidden_dim)
            stoch_samples = out_stoch.reshape(n_samples, batch_size, horizon, self.latent_dim)

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
                "prior_logits": (prior_logits_samples.mean(dim=0) if prior_logits_samples is not None else None),
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
            "predictions": out_pred_loc,
            "dist_scale": out_pred_scale,
            "dist_loc_latent": out_pred_loc_lat,
            "aux_loc": out_aux_loc,
            "aux_scale": out_aux_scale,
            "prior_mu": out_prior_mu,
            "prior_logvar": out_prior_lv,
            "prior_logits": out_prior_logits,
            "deter": out_deter,
            "stoch": out_stoch,
            "objective_dists": pred_dists,
            "objective_dists_latent": pred_dists_latent,
            "aux_dists": aux_dists,
            "state": state,
        }

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
        use_free_bits: bool = False,
        kl_free_bits: float = 0.0,
        kl_balance: float = 0.8,
        use_kl_balancing: bool = False,
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
            prior_logits=observed.get("prior_logits"),  # type: ignore[arg-type]
            posterior_logits=observed.get("posterior_logits"),  # type: ignore[arg-type]
            groups=self.stochastic_groups,
            classes=self.stochastic_classes,
            use_free_bits=use_free_bits,
            kl_free_bits=kl_free_bits,
            kl_balance=kl_balance,
            use_kl_balancing=use_kl_balancing,
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
            if allow_none and len(dist_list) == 0:
                return torch.zeros((), dtype=targets.dtype, device=targets.device)
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
        prior_logits: Optional[torch.Tensor] = None,
        posterior_logits: Optional[torch.Tensor] = None,
        groups: int = 32,
        classes: int = 32,
        use_free_bits: bool = False,
        kl_free_bits: float = 0.0,
        kl_balance: float = 0.8,
        use_kl_balancing: bool = False,
    ) -> torch.Tensor:
        if prior_logits is not None and posterior_logits is not None:
            # Local import avoids potential circular import at module load time.
            from ..training.losses import fast_kl_balancing_loss

            post_logits_4d = posterior_logits.view(
                *posterior_logits.shape[:-1], int(groups), int(classes)
            )
            prior_logits_4d = prior_logits.view(
                *prior_logits.shape[:-1], int(groups), int(classes)
            )
            if bool(use_kl_balancing) or (bool(use_free_bits) and float(kl_free_bits) > 0.0):
                kl_groups = fast_kl_balancing_loss(
                    post_logits_4d,
                    prior_logits_4d,
                    alpha=float(kl_balance),
                    free_nats=float(kl_free_bits),
                    use_kl_balancing=bool(use_kl_balancing),
                    use_free_bits=bool(use_free_bits),
                )
            else:
                kl_groups = LatentSSMWorldModel._kl_categorical_logits(
                    posterior_logits,
                    prior_logits,
                    groups,
                    classes,
                )
            return kl_groups.sum(dim=-1).mean()
        min_logvar = 2.0 * torch.tensor(float(min_std), device=prior_mu.device, dtype=prior_mu.dtype).log()
        p_lv = prior_logvar.clamp_min(min_logvar.item())
        q_lv = post_logvar.clamp_min(min_logvar.item())
        var_ratio = (q_lv - p_lv).exp()
        delta = prior_mu - post_mu
        kl_elem = 0.5 * (p_lv - q_lv + var_ratio + delta.pow(2) / p_lv.exp() - 1.0)
        kl_steps = kl_elem.sum(dim=-1)
        if bool(use_free_bits) and float(kl_free_bits) > 0.0:
            kl_steps = torch.clamp(kl_steps, min=float(kl_free_bits))
        return kl_steps.mean()

    def _rollout_mixed(
        self,
        initial_state: RSSMState,
        controls: torch.Tensor,
        exogenous: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> Dict[str, object]:
        batch_size, horizon, _ = controls.shape
        device, dtype = controls.device, controls.dtype
        state = initial_state
        exo_dim = exogenous.shape[-1]
        BT = batch_size * horizon

        # --- PRE-ENCODE: batched over full time dimension ---
        obs_for_enc = self.symlog(targets) if self.use_symlog else targets
        c_enc_all = self._encode_control(controls.reshape(BT, -1)).reshape(batch_size, horizon, -1)
        x_enc_all = self._encode_exogenous(exogenous.reshape(BT, -1)).reshape(batch_size, horizon, -1)
        y_enc_all = self._encode_observation(obs_for_enc.reshape(BT, -1)).reshape(batch_size, horizon, -1)

        predictions = torch.empty(batch_size, horizon, self.output_dim, device=device, dtype=dtype)
        dist_loc = torch.empty_like(predictions)
        dist_loc_latent = torch.empty_like(predictions)
        dist_scale = torch.empty_like(predictions)
        kl_terms = torch.empty(batch_size, horizon, device=device, dtype=dtype)
        prior_mu_all = torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
        prior_logvar_all = torch.empty_like(prior_mu_all)
        post_mu_all = torch.empty_like(prior_mu_all)
        post_logvar_all = torch.empty_like(prior_mu_all)
        prior_logits_all = (
            torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
            if self.latent_distribution == "categorical"
            else None
        )
        post_logits_all = (
            torch.empty(batch_size, horizon, self.latent_dim, device=device, dtype=dtype)
            if self.latent_distribution == "categorical"
            else None
        )
        aux_out_dim = exo_dim if self.decode_exogenous else 0
        aux_loc = torch.empty(batch_size, horizon, aux_out_dim, device=device, dtype=dtype)
        aux_scale = torch.empty_like(aux_loc)
        objective_dists = []
        objective_dists_latent = []
        aux_dists = []

        for t in range(horizon):
            step_out = self.rssm_cell.observe(
                prev_state=state,
                control_embed=c_enc_all[:, t],
                exogenous_embed=x_enc_all[:, t],
                observation_embed=y_enc_all[:, t],
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
                torch.rand(batch_size, 1, device=device) < float(teacher_forcing_ratio)
            ).to(dtype=dtype)
            if self.latent_distribution == "categorical":
                if step_out.prior_logits is None or step_out.posterior_logits is None:
                    raise RuntimeError("Categorical latent mode requires prior/posterior logits.")
                prior_s = fast_sample(
                    step_out.prior_logits.view(
                        batch_size,
                        self.stochastic_groups,
                        self.stochastic_classes,
                    )
                )
                post_s = fast_sample(
                    step_out.posterior_logits.view(
                        batch_size,
                        self.stochastic_groups,
                        self.stochastic_classes,
                    )
                )
                z_t = teacher_mask * post_s + (1.0 - teacher_mask) * prior_s
            else:
                z_t = teacher_mask * post_mu + (1.0 - teacher_mask) * prior_mu

            y_dist, y_dist_latent, y_loc, y_scale, y_loc_latent = self._decode_obs(h_t, z_t)
            x_dist, x_loc, x_scale = self._decode_exogenous(h_t, z_t, exo_dim)
            if self.latent_distribution == "categorical":
                if step_out.prior_logits is None or step_out.posterior_logits is None:
                    raise RuntimeError("Categorical latent mode requires prior/posterior logits.")
                kl_t = self._kl_categorical_logits(
                    step_out.posterior_logits,
                    step_out.prior_logits,
                    self.stochastic_groups,
                    self.stochastic_classes,
                )
                if prior_logits_all is not None and post_logits_all is not None:
                    prior_logits_all[:, t, :] = step_out.prior_logits
                    post_logits_all[:, t, :] = step_out.posterior_logits
            else:
                kl_t = self._kl_diagonal_normal(post_mu, post_logvar, prior_mu, prior_logvar)

            predictions[:, t, :] = y_loc
            dist_loc[:, t, :] = y_loc
            dist_loc_latent[:, t, :] = y_loc_latent
            dist_scale[:, t, :] = y_scale
            kl_terms[:, t] = kl_t.sum(dim=-1)
            prior_mu_all[:, t, :] = prior_mu
            prior_logvar_all[:, t, :] = prior_logvar
            post_mu_all[:, t, :] = post_mu
            post_logvar_all[:, t, :] = post_logvar
            objective_dists.append(y_dist)
            objective_dists_latent.append(y_dist_latent)
            if x_dist is not None:
                aux_dists.append(x_dist)
            if aux_out_dim > 0:
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
            "prior_logits": prior_logits_all,
            "posterior_logits": post_logits_all,
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
                sample_posterior=(self.latent_distribution == "categorical"),
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
                "prior_logits": posterior.get("prior_logits"),
                "posterior_logits": posterior.get("posterior_logits"),
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
        prior_logits = imagined.get("prior_logits")
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
            "prior_logits": prior_logits,
            "posterior_logits": (torch.zeros_like(prior_logits) if torch.is_tensor(prior_logits) else None),
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
