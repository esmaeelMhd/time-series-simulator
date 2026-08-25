"""LightningModule wrapper for RSSM world model training."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .rollout import get_rollout_schedule

try:
    import pytorch_lightning as pl  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "WorldModelLightningModule requires pytorch-lightning. "
        "Install with: pip install pytorch-lightning"
    ) from exc

logger = logging.getLogger(__name__)


class WorldModelLightningModule(pl.LightningModule):  # type: ignore[misc]
    """Lightning wrapper with RSSM combined-loss support and fallback MSE mode."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-6,
        scheduler_warmup_steps: int = 1000,
        scheduler_min_ratio: float = 0.01,
        grad_clip_norm: float = 100.0,
        probabilistic_cfg: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler_warmup_steps = int(max(1, scheduler_warmup_steps))
        self.scheduler_min_ratio = float(max(0.0, min(1.0, scheduler_min_ratio)))
        self.grad_clip_norm = float(max(0.0, grad_clip_norm))

        p = dict(probabilistic_cfg or {})
        self.recon_weight = float(p.get("recon_weight", p.get("elbo_weight", 1.0)))
        self.kl_weight = float(p.get("kl_weight", 1.0))
        self.aux_weight = float(p.get("aux_weight", p.get("rollout_mse_weight", 1.0)))
        self.rollout_weight = float(p.get("rollout_weight", 0.0))
        self.rollout_dtw_weight = float(p.get("rollout_dtw_weight", 0.0))
        self.rollout_dtw_gamma = float(p.get("rollout_dtw_gamma", 0.1))
        self.rollout_warmup_fraction = float(p.get("rollout_warmup_fraction", 0.30))
        self.rollout_max_horizon = int(max(0, p.get("rollout_max_horizon", 0)))
        self.rollout_start_epoch = p.get("rollout_start_epoch")
        self.rollout_full_epoch = p.get("rollout_full_epoch")
        self.validation_rollout_horizon = int(
            max(1, p.get("validation_rollout_horizon", p.get("eval_rollout_horizon", 30)))
        )
        self.min_context = int(max(1, p.get("min_context", 16)))
        self.use_free_bits = bool(p.get("use_free_bits", False))
        self.kl_free_bits = float(max(0.0, p.get("kl_free_bits", 0.0)))
        self.kl_balance = float(p.get("kl_balance", 0.8))
        self.use_kl_balancing = bool(p.get("use_kl_balancing", False))
        self._log_aux_metrics = self.aux_weight > 0.0
        self._log_rollout_metrics = self.rollout_weight > 0.0
        self._log_rollout_dtw_metric = self._log_rollout_metrics and self.rollout_dtw_weight > 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            if "predictions" in out:
                return out["predictions"]
            if "mean" in out:
                return out["mean"]
        if torch.is_tensor(out):
            return out
        raise TypeError("Wrapped model output is not compatible with Lightning wrapper.")

    @staticmethod
    def _safe_metric(value: torch.Tensor) -> torch.Tensor:
        """Ensure logged metrics are finite to avoid logger backend JSON failures."""
        return torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def _gaussian_crps(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Closed-form CRPS for Gaussian outputs, averaged over batch/time/dim."""
        sigma = scale.clamp_min(1e-6)
        z = (target - mean) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - (1.0 / math.sqrt(math.pi)))
        return crps.mean()

    def _rollout_schedule(self, seq_len: int) -> tuple[int, int, float]:
        """(horizon, context_len, ramp_factor) from the shared scheduler."""
        total_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1) if self.trainer is not None else 1
        ep = int(self.current_epoch) + 1
        cfg: Dict[str, Any] = {
            "epochs": total_epochs,
            "seq_len": int(seq_len),
            "min_context": self.min_context,
            "rollout_max_horizon": self.rollout_max_horizon,
            "rollout_warmup_fraction": self.rollout_warmup_fraction,
        }
        if self.rollout_start_epoch is not None:
            cfg["rollout_start_epoch"] = int(self.rollout_start_epoch)
        if self.rollout_full_epoch is not None:
            cfg["rollout_full_epoch"] = int(self.rollout_full_epoch)
        return get_rollout_schedule(epoch=ep, cfg=cfg)

    def _compute_rssm_combined_losses(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        stage: str,
        compute_rollout_dtw: bool,
        batch_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        objective = batch["objective"]
        seq_len = int(objective.shape[1])
        if stage == "train":
            horizon, context_len, ramp = self._rollout_schedule(seq_len=seq_len)
            scheduled_horizon = int(horizon)
            effective_rollout_weight = float(self.rollout_weight) * float(ramp)
        elif stage == "val":
            # Force fixed validation horizon (default: 30), independent of curriculum/ramp.
            desired_h = int(self.validation_rollout_horizon)
            scheduled_horizon = int(desired_h)
            max_possible = max(0, int(seq_len) - int(self.min_context))
            horizon = min(int(desired_h), int(max_possible))
            if horizon <= 0:
                context_len = max(1, int(seq_len) // 2)
                horizon = max(0, int(seq_len) - int(context_len))
            else:
                context_len = max(int(self.min_context), int(seq_len) - int(horizon))
                if context_len <= 0:
                    context_len = max(1, int(seq_len) // 2)
                    horizon = max(0, int(seq_len) - int(context_len))
            ramp = 1.0 if horizon > 0 else 0.0
            effective_rollout_weight = float(self.rollout_weight)
        else:
            # Test: evaluate open-loop rollout with configured horizon cap.
            max_possible = max(0, int(seq_len) - int(self.min_context))
            eval_cap = int(self.rollout_max_horizon) if int(self.rollout_max_horizon) > 0 else int(max_possible)
            scheduled_horizon = int(eval_cap)
            horizon = min(int(eval_cap), int(max_possible))
            context_len = max(int(self.min_context), int(seq_len) - int(horizon))
            horizon = max(0, int(seq_len) - int(context_len))
            ramp = 1.0 if horizon > 0 else 0.0
            effective_rollout_weight = float(self.rollout_weight)

        out = self.model.imagine_rollout_with_loss(
            batch=batch,  # type: ignore[arg-type]
            context_len=context_len,
            horizon=horizon,
            sample_prior=False,
            compute_rollout_dtw=bool(compute_rollout_dtw and self.rollout_dtw_weight > 0.0),
            rollout_dtw_gamma=float(self.rollout_dtw_gamma),
            use_free_bits=self.use_free_bits,
            kl_free_bits=self.kl_free_bits,
            kl_balance=self.kl_balance,
            use_kl_balancing=self.use_kl_balancing,
        )
        obs_recon = out["obs_recon_nll"]
        obs_kl = out["obs_kl"]
        obs_aux = out["obs_aux_nll"]
        obs = out.get("observed")
        loss_std = self.recon_weight * obs_recon + self.kl_weight * obs_kl + self.aux_weight * obs_aux

        rollout_nll = torch.zeros((), device=loss_std.device, dtype=loss_std.dtype)
        rollout_dtw = torch.zeros_like(rollout_nll)
        rollout_total = torch.zeros_like(rollout_nll)
        if horizon > 0 and effective_rollout_weight > 0.0:
            rollout_nll = out["rollout_nll"]
            rollout_dtw = out["rollout_dtw"]
            rollout_total = rollout_nll + self.rollout_dtw_weight * rollout_dtw
        loss_total = loss_std + effective_rollout_weight * rollout_total

        open_loop_crps = torch.zeros_like(loss_total)
        imagined = out.get("imagined")
        pred_mean_candidate = None
        pred_scale_candidate = None
        if isinstance(imagined, Mapping):
            pred_mean_candidate = imagined.get("dist_loc", imagined.get("predictions", imagined.get("mean")))
            pred_scale_candidate = imagined.get("dist_scale")
        if horizon > 0 and torch.is_tensor(pred_mean_candidate):
            pred_mean = pred_mean_candidate
            pred_scale = pred_scale_candidate if torch.is_tensor(pred_scale_candidate) else None
            target_y = objective[:, -horizon:, :]

            # Handle single-output squeeze edge-cases by restoring feature dim.
            if pred_mean.ndim == 2:
                pred_mean = pred_mean.unsqueeze(-1)
            if pred_scale is not None and pred_scale.ndim == 2:
                pred_scale = pred_scale.unsqueeze(-1)
            if target_y.ndim == 2:
                target_y = target_y.unsqueeze(-1)

            if pred_mean.ndim == 3 and target_y.ndim == 3:
                time_n = min(int(pred_mean.shape[1]), int(target_y.shape[1]))
                feat_n = min(int(pred_mean.shape[2]), int(target_y.shape[2]))
                if pred_scale is not None and pred_scale.ndim == 3:
                    time_n = min(time_n, int(pred_scale.shape[1]))
                    feat_n = min(feat_n, int(pred_scale.shape[2]))
                if time_n > 0 and feat_n > 0:
                    pred_mean = pred_mean[:, :time_n, :feat_n]
                    target_y = target_y[:, :time_n, :feat_n]
                    if pred_scale is not None and pred_scale.ndim == 3:
                        pred_scale = pred_scale[:, :time_n, :feat_n]
                    # Prefer Gaussian CRPS when scale is valid; otherwise fallback to MAE proxy.
                    used_gaussian = False
                    if pred_scale is not None and torch.all(torch.isfinite(pred_scale)):
                        crps_gauss = self._gaussian_crps(mean=pred_mean, scale=pred_scale, target=target_y)
                        if torch.isfinite(crps_gauss):
                            open_loop_crps = crps_gauss
                            used_gaussian = True
                    if not used_gaussian:
                        open_loop_crps = torch.mean(torch.abs(pred_mean - target_y))

        # Health-check diagnostics (used by TrainingHealthCheck callback).
        kl_raw = obs_kl.detach()
        kl_mean = obs_kl.detach()
        kl_active = torch.zeros_like(loss_total)
        dec_std_mean = torch.zeros_like(loss_total)
        dec_std_min = torch.zeros_like(loss_total)
        prior_std_mean = torch.zeros_like(loss_total)
        prior_std_max = torch.zeros_like(loss_total)
        post_std_mean = torch.zeros_like(loss_total)
        post_std_max = torch.zeros_like(loss_total)
        if isinstance(obs, Mapping):
            prior_mu = obs.get("prior_mu")
            prior_logvar = obs.get("prior_logvar")
            post_mu = obs.get("posterior_mu")
            post_logvar = obs.get("posterior_logvar")
            dist_scale = obs.get("dist_scale")
            if (
                torch.is_tensor(prior_mu)
                and torch.is_tensor(prior_logvar)
                and torch.is_tensor(post_mu)
                and torch.is_tensor(post_logvar)
            ):
                var_ratio = (post_logvar - prior_logvar).exp()
                delta = prior_mu - post_mu
                raw_kl_elem = 0.5 * (
                    prior_logvar - post_logvar + var_ratio + delta.pow(2) / prior_logvar.exp() - 1.0
                )
                kl_raw = raw_kl_elem.sum(dim=-1).mean().detach()
                kl_mean = kl_raw
                kl_per_dim = raw_kl_elem.mean(dim=(0, 1))
                kl_active = (kl_per_dim > 0.1).sum().to(dtype=loss_total.dtype)
                prior_std = torch.exp(0.5 * prior_logvar)
                post_std = torch.exp(0.5 * post_logvar)
                prior_std_mean = prior_std.mean().detach().to(dtype=loss_total.dtype)
                prior_std_max = prior_std.max().detach().to(dtype=loss_total.dtype)
                post_std_mean = post_std.mean().detach().to(dtype=loss_total.dtype)
                post_std_max = post_std.max().detach().to(dtype=loss_total.dtype)
            if torch.is_tensor(dist_scale):
                dec_std_mean = dist_scale.mean().detach().to(dtype=loss_total.dtype)
                dec_std_min = dist_scale.min().detach().to(dtype=loss_total.dtype)

        return {
            "loss_total": self._safe_metric(loss_total),
            "loss_std": self._safe_metric(loss_std),
            "recon": self._safe_metric(obs_recon),
            "recon_nll": self._safe_metric(obs_recon),
            "kl": self._safe_metric(obs_kl),
            "kl_mean": self._safe_metric(kl_mean.to(dtype=loss_total.dtype)),
            "kl_raw": self._safe_metric(kl_raw.to(dtype=loss_total.dtype)),
            "kl_active": self._safe_metric(kl_active),
            "aux_nll": self._safe_metric(obs_aux),
            "decoder_std_mean": self._safe_metric(dec_std_mean),
            "dec_std_min": self._safe_metric(dec_std_min),
            "prior_std_mean": self._safe_metric(prior_std_mean),
            "prior_std_max": self._safe_metric(prior_std_max),
            "posterior_std_mean": self._safe_metric(post_std_mean),
            "posterior_std_max": self._safe_metric(post_std_max),
            "rollout_nll": self._safe_metric(rollout_nll),
            "rollout_dtw": self._safe_metric(rollout_dtw),
            "rollout_total": self._safe_metric(rollout_total),
            "rollout_weight_eff": self._safe_metric(torch.tensor(
                float(effective_rollout_weight), device=loss_total.device, dtype=loss_total.dtype
            )),
            "rollout_ramp": self._safe_metric(torch.tensor(float(ramp), device=loss_total.device, dtype=loss_total.dtype)),
            "horizon": self._safe_metric(torch.tensor(float(horizon), device=loss_total.device, dtype=loss_total.dtype)),
            "horizon_schedule": self._safe_metric(
                torch.tensor(float(scheduled_horizon), device=loss_total.device, dtype=loss_total.dtype)
            ),
            "context_len": self._safe_metric(torch.tensor(float(context_len), device=loss_total.device, dtype=loss_total.dtype)),
            "open_loop_crps": self._safe_metric(open_loop_crps),
        }

    @staticmethod
    def _is_role_batch(batch: Any) -> bool:
        return isinstance(batch, Mapping) and {"control", "exogenous", "objective"}.issubset(batch.keys())

    def _should_log_metric(self, key: str) -> bool:
        if key == "aux_nll" and not self._log_aux_metrics:
            return False
        if key in {"rollout_nll", "rollout_total", "rollout_weight_eff", "rollout_ramp", "horizon", "horizon_schedule", "context_len"}:
            return self._log_rollout_metrics
        if key == "rollout_dtw":
            return self._log_rollout_dtw_metric
        return True

    def _shared_step(self, batch: Any, stage: str, batch_idx: Optional[int] = None) -> torch.Tensor:
        if self._is_role_batch(batch) and hasattr(self.model, "imagine_rollout_with_loss"):
            metrics = self._compute_rssm_combined_losses(
                batch=batch,  # type: ignore[arg-type]
                stage=stage,
                compute_rollout_dtw=(stage != "train"),
                batch_idx=batch_idx,
            )
            for key, value in metrics.items():
                if not self._should_log_metric(key):
                    continue
                self.log(
                    f"{stage}/{key}",
                    self._safe_metric(value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(key in {"loss_total", "open_loop_crps"} and stage != "train"),
                )
            if stage == "train":
                self.log("train/loss", self._safe_metric(metrics["loss_total"]), on_step=False, on_epoch=True, prog_bar=True)
            return metrics["loss_total"]

        x, y = batch
        pred = self.forward(x)
        if pred.shape != y.shape:
            pred = pred[:, -y.shape[1]:, : y.shape[-1]]
        loss = F.mse_loss(pred, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train", batch_idx=batch_idx)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Non-finite training loss; skipping step.")
            return None  # type: ignore[return-value]
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx=batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", batch_idx=batch_idx)

    def on_after_backward(self) -> None:
        if self.grad_clip_norm <= 0.0:
            return
        sq = torch.zeros((), device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                sq = sq + torch.sum(p.grad.detach() ** 2)
        grad_norm = torch.sqrt(sq)
        self.log("train/grad_norm", self._safe_metric(grad_norm), on_step=True, on_epoch=False, prog_bar=False)

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self.grad_clip_norm > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.grad_clip_norm),
                gradient_clip_algorithm="norm",
            )
            return
        if gradient_clip_val is not None and float(gradient_clip_val) > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(gradient_clip_val),
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = 100000
        if self.trainer is not None:
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if isinstance(est, int) and est > 0:
                total_steps = int(est)
        warmup_steps = int(max(1, self.scheduler_warmup_steps))
        decay_steps = int(max(1, total_steps - warmup_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            s = max(0, int(step) - warmup_steps)
            progress = min(1.0, float(s) / float(decay_steps))
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.scheduler_min_ratio + (1.0 - self.scheduler_min_ratio) * decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
