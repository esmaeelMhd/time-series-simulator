"""LightningModule wrapper for RSSM world model training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import math

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "timesim.lightning requires pytorch-lightning. "
        "Install with: pip install pytorch-lightning"
    ) from exc


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
        # Mandatory warmup for stable latent model training.
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
        self.min_context = int(max(1, p.get("min_context", 16)))

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
    def _gaussian_crps(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Closed-form CRPS for Gaussian outputs, averaged over batch/time/dim."""
        sigma = scale.clamp_min(1e-6)
        z = (target - mean) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - (1.0 / math.sqrt(math.pi)))
        return crps.mean()

    def _rollout_schedule(self, seq_len: int) -> tuple[int, int, float]:
        """(horizon, context_len, ramp_factor) with warmup + linear ramp."""
        total_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1) if self.trainer is not None else 1
        ep = int(self.current_epoch) + 1
        max_horizon = min(int(self.rollout_max_horizon), max(0, int(seq_len) - int(self.min_context)))
        if max_horizon <= 0:
            return 0, int(seq_len), 0.0

        frac = max(0.0, min(1.0, float(self.rollout_warmup_fraction)))
        warmup_epochs = int(math.ceil(total_epochs * frac))
        warmup_epochs = min(total_epochs, max(0, warmup_epochs))
        if ep <= warmup_epochs:
            return 0, int(seq_len), 0.0

        denom = max(1, total_epochs - warmup_epochs)
        ramp = float(ep - warmup_epochs) / float(denom)
        ramp = max(0.0, min(1.0, ramp))
        horizon = int(round(max_horizon * ramp))
        if horizon == 0 and ramp > 0.0:
            horizon = 1
        horizon = min(max_horizon, max(0, horizon))
        context_len = max(int(self.min_context), int(seq_len) - int(horizon))
        horizon = max(0, int(seq_len) - int(context_len))
        return int(horizon), int(context_len), float(ramp)

    def _compute_rssm_combined_losses(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        compute_rollout_dtw: bool,
    ) -> Dict[str, torch.Tensor]:
        objective = batch["objective"]
        seq_len = int(objective.shape[1])
        horizon, context_len, ramp = self._rollout_schedule(seq_len=seq_len)
        effective_rollout_weight = float(self.rollout_weight) * float(ramp)

        out = self.model.imagine_rollout_with_loss(
            batch=batch,  # type: ignore[arg-type]
            context_len=context_len,
            horizon=horizon,
            sample_prior=False,
            compute_rollout_dtw=bool(compute_rollout_dtw and self.rollout_dtw_weight > 0.0),
            rollout_dtw_gamma=float(self.rollout_dtw_gamma),
        )
        obs_recon = out["obs_recon_nll"]
        obs_kl = out["obs_kl"]
        obs_aux = out["obs_aux_nll"]
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
        if (
            horizon > 0
            and isinstance(imagined, Mapping)
            and torch.is_tensor(imagined.get("dist_loc"))
            and torch.is_tensor(imagined.get("dist_scale"))
        ):
            target_y = objective[:, -horizon:, :]
            open_loop_crps = self._gaussian_crps(
                mean=imagined["dist_loc"],  # type: ignore[index]
                scale=imagined["dist_scale"],  # type: ignore[index]
                target=target_y,
            )

        return {
            "loss_total": loss_total,
            "loss_std": loss_std,
            "recon_nll": obs_recon,
            "kl": obs_kl,
            "aux_nll": obs_aux,
            "rollout_nll": rollout_nll,
            "rollout_dtw": rollout_dtw,
            "rollout_total": rollout_total,
            "rollout_weight_eff": torch.tensor(
                float(effective_rollout_weight), device=loss_total.device, dtype=loss_total.dtype
            ),
            "rollout_ramp": torch.tensor(float(ramp), device=loss_total.device, dtype=loss_total.dtype),
            "horizon": torch.tensor(float(horizon), device=loss_total.device, dtype=loss_total.dtype),
            "context_len": torch.tensor(float(context_len), device=loss_total.device, dtype=loss_total.dtype),
            "open_loop_crps": open_loop_crps,
        }

    @staticmethod
    def _is_role_batch(batch: Any) -> bool:
        return isinstance(batch, Mapping) and {"control", "exogenous", "objective"}.issubset(batch.keys())

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        if self._is_role_batch(batch) and hasattr(self.model, "imagine_rollout_with_loss"):
            metrics = self._compute_rssm_combined_losses(
                batch=batch,  # type: ignore[arg-type]
                compute_rollout_dtw=(stage != "train"),
            )
            for key, value in metrics.items():
                self.log(
                    f"{stage}/{key}",
                    value,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    prog_bar=(key in {"loss_total", "open_loop_crps"} and stage != "train"),
                )
            if stage == "train":
                self.log("train/loss", metrics["loss_total"], on_step=True, on_epoch=True, prog_bar=True)
            return metrics["loss_total"]

        x, y = batch
        pred = self.forward(x)
        if pred.shape != y.shape:
            # Common case: model predicts full sequence but target is horizon-only.
            pred = pred[:, -y.shape[1]:, : y.shape[-1]]
        loss = F.mse_loss(pred, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_after_backward(self) -> None:
        if self.grad_clip_norm <= 0.0:
            return
        sq = torch.zeros((), device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                sq = sq + torch.sum(p.grad.detach() ** 2)
        grad_norm = torch.sqrt(sq)
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

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
