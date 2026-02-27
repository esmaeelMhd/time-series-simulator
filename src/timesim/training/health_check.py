"""Lightning callback for per-epoch RSSM health diagnostics."""

from __future__ import annotations

import os
from typing import Any
import logging

try:
    import pytorch_lightning as pl  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "TrainingHealthCheck requires pytorch-lightning. "
        "Install with: pip install pytorch-lightning"
    ) from exc


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert logged metric value to float safely."""
    if value is None:
        return float(default)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


class TrainingHealthCheck(pl.Callback):  # type: ignore[misc]
    """Epoch-end training health report based on logged Lightning metrics."""

    def __init__(self, check_every_n_epochs: int = 1):
        super().__init__()
        self.check_every_n_epochs = max(1, int(check_every_n_epochs))
        self.history: dict[str, list[float]] = {}
        self._use_color = (
            os.environ.get("NO_COLOR", "").strip() == ""
            and os.environ.get("TERM", "").strip() not in {"", "dumb"}
        )
        self._logger = logging.getLogger(__name__)

    def _tag(self, label: str, color: str) -> str:
        if not self._use_color:
            return label
        colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{label}{colors['reset']}"

    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if (trainer.current_epoch + 1) % self.check_every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch)

        kl_raw = _to_float(metrics.get("train/kl_raw"), 0.0)
        kl_active = _to_float(metrics.get("train/kl_active"), 0.0)
        recon_loss = _to_float(metrics.get("train/recon"), 0.0)
        dec_std_min = _to_float(metrics.get("train/dec_std_min"), 0.0)
        prior_std_max = _to_float(metrics.get("train/prior_std_max"), 0.0)
        val_crps = _to_float(metrics.get("val/open_loop_crps"), 0.0)

        status: list[str] = []
        warnings: list[str] = []

        ok = self._tag("[OK]", "green")
        warn = self._tag("[WARN]", "yellow")
        alert = self._tag("[ALERT]", "red")
        info = self._tag("[INFO]", "cyan")

        if kl_raw < 1.0 and epoch > 5:
            warnings.append(
                f"{warn} POSTERIOR COLLAPSE: KL is tiny ({kl_raw:.4f}). Latents are likely inactive."
            )
        elif kl_active < 2 and epoch > 5:
            warnings.append(
                f"{warn} LOW ACTIVITY: Only {int(kl_active)} latent dims active."
            )
        else:
            status.append(
                f"{ok} Latent Space: active_dims={int(kl_active)}, kl_raw={kl_raw:.2f}"
            )

        if dec_std_min < 0.1:
            warnings.append(
                f"{alert} VARIANCE COLLAPSE: decoder min std={dec_std_min:.4f}."
            )
        elif prior_std_max > 5.0:
            warnings.append(
                f"{alert} EXPLODING PRIOR: prior max std={prior_std_max:.2f}."
            )
        else:
            status.append(
                f"{ok} Uncertainty: dec_std_min={dec_std_min:.2f}, prior_std_max={prior_std_max:.2f}"
            )

        if recon_loss > 5.0 and epoch > 10:
            warnings.append(
                f"{warn} UNDERFITTING: recon loss high ({recon_loss:.2f})."
            )

        self._logger.info(f"\n{'=' * 30} EPOCH {epoch} HEALTH CHECK {'=' * 30}")
        if warnings:
            self._logger.warning(f"{warn} ISSUES DETECTED:")
            for msg in warnings:
                self._logger.warning(msg)
        else:
            self._logger.info(f"{ok} TRAINING LOOKS HEALTHY.")

        for msg in status:
            self._logger.info(msg)

        self._logger.info(f"{info} Current CRPS: {val_crps:.4f}")
        self._logger.info("=" * 80 + "\n")


__all__ = ["TrainingHealthCheck"]
