"""Unified trainer for world models with multi-step rollout training.

HOT PATH: The training loop (_train_step, _validate) runs every step/epoch.
Performance optimizations applied:
- Batched rollouts for uniform horizons
- Vectorized mask operations
- Minimal allocations in training loop
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Callable, Dict, Any
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..data.dataset import GroupedTimeSeriesDataset
from ..data.sampling import SamplingStrategy, RandomStartFixedHorizon
from ..models.base import WorldModelBase
from ..utils.early_stop import EarlyStopping
from .losses import (
    OneStepLoss,
    MultiStepLoss,
    CombinedLoss,
    ProbabilisticRolloutLoss,
    soft_dtw_distance,
)
from .rollout import batch_rollout_padded, get_rollout_schedule


class WorldModelTrainer:
    """Unified trainer for world models with multi-step rollout training.
    
    This trainer implements the "see every possible path" approach by:
    1. Sampling diverse (start_index, horizon) pairs using a SamplingStrategy
    2. Performing multi-environment rollouts in parallel
    3. Computing multi-step losses to reduce compounding errors
    
    Parameters
    ----------
    model : WorldModelBase
        World model to train.
    dataset : GroupedTimeSeriesDataset
        Training dataset.
    val_dataset : GroupedTimeSeriesDataset, optional
        Validation dataset.
    sampling_strategy : SamplingStrategy
        Strategy for sampling rollout starting points and horizons.
    warmup_len : int
        Length of warmup sequence for state initialization.
    batch_size : int, default 32
        Number of rollouts per training batch.
    loss_type : {"mse", "mae", "huber", "shape"}, default "mse"
        Base loss function type.
    loss_weighting : {"uniform", "linear", "exponential"}, default "uniform"
        Time-step weighting scheme for multi-step loss.
    loss_weight_scale : float, default 1.0
        Scale factor for weighted multi-step losses.
    training_mode : {"multi_step", "one_step", "combined"}, default "multi_step"
        Training mode:
        - "multi_step": Pure autoregressive rollout loss
        - "one_step": Teacher-forced one-step loss
        - "combined": Weighted combination of both
    feedback : {"model", "teacher", "mixed"}, default "model"
        Feedback mode for multi-step rollouts.
    teacher_forcing_ratio : float, default 0.0
        Ratio for mixed feedback mode.
    one_step_weight : float, default 0.5
        Weight for one-step loss in combined mode.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use. If None, creates Adam with lr=1e-3.
    device : torch.device or str, default "cpu"
        Device for training. Ignored if use_gpu=True.
    use_gpu : bool, default False
        If True, automatically use GPU if available, otherwise use CPU.
        Takes precedence over device parameter.
    early_stopping : bool, default False
        Whether to use early stopping.
    patience : int, default 5
        Patience for early stopping.
    min_delta : float, default 0.0
        Minimum validation improvement to reset early-stopping patience.
    run_dir : str or Path, optional
        Directory for saving outputs and logs.
    writer : SummaryWriter, optional
        TensorBoard writer for logging.
    """
    
    def __init__(
        self,
        model: WorldModelBase,
        dataset: GroupedTimeSeriesDataset,
        val_dataset: Optional[GroupedTimeSeriesDataset] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        warmup_len: int = 24,
        batch_size: int = 32,
        loss_type: Literal["mse", "mae", "huber", "shape"] = "mse",
        loss_weighting: Literal["uniform", "linear", "exponential"] = "uniform",
        loss_weight_scale: float = 1.0,
        shape_loss_cfg: Optional[Dict[str, Any]] = None,
        training_mode: Literal["multi_step", "one_step", "combined"] = "multi_step",
        feedback: Literal["model", "teacher", "mixed"] = "model",
        teacher_forcing_ratio: float = 0.0,
        one_step_weight: float = 0.5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
        use_gpu: bool = False,
        use_amp: bool = False,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.0,
        run_dir: Optional[str | Path] = None,
        writer: Optional["SummaryWriter"] = None, # type: ignore
        probabilistic_cfg: Optional[Dict[str, Any]] = None,
        sequence_curriculum_cfg: Optional[Dict[str, Any]] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.warmup_len = warmup_len
        self.batch_size = batch_size
        self.training_mode = training_mode
        self.feedback = feedback
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_amp = use_amp
        self.is_probabilistic_model = bool(getattr(self.model, "is_probabilistic", False))
        self.disable_aux_loss = not bool(getattr(self.model, "use_aux_decoder", True))
        
        # Device setup
        if use_gpu:
            # Automatically use GPU if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("Warning: use_gpu=True but CUDA is not available. Using CPU instead.")
        else:
            # Use the specified device
            self.device = torch.device(device)
        self.model.to(self.device)
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16
        if self.amp_enabled:
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
            except Exception:
                self.amp_dtype = torch.float16
        self.use_grad_scaler = bool(self.amp_enabled and self.amp_dtype == torch.float16)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        
        # Sampling strategy
        if sampling_strategy is None:
            # Default: random start with fixed horizon
            self.sampling_strategy = RandomStartFixedHorizon(horizon=dataset.pred_len)
        else:
            self.sampling_strategy = sampling_strategy
        
        # Loss function
        if training_mode == "one_step":
            self.loss_fn = OneStepLoss(
                loss_type=loss_type,
                shape_loss_cfg=shape_loss_cfg,
            )
        elif training_mode == "multi_step":
            self.loss_fn = MultiStepLoss(
                loss_type=loss_type,
                weighting=loss_weighting,
                weight_scale=loss_weight_scale,
                shape_loss_cfg=shape_loss_cfg,
            )
        elif training_mode == "combined":
            self.loss_fn = CombinedLoss(
                one_step_weight=one_step_weight,
                multi_step_weight=1.0 - one_step_weight,
                loss_type=loss_type,
                multi_step_weighting=loss_weighting,
                multi_step_weight_scale=loss_weight_scale,
                shape_loss_cfg=shape_loss_cfg,
            )
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        self._val_multi_step_loss = MultiStepLoss(
            loss_type=loss_type,
            weighting=loss_weighting,
            weight_scale=loss_weight_scale,
            shape_loss_cfg=shape_loss_cfg,
        )
        prob_cfg = probabilistic_cfg or {}
        self.probabilistic_loss_fn = ProbabilisticRolloutLoss(
            recon_weight=prob_cfg.get("recon_weight", prob_cfg.get("elbo_weight", 1.0)),
            kl_weight=prob_cfg.get("kl_weight", 1.0),
            aux_weight=prob_cfg.get("aux_weight", prob_cfg.get("rollout_mse_weight", 1.0)),
            kl_free_bits=prob_cfg.get("kl_free_bits", 1.0),
            kl_balance=prob_cfg.get("kl_balance", 0.8),
            use_kl_balancing=prob_cfg.get("use_kl_balancing", True),
            use_free_bits=prob_cfg.get("use_free_bits", True),
            use_symlog=prob_cfg.get("use_symlog", False),
        )
        self.prob_objective = str(prob_cfg.get("objective", "rssm"))
        self.rollout_weight = float(prob_cfg.get("rollout_weight", 0.0))
        self.rollout_dtw_weight = float(prob_cfg.get("rollout_dtw_weight", 0.0))
        self.rollout_dtw_gamma = float(prob_cfg.get("rollout_dtw_gamma", 0.1))
        self.rollout_warmup_fraction = float(prob_cfg.get("rollout_warmup_fraction", 0.30))
        self.rollout_max_horizon = max(
            0,
            int(prob_cfg.get("rollout_max_horizon", max(1, getattr(self.dataset, "pred_len", 1)))),
        )
        self.min_context = max(1, int(prob_cfg.get("min_context", 16)))
        self.kl_warmup_enabled = bool(prob_cfg.get("kl_warmup_enabled", False))
        self.kl_beta_start = float(prob_cfg.get("kl_beta_start", 1.0))
        self.kl_beta_end = float(prob_cfg.get("kl_beta_end", 1.0))
        self.kl_warmup_epochs = max(1, int(prob_cfg.get("kl_warmup_epochs", 1)))
        self.grad_clip_norm = float(prob_cfg.get("grad_clip_norm", 100.0))
        # Mandatory warmup for stable RSSM optimization.
        self.lr_warmup_steps = max(1, int(prob_cfg.get("lr_warmup_steps", 1000)))
        self.lr_min_ratio = float(prob_cfg.get("lr_min_ratio", 0.01))
        self.collapse_kl_threshold = float(prob_cfg.get("collapse_kl_threshold", 0.1))
        self.collapse_patience_epochs = max(1, int(prob_cfg.get("collapse_patience_epochs", 3)))
        self.checkpoint_top_k = max(1, int(prob_cfg.get("checkpoint_top_k", 3)))
        self.early_stopping_monitor = str(
            prob_cfg.get(
                "early_stopping_monitor",
                "open_loop_crps" if self.is_probabilistic_model else "val_loss",
            )
        ).lower()
        self._collapse_counter = 0
        self._last_prob_info: Dict[str, float] = {}
        self._current_epoch = 1
        self._fit_epochs = 1
        default_checkpoint_metric = "open_loop_crps" if self.is_probabilistic_model else "val_loss"
        self.checkpoint_metric = str(prob_cfg.get("checkpoint_metric", default_checkpoint_metric)).lower()
        self.checkpoint_open_loop_horizon = max(
            1,
            int(prob_cfg.get("checkpoint_open_loop_horizon", getattr(self.dataset, "pred_len", 1))),
        )
        self.checkpoint_open_loop_windows = max(
            1,
            int(prob_cfg.get("checkpoint_open_loop_windows", 4)),
        )
        self.checkpoint_open_loop_samples = max(
            1,
            int(prob_cfg.get("checkpoint_open_loop_samples", 32)),
        )
        if self.disable_aux_loss:
            self.probabilistic_loss_fn.aux_weight = 0.0

        curriculum_cfg = dict(sequence_curriculum_cfg or {})
        self.sequence_curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.curriculum_start_horizon = max(
            1,
            int(curriculum_cfg.get("start_horizon", curriculum_cfg.get("start_seq_len", 16))),
        )
        self.curriculum_target_horizon = max(
            self.curriculum_start_horizon,
            int(
                curriculum_cfg.get(
                    "target_horizon",
                    curriculum_cfg.get("target_seq_len", getattr(self.dataset, "pred_len", 1)),
                )
            ),
        )
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-6)
        else:
            self.optimizer = optimizer
        self._base_lrs = [float(pg.get("lr", 1e-3)) for pg in self.optimizer.param_groups]
        self._global_step = 0
        self._total_train_steps = 1
        
        # Early stopping
        self.early_stopping = (
            EarlyStopping(patience=patience, min_delta=min_delta)
            if early_stopping else None
        )
        
        # Logging
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if writer is not None:
                self.writer = writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(log_dir=self.run_dir)
                except ModuleNotFoundError:
                    self.writer = None
            
            # CSV metrics file
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write(
                        "epoch,train_loss,val_loss,loss_std,loss_total,recon_nll,kl,kl_raw,aux_nll,"
                        "rollout_nll,rollout_dtw,rollout_total,rollout_weight_eff,"
                        "rollout_ramp,horizon_schedule,context_len,grad_norm,grad_norm_pre,lr\n"
                    )
        else:
            self.writer = writer
            self.metrics_path = None
        
        # Random number generator for reproducibility
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.checkpoint_metadata: Dict[str, Any] = dict(checkpoint_metadata or {})

    def _rollout_schedule(self) -> tuple[int, int, float]:
        """Return (horizon, context_len, ramp_factor) for rollout losses."""
        cfg = {
            "epochs": int(self._fit_epochs),
            "seq_len": int(self.warmup_len),
            "rollout_warmup_fraction": float(self.rollout_warmup_fraction),
            "rollout_max_horizon": int(self.rollout_max_horizon),
            "min_context": int(self.min_context),
        }
        return get_rollout_schedule(epoch=int(self._current_epoch), cfg=cfg)

    def _current_kl_beta(self) -> float:
        """Current KL beta with optional linear warmup."""
        if not self.kl_warmup_enabled:
            return self.kl_beta_end
        if self.kl_warmup_epochs <= 1:
            return self.kl_beta_end
        ep = int(self._current_epoch)
        if ep >= self.kl_warmup_epochs:
            return self.kl_beta_end
        frac = float(ep - 1) / float(self.kl_warmup_epochs - 1)
        return self.kl_beta_start + frac * (self.kl_beta_end - self.kl_beta_start)

    def _update_learning_rate(self):
        """Linear warmup then cosine decay on optimizer LR."""
        if not self._base_lrs:
            return
        step = int(self._global_step)
        total = max(1, int(self._total_train_steps))
        warmup = int(self.lr_warmup_steps)

        if warmup > 0 and step < warmup:
            mult = float(step + 1) / float(warmup)
        else:
            decay_steps = max(1, total - warmup)
            progress = float(max(0, step - warmup)) / float(decay_steps)
            progress = max(0.0, min(1.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            mult = self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine

        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = float(base_lr) * float(mult)

    def _current_curriculum_horizon(self) -> int:
        """Current horizon cap for sequence-length curriculum."""
        if not self.sequence_curriculum_enabled:
            return self.curriculum_target_horizon
        if self._fit_epochs <= 1:
            return self.curriculum_target_horizon
        frac = float(self._current_epoch - 1) / float(max(1, self._fit_epochs - 1))
        frac = max(0.0, min(1.0, frac))
        curr = self.curriculum_start_horizon + frac * (
            self.curriculum_target_horizon - self.curriculum_start_horizon
        )
        return max(1, int(round(curr)))

    def _apply_horizon_curriculum(self, horizons: np.ndarray) -> np.ndarray:
        if not self.sequence_curriculum_enabled:
            return horizons
        cap = self._current_curriculum_horizon()
        return np.clip(horizons.astype(np.int64, copy=False), 1, cap)

    def _checkpoint_score(self, val_loss: Optional[float]) -> tuple[float, str]:
        """Compute score used to select best checkpoint (lower is better)."""
        if (
            self.checkpoint_metric == "open_loop_crps"
            and self.is_probabilistic_model
            and self.val_dataset is not None
        ):
            max_h = max(1, len(self.val_dataset.values) - self.warmup_len)
            horizon = max(1, min(self.checkpoint_open_loop_horizon, max_h))
            try:
                from ..evaluation import open_loop_evaluate

                curves = open_loop_evaluate(
                    model=self.model,
                    dataset=self.val_dataset,
                    warmup_len=self.warmup_len,
                    horizon=horizon,
                    n_windows=self.checkpoint_open_loop_windows,
                    n_samples=self.checkpoint_open_loop_samples,
                    device=self.device,
                )
                crps = curves.get("crps")
                if crps is not None and len(crps) > 0 and np.all(np.isfinite(crps)):
                    return float(np.mean(crps)), "open_loop_crps"
            except Exception:
                pass

        if val_loss is None or not np.isfinite(val_loss):
            return float("inf"), "val_loss"
        return float(val_loss), "val_loss"
    
    def _train_step(self) -> float:
        """Perform one training step (one batch of rollouts).
        
        Returns
        -------
        float
            Training loss for this batch.
        """
        self.model.train()
        self._update_learning_rate()
        
        # Sample rollout starting points and horizons
        start_indices, horizons = self.sampling_strategy.sample(
            dataset_length=len(self.dataset.values),
            batch_size=self.batch_size,
            warmup_len=self.warmup_len,
            rng=self.rng,
        )
        horizons = self._apply_horizon_curriculum(horizons)
        sched_horizon, context_len, rollout_ramp = self._rollout_schedule()
        effective_rollout_weight = max(0.0, self.rollout_weight * rollout_ramp)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Perform batched rollouts
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            if self.is_probabilistic_model:
                result_teacher = batch_rollout_padded(
                    self.model, self.dataset, start_indices, horizons,
                    self.warmup_len, feedback="teacher",
                    device=self.device
                )
                targets = result_teacher["targets"]
                mask = result_teacher["mask"]
                exogenous_targets = result_teacher.get("exogenous")
                dist_loc_latent = result_teacher.get("dist_loc_latent")
                dist_loc = result_teacher.get("dist_loc")
                dist_scale = result_teacher.get("dist_scale")
                prior_mu = result_teacher.get("prior_mu")
                prior_logvar = result_teacher.get("prior_logvar")
                posterior_mu = result_teacher.get("posterior_mu")
                posterior_logvar = result_teacher.get("posterior_logvar")
                aux_loc = result_teacher.get("aux_loc")
                aux_scale = result_teacher.get("aux_scale")
                if self.disable_aux_loss:
                    exogenous_targets = None
                    aux_loc = None
                    aux_scale = None

                if dist_loc_latent is None:
                    if dist_loc is None:
                        raise ValueError("Probabilistic rollout must return dist_loc or dist_loc_latent")
                    if self.probabilistic_loss_fn.use_symlog:
                        dist_loc_latent = self.probabilistic_loss_fn._symlog(dist_loc)  # pylint: disable=protected-access
                    else:
                        dist_loc_latent = dist_loc
                if (
                    dist_scale is None
                    or prior_mu is None
                    or prior_logvar is None
                    or posterior_mu is None
                    or posterior_logvar is None
                ):
                    raise ValueError(
                        "Probabilistic model rollout must return "
                        "dist_scale/prior_mu/prior_logvar/posterior_mu/posterior_logvar"
                    )

                kl_beta = self._current_kl_beta()
                std_loss, info = self.probabilistic_loss_fn(
                    targets=targets,
                    dist_loc_latent=dist_loc_latent,
                    dist_scale=dist_scale,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    posterior_mu=posterior_mu,
                    posterior_logvar=posterior_logvar,
                    exogenous_targets=exogenous_targets,
                    aux_loc=aux_loc,
                    aux_scale=aux_scale,
                    mask=mask,
                    kl_beta=kl_beta,
                )
                raw_kl_elem = self.probabilistic_loss_fn._balanced_kl(  # pylint: disable=protected-access
                    posterior_mu=posterior_mu,
                    posterior_logvar=posterior_logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                )
                raw_kl = self.probabilistic_loss_fn._sum_time_mean_batch(  # pylint: disable=protected-access
                    raw_kl_elem.sum(dim=-1), mask
                )

                rollout_nll = torch.zeros((), dtype=std_loss.dtype, device=std_loss.device)
                rollout_dtw = torch.zeros_like(rollout_nll)
                rollout_total = torch.zeros_like(rollout_nll)
                combined_loss = std_loss
                rollout_computed = False
                rollout_horizon = int(max(0, sched_horizon))
                if rollout_horizon > 0 and effective_rollout_weight > 0.0:
                    rollout_horizons = np.minimum(
                        horizons.astype(np.int64, copy=False),
                        np.full_like(horizons, fill_value=rollout_horizon),
                    )
                    rollout_horizons = np.clip(rollout_horizons, 1, None)
                    result_model = batch_rollout_padded(
                        self.model, self.dataset, start_indices, rollout_horizons,
                        self.warmup_len, feedback="model",
                        device=self.device
                    )
                    rollout_targets = result_model["targets"]
                    rollout_mask = result_model["mask"]
                    rollout_dist_loc_latent = result_model.get("dist_loc_latent")
                    rollout_dist_loc = result_model.get("dist_loc")
                    rollout_dist_scale = result_model.get("dist_scale")
                    if rollout_dist_loc_latent is None:
                        if rollout_dist_loc is None:
                            raise ValueError("Model-feedback rollout must return dist_loc or dist_loc_latent")
                        if self.probabilistic_loss_fn.use_symlog:
                            rollout_dist_loc_latent = self.probabilistic_loss_fn._symlog(rollout_dist_loc)  # pylint: disable=protected-access
                        else:
                            rollout_dist_loc_latent = rollout_dist_loc
                    if rollout_dist_scale is None:
                        raise ValueError("Model-feedback rollout must return dist_scale")

                    rollout_nll = self.probabilistic_loss_fn.compute_recon_nll(
                        targets=rollout_targets,
                        dist_loc_latent=rollout_dist_loc_latent,
                        dist_scale=rollout_dist_scale,
                        mask=rollout_mask,
                    )

                    if self.rollout_dtw_weight > 0.0:
                        preds_roll = result_model["predictions"]
                        dtw_vals = []
                        for i in range(preds_roll.shape[0]):
                            h_i = int(rollout_horizons[i])
                            if h_i <= 1:
                                continue
                            dtw_vals.append(
                                soft_dtw_distance(
                                    preds_roll[i:i + 1, :h_i, :],
                                    rollout_targets[i:i + 1, :h_i, :],
                                    gamma=self.rollout_dtw_gamma,
                                )
                            )
                        if dtw_vals:
                            rollout_dtw = torch.stack(dtw_vals).mean()
                    rollout_total = rollout_nll + self.rollout_dtw_weight * rollout_dtw
                    combined_loss = std_loss + effective_rollout_weight * rollout_total
                    rollout_computed = True
                loss = combined_loss

                self._last_prob_info = {
                    **info,
                    "loss_std": float(std_loss.detach().item()),
                    "loss_total": float(combined_loss.detach().item()),
                    "kl_raw": float(raw_kl.detach().item()),
                    "rollout_horizon": float(rollout_horizon),
                    "context_len": float(context_len),
                    "rollout_ramp": float(rollout_ramp),
                    "rollout_weight_eff": float(effective_rollout_weight),
                    "rollout_nll": float(rollout_nll.detach().item()),
                    "rollout_dtw": float(rollout_dtw.detach().item()),
                    "rollout_total": float(rollout_total.detach().item()),
                    "rollout_computed": 1.0 if rollout_computed else 0.0,
                }
            elif self.training_mode == "combined":
                # Need both teacher-forced and model-feedback rollouts
                result_teacher = batch_rollout_padded(
                    self.model, self.dataset, start_indices, horizons,
                    self.warmup_len, feedback="teacher", device=self.device
                )
                result_model = batch_rollout_padded(
                    self.model, self.dataset, start_indices, horizons,
                    self.warmup_len, feedback="model", device=self.device
                )
                
                predictions_teacher = result_teacher["predictions"]
                predictions_model = result_model["predictions"]
                targets = result_teacher["targets"]
                mask = result_teacher["mask"]
                
                # Compute loss
                loss, info = self.loss_fn(predictions_teacher, predictions_model, targets)
                self._last_prob_info = {}
                
            else:
                # Single rollout mode
                result = batch_rollout_padded(
                    self.model, self.dataset, start_indices, horizons,
                    self.warmup_len, feedback=self.feedback,
                    teacher_forcing_ratio=self.teacher_forcing_ratio,
                    device=self.device
                )
                
                predictions = result["predictions"]
                targets = result["targets"]
                mask = result["mask"]
                
                # Mask out padded values for loss computation
                # Expand mask to match output dimensions
                mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
                predictions_masked = predictions * mask_expanded
                targets_masked = targets * mask_expanded
                
                # Compute loss
                loss = self.loss_fn(predictions_masked, targets_masked)
                self._last_prob_info = {}
        
        # Guard against NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in training loss. Check data and model stability.")
        
        # Backward pass
        grad_norm_value = float("nan")
        grad_norm_preclip = float("nan")
        if self.amp_enabled and self.use_grad_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm > 0:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                grad_norm_preclip = float(gn.detach().item() if torch.is_tensor(gn) else gn)
                grad_norm_value = float(min(grad_norm_preclip, self.grad_clip_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_norm > 0:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                grad_norm_preclip = float(gn.detach().item() if torch.is_tensor(gn) else gn)
                grad_norm_value = float(min(grad_norm_preclip, self.grad_clip_norm))
            self.optimizer.step()
        if self.grad_clip_norm <= 0:
            sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    sq += float(torch.sum(g * g).item())
            grad_norm_value = float(math.sqrt(max(0.0, sq)))
            grad_norm_preclip = grad_norm_value

        if self.is_probabilistic_model and self._last_prob_info is not None:
            self._last_prob_info["grad_norm"] = float(grad_norm_value)
            self._last_prob_info["grad_norm_pre"] = float(grad_norm_preclip)
            self._last_prob_info["lr"] = float(self.optimizer.param_groups[0].get("lr", float("nan")))
            self._last_prob_info["horizon_schedule"] = float(sched_horizon)
            self._last_prob_info["context_len"] = float(context_len)
            self._last_prob_info["rollout_ramp"] = float(rollout_ramp)

            if self.writer is not None:
                gs = int(self._global_step)
                for key in [
                    "loss_std",
                    "loss_total",
                    "recon_nll",
                    "kl",
                    "kl_raw",
                    "aux_nll",
                    "rollout_nll",
                    "rollout_dtw",
                    "rollout_total",
                    "rollout_weight_eff",
                    "rollout_ramp",
                    "horizon_schedule",
                    "context_len",
                    "grad_norm",
                    "grad_norm_pre",
                    "lr",
                ]:
                    if key in self._last_prob_info:
                        self.writer.add_scalar(f"Train/{key}", self._last_prob_info[key], gs)
        self._global_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def _validate(self) -> Optional[float]:
        """Validate on validation dataset.
        
        Returns
        -------
        float or None
            Validation loss, or None if no validation dataset.
        """
        if self.val_dataset is None:
            return None
        
        self.model.eval()
        
        # Sample validation rollouts
        val_batches = max(1, len(self.val_dataset) // (self.batch_size * self.warmup_len))
        val_losses = []
        sched_horizon, context_len, rollout_ramp = self._rollout_schedule()
        effective_rollout_weight = max(0.0, self.rollout_weight * rollout_ramp)
        
        for _ in range(val_batches):
            start_indices, horizons = self.sampling_strategy.sample(
                dataset_length=len(self.val_dataset.values),
                batch_size=self.batch_size,
                warmup_len=self.warmup_len,
                rng=self.rng,
            )
            horizons = self._apply_horizon_curriculum(horizons)
            
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                if self.is_probabilistic_model:
                    result_teacher = batch_rollout_padded(
                        self.model, self.val_dataset, start_indices, horizons,
                        self.warmup_len, feedback="teacher", device=self.device
                    )
                    targets = result_teacher["targets"]
                    mask = result_teacher["mask"]
                    exogenous_targets = result_teacher.get("exogenous")
                    dist_loc_latent = result_teacher.get("dist_loc_latent")
                    dist_loc = result_teacher.get("dist_loc")
                    dist_scale = result_teacher.get("dist_scale")
                    prior_mu = result_teacher.get("prior_mu")
                    prior_logvar = result_teacher.get("prior_logvar")
                    posterior_mu = result_teacher.get("posterior_mu")
                    posterior_logvar = result_teacher.get("posterior_logvar")
                    aux_loc = result_teacher.get("aux_loc")
                    aux_scale = result_teacher.get("aux_scale")
                    if self.disable_aux_loss:
                        exogenous_targets = None
                        aux_loc = None
                        aux_scale = None
                    if dist_loc_latent is None:
                        if dist_loc is None:
                            raise ValueError("Probabilistic rollout must return dist_loc or dist_loc_latent")
                        if self.probabilistic_loss_fn.use_symlog:
                            dist_loc_latent = self.probabilistic_loss_fn._symlog(dist_loc)  # pylint: disable=protected-access
                        else:
                            dist_loc_latent = dist_loc
                    if (
                        dist_scale is None
                        or prior_mu is None
                        or prior_logvar is None
                        or posterior_mu is None
                        or posterior_logvar is None
                    ):
                        raise ValueError(
                            "Probabilistic model rollout must return "
                            "dist_scale/prior_mu/prior_logvar/posterior_mu/posterior_logvar"
                        )
                    kl_beta = self._current_kl_beta()
                    loss, _ = self.probabilistic_loss_fn(
                        targets=targets,
                        dist_loc_latent=dist_loc_latent,
                        dist_scale=dist_scale,
                        prior_mu=prior_mu,
                        prior_logvar=prior_logvar,
                        posterior_mu=posterior_mu,
                        posterior_logvar=posterior_logvar,
                        exogenous_targets=exogenous_targets,
                        aux_loc=aux_loc,
                        aux_scale=aux_scale,
                        mask=mask,
                        kl_beta=kl_beta,
                    )
                    if sched_horizon > 0 and effective_rollout_weight > 0.0:
                        rollout_horizons = np.minimum(
                            horizons.astype(np.int64, copy=False),
                            np.full_like(horizons, fill_value=int(sched_horizon)),
                        )
                        rollout_horizons = np.clip(rollout_horizons, 1, None)
                        result_model = batch_rollout_padded(
                            self.model, self.val_dataset, start_indices, rollout_horizons,
                            self.warmup_len, feedback="model", device=self.device
                        )
                        rollout_targets = result_model["targets"]
                        rollout_mask = result_model["mask"]
                        rollout_dist_loc_latent = result_model.get("dist_loc_latent")
                        rollout_dist_loc = result_model.get("dist_loc")
                        rollout_dist_scale = result_model.get("dist_scale")
                        if rollout_dist_loc_latent is None:
                            if rollout_dist_loc is None:
                                raise ValueError("Model-feedback rollout must return dist_loc or dist_loc_latent")
                            if self.probabilistic_loss_fn.use_symlog:
                                rollout_dist_loc_latent = self.probabilistic_loss_fn._symlog(rollout_dist_loc)  # pylint: disable=protected-access
                            else:
                                rollout_dist_loc_latent = rollout_dist_loc
                        if rollout_dist_scale is None:
                            raise ValueError("Model-feedback rollout must return dist_scale")

                        rollout_nll = self.probabilistic_loss_fn.compute_recon_nll(
                            targets=rollout_targets,
                            dist_loc_latent=rollout_dist_loc_latent,
                            dist_scale=rollout_dist_scale,
                            mask=rollout_mask,
                        )
                        rollout_dtw = torch.zeros((), dtype=rollout_nll.dtype, device=rollout_nll.device)
                        if self.rollout_dtw_weight > 0.0:
                            preds_roll = result_model["predictions"]
                            dtw_vals = []
                            for i in range(preds_roll.shape[0]):
                                h_i = int(rollout_horizons[i])
                                if h_i <= 1:
                                    continue
                                dtw_vals.append(
                                    soft_dtw_distance(
                                        preds_roll[i:i + 1, :h_i, :],
                                        rollout_targets[i:i + 1, :h_i, :],
                                        gamma=self.rollout_dtw_gamma,
                                    )
                                )
                            if dtw_vals:
                                rollout_dtw = torch.stack(dtw_vals).mean()
                        rollout_total = rollout_nll + self.rollout_dtw_weight * rollout_dtw
                        loss = loss + effective_rollout_weight * rollout_total
                else:
                    result = batch_rollout_padded(
                        self.model, self.val_dataset, start_indices, horizons,
                        self.warmup_len, feedback="model", device=self.device
                    )

                    predictions = result["predictions"]
                    targets = result["targets"]
                    mask = result["mask"]

                    # Mask out padded values
                    mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
                    predictions_masked = predictions * mask_expanded
                    targets_masked = targets * mask_expanded

                    # Compute loss
                    if self.training_mode == "combined":
                        # For validation, just use multi-step loss
                        loss = self._val_multi_step_loss(predictions_masked, targets_masked)
                    else:
                        loss = self.loss_fn(predictions_masked, targets_masked)
            
            val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def fit(
        self,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        verbose: bool = True,
        checkpoint_path: Optional[str | Path] = None,
        on_checkpoint_saved: Optional[Callable[[int, float], None]] = None,
    ) -> tuple[list[float], list[Optional[float]]]:
        """Train the world model.
        
        Parameters
        ----------
        epochs : int, default 10
            Number of training epochs.
        steps_per_epoch : int, optional
            Number of training steps per epoch. If None, uses a heuristic
            based on dataset size.
        verbose : bool, default True
            Whether to print progress.
        checkpoint_path : str or Path, optional
            If provided, save checkpoint only when checkpoint metric improves.
        on_checkpoint_saved : callable, optional
            Callback invoked as ``on_checkpoint_saved(epoch, score)`` when
            an improved validation checkpoint is saved.
        
        Returns
        -------
        train_losses : list of float
            Training loss per epoch.
        val_losses : list of float or None
            Validation loss per epoch.
        """
        if steps_per_epoch is None:
            # Heuristic: aim to see each starting point ~once per epoch
            dataset_len = len(self.dataset.values) - self.warmup_len
            steps_per_epoch = max(1, dataset_len // self.batch_size)
        self._total_train_steps = max(1, int(epochs) * int(steps_per_epoch))
        self._global_step = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch")
            print(f"Batch size: {self.batch_size}, Training mode: {self.training_mode}")
            print(f"{'='*70}\n")
        
        train_losses = []
        val_losses = []
        start_time = time.time()
        epoch_times = []
        best_val_loss: Optional[float] = None
        best_checkpoint_score: Optional[float] = None
        best_checkpoint_label: str = "val_loss"
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        checkpoint_target: Optional[Path] = None
        checkpoint_bundle_dir: Optional[Path] = None
        topk_checkpoints: list[tuple[float, Path]] = []
        if checkpoint_path is not None:
            checkpoint_target = Path(checkpoint_path)
        elif self.run_dir:
            checkpoint_target = self.run_dir / "best_checkpoint.pth"
        if checkpoint_target is not None:
            checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_bundle_dir = checkpoint_target.parent / "checkpoints"
            checkpoint_bundle_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            self._fit_epochs = epochs
            self._current_epoch = epoch
            epoch_start_time = time.time()
            if verbose:
                print(f"Epoch {epoch}/{epochs}...")
                if self.sequence_curriculum_enabled:
                    print(
                        "  Curriculum horizon cap: "
                        f"{self._current_curriculum_horizon()} "
                        f"(start={self.curriculum_start_horizon}, target={self.curriculum_target_horizon})"
                    )
            
            # Training
            self.model.train()
            epoch_losses = []
            epoch_kl_values = []
            epoch_prob_stats: Dict[str, list[float]] = {
                "loss_std": [],
                "loss_total": [],
                "recon_nll": [],
                "kl": [],
                "kl_raw": [],
                "aux_nll": [],
                "rollout_nll": [],
                "rollout_dtw": [],
                "rollout_total": [],
                "rollout_weight_eff": [],
                "rollout_ramp": [],
                "horizon_schedule": [],
                "context_len": [],
                "grad_norm": [],
                "grad_norm_pre": [],
                "lr": [],
            }
            
            if verbose:
                # Create progress bar for training steps
                pbar = tqdm(
                    range(steps_per_epoch),
                    desc=f"Epoch {epoch}/{epochs} [Train]",
                    unit="step",
                    leave=False,
                    ncols=100,
                )
            else:
                pbar = range(steps_per_epoch)
            
            for step in pbar:
                batch_loss = self._train_step()
                epoch_losses.append(batch_loss)
                if self.is_probabilistic_model and self._last_prob_info:
                    kl_val = self._last_prob_info.get("kl")
                    if kl_val is not None and np.isfinite(kl_val):
                        epoch_kl_values.append(float(kl_val))
                    for k in epoch_prob_stats.keys():
                        v = self._last_prob_info.get(k)
                        if v is not None and np.isfinite(v):
                            epoch_prob_stats[k].append(float(v))
                
                if verbose:
                    # Update progress bar with current loss
                    current_avg_loss = np.mean(epoch_losses)
                    pbar.set_postfix({"loss": f"{current_avg_loss:.4f}"})
            
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            mean_epoch_kl = float(np.mean(epoch_kl_values)) if epoch_kl_values else float("nan")
            if self.is_probabilistic_model and np.isfinite(mean_epoch_kl):
                if mean_epoch_kl < self.collapse_kl_threshold:
                    self._collapse_counter += 1
                else:
                    self._collapse_counter = 0
                if self._collapse_counter >= self.collapse_patience_epochs and verbose:
                    print(
                        "  Warning: possible posterior collapse detected "
                        f"(mean KL {mean_epoch_kl:.4f} < {self.collapse_kl_threshold:.4f} "
                        f"for {self._collapse_counter} epoch(s))."
                    )
            
            # Validation
            if verbose:
                print("  Validating...", flush=True)
            val_start_time = time.time()
            val_loss = self._validate()
            val_losses.append(val_loss)
            val_time = time.time() - val_start_time
            if verbose:
                print(f"  Validation done ({val_time:.2f}s)")
            if val_loss is not None and np.isfinite(val_loss):
                if best_val_loss is None or float(val_loss) < best_val_loss:
                    best_val_loss = float(val_loss)

            checkpoint_score, checkpoint_label = self._checkpoint_score(val_loss)
            if np.isfinite(checkpoint_score):
                if best_checkpoint_score is None or checkpoint_score < best_checkpoint_score:
                    prev_best = best_checkpoint_score
                    best_checkpoint_score = float(checkpoint_score)
                    best_checkpoint_label = checkpoint_label
                    # Keep a CPU copy so we can restore best weights after training.
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    if checkpoint_target is not None:
                        best_bundle = {
                            "epoch": int(epoch),
                            "checkpoint_label": str(checkpoint_label),
                            "checkpoint_score": float(checkpoint_score),
                            "model_state_dict": best_state_dict,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "trainer_state": {
                                "global_step": int(self._global_step),
                                "fit_epochs": int(self._fit_epochs),
                                "current_epoch": int(self._current_epoch),
                                "base_lrs": list(self._base_lrs),
                            },
                        }
                        if self.checkpoint_metadata:
                            best_bundle["metadata"] = self.checkpoint_metadata
                        torch.save(best_bundle, checkpoint_target)
                        if verbose:
                            if prev_best is None:
                                print(
                                    f"  Saved checkpoint: {checkpoint_target} "
                                    f"({checkpoint_label}={checkpoint_score:.6f})"
                                )
                            else:
                                print(
                                    f"  Saved checkpoint: {checkpoint_target} "
                                    f"({checkpoint_label} improved "
                                    f"{prev_best:.6f} -> {checkpoint_score:.6f})"
                                )
                        if on_checkpoint_saved is not None:
                            try:
                                on_checkpoint_saved(epoch, float(checkpoint_score))
                            except Exception as cb_exc:
                                if verbose:
                                    print(f"  Warning: checkpoint callback failed: {cb_exc}")
                if checkpoint_bundle_dir is not None:
                    should_save_topk = (
                        len(topk_checkpoints) < self.checkpoint_top_k
                        or checkpoint_score < max(s for s, _ in topk_checkpoints)
                    )
                    if should_save_topk:
                        bundle_name = (
                            f"epoch{epoch:04d}_{checkpoint_label}_"
                            f"{checkpoint_score:.6f}.pth"
                        ).replace(":", "_")
                        bundle_path = checkpoint_bundle_dir / bundle_name
                        checkpoint_bundle = {
                            "epoch": int(epoch),
                            "checkpoint_label": str(checkpoint_label),
                            "checkpoint_score": float(checkpoint_score),
                            "model_state_dict": {
                                k: v.detach().cpu().clone()
                                for k, v in self.model.state_dict().items()
                            },
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "trainer_state": {
                                "global_step": int(self._global_step),
                                "fit_epochs": int(self._fit_epochs),
                                "current_epoch": int(self._current_epoch),
                                "base_lrs": list(self._base_lrs),
                            },
                        }
                        if self.checkpoint_metadata:
                            checkpoint_bundle["metadata"] = self.checkpoint_metadata
                        torch.save(checkpoint_bundle, bundle_path)
                        topk_checkpoints.append((float(checkpoint_score), bundle_path))
                        topk_checkpoints.sort(key=lambda x: x[0])
                        while len(topk_checkpoints) > self.checkpoint_top_k:
                            _, rm_path = topk_checkpoints.pop(-1)
                            try:
                                rm_path.unlink(missing_ok=True)
                            except Exception:
                                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            elapsed_time = time.time() - start_time
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = epochs - epoch
            eta = avg_epoch_time * remaining_epochs if remaining_epochs > 0 else 0
            
            # Logging - now print every epoch when verbose
            if verbose:
                msg = f"[Epoch {epoch}/{epochs}] "
                msg += f"train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                if best_checkpoint_score is not None:
                    msg += (
                        f" | ckpt_{best_checkpoint_label}={best_checkpoint_score:.6f}"
                    )
                if self.is_probabilistic_model:
                    msg += f" | kl_beta={self._current_kl_beta():.4f}"
                    if np.isfinite(mean_epoch_kl):
                        msg += f" | kl_mean={mean_epoch_kl:.4f}"
                    recon_mean = np.mean(epoch_prob_stats["recon_nll"]) if epoch_prob_stats["recon_nll"] else float("nan")
                    aux_mean = np.mean(epoch_prob_stats["aux_nll"]) if epoch_prob_stats["aux_nll"] else float("nan")
                    roll_mean = np.mean(epoch_prob_stats["rollout_nll"]) if epoch_prob_stats["rollout_nll"] else float("nan")
                    horizon_mean = np.mean(epoch_prob_stats["horizon_schedule"]) if epoch_prob_stats["horizon_schedule"] else float("nan")
                    ramp_mean = np.mean(epoch_prob_stats["rollout_ramp"]) if epoch_prob_stats["rollout_ramp"] else float("nan")
                    grad_mean = np.mean(epoch_prob_stats["grad_norm"]) if epoch_prob_stats["grad_norm"] else float("nan")
                    lr_mean = np.mean(epoch_prob_stats["lr"]) if epoch_prob_stats["lr"] else float("nan")
                    if np.isfinite(recon_mean):
                        msg += f" | recon={recon_mean:.4f}"
                    if np.isfinite(aux_mean):
                        msg += f" | aux={aux_mean:.4f}"
                    if np.isfinite(roll_mean):
                        msg += f" | rollout_nll={roll_mean:.4f}"
                    if np.isfinite(horizon_mean):
                        msg += f" | h={horizon_mean:.2f}"
                    if np.isfinite(ramp_mean):
                        msg += f" | ramp={ramp_mean:.2f}"
                    if np.isfinite(grad_mean):
                        msg += f" | grad={grad_mean:.2f}"
                    if np.isfinite(lr_mean):
                        msg += f" | lr={lr_mean:.2e}"
                msg += f" | time={epoch_time:.2f}s"
                if epoch > 1:
                    msg += f" | ETA={eta:.1f}s"
                print(msg)
            
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                if best_checkpoint_score is not None:
                    self.writer.add_scalar(
                        f"Checkpoint/{best_checkpoint_label}",
                        best_checkpoint_score,
                        epoch,
                    )
                if self.is_probabilistic_model:
                    for k, vals in epoch_prob_stats.items():
                        if vals:
                            self.writer.add_scalar(f"Epoch/{k}", float(np.mean(vals)), epoch)
                self.writer.add_scalar("Time/epoch", epoch_time, epoch)
            
            if self.metrics_path is not None:
                def _m(name: str) -> str:
                    vals = epoch_prob_stats.get(name, [])
                    if not vals:
                        return ""
                    v = float(np.mean(vals))
                    return f"{v:.10f}" if np.isfinite(v) else ""
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},{train_loss},{val_loss if val_loss is not None else ''},"
                        f"{_m('loss_std')},{_m('loss_total')},"
                        f"{_m('recon_nll')},{_m('kl')},{_m('kl_raw')},{_m('aux_nll')},"
                        f"{_m('rollout_nll')},{_m('rollout_dtw')},{_m('rollout_total')},"
                        f"{_m('rollout_weight_eff')},{_m('rollout_ramp')},"
                        f"{_m('horizon_schedule')},{_m('context_len')},{_m('grad_norm')},{_m('grad_norm_pre')},{_m('lr')}\n"
                    )
            
            # Early stopping
            if self.early_stopping:
                monitor_value: Optional[float] = val_loss
                monitor_label = "val_loss"
                if (
                    self.early_stopping_monitor in {"open_loop_crps", "checkpoint_metric"}
                    and np.isfinite(checkpoint_score)
                ):
                    monitor_value = float(checkpoint_score)
                    monitor_label = str(checkpoint_label)
                self.early_stopping(monitor_value)
                if self.early_stopping.early_stop:
                    if verbose:
                        print(f"\n{'='*70}")
                        print("Early stopping triggered.")
                        print(
                            f"Reason: no validation improvement for "
                            f"{self.early_stopping.counter} epoch(s) "
                            f"(patience={self.early_stopping.patience})."
                        )
                        if monitor_value is not None and np.isfinite(monitor_value):
                            print(
                                f"Current monitored {monitor_label}: "
                                f"{float(monitor_value):.6f}"
                            )
                        if self.early_stopping.best_loss is not None:
                            print(
                                f"Best monitored {monitor_label}: "
                                f"{self.early_stopping.best_loss:.6f}"
                            )
                            if monitor_value is not None and np.isfinite(monitor_value):
                                diff = float(monitor_value) - float(self.early_stopping.best_loss)
                                print(f"Monitored gap vs best: {diff:+.6f}")
                        print(f"Stopped at epoch {epoch}/{epochs}")
                        print(f"{'='*70}\n")
                    break

        # Ensure caller gets the best validation model, not the final epoch model.
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training completed!")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
            if len(train_losses) > 0:
                print(f"Final train loss: {train_losses[-1]:.6f}")
                if val_losses[-1] is not None:
                    print(f"Final val loss: {val_losses[-1]:.6f}")
                if best_val_loss is not None:
                    print(f"Best val loss: {best_val_loss:.6f}")
                if best_checkpoint_score is not None:
                    print(
                        f"Best checkpoint {best_checkpoint_label}: "
                        f"{best_checkpoint_score:.6f}"
                    )
                if checkpoint_bundle_dir is not None and topk_checkpoints:
                    print(
                        f"Saved top-{len(topk_checkpoints)} checkpoints to: "
                        f"{checkpoint_bundle_dir}"
                    )
            print(f"{'='*70}\n")
        
        return train_losses, val_losses
    
    def save(self, path: str | Path):
        """Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to save checkpoint.
        """
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str | Path):
        """Load model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to load checkpoint from.
        """
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()


# Backward compatibility: alias to old Trainer
class Trainer(nn.Module):
    """Legacy trainer for backward compatibility.
    
    This maintains the old Trainer interface for existing code.
    For new code, use WorldModelTrainer instead.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss: str = "mse",
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device | str = "cpu",
        early_stopping: bool = False,
        patience: int = 5,
        run_dir: Optional[str] = None,
        writer: Optional["SummaryWriter"] = None,
    ):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "dilate":
            self.loss_fn = None  # Will use dilate_loss function
        else:
            raise ValueError("loss must be mse or dilate")
        
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_type = loss
        self.early_stopping = EarlyStopping(patience=patience) if early_stopping else None
        
        # Logging
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if writer is not None:
                self.writer = writer
            else:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(log_dir=self.run_dir)
                except ModuleNotFoundError:
                    self.writer = None
            
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,train_loss,val_loss\n")
        else:
            self.writer = writer
            self.metrics_path = None
    
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        
        if self.loss_type == "mse":
            loss = self.loss_fn(pred, y)
        else:
            from ..utils.dilate import dilate_loss
            loss, _, _ = dilate_loss(y, pred, device=self.device)
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN/Inf in training loss.")
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader):
        self.model.eval()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            
            if self.loss_type == "mse":
                batch_loss = self.loss_fn(pred, y)
            else:
                from ..utils.dilate import dilate_loss
                batch_loss, _, _ = dilate_loss(y, pred, device=self.device)
            
            total += batch_loss.item() * len(x)
            n += len(x)
        self.model.train()
        return total / max(n, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ):
        train_losses, val_losses = [], []
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch in train_loader:
                batch_loss = self._step(batch)
                epoch_losses.append(batch_loss)
            
            train_loss = sum(epoch_losses) / len(epoch_losses)
            val_loss = self._validate(val_loader) if val_loader else None
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose and (epoch == 1 or epoch == epochs or epoch % 5 == 0):
                msg = f"[Epoch {epoch}/{epochs}] train={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val={val_loss:.4f}"
                print(msg)
            
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    if verbose:
                        print("Early stopping triggered.")
                        print(
                            f"Reason: no validation improvement for "
                            f"{self.early_stopping.counter} epoch(s) "
                            f"(patience={self.early_stopping.patience})."
                        )
                        if val_loss is not None:
                            print(f"Current validation loss: {val_loss:.6f}")
                        if self.early_stopping.best_loss is not None:
                            print(f"Best validation loss: {self.early_stopping.best_loss:.6f}")
                    break
            
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
            
            if self.metrics_path is not None:
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{train_loss},{val_loss if val_loss is not None else ''}\n")
        
        return train_losses, val_losses
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
