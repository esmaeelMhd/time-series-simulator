"""Loss functions for world model training.

HOT PATH: Loss computation happens every training step.
Optimizations:
- Use vectorized PyTorch operations (no Python loops)
- Avoid intermediate tensor allocations
- Use in-place operations where safe
"""

from __future__ import annotations

from typing import Literal, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LossType = Literal["mse", "mae", "huber", "shape"]

_DEFAULT_SHAPE_LOSS_CFG: Dict[str, float] = {
    "w_level": 0.5,
    "w_slope": 0.3,
    "w_curvature": 0.1,
    "w_stats": 0.1,
    "robust_beta": 1.0,
}


def _merged_shape_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    out = dict(_DEFAULT_SHAPE_LOSS_CFG)
    if cfg:
        for k in out:
            if k in cfg:
                out[k] = float(cfg[k])
    return out


def _weighted_time_mean(step_losses: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Average losses over time with optional per-step weights.

    step_losses shape: (batch, horizon)
    """
    if step_losses.numel() == 0:
        return torch.zeros((), dtype=step_losses.dtype, device=step_losses.device)
    if weights is None:
        return step_losses.mean()
    return (step_losses * weights.unsqueeze(0)).mean()


def _shape_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    cfg: Dict[str, float],
    time_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast DTW-like surrogate: level + slope + curvature + summary stats.

    Complexity is linear in horizon and fully vectorized.
    """
    beta = max(1e-6, cfg["robust_beta"])
    w_level = cfg["w_level"]
    w_slope = cfg["w_slope"]
    w_curv = cfg["w_curvature"]
    w_stats = cfg["w_stats"]

    # Level term: robust point-wise fit
    level_steps = F.smooth_l1_loss(predictions, targets, beta=beta, reduction="none").mean(dim=-1)
    loss_level = _weighted_time_mean(level_steps, time_weights)

    # Slope term: first-derivative dynamics matching
    if predictions.shape[1] > 1:
        dp = predictions[:, 1:, :] - predictions[:, :-1, :]
        dt = targets[:, 1:, :] - targets[:, :-1, :]
        slope_steps = F.smooth_l1_loss(dp, dt, beta=beta, reduction="none").mean(dim=-1)
        slope_w = time_weights[1:] if time_weights is not None else None
        loss_slope = _weighted_time_mean(slope_steps, slope_w)
    else:
        loss_slope = torch.zeros_like(loss_level)

    # Curvature term: second-derivative dynamics matching
    if predictions.shape[1] > 2:
        d2p = dp[:, 1:, :] - dp[:, :-1, :]
        d2t = dt[:, 1:, :] - dt[:, :-1, :]
        curv_steps = F.smooth_l1_loss(d2p, d2t, beta=beta, reduction="none").mean(dim=-1)
        curv_w = time_weights[2:] if time_weights is not None else None
        loss_curv = _weighted_time_mean(curv_steps, curv_w)
    else:
        loss_curv = torch.zeros_like(loss_level)

    # Cheap long-term shape statistics (trend/amplitude summary)
    pred_mean = predictions.mean(dim=1)
    targ_mean = targets.mean(dim=1)
    pred_std = predictions.std(dim=1, unbiased=False)
    targ_std = targets.std(dim=1, unbiased=False)
    loss_stats = 0.5 * F.mse_loss(pred_mean, targ_mean) + 0.5 * F.mse_loss(pred_std, targ_std)

    denom = max(1e-8, w_level + w_slope + w_curv + w_stats)
    total = (
        w_level * loss_level +
        w_slope * loss_slope +
        w_curv * loss_curv +
        w_stats * loss_stats
    ) / denom
    return total


class OneStepLoss(nn.Module):
    """Standard one-step prediction loss with teacher forcing.
    
    This is the traditional supervised learning loss where the model
    predicts one step ahead given ground truth history.
    
    Parameters
    ----------
    loss_type : {"mse", "mae", "huber", "shape"}
        Type of loss function.
    """
    
    def __init__(
        self,
        loss_type: LossType = "mse",
        shape_loss_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.shape_loss_cfg = _merged_shape_cfg(shape_loss_cfg)
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        elif loss_type == "shape":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one-step loss.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape (batch_size, horizon, output_dim).
        targets : torch.Tensor
            Ground truth targets, shape (batch_size, horizon, output_dim).
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if self.loss_type == "shape":
            return _shape_loss(predictions, targets, self.shape_loss_cfg)
        return self.loss_fn(predictions, targets)


class MultiStepLoss(nn.Module):
    """Multi-step prediction loss for world model training.
    
    This loss computes the error over multiple autoregressive steps,
    which helps reduce compounding errors in long-horizon predictions.
    
    Parameters
    ----------
    loss_type : {"mse", "mae", "huber", "shape"}
        Base loss function type.
    weighting : {"uniform", "linear", "exponential"}
        How to weight losses across time steps:
        - "uniform": All steps weighted equally
        - "linear": Linearly increasing weights (emphasize later steps)
        - "exponential": Exponentially increasing weights
    weight_scale : float, default 1.0
        Scaling factor for non-uniform weighting.
    """
    
    def __init__(
        self,
        loss_type: LossType = "mse",
        weighting: Literal["uniform", "linear", "exponential"] = "uniform",
        weight_scale: float = 1.0,
        shape_loss_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weighting = weighting
        self.weight_scale = weight_scale
        self.shape_loss_cfg = _merged_shape_cfg(shape_loss_cfg)
        
        if loss_type == "mse":
            self.base_loss = F.mse_loss
        elif loss_type == "mae":
            self.base_loss = F.l1_loss
        elif loss_type == "huber":
            self.base_loss = F.huber_loss
        elif loss_type == "shape":
            self.base_loss = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_weights(self, horizon: int, device: torch.device) -> torch.Tensor:
        """Compute time-step weights.
        
        Parameters
        ----------
        horizon : int
            Number of time steps.
        device : torch.device
            Device for the weights tensor.
        
        Returns
        -------
        torch.Tensor
            Weights of shape (horizon,).
        """
        if self.weighting == "uniform":
            weights = torch.ones(horizon, device=device)
        elif self.weighting == "linear":
            # Linearly increasing: 1, 2, 3, ..., horizon
            weights = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
            weights = weights * self.weight_scale
        elif self.weighting == "exponential":
            # Exponentially increasing: exp(0), exp(1), ..., exp(horizon-1)
            weights = torch.exp(
                torch.arange(horizon, device=device, dtype=torch.float32) * self.weight_scale
            )
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
        
        # Normalize weights to sum to horizon (so average is comparable to uniform)
        weights = weights / weights.mean()
        return weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-step loss.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape (batch_size, horizon, output_dim).
        targets : torch.Tensor
            Ground truth targets, shape (batch_size, horizon, output_dim).
        weights : torch.Tensor, optional
            Custom per-step weights, shape (horizon,).
            If None, uses the weighting scheme from __init__.
        
        Returns
        -------
        torch.Tensor
            Scalar loss value (weighted average over time steps).
        """
        batch_size, horizon, output_dim = predictions.shape
        device = predictions.device
        
        # Get time-step weights
        if weights is None:
            weights = self._compute_weights(horizon, device)

        if self.loss_type == "shape":
            return _shape_loss(predictions, targets, self.shape_loss_cfg, time_weights=weights)

        # Compute per-step losses: (batch_size, horizon, output_dim)
        if self.loss_type == "mse":
            step_losses = (predictions - targets) ** 2
        elif self.loss_type == "mae":
            step_losses = torch.abs(predictions - targets)
        elif self.loss_type == "huber":
            step_losses = F.smooth_l1_loss(predictions, targets, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Average over output dimensions: (batch_size, horizon)
        step_losses = step_losses.mean(dim=-1)
        weighted_losses = step_losses * weights.unsqueeze(0)
        return weighted_losses.mean()


class CombinedLoss(nn.Module):
    """Combined one-step and multi-step loss.
    
    This loss combines teacher-forced one-step prediction with multi-step
    autoregressive rollout loss. The combination helps with both:
    1. Learning accurate single-step dynamics (one-step loss)
    2. Reducing compounding errors (multi-step loss)
    
    Parameters
    ----------
    one_step_weight : float, default 0.5
        Weight for one-step loss (lambda in the paper).
    multi_step_weight : float, default 0.5
        Weight for multi-step loss (1 - lambda).
    loss_type : {"mse", "mae", "huber", "shape"}
        Base loss function type.
    multi_step_weighting : {"uniform", "linear", "exponential"}
        Weighting scheme for multi-step loss.
    multi_step_weight_scale : float, default 1.0
        Scale factor for the multi-step weighting scheme.
    """
    
    def __init__(
        self,
        one_step_weight: float = 0.5,
        multi_step_weight: float = 0.5,
        loss_type: LossType = "mse",
        multi_step_weighting: Literal["uniform", "linear", "exponential"] = "uniform",
        multi_step_weight_scale: float = 1.0,
        shape_loss_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.one_step_weight = one_step_weight
        self.multi_step_weight = multi_step_weight
        
        self.one_step_loss = OneStepLoss(
            loss_type=loss_type,
            shape_loss_cfg=shape_loss_cfg,
        )
        self.multi_step_loss = MultiStepLoss(
            loss_type=loss_type,
            weighting=multi_step_weighting,
            weight_scale=multi_step_weight_scale,
            shape_loss_cfg=shape_loss_cfg,
        )
    
    def forward(
        self,
        predictions_teacher: torch.Tensor,
        predictions_model: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Parameters
        ----------
        predictions_teacher : torch.Tensor
            Predictions with teacher forcing, shape (batch_size, horizon, output_dim).
        predictions_model : torch.Tensor
            Predictions with model feedback (autoregressive),
            shape (batch_size, horizon, output_dim).
        targets : torch.Tensor
            Ground truth targets, shape (batch_size, horizon, output_dim).
        
        Returns
        -------
        loss : torch.Tensor
            Combined scalar loss.
        info : dict
            Dictionary with individual loss components for logging.
        """
        loss_1step = self.one_step_loss(predictions_teacher, targets)
        loss_multi = self.multi_step_loss(predictions_model, targets)
        
        total_loss = (
            self.one_step_weight * loss_1step +
            self.multi_step_weight * loss_multi
        )
        
        info = {
            "loss_1step": loss_1step.item(),
            "loss_multi": loss_multi.item(),
            "loss_total": total_loss.item(),
        }
        
        return total_loss, info


class ProbabilisticRolloutLoss(nn.Module):
    """RSSM probabilistic loss: reconstruction NLL + balanced/free-bits KL + aux NLL."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        aux_weight: float = 1.0,
        kl_free_bits: float = 1.0,
        kl_balance: float = 0.8,
        use_kl_balancing: bool = True,
        use_free_bits: bool = True,
        use_symlog: bool = False,
        # Backward-compat aliases:
        elbo_weight: Optional[float] = None,
        rollout_mse_weight: Optional[float] = None,
    ):
        super().__init__()
        if elbo_weight is not None:
            recon_weight = float(elbo_weight)
        if rollout_mse_weight is not None:
            aux_weight = float(rollout_mse_weight)
        self.recon_weight = float(recon_weight)
        self.kl_weight = float(kl_weight)
        self.aux_weight = float(aux_weight)
        self.kl_free_bits = float(kl_free_bits)
        self.kl_balance = float(kl_balance)
        self.use_kl_balancing = bool(use_kl_balancing)
        self.use_free_bits = bool(use_free_bits)
        self.use_symlog = bool(use_symlog)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean()
        mask_dims = mask.dim()
        mask_f = mask.to(dtype=x.dtype)
        while mask_f.dim() < x.dim():
            mask_f = mask_f.unsqueeze(-1)
        extra = 1
        for d in range(mask_dims, x.dim()):
            extra *= int(x.shape[d])
        denom = (mask_f.sum() * float(extra)).clamp_min(1.0)
        return (x * mask_f).sum() / denom

    @staticmethod
    def _sum_time_mean_batch(step_losses: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Reduce (B,T) step losses as sum over time then mean over batch."""
        if step_losses.numel() == 0:
            return torch.zeros((), dtype=step_losses.dtype, device=step_losses.device)
        if step_losses.dim() != 2:
            raise ValueError(
                f"Expected step_losses with shape (B,T), got {tuple(step_losses.shape)}"
            )
        if mask is None:
            return step_losses.sum(dim=1).mean()
        if mask.shape != step_losses.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} must match step losses {tuple(step_losses.shape)}"
            )
        mask_f = mask.to(dtype=step_losses.dtype)
        return (step_losses * mask_f).sum(dim=1).mean()

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def _normal_from_mu_logvar(mu: torch.Tensor, logvar: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(
            loc=mu,
            scale=torch.exp(0.5 * logvar).clamp_min(1e-6),
        )

    def _balanced_kl(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        post = self._normal_from_mu_logvar(posterior_mu, posterior_logvar)
        prior = self._normal_from_mu_logvar(prior_mu, prior_logvar)
        raw_kl = torch.distributions.kl_divergence(post, prior)
        if not self.use_kl_balancing:
            return raw_kl
        post_sg = self._normal_from_mu_logvar(
            posterior_mu.detach(),
            posterior_logvar.detach(),
        )
        prior_sg = self._normal_from_mu_logvar(
            prior_mu.detach(),
            prior_logvar.detach(),
        )
        kl_prior_fit = torch.distributions.kl_divergence(post_sg, prior)
        kl_post_fit = torch.distributions.kl_divergence(post, prior_sg)
        return self.kl_balance * kl_prior_fit + (1.0 - self.kl_balance) * kl_post_fit

    def forward(
        self,
        targets: torch.Tensor,
        dist_loc_latent: Optional[torch.Tensor],
        dist_scale: Optional[torch.Tensor],
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        recon_dist: Optional[torch.distributions.Distribution] = None,
        exogenous_targets: Optional[torch.Tensor] = None,
        aux_loc: Optional[torch.Tensor] = None,
        aux_scale: Optional[torch.Tensor] = None,
        aux_dist: Optional[torch.distributions.Distribution] = None,
        mask: Optional[torch.Tensor] = None,
        kl_beta: float = 1.0,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        recon_nll, kl, aux_nll = self.compute_terms(
            targets=targets,
            dist_loc_latent=dist_loc_latent,
            dist_scale=dist_scale,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            recon_dist=recon_dist,
            exogenous_targets=exogenous_targets,
            aux_loc=aux_loc,
            aux_scale=aux_scale,
            aux_dist=aux_dist,
            mask=mask,
        )
        total = (
            self.recon_weight * recon_nll
            + self.kl_weight * float(kl_beta) * kl
            + self.aux_weight * aux_nll
        )
        info = {
            "loss_total": float(total.detach().item()),
            "recon_nll": float(recon_nll.detach().item()),
            "kl": float(kl.detach().item()),
            "aux_nll": float(aux_nll.detach().item()),
            "kl_beta": float(kl_beta),
        }
        return total, info

    def compute_terms(
        self,
        targets: torch.Tensor,
        dist_loc_latent: Optional[torch.Tensor],
        dist_scale: Optional[torch.Tensor],
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        recon_dist: Optional[torch.distributions.Distribution] = None,
        exogenous_targets: Optional[torch.Tensor] = None,
        aux_loc: Optional[torch.Tensor] = None,
        aux_scale: Optional[torch.Tensor] = None,
        aux_dist: Optional[torch.distributions.Distribution] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_nll = self.compute_recon_nll(
            targets=targets,
            dist_loc_latent=dist_loc_latent,
            dist_scale=dist_scale,
            recon_dist=recon_dist,
            mask=mask,
        )

        kl_elem = self._balanced_kl(
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )
        if self.use_free_bits:
            kl_elem = torch.maximum(
                kl_elem,
                torch.full_like(kl_elem, fill_value=self.kl_free_bits),
            )
        kl_steps = kl_elem.sum(dim=-1)
        kl = self._sum_time_mean_batch(kl_steps, mask)

        aux_nll = torch.zeros((), dtype=recon_nll.dtype, device=recon_nll.device)
        if (
            exogenous_targets is not None
            and exogenous_targets.shape[-1] > 0
        ):
            aux_nll = self.compute_aux_nll(
                targets=exogenous_targets,
                aux_loc=aux_loc,
                aux_scale=aux_scale,
                aux_dist=aux_dist,
                mask=mask,
            )

        return recon_nll, kl, aux_nll

    def compute_recon_nll(
        self,
        targets: torch.Tensor,
        dist_loc_latent: Optional[torch.Tensor],
        dist_scale: Optional[torch.Tensor],
        recon_dist: Optional[torch.distributions.Distribution] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruction NLL using distribution.log_prob()."""
        targets_latent = self._symlog(targets) if self.use_symlog else targets
        if recon_dist is None:
            if dist_loc_latent is None or dist_scale is None:
                raise ValueError("Either recon_dist or dist_loc_latent/dist_scale must be provided.")
            recon_dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=dist_loc_latent, scale=dist_scale),
                1,
            )
        recon_nll_steps = -recon_dist.log_prob(targets_latent)
        if recon_nll_steps.dim() != 2:
            raise ValueError(
                f"Expected reconstruction log_prob to return (B,T), got {tuple(recon_nll_steps.shape)}"
            )
        return self._sum_time_mean_batch(recon_nll_steps, mask)

    def compute_aux_nll(
        self,
        targets: torch.Tensor,
        aux_loc: Optional[torch.Tensor],
        aux_scale: Optional[torch.Tensor],
        aux_dist: Optional[torch.distributions.Distribution] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Auxiliary exogenous NLL using distribution.log_prob()."""
        if aux_dist is None:
            if aux_loc is None or aux_scale is None:
                raise ValueError("Either aux_dist or aux_loc/aux_scale must be provided.")
            aux_dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=aux_loc, scale=aux_scale),
                1,
            )
        aux_nll_steps = -aux_dist.log_prob(targets)
        if aux_nll_steps.dim() != 2:
            raise ValueError(
                f"Expected auxiliary log_prob to return (B,T), got {tuple(aux_nll_steps.shape)}"
            )
        return self._sum_time_mean_batch(aux_nll_steps, mask)


def soft_dtw_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Differentiable Soft-DTW distance averaged across output dimensions.

    Parameters
    ----------
    x, y : torch.Tensor
        Shape (batch, time, dim).
    gamma : float
        Smoothing parameter (>0).
    """
    if x.shape != y.shape:
        raise ValueError(f"soft_dtw_distance requires matching shapes, got {x.shape} vs {y.shape}")
    if x.dim() != 3:
        raise ValueError(f"soft_dtw_distance expects rank-3 tensors, got rank {x.dim()}")
    bsz, tlen, _ = x.shape
    if tlen == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)

    g = max(1e-6, float(gamma))
    # Pairwise squared distances: (B, T, T)
    dist = torch.cdist(x, y, p=2) ** 2
    inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    r = torch.full((bsz, tlen + 1, tlen + 1), inf, dtype=x.dtype, device=x.device)
    r[:, 0, 0] = 0.0

    for i in range(1, tlen + 1):
        for j in range(1, tlen + 1):
            prev = torch.stack(
                [
                    r[:, i - 1, j],
                    r[:, i, j - 1],
                    r[:, i - 1, j - 1],
                ],
                dim=-1,
            )
            softmin = -g * torch.logsumexp(-prev / g, dim=-1)
            r[:, i, j] = dist[:, i - 1, j - 1] + softmin
    return r[:, tlen, tlen].mean()


def dilate_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 0.01,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DILATE loss (shape + temporal) for sequences.
    
    This is a simplified version of the DILATE loss from the paper:
    "A Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models"
    
    Parameters
    ----------
    target : torch.Tensor
        Ground truth, shape (batch_size, seq_len, features).
    prediction : torch.Tensor
        Predictions, shape (batch_size, seq_len, features).
    alpha : float, default 0.5
        Weight for shape loss vs temporal loss.
    gamma : float, default 0.01
        Regularization for temporal loss.
    device : torch.device or str
        Device for computation.
    
    Returns
    -------
    loss : torch.Tensor
        Total DILATE loss.
    loss_shape : torch.Tensor
        Shape component (MSE).
    loss_temporal : torch.Tensor
        Temporal component (derivative matching).
    """
    device = torch.device(device)
    target = target.to(device)
    prediction = prediction.to(device)
    
    # Shape loss (MSE)
    loss_shape = F.mse_loss(prediction, target)
    
    # Temporal loss (first derivative matching)
    def _first_derivative(x):
        return x[:, 1:] - x[:, :-1]
    
    loss_temporal = F.l1_loss(_first_derivative(prediction), _first_derivative(target))
    
    # Combined loss
    loss = alpha * loss_shape + (1 - alpha) * (loss_temporal + gamma)
    
    return loss, loss_shape, loss_temporal
