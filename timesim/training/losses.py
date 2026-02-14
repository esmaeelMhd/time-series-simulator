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
