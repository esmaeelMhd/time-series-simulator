"""Loss functions for world model training.

HOT PATH: Loss computation happens every training step.
Optimizations:
- Use vectorized PyTorch operations (no Python loops)
- Avoid intermediate tensor allocations
- Use in-place operations where safe
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneStepLoss(nn.Module):
    """Standard one-step prediction loss with teacher forcing.
    
    This is the traditional supervised learning loss where the model
    predicts one step ahead given ground truth history.
    
    Parameters
    ----------
    loss_type : {"mse", "mae", "huber"}
        Type of loss function.
    """
    
    def __init__(self, loss_type: Literal["mse", "mae", "huber"] = "mse"):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
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
        return self.loss_fn(predictions, targets)


class MultiStepLoss(nn.Module):
    """Multi-step prediction loss for world model training.
    
    This loss computes the error over multiple autoregressive steps,
    which helps reduce compounding errors in long-horizon predictions.
    
    Parameters
    ----------
    loss_type : {"mse", "mae", "huber"}
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
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        weighting: Literal["uniform", "linear", "exponential"] = "uniform",
        weight_scale: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weighting = weighting
        self.weight_scale = weight_scale
        
        if loss_type == "mse":
            self.base_loss = F.mse_loss
        elif loss_type == "mae":
            self.base_loss = F.l1_loss
        elif loss_type == "huber":
            self.base_loss = F.huber_loss
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
        
        # Compute per-step losses
        # Shape: (batch_size, horizon, output_dim)
        step_losses = (predictions - targets) ** 2 if self.loss_type == "mse" else torch.abs(predictions - targets)
        
        # Average over output dimensions: (batch_size, horizon)
        step_losses = step_losses.mean(dim=-1)
        
        # Get time-step weights
        if weights is None:
            weights = self._compute_weights(horizon, device)
        
        # Weight and average over time: (batch_size,)
        weighted_losses = step_losses * weights.unsqueeze(0)
        
        # Average over batch
        loss = weighted_losses.mean()
        
        return loss


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
    loss_type : {"mse", "mae", "huber"}
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
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        multi_step_weighting: Literal["uniform", "linear", "exponential"] = "uniform",
        multi_step_weight_scale: float = 1.0,
    ):
        super().__init__()
        self.one_step_weight = one_step_weight
        self.multi_step_weight = multi_step_weight
        
        self.one_step_loss = OneStepLoss(loss_type=loss_type)
        self.multi_step_loss = MultiStepLoss(
            loss_type=loss_type,
            weighting=multi_step_weighting,
            weight_scale=multi_step_weight_scale,
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
