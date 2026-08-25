"""Shape + temporal sequence losses.

``shape_and_temporal_loss`` is a lightweight approximation of the DILATE
paper's shape+time objective (MSE plus first-derivative matching), not the
original DTW-based DILATE loss. ``dilate_loss`` remains as a compatibility alias.
"""

import torch
import torch.nn.functional as F


def shape_and_temporal_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 0.01,
    device: torch.device | str = "cpu",
):
    """Shape (MSE) plus first-derivative temporal penalty."""
    device = torch.device(device)
    target = target.to(device)
    prediction = prediction.to(device)

    loss_shape = F.mse_loss(prediction, target)

    def _first_derivative(x):
        return x[:, 1:] - x[:, :-1]

    loss_temporal = F.l1_loss(_first_derivative(prediction), _first_derivative(target))
    loss = alpha * loss_shape + (1 - alpha) * (loss_temporal + gamma)
    return loss, loss_shape, loss_temporal


def dilate_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 0.01,
    device: torch.device | str = "cpu",
):
    """Backward-compatible alias for :func:`shape_and_temporal_loss`."""
    return shape_and_temporal_loss(
        target, prediction, alpha=alpha, gamma=gamma, device=device
    )
