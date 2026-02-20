import torch
import torch.nn.functional as F


def dilate_loss(target: torch.Tensor,
                prediction: torch.Tensor,
                alpha: float = 0.5,
                gamma: float = 0.01,
                device: torch.device | str = "cpu"):
    """Compute DILATE loss (shape + temporal) for 1-D sequences.
    This is a *simplified* version adequate for small tests.
    Original paper: Shape and time distortions measure.
    """
    device = torch.device(device)
    target = target.to(device)
    prediction = prediction.to(device)
    batch_size, seq_len = target.shape[0], target.shape[1]

    # Shape loss (MSE)
    loss_shape = F.mse_loss(prediction, target)

    # Temporal loss (soft-DTW approximation)
    # Here we use a crude approximation: mean absolute difference of first derivatives
    def _first_derivative(x):
        return x[:, 1:] - x[:, :-1]

    loss_temporal = F.l1_loss(_first_derivative(prediction), _first_derivative(target))

    loss = alpha * loss_shape + (1 - alpha) * (loss_temporal + gamma)
    return loss, loss_shape, loss_temporal 