import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def crps_ensemble(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CRPS for an ensemble forecast.

    Parameters
    ----------
    samples : torch.Tensor
        Shape (n_samples, ..., output_dim).
    target : torch.Tensor
        Shape (..., output_dim).
    """
    if samples.dim() < 2:
        raise ValueError("samples must be at least rank-2: (n_samples, ...)")
    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"samples shape {tuple(samples.shape)} incompatible with target shape {tuple(target.shape)}"
        )
    n = samples.shape[0]
    if n < 1:
        raise ValueError("samples must contain at least one ensemble member")

    term_1 = torch.mean(torch.abs(samples - target.unsqueeze(0)), dim=0)
    diffs = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))
    term_2 = 0.5 * torch.mean(diffs, dim=(0, 1))
    return term_1 - term_2


def interval_coverage(
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> float:
    inside = (target >= lower) & (target <= upper)
    return float(inside.float().mean().item())
