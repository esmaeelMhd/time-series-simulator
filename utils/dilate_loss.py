"""DILATE loss for time series forecasting.

HOT PATH: This loss is computed every training step when using DILATE.
Optimizations:
- Batched pairwise distance computation (no Python loop)
- Single tensor allocation for distance matrix
- Precomputed Omega matrix via caching
"""

import torch
from . import soft_dtw
from . import path_soft_dtw

# Module-level cache for Omega matrices (Rule 5: avoid repeated allocations)
_omega_cache: dict[tuple[int, str], torch.Tensor] = {}


def _get_omega(n_output: int, device: torch.device) -> torch.Tensor:
    """Get or create cached Omega matrix for temporal loss.
    
    HOT PATH: Avoids recreating Omega every call.
    """
    cache_key = (n_output, str(device))
    if cache_key not in _omega_cache:
        # Create index range tensor (use arange instead of deprecated range)
        indices = torch.arange(1, n_output + 1, dtype=torch.float32, device=device).view(-1, 1)
        _omega_cache[cache_key] = soft_dtw.pairwise_distances(indices)
    return _omega_cache[cache_key]


def _batched_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances for all batch items at once.
    
    HOT PATH: Vectorized computation, no Python loop over batch.
    
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, N, 1)
    y : torch.Tensor
        Shape (batch_size, M, 1)
    
    Returns
    -------
    torch.Tensor
        Shape (batch_size, N, M) where D[b,i,j] = ||x[b,i] - y[b,j]||^2
    """
    # x: (B, N, 1), y: (B, M, 1)
    # Compute squared distances using broadcasting
    # x_norm: (B, N, 1), y_norm: (B, 1, M)
    x_sq = (x ** 2).squeeze(-1)  # (B, N)
    y_sq = (y ** 2).squeeze(-1)  # (B, M)
    
    # x @ y^T: (B, N, M)
    xy = torch.bmm(x, y.transpose(1, 2))
    
    # dist[b,i,j] = x_sq[b,i] + y_sq[b,j] - 2*xy[b,i,j]
    dist = x_sq.unsqueeze(2) + y_sq.unsqueeze(1) - 2.0 * xy
    return torch.clamp(dist, min=0.0)


def dilate_loss(outputs, targets, alpha, gamma, device):
    """Compute DILATE loss (shape + temporal) for sequences.
    
    HOT PATH: Optimized for batched computation.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions, shape (batch_size, N_output, 1)
    targets : torch.Tensor
        Ground truth, shape (batch_size, N_output, 1)
    alpha : float
        Weight for shape loss (1-alpha for temporal)
    gamma : float
        Softmin temperature parameter
    device : torch.device or str
        Computation device
    
    Returns
    -------
    loss : torch.Tensor
        Total DILATE loss
    loss_shape : torch.Tensor
        Shape component (soft-DTW)
    loss_temporal : torch.Tensor
        Temporal component (path-based)
    """
    batch_size, N_output = outputs.shape[0:2]
    device = torch.device(device)
    
    # HOT PATH: Batched pairwise distance computation (no Python loop)
    # This replaces the for loop over batch_size
    D = _batched_pairwise_distances(targets, outputs)  # (B, N, N)
    
    # Soft-DTW shape loss
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    loss_shape = softdtw_batch(D, gamma)
    
    # Path-based temporal loss
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    
    # Get cached Omega matrix (Rule 5: no repeated allocations)
    Omega = _get_omega(N_output, device)
    
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    
    return loss, loss_shape, loss_temporal