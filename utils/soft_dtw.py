"""Soft-DTW implementation for differentiable time warping.

HOT PATH: This module is called during loss computation.
Optimizations:
- Batched CPU/GPU transfers (single transfer per batch)
- Numba JIT for inner loops (unavoidable due to DTW structure)
- Preallocated output tensors
"""

import numpy as np
import torch
from numba import jit
from torch.autograd import Function


def pairwise_distances(x, y=None):
    """Compute pairwise squared Euclidean distances.
    
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] = ||x[i,:]-y[j,:]||^2
    
    HOT PATH: Used for distance matrix computation.
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    """Batched Soft-DTW for differentiable time warping.
    
    HOT PATH: Called during loss computation.
    Optimizations:
    - Single CPU transfer per batch (not per item)
    - Preallocated numpy arrays for results
    - Single GPU transfer after all computations
    """
    
    @staticmethod
    def forward(ctx, D, gamma=1.0):  # D.shape: [batch_size, N, N]
        dev = D.device
        batch_size, N, _ = D.shape
        gamma_tensor = torch.tensor([gamma], dtype=torch.float32, device=dev)
        g_ = gamma
        
        # HOT PATH: Single CPU transfer for entire batch (Rule 7)
        D_ = D.detach().cpu().numpy()
        
        # Preallocate numpy array for results (Rule 5: no allocations in loop)
        R_np = np.zeros((batch_size, N + 2, N + 2), dtype=np.float32)
        total_loss = 0.0
        
        # Numba JIT loop (unavoidable due to DTW recurrence structure)
        for k in range(batch_size):
            R_np[k] = compute_softdtw(D_[k], g_)
            total_loss += R_np[k, -2, -2]
        
        # HOT PATH: Single GPU transfer after all computations (Rule 7)
        R = torch.from_numpy(R_np).to(dev)
        
        ctx.save_for_backward(D, R, gamma_tensor)
        return torch.tensor(total_loss / batch_size, device=dev)
    
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, _ = D.shape
        g_ = gamma.item()
        
        # HOT PATH: Single CPU transfer for entire batch
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        
        # Preallocate numpy array for gradients (Rule 5)
        E_np = np.zeros((batch_size, N, N), dtype=np.float32)
        
        # Numba JIT loop (unavoidable)
        for k in range(batch_size):
            E_np[k] = compute_softdtw_backward(D_[k], R_[k], g_)
        
        # HOT PATH: Single GPU transfer after all computations
        E = torch.from_numpy(E_np).to(dev)
        
        return grad_output * E, None


