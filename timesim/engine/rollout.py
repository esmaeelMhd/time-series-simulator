from __future__ import annotations

import torch
from torch.nn import Module


def rollout_autoregressive(model: Module,
                           x0: torch.Tensor,  # (B, seq_len, F_in)
                           h_max: int,
                           device: torch.device | str = "cpu") -> torch.Tensor:
    """Unroll *model* autoregressively for *h_max* steps.

    Assumes model returns a sequence of length pred_len; we take only the last
    timestep (pred_len==1 recommended).  The output tensor has shape
    (B, h_max, F_out).
    """
    device = torch.device(device)
    model.eval()
    preds = []
    x = x0.to(device)
    with torch.no_grad():
        for _ in range(h_max):
            y = model(x)  # (B, pred_len, F_out)
            step = y[:, -1:, :]          # last step, keep time dim 1
            preds.append(step.cpu())
            x = torch.cat([x, step], dim=1)[:, 1:, :]  # slide window
    return torch.cat(preds, dim=1)  # (B, h_max, F_out) 