"""Evaluation helpers for RSSM world models."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import GroupedTimeSeriesDataset
from ..utils.metrics import crps_ensemble


def _window_starts(total_len: int, warmup_len: int, horizon: int, n_windows: int) -> List[int]:
    max_start = total_len - (warmup_len + horizon)
    if max_start < 0:
        return []
    if n_windows <= 1:
        return [0]
    return np.linspace(0, max_start, num=n_windows, dtype=int).tolist()


def _split_controls_exo(dataset: GroupedTimeSeriesDataset, input_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cpos = list(getattr(dataset, "control_positions", []))
    xpos = list(getattr(dataset, "known_exo_positions", []))
    controls = input_arr[:, cpos] if cpos else np.zeros((input_arr.shape[0], 0), dtype=np.float32)
    exogenous = input_arr[:, xpos] if xpos else np.zeros((input_arr.shape[0], 0), dtype=np.float32)
    return controls.astype(np.float32, copy=False), exogenous.astype(np.float32, copy=False)


def _aggregate_error_curves(
    means: List[np.ndarray],
    targets: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    if not means:
        return {"rmse": np.empty((0,), dtype=np.float32), "mae": np.empty((0,), dtype=np.float32)}
    err = np.stack(means, axis=0) - np.stack(targets, axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=(0, 2)))
    mae = np.mean(np.abs(err), axis=(0, 2))
    return {"rmse": rmse, "mae": mae}


def open_loop_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    device: str | torch.device = "cpu",
) -> Dict[str, np.ndarray]:
    """Open-loop rollout quality: observe context, then imagine with known C/X."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    crps_list: List[np.ndarray] = []

    with torch.no_grad():
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]

            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_y = warmup[:, dataset.out_idx]
            target_y = future[:, dataset.out_idx]

            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            future_c, future_x = _split_controls_exo(dataset, future_inputs)

            out = model.condition_then_simulate(
                history_controls=torch.from_numpy(history_c).unsqueeze(0).to(device),
                history_exogenous=torch.from_numpy(history_x).unsqueeze(0).to(device),
                history_objectives=torch.from_numpy(history_y).unsqueeze(0).to(device),
                future_controls=torch.from_numpy(future_c).unsqueeze(0).to(device),
                future_exogenous=torch.from_numpy(future_x).unsqueeze(0).to(device),
                n_steps=horizon,
                n_samples=n_samples,
            )

            if "samples" in out:
                samples = out["samples"].squeeze(1).cpu()  # (N, H, O)
                mean = samples.mean(dim=0).numpy()
                target_t = torch.from_numpy(target_y)
                crps = crps_ensemble(samples, target_t).mean(dim=-1).numpy()
            else:
                mean = out["predictions"].squeeze(0).cpu().numpy()
                crps = np.mean(np.abs(mean - target_y), axis=-1)

            mean_preds.append(mean)
            gt_list.append(target_y)
            crps_list.append(crps)

    curves = _aggregate_error_curves(mean_preds, gt_list)
    curves["crps"] = np.mean(np.stack(crps_list, axis=0), axis=0) if crps_list else np.empty((0,), dtype=np.float32)
    return curves


def closed_loop_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    device: str | torch.device = "cpu",
) -> Dict[str, np.ndarray]:
    """Closed-loop one-step quality with reconditioning on true Y every step."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []

    with torch.no_grad():
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            future_inputs = future[:, dataset.in_idx]
            target_y = future[:, dataset.out_idx].astype(np.float32, copy=False)

            hist_inputs = warmup[:, dataset.in_idx].astype(np.float32, copy=True)
            hist_y = warmup[:, dataset.out_idx].astype(np.float32, copy=True)
            preds = np.zeros_like(target_y)

            for t in range(horizon):
                hc, hx = _split_controls_exo(dataset, hist_inputs)
                fc, fx = _split_controls_exo(dataset, future_inputs[t:t + 1])
                step_out = model.condition_then_simulate(
                    history_controls=torch.from_numpy(hc).unsqueeze(0).to(device),
                    history_exogenous=torch.from_numpy(hx).unsqueeze(0).to(device),
                    history_objectives=torch.from_numpy(hist_y).unsqueeze(0).to(device),
                    future_controls=torch.from_numpy(fc).unsqueeze(0).to(device),
                    future_exogenous=torch.from_numpy(fx).unsqueeze(0).to(device),
                    n_steps=1,
                    n_samples=n_samples,
                )
                if "samples" in step_out:
                    preds[t] = step_out["samples"].mean(dim=0).squeeze(0).squeeze(0).cpu().numpy()
                else:
                    preds[t] = step_out["predictions"].squeeze(0).squeeze(0).cpu().numpy()

                # Recondition with ground truth at the current step.
                hist_inputs = np.concatenate([hist_inputs, future_inputs[t:t + 1]], axis=0)
                hist_y = np.concatenate([hist_y, target_y[t:t + 1]], axis=0)

            mean_preds.append(preds)
            gt_list.append(target_y)

    return _aggregate_error_curves(mean_preds, gt_list)


def calibration_check(
    samples: np.ndarray,
    targets: np.ndarray,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> Dict[float, float]:
    """Coverage for predictive intervals across probability levels.

    samples shape: (N, B, H, O)
    targets shape: (B, H, O)
    """
    if samples.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected samples=(N,B,H,O) and targets=(B,H,O)")

    coverages: Dict[float, float] = {}
    for lvl in levels:
        alpha = (1.0 - float(lvl)) / 2.0
        lo = np.quantile(samples, alpha, axis=0)
        hi = np.quantile(samples, 1.0 - alpha, axis=0)
        inside = (targets >= lo) & (targets <= hi)
        coverages[float(lvl)] = float(np.mean(inside))
    return coverages


def latent_diagnostics(
    model,
    controls: torch.Tensor,
    exogenous: torch.Tensor,
    observations: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return tensors used for KL/prior-posterior/latent diagnostics plots."""
    out = model.observe(
        controls=controls,
        exogenous=exogenous,
        observations=observations,
        sample_posterior=False,
    )
    return {
        "kl_terms": out["kl_terms"],
        "prior_mu": out["prior_mu"],
        "prior_logvar": out["prior_logvar"],
        "posterior_mu": out["posterior_mu"],
        "posterior_logvar": out["posterior_logvar"],
        "deter": out["deter"],
        "stoch": out["stoch"],
    }
