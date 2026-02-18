"""Evaluation helpers for RSSM world models."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Literal, Any

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


def _split_controls_exo(
    dataset: GroupedTimeSeriesDataset,
    input_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    return {"rmse": rmse.astype(np.float32), "mae": mae.astype(np.float32)}


def _inverse_outputs(dataset: GroupedTimeSeriesDataset, arr: np.ndarray) -> np.ndarray:
    """Inverse-transform arrays with last-dim = objective dim."""
    scaler = getattr(dataset, "scaler", None)
    out = np.asarray(arr, dtype=np.float32)
    if scaler is None:
        return out
    if out.ndim < 2 or out.shape[-1] != len(dataset.out_idx):
        return out
    flat = out.reshape(-1, out.shape[-1])
    full = np.zeros((flat.shape[0], dataset.values.shape[1]), dtype=np.float32)
    full[:, dataset.out_idx] = flat
    inv = scaler.inverse_transform(full)[:, dataset.out_idx]
    return inv.reshape(out.shape).astype(np.float32, copy=False)


def _interval_curves(
    samples: np.ndarray,
    targets: np.ndarray,
    levels: Sequence[float],
) -> tuple[Dict[float, np.ndarray], np.ndarray]:
    """Return coverage curves per level and sharpness at 90%."""
    coverage_by_level: Dict[float, np.ndarray] = {}
    for lvl in levels:
        alpha = (1.0 - float(lvl)) / 2.0
        lo = np.quantile(samples, alpha, axis=0)
        hi = np.quantile(samples, 1.0 - alpha, axis=0)
        inside = (targets >= lo) & (targets <= hi)
        coverage_by_level[float(lvl)] = np.mean(inside, axis=-1).astype(np.float32)

    lo90 = np.quantile(samples, 0.05, axis=0)
    hi90 = np.quantile(samples, 0.95, axis=0)
    sharpness90 = np.mean(hi90 - lo90, axis=-1).astype(np.float32)
    return coverage_by_level, sharpness90


def open_loop_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    device: str | torch.device = "cpu",
    denormalize: bool = True,
    interval_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Dict[str, Any]:
    """Open-loop rollout quality: observe context, then imagine with known C/X."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    crps_list: List[np.ndarray] = []
    nll_list: List[np.ndarray] = []
    coverage_lists: Dict[float, List[np.ndarray]] = {float(l): [] for l in interval_levels}
    sharpness90_list: List[np.ndarray] = []

    with torch.no_grad():
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]

            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)
            target_y = future[:, dataset.out_idx].astype(np.float32, copy=False)

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
                samples = out["samples"].squeeze(1).cpu().numpy().astype(np.float32, copy=False)  # (N, H, O)
                if denormalize:
                    samples = _inverse_outputs(dataset, samples)
                    target_eval = _inverse_outputs(dataset, target_y)
                else:
                    target_eval = target_y

                mean = np.mean(samples, axis=0).astype(np.float32, copy=False)
                crps_t = crps_ensemble(
                    torch.from_numpy(samples),
                    torch.from_numpy(target_eval),
                ).mean(dim=-1).cpu().numpy().astype(np.float32, copy=False)
                coverage_t, sharpness90_t = _interval_curves(samples, target_eval, interval_levels)
                for lvl, curve in coverage_t.items():
                    coverage_lists[lvl].append(curve)
                sharpness90_list.append(sharpness90_t)

                if "dist_loc_latent" in out and "dist_scale" in out:
                    loc_lat = out["dist_loc_latent"].squeeze(0)
                    scale = out["dist_scale"].squeeze(0)
                    y_t = torch.from_numpy(target_y).to(loc_lat.device)
                    if bool(getattr(model, "use_symlog", False)):
                        y_t = model.symlog(y_t)
                    nll_t = (-torch.distributions.Normal(loc=loc_lat, scale=scale).log_prob(y_t))
                    nll_list.append(nll_t.mean(dim=-1).cpu().numpy().astype(np.float32, copy=False))
            else:
                mean = out["predictions"].squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                target_eval = _inverse_outputs(dataset, target_y) if denormalize else target_y
                if denormalize:
                    mean = _inverse_outputs(dataset, mean)
                crps_t = np.mean(np.abs(mean - target_eval), axis=-1).astype(np.float32, copy=False)

            mean_preds.append(mean)
            gt_list.append(target_eval)
            crps_list.append(crps_t)

    curves = _aggregate_error_curves(mean_preds, gt_list)
    curves["starts"] = np.asarray(starts, dtype=np.int64)
    curves["crps"] = (
        np.mean(np.stack(crps_list, axis=0), axis=0).astype(np.float32)
        if crps_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["nll"] = (
        np.mean(np.stack(nll_list, axis=0), axis=0).astype(np.float32)
        if nll_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["sharpness_90"] = (
        np.mean(np.stack(sharpness90_list, axis=0), axis=0).astype(np.float32)
        if sharpness90_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["coverage"] = {
        lvl: (
            np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
            if vals
            else np.empty((0,), dtype=np.float32)
        )
        for lvl, vals in coverage_lists.items()
    }
    return curves


def closed_loop_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    device: str | torch.device = "cpu",
    denormalize: bool = True,
    interval_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Dict[str, Any]:
    """Closed-loop one-step quality with reconditioning on true Y every step."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    crps_list: List[np.ndarray] = []
    nll_list: List[np.ndarray] = []
    coverage_lists: Dict[float, List[np.ndarray]] = {float(l): [] for l in interval_levels}
    sharpness90_list: List[np.ndarray] = []

    with torch.no_grad():
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            future_inputs = future[:, dataset.in_idx]
            target_y = future[:, dataset.out_idx].astype(np.float32, copy=False)

            hist_inputs = warmup[:, dataset.in_idx].astype(np.float32, copy=True)
            hist_y = warmup[:, dataset.out_idx].astype(np.float32, copy=True)
            preds = np.zeros_like(target_y, dtype=np.float32)
            crps_curve = np.zeros((horizon,), dtype=np.float32)
            nll_curve = np.zeros((horizon,), dtype=np.float32)
            coverage_curves = {
                float(lvl): np.zeros((horizon,), dtype=np.float32) for lvl in interval_levels
            }
            sharpness90_curve = np.zeros((horizon,), dtype=np.float32)

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

                y_t = target_y[t:t + 1]
                if "samples" in step_out:
                    samples_t = step_out["samples"].squeeze(2).squeeze(1).cpu().numpy().astype(np.float32, copy=False)  # (N, O)
                    if denormalize:
                        samples_eval = _inverse_outputs(dataset, samples_t)
                        y_eval = _inverse_outputs(dataset, y_t)
                    else:
                        samples_eval = samples_t
                        y_eval = y_t

                    preds[t] = np.mean(samples_eval, axis=0)
                    crps_curve[t] = float(
                        crps_ensemble(
                            torch.from_numpy(samples_eval),
                            torch.from_numpy(y_eval[0]),
                        ).mean().item()
                    )
                    cov_t, sharp_t = _interval_curves(
                        samples_eval[:, None, :],
                        y_eval,
                        interval_levels,
                    )
                    for lvl in interval_levels:
                        coverage_curves[float(lvl)][t] = float(cov_t[float(lvl)][0])
                    sharpness90_curve[t] = float(sharp_t[0])

                    if "dist_loc_latent" in step_out and "dist_scale" in step_out:
                        loc_lat = step_out["dist_loc_latent"].squeeze(1).squeeze(0)
                        scale = step_out["dist_scale"].squeeze(1).squeeze(0)
                        yt_t = torch.from_numpy(target_y[t]).to(loc_lat.device)
                        if bool(getattr(model, "use_symlog", False)):
                            yt_t = model.symlog(yt_t)
                        nll_curve[t] = float(
                            (-torch.distributions.Normal(loc=loc_lat, scale=scale).log_prob(yt_t)).mean().item()
                        )
                else:
                    pred_t = step_out["predictions"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                    if denormalize:
                        pred_t = _inverse_outputs(dataset, pred_t[None, :])[0]
                        y_eval = _inverse_outputs(dataset, y_t)[0]
                    else:
                        y_eval = y_t[0]
                    preds[t] = pred_t
                    crps_curve[t] = float(np.mean(np.abs(pred_t - y_eval)))

                # Recondition with ground truth at the current step.
                hist_inputs = np.concatenate([hist_inputs, future_inputs[t:t + 1]], axis=0)
                hist_y = np.concatenate([hist_y, target_y[t:t + 1]], axis=0)

            target_eval_full = _inverse_outputs(dataset, target_y) if denormalize else target_y
            mean_preds.append(preds)
            gt_list.append(target_eval_full)
            crps_list.append(crps_curve)
            if np.any(np.isfinite(nll_curve)):
                nll_list.append(nll_curve)
            for lvl in interval_levels:
                if np.any(np.isfinite(coverage_curves[float(lvl)])):
                    coverage_lists[float(lvl)].append(coverage_curves[float(lvl)])
            if np.any(np.isfinite(sharpness90_curve)):
                sharpness90_list.append(sharpness90_curve)

    curves = _aggregate_error_curves(mean_preds, gt_list)
    curves["starts"] = np.asarray(starts, dtype=np.int64)
    curves["crps"] = (
        np.mean(np.stack(crps_list, axis=0), axis=0).astype(np.float32)
        if crps_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["nll"] = (
        np.mean(np.stack(nll_list, axis=0), axis=0).astype(np.float32)
        if nll_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["sharpness_90"] = (
        np.mean(np.stack(sharpness90_list, axis=0), axis=0).astype(np.float32)
        if sharpness90_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["coverage"] = {
        lvl: (
            np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
            if vals
            else np.empty((0,), dtype=np.float32)
        )
        for lvl, vals in coverage_lists.items()
    }
    return curves


def interventional_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    scenario: Literal["constant", "step", "ramp"] = "step",
    control_step_size: float = 1.0,
    control_index: Optional[int] = None,
    objective_index: Optional[int] = None,
    device: str | torch.device = "cpu",
    denormalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Interventional sensitivity: perturb future controls and measure Y response."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)
    if not starts:
        return {
            "delta_abs": np.empty((0,), dtype=np.float32),
            "delta_signed": np.empty((0,), dtype=np.float32),
            "direction_score": np.empty((0,), dtype=np.float32),
        }

    control_cols = [dataset.input_cols[i] for i in getattr(dataset, "control_positions", [])]
    if not control_cols:
        return {
            "delta_abs": np.zeros((horizon,), dtype=np.float32),
            "delta_signed": np.zeros((horizon,), dtype=np.float32),
            "direction_score": np.zeros((horizon,), dtype=np.float32),
        }
    feature_idx = {c: i for i, c in enumerate(dataset.feature_cols)}
    control_idx = [feature_idx[c] for c in control_cols]
    control_std = np.std(dataset.values[:, control_idx], axis=0).astype(np.float32, copy=False)
    control_std = np.where(control_std < 1e-6, 1.0, control_std)
    if control_index is not None:
        cidx = int(control_index)
        if cidx < 0 or cidx >= len(control_cols):
            raise ValueError(
                f"control_index={cidx} out of range for {len(control_cols)} controls"
            )
    else:
        cidx = None

    delta_abs_all: List[np.ndarray] = []
    delta_signed_all: List[np.ndarray] = []
    direction_scores: List[np.ndarray] = []

    with torch.no_grad():
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]

            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_y = warmup[:, dataset.out_idx]

            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            base_c, future_x = _split_controls_exo(dataset, future_inputs)

            pert_c = np.array(base_c, copy=True)
            step_vec = control_std.reshape(1, -1)
            if cidx is not None:
                mask = np.zeros((1, step_vec.shape[1]), dtype=np.float32)
                mask[0, cidx] = 1.0
                step_vec = step_vec * mask
            if scenario == "constant":
                pert_c[:] = base_c[0:1]
            elif scenario == "step":
                split = max(1, horizon // 2)
                pert_c[split:] = pert_c[split:] + control_step_size * step_vec
            elif scenario == "ramp":
                ramp = np.linspace(0.0, control_step_size, num=horizon, dtype=np.float32).reshape(-1, 1)
                pert_c = pert_c + ramp * step_vec
            else:
                raise ValueError(f"Unsupported intervention scenario: {scenario}")

            base_out = model.condition_then_simulate(
                history_controls=torch.from_numpy(history_c).unsqueeze(0).to(device),
                history_exogenous=torch.from_numpy(history_x).unsqueeze(0).to(device),
                history_objectives=torch.from_numpy(history_y).unsqueeze(0).to(device),
                future_controls=torch.from_numpy(base_c).unsqueeze(0).to(device),
                future_exogenous=torch.from_numpy(future_x).unsqueeze(0).to(device),
                n_steps=horizon,
                n_samples=n_samples,
            )
            pert_out = model.condition_then_simulate(
                history_controls=torch.from_numpy(history_c).unsqueeze(0).to(device),
                history_exogenous=torch.from_numpy(history_x).unsqueeze(0).to(device),
                history_objectives=torch.from_numpy(history_y).unsqueeze(0).to(device),
                future_controls=torch.from_numpy(pert_c).unsqueeze(0).to(device),
                future_exogenous=torch.from_numpy(future_x).unsqueeze(0).to(device),
                n_steps=horizon,
                n_samples=n_samples,
            )

            if "samples" in base_out:
                base_mean = base_out["samples"].mean(dim=0).squeeze(0).cpu().numpy()
            else:
                base_mean = base_out["predictions"].squeeze(0).cpu().numpy()
            if "samples" in pert_out:
                pert_mean = pert_out["samples"].mean(dim=0).squeeze(0).cpu().numpy()
            else:
                pert_mean = pert_out["predictions"].squeeze(0).cpu().numpy()

            if denormalize:
                base_mean = _inverse_outputs(dataset, base_mean)
                pert_mean = _inverse_outputs(dataset, pert_mean)

            delta_y = pert_mean - base_mean
            if objective_index is not None:
                yidx = int(objective_index)
                if yidx < 0 or yidx >= delta_y.shape[-1]:
                    raise ValueError(
                        f"objective_index={yidx} out of range for output dim {delta_y.shape[-1]}"
                    )
                delta_y_dir = delta_y[:, yidx:yidx + 1]
            else:
                delta_y_dir = np.mean(delta_y, axis=-1, keepdims=True)
            if cidx is not None:
                delta_u = (pert_c - base_c)[:, cidx:cidx + 1]
            else:
                delta_u = np.mean(pert_c - base_c, axis=-1, keepdims=True)
            active = np.abs(delta_u) > 1e-8
            direction = np.full(delta_y_dir.shape, np.nan, dtype=np.float32)
            direction[active] = (
                np.sign(delta_y_dir[active]) * np.sign(delta_u[active])
            ).astype(np.float32, copy=False)
            direction_valid = np.isfinite(direction)
            direction_num = np.where(direction_valid, direction, 0.0).sum(axis=-1)
            direction_den = direction_valid.sum(axis=-1)
            direction_step = np.divide(
                direction_num,
                np.maximum(direction_den, 1),
                where=direction_den > 0,
            )
            direction_step = np.where(
                direction_den > 0,
                direction_step,
                np.nan,
            ).astype(np.float32, copy=False)

            delta_abs_all.append(np.mean(np.abs(delta_y), axis=-1))
            delta_signed_all.append(np.mean(delta_y, axis=-1))
            direction_scores.append(direction_step)

    direction_arr = np.stack(direction_scores, axis=0)
    direction_valid = np.isfinite(direction_arr)
    direction_num = np.where(direction_valid, direction_arr, 0.0).sum(axis=0)
    direction_den = direction_valid.sum(axis=0)
    direction_curve = np.divide(
        direction_num,
        np.maximum(direction_den, 1),
        where=direction_den > 0,
    )
    direction_curve = np.where(direction_den > 0, direction_curve, np.nan).astype(
        np.float32,
        copy=False,
    )

    return {
        "delta_abs": np.mean(np.stack(delta_abs_all, axis=0), axis=0).astype(np.float32),
        "delta_signed": np.mean(np.stack(delta_signed_all, axis=0), axis=0).astype(np.float32),
        "direction_score": direction_curve,
    }


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


def summarize_horizons(
    curves: Dict[str, np.ndarray],
    horizons: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, Dict[int, float]]:
    """Extract scalar metrics at selected horizons (1-indexed)."""
    summary: Dict[str, Dict[int, float]] = {}
    for name in ["rmse", "mae", "crps", "nll", "sharpness_90"]:
        arr = curves.get(name, None)
        if arr is None or len(arr) == 0:
            continue
        vals: Dict[int, float] = {}
        for h in horizons:
            idx = int(h) - 1
            if 0 <= idx < len(arr):
                vals[int(h)] = float(arr[idx])
        if vals:
            summary[name] = vals
    return summary


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
    latent = torch.cat([out["deter"], out["stoch"]], dim=-1)
    latent_proj = torch.zeros(
        latent.shape[0],
        latent.shape[1],
        2,
        dtype=latent.dtype,
        device=latent.device,
    )
    flat = latent.reshape(-1, latent.shape[-1])
    if flat.shape[0] >= 2 and flat.shape[1] >= 2:
        try:
            centered = flat - flat.mean(dim=0, keepdim=True)
            _, _, v = torch.pca_lowrank(centered, q=2)
            latent_proj = torch.matmul(centered, v[:, :2]).reshape(
                latent.shape[0], latent.shape[1], 2
            )
        except Exception:
            pass
    return {
        "kl_terms": out["kl_terms"],
        "prior_mu": out["prior_mu"],
        "prior_logvar": out["prior_logvar"],
        "posterior_mu": out["posterior_mu"],
        "posterior_logvar": out["posterior_logvar"],
        "deter": out["deter"],
        "stoch": out["stoch"],
        "latent": latent,
        "latent_proj_2d": latent_proj,
    }
