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
        return {
            "rmse": np.empty((0,), dtype=np.float32),
            "mae": np.empty((0,), dtype=np.float32),
            "rmse_per_dim": np.empty((0, 0), dtype=np.float32),
            "mae_per_dim": np.empty((0, 0), dtype=np.float32),
        }
    err = np.stack(means, axis=0) - np.stack(targets, axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=(0, 2)))
    mae = np.mean(np.abs(err), axis=(0, 2))
    rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=0))
    mae_per_dim = np.mean(np.abs(err), axis=0)
    return {
        "rmse": rmse.astype(np.float32),
        "mae": mae.astype(np.float32),
        "rmse_per_dim": rmse_per_dim.astype(np.float32),
        "mae_per_dim": mae_per_dim.astype(np.float32),
    }


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


def _apply_sigma_scale_to_samples(samples: np.ndarray, sigma_scale: float) -> np.ndarray:
    """Post-hoc calibration on ensemble samples by scaling spread around the mean."""
    scale = float(max(1e-6, sigma_scale))
    if np.isclose(scale, 1.0):
        return samples
    mean = np.mean(samples, axis=0, keepdims=True)
    return (mean + scale * (samples - mean)).astype(np.float32, copy=False)


def _nanmean_curves(curves: List[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(curves, axis=0).astype(np.float32, copy=False)
    valid = np.isfinite(stacked)
    num = np.where(valid, stacked, 0.0).sum(axis=0)
    den = valid.sum(axis=0)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, np.maximum(den, 1), where=den > 0, out=out)
    out = np.where(den > 0, out, np.nan)
    return out.astype(np.float32, copy=False)


def _condition_then_rollout(
    model,
    *,
    history_controls: torch.Tensor,
    history_exogenous: torch.Tensor,
    history_objectives: torch.Tensor,
    future_controls: torch.Tensor,
    future_exogenous: torch.Tensor,
    n_steps: Optional[int] = None,
    n_samples: int = 50,
) -> Dict[str, Any]:
    horizon = int(n_steps if n_steps is not None else future_controls.shape[1])
    warmup_seq = {
        "controls": history_controls,
        "exogenous": history_exogenous,
        "outputs": history_objectives,
        "inputs": torch.cat([history_controls, history_exogenous, history_objectives], dim=-1),
    }
    rollout_inputs = {
        "controls": future_controls[:, :horizon, :],
        "exogenous": future_exogenous[:, :horizon, :],
    }
    base_out = model.rollout(
        warmup_seq=warmup_seq,
        rollout_inputs=rollout_inputs,
        horizon=horizon,
    )
    out: Dict[str, Any] = dict(base_out)
    if n_samples > 1 and hasattr(model, "rollout_mc"):
        mc_out = model.rollout_mc(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            n_samples=n_samples,
        )
        out["samples"] = mc_out["samples"]
        out["predictions"] = mc_out.get("mean", out.get("predictions"))
    return out


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
    sigma_scale: float = 1.0,
) -> Dict[str, Any]:
    """Open-loop rollout quality: observe context, then imagine with known C/X."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    crps_list: List[np.ndarray] = []
    crps_per_dim_list: List[np.ndarray] = []
    nll_list: List[np.ndarray] = []
    nll_per_dim_list: List[np.ndarray] = []
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

            out = _condition_then_rollout(
                model,
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
                samples = _apply_sigma_scale_to_samples(samples, sigma_scale=sigma_scale)
                if denormalize:
                    samples = _inverse_outputs(dataset, samples)
                    target_eval = _inverse_outputs(dataset, target_y)
                else:
                    target_eval = target_y

                mean = np.mean(samples, axis=0).astype(np.float32, copy=False)
                crps_t_per_dim = crps_ensemble(
                    torch.from_numpy(samples),
                    torch.from_numpy(target_eval),
                ).cpu().numpy().astype(np.float32, copy=False)
                crps_t = np.mean(crps_t_per_dim, axis=-1).astype(np.float32, copy=False)
                coverage_t, sharpness90_t = _interval_curves(samples, target_eval, interval_levels)
                for lvl, curve in coverage_t.items():
                    coverage_lists[lvl].append(curve)
                sharpness90_list.append(sharpness90_t)

                if "dist_loc_latent" in out and "dist_scale" in out:
                    loc_lat = out["dist_loc_latent"].squeeze(0)
                    scale = out["dist_scale"].squeeze(0) * float(max(1e-6, sigma_scale))
                    y_t = torch.from_numpy(target_y).to(loc_lat.device)
                    if bool(getattr(model, "use_symlog", False)):
                        y_t = model.symlog(y_t)
                    nll_t = (-torch.distributions.Normal(loc=loc_lat, scale=scale).log_prob(y_t))
                    nll_t_per_dim = nll_t.cpu().numpy().astype(np.float32, copy=False)
                    nll_list.append(np.mean(nll_t_per_dim, axis=-1).astype(np.float32, copy=False))
                    nll_per_dim_list.append(nll_t_per_dim)
            else:
                mean = out["predictions"].squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                target_eval = _inverse_outputs(dataset, target_y) if denormalize else target_y
                if denormalize:
                    mean = _inverse_outputs(dataset, mean)
                crps_t = np.mean(np.abs(mean - target_eval), axis=-1).astype(np.float32, copy=False)
                crps_t_per_dim = np.abs(mean - target_eval).astype(np.float32, copy=False)

            mean_preds.append(mean)
            gt_list.append(target_eval)
            crps_list.append(crps_t)
            crps_per_dim_list.append(crps_t_per_dim)

    curves = _aggregate_error_curves(mean_preds, gt_list)
    curves["starts"] = np.asarray(starts, dtype=np.int64)
    curves["crps"] = (
        np.mean(np.stack(crps_list, axis=0), axis=0).astype(np.float32)
        if crps_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["crps_per_dim"] = (
        np.mean(np.stack(crps_per_dim_list, axis=0), axis=0).astype(np.float32)
        if crps_per_dim_list
        else np.empty((0, 0), dtype=np.float32)
    )
    curves["nll"] = (
        np.mean(np.stack(nll_list, axis=0), axis=0).astype(np.float32)
        if nll_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["nll_per_dim"] = _nanmean_curves(nll_per_dim_list)
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
    sigma_scale: float = 1.0,
) -> Dict[str, Any]:
    """Closed-loop one-step quality with reconditioning on true Y every step."""
    device = torch.device(device)
    model.eval()
    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)

    mean_preds: List[np.ndarray] = []
    gt_list: List[np.ndarray] = []
    crps_list: List[np.ndarray] = []
    crps_per_dim_list: List[np.ndarray] = []
    nll_list: List[np.ndarray] = []
    nll_per_dim_list: List[np.ndarray] = []
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
            crps_curve_per_dim = np.zeros((horizon, target_y.shape[-1]), dtype=np.float32)
            nll_curve = np.zeros((horizon,), dtype=np.float32)
            nll_curve_per_dim = np.full((horizon, target_y.shape[-1]), np.nan, dtype=np.float32)
            coverage_curves = {
                float(lvl): np.zeros((horizon,), dtype=np.float32) for lvl in interval_levels
            }
            sharpness90_curve = np.zeros((horizon,), dtype=np.float32)

            for t in range(horizon):
                hc, hx = _split_controls_exo(dataset, hist_inputs)
                fc, fx = _split_controls_exo(dataset, future_inputs[t:t + 1])
                step_out = _condition_then_rollout(
                    model,
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
                    samples_t = _apply_sigma_scale_to_samples(
                        samples_t[:, None, :],
                        sigma_scale=sigma_scale,
                    )[:, 0, :]
                    if denormalize:
                        samples_eval = _inverse_outputs(dataset, samples_t)
                        y_eval = _inverse_outputs(dataset, y_t)
                    else:
                        samples_eval = samples_t
                        y_eval = y_t

                    preds[t] = np.mean(samples_eval, axis=0)
                    crps_vec = crps_ensemble(
                        torch.from_numpy(samples_eval),
                        torch.from_numpy(y_eval[0]),
                    ).cpu().numpy().astype(np.float32, copy=False)
                    crps_curve[t] = float(np.mean(crps_vec))
                    crps_curve_per_dim[t] = crps_vec
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
                        scale = step_out["dist_scale"].squeeze(1).squeeze(0) * float(max(1e-6, sigma_scale))
                        yt_t = torch.from_numpy(target_y[t]).to(loc_lat.device)
                        if bool(getattr(model, "use_symlog", False)):
                            yt_t = model.symlog(yt_t)
                        nll_vec = (
                            -torch.distributions.Normal(loc=loc_lat, scale=scale).log_prob(yt_t)
                        ).cpu().numpy().astype(np.float32, copy=False)
                        nll_curve[t] = float(np.mean(nll_vec))
                        nll_curve_per_dim[t] = nll_vec
                else:
                    pred_t = step_out["predictions"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                    if denormalize:
                        pred_t = _inverse_outputs(dataset, pred_t[None, :])[0]
                        y_eval = _inverse_outputs(dataset, y_t)[0]
                    else:
                        y_eval = y_t[0]
                    preds[t] = pred_t
                    crps_curve[t] = float(np.mean(np.abs(pred_t - y_eval)))
                    crps_curve_per_dim[t] = np.abs(pred_t - y_eval).astype(np.float32, copy=False)

                # Recondition with ground truth at the current step.
                hist_inputs = np.concatenate([hist_inputs, future_inputs[t:t + 1]], axis=0)
                hist_y = np.concatenate([hist_y, target_y[t:t + 1]], axis=0)

            target_eval_full = _inverse_outputs(dataset, target_y) if denormalize else target_y
            mean_preds.append(preds)
            gt_list.append(target_eval_full)
            crps_list.append(crps_curve)
            crps_per_dim_list.append(crps_curve_per_dim)
            if np.any(np.isfinite(nll_curve)):
                nll_list.append(nll_curve)
                nll_per_dim_list.append(nll_curve_per_dim)
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
    curves["crps_per_dim"] = (
        np.mean(np.stack(crps_per_dim_list, axis=0), axis=0).astype(np.float32)
        if crps_per_dim_list
        else np.empty((0, 0), dtype=np.float32)
    )
    curves["nll"] = (
        np.mean(np.stack(nll_list, axis=0), axis=0).astype(np.float32)
        if nll_list
        else np.empty((0,), dtype=np.float32)
    )
    curves["nll_per_dim"] = _nanmean_curves(nll_per_dim_list)
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

            base_out = _condition_then_rollout(
                model,
                history_controls=torch.from_numpy(history_c).unsqueeze(0).to(device),
                history_exogenous=torch.from_numpy(history_x).unsqueeze(0).to(device),
                history_objectives=torch.from_numpy(history_y).unsqueeze(0).to(device),
                future_controls=torch.from_numpy(base_c).unsqueeze(0).to(device),
                future_exogenous=torch.from_numpy(future_x).unsqueeze(0).to(device),
                n_steps=horizon,
                n_samples=n_samples,
            )
            pert_out = _condition_then_rollout(
                model,
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
            direction_step = np.full(direction_num.shape, np.nan, dtype=np.float32)
            np.divide(
                direction_num,
                np.maximum(direction_den, 1),
                where=direction_den > 0,
                out=direction_step,
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
    direction_curve = np.full(direction_num.shape, np.nan, dtype=np.float32)
    np.divide(
        direction_num,
        np.maximum(direction_den, 1),
        where=direction_den > 0,
        out=direction_curve,
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


def _random_window_starts(
    total_len: int,
    warmup_len: int,
    horizon: int,
    n_windows: int,
    seed: int = 42,
) -> List[int]:
    max_start = total_len - (warmup_len + horizon)
    if max_start < 0:
        return []
    n_windows = max(1, int(n_windows))
    rng = np.random.default_rng(seed)
    population = max_start + 1
    replace = n_windows > population
    starts = rng.choice(population, size=n_windows, replace=replace)
    return starts.astype(int).tolist()


def _simulate_condition_then_simulate(
    model,
    dataset: GroupedTimeSeriesDataset,
    history_c: np.ndarray,
    history_x: np.ndarray,
    history_y: np.ndarray,
    future_c: np.ndarray,
    future_x: np.ndarray,
    *,
    n_samples: int,
    device: torch.device,
    denormalize: bool,
    sigma_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        out = _condition_then_rollout(
            model,
            history_controls=torch.from_numpy(history_c).unsqueeze(0).to(device),
            history_exogenous=torch.from_numpy(history_x).unsqueeze(0).to(device),
            history_objectives=torch.from_numpy(history_y).unsqueeze(0).to(device),
            future_controls=torch.from_numpy(future_c).unsqueeze(0).to(device),
            future_exogenous=torch.from_numpy(future_x).unsqueeze(0).to(device),
            n_steps=future_c.shape[0],
            n_samples=n_samples,
        )

    samples = None
    if "samples" in out:
        # (N, B=1, H, O) -> (N, H, O)
        samples = out["samples"].squeeze(1).cpu().numpy().astype(np.float32, copy=False)
        samples = _apply_sigma_scale_to_samples(samples, sigma_scale=sigma_scale)
        mean = np.mean(samples, axis=0).astype(np.float32, copy=False)
        std = np.std(samples, axis=0).astype(np.float32, copy=False)
    else:
        mean = out["predictions"].squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        std = np.full_like(mean, np.nan, dtype=np.float32)

    if denormalize:
        if samples is not None:
            samples = _inverse_outputs(dataset, samples)
            mean = np.mean(samples, axis=0).astype(np.float32, copy=False)
            std = np.std(samples, axis=0).astype(np.float32, copy=False)
        else:
            mean = _inverse_outputs(dataset, mean)

    finite = bool(np.isfinite(mean).all() and np.isfinite(std).all())
    return {
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "samples": samples if samples is not None else np.empty((0,), dtype=np.float32),
        "finite": np.asarray([finite], dtype=np.bool_),
    }


def _select_objective_traj(arr: np.ndarray, objective_index: Optional[int]) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected trajectory shape (H,O), got {tuple(arr.shape)}")
    if objective_index is None:
        return np.mean(arr, axis=-1, keepdims=True).astype(np.float32, copy=False)
    oidx = int(objective_index)
    if oidx < 0 or oidx >= arr.shape[-1]:
        raise ValueError(f"objective_index={oidx} out of range for output dim {arr.shape[-1]}")
    return arr[:, oidx:oidx + 1].astype(np.float32, copy=False)


def _control_step_vector(
    control_std: np.ndarray,
    control_index: Optional[int],
    step_scale: float,
) -> np.ndarray:
    if control_std.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    step_vec = control_std.reshape(1, -1).astype(np.float32, copy=False)
    if control_index is not None:
        cidx = int(control_index)
        if cidx < 0 or cidx >= step_vec.shape[-1]:
            raise ValueError(
                f"control_index={cidx} out of range for {step_vec.shape[-1]} controls"
            )
        mask = np.zeros_like(step_vec)
        mask[0, cidx] = 1.0
        step_vec = step_vec * mask
    return (float(step_scale) * step_vec).astype(np.float32, copy=False)


def _exogenous_step_vector(
    exogenous_std: np.ndarray,
    exogenous_index: Optional[int],
    step_scale: float,
) -> np.ndarray:
    if exogenous_std.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    step_vec = exogenous_std.reshape(1, -1).astype(np.float32, copy=False)
    if exogenous_index is not None:
        xidx = int(exogenous_index)
        if xidx < 0 or xidx >= step_vec.shape[-1]:
            raise ValueError(
                f"exogenous_index={xidx} out of range for {step_vec.shape[-1]} exogenous dims"
            )
        mask = np.zeros_like(step_vec)
        mask[0, xidx] = 1.0
        step_vec = step_vec * mask
    return (float(step_scale) * step_vec).astype(np.float32, copy=False)


def interventional_suite_evaluate(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int = 8,
    n_samples: int = 50,
    control_index: Optional[int] = None,
    objective_index: Optional[int] = None,
    exogenous_index: Optional[int] = None,
    expected_direction_sign: Optional[float] = None,
    direction_n_windows: int = 100,
    control_step_size: float = 1.0,
    exogenous_step_size: float = 1.0,
    sensitivity_threshold_ratio: float = 0.01,
    irrelevance_pairs: Optional[Sequence[Dict[str, int]]] = None,
    irrelevance_tolerance_ratio: float = 0.05,
    irrelevance_tolerance_abs: float = 1e-4,
    extreme_sigma: float = 3.0,
    extreme_widen_ratio: float = 1.2,
    extreme_min_std_ratio: float = 0.05,
    extreme_min_std_abs: float = 1e-4,
    sigma_scale: float = 1.0,
    random_seed: int = 42,
    device: str | torch.device = "cpu",
    denormalize: bool = True,
) -> Dict[str, Any]:
    """Comprehensive interventional diagnostics for simulator quality."""
    device_t = torch.device(device)
    model.eval()

    starts = _window_starts(len(dataset.values), warmup_len, horizon, n_windows)
    direction_starts = _random_window_starts(
        len(dataset.values),
        warmup_len,
        horizon,
        max(1, int(direction_n_windows)),
        seed=int(random_seed),
    )

    inputs_all = dataset.values[:, dataset.in_idx]
    controls_all, exogenous_all = _split_controls_exo(dataset, inputs_all)
    control_std = (
        np.std(controls_all, axis=0).astype(np.float32, copy=False)
        if controls_all.size > 0
        else np.zeros((0,), dtype=np.float32)
    )
    exogenous_std = (
        np.std(exogenous_all, axis=0).astype(np.float32, copy=False)
        if exogenous_all.size > 0
        else np.zeros((0,), dtype=np.float32)
    )
    control_std = np.where(control_std < 1e-6, 1.0, control_std)
    exogenous_std = np.where(exogenous_std < 1e-6, 1.0, exogenous_std)

    y_all = dataset.values[:, dataset.out_idx].astype(np.float32, copy=False)
    y_all_eval = _inverse_outputs(dataset, y_all) if denormalize else y_all
    y_std = np.std(y_all_eval, axis=0).astype(np.float32, copy=False)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_std_sel = float(
        y_std[int(objective_index)]
        if objective_index is not None and 0 <= int(objective_index) < y_std.shape[0]
        else np.mean(y_std)
    )
    sensitivity_threshold = max(1e-6, float(sensitivity_threshold_ratio) * y_std_sel)

    control_step_vec = _control_step_vector(control_std, control_index, control_step_size)
    exogenous_step_vec = _exogenous_step_vector(exogenous_std, exogenous_index, exogenous_step_size)

    # 5C.1 Control sensitivity (C_low, C_mid, C_high).
    control_window_rows: List[Dict[str, Any]] = []
    control_traj = {"low": [], "mid": [], "high": []}
    if controls_all.shape[1] > 0 and starts:
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            base_c, base_x = _split_controls_exo(dataset, future_inputs)
            history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)

            scenario_means: Dict[str, np.ndarray] = {}
            for name, mult in (("low", -1.0), ("mid", 0.0), ("high", 1.0)):
                c_mod = base_c + float(mult) * control_step_vec
                sim = _simulate_condition_then_simulate(
                    model=model,
                    dataset=dataset,
                    history_c=history_c,
                    history_x=history_x,
                    history_y=history_y,
                    future_c=c_mod.astype(np.float32, copy=False),
                    future_x=base_x.astype(np.float32, copy=False),
                    n_samples=n_samples,
                    device=device_t,
                    denormalize=denormalize,
                    sigma_scale=sigma_scale,
                )
                scenario_means[name] = sim["mean"]
                control_traj[name].append(sim["mean"])

            low_sel = _select_objective_traj(scenario_means["low"], objective_index)
            mid_sel = _select_objective_traj(scenario_means["mid"], objective_index)
            high_sel = _select_objective_traj(scenario_means["high"], objective_index)
            delta_lm = float(np.mean(np.abs(low_sel - mid_sel)))
            delta_mh = float(np.mean(np.abs(mid_sel - high_sel)))
            delta_lh = float(np.mean(np.abs(low_sel - high_sel)))
            differs = bool(
                delta_lm > sensitivity_threshold
                and delta_mh > sensitivity_threshold
                and delta_lh > sensitivity_threshold
            )
            control_window_rows.append(
                {
                    "start": int(s),
                    "delta_low_mid": delta_lm,
                    "delta_mid_high": delta_mh,
                    "delta_low_high": delta_lh,
                    "threshold": float(sensitivity_threshold),
                    "differs": differs,
                }
            )

    control_traj_mean: Dict[str, np.ndarray] = {}
    for key, vals in control_traj.items():
        if vals:
            control_traj_mean[key] = np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32, copy=False)
        else:
            control_traj_mean[key] = np.empty((0, y_all.shape[-1]), dtype=np.float32)
    control_diff_rate = float(
        np.mean([1.0 if bool(r["differs"]) else 0.0 for r in control_window_rows])
    ) if control_window_rows else np.nan

    # 5C.2 Direction agreement across random windows.
    direction_rows: List[Dict[str, Any]] = []
    if controls_all.shape[1] > 0 and direction_starts:
        for s in direction_starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            base_c, base_x = _split_controls_exo(dataset, future_inputs)
            history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)

            low = _simulate_condition_then_simulate(
                model=model,
                dataset=dataset,
                history_c=history_c,
                history_x=history_x,
                history_y=history_y,
                future_c=(base_c - control_step_vec).astype(np.float32, copy=False),
                future_x=base_x.astype(np.float32, copy=False),
                n_samples=n_samples,
                device=device_t,
                denormalize=denormalize,
                sigma_scale=sigma_scale,
            )
            high = _simulate_condition_then_simulate(
                model=model,
                dataset=dataset,
                history_c=history_c,
                history_x=history_x,
                history_y=history_y,
                future_c=(base_c + control_step_vec).astype(np.float32, copy=False),
                future_x=base_x.astype(np.float32, copy=False),
                n_samples=n_samples,
                device=device_t,
                denormalize=denormalize,
                sigma_scale=sigma_scale,
            )
            low_sel = _select_objective_traj(low["mean"], objective_index)
            high_sel = _select_objective_traj(high["mean"], objective_index)
            effect = float(np.mean(high_sel - low_sel))
            agree = np.nan
            if expected_direction_sign is not None:
                exp = 1.0 if float(expected_direction_sign) >= 0.0 else -1.0
                agree = float(np.sign(effect) == np.sign(exp))
            direction_rows.append(
                {
                    "start": int(s),
                    "effect": effect,
                    "expected_sign": expected_direction_sign,
                    "agree": agree,
                }
            )
    direction_agreement_rate = np.nan
    if direction_rows and expected_direction_sign is not None:
        vals = np.asarray([r["agree"] for r in direction_rows], dtype=np.float32)
        direction_agreement_rate = float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else np.nan

    # 5C.3 Exogenous sensitivity.
    exogenous_window_rows: List[Dict[str, Any]] = []
    exogenous_traj = {"low": [], "mid": [], "high": []}
    if exogenous_all.shape[1] > 0 and starts:
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            base_c, base_x = _split_controls_exo(dataset, future_inputs)
            history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)

            scenario_means: Dict[str, np.ndarray] = {}
            for name, mult in (("low", -1.0), ("mid", 0.0), ("high", 1.0)):
                x_mod = base_x + float(mult) * exogenous_step_vec
                sim = _simulate_condition_then_simulate(
                    model=model,
                    dataset=dataset,
                    history_c=history_c,
                    history_x=history_x,
                    history_y=history_y,
                    future_c=base_c.astype(np.float32, copy=False),
                    future_x=x_mod.astype(np.float32, copy=False),
                    n_samples=n_samples,
                    device=device_t,
                    denormalize=denormalize,
                    sigma_scale=sigma_scale,
                )
                scenario_means[name] = sim["mean"]
                exogenous_traj[name].append(sim["mean"])

            low_sel = _select_objective_traj(scenario_means["low"], objective_index)
            mid_sel = _select_objective_traj(scenario_means["mid"], objective_index)
            high_sel = _select_objective_traj(scenario_means["high"], objective_index)
            delta_lm = float(np.mean(np.abs(low_sel - mid_sel)))
            delta_mh = float(np.mean(np.abs(mid_sel - high_sel)))
            delta_lh = float(np.mean(np.abs(low_sel - high_sel)))
            differs = bool(
                delta_lm > sensitivity_threshold
                and delta_mh > sensitivity_threshold
                and delta_lh > sensitivity_threshold
            )
            exogenous_window_rows.append(
                {
                    "start": int(s),
                    "delta_low_mid": delta_lm,
                    "delta_mid_high": delta_mh,
                    "delta_low_high": delta_lh,
                    "threshold": float(sensitivity_threshold),
                    "differs": differs,
                }
            )

    exogenous_traj_mean: Dict[str, np.ndarray] = {}
    for key, vals in exogenous_traj.items():
        if vals:
            exogenous_traj_mean[key] = np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32, copy=False)
        else:
            exogenous_traj_mean[key] = np.empty((0, y_all.shape[-1]), dtype=np.float32)
    exogenous_diff_rate = float(
        np.mean([1.0 if bool(r["differs"]) else 0.0 for r in exogenous_window_rows])
    ) if exogenous_window_rows else np.nan

    # 5C.4 Control irrelevance test.
    irrelevance_details: List[Dict[str, Any]] = []
    irrelevance_summary: List[Dict[str, Any]] = []
    if irrelevance_pairs:
        for pair in irrelevance_pairs:
            cidx = int(pair.get("control_index", -1))
            oidx = int(pair.get("objective_index", -1))
            if cidx < 0 or oidx < 0:
                continue
            local_step = _control_step_vector(control_std, cidx, control_step_size)
            tol = max(float(irrelevance_tolerance_abs), float(irrelevance_tolerance_ratio) * float(y_std[oidx]))
            pass_flags: List[float] = []
            effects: List[float] = []

            for s in starts:
                warmup = dataset.values[s:s + warmup_len]
                future = dataset.values[s + warmup_len:s + warmup_len + horizon]
                warmup_inputs = warmup[:, dataset.in_idx]
                future_inputs = future[:, dataset.in_idx]
                history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
                base_c, base_x = _split_controls_exo(dataset, future_inputs)
                history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)

                low = _simulate_condition_then_simulate(
                    model=model,
                    dataset=dataset,
                    history_c=history_c,
                    history_x=history_x,
                    history_y=history_y,
                    future_c=(base_c - local_step).astype(np.float32, copy=False),
                    future_x=base_x.astype(np.float32, copy=False),
                    n_samples=n_samples,
                    device=device_t,
                    denormalize=denormalize,
                    sigma_scale=sigma_scale,
                )
                high = _simulate_condition_then_simulate(
                    model=model,
                    dataset=dataset,
                    history_c=history_c,
                    history_x=history_x,
                    history_y=history_y,
                    future_c=(base_c + local_step).astype(np.float32, copy=False),
                    future_x=base_x.astype(np.float32, copy=False),
                    n_samples=n_samples,
                    device=device_t,
                    denormalize=denormalize,
                    sigma_scale=sigma_scale,
                )
                low_sel = _select_objective_traj(low["mean"], oidx)
                high_sel = _select_objective_traj(high["mean"], oidx)
                effect_abs = float(np.mean(np.abs(high_sel - low_sel)))
                passed = bool(effect_abs <= tol)
                effects.append(effect_abs)
                pass_flags.append(1.0 if passed else 0.0)
                irrelevance_details.append(
                    {
                        "control_index": cidx,
                        "objective_index": oidx,
                        "start": int(s),
                        "effect_abs": effect_abs,
                        "tolerance": tol,
                        "pass": passed,
                    }
                )
            if effects:
                irrelevance_summary.append(
                    {
                        "control_index": cidx,
                        "objective_index": oidx,
                        "mean_effect_abs": float(np.mean(np.asarray(effects, dtype=np.float32))),
                        "tolerance": float(tol),
                        "pass_rate": float(np.mean(np.asarray(pass_flags, dtype=np.float32))),
                    }
                )

    # 5C.5 Extreme control stress test.
    extreme_rows: List[Dict[str, Any]] = []
    if controls_all.shape[1] > 0 and starts:
        extreme_step = _control_step_vector(control_std, control_index, extreme_sigma)
        for s in starts:
            warmup = dataset.values[s:s + warmup_len]
            future = dataset.values[s + warmup_len:s + warmup_len + horizon]
            warmup_inputs = warmup[:, dataset.in_idx]
            future_inputs = future[:, dataset.in_idx]
            history_c, history_x = _split_controls_exo(dataset, warmup_inputs)
            base_c, base_x = _split_controls_exo(dataset, future_inputs)
            history_y = warmup[:, dataset.out_idx].astype(np.float32, copy=False)

            base = _simulate_condition_then_simulate(
                model=model,
                dataset=dataset,
                history_c=history_c,
                history_x=history_x,
                history_y=history_y,
                future_c=base_c.astype(np.float32, copy=False),
                future_x=base_x.astype(np.float32, copy=False),
                n_samples=n_samples,
                device=device_t,
                denormalize=denormalize,
                sigma_scale=sigma_scale,
            )
            low = _simulate_condition_then_simulate(
                model=model,
                dataset=dataset,
                history_c=history_c,
                history_x=history_x,
                history_y=history_y,
                future_c=(base_c - extreme_step).astype(np.float32, copy=False),
                future_x=base_x.astype(np.float32, copy=False),
                n_samples=n_samples,
                device=device_t,
                denormalize=denormalize,
                sigma_scale=sigma_scale,
            )
            high = _simulate_condition_then_simulate(
                model=model,
                dataset=dataset,
                history_c=history_c,
                history_x=history_x,
                history_y=history_y,
                future_c=(base_c + extreme_step).astype(np.float32, copy=False),
                future_x=base_x.astype(np.float32, copy=False),
                n_samples=n_samples,
                device=device_t,
                denormalize=denormalize,
                sigma_scale=sigma_scale,
            )

            base_std_sel = _select_objective_traj(base["std"], objective_index)
            low_std_sel = _select_objective_traj(low["std"], objective_index)
            high_std_sel = _select_objective_traj(high["std"], objective_index)
            base_std_mean = float(np.mean(base_std_sel[np.isfinite(base_std_sel)])) if np.isfinite(base_std_sel).any() else np.nan
            low_std_mean = float(np.mean(low_std_sel[np.isfinite(low_std_sel)])) if np.isfinite(low_std_sel).any() else np.nan
            high_std_mean = float(np.mean(high_std_sel[np.isfinite(high_std_sel)])) if np.isfinite(high_std_sel).any() else np.nan
            extreme_std_mean = float(np.nanmax(np.asarray([low_std_mean, high_std_mean], dtype=np.float32)))
            widen_ratio = float(extreme_std_mean / max(base_std_mean, 1e-8)) if np.isfinite(extreme_std_mean) and np.isfinite(base_std_mean) else np.nan

            finite_pass = bool(base["finite"][0] and low["finite"][0] and high["finite"][0])
            widen_pass = bool(np.isfinite(widen_ratio) and widen_ratio >= float(extreme_widen_ratio))
            conf_floor = max(float(extreme_min_std_abs), float(extreme_min_std_ratio) * y_std_sel)
            not_confident_pass = bool(np.isfinite(extreme_std_mean) and extreme_std_mean >= conf_floor)

            extreme_rows.append(
                {
                    "start": int(s),
                    "finite_pass": finite_pass,
                    "widen_ratio": widen_ratio,
                    "widen_pass": widen_pass,
                    "extreme_std_mean": extreme_std_mean,
                    "base_std_mean": base_std_mean,
                    "confidence_floor": conf_floor,
                    "not_confident_pass": not_confident_pass,
                }
            )

    def _rate(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return np.nan
        vals = np.asarray([1.0 if bool(r[key]) else 0.0 for r in rows], dtype=np.float32)
        return float(np.mean(vals))

    return {
        "control_sensitivity": {
            "window_rows": control_window_rows,
            "trajectory_means": control_traj_mean,
            "diff_rate": control_diff_rate,
            "threshold": float(sensitivity_threshold),
        },
        "direction_check": {
            "window_rows": direction_rows,
            "agreement_rate": direction_agreement_rate,
            "expected_sign": expected_direction_sign,
            "n_windows": int(len(direction_rows)),
        },
        "exogenous_sensitivity": {
            "window_rows": exogenous_window_rows,
            "trajectory_means": exogenous_traj_mean,
            "diff_rate": exogenous_diff_rate,
            "threshold": float(sensitivity_threshold),
        },
        "control_irrelevance": {
            "rows": irrelevance_details,
            "summary": irrelevance_summary,
            "overall_pass_rate": float(
                np.mean(np.asarray([r["pass_rate"] for r in irrelevance_summary], dtype=np.float32))
            ) if irrelevance_summary else np.nan,
        },
        "extreme_control": {
            "window_rows": extreme_rows,
            "finite_rate": _rate(extreme_rows, "finite_pass"),
            "widen_rate": _rate(extreme_rows, "widen_pass"),
            "not_confident_rate": _rate(extreme_rows, "not_confident_pass"),
        },
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
