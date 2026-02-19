#!/usr/bin/env python3
"""Comprehensive RSSM evaluation suite (open/closed/interventional/latent)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import load as joblib_load

from timesim.utils.config import load_config
from timesim.data.loader import (
    load_csv_dataset,
    resolve_split_ratios,
    chronological_split_dataframe,
)
from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.schema import VariableSchema
from timesim.data.stamps import get_time_feature_columns
from timesim.utils.misc import seed_everything, resolve_device
from timesim.evaluation import (
    open_loop_evaluate,
    closed_loop_evaluate,
    interventional_evaluate,
    interventional_suite_evaluate,
    summarize_horizons,
    latent_diagnostics,
)
from timesim.models.factory import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Run RSSM training/eval checklist suite")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n-windows", type=int, default=None)
    p.add_argument("--mc-samples", type=int, default=None)
    p.add_argument("--sigma-scale", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def _resolve_checkpoint(config: Dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    output_cfg = config.get("output", {})
    run_dir = Path(output_cfg.get("runs_dir", "runs")) / config["dataset"]["name"]
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str) and run_name.strip():
        run_dir = run_dir / run_name.strip()
    model_dir = run_dir / "latent_ssm"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates = sorted(model_dir.glob("*_checkpoint.pth"))
    if not candidates:
        candidates = sorted((model_dir / "checkpoints").glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in: {model_dir}")
    return candidates[-1]


def _load_model(model, checkpoint: Path, device: str):
    try:
        raw = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception:
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    try:
        model.load_state_dict(state)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                "Warning: non-strict checkpoint load "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )
    model.to(device)
    model.eval()


def _build_test_dataset(config: Dict[str, Any], scaler) -> GroupedTimeSeriesDataset:
    dcfg = config["dataset"]
    data_cfg = config.get("data", {})
    groups = dcfg["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]
    seq_len = int(dcfg["seq_len"])
    pred_len = int(dcfg["pred_len"])

    df = load_csv_dataset(
        dcfg["csv"],
        index_col=dcfg.get("index_col", data_cfg.get("index_col", "date")),
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=dcfg.get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )

    eval_cfg = config.get("evaluation", {}) or {}
    test_split = eval_cfg.get("test_split", None)
    if test_split is None:
        ratios = resolve_split_ratios(
            split_cfg=data_cfg.get("splits", None),
            train_split=dcfg.get("train_split", data_cfg.get("train_split", None)),
            default=(
                float(data_cfg.get("default_train_ratio", 0.70)),
                float(data_cfg.get("default_val_ratio", 0.15)),
                float(data_cfg.get("default_test_ratio", 0.15)),
            ),
        )
        _, _, test_df = chronological_split_dataframe(df, split_ratios=ratios)
        test_start = len(df) - len(test_df)
    else:
        test_split = float(test_split)
        test_count = max(seq_len + pred_len + 1, int(round(len(df) * test_split)))
        test_start = max(0, len(df) - test_count - seq_len)
        test_df = df.iloc[test_start:].copy()

    add_time = bool(data_cfg.get("add_time_features", False))
    tf_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(tf_cfg, dict) and "enabled" in tf_cfg:
        add_time = bool(tf_cfg.get("enabled")) or add_time

    return GroupedTimeSeriesDataset(
        test_df,
        groups,
        input_groups,
        output_groups,
        seq_len=seq_len,
        pred_len=pred_len,
        scale=True,
        add_time=add_time,
        time_features_cfg=tf_cfg,
        scaler=scaler,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
    )


def _plot_curve(y: np.ndarray, title: str, ylabel: str, out_path: Path):
    if y.size == 0:
        return
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=2.0)
    plt.xlabel("Horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_horizon_csv(
    out_path: Path,
    curves: Dict[str, Any],
    coverage_levels: Sequence[float],
):
    horizon = len(curves.get("rmse", []))
    rows = []
    cov = curves.get("coverage", {})
    for t in range(horizon):
        row = {
            "horizon": t + 1,
            "rmse": float(curves.get("rmse", [np.nan] * horizon)[t]),
            "mae": float(curves.get("mae", [np.nan] * horizon)[t]),
            "crps": float(curves.get("crps", [np.nan] * horizon)[t]),
            "nll": float(curves.get("nll", [np.nan] * horizon)[t]) if len(curves.get("nll", [])) == horizon else np.nan,
            "sharpness_90": float(curves.get("sharpness_90", [np.nan] * horizon)[t])
            if len(curves.get("sharpness_90", [])) == horizon else np.nan,
        }
        for lvl in coverage_levels:
            c = cov.get(float(lvl), np.empty((0,), dtype=np.float32))
            row[f"coverage_{int(round(100 * float(lvl)))}"] = float(c[t]) if len(c) == horizon else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _save_selected_horizons_csv(
    out_path: Path,
    curves: Dict[str, Any],
    horizons: Sequence[int] = (1, 5, 10, 20),
):
    rows = []
    for h in horizons:
        idx = int(h) - 1
        row = {
            "horizon": int(h),
            "rmse": np.nan,
            "mae": np.nan,
            "crps": np.nan,
            "nll": np.nan,
            "sharpness_90": np.nan,
        }
        for key in ("rmse", "mae", "crps", "nll", "sharpness_90"):
            arr = np.asarray(curves.get(key, []), dtype=np.float32)
            if 0 <= idx < arr.size:
                row[key] = float(arr[idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _save_per_objective_horizon_csv(
    out_path: Path,
    curves: Dict[str, Any],
    objective_names: Sequence[str],
):
    rmse = np.asarray(curves.get("rmse_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    mae = np.asarray(curves.get("mae_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    crps = np.asarray(curves.get("crps_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    nll = np.asarray(curves.get("nll_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    horizon = int(rmse.shape[0]) if rmse.ndim == 2 else 0
    out_dim = int(rmse.shape[1]) if rmse.ndim == 2 else 0

    rows = []
    for t in range(horizon):
        for j in range(out_dim):
            rows.append(
                {
                    "horizon": t + 1,
                    "objective_index": j,
                    "objective": objective_names[j] if j < len(objective_names) else f"y{j}",
                    "rmse": float(rmse[t, j]) if rmse.size else np.nan,
                    "mae": float(mae[t, j]) if mae.shape == rmse.shape else np.nan,
                    "crps": float(crps[t, j]) if crps.shape == rmse.shape else np.nan,
                    "nll": float(nll[t, j]) if nll.shape == rmse.shape else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _save_per_objective_summary_csv(
    out_path: Path,
    curves: Dict[str, Any],
    objective_names: Sequence[str],
):
    rmse = np.asarray(curves.get("rmse_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    mae = np.asarray(curves.get("mae_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    crps = np.asarray(curves.get("crps_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    nll = np.asarray(curves.get("nll_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    out_dim = int(rmse.shape[1]) if rmse.ndim == 2 else 0

    rows = []
    for j in range(out_dim):
        rows.append(
            {
                "objective_index": j,
                "objective": objective_names[j] if j < len(objective_names) else f"y{j}",
                "rmse_mean_over_horizon": _finite_mean(rmse[:, j]) if rmse.size else np.nan,
                "mae_mean_over_horizon": _finite_mean(mae[:, j]) if mae.shape == rmse.shape else np.nan,
                "crps_mean_over_horizon": _finite_mean(crps[:, j]) if crps.shape == rmse.shape else np.nan,
                "nll_mean_over_horizon": _finite_mean(nll[:, j]) if nll.shape == rmse.shape else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _calibration_summary(curves: Dict[str, Any], levels: Sequence[float]) -> pd.DataFrame:
    cov = curves.get("coverage", {})
    rows = []
    for lvl in levels:
        arr = cov.get(float(lvl), np.empty((0,), dtype=np.float32))
        rows.append({
            "nominal": float(lvl),
            "actual": float(np.mean(arr)) if arr.size > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _coverage_table(
    curves: Dict[str, Any],
    levels: Sequence[float],
    split: str,
) -> pd.DataFrame:
    cov = curves.get("coverage", {})
    rows = []
    for lvl in levels:
        arr = cov.get(float(lvl), np.empty((0,), dtype=np.float32))
        rows.append(
            {
                "split": split,
                "nominal": float(lvl),
                "actual": float(np.mean(arr)) if arr.size > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _sharpness_summary(
    curves: Dict[str, Any],
    split: str,
) -> pd.DataFrame:
    sharp = np.asarray(curves.get("sharpness_90", []), dtype=np.float32)
    rows = [
        {
            "split": split,
            "metric": "sharpness_90_mean",
            "value": _finite_mean(sharp),
        }
    ]
    for h in (1, 5, 10, 20):
        idx = h - 1
        rows.append(
            {
                "split": split,
                "metric": f"sharpness_90_h{h}",
                "value": float(sharp[idx]) if 0 <= idx < sharp.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _per_objective_summary_dict(
    curves: Dict[str, Any],
    objective_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    rmse = np.asarray(curves.get("rmse_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    mae = np.asarray(curves.get("mae_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    crps = np.asarray(curves.get("crps_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    nll = np.asarray(curves.get("nll_per_dim", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    out_dim = int(rmse.shape[1]) if rmse.ndim == 2 else 0
    out: Dict[str, Dict[str, float]] = {}
    for j in range(out_dim):
        key = objective_names[j] if j < len(objective_names) else f"y{j}"
        out[key] = {
            "rmse_mean_over_horizon": _finite_mean(rmse[:, j]) if rmse.size else np.nan,
            "mae_mean_over_horizon": _finite_mean(mae[:, j]) if mae.shape == rmse.shape else np.nan,
            "crps_mean_over_horizon": _finite_mean(crps[:, j]) if crps.shape == rmse.shape else np.nan,
            "nll_mean_over_horizon": _finite_mean(nll[:, j]) if nll.shape == rmse.shape else np.nan,
        }
    return out


def _finite_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(arr[mask].mean())


def _save_records_csv(out_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(out_path, index=False)


def _save_trajectory_means_csv(
    out_path: Path,
    trajectory_means: Dict[str, np.ndarray],
    objective_names: Sequence[str],
) -> None:
    rows = []
    for scenario, traj in trajectory_means.items():
        arr = np.asarray(traj, dtype=np.float32)
        if arr.ndim != 2:
            continue
        horizon, out_dim = arr.shape
        for t in range(horizon):
            for j in range(out_dim):
                rows.append(
                    {
                        "scenario": str(scenario),
                        "horizon": int(t + 1),
                        "objective_index": int(j),
                        "objective": objective_names[j] if j < len(objective_names) else f"y{j}",
                        "value": float(arr[t, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _inverse_outputs(dataset: GroupedTimeSeriesDataset, arr: np.ndarray) -> np.ndarray:
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


def _decode_from_state(model, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_decode_obs"):
        _, _, loc, _, _ = model._decode_obs(h, z)  # type: ignore[attr-defined]
        return loc
    if hasattr(model, "obs_decoder"):
        latent = torch.cat([h, z], dim=-1)
        min_scale = max(0.01, float(getattr(model, "min_scale", 1e-4)))
        _, loc_latent, _ = model.obs_decoder(latent, min_scale=min_scale)
        if bool(getattr(model, "use_symlog", False)) and hasattr(model, "symexp"):
            return model.symexp(loc_latent)
        return loc_latent
    raise RuntimeError("Model does not expose a supported decode path for latent traversal.")


def _latent_traversal(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    start_idx: int,
    device: str | torch.device,
) -> Dict[str, Any]:
    values = dataset.values
    if start_idx + warmup_len > len(values):
        return {
            "deltas_sigma": np.empty((0,), dtype=np.float32),
            "trajectory": np.empty((0, len(dataset.out_idx)), dtype=np.float32),
            "latent_dim": -1,
            "latent_sigma": np.nan,
            "smoothness_ratio": np.nan,
            "effect_range_mean": np.nan,
        }
    w = values[start_idx:start_idx + warmup_len]
    w_inputs = w[:, dataset.in_idx]
    w_out = w[:, dataset.out_idx]
    c = (
        w_inputs[:, dataset.control_positions]
        if dataset.control_positions
        else np.zeros((warmup_len, 0), dtype=np.float32)
    )
    x = (
        w_inputs[:, dataset.known_exo_positions]
        if dataset.known_exo_positions
        else np.zeros((warmup_len, 0), dtype=np.float32)
    )
    latent = latent_diagnostics(
        model=model,
        controls=torch.from_numpy(c).unsqueeze(0).to(device),
        exogenous=torch.from_numpy(x).unsqueeze(0).to(device),
        observations=torch.from_numpy(w_out.astype(np.float32)).unsqueeze(0).to(device),
    )
    h_last = latent["deter"][0, -1, :]
    z_last = latent["stoch"][0, -1, :]
    post_logvar_last = latent["posterior_logvar"][0, -1, :]
    post_std_last = torch.exp(0.5 * post_logvar_last).clamp_min(1e-6)
    dim = int(torch.argmax(post_std_last).item())
    sigma = float(post_std_last[dim].item())

    deltas = np.linspace(-2.0, 2.0, num=9, dtype=np.float32)
    traj = []
    with torch.no_grad():
        for d in deltas:
            z_t = z_last.clone()
            z_t[dim] = z_t[dim] + float(d) * sigma
            loc = _decode_from_state(model, h_last.unsqueeze(0), z_t.unsqueeze(0))
            traj.append(loc.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False))
    traj_arr = np.stack(traj, axis=0).astype(np.float32, copy=False)
    traj_eval = _inverse_outputs(dataset, traj_arr)

    if traj_eval.shape[0] >= 3:
        d1 = np.diff(traj_eval, axis=0)
        d2 = np.diff(d1, axis=0)
        smoothness = float(np.mean(np.abs(d2)) / (np.mean(np.abs(d1)) + 1e-8))
    else:
        smoothness = np.nan
    effect_range_mean = float(np.mean(np.ptp(traj_eval, axis=0)))

    return {
        "deltas_sigma": deltas,
        "trajectory": traj_eval,
        "latent_dim": dim,
        "latent_sigma": sigma,
        "smoothness_ratio": smoothness,
        "effect_range_mean": effect_range_mean,
    }


def _plot_latent_traversal(
    deltas_sigma: np.ndarray,
    trajectory: np.ndarray,
    objective_names: Sequence[str],
    out_path: Path,
) -> None:
    if deltas_sigma.size == 0 or trajectory.size == 0:
        return
    plt.figure(figsize=(8, 4))
    for j in range(trajectory.shape[-1]):
        label = objective_names[j] if j < len(objective_names) else f"y{j}"
        plt.plot(deltas_sigma, trajectory[:, j], marker="o", linewidth=1.8, label=label)
    plt.xlabel("Latent Perturbation (sigma)")
    plt.ylabel("Decoded Objective")
    plt.title("Latent Traversal")
    plt.grid(True, alpha=0.3)
    if trajectory.shape[-1] <= 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _reconstruction_sanity(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    horizon: int,
    n_windows: int,
    device: str | torch.device,
) -> Dict[str, Any]:
    starts = np.linspace(
        0,
        max(0, len(dataset.values) - (warmup_len + horizon)),
        num=max(1, int(n_windows)),
        dtype=int,
    ).tolist()
    true_list = []
    recon_list = []
    overlay_true = np.empty((0, len(dataset.out_idx)), dtype=np.float32)
    overlay_recon = np.empty((0, len(dataset.out_idx)), dtype=np.float32)

    with torch.no_grad():
        for idx, s in enumerate(starts):
            seq = dataset.values[s:s + warmup_len + horizon]
            if seq.shape[0] < 2:
                continue
            seq_inputs = seq[:, dataset.in_idx]
            seq_out = seq[:, dataset.out_idx]
            c = (
                seq_inputs[:, dataset.control_positions]
                if dataset.control_positions
                else np.zeros((seq.shape[0], 0), dtype=np.float32)
            )
            x = (
                seq_inputs[:, dataset.known_exo_positions]
                if dataset.known_exo_positions
                else np.zeros((seq.shape[0], 0), dtype=np.float32)
            )
            observed = model.observe(
                controls=torch.from_numpy(c).unsqueeze(0).to(device),
                exogenous=torch.from_numpy(x).unsqueeze(0).to(device),
                observations=torch.from_numpy(seq_out.astype(np.float32)).unsqueeze(0).to(device),
                initial_state=None,
                sample_posterior=False,
            )
            recon = observed["predictions"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            true_eval = _inverse_outputs(dataset, seq_out.astype(np.float32, copy=False))
            recon_eval = _inverse_outputs(dataset, recon)
            true_list.append(true_eval)
            recon_list.append(recon_eval)
            if idx == 0:
                overlay_true = true_eval
                overlay_recon = recon_eval

    if not true_list:
        return {
            "overlay_true": overlay_true,
            "overlay_recon": overlay_recon,
            "rmse_per_dim": np.empty((0,), dtype=np.float32),
            "mae_per_dim": np.empty((0,), dtype=np.float32),
            "corr_per_dim": np.empty((0,), dtype=np.float32),
            "rmse_mean": np.nan,
            "corr_mean": np.nan,
        }

    true_arr = np.stack(true_list, axis=0).astype(np.float32, copy=False)
    recon_arr = np.stack(recon_list, axis=0).astype(np.float32, copy=False)
    err = recon_arr - true_arr
    rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=(0, 1))).astype(np.float32, copy=False)
    mae_per_dim = np.mean(np.abs(err), axis=(0, 1)).astype(np.float32, copy=False)
    corr = []
    for j in range(true_arr.shape[-1]):
        y_true = true_arr[:, :, j].reshape(-1)
        y_pred = recon_arr[:, :, j].reshape(-1)
        if y_true.size < 2 or np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(y_true, y_pred)[0, 1]))
    corr_per_dim = np.asarray(corr, dtype=np.float32)
    return {
        "overlay_true": overlay_true,
        "overlay_recon": overlay_recon,
        "rmse_per_dim": rmse_per_dim,
        "mae_per_dim": mae_per_dim,
        "corr_per_dim": corr_per_dim,
        "rmse_mean": _finite_mean(rmse_per_dim),
        "corr_mean": _finite_mean(corr_per_dim),
    }


def _plot_reconstruction_overlay(
    true_arr: np.ndarray,
    recon_arr: np.ndarray,
    objective_names: Sequence[str],
    out_path: Path,
) -> None:
    if true_arr.size == 0 or recon_arr.size == 0:
        return
    n_obj = true_arr.shape[-1]
    obj = 0
    y_true = true_arr[:, obj]
    y_recon = recon_arr[:, obj]
    x = np.arange(1, y_true.shape[0] + 1)
    label = objective_names[obj] if obj < len(objective_names) else f"y{obj}"
    plt.figure(figsize=(9, 4))
    plt.plot(x, y_true, linewidth=2.0, label=f"{label} true")
    plt.plot(x, y_recon, linewidth=1.8, linestyle="--", label=f"{label} recon")
    plt.xlabel("Timestep")
    plt.ylabel("Objective Value")
    plt.title("Reconstruction Sanity (Observe Mode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _compute_latent_kl_windows(
    model,
    dataset: GroupedTimeSeriesDataset,
    warmup_len: int,
    n_windows: int,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute KL-per-timestep and KL-per-window over multiple windows."""
    values = dataset.values
    max_start = len(values) - warmup_len
    if max_start <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    n_windows = max(1, int(n_windows))
    starts = np.linspace(0, max_start, num=n_windows, dtype=int)
    kl_curves = []
    kl_means = []

    with torch.no_grad():
        for s in starts:
            w = values[s:s + warmup_len]
            w_inputs = w[:, dataset.in_idx]
            w_out = w[:, dataset.out_idx]
            c = (
                w_inputs[:, dataset.control_positions]
                if dataset.control_positions
                else np.zeros((warmup_len, 0), dtype=np.float32)
            )
            x = (
                w_inputs[:, dataset.known_exo_positions]
                if dataset.known_exo_positions
                else np.zeros((warmup_len, 0), dtype=np.float32)
            )
            latent = latent_diagnostics(
                model=model,
                controls=torch.from_numpy(c).unsqueeze(0).to(device),
                exogenous=torch.from_numpy(x).unsqueeze(0).to(device),
                observations=torch.from_numpy(w_out.astype(np.float32)).unsqueeze(0).to(device),
            )
            kl_terms = latent["kl_terms"].squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            kl_curves.append(kl_terms)
            kl_means.append(float(np.mean(kl_terms)))

    return (
        np.mean(np.stack(kl_curves, axis=0), axis=0).astype(np.float32, copy=False),
        np.asarray(kl_means, dtype=np.float32),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("misc", {})
        config["misc"]["device"] = args.device
    seed = int(config.get("misc", {}).get("seed", 42))
    deterministic = bool(config.get("misc", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(config.get("misc", {}).get("device", "auto"))
    config.setdefault("misc", {})
    config["misc"]["device"] = device

    checkpoint = _resolve_checkpoint(config, args.checkpoint)
    model_dir = checkpoint.resolve().parent
    if model_dir.name == "checkpoints":
        model_dir = model_dir.parent

    out_dir = Path(args.output_dir) if args.output_dir else (model_dir / "eval_suite")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        parent_scaler = model_dir.parent / "scaler.pkl"
        if parent_scaler.exists():
            scaler_path = parent_scaler
        else:
            raise FileNotFoundError(
                f"Scaler not found in {model_dir} or {model_dir.parent}"
            )
    scaler = joblib_load(scaler_path)

    dataset = _build_test_dataset(config, scaler=scaler)

    dcfg = config["dataset"]
    groups = dcfg["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]
    input_cols = schema.columns_for_group_names(input_groups)
    output_cols = schema.columns_for_group_names(output_groups)
    input_dim = len(set(input_cols) | set(output_cols))

    data_cfg = config.get("data", {})
    add_time = bool(data_cfg.get("add_time_features", False))
    tf_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(tf_cfg, dict) and "enabled" in tf_cfg:
        add_time = bool(tf_cfg.get("enabled")) or add_time
    if add_time:
        input_dim += len(
            get_time_feature_columns(
                features=tf_cfg.get("features"),
                encoding=tf_cfg.get("encoding", "cyclical"),
            )
        )
    output_dim = len(output_cols)
    seq_len = int(dcfg["seq_len"])
    pred_len = int(dcfg["pred_len"])

    models_cfg = {m["type"]: m for m in config.get("models", [])}
    latent_cfg = models_cfg.get("latent_ssm", {"type": "latent_ssm"})
    model = build_model(
        "latent_ssm",
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        per_model_cfg=latent_cfg,
        model_defaults_cfg=config.get("model_defaults", {}),
    )
    _load_model(model, checkpoint, device)

    eval_cfg = config.get("evaluation", {}) or {}
    prob_eval_cfg = eval_cfg.get("probabilistic", {}) or {}
    warmup_len = int(config.get("training", {}).get("warmup_len", seq_len))
    horizon = int(args.horizon or eval_cfg.get("horizon", max(pred_len, 20)))
    n_windows = int(args.n_windows or eval_cfg.get("n_windows", 8))
    mc_samples = int(args.mc_samples or prob_eval_cfg.get("mc_samples", 128))
    sigma_scale = float(
        args.sigma_scale
        if args.sigma_scale is not None
        else prob_eval_cfg.get("sigma_scale", 1.0)
    )
    sigma_scale = float(max(1e-6, sigma_scale))

    coverage_table_levels_cfg = prob_eval_cfg.get(
        "coverage_table_levels",
        [0.5, 0.8, 0.9, 0.95],
    )
    coverage_table_levels = tuple(
        sorted({float(x) for x in coverage_table_levels_cfg if 0.0 < float(x) < 1.0})
    )
    if not coverage_table_levels:
        coverage_table_levels = (0.5, 0.8, 0.9, 0.95)

    calibration_levels_cfg = prob_eval_cfg.get(
        "calibration_levels",
        [round(v, 2) for v in np.arange(0.50, 1.00, 0.01)],
    )
    calibration_levels = tuple(
        sorted({float(x) for x in calibration_levels_cfg if 0.0 < float(x) < 1.0})
    )
    if not calibration_levels:
        calibration_levels = tuple(round(v, 2) for v in np.arange(0.50, 1.00, 0.01))

    interval_levels = tuple(
        sorted(set(coverage_table_levels) | set(calibration_levels))
    )

    open_curves = open_loop_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        horizon=horizon,
        n_windows=n_windows,
        n_samples=mc_samples,
        device=device,
        denormalize=True,
        interval_levels=interval_levels,
        sigma_scale=sigma_scale,
    )
    closed_curves = closed_loop_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        horizon=horizon,
        n_windows=n_windows,
        n_samples=mc_samples,
        device=device,
        denormalize=True,
        interval_levels=interval_levels,
        sigma_scale=sigma_scale,
    )
    inter_cfg = eval_cfg.get("interventional", {}) or {}
    inter_control_index: Optional[int] = (
        int(inter_cfg["control_index"]) if inter_cfg.get("control_index", None) is not None else None
    )
    inter_objective_index: Optional[int] = (
        int(inter_cfg["objective_index"]) if inter_cfg.get("objective_index", None) is not None else None
    )
    inter_exogenous_index: Optional[int] = (
        int(inter_cfg["exogenous_index"]) if inter_cfg.get("exogenous_index", None) is not None else None
    )
    expected_sign_raw = inter_cfg.get("expected_direction_sign", None)
    expected_sign: Optional[float]
    if expected_sign_raw is None:
        expected_sign = None
    else:
        exp = float(expected_sign_raw)
        expected_sign = 1.0 if exp >= 0.0 else -1.0
    inter_direction_n_windows = int(inter_cfg.get("direction_n_windows", 100))
    inter_exogenous_step_size = float(inter_cfg.get("exogenous_step_size", inter_cfg.get("control_step_size", 1.0)))
    inter_sensitivity_threshold_ratio = float(inter_cfg.get("sensitivity_threshold_ratio", 0.01))
    inter_irrelevance_pairs = inter_cfg.get("control_irrelevance_pairs", [])
    inter_irrelevance_tol_ratio = float(inter_cfg.get("irrelevance_tolerance_ratio", 0.05))
    inter_irrelevance_tol_abs = float(inter_cfg.get("irrelevance_tolerance_abs", 1e-4))
    inter_extreme_sigma = float(inter_cfg.get("extreme_sigma", 3.0))
    inter_extreme_widen_ratio = float(inter_cfg.get("extreme_widen_ratio", 1.2))
    inter_extreme_min_std_ratio = float(inter_cfg.get("extreme_min_std_ratio", 0.05))
    inter_extreme_min_std_abs = float(inter_cfg.get("extreme_min_std_abs", 1e-4))

    intervention = interventional_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        horizon=horizon,
        n_windows=n_windows,
        n_samples=mc_samples,
        scenario=str(inter_cfg.get("scenario", "step")),
        control_step_size=float(inter_cfg.get("control_step_size", 1.0)),
        control_index=inter_control_index,
        objective_index=inter_objective_index,
        device=device,
        denormalize=True,
    )
    intervention_suite = interventional_suite_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        horizon=horizon,
        n_windows=n_windows,
        n_samples=mc_samples,
        control_index=inter_control_index,
        objective_index=inter_objective_index,
        exogenous_index=inter_exogenous_index,
        expected_direction_sign=expected_sign,
        direction_n_windows=inter_direction_n_windows,
        control_step_size=float(inter_cfg.get("control_step_size", 1.0)),
        exogenous_step_size=inter_exogenous_step_size,
        sensitivity_threshold_ratio=inter_sensitivity_threshold_ratio,
        irrelevance_pairs=inter_irrelevance_pairs,
        irrelevance_tolerance_ratio=inter_irrelevance_tol_ratio,
        irrelevance_tolerance_abs=inter_irrelevance_tol_abs,
        extreme_sigma=inter_extreme_sigma,
        extreme_widen_ratio=inter_extreme_widen_ratio,
        extreme_min_std_ratio=inter_extreme_min_std_ratio,
        extreme_min_std_abs=inter_extreme_min_std_abs,
        sigma_scale=sigma_scale,
        random_seed=seed,
        device=device,
        denormalize=True,
    )

    direction_curve_raw = intervention["direction_score"]
    direction_curve_aligned = (
        direction_curve_raw * float(expected_sign)
        if expected_sign is not None
        else direction_curve_raw
    )
    direction_mean_raw = _finite_mean(direction_curve_raw)
    direction_mean_aligned = _finite_mean(direction_curve_aligned)

    latent_n_windows = int(eval_cfg.get("latent_n_windows", n_windows))
    kl_per_timestep, kl_window_means = _compute_latent_kl_windows(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        n_windows=latent_n_windows,
        device=device,
    )
    mean_test_kl = _finite_mean(kl_window_means)
    training_prob_cfg = config.get("training", {}).get("probabilistic", {}) or {}
    free_nats = float(training_prob_cfg.get("kl_free_bits", 1.0))
    kl_above_free_ratio = float(
        np.mean((kl_per_timestep > free_nats).astype(np.float32))
    ) if kl_per_timestep.size > 0 else np.nan
    kl_timestep_mean = _finite_mean(kl_per_timestep)
    kl_timestep_std = float(np.nanstd(kl_per_timestep)) if kl_per_timestep.size > 0 else np.nan
    free_tol = 0.10 * max(abs(free_nats), 1e-6)
    kl_flat_at_free = bool(
        np.isfinite(kl_timestep_mean)
        and np.isfinite(kl_timestep_std)
        and abs(kl_timestep_mean - free_nats) <= free_tol
        and kl_timestep_std <= free_tol
    )

    traversal_start = int(open_curves.get("starts", np.asarray([0], dtype=np.int64))[0]) if len(open_curves.get("starts", [])) > 0 else 0
    latent_traversal = _latent_traversal(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        start_idx=traversal_start,
        device=device,
    )
    reconstruction = _reconstruction_sanity(
        model=model,
        dataset=dataset,
        warmup_len=warmup_len,
        horizon=horizon,
        n_windows=max(1, n_windows),
        device=device,
    )

    # Persist metrics.
    _save_horizon_csv(out_dir / "open_loop_horizon_metrics.csv", open_curves, coverage_table_levels)
    _save_horizon_csv(out_dir / "closed_loop_horizon_metrics.csv", closed_curves, coverage_table_levels)
    _save_selected_horizons_csv(out_dir / "open_loop_horizons_1_5_10_20.csv", open_curves)
    _save_selected_horizons_csv(out_dir / "closed_loop_horizons_1_5_10_20.csv", closed_curves)
    _save_per_objective_horizon_csv(
        out_dir / "open_loop_per_objective_horizon_metrics.csv",
        open_curves,
        output_cols,
    )
    _save_per_objective_horizon_csv(
        out_dir / "closed_loop_per_objective_horizon_metrics.csv",
        closed_curves,
        output_cols,
    )
    _save_per_objective_summary_csv(
        out_dir / "open_loop_per_objective_summary.csv",
        open_curves,
        output_cols,
    )
    _save_per_objective_summary_csv(
        out_dir / "closed_loop_per_objective_summary.csv",
        closed_curves,
        output_cols,
    )

    inter_df = pd.DataFrame({
        "horizon": np.arange(1, len(intervention["delta_abs"]) + 1),
        "delta_abs": intervention["delta_abs"],
        "delta_signed": intervention["delta_signed"],
        "direction_score_raw": direction_curve_raw,
        "direction_score_aligned": direction_curve_aligned,
    })
    inter_df.to_csv(out_dir / "interventional_metrics.csv", index=False)
    _save_records_csv(
        out_dir / "interventional_control_sensitivity_windows.csv",
        intervention_suite["control_sensitivity"]["window_rows"],
    )
    _save_trajectory_means_csv(
        out_dir / "interventional_control_sensitivity_trajectories.csv",
        intervention_suite["control_sensitivity"]["trajectory_means"],
        output_cols,
    )
    _save_records_csv(
        out_dir / "interventional_direction_check_windows.csv",
        intervention_suite["direction_check"]["window_rows"],
    )
    _save_records_csv(
        out_dir / "interventional_exogenous_sensitivity_windows.csv",
        intervention_suite["exogenous_sensitivity"]["window_rows"],
    )
    _save_trajectory_means_csv(
        out_dir / "interventional_exogenous_sensitivity_trajectories.csv",
        intervention_suite["exogenous_sensitivity"]["trajectory_means"],
        output_cols,
    )
    _save_records_csv(
        out_dir / "interventional_control_irrelevance_windows.csv",
        intervention_suite["control_irrelevance"]["rows"],
    )
    _save_records_csv(
        out_dir / "interventional_control_irrelevance_summary.csv",
        intervention_suite["control_irrelevance"]["summary"],
    )
    _save_records_csv(
        out_dir / "interventional_extreme_control_windows.csv",
        intervention_suite["extreme_control"]["window_rows"],
    )

    coverage_open_df = _coverage_table(open_curves, coverage_table_levels, split="open_loop")
    coverage_closed_df = _coverage_table(closed_curves, coverage_table_levels, split="closed_loop")
    coverage_df = pd.concat([coverage_open_df, coverage_closed_df], ignore_index=True)
    coverage_df.to_csv(out_dir / "coverage_table.csv", index=False)

    sharpness_df = pd.concat(
        [
            _sharpness_summary(open_curves, split="open_loop"),
            _sharpness_summary(closed_curves, split="closed_loop"),
        ],
        ignore_index=True,
    )
    sharpness_df.to_csv(out_dir / "sharpness_summary.csv", index=False)

    cal_df = _calibration_summary(open_curves, calibration_levels)
    cal_df.to_csv(out_dir / "calibration_curve.csv", index=False)

    pd.DataFrame(
        {
            "timestep": np.arange(1, len(kl_per_timestep) + 1),
            "kl": kl_per_timestep,
        }
    ).to_csv(out_dir / "latent_kl_per_timestep.csv", index=False)
    pd.DataFrame(
        {
            "window_index": np.arange(1, len(kl_window_means) + 1),
            "kl_mean": kl_window_means,
        }
    ).to_csv(out_dir / "latent_kl_per_window.csv", index=False)

    traversal_rows = []
    traversal_vals = np.asarray(latent_traversal.get("trajectory", np.empty((0, 0))), dtype=np.float32)
    traversal_deltas = np.asarray(latent_traversal.get("deltas_sigma", np.empty((0,))), dtype=np.float32)
    if traversal_vals.ndim == 2 and traversal_deltas.ndim == 1 and traversal_vals.shape[0] == traversal_deltas.shape[0]:
        for i in range(traversal_vals.shape[0]):
            for j in range(traversal_vals.shape[1]):
                traversal_rows.append(
                    {
                        "delta_sigma": float(traversal_deltas[i]),
                        "objective_index": int(j),
                        "objective": output_cols[j] if j < len(output_cols) else f"y{j}",
                        "value": float(traversal_vals[i, j]),
                        "latent_dim": int(latent_traversal.get("latent_dim", -1)),
                        "latent_sigma": float(latent_traversal.get("latent_sigma", np.nan)),
                    }
                )
    pd.DataFrame(traversal_rows).to_csv(out_dir / "latent_traversal.csv", index=False)

    recon_rmse = np.asarray(reconstruction.get("rmse_per_dim", np.empty((0,))), dtype=np.float32)
    recon_mae = np.asarray(reconstruction.get("mae_per_dim", np.empty((0,))), dtype=np.float32)
    recon_corr = np.asarray(reconstruction.get("corr_per_dim", np.empty((0,))), dtype=np.float32)
    recon_rows = []
    for j in range(recon_rmse.shape[0]):
        recon_rows.append(
            {
                "objective_index": int(j),
                "objective": output_cols[j] if j < len(output_cols) else f"y{j}",
                "rmse": float(recon_rmse[j]),
                "mae": float(recon_mae[j]) if j < recon_mae.shape[0] else np.nan,
                "corr": float(recon_corr[j]) if j < recon_corr.shape[0] else np.nan,
            }
        )
    pd.DataFrame(recon_rows).to_csv(out_dir / "reconstruction_summary.csv", index=False)

    overlay_true = np.asarray(reconstruction.get("overlay_true", np.empty((0, 0))), dtype=np.float32)
    overlay_recon = np.asarray(reconstruction.get("overlay_recon", np.empty((0, 0))), dtype=np.float32)
    overlay_rows = []
    if overlay_true.ndim == 2 and overlay_recon.shape == overlay_true.shape:
        for t in range(overlay_true.shape[0]):
            for j in range(overlay_true.shape[1]):
                overlay_rows.append(
                    {
                        "timestep": int(t + 1),
                        "objective_index": int(j),
                        "objective": output_cols[j] if j < len(output_cols) else f"y{j}",
                        "true": float(overlay_true[t, j]),
                        "recon": float(overlay_recon[t, j]),
                    }
                )
    pd.DataFrame(overlay_rows).to_csv(out_dir / "reconstruction_overlay_series.csv", index=False)

    _plot_curve(open_curves.get("rmse", np.empty((0,))), "Open-loop RMSE vs Horizon", "RMSE", out_dir / "open_loop_rmse.png")
    _plot_curve(open_curves.get("crps", np.empty((0,))), "Open-loop CRPS vs Horizon", "CRPS", out_dir / "open_loop_crps.png")
    _plot_curve(closed_curves.get("rmse", np.empty((0,))), "Closed-loop RMSE vs Horizon", "RMSE", out_dir / "closed_loop_rmse.png")
    _plot_curve(closed_curves.get("crps", np.empty((0,))), "Closed-loop CRPS vs Horizon", "CRPS", out_dir / "closed_loop_crps.png")
    _plot_curve(open_curves.get("sharpness_90", np.empty((0,))), "Open-loop 90% Interval Width", "Width", out_dir / "open_loop_sharpness90.png")
    _plot_curve(kl_per_timestep, "KL per Timestep (observe)", "KL", out_dir / "latent_kl_per_timestep.png")
    _plot_latent_traversal(
        traversal_deltas,
        traversal_vals,
        output_cols,
        out_dir / "latent_traversal.png",
    )
    _plot_reconstruction_overlay(
        overlay_true,
        overlay_recon,
        output_cols,
        out_dir / "reconstruction_overlay.png",
    )

    if not cal_df.empty:
        plt.figure(figsize=(5, 5))
        plt.plot(cal_df["nominal"], cal_df["actual"], marker="o", linewidth=1.8)
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
        plt.xlim(0.45, 1.0)
        plt.ylim(0.45, 1.0)
        plt.xlabel("Nominal Coverage")
        plt.ylabel("Actual Coverage")
        plt.title("Calibration Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_curve.png", dpi=140)
        plt.close()

    open_hsum = summarize_horizons(open_curves)
    closed_hsum = summarize_horizons(closed_curves)
    rmse_h1 = float(open_hsum.get("rmse", {}).get(1, np.nan))
    rmse_h20 = float(open_hsum.get("rmse", {}).get(20, np.nan))
    rmse_ratio_h20_h1 = float(rmse_h20 / rmse_h1) if np.isfinite(rmse_h1) and rmse_h1 > 0 else np.nan

    coverage95 = np.nan
    cov95_curve = open_curves.get("coverage", {}).get(0.95, np.empty((0,), dtype=np.float32))
    if cov95_curve.size > 0:
        coverage95 = float(np.mean(cov95_curve))

    summary = {
        "checkpoint": str(checkpoint),
        "test_windows": int(n_windows),
        "horizon": int(horizon),
        "open_loop_horizon_summary": open_hsum,
        "closed_loop_horizon_summary": closed_hsum,
        "open_loop_per_objective_summary": _per_objective_summary_dict(open_curves, output_cols),
        "closed_loop_per_objective_summary": _per_objective_summary_dict(closed_curves, output_cols),
        "coverage_table_levels": list(coverage_table_levels),
        "calibration_levels": list(calibration_levels),
        "coverage_table_open_loop": {
            str(int(round(100 * float(row["nominal"])))): float(row["actual"])
            for _, row in coverage_open_df.iterrows()
        },
        "coverage_table_closed_loop": {
            str(int(round(100 * float(row["nominal"])))): float(row["actual"])
            for _, row in coverage_closed_df.iterrows()
        },
        "sharpness_90_open_loop_mean": _finite_mean(
            np.asarray(open_curves.get("sharpness_90", []), dtype=np.float32)
        ),
        "sharpness_90_closed_loop_mean": _finite_mean(
            np.asarray(closed_curves.get("sharpness_90", []), dtype=np.float32)
        ),
        "sigma_scale": float(sigma_scale),
        "coverage_95_mean": coverage95,
        "open_loop_rmse_h20_over_h1": rmse_ratio_h20_h1,
        "intervention_direction_mean_raw": direction_mean_raw,
        "intervention_direction_mean_aligned": direction_mean_aligned,
        "intervention_expected_sign": expected_sign,
        "interventional_suite": {
            "control_sensitivity_diff_rate": intervention_suite["control_sensitivity"]["diff_rate"],
            "exogenous_sensitivity_diff_rate": intervention_suite["exogenous_sensitivity"]["diff_rate"],
            "direction_agreement_rate": intervention_suite["direction_check"]["agreement_rate"],
            "control_irrelevance_overall_pass_rate": intervention_suite["control_irrelevance"]["overall_pass_rate"],
            "extreme_control_finite_rate": intervention_suite["extreme_control"]["finite_rate"],
            "extreme_control_widen_rate": intervention_suite["extreme_control"]["widen_rate"],
            "extreme_control_not_confident_rate": intervention_suite["extreme_control"]["not_confident_rate"],
        },
        "latent_space": {
            "free_nats": float(free_nats),
            "kl_timestep_mean": kl_timestep_mean,
            "kl_timestep_std": kl_timestep_std,
            "kl_positions_above_free_ratio": kl_above_free_ratio,
            "kl_flat_at_free": bool(kl_flat_at_free),
            "prior_posterior_kl_mean": mean_test_kl,
            "prior_posterior_kl_reasonable_range": bool(
                np.isfinite(mean_test_kl) and 1.0 <= mean_test_kl <= 50.0
            ),
            "latent_traversal_dim": int(latent_traversal.get("latent_dim", -1)),
            "latent_traversal_sigma": float(latent_traversal.get("latent_sigma", np.nan)),
            "latent_traversal_smoothness_ratio": float(latent_traversal.get("smoothness_ratio", np.nan)),
            "latent_traversal_effect_range_mean": float(latent_traversal.get("effect_range_mean", np.nan)),
            "reconstruction_rmse_mean": float(reconstruction.get("rmse_mean", np.nan)),
            "reconstruction_corr_mean": float(reconstruction.get("corr_mean", np.nan)),
        },
        "latent_kl_mean": mean_test_kl,
        "latent_kl_window_count": int(kl_window_means.size),
        "latent_kl_window_p25": float(np.nanpercentile(kl_window_means, 25))
        if kl_window_means.size > 0 else np.nan,
        "latent_kl_window_p50": float(np.nanpercentile(kl_window_means, 50))
        if kl_window_means.size > 0 else np.nan,
        "latent_kl_window_p75": float(np.nanpercentile(kl_window_means, 75))
        if kl_window_means.size > 0 else np.nan,
        "checks": {
            "latent_kl_gt_0_5": bool(np.isfinite(mean_test_kl) and mean_test_kl > 0.5),
            "rmse_h20_lt_3x_h1": bool(np.isfinite(rmse_ratio_h20_h1) and rmse_ratio_h20_h1 < 3.0),
            "coverage95_in_90_100": bool(np.isfinite(coverage95) and 0.90 <= coverage95 <= 1.00),
            "direction_score_positive": bool(
                np.isfinite(direction_mean_aligned)
                and direction_mean_aligned > 0.0
            ),
            "control_sensitivity_diff_positive": bool(
                np.isfinite(float(intervention_suite["control_sensitivity"]["diff_rate"]))
                and float(intervention_suite["control_sensitivity"]["diff_rate"]) > 0.0
            ),
            "exogenous_sensitivity_diff_positive": bool(
                np.isfinite(float(intervention_suite["exogenous_sensitivity"]["diff_rate"]))
                and float(intervention_suite["exogenous_sensitivity"]["diff_rate"]) > 0.0
            ),
            "direction_agreement_gt_50pct": bool(
                np.isfinite(float(intervention_suite["direction_check"]["agreement_rate"]))
                and float(intervention_suite["direction_check"]["agreement_rate"]) >= 0.5
            ),
            "extreme_control_finite_all": bool(
                np.isfinite(float(intervention_suite["extreme_control"]["finite_rate"]))
                and float(intervention_suite["extreme_control"]["finite_rate"]) >= 1.0
            ),
            "extreme_control_widen_majority": bool(
                np.isfinite(float(intervention_suite["extreme_control"]["widen_rate"]))
                and float(intervention_suite["extreme_control"]["widen_rate"]) >= 0.5
            ),
            "extreme_control_not_confident_majority": bool(
                np.isfinite(float(intervention_suite["extreme_control"]["not_confident_rate"]))
                and float(intervention_suite["extreme_control"]["not_confident_rate"]) >= 0.5
            ),
            "control_irrelevance_majority_pass": bool(
                np.isfinite(float(intervention_suite["control_irrelevance"]["overall_pass_rate"]))
                and float(intervention_suite["control_irrelevance"]["overall_pass_rate"]) >= 0.5
            ),
            "latent_kl_most_positions_gt_free": bool(
                np.isfinite(kl_above_free_ratio) and kl_above_free_ratio >= 0.5
            ),
            "latent_kl_not_flat_at_free": bool(not kl_flat_at_free),
            "prior_posterior_kl_gt_1": bool(np.isfinite(mean_test_kl) and mean_test_kl > 1.0),
            "prior_posterior_kl_lt_50": bool(np.isfinite(mean_test_kl) and mean_test_kl < 50.0),
            "latent_traversal_smooth": bool(
                np.isfinite(float(latent_traversal.get("smoothness_ratio", np.nan)))
                and float(latent_traversal.get("smoothness_ratio", np.nan)) < 2.0
            ),
            "latent_traversal_changes_output": bool(
                np.isfinite(float(latent_traversal.get("effect_range_mean", np.nan)))
                and float(latent_traversal.get("effect_range_mean", np.nan)) > 0.0
            ),
            "reconstruction_corr_gt_0_9": bool(
                np.isfinite(float(reconstruction.get("corr_mean", np.nan)))
                and float(reconstruction.get("corr_mean", np.nan)) >= 0.9
            ),
        },
    }
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("\nEvaluation suite complete")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output dir: {out_dir}")
    print(f"  Sigma scale: {sigma_scale:.4f}")
    print(f"  Open RMSE h1/h20 ratio: {rmse_ratio_h20_h1:.4f}" if np.isfinite(rmse_ratio_h20_h1) else "  Open RMSE h1/h20 ratio: nan")
    print(f"  Coverage@95 mean: {coverage95:.4f}" if np.isfinite(coverage95) else "  Coverage@95 mean: nan")
    print(f"  Latent KL mean: {mean_test_kl:.4f}" if np.isfinite(mean_test_kl) else "  Latent KL mean: nan")
    print(
        f"  KL>free_nats ratio: {kl_above_free_ratio:.4f}"
        if np.isfinite(kl_above_free_ratio)
        else "  KL>free_nats ratio: nan"
    )
    print(
        f"  Latent traversal smoothness: {float(latent_traversal.get('smoothness_ratio', np.nan)):.4f}"
        if np.isfinite(float(latent_traversal.get("smoothness_ratio", np.nan)))
        else "  Latent traversal smoothness: nan"
    )
    print(
        f"  Recon corr mean: {float(reconstruction.get('corr_mean', np.nan)):.4f}"
        if np.isfinite(float(reconstruction.get("corr_mean", np.nan)))
        else "  Recon corr mean: nan"
    )


if __name__ == "__main__":
    main()
