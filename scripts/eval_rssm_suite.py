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


def _finite_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(arr[mask].mean())


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
    interval_levels = (0.5, 0.8, 0.9, 0.95)

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
    expected_sign_raw = inter_cfg.get("expected_direction_sign", None)
    expected_sign: Optional[float]
    if expected_sign_raw is None:
        expected_sign = None
    else:
        exp = float(expected_sign_raw)
        expected_sign = 1.0 if exp >= 0.0 else -1.0

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

    # Persist metrics.
    _save_horizon_csv(out_dir / "open_loop_horizon_metrics.csv", open_curves, interval_levels)
    _save_horizon_csv(out_dir / "closed_loop_horizon_metrics.csv", closed_curves, interval_levels)

    inter_df = pd.DataFrame({
        "horizon": np.arange(1, len(intervention["delta_abs"]) + 1),
        "delta_abs": intervention["delta_abs"],
        "delta_signed": intervention["delta_signed"],
        "direction_score_raw": direction_curve_raw,
        "direction_score_aligned": direction_curve_aligned,
    })
    inter_df.to_csv(out_dir / "interventional_metrics.csv", index=False)

    cal_df = _calibration_summary(open_curves, interval_levels)
    cal_df.to_csv(out_dir / "calibration_curve.csv", index=False)

    _plot_curve(open_curves.get("rmse", np.empty((0,))), "Open-loop RMSE vs Horizon", "RMSE", out_dir / "open_loop_rmse.png")
    _plot_curve(open_curves.get("crps", np.empty((0,))), "Open-loop CRPS vs Horizon", "CRPS", out_dir / "open_loop_crps.png")
    _plot_curve(open_curves.get("sharpness_90", np.empty((0,))), "Open-loop 90% Interval Width", "Width", out_dir / "open_loop_sharpness90.png")
    _plot_curve(kl_per_timestep, "KL per Timestep (observe)", "KL", out_dir / "latent_kl_per_timestep.png")

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
        "sigma_scale": float(sigma_scale),
        "coverage_95_mean": coverage95,
        "open_loop_rmse_h20_over_h1": rmse_ratio_h20_h1,
        "intervention_direction_mean_raw": direction_mean_raw,
        "intervention_direction_mean_aligned": direction_mean_aligned,
        "intervention_expected_sign": expected_sign,
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


if __name__ == "__main__":
    main()
