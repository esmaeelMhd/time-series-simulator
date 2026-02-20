#!/usr/bin/env python3
"""Streamlit monitoring dashboard for RSSM training/evaluation quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from joblib import load as joblib_load

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.loader import (
    chronological_split_dataframe,
    load_csv_dataset,
    resolve_split_ratios,
)
from timesim.data.schema import VariableSchema
from timesim.data.stamps import get_time_feature_columns
from timesim.models.factory import build_model
from timesim.utils.config import load_config
from timesim.utils.misc import resolve_device


st.set_page_config(page_title="TimeSim Monitoring", layout="wide")
st.title("TimeSim RSSM Monitoring View")

st.markdown(
    """
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
  h1, h2, h3 {letter-spacing: 0.02em;}
  .row-label {font-weight: 700; color: #0f172a; margin-top: 0.8rem;}
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class RunArtifacts:
    run_dir: Path
    model_dir: Path
    metrics_path: Path
    eval_dir: Optional[Path]
    config_path: Optional[Path]
    scaler_path: Optional[Path]
    checkpoint_path: Optional[Path]


def _discover_runs(runs_root: Path) -> list[RunArtifacts]:
    out: list[RunArtifacts] = []
    for metrics in sorted(runs_root.glob("**/latent_ssm/metrics.csv")):
        model_dir = metrics.parent
        run_dir = model_dir.parent
        eval_dir = model_dir / "eval_suite"
        config_path = run_dir / "config.yaml"
        scaler_path = run_dir / "scaler.pkl"
        checkpoint = model_dir / "train_checkpoint.pth"
        if not checkpoint.exists():
            cands = sorted(model_dir.glob("*_checkpoint.pth"))
            checkpoint = cands[-1] if cands else None
        out.append(
            RunArtifacts(
                run_dir=run_dir,
                model_dir=model_dir,
                metrics_path=metrics,
                eval_dir=eval_dir if eval_dir.exists() else None,
                config_path=config_path if config_path.exists() else None,
                scaler_path=scaler_path if scaler_path.exists() else None,
                checkpoint_path=checkpoint if isinstance(checkpoint, Path) else None,
            )
        )
    return out


def _split_metric_segments(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns or df.empty:
        df = df.copy()
        df["segment"] = 0
        return df
    seg_break = df["epoch"].diff().fillna(1).le(0)
    out = df.copy()
    out["segment"] = seg_break.cumsum().astype(int)
    return out


def _resolve_eval_csv(eval_dir: Optional[Path], name: str) -> Optional[Path]:
    if eval_dir is None:
        return None
    p = eval_dir / name
    return p if p.exists() else None


def _load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


_RICH_METRICS_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "loss_std",
    "loss_total",
    "recon_nll",
    "kl",
    "kl_raw",
    "aux_nll",
    "rollout_nll",
    "rollout_dtw",
    "rollout_total",
    "rollout_weight_eff",
    "rollout_ramp",
    "horizon_schedule",
    "context_len",
    "grad_norm",
    "grad_norm_pre",
    "lr",
]


def _load_metrics_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Load metrics.csv robustly, including mixed-schema files from legacy runs."""
    try:
        return pd.read_csv(path), ""
    except Exception:
        pass

    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except Exception as exc:
        raise ValueError(f"Could not read metrics text: {exc}") from exc
    if not lines:
        return pd.DataFrame(columns=["epoch"]), "metrics.csv is empty."

    header = [x.strip() for x in lines[0].split(",")]
    data = lines[1:]
    if not data:
        return pd.DataFrame(columns=header), "metrics.csv has header only."

    split_rows = [ln.split(",") for ln in data]
    counts = [len(r) for r in split_rows]
    rich_n = len(_RICH_METRICS_COLUMNS)
    header_n = len(header)
    count_hist = Counter(counts)

    if rich_n in count_hist:
        target_n = rich_n
        columns = list(_RICH_METRICS_COLUMNS)
    elif header_n in count_hist:
        target_n = header_n
        columns = header
    else:
        target_n = count_hist.most_common(1)[0][0]
        columns = [f"col_{i+1}" for i in range(target_n)]

    kept_rows = [r[:target_n] for r in split_rows if len(r) == target_n]
    dropped = len(split_rows) - len(kept_rows)
    df = pd.DataFrame(kept_rows, columns=columns)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    msg = ""
    if dropped > 0:
        msg = (
            f"Loaded mixed metrics schema from `{path.name}`: kept {len(kept_rows)} rows with "
            f"{target_n} columns, dropped {dropped} incompatible rows."
        )
    return df, msg


def _line_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "",
    *,
    color: str = "#0f766e",
):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(x, y, color=color, linewidth=1.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _heatmap_plot(arr: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5.3, 3.0))
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _find_checkpoint(model_dir: Path) -> Optional[Path]:
    p = model_dir / "train_checkpoint.pth"
    if p.exists():
        return p
    cands = sorted(model_dir.glob("*_checkpoint.pth"))
    if cands:
        return cands[-1]
    cands = sorted((model_dir / "checkpoints").glob("*.pth"))
    if cands:
        return cands[-1]
    return None


def _load_model_state(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)


def _build_test_dataset(cfg: Dict[str, Any], scaler) -> GroupedTimeSeriesDataset:
    dcfg = cfg["dataset"]
    data_cfg = cfg.get("data", {})
    groups = dcfg["variables"]
    input_groups = cfg["model_io"]["input_groups"]
    output_groups = cfg["model_io"]["output_groups"]
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
    eval_cfg = cfg.get("evaluation", {}) or {}
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
    else:
        frac = float(test_split)
        n_test = max(seq_len + pred_len + 1, int(round(len(df) * frac)))
        start = max(0, len(df) - n_test - seq_len)
        test_df = df.iloc[start:].copy()

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


def _compute_latent_diagnostics(
    cfg_path: Path,
    scaler_path: Path,
    checkpoint_path: Path,
    *,
    n_windows: int,
    device: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    try:
        cfg = load_config(str(cfg_path))
        dcfg = cfg["dataset"]
        groups = dcfg["variables"]
        schema = VariableSchema.from_groups(groups)
        input_groups = cfg["model_io"]["input_groups"]
        output_groups = cfg["model_io"]["output_groups"]
        input_cols = schema.columns_for_group_names(input_groups)
        output_cols = schema.columns_for_group_names(output_groups)
        input_dim = len(set(input_cols) | set(output_cols))
        add_time = bool(cfg.get("data", {}).get("add_time_features", False))
        tf_cfg = cfg.get("data", {}).get("time_features", {}) or {}
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

        model_cfg_map = {m["type"]: m for m in cfg.get("models", [])}
        per_model_cfg = model_cfg_map.get("latent_ssm", {"type": "latent_ssm"})
        model = build_model(
            "latent_ssm",
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=int(dcfg["seq_len"]),
            pred_len=int(dcfg["pred_len"]),
            per_model_cfg=per_model_cfg,
            model_defaults_cfg=cfg.get("model_defaults", {}),
        )
        dev = resolve_device(device)
        _load_model_state(model, checkpoint_path, dev)
        model.to(dev)
        model.eval()

        scaler = joblib_load(scaler_path)
        dataset = _build_test_dataset(cfg, scaler=scaler)
        warmup_len = int(cfg.get("training", {}).get("warmup_len", int(dcfg["seq_len"])))
        warmup_len = max(1, min(warmup_len, len(dataset.values)))
        max_start = len(dataset.values) - warmup_len
        if max_start < 0:
            return None, None, "Not enough test data for latent diagnostics."
        starts = np.linspace(0, max_start, num=max(1, int(n_windows)), dtype=int).tolist()

        kl_windows: list[torch.Tensor] = []
        overlap_windows: list[torch.Tensor] = []
        cpos = list(getattr(dataset, "control_positions", []))
        xpos = list(getattr(dataset, "known_exo_positions", []))

        with torch.no_grad():
            for s in starts:
                warm = dataset.values[s : s + warmup_len]
                warm_inputs = warm[:, dataset.in_idx]
                history_y = warm[:, dataset.out_idx].astype(np.float32, copy=False)
                history_c = (
                    warm_inputs[:, cpos]
                    if cpos
                    else np.zeros((warm_inputs.shape[0], 0), dtype=np.float32)
                )
                history_x = (
                    warm_inputs[:, xpos]
                    if xpos
                    else np.zeros((warm_inputs.shape[0], 0), dtype=np.float32)
                )

                obs = model.observe(
                    controls=torch.from_numpy(history_c).unsqueeze(0).to(dev),
                    exogenous=torch.from_numpy(history_x).unsqueeze(0).to(dev),
                    observations=torch.from_numpy(history_y).unsqueeze(0).to(dev),
                    sample_posterior=False,
                )
                prior_mu = obs["prior_mu"]
                prior_std = torch.exp(0.5 * obs["prior_logvar"]).clamp_min(1e-6)
                post_mu = obs["posterior_mu"]
                post_std = torch.exp(0.5 * obs["posterior_logvar"]).clamp_min(1e-6)

                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(post_mu, post_std),
                    torch.distributions.Normal(prior_mu, prior_std),
                ).squeeze(0)  # (T, Z)
                kl_windows.append(kl)

                denom = prior_std.pow(2) + post_std.pow(2) + 1e-8
                overlap = torch.sqrt((2.0 * prior_std * post_std) / denom) * torch.exp(
                    -((prior_mu - post_mu).pow(2)) / (4.0 * denom)
                )
                overlap_windows.append(overlap.squeeze(0))  # (T, Z)

        if not kl_windows:
            return None, None, "No latent windows were computed."

        kl_heat = torch.stack(kl_windows, dim=0).mean(dim=0).cpu().numpy().astype(np.float32)
        overlap_curve = (
            torch.stack(overlap_windows, dim=0).mean(dim=(0, 2)).cpu().numpy().astype(np.float32)
        )
        return kl_heat, overlap_curve, ""
    except Exception as exc:
        return None, None, f"Latent diagnostics unavailable: {exc}"


@st.cache_data(show_spinner=False)
def _compute_latent_diagnostics_cached(
    cfg_path_str: str,
    scaler_path_str: str,
    checkpoint_path_str: str,
    cfg_mtime: float,
    scaler_mtime: float,
    checkpoint_mtime: float,
    n_windows: int,
    device: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    # mtime args are part of the cache key so cache invalidates when artifacts change.
    _ = (cfg_mtime, scaler_mtime, checkpoint_mtime)
    return _compute_latent_diagnostics(
        Path(cfg_path_str),
        Path(scaler_path_str),
        Path(checkpoint_path_str),
        n_windows=n_windows,
        device=device,
    )


st.sidebar.header("Run Selection")
runs_root = st.sidebar.text_input("Runs root", value="runs")
runs_root_path = Path(runs_root)
artifacts = _discover_runs(runs_root_path) if runs_root_path.exists() else []
if not artifacts:
    st.error(f"No latent_ssm metrics found under '{runs_root_path}'.")
    st.stop()

labels = [str(a.run_dir) for a in artifacts]
idx_default = max(0, len(labels) - 1)
selected_label = st.sidebar.selectbox("Run", labels, index=idx_default)
selected = artifacts[labels.index(selected_label)]

st.sidebar.write(f"Metrics: `{selected.metrics_path}`")
if selected.eval_dir is not None:
    st.sidebar.write(f"Eval: `{selected.eval_dir}`")
if selected.config_path is not None:
    st.sidebar.write(f"Config: `{selected.config_path}`")
if selected.checkpoint_path is not None:
    st.sidebar.write(f"Checkpoint: `{selected.checkpoint_path}`")

try:
    metrics_all, metrics_read_note = _load_metrics_csv(selected.metrics_path)
except Exception as exc:
    st.error(f"Could not read metrics file: {selected.metrics_path} ({exc})")
    st.stop()
if metrics_read_note:
    st.sidebar.info(metrics_read_note)

metrics_all = _split_metric_segments(metrics_all)
segments = sorted(metrics_all["segment"].unique().tolist()) if "segment" in metrics_all.columns else []
if not segments:
    st.warning("Metrics file has no rows yet. Showing eval-only panels where available.")
    segment = 0
    metrics = pd.DataFrame(columns=["epoch"])
    epochs = np.array([], dtype=np.float32)
else:
    seg_default = segments[-1]
    segment = st.sidebar.selectbox("Training segment", segments, index=segments.index(seg_default))
    metrics = metrics_all[metrics_all["segment"] == segment].copy()
    metrics = metrics.sort_values("epoch").reset_index(drop=True)
    epochs = (
        metrics["epoch"].to_numpy(dtype=np.float32)
        if "epoch" in metrics.columns
        else np.arange(len(metrics))
    )

st.caption(
    f"Showing segment {segment} with {len(metrics)} epoch rows from `{selected.run_dir}`."
)

# Row 1: Training curves
st.markdown('<div class="row-label">Row 1: Training curves</div>', unsafe_allow_html=True)
r1 = st.columns(5)
curve_specs = [
    ("loss_total", "Loss total", "train_loss"),
    ("recon_nll", "Recon NLL", None),
    ("kl", "KL divergence", None),
    ("aux_nll", "Aux NLL", None),
    ("rollout_nll", "Rollout NLL", None),
]
for col, title, fallback in curve_specs:
    with r1[curve_specs.index((col, title, fallback))]:
        use_col = col if col in metrics.columns else fallback
        if use_col is None or use_col not in metrics.columns:
            st.info(f"{title}: not logged")
            continue
        y = metrics[use_col].astype(float).to_numpy()
        _line_plot(epochs, y, title=title, ylabel=use_col)

# Row 2: Training health
st.markdown('<div class="row-label">Row 2: Training health</div>', unsafe_allow_html=True)
r2 = st.columns(4)
health_specs = [
    ("grad_norm", "Gradient norm", "grad_norm_pre"),
    ("lr", "Learning rate", None),
    ("horizon_schedule", "Current horizon", None),
    ("rollout_ramp", "Rollout ramp", None),
]
for col, title, fallback in health_specs:
    with r2[health_specs.index((col, title, fallback))]:
        use_col = col if col in metrics.columns else fallback
        if use_col is None or use_col not in metrics.columns:
            st.info(f"{title}: not logged")
            continue
        y = metrics[use_col].astype(float).to_numpy()
        _line_plot(epochs, y, title=title, ylabel=use_col, color="#1d4ed8")

# Row 3: KL diagnostics
st.markdown('<div class="row-label">Row 3: KL diagnostics</div>', unsafe_allow_html=True)
r3 = st.columns(3)
with r3[0]:
    if "kl_raw" in metrics.columns:
        _line_plot(
            epochs,
            metrics["kl_raw"].astype(float).to_numpy(),
            title="KL raw (before free bits)",
            ylabel="kl_raw",
            color="#b45309",
        )
    else:
        st.info("KL raw: not logged")

diag_windows = st.sidebar.slider("Latent diagnostic windows", min_value=2, max_value=32, value=8, step=1)
diag_device = st.sidebar.selectbox("Latent diagnostic device", ["cpu", "cuda", "auto"], index=0)
if diag_device == "cuda" and not torch.cuda.is_available():
    st.sidebar.warning("CUDA is not available; using CPU for latent diagnostics.")
    diag_device = "cpu"
kl_heat: Optional[np.ndarray] = None
overlap_curve: Optional[np.ndarray] = None
diag_msg = ""
if selected.config_path and selected.scaler_path and selected.checkpoint_path:
    with st.spinner("Computing KL heatmap + prior/posterior overlap..."):
        kl_heat, overlap_curve, diag_msg = _compute_latent_diagnostics_cached(
            str(selected.config_path),
            str(selected.scaler_path),
            str(selected.checkpoint_path),
            selected.config_path.stat().st_mtime,
            selected.scaler_path.stat().st_mtime,
            selected.checkpoint_path.stat().st_mtime,
            n_windows=int(diag_windows),
            device=diag_device,
        )
else:
    diag_msg = "Config, scaler, or checkpoint missing for latent diagnostics."

with r3[1]:
    if kl_heat is not None:
        _heatmap_plot(
            kl_heat,
            title="KL per latent dim heatmap",
            xlabel="Latent dimension",
            ylabel="Time step",
        )
    else:
        st.info(diag_msg)

with r3[2]:
    if overlap_curve is not None:
        x = np.arange(1, overlap_curve.shape[0] + 1)
        _line_plot(
            x,
            overlap_curve,
            title="Prior-Posterior overlap",
            xlabel="Time step",
            ylabel="overlap [0,1]",
            color="#15803d",
        )
        st.metric("Mean overlap", f"{float(np.nanmean(overlap_curve)):.4f}")
    else:
        st.info(diag_msg)

# Load eval artifacts
open_loop_df = _load_csv(_resolve_eval_csv(selected.eval_dir, "open_loop_horizon_metrics.csv"))
calib_df = _load_csv(_resolve_eval_csv(selected.eval_dir, "calibration_curve.csv"))
interv_df = _load_csv(_resolve_eval_csv(selected.eval_dir, "interventional_metrics.csv"))

# Row 4: Validation forecasting
st.markdown('<div class="row-label">Row 4: Validation forecasting</div>', unsafe_allow_html=True)
r4 = st.columns(3)
with r4[0]:
    if open_loop_df is not None and {"horizon", "rmse"}.issubset(open_loop_df.columns):
        _line_plot(
            open_loop_df["horizon"].to_numpy(dtype=np.float32),
            open_loop_df["rmse"].to_numpy(dtype=np.float32),
            title="RMSE vs horizon",
            xlabel="Horizon",
            ylabel="RMSE",
            color="#334155",
        )
    else:
        st.info("open_loop_horizon_metrics.csv missing")

with r4[1]:
    if open_loop_df is not None and {"horizon", "crps"}.issubset(open_loop_df.columns):
        _line_plot(
            open_loop_df["horizon"].to_numpy(dtype=np.float32),
            open_loop_df["crps"].to_numpy(dtype=np.float32),
            title="CRPS vs horizon",
            xlabel="Horizon",
            ylabel="CRPS",
            color="#0f766e",
        )
    else:
        st.info("CRPS not available")

with r4[2]:
    if calib_df is not None and {"nominal", "actual"}.issubset(calib_df.columns):
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        nominal = calib_df["nominal"].to_numpy(dtype=np.float32)
        actual = calib_df["actual"].to_numpy(dtype=np.float32)
        ax.plot(nominal, actual, marker="o", linewidth=1.8, color="#7c3aed", label="Observed")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#64748b", label="Ideal")
        ax.set_title("Calibration plot", fontsize=11)
        ax.set_xlabel("Nominal coverage", fontsize=9)
        ax.set_ylabel("Actual coverage", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("calibration_curve.csv missing")

# Row 5: Simulator quality
st.markdown('<div class="row-label">Row 5: Simulator quality</div>', unsafe_allow_html=True)
r5 = st.columns(2)
with r5[0]:
    if interv_df is not None and {"horizon", "delta_signed", "delta_abs"}.issubset(interv_df.columns):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        h = interv_df["horizon"].to_numpy(dtype=np.float32)
        ax.plot(h, interv_df["delta_signed"].to_numpy(dtype=np.float32), label="delta_signed", color="#0369a1")
        ax.plot(h, interv_df["delta_abs"].to_numpy(dtype=np.float32), label="delta_abs", color="#c2410c")
        ax.set_title("Control sweep -> Y response curves", fontsize=11)
        ax.set_xlabel("Horizon", fontsize=9)
        ax.set_ylabel("Response delta", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("interventional_metrics.csv missing")

with r5[1]:
    if open_loop_df is not None and {"horizon", "sharpness_90"}.issubset(open_loop_df.columns):
        _line_plot(
            open_loop_df["horizon"].to_numpy(dtype=np.float32),
            open_loop_df["sharpness_90"].to_numpy(dtype=np.float32),
            title="Uncertainty vs horizon",
            xlabel="Horizon",
            ylabel="sharpness_90",
            color="#7e22ce",
        )
    else:
        st.info("sharpness_90 not available")

with st.expander("Raw artifacts"):
    st.write("Run dir:", selected.run_dir)
    st.write("Model dir:", selected.model_dir)
    st.write("Metrics:", selected.metrics_path)
    st.write("Eval dir:", selected.eval_dir)
    st.write("Config:", selected.config_path)
    st.write("Checkpoint:", selected.checkpoint_path)
    st.write("Scaler:", selected.scaler_path)
    summary_path = selected.eval_dir / "summary.yaml" if selected.eval_dir else None
    if summary_path is not None and summary_path.exists():
        try:
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
            st.json(summary)
        except Exception:
            st.info("Could not parse summary.yaml")
