#!/usr/bin/env python3
"""Compare trained world models (no training -- evaluation and plotting only).

Loads saved checkpoints and scalers from ``<runs_dir>/<dataset>[/<run_name>]/``, runs
evaluation and recursive simulation on every available checkpoint,
and produces:

- Per-model outputs in ``<runs_dir>/<dataset>[/<run_name>]/<model>/``:
  ``<round>_forecast.png``, ``<round>_simulation.png/.csv``

- Cross-model comparisons in ``<runs_dir>/<dataset>[/<run_name>]/figures/``:
  ``comparison_forecast.png``, ``comparison_losses.png``,
  ``comparison_metrics.png``, ``simulation_trajectory.png``,
  ``comparison_results.csv``, ``simulation_metrics.csv``

Usage:
    python scripts/compare.py --config configs/wastewater.small.yaml
    python scripts/compare.py --config configs/wastewater.yaml --models lstm transformer
"""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch

from timesim.utils.config import load_config
from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.utils.plotting import save_loss_plot, save_forecast_plot
from timesim.models.factory import build_model, count_parameters, NEURAL_MODELS

# Try importing XGBoost
try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import matplotlib
matplotlib.use("Agg")

# Shared evaluation / simulation / plotting utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (
    evaluate_neural_model,
    evaluate_xgboost_model,
    simulate_recursive_neural,
    simulate_recursive_xgboost,
    save_per_model_simulation_plot,
    save_per_model_simulation_csv,
    save_comparison_forecast_plot,
    save_comparison_loss_plot,
    save_metrics_bar_chart,
    save_simulation_trajectory_plot,
    save_simulation_csv,
)


# ─────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────

def discover_checkpoints(model_dir: Path, model_type: str):
    """Find available checkpoints in a model directory.

    Returns list of (round_name, path) tuples.
    """
    found = []
    if model_type in NEURAL_MODELS:
        for p in sorted(model_dir.glob("*_checkpoint.pth")):
            round_name = p.stem.replace("_checkpoint", "")
            found.append((round_name, p))
    else:
        for p in sorted(model_dir.glob("*_model.pkl")):
            round_name = p.stem.replace("_model", "")
            found.append((round_name, p))
    return found


def load_loss_history(model_dir: Path):
    """Load cumulative train/val loss from metrics.csv if available."""
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        return [], []
    try:
        df = pd.read_csv(metrics_path)
        train_losses = df["train_loss"].tolist()
        val_losses = df["val_loss"].tolist()
        return train_losses, val_losses
    except Exception:
        return [], []


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trained models (evaluation only -- no training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (supports _base chain)")
    parser.add_argument("--models", nargs="*",
                        help="Override: compare only these model types")
    parser.add_argument("--device", type=str,
                        help="Override device (cpu / cuda)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config = load_config(args.config)
    if args.device:
        config["misc"]["device"] = args.device

    seed = config["misc"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config["misc"].get("device", "cpu")

    # ── Config sub-dicts ──────────────────────────────────────────────
    data_cfg = config.get("data", {})
    model_defaults_cfg = config.get("model_defaults", {})
    plot_cfg = config.get("plotting", {})
    output_cfg = config.get("output", {})
    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None

    # ── Model info ────────────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}

    if args.models:
        model_names = args.models
    elif models_cfg_list:
        model_names = [m["type"] for m in models_cfg_list]
    else:
        model_names = [config.get("model", {}).get("type", "lstm")]

    # ── Dimensions ────────────────────────────────────────────────────
    groups = config["dataset"]["variables"]
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    input_cols = sum((groups[g] for g in input_groups), [])
    output_cols = sum((groups[g] for g in output_groups), [])

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    # Use union to avoid double-counting when output_cols are in input_groups
    all_input_features = set(input_cols) | set(output_cols)
    input_dim = len(all_input_features)
    output_dim = len(output_cols)

    control_cols = groups.get("control", [])
    exo_cols_list = groups.get("exogenous", [])
    control_dim = len([c for c in input_cols if c in control_cols])
    exo_dim = len([c for c in input_cols if c in exo_cols_list])

    warmup_len = config["training"].get("warmup_len", seq_len)
    eval_cfg = config.get("evaluation", {}) or {}
    eval_horizon = eval_cfg.get("horizon", max(pred_len, 12))
    n_windows = eval_cfg.get("n_windows", 4)
    prob_eval_cfg = eval_cfg.get("probabilistic", {}) or {}

    sim_cfg = config.get("simulation", {})
    sim_start = sim_cfg.get("start_idx", 0)

    # ── Run directory ─────────────────────────────────────────────────
    dataset_name = config["dataset"]["name"]
    run_dir = runs_dir / dataset_name
    if run_name is not None:
        run_dir = run_dir / run_name
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        print("  Run scripts/train.py first.")
        return

    # ── Load scaler & dataset ─────────────────────────────────────────
    index_col = config["dataset"].get("index_col",
                                       data_cfg.get("index_col", "date"))
    train_split = config["dataset"].get("train_split",
                                         data_cfg.get("train_split", 0.8))

    scaler_path = run_dir / "scaler.pkl"
    if not scaler_path.exists():
        print(f"ERROR: scaler not found: {scaler_path}")
        print("  Run scripts/train.py first to save scaler.")
        return

    from joblib import load
    scaler = load(scaler_path)
    print(f"Loaded scaler from {scaler_path}")

    print("Loading dataset...")
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        slice_cfg=config["dataset"].get("slice"),
    )
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    train_loader, val_loader, _ = build_grouped_dataloaders(
        df, groups, input_groups, output_groups,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=config["dataset"]["batch_size"],
        train_split=train_split,
        existing_scaler=scaler,
    )

    val_dataset = val_loader.dataset
    print(f"  Val samples: {len(val_dataset)}")

    sim_horizon = sim_cfg.get("horizon", None)
    if sim_horizon is None:
        sim_horizon = len(val_dataset.values) - seq_len

    # ── Banner ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (evaluation & simulation only)")
    print("=" * 70)
    print(f"  Run directory  : {run_dir}")
    print(f"  Models         : {model_names}")
    print(f"  Eval horizon   : {eval_horizon}")
    print(f"  Sim horizon    : {sim_horizon}")
    print(f"  Device         : {device}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATE & SIMULATE EVERY CHECKPOINT
    # ══════════════════════════════════════════════════════════════════

    all_results = {}           # latest checkpoint per model → comparison
    sim_results_latest = {}

    for model_type in model_names:
        mc = models_cfg_map.get(model_type, {"type": model_type})
        model_dir = run_dir / model_type

        if not model_dir.exists():
            print(f"\n  {model_type}: directory not found, skipping")
            continue

        checkpoints = discover_checkpoints(model_dir, model_type)
        if not checkpoints:
            print(f"\n  {model_type}: no checkpoints found, skipping")
            continue

        n_params = 0
        train_losses, val_losses = load_loss_history(model_dir)

        for round_name, ckpt_path in checkpoints:
            # ── Load model ────────────────────────────────────────
            if model_type in NEURAL_MODELS:
                model = build_model(
                    model_type, input_dim, output_dim, seq_len, pred_len,
                    per_model_cfg=mc, model_defaults_cfg=model_defaults_cfg)
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True))
                model.to(device)
                model.eval()
                n_params = count_parameters(model)
            else:
                model = XGBoostForecaster.load(str(ckpt_path))

            print(f"\n{'─' * 70}")
            print(f"  {model_type.upper()} [{round_name}]")
            print(f"{'─' * 70}")

            # ── Evaluate ──────────────────────────────────────────
            try:
                if model_type in NEURAL_MODELS:
                    gt_list, pred_list, eval_info = evaluate_neural_model(
                        model, val_dataset, warmup_len, eval_horizon,
                        control_dim, exo_dim, device, n_windows,
                        probabilistic_cfg=prob_eval_cfg,
                        return_info=True,
                    )
                else:
                    gt_list, pred_list = evaluate_xgboost_model(
                        model, val_dataset, seq_len, eval_horizon, n_windows,
                    )
                    eval_info = {
                        "is_probabilistic": False,
                        "rollout_nll": float("nan"),
                        "coverage_90": float("nan"),
                        "interval_width_90": float("nan"),
                    }

                if gt_list and pred_list:
                    mean_mse = float(np.mean(
                        [np.mean((g - p) ** 2) for g, p in zip(gt_list, pred_list)]))
                    mean_mae = float(np.mean(
                        [np.mean(np.abs(g - p)) for g, p in zip(gt_list, pred_list)]))
                else:
                    mean_mse = mean_mae = float("nan")

                print(f"    Eval  MSE={mean_mse:.6f}  MAE={mean_mae:.6f}")
                if bool(eval_info.get("is_probabilistic", False)):
                    print(
                        "    Eval  "
                        f"NLL={eval_info.get('rollout_nll', float('nan')):.6f}  "
                        f"Coverage@90={eval_info.get('coverage_90', float('nan')):.6f}  "
                        f"Width@90={eval_info.get('interval_width_90', float('nan')):.6f}"
                    )

                if gt_list and pred_list:
                    save_forecast_plot(
                        gt_list[0], pred_list[0], output_cols,
                        model_dir / f"{round_name}_forecast.png",
                        title=f"{model_type.upper()} [{round_name}] "
                              f"(horizon={eval_horizon})",
                        show_metrics=True,
                    )
            except Exception as exc:
                print(f"    Eval ERROR: {exc}")
                gt_list, pred_list = [], []
                mean_mse = mean_mae = float("nan")
                eval_info = {
                    "is_probabilistic": False,
                    "rollout_nll": float("nan"),
                    "coverage_90": float("nan"),
                    "interval_width_90": float("nan"),
                }

            # ── Simulate ──────────────────────────────────────────
            sim_result = None
            try:
                if model_type in NEURAL_MODELS:
                    sim_result = simulate_recursive_neural(
                        model, val_dataset, seq_len, sim_horizon,
                        device, start_idx=sim_start,
                        probabilistic_cfg=prob_eval_cfg,
                    )
                else:
                    sim_result = simulate_recursive_xgboost(
                        model, val_dataset, seq_len, sim_horizon,
                        start_idx=sim_start,
                    )

                n = sim_result["n_steps"]
                if n > 0:
                    gt_s = sim_result["ground_truths"]
                    pr_s = sim_result["predictions"]
                    sim_mse = float(np.mean((gt_s - pr_s) ** 2))
                    sim_mae = float(np.mean(np.abs(gt_s - pr_s)))
                    print(f"    Sim   MSE={sim_mse:.6f}  MAE={sim_mae:.6f}  "
                          f"({n} steps)")

                    save_per_model_simulation_plot(
                        sim_result, output_cols, model_type,
                        model_dir / f"{round_name}_simulation.png",
                        plot_cfg=plot_cfg,
                    )
                    save_per_model_simulation_csv(
                        sim_result, output_cols,
                        model_dir, round_name,
                    )
                else:
                    sim_result = None
            except Exception as exc:
                print(f"    Sim ERROR: {exc}")
                sim_result = None

            # ── Store latest for comparison (overwritten each round) ──
            all_results[model_type] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "gt_list": gt_list,
                "pred_list": pred_list,
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "rollout_nll": float(eval_info.get("rollout_nll", float("nan"))),
                "coverage_90": float(eval_info.get("coverage_90", float("nan"))),
                "interval_width_90": float(eval_info.get("interval_width_90", float("nan"))),
                "n_params": n_params,
                "last_round_name": round_name,
                "total_epochs": len(train_losses),
            }
            if sim_result and sim_result["n_steps"] > 0:
                sim_results_latest[model_type] = sim_result

    # ══════════════════════════════════════════════════════════════════
    #  COMPARATIVE OUTPUTS  →  figures/
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  GENERATING COMPARATIVE RESULTS")
    print(f"{'=' * 70}")

    if len(all_results) >= 2:
        save_comparison_forecast_plot(
            all_results, output_cols,
            figs_dir / "comparison_forecast.png", eval_horizon,
            plot_cfg=plot_cfg)
        save_comparison_loss_plot(
            all_results, figs_dir / "comparison_losses.png",
            plot_cfg=plot_cfg)
        save_metrics_bar_chart(
            all_results, figs_dir / "comparison_metrics.png",
            plot_cfg=plot_cfg)
        print("  Saved comparison plots → figures/")

    rows = []
    for mn, res in all_results.items():
        rows.append({
            "model": mn,
            "last_round": res.get("last_round_name", "train"),
            "total_epochs": res.get("total_epochs", 0),
            "parameters": res["n_params"],
            "final_train_loss": (res["train_losses"][-1]
                                 if res["train_losses"] else None),
            "final_val_loss": (res["val_losses"][-1]
                               if res["val_losses"] else None),
            "rollout_mse": res["mean_mse"],
            "rollout_mae": res["mean_mae"],
            "rollout_nll": res.get("rollout_nll"),
            "coverage_90": res.get("coverage_90"),
            "interval_width_90": res.get("interval_width_90"),
        })
    results_df = pd.DataFrame(rows)
    csv_path = figs_dir / "comparison_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(results_df.to_string(index=False))
    print(f"\n  Figures : {figs_dir}")
    print(f"  CSV     : {csv_path}")
    print(f"{'=' * 70}")

    # ── Simulation comparison ─────────────────────────────────────────
    if sim_results_latest:
        save_simulation_trajectory_plot(
            sim_results_latest, output_cols,
            figs_dir / "simulation_trajectory.png",
            plot_cfg=plot_cfg,
        )
        save_simulation_csv(
            sim_results_latest, output_cols,
            figs_dir / "simulation_results.csv",
        )

        sim_rows = []
        for mn, sr in sim_results_latest.items():
            if sr["n_steps"] > 0:
                gt_s = sr["ground_truths"]
                pr_s = sr["predictions"]
                sim_rows.append({
                    "model": mn,
                    "last_round": all_results[mn].get("last_round_name", "train"),
                    "sim_steps": sr["n_steps"],
                    "sim_mse": float(np.mean((gt_s - pr_s) ** 2)),
                    "sim_mae": float(np.mean(np.abs(gt_s - pr_s))),
                })
        if sim_rows:
            sim_df = pd.DataFrame(sim_rows)
            sim_csv_path = figs_dir / "simulation_metrics.csv"
            sim_df.to_csv(sim_csv_path, index=False)
            print(f"\n{'=' * 70}")
            print("  SIMULATION SUMMARY")
            print(f"{'=' * 70}")
            print(sim_df.to_string(index=False))
            print(f"\n  Simulation plot : {figs_dir / 'simulation_trajectory.png'}")
            print(f"  Sim metrics     : {sim_csv_path}")
            print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
