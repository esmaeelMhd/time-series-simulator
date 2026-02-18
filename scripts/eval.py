#!/usr/bin/env python3
"""Evaluate a single trained model without retraining.

Loads a saved checkpoint from ``<runs_dir>/<dataset>/<model>/`` and runs
evaluation (multi-window rollout) and/or recursive simulation using the
current config settings.  This lets you re-evaluate a trained model with
different horizons, simulation lengths, number of windows, etc. without
touching the training at all.

Outputs are saved **next to the checkpoint** in the model directory,
with a configurable prefix (default ``eval_``) so they don't overwrite
the round-specific artifacts produced during training.

Examples:

    # Evaluate the latest LSTM checkpoint with default config
    python scripts/eval.py --config configs/wastewater.small.yaml --model lstm

    # Evaluate with a custom simulation horizon
    python scripts/eval.py --config configs/wastewater.small.yaml --model lstm \\
        --sim-horizon 500

    # Evaluate a specific checkpoint round (e.g. after initial training)
    python scripts/eval.py --config configs/wastewater.small.yaml --model lstm \\
        --round train

    # Change evaluation horizon and number of windows
    python scripts/eval.py --config configs/wastewater.small.yaml --model transformer \\
        --eval-horizon 48 --n-windows 16

    # Skip simulation and only produce forecast plots
    python scripts/eval.py --config configs/wastewater.small.yaml --model tft \\
        --no-sim

    # Skip evaluation and only produce simulation
    python scripts/eval.py --config configs/wastewater.small.yaml --model tft \\
        --no-eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from timesim.utils.config import load_config
from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.data.stamps import get_time_feature_columns
from timesim.utils.plotting import save_forecast_plot
from timesim.models.factory import build_model, count_parameters, NEURAL_MODELS

try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import matplotlib
matplotlib.use("Agg")

# Shared eval / simulation utilities (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (
    evaluate_neural_model,
    evaluate_xgboost_model,
    simulate_recursive_neural,
    simulate_recursive_xgboost,
    save_per_model_simulation_plot,
    save_per_model_simulation_csv,
)


# ─────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────

def discover_checkpoints(model_dir: Path, model_type: str):
    """Return list of (round_name, path) tuples."""
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


MODEL_PARAM_KEYS_BY_TYPE = {
    "lstm": {"hidden_dim", "num_layers", "dropout"},
    "dlinear": {"kernel_size", "individual"},
    "nlinear": {"individual"},
    "tft": {"hidden_dim", "n_heads", "num_lstm_layers", "dropout"},
    "transformer": {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout"},
    "latent_ssm": {
        "hidden_dim", "latent_dim", "num_layers", "dropout",
        "min_scale", "min_df", "encoder_dim", "decoder_layers", "use_symlog",
        "use_aux_decoder", "use_dual_path", "leak_objective_to_transition",
    },
    "xgboost": {"strategy", "n_estimators", "max_depth", "learning_rate"},
}


def _resolve_optuna_summary_path(config, run_dir: Path, cli_path: str | None = None) -> Path:
    if cli_path:
        return Path(cli_path)
    cfg_path = config.get("training", {}).get("optuna_summary_path", None)
    if cfg_path:
        return Path(cfg_path)
    return run_dir / "optuna" / "summary.yaml"


def _load_optuna_best_params(summary_path: Path):
    if not summary_path.exists():
        print(f"  Optuna summary not found: {summary_path}")
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        print(f"  Warning: Optuna summary is not a mapping: {summary_path}")
        return {}
    best_by_model = {}
    for model_type, entry in raw.items():
        if isinstance(entry, dict):
            best_params = entry.get("best_params", {})
            if isinstance(best_params, dict):
                best_by_model[str(model_type)] = dict(best_params)
    return best_by_model


def _split_optuna_params(model_type: str, best_params):
    model_keys = MODEL_PARAM_KEYS_BY_TYPE.get(model_type, set())
    model_overrides = {k: v for k, v in best_params.items() if k in model_keys}
    training_overrides = {
        k: v
        for k, v in best_params.items()
        if k in {
            "learning_rate",
            "optimizer",
            "weight_decay",
            "loss_type",
            "loss_weighting",
            "loss_weight_scale",
            "mode",
            "feedback",
            "teacher_forcing_ratio",
            "one_step_weight",
            "recon_weight",
            "elbo_weight",
            "kl_weight",
            "aux_weight",
            "rollout_mse_weight",
            "rollout_weight",
            "rollout_dtw_weight",
            "rollout_dtw_gamma",
            "rollout_warmup_fraction",
            "rollout_max_horizon",
            "min_context",
            "kl_free_bits",
            "kl_balance",
            "use_kl_balancing",
            "use_free_bits",
            "use_symlog",
            "grad_clip_norm",
            "lr_warmup_steps",
            "lr_min_ratio",
            "checkpoint_top_k",
            "early_stopping_monitor",
            "objective",
            "kl_warmup_enabled",
            "kl_beta_start",
            "kl_beta_end",
            "kl_warmup_epochs",
            "checkpoint_metric",
            "checkpoint_open_loop_horizon",
            "checkpoint_open_loop_windows",
            "checkpoint_open_loop_samples",
        }
    }
    return model_overrides, training_overrides


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model (no training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (supports _base chain)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model type to evaluate (e.g. lstm, transformer, xgboost)")
    parser.add_argument("--round", type=str, default=None,
                        help="Specific round to load (e.g. 'train', 'retrain'). "
                             "Default: latest available checkpoint.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu / cuda)")
    parser.add_argument(
        "--use-optuna-best-params",
        dest="use_optuna_best_params",
        action="store_true",
        help="Use per-model best params from Optuna summary if available",
    )
    parser.add_argument(
        "--no-use-optuna-best-params",
        dest="use_optuna_best_params",
        action="store_false",
        help="Disable Optuna best params even if enabled in config",
    )
    parser.set_defaults(use_optuna_best_params=None)
    parser.add_argument(
        "--optuna-summary",
        type=str,
        default=None,
        help="Path to Optuna summary.yaml (default: <run_dir>/optuna/summary.yaml)",
    )

    # ── Evaluation overrides ──────────────────────────────────────────
    eval_grp = parser.add_argument_group("evaluation overrides")
    eval_grp.add_argument("--eval-horizon", type=int, default=None,
                          help="Override evaluation rollout horizon")
    eval_grp.add_argument("--n-windows", type=int, default=None,
                          help="Override number of validation windows")
    eval_grp.add_argument("--no-eval", action="store_true",
                          help="Skip evaluation (multi-window rollout)")

    # ── Simulation overrides ──────────────────────────────────────────
    sim_grp = parser.add_argument_group("simulation overrides")
    sim_grp.add_argument("--sim-horizon", type=int, default=None,
                         help="Override simulation horizon (number of recursive steps)")
    sim_grp.add_argument("--sim-start", type=int, default=None,
                         help="Override simulation start index in validation data")
    sim_grp.add_argument("--no-sim", action="store_true",
                         help="Skip recursive simulation")

    # ── Output ────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument("--prefix", type=str, default="eval",
                         help="Prefix for output file names (default: 'eval')")

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
    if args.use_optuna_best_params is not None:
        config.setdefault("training", {})
        config["training"]["use_optuna_best_params"] = args.use_optuna_best_params
    if args.optuna_summary:
        config.setdefault("training", {})
        config["training"]["optuna_summary_path"] = args.optuna_summary

    seed = config["misc"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config["misc"].get("device", "cpu")

    model_type = args.model.lower()

    # ── Config sub-dicts ──────────────────────────────────────────────
    data_cfg = config.get("data", {})
    model_defaults_cfg = config.get("model_defaults", {})
    plot_cfg = config.get("plotting", {})
    output_cfg = config.get("output", {})
    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None

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
    add_time_features = bool(data_cfg.get("add_time_features", False))
    time_features_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(time_features_cfg, dict) and "enabled" in time_features_cfg:
        add_time_features = bool(time_features_cfg.get("enabled")) or add_time_features
    if add_time_features:
        input_dim += len(
            get_time_feature_columns(
                features=time_features_cfg.get("features"),
                encoding=time_features_cfg.get("encoding", "cyclical"),
            )
        )
    output_dim = len(output_cols)

    control_cols = groups.get("control", [])
    exo_cols_list = groups.get("exogenous", [])
    control_dim = len([c for c in input_cols if c in control_cols])
    exo_dim = len([c for c in input_cols if c in exo_cols_list])

    warmup_len = config["training"].get("warmup_len", seq_len)

    # ── Resolve eval/sim parameters (CLI overrides > config) ──────────
    eval_horizon = (args.eval_horizon
                    or config.get("evaluation", {}).get("horizon", max(pred_len, 12)))
    n_windows = (args.n_windows
                 or config.get("evaluation", {}).get("n_windows", 4))

    sim_cfg = config.get("simulation", {})
    sim_start = args.sim_start if args.sim_start is not None else sim_cfg.get("start_idx", 0)
    # sim_horizon resolved after loading data (may be None = all val data)
    sim_horizon_override = args.sim_horizon  # may be None

    prefix = args.prefix

    # ── Locate model directory & checkpoint ───────────────────────────
    dataset_name = config["dataset"]["name"]
    run_dir = runs_dir / dataset_name
    if run_name is not None:
        run_dir = run_dir / run_name
    use_optuna_best_params = bool(
        config.get("training", {}).get("use_optuna_best_params", False)
    )
    optuna_summary_path = _resolve_optuna_summary_path(config, run_dir, args.optuna_summary)
    optuna_best_by_model = {}
    if use_optuna_best_params:
        print("Loading Optuna best params...")
        print(f"  Summary path: {optuna_summary_path}")
        optuna_best_by_model = _load_optuna_best_params(optuna_summary_path)
        print(f"  Models with best params: {sorted(optuna_best_by_model.keys())}")
    model_dir = run_dir / model_type

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        print(f"  Available models: "
              f"{[d.name for d in run_dir.iterdir() if d.is_dir() and d.name != 'figures']}")
        return

    checkpoints = discover_checkpoints(model_dir, model_type)
    if not checkpoints:
        print(f"ERROR: no checkpoints found in {model_dir}")
        return

    # Pick checkpoint
    if args.round:
        matches = [(rn, p) for rn, p in checkpoints if rn == args.round]
        if not matches:
            avail = [rn for rn, _ in checkpoints]
            print(f"ERROR: round '{args.round}' not found. Available: {avail}")
            return
        round_name, ckpt_path = matches[0]
    else:
        # Latest checkpoint (last in sorted order)
        round_name, ckpt_path = checkpoints[-1]

    # ── Load model ────────────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}
    mc = models_cfg_map.get(model_type, {"type": model_type})
    model_overrides = {}
    if use_optuna_best_params and model_type in optuna_best_by_model:
        model_overrides, _ = _split_optuna_params(
            model_type, optuna_best_by_model[model_type]
        )
        print(f"  Optuna model overrides for {model_type}: {model_overrides}")

    if model_type in NEURAL_MODELS:
        model = build_model(
            model_type, input_dim, output_dim, seq_len, pred_len,
            per_model_cfg=mc, model_defaults_cfg=model_defaults_cfg,
            overrides=model_overrides)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    "  Warning: non-strict checkpoint load "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
        model.to(device)
        model.eval()
        n_params = count_parameters(model)
    else:
        if not HAS_XGBOOST:
            print("ERROR: xgboost is not installed.  pip install xgboost")
            return
        model = XGBoostForecaster.load(str(ckpt_path))
        n_params = 0

    # ── Load dataset ──────────────────────────────────────────────────
    index_col = config["dataset"].get("index_col",
                                       data_cfg.get("index_col", "date"))
    train_split = config["dataset"].get("train_split",
                                         data_cfg.get("train_split", 0.8))

    scaler_path = run_dir / "scaler.pkl"
    if not scaler_path.exists():
        print(f"ERROR: scaler not found: {scaler_path}")
        print("  Run scripts/train.py first.")
        return

    from joblib import load
    scaler = load(scaler_path)

    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        slice_cfg=config["dataset"].get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )

    _, val_loader, _ = build_grouped_dataloaders(
        df, groups, input_groups, output_groups,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=config["dataset"]["batch_size"],
        train_split=train_split,
        add_time=add_time_features,
        time_features_cfg=time_features_cfg,
        existing_scaler=scaler,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
    )
    val_dataset = val_loader.dataset

    # Resolve simulation horizon
    sim_horizon = sim_horizon_override
    if sim_horizon is None:
        cfg_horizon = sim_cfg.get("horizon", None)
        if cfg_horizon is not None:
            sim_horizon = cfg_horizon
        else:
            sim_horizon = len(val_dataset.values) - seq_len

    # ── Banner ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SINGLE-MODEL EVALUATION")
    print("=" * 70)
    print(f"  Config         : {args.config}")
    print(f"  Model          : {model_type}")
    print(f"  Checkpoint     : {ckpt_path.name} (round: {round_name})")
    print(f"  Parameters     : {n_params:,}" if n_params
          else f"  Parameters     : N/A (tree-based)")
    print(f"  Device         : {device}")
    print(f"  Eval horizon   : {eval_horizon}  (windows: {n_windows})"
          + ("  [SKIPPED]" if args.no_eval else ""))
    print(f"  Sim horizon    : {sim_horizon}  (start: {sim_start})"
          + ("  [SKIPPED]" if args.no_sim else ""))
    print(f"  Output prefix  : {prefix}_")
    print(f"  Output dir     : {model_dir}")
    print("=" * 70 + "\n")

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATION (multi-window rollout)
    # ══════════════════════════════════════════════════════════════════
    if not args.no_eval:
        print("Running evaluation (multi-window rollout)...")
        try:
            if model_type in NEURAL_MODELS:
                gt_list, pred_list = evaluate_neural_model(
                    model, val_dataset, warmup_len, eval_horizon,
                    control_dim, exo_dim, device, n_windows,
                )
            else:
                gt_list, pred_list = evaluate_xgboost_model(
                    model, val_dataset, seq_len, eval_horizon, n_windows,
                )

            if gt_list and pred_list:
                mean_mse = float(np.mean(
                    [np.mean((g - p) ** 2) for g, p in zip(gt_list, pred_list)]))
                mean_mae = float(np.mean(
                    [np.mean(np.abs(g - p)) for g, p in zip(gt_list, pred_list)]))
                print(f"  Eval MSE = {mean_mse:.6f}")
                print(f"  Eval MAE = {mean_mae:.6f}")
                print(f"  Windows  = {len(gt_list)}")

                # Save forecast plot (first window)
                forecast_path = model_dir / f"{prefix}_forecast.png"
                save_forecast_plot(
                    gt_list[0], pred_list[0], output_cols,
                    forecast_path,
                    title=f"{model_type.upper()} [{round_name}] – "
                          f"Forecast (horizon={eval_horizon})",
                    show_metrics=True,
                )
                print(f"  Saved -> {forecast_path}")

                # Save per-window metrics CSV
                metrics_rows = []
                for w, (gt, pred) in enumerate(zip(gt_list, pred_list)):
                    w_mse = float(np.mean((gt - pred) ** 2))
                    w_mae = float(np.mean(np.abs(gt - pred)))
                    metrics_rows.append({
                        "window": w,
                        "mse": w_mse,
                        "mae": w_mae,
                    })
                metrics_rows.append({
                    "window": "mean",
                    "mse": mean_mse,
                    "mae": mean_mae,
                })
                metrics_csv = model_dir / f"{prefix}_eval_metrics.csv"
                pd.DataFrame(metrics_rows).to_csv(
                    metrics_csv, index=False, float_format="%.6f")
                print(f"  Saved -> {metrics_csv}")
            else:
                print("  Warning: no evaluation windows produced (data too short?)")

        except Exception as exc:
            print(f"  Eval ERROR: {exc}")
            import traceback; traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════
    #  SIMULATION (recursive step-by-step)
    # ══════════════════════════════════════════════════════════════════
    if not args.no_sim:
        print(f"\nRunning recursive simulation ({sim_horizon} steps)...")
        try:
            if model_type in NEURAL_MODELS:
                sim_result = simulate_recursive_neural(
                    model, val_dataset, seq_len,
                    sim_horizon, device, start_idx=sim_start,
                )
            else:
                sim_result = simulate_recursive_xgboost(
                    model, val_dataset, seq_len,
                    sim_horizon, start_idx=sim_start,
                )

            n_steps = sim_result["n_steps"]
            if n_steps > 0:
                gt_s = sim_result["ground_truths"]
                pr_s = sim_result["predictions"]
                sim_mse = float(np.mean((gt_s - pr_s) ** 2))
                sim_mae = float(np.mean(np.abs(gt_s - pr_s)))
                print(f"  Sim MSE  = {sim_mse:.6f}")
                print(f"  Sim MAE  = {sim_mae:.6f}")
                print(f"  Steps    = {n_steps}")

                # Simulation trajectory plot
                sim_plot_path = model_dir / f"{prefix}_simulation.png"
                save_per_model_simulation_plot(
                    sim_result, output_cols, model_type,
                    sim_plot_path, plot_cfg=plot_cfg,
                )
                print(f"  Saved -> {sim_plot_path}")

                # Simulation CSV
                save_per_model_simulation_csv(
                    sim_result, output_cols,
                    model_dir, prefix,
                )
                sim_csv_path = model_dir / f"{prefix}_simulation.csv"
                print(f"  Saved -> {sim_csv_path}")
            else:
                print("  Warning: 0 simulation steps (data too short?)")

        except Exception as exc:
            print(f"  Sim ERROR: {exc}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Model     : {model_type} [{round_name}]")
    print(f"  Outputs   : {model_dir}/{prefix}_*")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
