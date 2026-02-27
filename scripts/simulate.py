#!/usr/bin/env python3
"""Simulate a trained model on the full dataset from a chosen start index.

This script loads a trained checkpoint (neural or XGBoost), restores the saved
scaler from the run directory, builds a GroupedTimeSeriesDataset on the full
CSV, then runs recursive simulation for a chosen horizon.

Compared to eval.py:
- Uses the full dataset by default (ignores dataset.slice unless requested)
- Focuses only on environment-style recursive simulation
- Supports fixed or random start index selection
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from timesim.utils.misc import configure_torch_defaults
configure_torch_defaults()
import yaml

from timesim.utils.config import compose_config
from timesim.data.loader import load_csv_dataset
from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.schema import VariableSchema
from timesim.models.factory import build_model, count_parameters, NEURAL_MODELS
from timesim.utils.misc import seed_everything, resolve_device

try:
    from timesim.models.xgboost_model import XGBoostForecaster
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import matplotlib
matplotlib.use("Agg")

# Shared simulation / plotting utilities (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (
    simulate_recursive_neural,
    simulate_recursive_xgboost,
    save_per_model_simulation_plot,
    save_per_model_simulation_csv,
)


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


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Simulate a trained model on the full dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "--config-name", type=str, required=True,
                        help="Hydra config name or path to YAML config")
    parser.add_argument("--model", type=str, required=True,
                        help="Model type to simulate (e.g. lstm, transformer, xgboost)")
    parser.add_argument("--round", type=str, default=None,
                        help="Checkpoint round name to load (default: latest)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (auto / cpu / cuda)")

    sim_grp = parser.add_argument_group("simulation")
    sim_grp.add_argument("--horizon", type=int, default=None,
                         help="Simulation horizon (recursive steps). "
                              "Default: config.simulation.horizon or max possible.")
    sim_grp.add_argument("--start-idx", type=int, default=None,
                         help="Start index in full dataset (default: random valid index)")
    sim_grp.add_argument("--seed", type=int, default=None,
                         help="Seed used for random start index selection")
    sim_grp.add_argument("--use-config-slice", action="store_true",
                         help="Use dataset.slice from config instead of full CSV")

    out_grp = parser.add_argument_group("output")
    out_grp.add_argument("--prefix", type=str, default="simulate",
                         help="Output filename prefix (default: simulate)")

    return parser


def main():
    parser = _build_cli_parser()
    args, hydra_overrides = parser.parse_known_args()

    config = compose_config(args.config, overrides=hydra_overrides)
    if args.device:
        config["misc"]["device"] = args.device

    seed = int(args.seed if args.seed is not None else config["misc"].get("seed", 42))
    deterministic = bool(config.get("misc", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(config.get("misc", {}).get("device", "auto"))
    config.setdefault("misc", {})
    config["misc"]["device"] = device

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

    # ── Dimensions / groups ───────────────────────────────────────────
    groups = config["dataset"]["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    input_cols = schema.columns_for_group_names(input_groups)
    output_cols = schema.columns_for_group_names(output_groups)

    seq_len = config["dataset"]["seq_len"]
    pred_len = config["dataset"]["pred_len"]
    all_input_features = set(input_cols) | set(output_cols)
    input_dim = len(all_input_features)
    output_dim = len(output_cols)

    # ── Locate run/model/checkpoint ───────────────────────────────────
    dataset_name = config["dataset"]["name"]
    run_dir = runs_dir / dataset_name
    if run_name is not None:
        run_dir = run_dir / run_name
    model_dir = run_dir / model_type

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        return

    checkpoints = discover_checkpoints(model_dir, model_type)
    if not checkpoints:
        print(f"ERROR: no checkpoints found in {model_dir}")
        return

    if args.round:
        matches = [(rn, p) for rn, p in checkpoints if rn == args.round]
        if not matches:
            available = [rn for rn, _ in checkpoints]
            print(f"ERROR: round '{args.round}' not found. Available: {available}")
            return
        round_name, ckpt_path = matches[0]
    else:
        round_name, ckpt_path = checkpoints[-1]

    # ── Load model ────────────────────────────────────────────────────
    models_cfg_list = config.get("models", [])
    models_cfg_map = {m["type"]: m for m in models_cfg_list}
    mc = models_cfg_map.get(model_type, {"type": model_type})

    if model_type in NEURAL_MODELS:
        model = build_model(
            model_type, input_dim, output_dim, seq_len, pred_len,
            per_model_cfg=mc, model_defaults_cfg=model_defaults_cfg
        )
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        n_params = count_parameters(model)
    else:
        if not HAS_XGBOOST:
            print("ERROR: xgboost is not installed. pip install xgboost")
            return
        model = XGBoostForecaster.load(str(ckpt_path))
        n_params = 0

    # ── Load scaler + full dataset ────────────────────────────────────
    scaler_path = run_dir / "scaler.pkl"
    if not scaler_path.exists():
        print(f"ERROR: scaler not found: {scaler_path}")
        print("Run scripts/train.py first.")
        return

    from joblib import load
    scaler = load(scaler_path)

    index_col = config["dataset"].get("index_col", data_cfg.get("index_col", "date"))
    slice_cfg = config["dataset"].get("slice") if args.use_config_slice else None
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=slice_cfg,
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )

    dataset_full = GroupedTimeSeriesDataset(
        df=df,
        groups=groups,
        input_groups=input_groups,
        output_groups=output_groups,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        stride=int(data_cfg.get("window_stride", 1)),
        use_symlog=bool((data_cfg.get("symlog", {}) or {}).get("enabled", False)),
        symlog_columns=(data_cfg.get("symlog", {}) or {}).get("columns", None),
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
    )

    n_total = len(dataset_full.values)
    max_possible_horizon = n_total - seq_len
    if max_possible_horizon <= 0:
        print(
            f"ERROR: dataset too short for seq_len={seq_len}. "
            f"Need > {seq_len} rows, got {n_total}."
        )
        return

    # Resolve horizon
    cfg_horizon = config.get("simulation", {}).get("horizon", None)
    requested_horizon = args.horizon if args.horizon is not None else cfg_horizon

    if requested_horizon is not None and requested_horizon < 1:
        print("ERROR: --horizon must be >= 1")
        return

    # Resolve start index
    if args.start_idx is not None:
        start_idx = args.start_idx
        if start_idx < 0:
            print("ERROR: --start-idx must be >= 0")
            return
    else:
        if requested_horizon is None:
            start_max = n_total - seq_len - 1
        else:
            start_max = n_total - seq_len - requested_horizon
        if start_max < 0:
            start_max = 0
        start_idx = int(np.random.randint(0, start_max + 1))

    # Clip start index to valid bounds for at least one step
    max_valid_start = n_total - seq_len - 1
    if max_valid_start < 0:
        print("ERROR: no valid start index for simulation.")
        return
    if start_idx > max_valid_start:
        print(f"Warning: start_idx={start_idx} too large, clipping to {max_valid_start}")
        start_idx = max_valid_start

    # Final horizon given start index
    remaining = n_total - seq_len - start_idx
    if remaining <= 0:
        print("ERROR: no simulation steps available after start index.")
        return
    if requested_horizon is None:
        sim_horizon = remaining
    else:
        sim_horizon = min(requested_horizon, remaining)
        if sim_horizon < requested_horizon:
            print(
                f"Warning: requested horizon={requested_horizon} exceeds remaining "
                f"steps={remaining}; using {sim_horizon}."
            )

    # ── Simulate ───────────────────────────────────────────────────────
    if model_type in NEURAL_MODELS:
        sim_result = simulate_recursive_neural(
            model=model,
            val_dataset=dataset_full,
            seq_len=seq_len,
            sim_horizon=sim_horizon,
            device=device,
            start_idx=start_idx,
        )
    else:
        sim_result = simulate_recursive_xgboost(
            model=model,
            val_dataset=dataset_full,
            seq_len=seq_len,
            sim_horizon=sim_horizon,
            start_idx=start_idx,
        )

    if sim_result["n_steps"] <= 0:
        print("ERROR: simulation produced zero steps.")
        return

    gt = sim_result["ground_truths"]
    pred = sim_result["predictions"]
    sim_mse = float(np.mean((gt - pred) ** 2))
    sim_mae = float(np.mean(np.abs(gt - pred)))

    # ── Save outputs ───────────────────────────────────────────────────
    prefix = args.prefix.strip() or "simulate"
    plot_path = model_dir / f"{prefix}_simulation.png"
    save_per_model_simulation_plot(
        sim_result=sim_result,
        output_cols=output_cols,
        model_name=f"{model_type} ({round_name})",
        out_path=plot_path,
        plot_cfg=plot_cfg,
    )

    # Reuse helper: writes <prefix>_simulation.csv
    save_per_model_simulation_csv(
        sim_result=sim_result,
        output_cols=output_cols,
        model_dir=model_dir,
        round_name=prefix,
    )
    csv_path = model_dir / f"{prefix}_simulation.csv"

    meta = {
        "model_type": model_type,
        "round": round_name,
        "checkpoint": str(ckpt_path),
        "dataset_rows": int(n_total),
        "seq_len": int(seq_len),
        "start_idx": int(start_idx),
        "requested_horizon": None if requested_horizon is None else int(requested_horizon),
        "used_horizon": int(sim_result["n_steps"]),
        "mse": sim_mse,
        "mae": sim_mae,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "params": int(n_params),
        "used_config_slice": bool(args.use_config_slice),
    }
    meta_path = model_dir / f"{prefix}_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print("\n" + "=" * 70)
    print("  FULL-DATASET SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Model        : {model_type} ({round_name})")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Dataset rows : {n_total}")
    print(f"  Start idx    : {start_idx}")
    print(f"  Horizon      : {sim_result['n_steps']}")
    print(f"  MSE / MAE    : {sim_mse:.6f} / {sim_mae:.6f}")
    print(f"  Plot         : {plot_path}")
    print(f"  CSV          : {csv_path}")
    print(f"  Meta         : {meta_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

