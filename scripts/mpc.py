#!/usr/bin/env python3
"""Run latent CEM-MPC with a frozen LatentSSM checkpoint.

Outputs are saved in:
  <runs_dir>/<dataset>/<run_name>/<model>/

Artifacts:
- <prefix>_summary.yaml
- <prefix>_actions.csv
- <prefix>_predictions.csv
- <prefix>_cost_history.csv
- <prefix>_plan.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from timesim.utils.misc import configure_torch_defaults

configure_torch_defaults()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from joblib import load

from timesim.control import CEMController
from timesim.data.loader import (
    build_grouped_triplet_dataloaders,
    load_csv_dataset,
)
from timesim.data.schema import VariableRole, VariableSchema
from timesim.utils.config import compose_config
from timesim.utils.misc import resolve_device, seed_everything


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run latent CEM-MPC and save outputs.")
    p.add_argument("--config", "--config-name", required=True, type=str)
    p.add_argument("--model", default="latent_ssm", type=str)
    p.add_argument("--device", default=None, type=str, help="auto / cpu / cuda")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--horizon", default=None, type=int)
    p.add_argument("--start-idx", default=None, type=int, help="Start index in selected split values.")
    p.add_argument("--prefix", default="mpc", type=str)
    p.add_argument("--population", default=None, type=int)
    p.add_argument("--iterations", default=None, type=int)
    p.add_argument("--elite-frac", default=None, type=float)
    p.add_argument("--init-std", default=None, type=float)
    p.add_argument("--min-std", default=None, type=float)
    p.add_argument("--sample-latent", action="store_true")
    return p


def _infer_role_positions(dataset) -> tuple[list[int], list[int]]:
    schema = getattr(dataset, "variable_schema", None)
    input_cols = list(getattr(dataset, "input_cols", []))
    if schema is not None and input_cols:
        control_cols = list(schema.columns_for_role(VariableRole.CONTROL))
        exo_cols = list(schema.columns_for_role(VariableRole.EXOGENOUS))
        cpos = [i for i, c in enumerate(input_cols) if c in control_cols]
        xpos = [i for i, c in enumerate(input_cols) if c in exo_cols]
        return cpos, xpos
    cpos = list(getattr(dataset, "control_positions", []))
    xpos = list(getattr(dataset, "known_exo_positions", []))
    return cpos, xpos


def _build_split_dataset(config: dict[str, Any], df: pd.DataFrame, scaler, seed: int, split: str):
    data_cfg = config.get("data", {}) or {}
    dataset_cfg = config["dataset"]
    model_io_cfg = config["model_io"]
    time_features_cfg = data_cfg.get("time_features", {}) or {}
    add_time = bool(data_cfg.get("add_time_features", False))
    if isinstance(time_features_cfg, dict) and "enabled" in time_features_cfg:
        add_time = bool(time_features_cfg.get("enabled")) or add_time
    split_cfg = data_cfg.get("splits", None)
    train_split = float((split_cfg or {}).get("train", 0.7))

    _, val_loader, test_loader, _ = build_grouped_triplet_dataloaders(
        df=df,
        groups=dataset_cfg["variables"],
        input_groups=model_io_cfg["input_groups"],
        output_groups=model_io_cfg["output_groups"],
        seq_len=int(dataset_cfg["seq_len"]),
        pred_len=int(dataset_cfg["pred_len"]),
        batch_size=int(dataset_cfg["batch_size"]),
        train_split=train_split,
        split_cfg=split_cfg,
        add_time=add_time,
        time_features_cfg=time_features_cfg,
        existing_scaler=scaler,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
        seed=seed,
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        drop_last=bool(data_cfg.get("drop_last", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        stride=int(data_cfg.get("window_stride", 1)),
        use_symlog=bool((data_cfg.get("symlog", {}) or {}).get("enabled", False)),
        symlog_columns=(data_cfg.get("symlog", {}) or {}).get("columns", None),
    )
    return test_loader.dataset if split == "test" else val_loader.dataset


def _save_plot(
    pred: np.ndarray,
    gt: np.ndarray,
    actions: np.ndarray,
    output_cols: list[str],
    control_cols: list[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    h = np.arange(1, pred.shape[0] + 1)
    for j in range(pred.shape[1]):
        label = output_cols[j] if j < len(output_cols) else f"y{j}"
        axes[0].plot(h, pred[:, j], label=f"pred:{label}", linewidth=2.0)
        axes[0].plot(h, gt[:, j], "--", label=f"gt:{label}", linewidth=1.4, alpha=0.85)
    axes[0].set_ylabel("Objective")
    axes[0].set_title("MPC predicted trajectory vs ground truth")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    for j in range(actions.shape[1]):
        label = control_cols[j] if j < len(control_cols) else f"u{j}"
        axes[1].plot(h, actions[:, j], label=label, linewidth=2.0)
    axes[1].set_xlabel("Horizon step")
    axes[1].set_ylabel("Action")
    axes[1].set_title("Optimized control trajectory")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = _build_cli_parser()
    args, hydra_overrides = parser.parse_known_args()
    config = compose_config(args.config, overrides=hydra_overrides)

    model_type = str(args.model).lower().strip()
    if model_type != "latent_ssm":
        raise ValueError("This MPC script currently supports only model='latent_ssm'.")

    seed = int(config.get("misc", {}).get("seed", 42))
    seed_everything(seed)
    device = resolve_device(
        args.device if args.device is not None else config.get("misc", {}).get("device", "auto")
    )

    dataset_name = config["dataset"]["name"]
    output_cfg = config.get("output", {}) or {}
    run_name = output_cfg.get("run_name", None)
    if isinstance(run_name, str):
        run_name = run_name.strip() or None
    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_dir = runs_dir / dataset_name
    if run_name is not None:
        run_dir = run_dir / run_name
    model_dir = run_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build dims from config schema for readable output labels and bounds checks.
    groups = config["dataset"]["variables"]
    schema = VariableSchema.from_groups(groups)
    output_groups = config["model_io"]["output_groups"]
    output_cols = schema.columns_for_group_names(output_groups)
    control_cols = list(schema.columns_for_role(VariableRole.CONTROL))

    data_cfg = config.get("data", {}) or {}
    index_col = config["dataset"].get("index_col", data_cfg.get("index_col", "date"))
    df = load_csv_dataset(
        config["dataset"]["csv"],
        index_col=index_col,
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=config["dataset"].get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )
    scaler_path = run_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler at {scaler_path}")
    scaler = load(scaler_path)
    dataset = _build_split_dataset(config, df, scaler, seed, args.split)

    warmup_len = int(
        config.get("training", {}).get(
            "window_len",
            config.get("training", {}).get("warmup_len", int(config["dataset"]["seq_len"])),
        )
    )
    mpc_cfg = config.get("mpc", {}) or {}
    horizon = int(
        args.horizon
        if args.horizon is not None
        else mpc_cfg.get("horizon", config.get("evaluation", {}).get("horizon", 30))
    )
    start_idx = int(
        args.start_idx
        if args.start_idx is not None
        else mpc_cfg.get("start_idx", warmup_len)
    )

    total_len = len(dataset.values)
    if start_idx < warmup_len:
        raise ValueError(f"start_idx ({start_idx}) must be >= warmup_len ({warmup_len})")
    if start_idx + horizon > total_len:
        raise ValueError(
            f"start_idx+horizon exceeds split length: {start_idx}+{horizon}>{total_len}"
        )

    controller = CEMController.from_run_dir(
        model_dir,
        horizon=horizon,
        device=device,
        population=int(args.population if args.population is not None else mpc_cfg.get("population", 1000)),
        iterations=int(args.iterations if args.iterations is not None else mpc_cfg.get("iterations", 5)),
        elite_frac=float(args.elite_frac if args.elite_frac is not None else mpc_cfg.get("elite_frac", 0.1)),
        init_std=float(args.init_std if args.init_std is not None else mpc_cfg.get("init_std", 0.5)),
        min_std=float(args.min_std if args.min_std is not None else mpc_cfg.get("min_std", 1e-3)),
        sample_latent=bool(args.sample_latent),
        action_low=mpc_cfg.get("action_low", None),
        action_high=mpc_cfg.get("action_high", None),
        momentum=float(mpc_cfg.get("momentum", 0.0)),
    )
    model = controller.model

    # Extract role-indexed tensors
    cpos, xpos = _infer_role_positions(dataset)
    warmup_vals = dataset.values[start_idx - warmup_len : start_idx]
    future_vals = dataset.values[start_idx : start_idx + horizon]
    warmup_inputs = warmup_vals[:, dataset.in_idx]
    future_inputs = future_vals[:, dataset.in_idx]

    warmup_controls = (
        warmup_inputs[:, cpos] if cpos else np.zeros((warmup_len, 0), dtype=np.float32)
    )
    warmup_exo = (
        warmup_inputs[:, xpos] if xpos else np.zeros((warmup_len, 0), dtype=np.float32)
    )
    warmup_outputs = warmup_vals[:, dataset.out_idx]
    future_exo = future_inputs[:, xpos] if xpos else np.zeros((horizon, 0), dtype=np.float32)
    future_targets = future_vals[:, dataset.out_idx]

    with torch.no_grad():
        observed = model.observe(
            controls=torch.from_numpy(warmup_controls).float().unsqueeze(0).to(device),
            exogenous=torch.from_numpy(warmup_exo).float().unsqueeze(0).to(device),
            observations=torch.from_numpy(warmup_outputs).float().unsqueeze(0).to(device),
            initial_state=None,
            sample_posterior=False,
        )
        initial_state = observed["state"]

    target_t = torch.from_numpy(future_targets).float().to(device)
    w_track = float(mpc_cfg.get("cost_track_weight", 1.0))
    w_action = float(mpc_cfg.get("cost_action_weight", 0.01))
    w_smooth = float(mpc_cfg.get("cost_smooth_weight", 0.01))

    def cost_fn(y_preds: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # y_preds: [N,H,dim_y], actions: [N,H,dim_c]
        tgt = target_t.unsqueeze(0).expand(y_preds.shape[0], -1, -1).to(dtype=y_preds.dtype)
        track = ((y_preds - tgt) ** 2).mean(dim=(1, 2))
        effort = (actions ** 2).mean(dim=(1, 2))
        if actions.shape[1] > 1:
            smooth = ((actions[:, 1:, :] - actions[:, :-1, :]) ** 2).mean(dim=(1, 2))
        else:
            smooth = torch.zeros_like(track)
        return w_track * track + w_action * effort + w_smooth * smooth

    result = controller.optimize(
        initial_state=initial_state,
        future_exogenous=future_exo,
        cost_function=cost_fn,
    )

    prefix = str(args.prefix).strip() or "mpc"
    actions_np = result["best_actions"].detach().cpu().numpy()
    pred_np = result["trajectory"].detach().cpu().numpy()
    gt_np = future_targets.astype(np.float32, copy=False)

    actions_path = model_dir / f"{prefix}_actions.csv"
    preds_path = model_dir / f"{prefix}_predictions.csv"
    cost_path = model_dir / f"{prefix}_cost_history.csv"
    fig_path = model_dir / f"{prefix}_plan.png"
    summary_path = model_dir / f"{prefix}_summary.yaml"

    pd.DataFrame(
        {
            "horizon": np.arange(1, horizon + 1, dtype=int),
            **{f"action_{i}": actions_np[:, i] for i in range(actions_np.shape[1])},
        }
    ).to_csv(actions_path, index=False, float_format="%.6f")

    pd.DataFrame(
        {
            "horizon": np.arange(1, horizon + 1, dtype=int),
            **{f"pred_{i}": pred_np[:, i] for i in range(pred_np.shape[1])},
            **{f"gt_{i}": gt_np[:, i] for i in range(gt_np.shape[1])},
        }
    ).to_csv(preds_path, index=False, float_format="%.6f")

    pd.DataFrame(
        {
            "iteration": np.arange(1, len(result["cost_history"]) + 1, dtype=int),
            "best_cost": np.asarray(result["cost_history"], dtype=np.float32),
        }
    ).to_csv(cost_path, index=False, float_format="%.6f")

    _save_plot(pred_np, gt_np, actions_np, output_cols=output_cols, control_cols=control_cols, out_path=fig_path)

    action_t0 = result["action_t0"].detach().cpu().numpy()
    summary = {
        "config": args.config,
        "model": model_type,
        "split": args.split,
        "device": str(device),
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "warmup_len": int(warmup_len),
        "horizon": int(horizon),
        "start_idx": int(start_idx),
        "population": int(controller.population),
        "iterations": int(controller.iterations),
        "elite_frac": float(controller.elite_frac),
        "init_std": float(controller.init_std),
        "min_std": float(controller.min_std),
        "sample_latent": bool(controller.sample_latent),
        "best_cost": float(result["best_cost"]),
        "action_t0": action_t0.tolist(),
        "cost_weights": {
            "track": w_track,
            "action": w_action,
            "smooth": w_smooth,
        },
        "artifacts": {
            "actions_csv": str(actions_path),
            "predictions_csv": str(preds_path),
            "cost_history_csv": str(cost_path),
            "plan_png": str(fig_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("\n" + "=" * 70)
    print("  MPC COMPLETE")
    print("=" * 70)
    print(f"  Model dir   : {model_dir}")
    print(f"  Split       : {args.split}")
    print(f"  Start idx   : {start_idx}")
    print(f"  Horizon     : {horizon}")
    print(f"  Best cost   : {float(result['best_cost']):.6f}")
    print(f"  Action t=0  : {action_t0}")
    print(f"  Saved       : {summary_path.name}")
    print(f"  Saved       : {actions_path.name}")
    print(f"  Saved       : {preds_path.name}")
    print(f"  Saved       : {cost_path.name}")
    print(f"  Saved       : {fig_path.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

