#!/usr/bin/env python3
"""Serve a trained RSSM simulator via FastAPI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from joblib import load

from timesim.utils.config import load_config
from timesim.data.loader import build_grouped_dataloaders, load_csv_dataset
from timesim.data.schema import VariableSchema
from timesim.utils.misc import resolve_device
from timesim.models.factory import build_model
from timesim.serving import create_app
from timesim.simulator import RSSMSimulator


def _load_model_state(model: torch.nn.Module, checkpoint: str | Path, device: str):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Latent SSM may contain lazily-created aux decoder heads in checkpoints.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                "Warning: non-strict checkpoint load "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )


def parse_args():
    p = argparse.ArgumentParser(description="Serve RSSM simulator API")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--session-ttl", type=int, default=3600)
    p.add_argument("--sigma-scale", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device if args.device is not None else config.get("misc", {}).get("device", "auto"))
    seed = int(config.get("misc", {}).get("seed", 42))

    dcfg = config["dataset"]
    data_cfg = config.get("data", {})
    add_time = bool(data_cfg.get("add_time_features", False))
    tf_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(tf_cfg, dict) and "enabled" in tf_cfg:
        add_time = bool(tf_cfg.get("enabled")) or add_time
    groups = dcfg["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    seq_len = int(dcfg["seq_len"])
    pred_len = int(dcfg["pred_len"])
    batch_size = int(dcfg["batch_size"])

    input_cols = schema.columns_for_group_names(input_groups)
    output_cols = schema.columns_for_group_names(output_groups)
    input_dim = len(set(input_cols) | set(output_cols))
    output_dim = len(output_cols)

    df = load_csv_dataset(
        dcfg["csv"],
        index_col=dcfg.get("index_col", data_cfg.get("index_col", "date")),
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=dcfg.get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )

    scaler_path = Path(args.checkpoint).resolve().parent.parent / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = load(scaler_path)

    _, val_loader, _ = build_grouped_dataloaders(
        df,
        groups,
        input_groups,
        output_groups,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        train_split=dcfg.get("train_split", data_cfg.get("train_split", 0.7)),
        split_cfg=data_cfg.get("splits", None),
        seed=seed,
        shuffle_train=False,
        drop_last=bool(data_cfg.get("drop_last", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        stride=int(data_cfg.get("window_stride", 1)),
        use_symlog=bool((data_cfg.get("symlog", {}) or {}).get("enabled", False)),
        symlog_columns=(data_cfg.get("symlog", {}) or {}).get("columns", None),
        add_time=add_time,
        time_features_cfg=tf_cfg,
        existing_scaler=scaler,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
    )
    dataset = val_loader.dataset

    model_cfg = config.get("models", [{"type": "latent_ssm"}])
    latent_cfg = next((m for m in model_cfg if m.get("type") == "latent_ssm"), {"type": "latent_ssm"})

    model = build_model(
        "latent_ssm",
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        per_model_cfg=latent_cfg,
        model_defaults_cfg=config.get("model_defaults", {}),
    )
    _load_model_state(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    serving_cfg = config.get("serving", {}) or {}
    eval_prob_cfg = config.get("evaluation", {}).get("probabilistic", {}) or {}
    if args.sigma_scale is not None:
        sigma_scale = float(args.sigma_scale)
    else:
        sigma_scale = float(
            serving_cfg.get(
                "sigma_scale",
                eval_prob_cfg.get("sigma_scale", 1.0),
            )
        )
    sigma_scale = float(max(1e-6, sigma_scale))

    simulator_template = RSSMSimulator(
        model=model,
        feature_columns=dataset.feature_cols,
        input_columns=dataset.input_cols,
        output_columns=dataset.output_cols,
        in_idx=dataset.in_idx,
        out_idx=dataset.out_idx,
        control_positions=dataset.control_positions,
        known_exo_positions=dataset.known_exo_positions,
        scaler=dataset.scaler,
        sigma_scale=sigma_scale,
        device=device,
    )

    app = create_app(simulator_template, session_ttl_seconds=args.session_ttl)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
