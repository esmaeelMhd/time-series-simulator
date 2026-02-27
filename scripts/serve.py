#!/usr/bin/env python3
"""Serve a trained RSSM simulator via FastAPI."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import logging

import torch
from timesim.utils.misc import configure_torch_defaults
configure_torch_defaults()
from joblib import load
import numpy as np
import pandas as pd

from timesim.utils.config import compose_config
from timesim.data.loader import build_dataloaders_from_config, load_csv_dataset
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


def _build_cli_parser():
    p = argparse.ArgumentParser(description="Serve RSSM simulator API")
    p.add_argument("--config", "--config-name", type=str, required=True,
                   help="Hydra config name or path to YAML config")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--session-ttl", type=int, default=3600)
    p.add_argument("--sigma-scale", type=float, default=None)
    return p


def main():
    parser = _build_cli_parser()
    args, hydra_overrides = parser.parse_known_args()
    config = compose_config(args.config, overrides=hydra_overrides)
    device = resolve_device(args.device if args.device is not None else config.get("misc", {}).get("device", "auto"))
    seed = int(config.get("misc", {}).get("seed", 42))

    dcfg = config["dataset"]
    data_cfg = config.get("data", {})
    groups = dcfg["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = config["model_io"]["input_groups"]
    output_groups = config["model_io"]["output_groups"]

    dataset_seq_len = int(dcfg["seq_len"])
    pred_len = int(dcfg["pred_len"])
    seq_len = int(
        config.get("training", {}).get(
            "window_len",
            config.get("training", {}).get("warmup_len", dataset_seq_len),
        )
    )
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

    _, val_loader, _ = build_dataloaders_from_config(
        config=config,
        df=df,
        seed=seed,
        scaler=scaler,
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

    simulator_template = RSSMSimulator.from_dataset(
        model=model,
        dataset=dataset,
        sigma_scale=sigma_scale,
        device=device,
    )

    serving_cfg = config.get("serving", {}) or {}
    if bool(serving_cfg.get("benchmark_on_startup", True)):
        logger = logging.getLogger("timesim.serve")
        logger.setLevel(logging.INFO)
        try:
            sim = simulator_template.clone_empty()
            hist_df = df[dataset.feature_cols].tail(seq_len).copy()
            sim.reset(hist_df)
            control_last = {
                c: float(hist_df[c].iloc[-1]) for c in sim.control_columns
            }
            exo_last = {
                c: float(hist_df[c].iloc[-1]) for c in sim.exogenous_columns
            } if sim.exogenous_columns else {}

            t0 = time.perf_counter()
            _ = sim.step(control_last, exo_last if exo_last else None, n_samples=50)
            step_ms = (time.perf_counter() - t0) * 1000.0

            horizon = 50
            ctrl_df = pd.DataFrame(
                {
                    c: np.full((horizon,), control_last[c], dtype=np.float32)
                    for c in sim.control_columns
                }
            )
            exo_df = (
                pd.DataFrame(
                    {
                        c: np.full((horizon,), exo_last[c], dtype=np.float32)
                        for c in sim.exogenous_columns
                    }
                )
                if sim.exogenous_columns
                else None
            )
            t1 = time.perf_counter()
            _ = sim.rollout(ctrl_df, exo_df, n_samples=50)
            rollout_ms = (time.perf_counter() - t1) * 1000.0
            logger.info(
                "Serving benchmark: step_1x50=%.3fms (target<10ms), rollout_50x50=%.3fms (target<500ms)",
                step_ms,
                rollout_ms,
            )
        except Exception as exc:
            logger.warning("Serving benchmark failed: %s", exc)

    app = create_app(simulator_template, session_ttl_seconds=args.session_ttl)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
