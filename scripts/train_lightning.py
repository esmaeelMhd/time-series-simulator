#!/usr/bin/env python3
"""Experimental Lightning-based training entrypoint."""

from __future__ import annotations

import argparse

import torch

from timesim.utils.config import load_config
from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
from timesim.data.schema import VariableSchema
from timesim.data.stamps import get_time_feature_columns
from timesim.models.factory import build_model
from timesim.lightning import WorldModelLightningModule
from timesim.utils.misc import seed_everything, resolve_device


def parse_args():
    p = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, default="latent_ssm")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("misc", {})
        cfg["misc"]["device"] = args.device
    seed = int(cfg.get("misc", {}).get("seed", 42))
    deterministic = bool(cfg.get("misc", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(cfg.get("misc", {}).get("device", "auto"))
    cfg.setdefault("misc", {})
    cfg["misc"]["device"] = device

    dcfg = cfg["dataset"]
    data_cfg = cfg.get("data", {})
    seq_len = int(dcfg["seq_len"])
    pred_len = int(dcfg["pred_len"])
    batch_size = int(dcfg["batch_size"])

    groups = dcfg["variables"]
    schema = VariableSchema.from_groups(groups)
    input_groups = cfg["model_io"]["input_groups"]
    output_groups = cfg["model_io"]["output_groups"]

    add_time = bool(data_cfg.get("add_time_features", False))
    tf_cfg = data_cfg.get("time_features", {}) or {}
    if isinstance(tf_cfg, dict) and "enabled" in tf_cfg:
        add_time = bool(tf_cfg.get("enabled")) or add_time

    df = load_csv_dataset(
        dcfg["csv"],
        index_col=dcfg.get("index_col", data_cfg.get("index_col", "date")),
        parse_dates=bool(data_cfg.get("parse_dates", True)),
        slice_cfg=dcfg.get("slice"),
        engine=str(data_cfg.get("csv_engine", "pandas")),
        validation_cfg=data_cfg.get("validation", None),
    )
    train_loader, val_loader, _ = build_grouped_dataloaders(
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
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        drop_last=bool(data_cfg.get("drop_last", True)),
        add_time=add_time,
        time_features_cfg=tf_cfg,
        require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
    )

    input_cols = schema.columns_for_group_names(input_groups)
    output_cols = schema.columns_for_group_names(output_groups)
    input_dim = len(set(input_cols) | set(output_cols))
    if add_time:
        input_dim += len(
            get_time_feature_columns(
                features=tf_cfg.get("features"),
                encoding=tf_cfg.get("encoding", "cyclical"),
            )
        )
    output_dim = len(output_cols)

    models_cfg = {m["type"]: m for m in cfg.get("models", [])}
    model_cfg = models_cfg.get(args.model, {"type": args.model})
    model = build_model(
        args.model,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        per_model_cfg=model_cfg,
        model_defaults_cfg=cfg.get("model_defaults", {}),
    )
    module = WorldModelLightningModule(
        model=model,
        learning_rate=float(model_cfg.get("learning_rate", cfg.get("training", {}).get("learning_rate", 3e-4))),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-6)),
        scheduler_warmup_steps=int(
            cfg.get("training", {}).get("probabilistic", {}).get("lr_warmup_steps", 1000)
        ),
    )

    try:
        import pytorch_lightning as pl  # type: ignore
    except Exception as exc:
        raise ImportError("train_lightning.py requires pytorch-lightning") from exc

    epochs = int(args.epochs or cfg.get("training", {}).get("epochs", 10))
    accelerator = "gpu" if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=100.0,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
