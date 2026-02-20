#!/usr/bin/env python3
"""Hydra-native training entrypoint (no subprocess handoff)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import hydra  # type: ignore
from omegaconf import DictConfig, OmegaConf  # type: ignore

from timesim.config import coerce_and_validate_train_cfg

CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "configs").as_posix())
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_legacy as legacy


def _to_plain_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _as_model_overrides(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(model_cfg or {})
    if out.get("dim_h", None) is not None:
        out["hidden_dim"] = out["dim_h"]
    if out.get("dim_z", None) is not None:
        out["latent_dim"] = out["dim_z"]
    for k in ["name", "dim_h", "dim_z"]:
        out.pop(k, None)
    return out


def _build_legacy_config(cfg_d: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg_d.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "latent_ssm"))
    model_entry = {"type": model_type, **_as_model_overrides(model_cfg)}

    legacy_cfg: Dict[str, Any] = {
        "dataset": dict(cfg_d.get("dataset", {})),
        "data": dict(cfg_d.get("data", {})),
        "model_io": dict(cfg_d.get("model_io", {})),
        "models": [model_entry],
        "training": dict(cfg_d.get("training", {})),
        "training_rounds": list(cfg_d.get("training_rounds", []) or []),
        "evaluation": dict(cfg_d.get("evaluation", {})),
        "simulation": dict(cfg_d.get("simulation", {})),
        "output": dict(cfg_d.get("output", {})),
        "tracking": dict(cfg_d.get("tracking", {})),
        "plotting": dict(cfg_d.get("plotting", {})),
        "optimization": dict(cfg_d.get("optimization", {})),
        "model_defaults": dict(cfg_d.get("model_defaults", {})),
        "misc": dict(cfg_d.get("misc", {})),
        "architecture": dict(cfg_d.get("architecture", {})),
    }
    # Keep legacy codepath happy if not explicitly set.
    if "input_groups" not in legacy_cfg["model_io"]:
        legacy_cfg["model_io"]["input_groups"] = ["control", "exogenous", "objective"]
    if "output_groups" not in legacy_cfg["model_io"]:
        legacy_cfg["model_io"]["output_groups"] = ["objective"]
    return legacy_cfg


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_d = coerce_and_validate_train_cfg(_to_plain_dict(cfg))
    legacy_cfg = _build_legacy_config(cfg_d)

    args = argparse.Namespace(
        config="<hydra>",
        models=cfg_d.get("models"),
        epochs=cfg_d.get("epochs"),
        steps_per_epoch=cfg_d.get("steps_per_epoch"),
        device=cfg_d.get("device"),
        use_optuna_best_params=cfg_d.get("use_optuna_best_params"),
        optuna_summary=cfg_d.get("optuna_summary"),
    )
    legacy.main(config=legacy_cfg, args=args)


if __name__ == "__main__":
    main()
