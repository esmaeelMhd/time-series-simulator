#!/usr/bin/env python3
"""Hydra entrypoint for training.

This wraps the existing ``scripts/train.py`` pipeline so current
functionality is preserved while enabling Hydra/OmegaConf overrides.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import hydra  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception as exc:
    raise ImportError(
        "Hydra entrypoint requires hydra-core and omegaconf. "
        "Install with: pip install hydra-core omegaconf"
    ) from exc

from timesim.utils.config import deep_merge, load_config


def _to_plain_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _as_model_overrides(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(model_cfg or {})
    # Friendly aliases used in design docs.
    if out.get("dim_h", None) is not None:
        out["hidden_dim"] = out["dim_h"]
    if out.get("dim_z", None) is not None:
        out["latent_dim"] = out["dim_z"]
    for k in ["name", "dim_h", "dim_z"]:
        out.pop(k, None)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_d = _to_plain_dict(cfg)
    project_root = Path(__file__).resolve().parent.parent
    base_config_path = Path(str(cfg_d.get("base_config", "configs/wastewater.yaml")))
    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path
    base = load_config(base_config_path)
    overrides = cfg_d.get("overrides", {}) or {}

    hydra_overrides: Dict[str, Any] = {}
    if isinstance(cfg_d.get("training"), dict):
        hydra_overrides["training"] = dict(cfg_d["training"])
    if isinstance(cfg_d.get("data"), dict):
        hydra_overrides["data"] = dict(cfg_d["data"])
    if isinstance(cfg_d.get("serving"), dict):
        hydra_overrides["serving"] = dict(cfg_d["serving"])

    model_cfg = cfg_d.get("model", {}) or {}
    if isinstance(model_cfg, dict) and model_cfg:
        model_type = str(model_cfg.get("type", "latent_ssm"))
        model_entry = {"type": model_type, **_as_model_overrides(model_cfg)}
        hydra_overrides["models"] = [model_entry]

    seed_val = cfg_d.get("seed", None)
    deterministic_val = cfg_d.get("deterministic", None)
    if seed_val is not None or deterministic_val is not None:
        hydra_overrides.setdefault("misc", {})
        if seed_val is not None:
            hydra_overrides["misc"]["seed"] = int(seed_val)
        if deterministic_val is not None:
            hydra_overrides["misc"]["deterministic"] = bool(deterministic_val)

    merged = deep_merge(base, hydra_overrides)
    merged = deep_merge(merged, overrides)

    run_dir = Path.cwd()
    resolved_cfg_path = run_dir / "hydra_resolved_config.yaml"
    OmegaConf.save(config=OmegaConf.create(merged), f=str(resolved_cfg_path))

    cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--config", str(resolved_cfg_path)]

    if cfg_d.get("models"):
        models = cfg_d["models"]
        if isinstance(models, str):
            models = [models]
        cmd += ["--models", *[str(m) for m in models]]
    if cfg_d.get("epochs") is not None:
        cmd += ["--epochs", str(cfg_d["epochs"])]
    if cfg_d.get("steps_per_epoch") is not None:
        cmd += ["--steps-per-epoch", str(cfg_d["steps_per_epoch"])]
    if cfg_d.get("device"):
        cmd += ["--device", str(cfg_d["device"])]

    print("Hydra launch command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
