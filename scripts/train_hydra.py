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


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="train")
def main(cfg: DictConfig) -> None:
    cfg_d = _to_plain_dict(cfg)
    project_root = Path(__file__).resolve().parent.parent
    base_config_path = Path(str(cfg_d.get("base_config", "configs/wastewater.yaml")))
    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path
    base = load_config(base_config_path)
    overrides = cfg_d.get("overrides", {}) or {}
    merged = deep_merge(base, overrides)

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
