#!/usr/bin/env python3
"""Hydra-native training entrypoint (no subprocess handoff)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import hydra  # type: ignore
from omegaconf import DictConfig, OmegaConf  # type: ignore

from timesim.config import coerce_and_validate_train_cfg
from timesim.utils.config import _materialize_legacy_shape

CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "configs").as_posix())
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train as legacy


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_d: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    cfg_d = coerce_and_validate_train_cfg(cfg_d)
    cfg_d = _materialize_legacy_shape(cfg_d)
    legacy.main(config=cfg_d)


if __name__ == "__main__":
    main()
