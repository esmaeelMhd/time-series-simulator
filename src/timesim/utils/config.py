"""Hydra-based config loader used by scripts outside hydra.main."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

__all__ = ["load_config"]


def _find_configs_root(cfg_path: Path) -> Path:
    for parent in [cfg_path.parent, *cfg_path.parents]:
        if parent.name == "configs":
            return parent
    return cfg_path.parent


def _compose_hydra_config(cfg_path: Path) -> Dict[str, Any]:
    root = _find_configs_root(cfg_path)
    config_name = cfg_path.relative_to(root).with_suffix("").as_posix()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name=config_name)

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _apply_cli_overrides(cfg: Dict[str, Any], cli_args: Namespace | None) -> Dict[str, Any]:
    if cli_args is None:
        return cfg
    for key, value in vars(cli_args).items():
        if value is not None:
            cfg[key] = value
    return cfg


def _materialize_legacy_shape(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        legacy_model = dict(model_cfg)
        if legacy_model.get("dim_h") is not None:
            legacy_model["hidden_dim"] = legacy_model["dim_h"]
        if legacy_model.get("dim_z") is not None:
            legacy_model["latent_dim"] = legacy_model["dim_z"]
        legacy_model.pop("name", None)
        legacy_model.pop("dim_h", None)
        legacy_model.pop("dim_z", None)
        legacy_model.setdefault("type", str(model_cfg.get("type", "latent_ssm")))

        if "models" not in cfg or not cfg.get("models"):
            cfg["models"] = [legacy_model]

    cfg.setdefault(
        "model_io",
        {"input_groups": ["control", "exogenous", "objective"], "output_groups": ["objective"]},
    )
    cfg.setdefault("data", {})
    cfg.setdefault("training_rounds", [])
    cfg.setdefault("evaluation", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("output", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("plotting", {})
    cfg.setdefault("optimization", {})
    cfg.setdefault("model_defaults", {})
    cfg.setdefault("misc", {})
    cfg.setdefault("architecture", {})
    return cfg


def load_config(cfg_path: str | Path, cli_args: Namespace | None = None) -> Dict[str, Any]:
    """Load config via Hydra defaults composition.

    Notes:
    - ``_base`` inheritance is no longer supported.
    - Plain YAML files without ``defaults`` are still loaded directly.
    """
    path = Path(cfg_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "_base" in raw:
        raise ValueError(
            f"Config uses deprecated _base inheritance: {path}. "
            "Migrate to Hydra defaults lists under configs/experiment/."
        )

    if "defaults" in raw:
        cfg = _compose_hydra_config(path)
    else:
        cfg = raw

    cfg = _materialize_legacy_shape(cfg)
    return _apply_cli_overrides(cfg, cli_args)
