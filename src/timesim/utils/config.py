"""Hierarchical YAML config loader with ``_base`` chain support.

Usage::

    from timesim.utils.config import load_config

    # Loads default.yaml → wastewater.yaml → wastewater.small.yaml
    config = load_config("configs/wastewater.small.yaml")
"""

from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


__all__ = ["load_config", "deep_merge"]


# ─────────────────────────────────────────────────────────────────────
# Deep merge
# ─────────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    - Dicts are merged recursively (keys in override win).
    - Lists and scalars in *override* **replace** those in *base*.
    - *base* is not mutated; a new dict is returned.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ─────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────

def load_config(
    cfg_path: str | Path,
    cli_args: Namespace | None = None,
) -> Dict[str, Any]:
    """Load a YAML config with optional ``_base`` chain resolution.

    Each config file may contain a ``_base`` key pointing to a parent
    config (path relative to the file's own directory).  The loader
    walks up the chain, deep-merging each layer on top of its parent::

        default.yaml                 # base defaults
          ← wastewater.yaml          # _base: default.yaml
            ← wastewater.small.yaml  # _base: wastewater.yaml

    After the chain is resolved, *cli_args* (if given) are applied as
    top-level overrides (non-None values only).

    Parameters
    ----------
    cfg_path : str or Path
        Path to a YAML config file.
    cli_args : argparse.Namespace, optional
        CLI overrides.  Non-None attributes replace top-level keys.

    Returns
    -------
    dict
        Fully resolved configuration as a plain Python dict.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ── Resolve _base chain ──────────────────────────────────────────
    base_ref = cfg.pop("_base", None)
    if base_ref is not None:
        base_path = cfg_path.parent / base_ref
        base_cfg = load_config(base_path)   # recursive
        cfg = deep_merge(base_cfg, cfg)

    # ── Apply CLI overrides ──────────────────────────────────────────
    if cli_args is not None:
        for key, value in vars(cli_args).items():
            if value is None:
                continue
            cfg[key] = value

    return cfg
