"""Hydra-based config composition for all CLI scripts.

Every script — training, evaluation, optimization, serving — loads config
through :func:`compose_config`, which delegates to ``hydra.compose()``.
"""

from __future__ import annotations

import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

__all__ = ["compose_config", "load_config"]

_REPO_CONFIGS_DIR: Optional[Path] = None


def _default_configs_dir() -> Path:
    """Resolve the canonical ``configs/`` directory relative to the repo root."""
    global _REPO_CONFIGS_DIR
    if _REPO_CONFIGS_DIR is not None:
        return _REPO_CONFIGS_DIR
    # Walk upwards from this file to find the repo root (contains configs/).
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if candidate.is_dir() and (candidate / "config.yaml").exists():
            _REPO_CONFIGS_DIR = candidate
            return _REPO_CONFIGS_DIR
    raise FileNotFoundError(
        "Cannot locate configs/ directory. Pass config_dir explicitly."
    )


def _resolve_config_name(config: str, config_dir: Path) -> str:
    """Accept either a Hydra config name or a file path and return the name.

    Examples::

        "wastewater.small"                     -> "wastewater.small"
        "configs/wastewater.small.yaml"        -> "wastewater.small"
        "/abs/path/configs/wastewater.small.yaml" -> "wastewater.small"
    """
    if "/" not in config and "\\" not in config and not config.endswith((".yaml", ".yml")):
        return config

    path = Path(config).resolve()
    if not path.exists():
        path = (Path.cwd() / config).resolve()

    try:
        return path.relative_to(config_dir).with_suffix("").as_posix()
    except ValueError:
        pass

    for parent in [path.parent, *path.parents]:
        if parent.name == "configs":
            return path.relative_to(parent).with_suffix("").as_posix()

    return path.stem


def _materialize_legacy_shape(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in default sections and build the ``models`` list from ``model``."""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        legacy_model = dict(model_cfg)
        legacy_model.pop("name", None)
        legacy_model.setdefault("type", str(model_cfg.get("type", "latent_ssm")))

        if "models" not in cfg or not cfg.get("models"):
            cfg["models"] = [legacy_model]

    cfg.setdefault(
        "model_io",
        {"input_groups": ["control", "exogenous", "objective"], "output_groups": ["objective"]},
    )
    for section in (
        "data", "training_rounds", "evaluation", "simulation", "output",
        "tracking", "plotting", "optimization", "model_defaults", "misc",
        "architecture",
    ):
        if section == "training_rounds":
            cfg.setdefault(section, [])
        else:
            cfg.setdefault(section, {})
    return cfg


def _scan_deprecated_patterns(obj: Any, path: str = "") -> List[str]:
    """Collect deprecated config key usages with dotted paths."""
    hits: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            child_path = f"{path}.{key_s}" if path else key_s
            if key_s == "_base":
                hits.append(child_path)
            if key_s in {"dim_h", "dim_z"}:
                hits.append(child_path)
            hits.extend(_scan_deprecated_patterns(value, child_path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            hits.extend(_scan_deprecated_patterns(value, child_path))
    return hits


def _validate_no_deprecated_patterns(cfg: Dict[str, Any]) -> None:
    """Fail fast when deprecated config patterns are present."""
    hits = _scan_deprecated_patterns(cfg)
    if not hits:
        return
    uniq_hits = sorted(set(hits))
    msg = (
        "Deprecated config patterns detected and rejected.\n"
        "Replace these keys/patterns:\n"
        "  - `_base` (unsupported)\n"
        "  - `dim_h` -> `hidden_dim`\n"
        "  - `dim_z` -> `latent_dim`\n"
        f"Locations: {uniq_hits}"
    )
    raise ValueError(msg)


def compose_config(
    config: str,
    overrides: Sequence[str] | None = None,
    config_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Load and compose a Hydra config, returning a plain dict.

    Parameters
    ----------
    config:
        Either a Hydra config name (``"wastewater.small"``) or a file path
        (``"configs/wastewater.small.yaml"``).  Both forms are accepted for
        backward compatibility.
    overrides:
        Optional Hydra-style ``key=value`` overrides, e.g.
        ``["misc.device=cpu", "evaluation.horizon=24"]``.
    config_dir:
        Absolute path to the ``configs/`` directory.  Auto-resolved from the
        repository root when *None*.

    Returns
    -------
    dict
        Fully composed and resolved configuration dictionary.
    """
    cfg_dir = Path(config_dir).resolve() if config_dir else _default_configs_dir()
    config_name = _resolve_config_name(config, cfg_dir)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    result: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    _validate_no_deprecated_patterns(result)
    return _materialize_legacy_shape(result)


# ------------------------------------------------------------------
# Deprecated compat shim
# ------------------------------------------------------------------

def load_config(cfg_path: str | Path, cli_args: Namespace | None = None) -> Dict[str, Any]:
    """Load config via Hydra defaults composition.

    .. deprecated::
        Use :func:`compose_config` instead.  This wrapper exists only for
        backward compatibility with code that has not been migrated yet.
    """
    warnings.warn(
        "load_config() is deprecated. Use compose_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cfg = compose_config(str(cfg_path))

    if cli_args is not None:
        for key, value in vars(cli_args).items():
            if value is not None:
                cfg[key] = value
    return cfg
