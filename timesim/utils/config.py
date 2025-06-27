from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def load_config(cfg_path: str | Path, cli_args: Namespace | None = None) -> Dict[str, Any]:
    """Load a YAML config file and merge *non-None* values coming from *cli_args*.

    Parameters
    ----------
    cfg_path: Path to a YAML file.
    cli_args: An ``argparse.Namespace`` produced from the CLI parser.  Every
        attribute that is not ``None`` will override the corresponding key in
        the YAML config (at the top level).

    Returns
    -------
    dict
        A plain Python dictionary with the merged configuration.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load with OmegaConf – it gives us attribute access & easy merging
    file_cfg = OmegaConf.load(cfg_path)
    merged = OmegaConf.to_container(file_cfg, resolve=True)  # type: ignore[arg-type]

    if cli_args is not None:
        for key, value in vars(cli_args).items():
            # If the parser didn't set the value, skip it so the YAML one wins
            if value is None:
                continue
            merged[key] = value

    return merged 