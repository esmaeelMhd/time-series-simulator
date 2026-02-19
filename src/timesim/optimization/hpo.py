"""Optuna HPO wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_optuna_search(config_path: str | Path) -> None:
    """Run optimization using the existing CLI implementation."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "scripts" / "optimize.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, check=True)


__all__ = ["run_optuna_search"]
