"""CLI entry point for training: ``timesim-train``."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from train import main as _train_main  # noqa: E402
    _train_main()


if __name__ == "__main__":
    main()
