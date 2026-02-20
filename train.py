#!/usr/bin/env python3
"""Root training entrypoint.

Dispatches to:
- `scripts/train_hydra.py` when Hydra-style key=value overrides are used
- `scripts/train.py` for legacy argparse workflow
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _is_hydra_arg(arg: str) -> bool:
    if arg.startswith("-"):
        return False
    return "=" in arg


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    args = sys.argv[1:]
    use_hydra = any(_is_hydra_arg(a) for a in args)
    target = repo_root / "scripts" / ("train_hydra.py" if use_hydra else "train.py")
    cmd = [sys.executable, str(target), *args]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
