#!/usr/bin/env python3
"""Wrapper for the packaged Optuna CLI.

Use either:
  - python scripts/optimize.py ...
  - timesim-optimize ...
"""

from pathlib import Path
import sys

# Ensure local package import works when called as:
#   python scripts/optimize.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timesim.cli.optimize import main


if __name__ == "__main__":
    main()
