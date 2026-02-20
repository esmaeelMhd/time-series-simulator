"""Streamlit dashboard launcher."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "dashboard.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
