"""Compatibility shim package.

Allows importing `timesim.*` when running scripts from repo root while
actual sources live in `src/timesim`.
"""

from pathlib import Path
from pkgutil import extend_path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "timesim"
__path__ = extend_path(__path__, __name__)

if _src_pkg.exists():
    __path__.append(str(_src_pkg))
