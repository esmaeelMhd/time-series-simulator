"""Custom callback helpers for Lightning training."""

from __future__ import annotations

import subprocess
from typing import Dict


def get_git_metadata() -> Dict[str, str]:
    """Collect lightweight git metadata for run logging."""
    meta = {"git_commit": "unknown", "git_branch": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        if commit:
            meta["git_commit"] = commit
        if branch:
            meta["git_branch"] = branch
    except Exception:
        pass
    return meta


__all__ = ["get_git_metadata"]
