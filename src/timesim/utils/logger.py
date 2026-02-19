from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

# Note: we *lazily* import SummaryWriter only when needed to avoid requiring
# tensorboard as a hard dependency for users who do not log.


__all__ = [
    "create_run_dir",
    "get_logger",
    "init_logging",
]


FMT = "[%(asctime)s] %(levelname)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def create_run_dir(base: str | os.PathLike = "./runs",
                   dataset: str | None = None,
                   model: str | None = None,
                   suffix: str | None = None) -> Path:
    """Create nested run directory runs/<dataset>/<model>/<timestamp>[_suffix]."""
    if dataset is None or model is None:
        raise ValueError("dataset and model names must be provided")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = ts if suffix is None else f"{ts}_{suffix}"

    run_dir = Path(base) / dataset / model / name
    run_dir.mkdir(parents=True, exist_ok=False)

    # subfolders
    (run_dir / "figs").mkdir(exist_ok=True)
    return run_dir


def get_logger(run_dir: str | os.PathLike, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to *train.log* under *run_dir* and stdout."""
    run_dir = Path(run_dir)
    logger = logging.getLogger(run_dir.name)
    # Avoid duplicate handlers when re-calling in notebooks/tests
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = logging.Formatter(FMT, datefmt=DATEFMT)

    # File handler
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def init_logging(run_dir: str | os.PathLike):
    """Convenience: create *run_dir*, logger and TensorBoard writer."""
    run_dir = Path(run_dir)
    logger = get_logger(run_dir)
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        tb_writer = SummaryWriter(log_dir=run_dir)
    except ModuleNotFoundError:
        tb_writer = None
    logger.info(f"Run directory: {run_dir}")
    return logger, tb_writer 