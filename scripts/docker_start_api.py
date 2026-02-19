#!/usr/bin/env python3
"""Container entrypoint for TimeSim API serving.

Behavior:
- If config + checkpoint exist, launch the full simulator API (scripts/serve.py).
- Otherwise, launch a minimal health API so `docker run` still starts an API server.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
import uvicorn


def _env() -> Dict[str, str]:
    return {
        "config": os.getenv("TIMESIM_CONFIG", "configs/wastewater.yaml"),
        "checkpoint": os.getenv(
            "TIMESIM_CHECKPOINT",
            "runs/wastewater/full_with_time/latent_ssm/train_checkpoint.pth",
        ),
        "host": os.getenv("TIMESIM_HOST", "0.0.0.0"),
        "port": os.getenv("PORT", os.getenv("TIMESIM_PORT", "8000")),
        "device": os.getenv("TIMESIM_DEVICE", "auto"),
        "session_ttl": os.getenv("TIMESIM_SESSION_TTL", "3600"),
        "sigma_scale": os.getenv("TIMESIM_SIGMA_SCALE", ""),
    }


def _launch_full_api(env: Dict[str, str]) -> int:
    cmd = [
        sys.executable,
        "scripts/serve.py",
        "--config",
        env["config"],
        "--checkpoint",
        env["checkpoint"],
        "--host",
        env["host"],
        "--port",
        str(env["port"]),
        "--device",
        env["device"],
        "--session-ttl",
        env["session_ttl"],
    ]
    if env["sigma_scale"].strip():
        cmd += ["--sigma-scale", env["sigma_scale"].strip()]
    return subprocess.call(cmd)


def _launch_minimal_health_api(env: Dict[str, str]) -> None:
    app = FastAPI(title="TimeSim API (No Model Loaded)", version="1.0.0")
    cfg = env["config"]
    ckpt = env["checkpoint"]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": "minimal",
            "message": "Model checkpoint not found; running health-only API.",
        }

    @app.get("/ready")
    def ready():
        return {
            "ready": False,
            "config_exists": Path(cfg).exists(),
            "checkpoint_exists": Path(ckpt).exists(),
            "config": cfg,
            "checkpoint": ckpt,
            "hint": "Mount a trained checkpoint and set TIMESIM_CONFIG/TIMESIM_CHECKPOINT.",
        }

    uvicorn.run(app, host=env["host"], port=int(env["port"]))


def main() -> None:
    env = _env()
    cfg_ok = Path(env["config"]).exists()
    ckpt_ok = Path(env["checkpoint"]).exists()
    if cfg_ok and ckpt_ok:
        raise SystemExit(_launch_full_api(env))
    _launch_minimal_health_api(env)


if __name__ == "__main__":
    main()

