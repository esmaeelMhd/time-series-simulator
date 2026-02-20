"""FastAPI serving utilities for stateful RSSM simulation."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..simulator import RSSMSimulator

logger = logging.getLogger(__name__)


class ResetRequest(BaseModel):
    historical: list[dict[str, float]]


class StepRequest(BaseModel):
    session_id: str
    controls: Mapping[str, float]
    exogenous: Optional[Mapping[str, float]] = None
    n_samples: int = Field(default=50, ge=1, le=512)


class RolloutRequest(BaseModel):
    session_id: str
    controls: list[dict[str, float]]
    exogenous: Optional[list[dict[str, float]]] = None
    n_samples: int = Field(default=50, ge=1, le=512)


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = int(ttl_seconds)
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _cleanup(self):
        now = time.time()
        expired = [
            sid
            for sid, obj in self._sessions.items()
            if now - float(obj.get("updated_at", now)) > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def put(self, simulator: RSSMSimulator) -> str:
        self._cleanup()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "sim": simulator,
            "updated_at": time.time(),
        }
        return session_id

    def get(self, session_id: str) -> RSSMSimulator:
        self._cleanup()
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found or expired")
        session["updated_at"] = time.time()
        return session["sim"]

    def count(self) -> int:
        self._cleanup()
        return int(len(self._sessions))


def create_app(simulator_template: RSSMSimulator, session_ttl_seconds: int = 3600) -> FastAPI:
    """Create FastAPI app exposing reset/step/rollout APIs."""
    store = SessionStore(ttl_seconds=session_ttl_seconds)
    app = FastAPI(title="TimeSim RSSM Simulator API", version="1.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "sessions": store.count(),
            "session_ttl_seconds": int(session_ttl_seconds),
        }

    @app.get("/schema")
    def schema():
        return simulator_template.schema()

    @app.post("/reset")
    def reset(req: ResetRequest):
        if not req.historical:
            raise HTTPException(status_code=400, detail="historical must contain at least one row")
        try:
            sim = simulator_template.clone_empty()
            hist_df = pd.DataFrame(req.historical)
            sim.reset(hist_df)
            sid = store.put(sim)
            return {
                "session_id": sid,
                "history_len": int(sim.last_reset_history_len),
                "expires_in_seconds": int(session_ttl_seconds),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/step")
    def step(req: StepRequest):
        try:
            sim = store.get(req.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        t0 = time.perf_counter()
        try:
            out = sim.step(
                control_values=req.controls,
                exogenous_values=req.exogenous,
                n_samples=req.n_samples,
                return_details=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("step latency_ms=%.3f session=%s", elapsed_ms, req.session_id)
            return {
                "predictions": out["predictions"],
                "warnings": list(out.get("warnings", [])),
                "latency_ms": float(out.get("latency_ms", elapsed_ms)),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/rollout")
    def rollout(req: RolloutRequest):
        try:
            sim = store.get(req.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        t0 = time.perf_counter()
        try:
            out = sim.rollout(
                control_trajectory=pd.DataFrame(req.controls),
                exogenous_trajectory=(
                    pd.DataFrame(req.exogenous)
                    if req.exogenous is not None
                    else None
                ),
                n_samples=req.n_samples,
                return_details=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("rollout latency_ms=%.3f session=%s", elapsed_ms, req.session_id)
            pred_df = out["predictions"]
            if not isinstance(pred_df, pd.DataFrame):
                raise RuntimeError("Simulator rollout returned invalid predictions payload.")
            return {
                "predictions": pred_df.to_dict(orient="records"),
                "n_steps": int(len(pred_df)),
                "warnings": list(out.get("warnings", [])),
                "latency_ms": float(out.get("latency_ms", elapsed_ms)),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

