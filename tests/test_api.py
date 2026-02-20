from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.serving.api import create_app
from timesim.serving.simulator import RSSMSimulator


class _DummyState:
    def __init__(self, h: torch.Tensor, z: torch.Tensor):
        self.h = h
        self.z = z


class _DummyWorldModel(torch.nn.Module):
    def __init__(self, output_dim: int = 1, hidden_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

    def observe(self, controls, exogenous, observations, sample_posterior=False):
        batch = controls.shape[0]
        dtype = controls.dtype
        device = controls.device
        h = torch.zeros(batch, self.hidden_dim, dtype=dtype, device=device)
        base = observations[:, -1, : self.output_dim]
        z = base.repeat(1, max(1, self.latent_dim // self.output_dim + 1))[:, : self.latent_dim]
        return {"state": _DummyState(h=h, z=z)}

    def imagine(
        self,
        initial_state: _DummyState,
        future_controls: torch.Tensor,
        future_exogenous: torch.Tensor,
        n_steps=None,
        n_samples: int = 50,
        sample_latent: bool = True,
    ):
        controls_term = future_controls.mean(dim=-1, keepdim=True)
        exogenous_term = (
            future_exogenous.mean(dim=-1, keepdim=True)
            if future_exogenous.shape[-1] > 0
            else torch.zeros_like(controls_term)
        )
        drift = torch.cumsum(0.03 * controls_term + 0.01 * exogenous_term, dim=1)
        base = initial_state.z[:, : self.output_dim].unsqueeze(1)
        pred = base + drift
        next_state = _DummyState(h=initial_state.h + 0.01, z=initial_state.z + 0.02)

        n_samples = max(1, int(n_samples))
        if n_samples > 1:
            offsets = torch.linspace(
                -0.2,
                0.2,
                steps=n_samples,
                device=future_controls.device,
                dtype=future_controls.dtype,
            ).view(n_samples, 1, 1, 1)
            samples = pred.unsqueeze(0) + offsets
            return {"samples": samples, "predictions": pred, "state": next_state}
        return {"predictions": pred, "state": next_state}


def _make_simulator() -> tuple[RSSMSimulator, pd.DataFrame, GroupedTimeSeriesDataset]:
    n = 256
    seq_len = 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    control = 20.0 + 4.0 * np.sin(np.linspace(0, 9, n))
    exogenous = 60.0 + 8.0 * np.cos(np.linspace(0, 9, n))
    objective = 0.4 * control + 0.08 * exogenous + 2.0
    df = pd.DataFrame(
        {
            "u": control.astype(np.float32),
            "x": exogenous.astype(np.float32),
            "y": objective.astype(np.float32),
        },
        index=idx,
    )
    groups = {"control": ["u"], "exogenous": ["x"], "objective": ["y"]}
    dataset = GroupedTimeSeriesDataset(
        df=df,
        groups=groups,
        input_groups=["control", "exogenous"],
        output_groups=["objective"],
        seq_len=seq_len,
        pred_len=1,
        scale=True,
    )
    model = _DummyWorldModel(output_dim=1)
    sim = RSSMSimulator.from_dataset(model=model, dataset=dataset, sigma_scale=1.0, device="cpu")
    return sim, df, dataset


def test_api_health_and_schema():
    simulator_template, _, _ = _make_simulator()
    client = TestClient(create_app(simulator_template, session_ttl_seconds=60))

    health = client.get("/health")
    assert health.status_code == 200
    health_json = health.json()
    assert health_json["status"] == "ok"
    assert health_json["sessions"] == 0

    schema = client.get("/schema")
    assert schema.status_code == 200
    schema_json = schema.json()
    assert schema_json["groups"]["control"] == ["u"]
    assert schema_json["groups"]["exogenous"] == ["x"]
    assert schema_json["groups"]["objective"] == ["y"]


def test_api_reset_step_rollout_flow():
    simulator_template, df, dataset = _make_simulator()
    client = TestClient(create_app(simulator_template, session_ttl_seconds=60))

    history_df = df[dataset.feature_cols].iloc[-dataset.seq_len :].copy()
    reset = client.post("/reset", json={"historical": history_df.to_dict(orient="records")})
    assert reset.status_code == 200
    reset_json = reset.json()
    assert "session_id" in reset_json
    session_id = reset_json["session_id"]

    step = client.post(
        "/step",
        json={
            "session_id": session_id,
            "controls": {"u": float(history_df["u"].iloc[-1])},
            "exogenous": {"x": float(history_df["x"].iloc[-1])},
            "n_samples": 16,
        },
    )
    assert step.status_code == 200
    step_json = step.json()
    assert "predictions" in step_json
    assert "y" in step_json["predictions"]
    assert "mean" in step_json["predictions"]["y"]
    assert "std" in step_json["predictions"]["y"]

    horizon = 10
    ctrl = [{"u": float(history_df["u"].iloc[-1])} for _ in range(horizon)]
    exo = [{"x": float(history_df["x"].iloc[-1])} for _ in range(horizon)]
    rollout = client.post(
        "/rollout",
        json={"session_id": session_id, "controls": ctrl, "exogenous": exo, "n_samples": 16},
    )
    assert rollout.status_code == 200
    rollout_json = rollout.json()
    assert rollout_json["n_steps"] == horizon
    assert len(rollout_json["predictions"]) == horizon
    assert {"y_mean", "y_std", "y_p5", "y_p95"}.issubset(rollout_json["predictions"][0].keys())

    missing_session = client.post(
        "/step",
        json={"session_id": "missing", "controls": {"u": 0.0}, "exogenous": {"x": 0.0}},
    )
    assert missing_session.status_code == 404
