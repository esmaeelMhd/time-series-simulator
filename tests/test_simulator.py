from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from timesim.data.dataset import GroupedTimeSeriesDataset
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
        batch, horizon, _ = future_controls.shape
        controls_term = future_controls.mean(dim=-1, keepdim=True)
        exogenous_term = (
            future_exogenous.mean(dim=-1, keepdim=True)
            if future_exogenous.shape[-1] > 0
            else torch.zeros_like(controls_term)
        )
        drift = torch.cumsum(0.03 * controls_term + 0.01 * exogenous_term, dim=1)
        base = initial_state.z[:, : self.output_dim].unsqueeze(1)
        pred = base + drift

        next_state = _DummyState(
            h=initial_state.h + 0.01,
            z=initial_state.z + 0.02,
        )

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


def _make_dataset(n: int = 240, seq_len: int = 24) -> tuple[pd.DataFrame, GroupedTimeSeriesDataset]:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    control = 20.0 + 5.0 * np.sin(np.linspace(0, 8, n))
    exogenous = 100.0 + 10.0 * np.cos(np.linspace(0, 8, n))
    objective = 0.4 * control + 0.06 * exogenous + 3.0
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
    return df, dataset


def test_simulator_reset_step_rollout_and_warnings():
    df, dataset = _make_dataset()
    model = _DummyWorldModel(output_dim=1)
    sim = RSSMSimulator.from_dataset(model=model, dataset=dataset, sigma_scale=1.0, device="cpu")

    history_df = df[dataset.feature_cols].iloc[-dataset.seq_len :].copy()
    assert sim.reset(history_df) is sim

    control_dict = {"u": float(history_df["u"].iloc[-1])}
    exogenous_dict = {"x": float(history_df["x"].iloc[-1])}
    pred = sim.step(control_dict, exogenous_dict, n_samples=32)
    assert "y" in pred
    assert "mean" in pred["y"] and "std" in pred["y"]

    values = []
    for _ in range(50):
        step_pred = sim.step(control_dict, exogenous_dict, n_samples=16)
        values.append(float(step_pred["y"]["mean"]))
    assert len(set(np.round(np.asarray(values), 6).tolist())) > 1

    horizon = 12
    control_df = pd.DataFrame({"u": np.full((horizon,), control_dict["u"], dtype=np.float32)})
    exogenous_df = pd.DataFrame({"x": np.full((horizon,), exogenous_dict["x"], dtype=np.float32)})
    rollout_df = sim.rollout(control_df, exogenous_df, n_samples=32)
    assert isinstance(rollout_df, pd.DataFrame)
    assert len(rollout_df) == horizon
    assert all(c in rollout_df.columns for c in ["y_mean", "y_std", "y_p5", "y_p95"])

    # Spot-check original scale (not normalized [0,1] range).
    assert float(rollout_df["y_mean"].mean()) > 1.0

    extreme = 5.0 * float(df["u"].max())
    step_out = sim.step({"u": extreme}, exogenous_dict, n_samples=8, return_details=True)
    warnings = step_out.get("warnings", [])
    assert any("u" in w and "sigma" in w for w in warnings)
