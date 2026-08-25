from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from timesim.control.mpc import CEMController, _resolve_best_checkpoint
from timesim.models.rssm import RSSMState


class _DummyLatentModel(torch.nn.Module):
    def __init__(self, control_dim: int = 3, exogenous_dim: int = 4, output_dim: int = 1):
        super().__init__()
        self.control_dim = int(control_dim)
        self.exogenous_dim = int(exogenous_dim)
        self.output_dim = int(output_dim)
        self.proj = torch.nn.Linear(self.control_dim + self.exogenous_dim, self.output_dim)

    def imagine(
        self,
        initial_state: RSSMState,
        future_controls: torch.Tensor,
        future_exogenous: torch.Tensor,
        n_steps: int | None = None,
        n_samples: int = 1,
        sample_latent: bool = False,
    ):
        _ = initial_state, n_steps, n_samples, sample_latent
        x = torch.cat([future_controls, future_exogenous], dim=-1)
        return {"predictions": self.proj(x)}


def test_cem_optimize_shapes_and_bounds():
    torch.manual_seed(0)
    model = _DummyLatentModel(control_dim=2, exogenous_dim=3, output_dim=1)
    controller = CEMController(
        model=model,
        horizon=6,
        action_dim=2,
        exogenous_dim=3,
        population=64,
        iterations=3,
        elite_frac=0.25,
        action_low=[-0.5, -0.2],
        action_high=[0.5, 0.2],
    )

    h = torch.zeros(1, 8)
    z = torch.zeros(1, 4)
    future_x = torch.zeros(6, 3)

    def cost_fn(y_preds: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Penalize action magnitude and output magnitude.
        return actions.square().mean(dim=(1, 2)) + y_preds.square().mean(dim=(1, 2))

    out = controller.optimize((h, z), future_x, cost_fn)

    assert out["action_t0"].shape == (2,)
    assert out["trajectory"].shape == (6, 1)
    assert out["best_actions"].shape == (6, 2)
    assert out["mean"].shape == (6, 2)
    assert out["std"].shape == (6, 2)
    assert np.isfinite(out["best_cost"])
    assert torch.all(out["best_actions"][:, 0] >= -0.5)
    assert torch.all(out["best_actions"][:, 0] <= 0.5)
    assert torch.all(out["best_actions"][:, 1] >= -0.2)
    assert torch.all(out["best_actions"][:, 1] <= 0.2)


def test_cem_cost_contract_validation():
    model = _DummyLatentModel(control_dim=1, exogenous_dim=1, output_dim=1)
    controller = CEMController(
        model=model,
        horizon=4,
        action_dim=1,
        exogenous_dim=1,
        population=16,
        iterations=1,
    )
    h = torch.zeros(1, 4)
    z = torch.zeros(1, 2)
    future_x = torch.zeros(4, 1)

    def bad_cost(y_preds: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _ = y_preds, actions
        return torch.tensor(0.0)  # wrong shape

    try:
        controller.optimize({"h": h, "z": z}, future_x, bad_cost)
        assert False, "Expected ValueError for invalid cost shape"
    except ValueError as exc:
        assert "shape [N]" in str(exc)


def test_cem_state_adapter_variants():
    model = _DummyLatentModel(control_dim=1, exogenous_dim=1, output_dim=1)
    controller = CEMController(
        model=model,
        horizon=2,
        action_dim=1,
        exogenous_dim=1,
        population=8,
        iterations=1,
    )
    h = torch.zeros(1, 5)
    z = torch.zeros(1, 3)
    future_x = torch.zeros(2, 1)

    def cost(y, a):
        return y.square().mean(dim=(1, 2)) + a.square().mean(dim=(1, 2))

    out1 = controller.optimize(RSSMState(h=h, z=z), future_x, cost)
    out2 = controller.optimize((h, z), future_x, cost)
    out3 = controller.optimize({"h": h, "z": z}, future_x, cost)
    assert out1["trajectory"].shape == out2["trajectory"].shape == out3["trajectory"].shape


def test_resolve_best_checkpoint_prefers_best_metric_epoch(tmp_path: Path):
    model_dir = tmp_path / "latent_ssm"
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "open_loop_crps": [0.4, 0.2, 0.3],
        }
    ).to_csv(model_dir / "metrics.csv", index=False)

    (ckpt_dir / "epoch000.ckpt").write_bytes(b"x")
    (ckpt_dir / "epoch001.ckpt").write_bytes(b"x")
    (ckpt_dir / "epoch002.ckpt").write_bytes(b"x")

    selected = _resolve_best_checkpoint(model_dir)
    assert selected.path.name == "epoch001.ckpt"
    assert selected.metric_name == "open_loop_crps"
    assert selected.epoch_1based == 2


def test_from_run_dir_loads_model_and_resolver(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "runs" / "wastewater" / "demo"
    model_dir = run_dir / "latent_ssm"
    model_dir.mkdir(parents=True)

    cfg = {
        "resolved": {
            "data": {
                "input_dim": 10,
                "output_dim": 1,
                "seq_len": 128,
                "pred_len": 1,
                "variable_schema": {"control": ["c1", "c2"], "exogenous": ["x1", "x2", "x3"]},
            },
            "model_params": {"hidden_dim": 8, "latent_dim": 4},
        }
    }
    with open(model_dir / "model_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    ckpt = model_dir / "train_checkpoint.pth"
    torch.save({"model_state_dict": {}}, ckpt)

    built_model = _DummyLatentModel(control_dim=2, exogenous_dim=3, output_dim=1)

    import timesim.control.mpc as mpc_mod

    monkeypatch.setattr(mpc_mod, "build_model", lambda **kwargs: built_model)
    monkeypatch.setattr(mpc_mod, "_load_model_state", lambda model, checkpoint_path, device: None)

    controller = CEMController.from_run_dir(run_dir, horizon=5, device="cpu")
    assert isinstance(controller, CEMController)
    assert controller.action_dim == 2
    assert controller.exogenous_dim == 3
    assert controller.horizon == 5

