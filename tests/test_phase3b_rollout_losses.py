import numpy as np
import pandas as pd
import pytest
import torch

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.models.latent_ssm import LatentSSMWorldModel
from timesim.training.losses import soft_dtw_distance
from timesim.training.rollout import (
    batch_rollout_padded,
    get_horizon,
    get_rollout_ramp,
    get_rollout_schedule,
)
from timesim.training.trainer import WorldModelTrainer


def _make_dataset(n: int = 260) -> GroupedTimeSeriesDataset:
    t = np.linspace(0, 18, n)
    control = np.sin(t)
    exogenous = np.cos(0.7 * t)
    output = 0.6 * control + 0.4 * exogenous
    df = pd.DataFrame(
        {
            "control": control,
            "exo": exogenous,
            "output": output,
        }
    )
    groups = {
        "control": ["control"],
        "exogenous": ["exo"],
        "objective": ["output"],
    }
    return GroupedTimeSeriesDataset(
        df,
        groups,
        ["control", "exogenous"],
        ["objective"],
        seq_len=12,
        pred_len=6,
        scale=True,
    )


def _make_prob_model() -> LatentSSMWorldModel:
    return LatentSSMWorldModel(
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
        control_dim=1,
        exogenous_dim=1,
    )


def test_get_horizon_scheduler_warmup_linear_ramp_and_context_floor() -> None:
    cfg = {
        "epochs": 10,
        "seq_len": 24,
        "rollout_warmup_fraction": 0.30,
        "rollout_max_horizon": 20,  # context floor should cap this to 8
        "min_context": 16,
    }
    horizons = [get_horizon(ep, cfg) for ep in range(1, 11)]
    ramps = [get_rollout_ramp(ep, cfg) for ep in range(1, 11)]

    assert horizons[:3] == [0, 0, 0]
    assert horizons[3] >= 1
    assert horizons[-1] == 8  # capped by context floor: 24 - 16
    assert all(h2 >= h1 for h1, h2 in zip(horizons, horizons[1:]))

    assert ramps[:3] == [0.0, 0.0, 0.0]
    assert ramps[-1] == pytest.approx(1.0)
    assert all(r2 >= r1 for r1, r2 in zip(ramps, ramps[1:]))

    horizon, context_len, ramp = get_rollout_schedule(epoch=10, cfg=cfg)
    assert horizon == 8
    assert context_len == 16
    assert ramp == pytest.approx(1.0)


def test_soft_dtw_is_differentiable() -> None:
    pred = torch.randn(2, 6, 1, requires_grad=True)
    target = torch.randn(2, 6, 1)
    dtw = soft_dtw_distance(pred, target, gamma=0.1)
    dtw.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_rollout_model_feedback_uses_prior_only_path() -> None:
    dataset = _make_dataset()
    model = _make_prob_model().eval()
    start_indices = np.array([40, 80], dtype=np.int64)
    horizons = np.array([5, 5], dtype=np.int64)

    with torch.no_grad():
        result = batch_rollout_padded(
            model=model,
            dataset=dataset,
            start_indices=start_indices,
            horizons=horizons,
            warmup_len=12,
            feedback="model",
            device="cpu",
        )

    assert "posterior_mu" in result
    assert "kl_terms" in result
    assert torch.isnan(result["posterior_mu"]).all()
    assert torch.isnan(result["kl_terms"]).all()


def test_rollout_loss_combination_and_weight_ramp_in_trainer() -> None:
    dataset = _make_dataset()
    model = _make_prob_model()
    trainer = WorldModelTrainer(
        model=model,
        dataset=dataset,
        val_dataset=dataset,
        sampling_strategy=RandomStartFixedHorizon(horizon=6),
        warmup_len=12,
        batch_size=4,
        training_mode="multi_step",
        feedback="model",
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device="cpu",
        early_stopping=False,
        run_dir=None,
        probabilistic_cfg={
            "recon_weight": 1.0,
            "kl_weight": 1.0,
            "aux_weight": 1.0,
            "rollout_weight": 0.8,
            "rollout_dtw_weight": 0.5,
            "rollout_dtw_gamma": 0.1,
            "rollout_warmup_fraction": 0.30,
            "rollout_max_horizon": 4,
            "rollout_start_epoch": 4,
            "rollout_full_epoch": 10,
            "min_context": 6,
            "checkpoint_metric": "open_loop_crps",
            "checkpoint_open_loop_horizon": 4,
            "checkpoint_open_loop_windows": 2,
            "checkpoint_open_loop_samples": 4,
        },
    )
    trainer._fit_epochs = 10
    trainer._current_epoch = 10

    train_loss = trainer._train_step()
    assert np.isfinite(train_loss)
    info = trainer._last_prob_info

    assert info["horizon_schedule"] >= 1.0
    assert info["rollout_ramp"] == pytest.approx(1.0)
    assert info["rollout_weight_eff"] == pytest.approx(0.8)
    assert "rollout_nll" in info
    assert "rollout_dtw" in info
    assert "rollout_total" in info
    assert info["rollout_total"] == pytest.approx(
        info["rollout_nll"] + 0.5 * info["rollout_dtw"],
        rel=1e-5,
        abs=1e-5,
    )
