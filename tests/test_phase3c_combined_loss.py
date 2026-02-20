import numpy as np
import pandas as pd
import pytest
import torch

import timesim.training.trainer as trainer_mod
from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.models.latent_ssm import LatentSSMWorldModel
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


def _make_trainer(dataset: GroupedTimeSeriesDataset) -> WorldModelTrainer:
    model = LatentSSMWorldModel(
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
        control_dim=1,
        exogenous_dim=1,
    )
    return WorldModelTrainer(
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
            "min_context": 6,
            "checkpoint_metric": "open_loop_crps",
            "checkpoint_open_loop_horizon": 4,
            "checkpoint_open_loop_windows": 2,
            "checkpoint_open_loop_samples": 4,
        },
    )


def test_combined_loss_skips_rollout_compute_when_horizon_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    trainer = _make_trainer(dataset)
    trainer._fit_epochs = 10
    trainer._current_epoch = 1  # warmup epoch => horizon=0

    call_counts = {"teacher": 0, "model": 0}
    original = trainer_mod.batch_rollout_padded

    def _wrapped_batch_rollout_padded(*args, **kwargs):
        feedback = kwargs.get("feedback")
        if feedback in call_counts:
            call_counts[feedback] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "batch_rollout_padded", _wrapped_batch_rollout_padded)

    step_loss = trainer._train_step()
    info = trainer._last_prob_info

    assert np.isfinite(step_loss)
    assert call_counts["teacher"] == 1
    assert call_counts["model"] == 0  # rollout not computed during warmup
    assert info["horizon_schedule"] == 0.0
    assert info["rollout_computed"] == 0.0
    assert info["rollout_nll"] == pytest.approx(0.0)
    assert info["rollout_dtw"] == pytest.approx(0.0)
    assert info["rollout_total"] == pytest.approx(0.0)
    assert info["loss_total"] == pytest.approx(info["loss_std"], rel=1e-6, abs=1e-6)


def test_combined_loss_logs_all_terms_and_formula() -> None:
    dataset = _make_dataset()
    trainer = _make_trainer(dataset)
    trainer._fit_epochs = 10
    trainer._current_epoch = 10  # post-warmup => rollout active

    step_loss = trainer._train_step()
    info = trainer._last_prob_info

    required_keys = [
        "recon_nll",
        "kl",
        "aux_nll",
        "rollout_nll",
        "rollout_dtw",
        "rollout_total",
        "loss_std",
        "loss_total",
    ]
    for key in required_keys:
        assert key in info
        assert np.isfinite(info[key])

    expected_total = info["loss_std"] + info["rollout_weight_eff"] * info["rollout_total"]
    assert info["loss_total"] == pytest.approx(expected_total, rel=1e-5, abs=1e-5)
    assert step_loss == pytest.approx(info["loss_total"], rel=1e-5, abs=1e-5)
