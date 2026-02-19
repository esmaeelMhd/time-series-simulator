"""Phase 5A evaluation outputs for RSSM models."""

import numpy as np
import pandas as pd

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.evaluation import (
    open_loop_evaluate,
    closed_loop_evaluate,
    interventional_suite_evaluate,
)
from timesim.models.latent_ssm import LatentSSMWorldModel
from scripts.eval_rssm_suite import _uncertainty_growth_summary


def _make_dataset(n: int = 240) -> GroupedTimeSeriesDataset:
    t = np.linspace(0.0, 20.0, n, dtype=np.float32)
    control = np.sin(t)
    exo = np.cos(0.5 * t)
    y1 = 0.7 * control + 0.2 * exo
    y2 = -0.3 * control + 0.5 * exo
    df = pd.DataFrame(
        {
            "control": control,
            "exogenous": exo,
            "objective_1": y1,
            "objective_2": y2,
        }
    )
    groups = {
        "control": ["control"],
        "exogenous": ["exogenous"],
        "objective": ["objective_1", "objective_2"],
    }
    return GroupedTimeSeriesDataset(
        df=df,
        groups=groups,
        input_groups=["control", "exogenous"],
        output_groups=["objective"],
        seq_len=16,
        pred_len=1,
        scale=True,
    )


def _make_model() -> LatentSSMWorldModel:
    # input_dim includes control + exogenous + objective columns.
    return LatentSSMWorldModel(
        input_dim=4,
        output_dim=2,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
    )


def test_open_loop_returns_per_objective_curves():
    dataset = _make_dataset()
    model = _make_model()
    curves = open_loop_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=16,
        horizon=6,
        n_windows=2,
        n_samples=4,
        device="cpu",
        denormalize=True,
        interval_levels=(0.5, 0.8, 0.9, 0.95, 0.99),
    )
    assert curves["rmse"].shape == (6,)
    assert curves["mae"].shape == (6,)
    assert curves["crps"].shape == (6,)
    assert curves["rmse_per_dim"].shape == (6, 2)
    assert curves["mae_per_dim"].shape == (6, 2)
    assert curves["crps_per_dim"].shape == (6, 2)
    assert curves["nll_per_dim"].shape == (6, 2)
    assert curves["sharpness_90"].shape == (6,)
    for lvl in (0.5, 0.8, 0.9, 0.95, 0.99):
        assert lvl in curves["coverage"]
        assert curves["coverage"][lvl].shape == (6,)


def test_closed_loop_returns_per_objective_curves():
    dataset = _make_dataset()
    model = _make_model()
    curves = closed_loop_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=16,
        horizon=6,
        n_windows=2,
        n_samples=4,
        device="cpu",
        denormalize=True,
        interval_levels=(0.5, 0.8, 0.9, 0.95, 0.99),
    )
    assert curves["rmse"].shape == (6,)
    assert curves["mae"].shape == (6,)
    assert curves["crps"].shape == (6,)
    assert curves["rmse_per_dim"].shape == (6, 2)
    assert curves["mae_per_dim"].shape == (6, 2)
    assert curves["crps_per_dim"].shape == (6, 2)
    assert curves["nll_per_dim"].shape == (6, 2)
    assert curves["sharpness_90"].shape == (6,)
    for lvl in (0.5, 0.8, 0.9, 0.95, 0.99):
        assert lvl in curves["coverage"]
        assert curves["coverage"][lvl].shape == (6,)


def test_interventional_suite_outputs():
    dataset = _make_dataset()
    model = _make_model()
    out = interventional_suite_evaluate(
        model=model,
        dataset=dataset,
        warmup_len=16,
        horizon=6,
        n_windows=2,
        n_samples=4,
        control_index=0,
        objective_index=0,
        exogenous_index=0,
        expected_direction_sign=1.0,
        direction_n_windows=3,
        device="cpu",
        denormalize=True,
    )
    assert "control_sensitivity" in out
    assert "direction_check" in out
    assert "exogenous_sensitivity" in out
    assert "control_irrelevance" in out
    assert "extreme_control" in out

    cs = out["control_sensitivity"]
    assert len(cs["window_rows"]) == 2
    assert cs["trajectory_means"]["low"].shape == (6, 2)
    assert cs["trajectory_means"]["mid"].shape == (6, 2)
    assert cs["trajectory_means"]["high"].shape == (6, 2)

    dc = out["direction_check"]
    assert len(dc["window_rows"]) == 3
    assert isinstance(dc["agreement_rate"], float)

    xs = out["exogenous_sensitivity"]
    assert len(xs["window_rows"]) == 2
    assert xs["trajectory_means"]["low"].shape == (6, 2)

    ec = out["extreme_control"]
    assert len(ec["window_rows"]) == 2


def test_uncertainty_growth_summary_flags():
    constant = _uncertainty_growth_summary(np.ones((10,), dtype=np.float32))
    assert constant["is_constant_like"]
    assert not constant["is_exploding"]

    linear = _uncertainty_growth_summary(np.linspace(0.2, 0.6, num=10, dtype=np.float32))
    assert not linear["is_constant_like"]
    assert not linear["is_exploding"]
