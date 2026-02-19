from __future__ import annotations

import numpy as np
import pandas as pd

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.evaluation import open_loop_evaluate
from timesim.models.factory import build_model
from timesim.serving.simulator import RSSMSimulator
from timesim.training import WorldModelTrainer


def _make_synth_df(n: int = 520) -> pd.DataFrame:
    t = np.arange(n, dtype=np.float32)
    u = 1.0 * np.sin(t / 16.0) + 0.15 * np.cos(t / 5.0)
    x = 0.9 * np.cos(t / 21.0) + 0.05 * np.sin(t / 4.0)
    y = 0.8 * u + 0.35 * x + 0.03 * np.sin(t / 9.0)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({"u": u, "x": x, "y": y}, index=idx)


def test_phase9_end_to_end_synthetic_train_eval_simulator():
    seq_len = 24
    pred_len = 1
    groups = {"control": ["u"], "exogenous": ["x"], "objective": ["y"]}
    df = _make_synth_df()

    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end - seq_len : val_end].copy()
    test_df = df.iloc[val_end - seq_len :].copy()

    train_ds = GroupedTimeSeriesDataset(
        df=train_df,
        groups=groups,
        input_groups=["control", "exogenous", "objective"],
        output_groups=["objective"],
        seq_len=seq_len,
        pred_len=pred_len,
        scale=True,
    )
    val_ds = GroupedTimeSeriesDataset(
        df=val_df,
        groups=groups,
        input_groups=["control", "exogenous", "objective"],
        output_groups=["objective"],
        seq_len=seq_len,
        pred_len=pred_len,
        scale=True,
        scaler=train_ds.scaler,
    )
    test_ds = GroupedTimeSeriesDataset(
        df=test_df,
        groups=groups,
        input_groups=["control", "exogenous", "objective"],
        output_groups=["objective"],
        seq_len=seq_len,
        pred_len=pred_len,
        scale=True,
        scaler=train_ds.scaler,
    )

    input_dim = len(set(train_ds.input_cols) | set(train_ds.output_cols))
    output_dim = len(train_ds.output_cols)
    model = build_model(
        "latent_ssm",
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        per_model_cfg={
            "type": "latent_ssm",
            "hidden_dim": 32,
            "latent_dim": 12,
            "num_layers": 1,
            "dropout": 0.0,
            "use_aux_decoder": True,
            "use_dual_path": True,
            "leak_objective_to_transition": False,
        },
        model_defaults_cfg={},
    )

    trainer = WorldModelTrainer(
        model=model,
        dataset=train_ds,
        val_dataset=val_ds,
        sampling_strategy=RandomStartFixedHorizon(horizon=8),
        warmup_len=seq_len,
        batch_size=16,
        training_mode="multi_step",
        feedback="model",
        device="cpu",
        use_amp=False,
        early_stopping=False,
        probabilistic_cfg={
            "objective": "rssm",
            "recon_weight": 1.0,
            "kl_weight": 0.2,
            "aux_weight": 1.0,
            "rollout_weight": 0.2,
            "rollout_dtw_weight": 0.0,
            "rollout_warmup_fraction": 0.30,
            "rollout_max_horizon": 8,
            "min_context": 16,
            "kl_free_bits": 0.5,
            "kl_balance": 0.8,
            "use_kl_balancing": True,
            "use_free_bits": True,
            "grad_clip_norm": 100.0,
            "lr_warmup_steps": 20,
            "lr_min_ratio": 0.01,
            "checkpoint_metric": "open_loop_crps",
            "checkpoint_open_loop_horizon": 10,
            "checkpoint_open_loop_windows": 2,
            "checkpoint_open_loop_samples": 8,
        },
        seed=42,
    )
    train_losses, val_losses = trainer.fit(epochs=20, steps_per_epoch=2, verbose=False)

    assert np.isfinite(float(train_losses[-1]))
    assert val_losses[-1] is not None and np.isfinite(float(val_losses[-1]))

    curves = open_loop_evaluate(
        model=model,
        dataset=test_ds,
        warmup_len=seq_len,
        horizon=10,
        n_windows=3,
        n_samples=8,
        device="cpu",
    )
    rmse = np.asarray(curves["rmse"], dtype=np.float32)
    assert rmse.shape == (10,)
    assert np.all(np.isfinite(rmse))

    sim = RSSMSimulator.from_dataset(model=model, dataset=test_ds, sigma_scale=1.0, device="cpu")
    hist_df = test_df[test_ds.feature_cols].iloc[:seq_len].copy()
    sim.reset(hist_df)
    control_dict = {"u": float(hist_df["u"].iloc[-1])}
    exogenous_dict = {"x": float(hist_df["x"].iloc[-1])}

    step_pred = sim.step(control_dict, exogenous_dict, n_samples=8)
    assert "y" in step_pred
    assert isinstance(step_pred["y"]["mean"], float)
    assert isinstance(step_pred["y"]["std"], float)

    horizon = 6
    c_df = pd.DataFrame({"u": np.full((horizon,), control_dict["u"], dtype=np.float32)})
    x_df = pd.DataFrame({"x": np.full((horizon,), exogenous_dict["x"], dtype=np.float32)})
    rollout_df = sim.rollout(c_df, x_df, n_samples=8)
    assert rollout_df.shape[0] == horizon
    assert {"y_mean", "y_std", "y_p5", "y_p95"}.issubset(set(rollout_df.columns))
    assert np.issubdtype(rollout_df["y_mean"].dtype, np.floating)

