#!/usr/bin/env python3
"""Fast RSSM training smoke test for CI (5 epochs on synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.models.factory import build_model
from timesim.training import WorldModelTrainer


def _make_synth_df(n: int = 420) -> pd.DataFrame:
    t = np.arange(n, dtype=np.float32)
    control = 1.5 * np.sin(t / 18.0) + 0.2 * np.cos(t / 7.0)
    exogenous = 0.8 * np.cos(t / 22.0) + 0.1 * np.sin(t / 5.0)
    noise = 0.02 * np.sin(t / 3.0)
    objective = 0.7 * control + 0.4 * exogenous + noise
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({"u": control, "x": exogenous, "y": objective}, index=idx)


def main() -> None:
    seq_len = 24
    pred_len = 1
    groups = {"control": ["u"], "exogenous": ["x"], "objective": ["y"]}
    df = _make_synth_df()
    split = int(0.8 * len(df))
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split - seq_len :].copy()

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
            "rollout_weight": 0.0,
            "rollout_dtw_weight": 0.0,
            "kl_free_bits": 0.5,
            "kl_balance": 0.8,
            "use_kl_balancing": True,
            "use_free_bits": True,
            "grad_clip_norm": 100.0,
            "lr_warmup_steps": 25,
            "lr_min_ratio": 0.01,
            "checkpoint_metric": "open_loop_crps",
            "checkpoint_open_loop_horizon": 8,
            "checkpoint_open_loop_windows": 2,
            "checkpoint_open_loop_samples": 8,
        },
        seed=42,
    )
    train_losses, val_losses = trainer.fit(epochs=5, steps_per_epoch=5, verbose=False)

    if not np.all(np.isfinite(np.asarray(train_losses, dtype=np.float32))):
        raise RuntimeError("Train loss contains non-finite values.")
    val_arr = np.asarray([v for v in val_losses if v is not None], dtype=np.float32)
    if val_arr.size == 0 or not np.all(np.isfinite(val_arr)):
        raise RuntimeError("Validation loss contains non-finite values.")

    print(
        "CI fast training smoke test passed: "
        f"train_last={float(train_losses[-1]):.6f}, val_last={float(val_arr[-1]):.6f}"
    )


if __name__ == "__main__":
    main()

