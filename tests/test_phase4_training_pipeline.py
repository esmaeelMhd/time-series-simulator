import numpy as np
import pandas as pd
import torch

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.models.latent_ssm import LatentSSMWorldModel
from timesim.training.trainer import WorldModelTrainer


def _make_dataset(n: int = 220) -> GroupedTimeSeriesDataset:
    t = np.linspace(0, 16, n)
    df = pd.DataFrame(
        {
            "control": np.sin(t),
            "exo": np.cos(0.7 * t),
            "output": 0.6 * np.sin(t) + 0.4 * np.cos(0.7 * t),
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


def _make_model() -> LatentSSMWorldModel:
    return LatentSSMWorldModel(
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        latent_dim=8,
        control_dim=1,
        exogenous_dim=1,
    )


def test_trainer_defaults_match_phase4_requirements() -> None:
    dataset = _make_dataset()
    model = _make_model()
    trainer = WorldModelTrainer(
        model=model,
        dataset=dataset,
        val_dataset=dataset,
        sampling_strategy=RandomStartFixedHorizon(horizon=6),
        warmup_len=12,
        batch_size=4,
        optimizer=None,  # should create AdamW defaults
        device="cpu",
        early_stopping=True,
        patience=30,
    )

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]["lr"] == 3e-4
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 1e-6
    assert trainer.grad_clip_norm == 100.0
    assert trainer.early_stopping_monitor == "open_loop_crps"
    assert trainer.lr_warmup_steps >= 1
    assert trainer.lr_min_ratio == 0.01


def test_checkpoint_bundle_contains_metadata_and_optimizer_state(tmp_path) -> None:
    dataset = _make_dataset()
    model = _make_model()
    ckpt_path = tmp_path / "latent_ssm" / "train_checkpoint.pth"
    metadata = {
        "normalization_stats": {"scale_": [1.0, 1.0, 1.0]},
        "variable_schema": {
            "control": ["control"],
            "exogenous": ["exo"],
            "objective": ["output"],
        },
        "config": {"training": {"epochs": 1}},
    }
    trainer = WorldModelTrainer(
        model=model,
        dataset=dataset,
        val_dataset=dataset,
        sampling_strategy=RandomStartFixedHorizon(horizon=6),
        warmup_len=12,
        batch_size=4,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6),
        device="cpu",
        early_stopping=False,
        run_dir=tmp_path / "latent_ssm",
        checkpoint_metadata=metadata,
        probabilistic_cfg={
            "checkpoint_metric": "val_loss",
            "rollout_weight": 0.0,
            "rollout_dtw_weight": 0.0,
            "lr_warmup_steps": 0,  # should still be treated as mandatory warmup >=1
        },
    )
    train_losses, val_losses = trainer.fit(
        epochs=1,
        steps_per_epoch=1,
        verbose=False,
        checkpoint_path=ckpt_path,
    )
    assert len(train_losses) == 1
    assert len(val_losses) == 1
    assert trainer.lr_warmup_steps == 1

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(saved, dict)
    assert "model_state_dict" in saved
    assert "optimizer_state_dict" in saved
    assert "trainer_state" in saved
    assert "metadata" in saved
    assert saved["metadata"]["normalization_stats"]["scale_"] == [1.0, 1.0, 1.0]
    assert saved["metadata"]["variable_schema"]["objective"] == ["output"]
    assert saved["metadata"]["config"]["training"]["epochs"] == 1

    topk_dir = ckpt_path.parent / "checkpoints"
    bundles = list(topk_dir.glob("*.pth"))
    assert bundles, "Expected at least one top-k checkpoint bundle."
    topk_bundle = torch.load(bundles[0], map_location="cpu", weights_only=False)
    assert "metadata" in topk_bundle
    assert "optimizer_state_dict" in topk_bundle
