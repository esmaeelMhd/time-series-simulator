"""Tests for rollout training functionality."""

import numpy as np
import pandas as pd
import torch
import pytest

from timesim.data.dataset import GroupedTimeSeriesDataset
from timesim.data.sampling import RandomStartFixedHorizon
from timesim.models.lstm import LSTMWorldModel
from timesim.models.latent_ssm import LatentSSMWorldModel
from timesim.training.rollout import batch_rollout, batch_rollout_padded
from timesim.training.losses import OneStepLoss, MultiStepLoss, ProbabilisticRolloutLoss
from timesim.training.trainer import WorldModelTrainer


def create_synthetic_dataset(n=500):
    """Create a synthetic dataset for testing."""
    df = pd.DataFrame({
        "control": np.sin(np.linspace(0, 10, n)),
        "exo": np.cos(np.linspace(0, 10, n)),
        "output": np.sin(np.linspace(0, 10, n)) * 0.5 + np.cos(np.linspace(0, 10, n)) * 0.5,
    })
    
    groups = {
        "control": ["control"],
        "exogenous": ["exo"],
        "objective": ["output"],
    }
    
    dataset = GroupedTimeSeriesDataset(
        df, groups, ["control", "exogenous"], ["objective"],
        seq_len=10, pred_len=5, scale=True
    )
    
    return dataset


def test_batch_rollout_basic():
    """Test basic batch rollout functionality."""
    dataset = create_synthetic_dataset(n=300)
    model = LSTMWorldModel(input_dim=3, output_dim=1, hidden_dim=16)
    model.eval()
    
    # Sample rollouts
    start_indices = np.array([50, 100, 150])
    horizons = np.array([10, 10, 10])
    warmup_len = 20
    
    with torch.no_grad():
        result = batch_rollout(
            model, dataset, start_indices, horizons, warmup_len,
            feedback="model", device="cpu"
        )
    
    # Check outputs
    assert "predictions" in result
    assert "targets" in result
    assert len(result["predictions"]) == 3
    assert len(result["targets"]) == 3
    
    # Check shapes
    for i in range(3):
        assert result["predictions"][i].shape == (10, 1)
        assert result["targets"][i].shape == (10, 1)


def test_batch_rollout_padded():
    """Test padded batch rollout with variable horizons."""
    dataset = create_synthetic_dataset(n=300)
    model = LSTMWorldModel(input_dim=3, output_dim=1, hidden_dim=16)
    model.eval()
    
    # Variable horizons
    start_indices = np.array([50, 100, 150])
    horizons = np.array([8, 12, 10])
    warmup_len = 20
    
    with torch.no_grad():
        result = batch_rollout_padded(
            model, dataset, start_indices, horizons, warmup_len,
            feedback="model", device="cpu"
        )
    
    # Check outputs
    assert result["predictions"].shape == (3, 12, 1)  # max horizon is 12
    assert result["targets"].shape == (3, 12, 1)
    assert result["mask"].shape == (3, 12)
    
    # Check mask
    assert result["mask"][0, :8].all()   # First 8 valid
    assert not result["mask"][0, 8:].any()  # Rest padded
    assert result["mask"][1, :12].all()  # All 12 valid
    assert result["mask"][2, :10].all()  # First 10 valid


def test_rollout_with_teacher_forcing():
    """Test rollout with teacher forcing feedback."""
    dataset = create_synthetic_dataset(n=300)
    model = LSTMWorldModel(input_dim=3, output_dim=1, hidden_dim=16)
    model.eval()
    
    start_indices = np.array([50, 100])
    horizons = np.array([10, 10])
    warmup_len = 20
    
    with torch.no_grad():
        result = batch_rollout_padded(
            model, dataset, start_indices, horizons, warmup_len,
            feedback="teacher", device="cpu"
        )
    
    assert result["predictions"].shape == (2, 10, 1)


def test_one_step_loss():
    """Test one-step loss computation."""
    loss_fn = OneStepLoss(loss_type="mse")
    
    predictions = torch.randn(4, 10, 2)
    targets = torch.randn(4, 10, 2)
    
    loss = loss_fn(predictions, targets)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_multi_step_loss_uniform():
    """Test multi-step loss with uniform weighting."""
    loss_fn = MultiStepLoss(loss_type="mse", weighting="uniform")
    
    predictions = torch.randn(4, 10, 2)
    targets = torch.randn(4, 10, 2)
    
    loss = loss_fn(predictions, targets)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_multi_step_loss_linear_weighting():
    """Test multi-step loss with linear weighting (emphasize later steps)."""
    loss_fn = MultiStepLoss(loss_type="mse", weighting="linear")
    
    predictions = torch.randn(4, 10, 2)
    targets = torch.randn(4, 10, 2)
    
    loss = loss_fn(predictions, targets)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_simple_training_reduces_loss():
    """Test that a few training steps reduce loss on a simple problem."""
    # Create a simple linear system: output = 0.5 * control + 0.3 * exo
    n = 500
    control = np.random.randn(n)
    exo = np.random.randn(n)
    output = 0.5 * control + 0.3 * exo + 0.01 * np.random.randn(n)
    
    df = pd.DataFrame({
        "control": control,
        "exo": exo,
        "output": output,
    })
    
    groups = {
        "control": ["control"],
        "exogenous": ["exo"],
        "objective": ["output"],
    }
    
    dataset = GroupedTimeSeriesDataset(
        df, groups, ["control", "exogenous"], ["objective"],
        seq_len=10, pred_len=5, scale=True
    )
    
    # Create model
    model = LSTMWorldModel(input_dim=3, output_dim=1, hidden_dim=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = MultiStepLoss(loss_type="mse")
    
    # Train for a few steps
    initial_loss = None
    final_loss = None
    
    for step in range(20):
        # Sample rollouts
        start_indices = np.array([50, 100, 150, 200])
        horizons = np.array([10, 10, 10, 10])
        warmup_len = 20
        
        result = batch_rollout_padded(
            model, dataset, start_indices, horizons, warmup_len,
            feedback="model", device="cpu"
        )
        
        predictions = result["predictions"]
        targets = result["targets"]
        
        optimizer.zero_grad()
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        if step == 19:
            final_loss = loss.item()
    
    # Loss should decrease
    assert final_loss < initial_loss * 0.8, f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"


def test_probabilistic_rollout_loss_masked_finite():
    loss_fn = ProbabilisticRolloutLoss(
        recon_weight=1.0,
        kl_weight=1.0,
        aux_weight=1.0,
        kl_free_bits=1.0,
        kl_balance=0.8,
        use_kl_balancing=True,
        use_free_bits=True,
    )
    bsz, horizon, out_dim = 4, 6, 2
    targets = torch.randn(bsz, horizon, out_dim)
    exogenous = torch.randn(bsz, horizon, 3)
    dist_loc_latent = torch.randn(bsz, horizon, out_dim)
    dist_scale = torch.rand(bsz, horizon, out_dim) + 0.01
    prior_mu = torch.randn(bsz, horizon, 5)
    prior_logvar = torch.randn(bsz, horizon, 5).clamp(min=-4.0, max=4.0)
    posterior_mu = torch.randn(bsz, horizon, 5)
    posterior_logvar = torch.randn(bsz, horizon, 5).clamp(min=-4.0, max=4.0)
    aux_loc = torch.randn_like(exogenous)
    aux_scale = torch.rand_like(exogenous) + 0.01
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    loss, info = loss_fn(
        targets=targets,
        dist_loc_latent=dist_loc_latent,
        dist_scale=dist_scale,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=posterior_mu,
        posterior_logvar=posterior_logvar,
        exogenous_targets=exogenous,
        aux_loc=aux_loc,
        aux_scale=aux_scale,
        mask=mask,
    )
    assert torch.isfinite(loss)
    assert np.isfinite(info["recon_nll"])
    assert np.isfinite(info["kl"])
    assert np.isfinite(info["aux_nll"])


def test_probabilistic_trainer_runs():
    dataset = create_synthetic_dataset(n=260)
    model = LatentSSMWorldModel(
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
    )
    trainer = WorldModelTrainer(
        model=model,
        dataset=dataset,
        val_dataset=dataset,
        sampling_strategy=RandomStartFixedHorizon(horizon=6),
        warmup_len=12,
        batch_size=8,
        training_mode="multi_step",
        feedback="model",
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device="cpu",
        early_stopping=False,
        run_dir=None,
        probabilistic_cfg={"elbo_weight": 1.0, "kl_weight": 1.0, "rollout_mse_weight": 1.0},
    )
    train_losses, val_losses = trainer.fit(epochs=2, steps_per_epoch=2, verbose=False)
    assert len(train_losses) == 2
    assert len(val_losses) == 2
    assert np.isfinite(train_losses[-1])
    assert val_losses[-1] is None or np.isfinite(val_losses[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
