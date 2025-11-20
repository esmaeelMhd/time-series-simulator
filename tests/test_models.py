"""Tests for world models."""

import numpy as np
import torch
import pytest

from timesim.models.base import WorldModelBase
from timesim.models.lstm import LSTMWorldModel


def test_lstm_world_model_init():
    """Test LSTMWorldModel initialization."""
    model = LSTMWorldModel(
        input_dim=10,
        output_dim=3,
        hidden_dim=32,
        num_layers=2,
    )
    
    assert model.input_dim == 10
    assert model.output_dim == 3
    assert model.hidden_dim == 32
    assert model.num_layers == 2


def test_lstm_init_state():
    """Test LSTM state initialization from warmup."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16, num_layers=2)
    
    batch_size = 4
    warmup_len = 10
    warmup_seq = torch.randn(batch_size, warmup_len, 5)
    
    h, c = model.init_state(warmup_seq)
    
    # Check shapes
    assert h.shape == (2, batch_size, 16)  # (num_layers, batch, hidden)
    assert c.shape == (2, batch_size, 16)


def test_lstm_step():
    """Test single-step prediction."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    model.eval()
    
    batch_size = 3
    control_dim = 2
    exo_dim = 1
    output_dim = 2
    
    # Initialize state
    warmup_seq = torch.randn(batch_size, 10, 5)
    state = model.init_state(warmup_seq)
    
    # Single step
    control_t = torch.randn(batch_size, control_dim)
    exo_t = torch.randn(batch_size, exo_dim)
    prev_output_t = torch.randn(batch_size, output_dim)
    
    new_state, pred = model.step(state, control_t, exo_t, prev_output_t)
    
    # Check output shape
    assert pred.shape == (batch_size, output_dim)
    
    # Check that state was updated
    h_new, c_new = new_state
    h_old, c_old = state
    assert not torch.allclose(h_new, h_old)


def test_lstm_rollout_model_feedback():
    """Test multi-step rollout with model feedback."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    model.eval()
    
    batch_size = 2
    warmup_len = 10
    horizon = 15
    control_dim = 2
    exo_dim = 1
    
    # Prepare inputs
    warmup_seq = {
        "inputs": torch.randn(batch_size, warmup_len, 5)
    }
    
    rollout_inputs = {
        "controls": torch.randn(batch_size, horizon, control_dim),
        "exogenous": torch.randn(batch_size, horizon, exo_dim),
    }
    
    # Rollout
    with torch.no_grad():
        result = model.rollout(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            feedback="model",
        )
    
    # Check output
    assert "predictions" in result
    assert result["predictions"].shape == (batch_size, horizon, 2)


def test_lstm_rollout_teacher_forcing():
    """Test multi-step rollout with teacher forcing."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    model.eval()
    
    batch_size = 2
    warmup_len = 10
    horizon = 15
    
    warmup_seq = {"inputs": torch.randn(batch_size, warmup_len, 5)}
    rollout_inputs = {
        "controls": torch.randn(batch_size, horizon, 2),
        "exogenous": torch.randn(batch_size, horizon, 1),
    }
    targets = torch.randn(batch_size, horizon, 2)
    
    with torch.no_grad():
        result = model.rollout(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            feedback="teacher",
            targets=targets,
        )
    
    assert result["predictions"].shape == (batch_size, horizon, 2)


def test_lstm_rollout_mixed_feedback():
    """Test rollout with mixed (scheduled sampling) feedback."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    model.eval()
    
    batch_size = 2
    horizon = 10
    
    warmup_seq = {"inputs": torch.randn(batch_size, 10, 5)}
    rollout_inputs = {
        "controls": torch.randn(batch_size, horizon, 2),
        "exogenous": torch.randn(batch_size, horizon, 1),
    }
    targets = torch.randn(batch_size, horizon, 2)
    
    with torch.no_grad():
        result = model.rollout(
            warmup_seq=warmup_seq,
            rollout_inputs=rollout_inputs,
            horizon=horizon,
            feedback="mixed",
            teacher_forcing_ratio=0.5,
            targets=targets,
        )
    
    assert result["predictions"].shape == (batch_size, horizon, 2)


def test_lstm_forward_backward_compatibility():
    """Test that forward() method works for backward compatibility."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16, pred_len=3)
    
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 5)
    
    pred = model(x)
    
    # Check output shape
    assert pred.shape == (batch_size, 3, 2)  # (batch, pred_len, output_dim)


def test_lstm_deterministic_with_seed():
    """Test that model is deterministic with fixed seed."""
    torch.manual_seed(42)
    model1 = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    
    torch.manual_seed(42)
    model2 = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    
    # Same input
    x = torch.randn(2, 10, 5)
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)
    
    # Should be identical
    assert torch.allclose(out1, out2)


def test_lstm_gradient_flow():
    """Test that gradients flow through the model."""
    model = LSTMWorldModel(input_dim=5, output_dim=2, hidden_dim=16)
    model.train()
    
    x = torch.randn(2, 10, 5, requires_grad=True)
    pred = model(x)
    loss = pred.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

