"""Tests for dataset classes."""

import numpy as np
import pandas as pd
import pytest
import torch

from timesim.data.dataset import TimeSeriesDataset, GroupedTimeSeriesDataset
from timesim.data.schema import VariableSchema


def test_time_series_dataset_basic():
    """Test basic TimeSeriesDataset functionality."""
    # Create synthetic data
    data = np.random.randn(100, 3).astype(np.float32)
    seq_len, pred_len = 10, 5
    
    dataset = TimeSeriesDataset(data, seq_len=seq_len, pred_len=pred_len, scale=False)
    
    # Check length
    expected_len = len(data) - (seq_len + pred_len)
    assert len(dataset) == expected_len
    
    # Check item shape
    x, y = dataset[0]
    assert x.shape == (seq_len, 3)
    assert y.shape == (pred_len, 3)
    
    # Check data type
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_time_series_dataset_scaling():
    """Test that scaling works correctly."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * 20, dtype=np.float32)
    dataset = TimeSeriesDataset(data, seq_len=5, pred_len=2, scale=True)
    
    # Check that scaler was created
    assert dataset.scaler is not None
    
    # Check that values are scaled (should be in [0, 1])
    assert dataset.values.min() >= 0.0
    assert dataset.values.max() <= 1.0


def test_grouped_time_series_dataset():
    """Test GroupedTimeSeriesDataset with control/exogenous/objective groups."""
    # Create synthetic dataframe
    n = 100
    df = pd.DataFrame({
        "control_1": np.random.randn(n),
        "control_2": np.random.randn(n),
        "exo_1": np.random.randn(n),
        "exo_2": np.random.randn(n),
        "output_1": np.random.randn(n),
    })
    
    groups = {
        "control": ["control_1", "control_2"],
        "exogenous": ["exo_1", "exo_2"],
        "objective": ["output_1"],
    }
    
    input_groups = ["control", "exogenous"]
    output_groups = ["objective"]
    
    dataset = GroupedTimeSeriesDataset(
        df, groups, input_groups, output_groups,
        seq_len=10, pred_len=5, scale=False
    )
    
    # Check that column indices are correct
    assert len(dataset.in_idx) == 4  # 2 controls + 2 exogenous
    assert len(dataset.out_idx) == 1  # 1 objective
    
    # Check item shapes
    x, y = dataset[0]
    assert x.shape == (10, 4)  # seq_len x input_dim
    assert y.shape == (5, 1)   # pred_len x output_dim


def test_grouped_dataset_warmup_and_rollout():
    """Test warmup window and rollout slice methods."""
    n = 200
    df = pd.DataFrame({
        "control": np.random.randn(n),
        "exo": np.random.randn(n),
        "output": np.random.randn(n),
    })
    
    groups = {
        "control": ["control"],
        "exogenous": ["exo"],
        "objective": ["output"],
    }
    
    dataset = GroupedTimeSeriesDataset(
        df, groups, ["control", "exogenous"], ["objective"],
        seq_len=10, pred_len=5, scale=False
    )
    
    # Test warmup window
    warmup_len = 20
    start_idx = 50
    warmup = dataset.get_warmup_window(start_idx, warmup_len)
    
    assert "inputs" in warmup
    assert "outputs" in warmup
    assert warmup["inputs"].shape == (warmup_len, 2)  # control + exo
    assert warmup["outputs"].shape == (warmup_len, 1)  # output
    
    # Test rollout slice
    horizon = 30
    rollout = dataset.get_rollout_slice(start_idx, horizon)
    
    assert "inputs" in rollout
    assert "targets" in rollout
    assert rollout["inputs"].shape == (horizon, 2)
    assert rollout["targets"].shape == (horizon, 1)


def test_dataset_nan_detection():
    """Test that NaN values are detected and raise an error."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]] * 20, dtype=np.float32)
    
    with pytest.raises(ValueError, match="NaNs remain"):
        TimeSeriesDataset(data, seq_len=5, pred_len=2, scale=True)


def test_variable_schema_rejects_duplicate_column_roles():
    with pytest.raises(ValueError, match="assigned to both"):
        VariableSchema.from_groups(
            {
                "control": ["u1"],
                "exogenous": ["u1"],
                "objective": ["y"],
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
