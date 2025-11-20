"""Tests for sampling strategies."""

import numpy as np
import pytest

from timesim.data.sampling import (
    RandomStartRandomHorizon,
    RandomStartFixedHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
    StrideBasedSampling,
)


def test_random_start_random_horizon():
    """Test RandomStartRandomHorizon sampling strategy."""
    strategy = RandomStartRandomHorizon(h_min=5, h_max=20)
    
    dataset_length = 1000
    batch_size = 32
    warmup_len = 24
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check shapes
    assert start_indices.shape == (batch_size,)
    assert horizons.shape == (batch_size,)
    
    # Check bounds
    assert np.all(start_indices >= warmup_len)
    assert np.all(start_indices + horizons <= dataset_length)
    assert np.all(horizons >= 5)
    assert np.all(horizons <= 20)


def test_random_start_fixed_horizon():
    """Test RandomStartFixedHorizon sampling strategy."""
    horizon = 24
    strategy = RandomStartFixedHorizon(horizon=horizon)
    
    dataset_length = 1000
    batch_size = 16
    warmup_len = 12
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check that all horizons are fixed
    assert np.all(horizons == horizon)
    
    # Check bounds
    assert np.all(start_indices >= warmup_len)
    assert np.all(start_indices + horizons <= dataset_length)


def test_daily_fixed_horizon():
    """Test DailyFixedHorizon sampling strategy."""
    # Simulate hourly data for 10 days
    samples_per_hour = 1
    dataset_length = 24 * 10 * samples_per_hour
    
    strategy = DailyFixedHorizon(
        start_hour=0,
        horizon=24,
        samples_per_hour=samples_per_hour,
    )
    
    batch_size = 5
    warmup_len = 24
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check that all horizons are 24
    assert np.all(horizons == 24)
    
    # Check that start indices are at daily boundaries (midnight)
    # After warmup, first valid start should be at a daily boundary
    for idx in start_indices:
        # Should be at start of day (modulo 24)
        assert (idx % 24) == 0 or (idx - warmup_len) % 24 == 0


def test_daily_fixed_horizon_2min_data():
    """Test DailyFixedHorizon with 2-minute data (30 samples/hour)."""
    samples_per_hour = 30
    dataset_length = 24 * 30 * 10  # 10 days of 2-min data
    
    strategy = DailyFixedHorizon(
        start_hour=0,
        horizon=24 * 30,  # 24 hours in 2-min samples
        samples_per_hour=samples_per_hour,
    )
    
    batch_size = 3
    warmup_len = 24 * 30  # 24 hours warmup
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check shapes and bounds
    assert len(start_indices) == batch_size
    assert np.all(horizons == 24 * 30)


def test_geometric_horizon_sampling():
    """Test GeometricHorizonSampling strategy."""
    strategy = GeometricHorizonSampling(pred_len=1, h_max=64)
    
    # Check that horizons are geometric: 1, 2, 4, 8, 16, 32, 64
    expected_horizons = [1, 2, 4, 8, 16, 32, 64]
    assert list(strategy.horizons) == expected_horizons
    
    dataset_length = 1000
    batch_size = 20
    warmup_len = 10
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check that sampled horizons are from the geometric sequence
    assert np.all(np.isin(horizons, expected_horizons))
    
    # Check bounds
    assert np.all(start_indices >= warmup_len)
    assert np.all(start_indices + horizons <= dataset_length)


def test_stride_based_sampling():
    """Test StrideBasedSampling strategy."""
    stride = 12
    h_max = 48
    strategy = StrideBasedSampling(stride=stride, h_max=h_max)
    
    dataset_length = 1000
    batch_size = 10
    warmup_len = 24
    
    start_indices, horizons = strategy.sample(dataset_length, batch_size, warmup_len)
    
    # Check that horizons are h_max
    assert np.all(horizons == h_max)
    
    # Check bounds
    assert np.all(start_indices >= warmup_len)
    assert np.all(start_indices + horizons <= dataset_length)


def test_sampling_with_seed():
    """Test that sampling is reproducible with a fixed seed."""
    strategy = RandomStartRandomHorizon(h_min=5, h_max=20)
    
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    start1, horizons1 = strategy.sample(1000, 10, 24, rng=rng1)
    start2, horizons2 = strategy.sample(1000, 10, 24, rng=rng2)
    
    # Should be identical with same seed
    assert np.array_equal(start1, start2)
    assert np.array_equal(horizons1, horizons2)


def test_sampling_dataset_too_short():
    """Test that sampling raises error when dataset is too short."""
    strategy = RandomStartFixedHorizon(horizon=100)
    
    # Dataset too short for horizon + warmup
    with pytest.raises(ValueError, match="Dataset too short"):
        strategy.sample(dataset_length=50, batch_size=5, warmup_len=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

