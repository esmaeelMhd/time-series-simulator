"""Sampling strategies for multi-step world model training.

HOT PATH: Sampling is called every training step.
Key optimizations:
- Vectorized start_indices computation (no Python loops)
- Preallocated numpy arrays with fixed dtypes
- Batch random number generation

These strategies determine which starting points and horizons to use when
training a world model with multi-step rollouts. Different strategies are
suitable for different domains and training objectives.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np


class SamplingStrategy(Protocol):
    """Protocol for sampling strategies.
    
    A sampling strategy determines which (start_index, horizon) pairs to use
    for training. This enables flexible training schemes like:
    - Random starts with random horizons
    - Fixed daily patterns (e.g., start at midnight, 24h horizon)
    - Curriculum learning (gradually increasing horizons)
    """

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample starting indices and horizons for a batch.
        
        Parameters
        ----------
        dataset_length : int
            Total length of the dataset (number of timesteps).
        batch_size : int
            Number of rollouts to sample.
        warmup_len : int
            Length of warmup sequence (must be available before start_index).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        
        Returns
        -------
        start_indices : np.ndarray
            Starting indices for each rollout, shape (batch_size,).
            These are the indices where the rollout begins (after warmup).
        horizons : np.ndarray
            Horizon length for each rollout, shape (batch_size,).
        
        Notes
        -----
        The implementation must ensure:
        - start_indices[i] >= warmup_len (warmup must fit before start)
        - start_indices[i] + horizons[i] <= dataset_length (rollout must fit)
        """
        ...


class RandomStartRandomHorizon:
    """Sample random starting points with random horizons.
    
    This is the most general strategy and provides maximum diversity in training.
    Good for learning robust dynamics across all time scales.
    
    HOT PATH: Called every training step.
    Optimizations:
    - Vectorized start_indices computation (no Python loop)
    - Single batch random number generation
    
    Parameters
    ----------
    h_min : int
        Minimum horizon length.
    h_max : int
        Maximum horizon length.
    """

    def __init__(self, h_min: int = 1, h_max: int = 24):
        if h_min < 1:
            raise ValueError("h_min must be >= 1")
        if h_max < h_min:
            raise ValueError("h_max must be >= h_min")
        self.h_min = h_min
        self.h_max = h_max

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Sample horizons first (vectorized)
        horizons = rng.integers(self.h_min, self.h_max + 1, size=batch_size, dtype=np.int64)

        # Validate that dataset is long enough for max horizon
        if dataset_length - self.h_max < warmup_len:
            raise ValueError(
                f"Dataset too short: need at least {warmup_len + self.h_max} "
                f"timesteps but only have {dataset_length}"
            )

        # HOT PATH: Vectorized start_indices computation
        # max_start[i] = dataset_length - horizons[i]
        # min_start = warmup_len (constant)
        # start_indices[i] ~ Uniform[min_start, max_start[i]]
        max_starts = dataset_length - horizons  # (batch_size,)

        # Generate uniform random in [0, 1) and scale to [min_start, max_start]
        # start = min_start + floor(random * (max_start - min_start + 1))
        ranges = max_starts - warmup_len + 1  # Number of valid positions per item
        random_offsets = rng.random(batch_size)  # (batch_size,) in [0, 1)
        start_indices = warmup_len + (random_offsets * ranges).astype(np.int64)

        return start_indices, horizons


class RandomStartFixedHorizon:
    """Sample random starting points with a fixed horizon.
    
    This is useful when you want to focus on a specific prediction horizon
    (e.g., always predict 24 steps ahead) but still want diversity in
    starting conditions.
    
    Parameters
    ----------
    horizon : int
        Fixed horizon length for all rollouts.
    """

    def __init__(self, horizon: int = 24):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        max_start = dataset_length - self.horizon
        min_start = warmup_len

        if max_start < min_start:
            raise ValueError(
                f"Dataset too short: need at least {warmup_len + self.horizon} "
                f"timesteps but only have {dataset_length}"
            )

        start_indices = rng.integers(min_start, max_start + 1, size=batch_size)
        horizons = np.full(batch_size, self.horizon, dtype=np.int64)

        return start_indices, horizons


class DailyFixedHorizon:
    """Sample starting points at fixed daily intervals with fixed horizon.
    
    This strategy is designed for domains with daily patterns (e.g., wastewater
    treatment, energy systems) where you want to train the model to predict
    a full day starting from a specific time (e.g., midnight).
    
    Parameters
    ----------
    start_hour : int, default 0
        Hour of day to start rollouts (0-23). Default is midnight.
    horizon : int, default 24
        Horizon length in hours. Default is 24 hours (one day).
    samples_per_hour : int, default 1
        Number of samples per hour in the dataset. For example:
        - 1 for hourly data
        - 2 for 30-minute data
        - 30 for 2-minute data
    
    Notes
    -----
    This strategy assumes the dataset has a regular sampling rate and that
    days are aligned in the data. It will sample starting points that fall
    at the specified hour of each day.
    """

    def __init__(
        self,
        start_hour: int = 0,
        horizon: int = 24,
        samples_per_hour: int = 1,
    ):
        if not 0 <= start_hour <= 23:
            raise ValueError("start_hour must be between 0 and 23")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if samples_per_hour < 1:
            raise ValueError("samples_per_hour must be >= 1")

        self.start_hour = start_hour
        self.horizon = horizon
        self.samples_per_hour = samples_per_hour
        self.timesteps_per_day = 24 * samples_per_hour
        self.start_offset = start_hour * samples_per_hour

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Find all valid daily starting points
        # Start from first occurrence of start_hour after warmup
        first_start = warmup_len + (self.start_offset - warmup_len % self.timesteps_per_day) % self.timesteps_per_day

        # Generate all daily starts
        daily_starts = []
        current = first_start
        while current + self.horizon <= dataset_length:
            daily_starts.append(current)
            current += self.timesteps_per_day

        if len(daily_starts) == 0:
            raise ValueError(
                f"No valid daily starting points found. Dataset length: {dataset_length}, "
                f"warmup: {warmup_len}, horizon: {self.horizon}, "
                f"timesteps_per_day: {self.timesteps_per_day}"
            )

        # Sample batch_size starting points (with replacement if needed)
        daily_starts = np.array(daily_starts)
        indices = rng.choice(len(daily_starts), size=batch_size, replace=True)
        start_indices = daily_starts[indices]
        horizons = np.full(batch_size, self.horizon, dtype=np.int64)

        return start_indices, horizons


class GeometricHorizonSampling:
    """Sample with geometrically spaced horizons for curriculum learning.
    
    This strategy samples horizons at geometrically increasing intervals
    (e.g., 1, 2, 4, 8, 16, 32, 64 steps). This is useful for training models
    that need to learn both short-term and long-term dynamics.
    
    HOT PATH: Called every training step.
    Optimizations:
    - Vectorized start_indices computation (no Python loop)
    - Precomputed geometric horizon sequence
    
    Based on the SEPP (See Every Possible Path) approach in the original code.
    
    Parameters
    ----------
    pred_len : int
        Base prediction length (minimum horizon).
    h_max : int
        Maximum horizon length.
    """

    def __init__(self, pred_len: int = 1, h_max: int = 64):
        if pred_len < 1:
            raise ValueError("pred_len must be >= 1")
        if h_max < pred_len:
            raise ValueError("h_max must be >= pred_len")

        self.pred_len = pred_len
        self.h_max = h_max

        # Compute geometric sequence of horizons (done once at init)
        horizons_list = []
        h = pred_len
        while h <= h_max:
            horizons_list.append(h)
            h *= 2
        self.horizons = np.array(horizons_list, dtype=np.int64)

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Validate dataset length for max horizon
        if dataset_length - self.h_max < warmup_len:
            raise ValueError(
                f"Dataset too short: need at least {warmup_len + self.h_max} "
                f"timesteps but only have {dataset_length}"
            )

        # Sample horizons from the geometric sequence (vectorized)
        horizon_indices = rng.choice(len(self.horizons), size=batch_size, replace=True)
        horizons = self.horizons[horizon_indices]

        # HOT PATH: Vectorized start_indices computation
        max_starts = dataset_length - horizons  # (batch_size,)
        ranges = max_starts - warmup_len + 1
        random_offsets = rng.random(batch_size)
        start_indices = warmup_len + (random_offsets * ranges).astype(np.int64)

        return start_indices, horizons


class StrideBasedSampling:
    """Sample starting points at regular stride intervals.
    
    This strategy samples starting points at fixed stride intervals, similar
    to the original SEPP implementation. It's deterministic (given a fixed order)
    and ensures even coverage of the dataset.
    
    Parameters
    ----------
    stride : int
        Stride between consecutive starting points.
    h_max : int
        Maximum horizon length.
    """

    def __init__(self, stride: int = 12, h_max: int = 64):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if h_max < 1:
            raise ValueError("h_max must be >= 1")

        self.stride = stride
        self.h_max = h_max

    def sample(
        self,
        dataset_length: int,
        batch_size: int,
        warmup_len: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Generate all valid starting points with stride
        max_start = dataset_length - self.h_max
        starts = np.arange(warmup_len, max_start + 1, self.stride)

        if len(starts) == 0:
            raise ValueError(
                f"No valid starting points with stride {self.stride}. "
                f"Dataset length: {dataset_length}, warmup: {warmup_len}, h_max: {self.h_max}"
            )

        # Sample batch_size starting points
        indices = rng.choice(len(starts), size=batch_size, replace=True)
        start_indices = starts[indices]

        # For now, use fixed h_max; could be extended to sample variable horizons
        horizons = np.full(batch_size, self.h_max, dtype=np.int64)

        return start_indices, horizons

