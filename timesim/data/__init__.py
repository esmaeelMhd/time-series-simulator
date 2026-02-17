"""Data handling for time-series world models."""

from .dataset import TimeSeriesDataset, GroupedTimeSeriesDataset
from .schema import VariableRole, VariableSchema
from .loader import (
    generate_sine_dataset,
    build_dataloaders,
    load_csv_dataset,
    build_grouped_dataloaders,
)
from .sampling import (
    SamplingStrategy,
    RandomStartRandomHorizon,
    RandomStartFixedHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
    StrideBasedSampling,
)

__all__ = [
    "TimeSeriesDataset",
    "GroupedTimeSeriesDataset",
    "VariableRole",
    "VariableSchema",
    "generate_sine_dataset",
    "build_dataloaders",
    "load_csv_dataset",
    "build_grouped_dataloaders",
    "SamplingStrategy",
    "RandomStartRandomHorizon",
    "RandomStartFixedHorizon",
    "DailyFixedHorizon",
    "GeometricHorizonSampling",
    "StrideBasedSampling",
]
