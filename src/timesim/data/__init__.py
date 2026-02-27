"""Data handling for time-series world models."""

from .dataset import TimeSeriesDataset, GroupedTimeSeriesDataset, SlidingWindowRoleDataset
from .schema import VariableRole, VariableSchema
from .loader import (
    generate_sine_dataset,
    build_dataloaders,
    load_csv_dataset,
    build_grouped_dataloaders,
    build_dataloaders_from_config,
    build_grouped_triplet_dataloaders,
    chronological_split_dataframe,
    resolve_split_ratios,
)
from .sampling import (
    SamplingStrategy,
    RandomStartRandomHorizon,
    RandomStartFixedHorizon,
    DailyFixedHorizon,
    GeometricHorizonSampling,
    StrideBasedSampling,
)
from .validation import validate_time_series_dataframe, validate_variable_groups
from .preprocessing import (
    NormalizationStats,
    fit_scaler,
    fit_normalization_stats,
    apply_scaler,
    normalize_array,
    denormalize_array,
    select_columns,
)
from .datamodule import TimeSeriesDataModule

__all__ = [
    "TimeSeriesDataset",
    "GroupedTimeSeriesDataset",
    "SlidingWindowRoleDataset",
    "VariableRole",
    "VariableSchema",
    "generate_sine_dataset",
    "build_dataloaders",
    "load_csv_dataset",
    "build_grouped_dataloaders",
    "build_dataloaders_from_config",
    "build_grouped_triplet_dataloaders",
    "chronological_split_dataframe",
    "resolve_split_ratios",
    "SamplingStrategy",
    "RandomStartRandomHorizon",
    "RandomStartFixedHorizon",
    "DailyFixedHorizon",
    "GeometricHorizonSampling",
    "StrideBasedSampling",
    "validate_time_series_dataframe",
    "validate_variable_groups",
    "NormalizationStats",
    "fit_scaler",
    "fit_normalization_stats",
    "apply_scaler",
    "normalize_array",
    "denormalize_array",
    "select_columns",
    "TimeSeriesDataModule",
]
