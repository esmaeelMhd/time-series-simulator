"""Data handling for time-series world models."""

from .datamodule import TimeSeriesDataModule
from .dataset import GroupedTimeSeriesDataset, SlidingWindowRoleDataset, TimeSeriesDataset
from .loader import (
    build_dataloaders,
    build_dataloaders_from_config,
    build_grouped_dataloaders,
    build_grouped_triplet_dataloaders,
    chronological_split_dataframe,
    generate_sine_dataset,
    held_out_eval_frame,
    load_csv_dataset,
    resolve_split_ratios,
)
from .preprocessing import (
    NormalizationStats,
    apply_scaler,
    denormalize_array,
    fit_normalization_stats,
    fit_scaler,
    normalize_array,
    select_columns,
)
from .sampling import (
    DailyFixedHorizon,
    GeometricHorizonSampling,
    RandomStartFixedHorizon,
    RandomStartRandomHorizon,
    SamplingStrategy,
    StrideBasedSampling,
)
from .schema import VariableRole, VariableSchema
from .validation import validate_time_series_dataframe, validate_variable_groups

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
    "held_out_eval_frame",
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
