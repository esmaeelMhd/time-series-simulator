"""Dataframe and schema validation for time-series pipelines."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def validate_variable_groups(groups: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for role, cols in groups.items():
        for c in cols:
            if c in seen:
                raise ValueError(
                    f"Column '{c}' appears in multiple variable groups: '{seen[c]}' and '{role}'."
                )
            seen[c] = str(role)


def validate_time_series_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: Optional[Iterable[str]] = None,
    strict: bool = False,
    require_datetime_index: bool = True,
    require_monotonic_index: bool = True,
    require_numeric_dtypes: bool = True,
    allow_nan: bool = False,
) -> pd.DataFrame:
    """Validate base invariants required by the training/eval pipeline."""
    req_cols = list(required_columns or [])
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataframe columns: {missing}")

    if require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected pandas.DatetimeIndex for time-series dataframe.")

    if require_monotonic_index:
        if not df.index.is_monotonic_increasing:
            raise ValueError("Time index must be monotonic increasing (chronological order).")
        if not df.index.is_unique:
            raise ValueError("Time index must be unique; duplicate timestamps found.")

    if require_numeric_dtypes:
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {non_numeric}.")

    if (not allow_nan) and df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs detected in dataframe columns: {nan_cols}.")

    try:
        import pandera as pa  # type: ignore
    except Exception:
        if strict:
            raise ImportError("Pandera strict validation requested but pandera is not installed.")
        return df

    schema_cols = {c: pa.Column(float, nullable=allow_nan, coerce=True) for c in df.columns}
    schema = pa.DataFrameSchema(
        columns=schema_cols,
        index=pa.Index(pa.DateTime, nullable=False) if require_datetime_index else None,
        strict=False,
        coerce=True,
    )
    validated = schema.validate(df, lazy=True)
    if require_monotonic_index and not validated.index.is_monotonic_increasing:
        raise ValueError("Validated dataframe index is not monotonic increasing.")
    if (not allow_nan) and np.isnan(validated.values).any():
        raise ValueError("Validated dataframe still contains NaNs.")
    return validated
