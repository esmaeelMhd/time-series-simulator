"""Optional dataframe validation powered by Pandera."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def validate_time_series_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: Optional[Iterable[str]] = None,
    strict: bool = False,
    require_datetime_index: bool = True,
) -> pd.DataFrame:
    """Validate time-series dataframe schema.

    Parameters
    ----------
    df:
        Input dataframe.
    required_columns:
        Optional expected columns. If provided, all must exist.
    strict:
        If True and Pandera is unavailable, raises ImportError.
    require_datetime_index:
        If True, validates index type is DatetimeIndex.
    """
    req_cols = list(required_columns or [])
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataframe columns: {missing}")

    if require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected pandas.DatetimeIndex for time-series dataframe")

    try:
        import pandera as pa  # type: ignore
    except Exception:
        if strict:
            raise ImportError(
                "Pandera validation requested in strict mode but pandera is not installed."
            )
        return df

    schema_cols = {c: pa.Column(float, nullable=False, coerce=True) for c in df.columns}
    schema = pa.DataFrameSchema(
        columns=schema_cols,
        index=pa.Index(pa.DateTime, nullable=False) if require_datetime_index else None,
        strict=False,
        coerce=True,
    )
    validated = schema.validate(df, lazy=True)
    return validated
