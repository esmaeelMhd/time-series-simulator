from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timesim.data import (
    SlidingWindowRoleDataset,
    TimeSeriesDataModule,
    chronological_split_dataframe,
    held_out_eval_frame,
    resolve_split_ratios,
)
from timesim.data.preprocessing import (
    denormalize_array,
    fit_normalization_stats,
    normalize_array,
)
from timesim.data.validation import validate_time_series_dataframe, validate_variable_groups


def _make_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "u": np.linspace(0.0, 10.0, n),
            "x": np.linspace(-5.0, 5.0, n),
            "y": np.sin(np.linspace(0.0, 6.28, n)),
        },
        index=idx,
    )


def test_chronological_split_70_15_15_defaults():
    df = _make_df(100)
    ratios = resolve_split_ratios()
    train_df, val_df, test_df = chronological_split_dataframe(df, split_ratios=ratios)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()


def test_held_out_eval_frame_warmup_does_not_leak_targets():
    df = _make_df(1000)
    split_cfg = {"train": 0.70, "val": 0.15, "test": 0.15}
    _, _, chrono_test = chronological_split_dataframe(
        df, split_ratios=resolve_split_ratios(split_cfg=split_cfg)
    )
    true_test_start = len(df) - len(chrono_test)
    warmup_len = 10
    seq_len = 50  # previously used as pad; must not pull targets into val

    sliced = held_out_eval_frame(
        df, split_cfg=split_cfg, eval_test_split=0.15, warmup_len=warmup_len
    )
    context_start = len(df) - len(sliced)
    first_target = context_start + warmup_len
    assert first_target == true_test_start
    assert context_start == true_test_start - warmup_len
    # Using seq_len as pad would have placed the first target before the test boundary.
    assert first_target > true_test_start - (seq_len - warmup_len)


def test_held_out_eval_frame_rejects_split_larger_than_configured_test():
    df = _make_df(200)
    with pytest.raises(ValueError, match="exceeds data.splits.test"):
        held_out_eval_frame(
            df,
            split_cfg={"train": 0.70, "val": 0.15, "test": 0.15},
            eval_test_split=0.20,
            warmup_len=8,
        )


def test_train_only_normalization_and_round_trip():
    df = _make_df(120)
    train_df, val_df, _ = chronological_split_dataframe(df, split_ratios=(0.7, 0.15, 0.15))

    stats = fit_normalization_stats(train_df.values, feature_names=list(train_df.columns))
    val_norm = normalize_array(val_df.values, stats)

    # Because the data trends upward over time, val points should exceed train range.
    assert float(np.max(val_norm[:, 0])) > 1.0

    val_back = denormalize_array(val_norm, stats)
    assert np.allclose(val_back, val_df.values, atol=1e-5)


def test_symlog_round_trip():
    arr = np.array(
        [
            [-1000.0, -1.0],
            [-10.0, 0.5],
            [0.0, 2.0],
            [10.0, 3.0],
            [1000.0, 4.0],
        ],
        dtype=np.float32,
    )
    stats = fit_normalization_stats(
        arr,
        feature_names=["a", "b"],
        use_symlog=True,
        symlog_columns=["a"],
    )
    back = denormalize_array(normalize_array(arr, stats), stats)
    assert np.allclose(back, arr, atol=1e-5)


def test_sliding_window_role_dataset_shapes():
    df = _make_df(64)
    groups = {"control": ["u"], "exogenous": ["x"], "objective": ["y"]}
    ds = SlidingWindowRoleDataset(df, groups=groups, seq_len=8, pred_len=3, stride=2)

    expected_len = ((len(df) - (8 + 3)) // 2) + 1
    assert len(ds) == expected_len

    sample = ds[0]
    assert sample["control"].shape == (8, 1)
    assert sample["exogenous"].shape == (8, 1)
    assert sample["objective"].shape == (8, 1)
    assert sample["target_objective"].shape == (3, 1)


def test_validation_fails_loud_and_early():
    validate_variable_groups({"control": ["u"], "exogenous": ["x"], "objective": ["y"]})

    with pytest.raises(ValueError, match="multiple variable groups"):
        validate_variable_groups({"control": ["u"], "exogenous": ["u"], "objective": ["y"]})

    df = _make_df(16)
    df_bad_nan = df.copy()
    df_bad_nan.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaNs detected"):
        validate_time_series_dataframe(df_bad_nan, allow_nan=False)

    df_bad_dtype = df.copy()
    df_bad_dtype["x"] = "bad"
    with pytest.raises(ValueError, match="Non-numeric columns"):
        validate_time_series_dataframe(df_bad_dtype)

    df_bad_index = df.sort_index(ascending=False)
    with pytest.raises(ValueError, match="monotonic increasing"):
        validate_time_series_dataframe(df_bad_index)


def test_datamodule_fits_train_stats_and_saves(tmp_path):
    df = _make_df(200).reset_index().rename(columns={"index": "date"})
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "dataset": {
            "csv": str(csv_path),
            "index_col": "date",
            "seq_len": 12,
            "pred_len": 2,
            "batch_size": 8,
            "variables": {"control": ["u"], "exogenous": ["x"], "objective": ["y"]},
        },
        "data": {
            "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
            "window_stride": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
            "shuffle_train": False,
            "require_full_role_mapping": True,
            "validation": {
                "strict": False,
                "require_datetime_index": True,
                "require_monotonic_index": True,
                "require_numeric_dtypes": True,
                "allow_nan": False,
            },
            "symlog": {"enabled": False, "columns": None},
        },
    }

    norm_path = tmp_path / "normalization_stats.pkl"
    dm = TimeSeriesDataModule(config=cfg, normalization_stats_path=norm_path)
    dm.setup()

    assert norm_path.exists()
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None
    assert dm.train_dataloader().drop_last is True
    assert dm.val_dataloader().drop_last is False
    assert dm.test_dataloader().drop_last is False
    assert dm.normalization_stats is not None

    raw = _make_df(200)
    train_df, _, _ = chronological_split_dataframe(raw, split_ratios=(0.7, 0.15, 0.15))
    expected_min = train_df[["u", "x", "y"]].values.min(axis=0)
    assert np.allclose(dm.normalization_stats.min, expected_min, atol=1e-6)
