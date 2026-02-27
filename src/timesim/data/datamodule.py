"""Lightning DataModule for chronological train/val/test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl  # type: ignore
except Exception:
    pl = None

from .dataset import SlidingWindowRoleDataset
from .loader import (
    chronological_split_dataframe,
    load_csv_dataset,
    resolve_split_ratios,
)
from .validation import validate_variable_groups
from .preprocessing import fit_normalization_stats


class TimeSeriesDataModule((pl.LightningDataModule if pl is not None else object)):  # type: ignore[misc]
    """Config-driven datamodule with train-only normalization fitting."""

    def __init__(self, config: Dict[str, Any], normalization_stats_path: Optional[str | Path] = None):
        if pl is not None:
            super().__init__()
        self.config = config
        self.normalization_stats_path = Path(normalization_stats_path) if normalization_stats_path else None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._stats = None

    def setup(self, stage: str | None = None) -> None:
        dcfg = self.config["dataset"]
        data_cfg = self.config.get("data", {})

        df = load_csv_dataset(
            dcfg["csv"],
            index_col=dcfg.get("index_col", data_cfg.get("index_col", "date")),
            parse_dates=bool(data_cfg.get("parse_dates", True)),
            slice_cfg=dcfg.get("slice"),
            engine=str(data_cfg.get("csv_engine", "pandas")),
            validation_cfg=data_cfg.get("validation", None),
        )
        groups = dcfg["variables"]
        validate_variable_groups(groups)

        split_cfg = data_cfg.get("splits", None)
        ratios = resolve_split_ratios(
            split_cfg=split_cfg,
            train_split=None,
            default=(0.70, 0.15, 0.15),
        )
        train_df, val_df, test_df = chronological_split_dataframe(df, split_ratios=ratios)

        seq_len = int(dcfg["seq_len"])
        pred_len = int(dcfg["pred_len"])
        stride = int(data_cfg.get("window_stride", 1))

        symlog_cfg = data_cfg.get("symlog", {}) or {}
        use_symlog = bool(symlog_cfg.get("enabled", False))
        symlog_columns = symlog_cfg.get("columns", None)
        if isinstance(symlog_columns, str) and symlog_columns.lower() == "all":
            symlog_columns = list(df.columns)

        self._stats = fit_normalization_stats(
            train_df[list(df.columns)].values,
            feature_names=list(df.columns),
            use_symlog=use_symlog,
            symlog_columns=symlog_columns,
        )

        if self.normalization_stats_path is not None:
            from joblib import dump

            self.normalization_stats_path.parent.mkdir(parents=True, exist_ok=True)
            dump(self._stats, self.normalization_stats_path)

        self._train_ds = SlidingWindowRoleDataset(
            train_df,
            groups=groups,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            normalization_stats=self._stats,
            fit_stats=False,
            require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
        )
        self._val_ds = SlidingWindowRoleDataset(
            val_df,
            groups=groups,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            normalization_stats=self._stats,
            fit_stats=False,
            require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
        )
        self._test_ds = SlidingWindowRoleDataset(
            test_df,
            groups=groups,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            normalization_stats=self._stats,
            fit_stats=False,
            require_full_role_mapping=bool(data_cfg.get("require_full_role_mapping", True)),
        )

        batch_size = int(dcfg.get("batch_size", data_cfg.get("batch_size", 32)))
        num_workers = int(data_cfg.get("num_workers", 0))
        pin_memory = bool(data_cfg.get("pin_memory", False))
        drop_last = bool(data_cfg.get("drop_last", True))
        shuffle_train = bool(data_cfg.get("shuffle_train", True))

        self._train_loader = DataLoader(
            self._train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self._val_loader = DataLoader(
            self._val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self._test_loader = DataLoader(
            self._test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    @property
    def normalization_stats(self):
        return self._stats

    @property
    def scaler(self):
        # backward-compatible alias
        return self._stats
