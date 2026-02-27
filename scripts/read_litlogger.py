#!/usr/bin/env python3
"""Read LitLogger .litbin metrics from a Lightning run directory."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class LitBinData:
    path: Path
    split: str
    metric: str
    header: dict
    rows: list[dict]


def _resolve_logs_root(base: Path) -> Path:
    if base.name == "lightning_logs":
        return base
    candidate = base / "lightning_logs"
    if candidate.exists():
        return candidate
    return base


def _pick_run_dir(logs_root: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = logs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run id not found: {run_dir}")
        return run_dir

    dirs = [p for p in logs_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found in {logs_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def read_litbin(path: Path, split: str, metric: str) -> LitBinData:
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{path} is too small to be a valid litbin file")

    header_len = struct.unpack(">I", raw[:4])[0]
    header_raw = raw[4 : 4 + header_len]
    header = json.loads(header_raw.decode("utf-8"))

    payload = raw[4 + header_len :]
    usable = (len(payload) // 16) * 16
    if usable != len(payload):
        payload = payload[:usable]

    rows = []
    base_time = _parse_iso_utc(header.get("created_at"))
    for t_ms, step, value in struct.iter_unpack(">qff", payload):
        row = {
            "split": split,
            "metric": metric,
            "t_ms": int(t_ms),
            "step": float(step),
            "value": float(value),
        }
        if base_time is not None:
            row["timestamp_utc"] = (base_time + timedelta(milliseconds=t_ms)).isoformat()
        rows.append(row)

    return LitBinData(path=path, split=split, metric=metric, header=header, rows=rows)


def _metric_files(run_dir: Path, split: str, wanted: set[str] | None) -> Iterable[tuple[str, Path]]:
    split_dir = run_dir / split
    if not split_dir.exists():
        return
    for p in sorted(split_dir.glob("*.litbin")):
        metric = p.stem
        if wanted is None or metric in wanted:
            yield metric, p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read LitLogger .litbin files from lightning_logs run directories."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("runs"),
        help="Base run path. Can be a run dir or a lightning_logs dir. Default: runs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run directory name inside lightning_logs. Default: latest by modification time.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which metric split to load.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=None,
        help="Metric names to load (without .litbin), e.g. loss loss_total",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics in selected split(s) and exit.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Rows to preview per metric when not exporting only. Default: 5",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path for long-format table (all selected metrics/timestamps). "
            "Default: <run_dir>/all_metrics_all_timestamps.csv"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logs_root = _resolve_logs_root(args.path)
    run_dir = _pick_run_dir(logs_root, args.run_id)
    wanted = set(args.metric) if args.metric else None
    splits = ["train", "val"] if args.split == "both" else [args.split]

    print(f"logs_root: {logs_root}")
    print(f"run_dir:   {run_dir}")

    discovered: dict[str, list[str]] = {}
    all_data: list[LitBinData] = []
    for split in splits:
        items = list(_metric_files(run_dir, split, wanted))
        discovered[split] = [name for name, _ in items]
        for metric, path in items:
            all_data.append(read_litbin(path, split=split, metric=metric))

    if args.list_metrics:
        for split in splits:
            names = discovered.get(split, [])
            print(f"\n[{split}] {len(names)} metrics")
            for name in names:
                print(f"  - {name}")
        return

    if not all_data:
        raise FileNotFoundError("No matching .litbin files found for requested split/metric filters.")

    for d in all_data:
        if not d.rows:
            print(f"\n{d.split}/{d.metric}: empty")
            continue
        first = d.rows[0]
        last = d.rows[-1]
        print(
            f"\n{d.split}/{d.metric}: n={len(d.rows)} "
            f"step[{first['step']:.3f} -> {last['step']:.3f}] "
            f"value[{first['value']:.6f} -> {last['value']:.6f}]"
        )
        if args.tail > 0:
            tail_rows = d.rows[-args.tail :]
            keys = list(tail_rows[0].keys())
            print(" | ".join(keys))
            for row in tail_rows:
                vals = []
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        vals.append(f"{v:.6f}")
                    else:
                        vals.append(str(v))
                print(" | ".join(vals))

    out_csv = args.out_csv if args.out_csv is not None else (run_dir / "all_metrics_all_timestamps.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "split", "metric", "t_ms", "step", "value", "timestamp_utc"]
    wrote = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in all_data:
            for row in d.rows:
                writer.writerow(
                    {
                        "run_id": run_dir.name,
                        "split": row.get("split", ""),
                        "metric": row.get("metric", ""),
                        "t_ms": row.get("t_ms", ""),
                        "step": row.get("step", ""),
                        "value": row.get("value", ""),
                        "timestamp_utc": row.get("timestamp_utc", ""),
                    }
                )
                wrote += 1
    print(f"\nWrote {wrote} rows to {out_csv}")


if __name__ == "__main__":
    main()
