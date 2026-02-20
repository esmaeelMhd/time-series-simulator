#!/usr/bin/env python3
"""Run Phase-8 RSSM ablation studies and compare against a full model baseline."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from timesim.utils.config import load_config


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    slug: str
    title: str
    expected: str
    model_overrides: Dict[str, Any]
    prob_overrides: Dict[str, Any]


ABLATIONS: List[AblationSpec] = [
    AblationSpec(
        ablation_id="8.1",
        slug="no_aux_decoder",
        title="No auxiliary decoder",
        expected="Worse open-loop RMSE, especially at longer horizons.",
        model_overrides={
            "use_aux_decoder": False,
            "allow_disable_aux_decoder_for_ablation": True,
        },
        prob_overrides={"aux_weight": 0.0},
    ),
    AblationSpec(
        ablation_id="8.2",
        slug="no_rollout_training",
        title="No rollout training",
        expected="Worse open-loop RMSE for horizon > 10; similar at horizon 1.",
        model_overrides={},
        prob_overrides={"rollout_weight": 0.0},
    ),
    AblationSpec(
        ablation_id="8.3",
        slug="no_free_bits",
        title="No free bits",
        expected="Higher posterior-collapse risk; latent KL drops toward 0.",
        model_overrides={},
        prob_overrides={"kl_free_bits": 0.0, "use_free_bits": False},
    ),
    AblationSpec(
        ablation_id="8.4",
        slug="no_kl_balancing",
        title="No KL balancing",
        expected="Slightly worse prior quality / open-loop quality.",
        model_overrides={},
        prob_overrides={"kl_balance": 0.5, "use_kl_balancing": True},
    ),
    AblationSpec(
        ablation_id="8.5",
        slug="y_in_transition",
        title="Y in transition (causal violation)",
        expected="Often better reconstruction but worse interventional behavior.",
        model_overrides={
            "leak_objective_to_transition": True,
            "allow_objective_leak_for_ablation": True,
        },
        prob_overrides={},
    ),
    AblationSpec(
        ablation_id="8.6",
        slug="shared_encoder",
        title="Shared encoder",
        expected="Worse interventional quality due entangled role representations.",
        model_overrides={
            "share_encoder_weights": True,
            "allow_shared_encoder_for_ablation": True,
        },
        prob_overrides={},
    ),
    AblationSpec(
        ablation_id="8.7",
        slug="no_stochastic_path",
        title="No stochastic path",
        expected="Uncertainty quality degrades; calibration worsens.",
        model_overrides={
            "use_stochastic_path": False,
            "allow_disable_stochastic_for_ablation": True,
        },
        prob_overrides={},
    ),
    AblationSpec(
        ablation_id="8.8",
        slug="no_dtw_loss",
        title="No DTW loss",
        expected="Shape quality degrades when DTW is otherwise enabled.",
        model_overrides={},
        prob_overrides={"rollout_dtw_weight": 0.0},
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RSSM Phase-8 ablation study suite.")
    p.add_argument("--base-config", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ablations", nargs="*", default=None, help="Subset by slug, e.g. no_aux_decoder no_rollout_training")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs for all runs.")
    p.add_argument("--steps-per-epoch", type=int, default=None, help="Override training steps/epoch.")
    p.add_argument("--eval-horizon", type=int, default=None)
    p.add_argument("--eval-windows", type=int, default=None)
    p.add_argument("--mc-samples", type=int, default=None)
    p.add_argument("--sigma-scale", type=float, default=None)
    p.add_argument(
        "--dtw-weight",
        type=float,
        default=None,
        help="Override baseline rollout_dtw_weight for the suite (e.g. 0.2).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Reuse existing checkpoints/eval outputs when present.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latent_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models = cfg.get("models", [])
    for m in models:
        if isinstance(m, dict) and m.get("type") == "latent_ssm":
            return m
    raise ValueError("Expected a latent_ssm entry in config.models")


def _apply_overrides(
    cfg: Dict[str, Any],
    *,
    spec: Optional[AblationSpec],
    run_name: str,
    device: Optional[str],
    epochs: Optional[int],
    steps_per_epoch: Optional[int],
    dtw_weight: Optional[float],
) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("output", {})
    out["output"]["run_name"] = run_name
    if device is not None:
        out.setdefault("misc", {})
        out["misc"]["device"] = str(device)

    if epochs is not None:
        out.setdefault("training", {})
        out["training"]["epochs"] = int(epochs)
        for rd in out.get("training_rounds", []):
            if isinstance(rd, dict):
                rd["epochs"] = int(epochs)

    if steps_per_epoch is not None:
        out.setdefault("training", {})
        out["training"]["steps_per_epoch"] = int(steps_per_epoch)

    if spec is None:
        if dtw_weight is not None:
            out.setdefault("training", {})
            out["training"].setdefault("probabilistic", {})
            out["training"]["probabilistic"]["rollout_dtw_weight"] = float(dtw_weight)
        return out

    latent_cfg = _latent_model_cfg(out)
    for k, v in spec.model_overrides.items():
        latent_cfg[k] = v

    out.setdefault("training", {})
    out["training"].setdefault("probabilistic", {})
    if dtw_weight is not None:
        out["training"]["probabilistic"]["rollout_dtw_weight"] = float(dtw_weight)
    for k, v in spec.prob_overrides.items():
        out["training"]["probabilistic"][k] = v
    return out


def _run(cmd: List[str], *, dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _run_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    runs_dir = Path(cfg.get("output", {}).get("runs_dir", "runs"))
    dataset = str(cfg["dataset"]["name"])
    run_name = str(cfg.get("output", {}).get("run_name", "")).strip()
    if run_name:
        return runs_dir / dataset / run_name
    return runs_dir / dataset


def _checkpoint_path(run_dir: Path) -> Path:
    model_dir = run_dir / "latent_ssm"
    ckpt = model_dir / "train_checkpoint.pth"
    if ckpt.exists():
        return ckpt
    cands = sorted(model_dir.glob("*_checkpoint.pth"))
    if cands:
        return cands[-1]
    cands = sorted((model_dir / "checkpoints").glob("*.pth"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"No checkpoint found under: {model_dir}")


def _expected_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "latent_ssm" / "train_checkpoint.pth"


def _summary_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    def h_metric(split: str, metric: str, horizon: int) -> float:
        d = summary.get(split, {}).get(metric, {})
        if isinstance(d, dict):
            if horizon in d:
                return float(d[horizon])
            if str(horizon) in d:
                return float(d[str(horizon)])
        return float("nan")

    return {
        "rmse_h1": h_metric("open_loop_horizon_summary", "rmse", 1),
        "rmse_h10": h_metric("open_loop_horizon_summary", "rmse", 10),
        "rmse_h20": h_metric("open_loop_horizon_summary", "rmse", 20),
        "closed_rmse_h1": h_metric("closed_loop_horizon_summary", "rmse", 1),
        "coverage95": float(summary.get("coverage_95_mean", float("nan"))),
        "latent_kl": float(summary.get("latent_kl_mean", float("nan"))),
        "direction_aligned": float(summary.get("intervention_direction_mean_aligned", float("nan"))),
        "rmse_h20_over_h1": float(summary.get("open_loop_rmse_h20_over_h1", float("nan"))),
    }


def _evaluate_expectation(
    spec: AblationSpec,
    base_metrics: Dict[str, float],
    ab_metrics: Dict[str, float],
    base_cfg: Dict[str, Any],
) -> Tuple[Optional[bool], str]:
    if spec.ablation_id == "8.1":
        ok = np.isfinite(ab_metrics["rmse_h20"]) and np.isfinite(base_metrics["rmse_h20"]) and (
            ab_metrics["rmse_h20"] > base_metrics["rmse_h20"] * 1.01
        )
        return bool(ok), "rmse_h20 increased vs full"
    if spec.ablation_id == "8.2":
        near_h1 = np.isfinite(ab_metrics["rmse_h1"]) and np.isfinite(base_metrics["rmse_h1"]) and (
            ab_metrics["rmse_h1"] <= base_metrics["rmse_h1"] * 1.20
        )
        worse_long = np.isfinite(ab_metrics["rmse_h20"]) and np.isfinite(base_metrics["rmse_h20"]) and (
            ab_metrics["rmse_h20"] > base_metrics["rmse_h20"] * 1.01
        )
        return bool(near_h1 and worse_long), "h1 near baseline and h20 worse"
    if spec.ablation_id == "8.3":
        ok = np.isfinite(ab_metrics["latent_kl"]) and np.isfinite(base_metrics["latent_kl"]) and (
            ab_metrics["latent_kl"] < max(0.5, 0.7 * base_metrics["latent_kl"])
        )
        return bool(ok), "latent KL dropped vs full"
    if spec.ablation_id == "8.4":
        ok = np.isfinite(ab_metrics["rmse_h20"]) and np.isfinite(base_metrics["rmse_h20"]) and (
            ab_metrics["rmse_h20"] >= base_metrics["rmse_h20"]
        )
        return bool(ok), "open-loop h20 not better than full"
    if spec.ablation_id == "8.5":
        recon_better = np.isfinite(ab_metrics["closed_rmse_h1"]) and np.isfinite(base_metrics["closed_rmse_h1"]) and (
            ab_metrics["closed_rmse_h1"] <= base_metrics["closed_rmse_h1"]
        )
        intervention_worse = (
            np.isfinite(ab_metrics["direction_aligned"])
            and np.isfinite(base_metrics["direction_aligned"])
            and (ab_metrics["direction_aligned"] < base_metrics["direction_aligned"])
        )
        return bool(recon_better and intervention_worse), "closed-loop improved while direction score worsened"
    if spec.ablation_id == "8.6":
        ok = (
            np.isfinite(ab_metrics["direction_aligned"])
            and np.isfinite(base_metrics["direction_aligned"])
            and (ab_metrics["direction_aligned"] < base_metrics["direction_aligned"])
        )
        return bool(ok), "intervention direction alignment decreased"
    if spec.ablation_id == "8.7":
        base_err = abs(base_metrics["coverage95"] - 0.95) if np.isfinite(base_metrics["coverage95"]) else float("nan")
        ab_err = abs(ab_metrics["coverage95"] - 0.95) if np.isfinite(ab_metrics["coverage95"]) else float("nan")
        ok = np.isfinite(base_err) and np.isfinite(ab_err) and (ab_err > base_err + 0.01)
        return bool(ok), "coverage@95 moved further away from 0.95"
    if spec.ablation_id == "8.8":
        base_dtw_w = float(base_cfg.get("training", {}).get("probabilistic", {}).get("rollout_dtw_weight", 0.0))
        if base_dtw_w <= 0.0:
            return None, "skipped (baseline DTW weight <= 0)"
        ok = np.isfinite(ab_metrics["rmse_h20"]) and np.isfinite(base_metrics["rmse_h20"]) and (
            ab_metrics["rmse_h20"] > base_metrics["rmse_h20"]
        )
        return bool(ok), "h20 degraded after removing DTW"
    return None, "no heuristic"


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    root = _repo_root()
    train_script = root / "scripts" / "train.py"
    eval_script = root / "scripts" / "eval_rssm_suite.py"
    if not train_script.exists() or not eval_script.exists():
        raise FileNotFoundError("Expected scripts/train.py and scripts/eval_rssm_suite.py in repository.")

    base_cfg = load_config(args.base_config)
    base_run_name = str(base_cfg.get("output", {}).get("run_name", "ablation_full")).strip() or "ablation_full"
    selected = {s.lower() for s in (args.ablations or [])}
    specs = [s for s in ABLATIONS if not selected or s.slug.lower() in selected]
    if not specs:
        raise ValueError("No ablations selected.")

    suite_root = _run_dir_from_cfg(base_cfg).parent / f"{base_run_name}_phase8_ablation_suite"
    cfg_dir = suite_root / "configs"
    results_dir = suite_root / "results"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (full model)
    full_cfg = _apply_overrides(
        base_cfg,
        spec=None,
        run_name=f"{base_run_name}_full",
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        dtw_weight=args.dtw_weight,
    )
    full_cfg_path = cfg_dir / "full.yaml"
    _write_yaml(full_cfg_path, full_cfg)
    full_run_dir = _run_dir_from_cfg(full_cfg)
    full_ckpt = None
    try:
        if not (args.skip_existing and (full_run_dir / "latent_ssm" / "train_checkpoint.pth").exists()):
            _run(
                [sys.executable, str(train_script), "--config", str(full_cfg_path), "--models", "latent_ssm"],
                dry_run=args.dry_run,
            )
        if args.dry_run:
            full_ckpt = _expected_checkpoint_path(full_run_dir)
            print(f"[dry-run] expected baseline checkpoint: {full_ckpt}")
            full_metrics = {
                "rmse_h1": float("nan"),
                "rmse_h10": float("nan"),
                "rmse_h20": float("nan"),
                "closed_rmse_h1": float("nan"),
                "coverage95": float("nan"),
                "latent_kl": float("nan"),
                "direction_aligned": float("nan"),
                "rmse_h20_over_h1": float("nan"),
            }
        else:
            full_ckpt = _checkpoint_path(full_run_dir)
        full_eval_dir = full_run_dir / "latent_ssm" / "eval_suite"
        if not (args.skip_existing and (full_eval_dir / "summary.yaml").exists()):
            eval_cmd = [
                sys.executable,
                str(eval_script),
                "--config",
                str(full_cfg_path),
                "--checkpoint",
                str(full_ckpt),
                "--output-dir",
                str(full_eval_dir),
            ]
            if args.eval_horizon is not None:
                eval_cmd += ["--horizon", str(int(args.eval_horizon))]
            if args.eval_windows is not None:
                eval_cmd += ["--n-windows", str(int(args.eval_windows))]
            if args.mc_samples is not None:
                eval_cmd += ["--mc-samples", str(int(args.mc_samples))]
            if args.sigma_scale is not None:
                eval_cmd += ["--sigma-scale", str(float(args.sigma_scale))]
            if args.device is not None:
                eval_cmd += ["--device", str(args.device)]
            _run(eval_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            full_summary_path = full_eval_dir / "summary.yaml"
            full_summary = yaml.safe_load(full_summary_path.read_text(encoding="utf-8")) if full_summary_path.exists() else {}
            full_metrics = _summary_metrics(full_summary if isinstance(full_summary, dict) else {})
    except Exception as exc:
        raise RuntimeError(f"Failed baseline full-model run: {exc}") from exc

    rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {
        "baseline": {
            "config": str(full_cfg_path),
            "checkpoint": str(full_ckpt) if full_ckpt is not None else None,
            "metrics": full_metrics,
        },
        "ablations": {},
    }

    for spec in specs:
        run_name = f"{base_run_name}_{spec.slug}"
        ab_cfg = _apply_overrides(
            base_cfg,
            spec=spec,
            run_name=run_name,
            device=args.device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            dtw_weight=args.dtw_weight,
        )
        ab_cfg_path = cfg_dir / f"{spec.ablation_id.replace('.', '_')}_{spec.slug}.yaml"
        _write_yaml(ab_cfg_path, ab_cfg)
        ab_run_dir = _run_dir_from_cfg(ab_cfg)

        row: Dict[str, Any] = {
            "ablation_id": spec.ablation_id,
            "slug": spec.slug,
            "title": spec.title,
            "expected": spec.expected,
            "run_name": run_name,
            "config_path": str(ab_cfg_path),
            "status": "ok",
        }

        try:
            if not (args.skip_existing and (ab_run_dir / "latent_ssm" / "train_checkpoint.pth").exists()):
                _run(
                    [sys.executable, str(train_script), "--config", str(ab_cfg_path), "--models", "latent_ssm"],
                    dry_run=args.dry_run,
                )
            ab_ckpt = _expected_checkpoint_path(ab_run_dir) if args.dry_run else _checkpoint_path(ab_run_dir)
            row["checkpoint"] = str(ab_ckpt)

            ab_eval_dir = ab_run_dir / "latent_ssm" / "eval_suite"
            if not (args.skip_existing and (ab_eval_dir / "summary.yaml").exists()):
                eval_cmd = [
                    sys.executable,
                    str(eval_script),
                    "--config",
                    str(ab_cfg_path),
                    "--checkpoint",
                    str(ab_ckpt),
                    "--compare-checkpoint",
                    str(full_ckpt),
                    "--compare-config",
                    str(full_cfg_path),
                    "--main-label",
                    spec.slug,
                    "--compare-label",
                    "full",
                    "--output-dir",
                    str(ab_eval_dir),
                ]
                if args.eval_horizon is not None:
                    eval_cmd += ["--horizon", str(int(args.eval_horizon))]
                if args.eval_windows is not None:
                    eval_cmd += ["--n-windows", str(int(args.eval_windows))]
                if args.mc_samples is not None:
                    eval_cmd += ["--mc-samples", str(int(args.mc_samples))]
                if args.sigma_scale is not None:
                    eval_cmd += ["--sigma-scale", str(float(args.sigma_scale))]
                if args.device is not None:
                    eval_cmd += ["--device", str(args.device)]
                _run(eval_cmd, dry_run=args.dry_run)

            if args.dry_run:
                row["status"] = "dry_run"
                row["expectation_pass"] = None
                row["expectation_note"] = "not evaluated in dry-run mode"
                detail["ablations"][spec.slug] = {
                    "ablation_id": spec.ablation_id,
                    "config": str(ab_cfg_path),
                    "checkpoint": str(ab_ckpt),
                    "status": "dry_run",
                }
            else:
                summary_path = ab_eval_dir / "summary.yaml"
                summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                metrics = _summary_metrics(summary if isinstance(summary, dict) else {})
                row.update(metrics)

                expectation_pass, note = _evaluate_expectation(spec, full_metrics, metrics, base_cfg)
                row["expectation_pass"] = expectation_pass
                row["expectation_note"] = note
                detail["ablations"][spec.slug] = {
                    "ablation_id": spec.ablation_id,
                    "config": str(ab_cfg_path),
                    "checkpoint": str(ab_ckpt),
                    "metrics": metrics,
                    "expectation_pass": expectation_pass,
                    "expectation_note": note,
                }
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            detail["ablations"][spec.slug] = {
                "ablation_id": spec.ablation_id,
                "config": str(ab_cfg_path),
                "status": "failed",
                "error": str(exc),
            }
        rows.append(row)

    results_csv = results_dir / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    detail_yaml = results_dir / "ablation_results.yaml"
    _write_yaml(detail_yaml, detail)

    print("\n" + "=" * 70)
    print("  PHASE-8 ABLATION SUITE COMPLETE")
    print("=" * 70)
    print(f"  Baseline config : {full_cfg_path}")
    print(f"  Baseline ckpt   : {full_ckpt}")
    print(f"  Results CSV     : {results_csv}")
    print(f"  Results YAML    : {detail_yaml}")
    print("=" * 70)


if __name__ == "__main__":
    main()
