"""Optional experiment tracking helpers (W&B / MLflow / no-op).

This module intentionally keeps third-party integrations optional so
the core training pipeline works without extra dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


class ExperimentTracker:
    """Backend-agnostic tracker wrapper.

    Supported backends:
    - ``none``: no-op
    - ``wandb``
    - ``mlflow``
    """

    def __init__(
        self,
        backend: str = "none",
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        run_dir: Optional[str | Path] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        self.backend = str(backend or "none").lower()
        self.project = project
        self.run_name = run_name
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._wandb = None
        self._mlflow = None
        self._active = False

        if self.backend in {"", "none", "disabled", "off"}:
            self.backend = "none"
            return

        if self.backend == "wandb":
            self._init_wandb(config=config, tags=tags)
            return
        if self.backend == "mlflow":
            self._init_mlflow(config=config, tags=tags)
            return

        print(f"Warning: unknown tracking backend '{self.backend}', disabling tracking.")
        self.backend = "none"

    def _init_wandb(self, config: Optional[Dict[str, Any]], tags: Optional[list[str]]) -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            print("Warning: wandb tracking requested but wandb is not installed. Disabling tracking.")
            self.backend = "none"
            return

        init_kwargs: Dict[str, Any] = {}
        if self.project:
            init_kwargs["project"] = self.project
        if self.run_name:
            init_kwargs["name"] = self.run_name
        if self.run_dir is not None:
            init_kwargs["dir"] = str(self.run_dir)
        if tags:
            init_kwargs["tags"] = list(tags)
        if config:
            init_kwargs["config"] = config
        wandb.init(**init_kwargs)
        self._wandb = wandb
        self._active = True

    def _init_mlflow(self, config: Optional[Dict[str, Any]], tags: Optional[list[str]]) -> None:
        try:
            import mlflow  # type: ignore
        except Exception:
            print("Warning: mlflow tracking requested but mlflow is not installed. Disabling tracking.")
            self.backend = "none"
            return

        if self.project:
            mlflow.set_experiment(self.project)
        mlflow.start_run(run_name=self.run_name)
        if tags:
            mlflow.set_tags({f"tag_{i}": t for i, t in enumerate(tags)})
        if config:
            mlflow.log_params({k: str(v) for k, v in _flatten_dict(config).items()})
        self._mlflow = mlflow
        self._active = True

    @property
    def is_active(self) -> bool:
        return bool(self._active)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._active:
            return
        flat = _flatten_dict(params)
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.config.update(flat, allow_val_change=True)
            return
        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.log_params({k: str(v) for k, v in flat.items()})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self._active:
            return
        cleaned = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if not cleaned:
            return
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.log(cleaned, step=step)
            return
        if self.backend == "mlflow" and self._mlflow is not None:
            if step is None:
                self._mlflow.log_metrics(cleaned)
            else:
                self._mlflow.log_metrics(cleaned, step=int(step))

    def log_artifact(self, path: str | Path, artifact_path: Optional[str] = None) -> None:
        if not self._active:
            return
        p = Path(path)
        if not p.exists():
            return
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.save(str(p))
            return
        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.log_artifact(str(p), artifact_path=artifact_path)

    def finish(self) -> None:
        if not self._active:
            return
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.finish()
        elif self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.end_run()
        self._active = False
