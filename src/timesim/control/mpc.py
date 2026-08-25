"""Latent-space MPC controller using Cross-Entropy Method (CEM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from timesim.models.factory import build_model
from timesim.models.rssm import RSSMState

Tensor = torch.Tensor


@dataclass
class _CheckpointSelection:
    path: Path
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    epoch_1based: Optional[int] = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data).__name__}")
    return data


def _resolve_best_checkpoint(model_dir: str | Path) -> _CheckpointSelection:
    """Resolve best checkpoint from model directory.

    Priority:
    1. Best epoch by ``open_loop_crps`` in ``metrics.csv`` -> ``checkpoints/epochXXX.ckpt``
    2. ``train_checkpoint.pth``
    3. Latest ``*_checkpoint.pth``
    4. Latest ``checkpoints/*.ckpt``
    """
    model_dir = Path(model_dir)
    metrics_path = model_dir / "metrics.csv"
    ckpt_dir = model_dir / "checkpoints"

    if metrics_path.exists() and ckpt_dir.exists():
        try:
            df = pd.read_csv(metrics_path)
            if "open_loop_crps" in df.columns and "epoch" in df.columns:
                valid = df[pd.notna(df["open_loop_crps"]) & pd.notna(df["epoch"])].copy()
                if len(valid) > 0:
                    best = valid.sort_values("open_loop_crps", ascending=True).iloc[0]
                    epoch_1based = int(best["epoch"])
                    epoch_0based = max(0, epoch_1based - 1)
                    candidate = ckpt_dir / f"epoch{epoch_0based:03d}.ckpt"
                    if candidate.exists():
                        return _CheckpointSelection(
                            path=candidate,
                            metric_name="open_loop_crps",
                            metric_value=float(best["open_loop_crps"]),
                            epoch_1based=epoch_1based,
                        )
        except Exception:
            pass

    candidate = model_dir / "train_checkpoint.pth"
    if candidate.exists():
        return _CheckpointSelection(path=candidate)

    pth_ckpts = sorted(model_dir.glob("*_checkpoint.pth"))
    if pth_ckpts:
        return _CheckpointSelection(path=pth_ckpts[-1])

    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if ckpts:
        return _CheckpointSelection(path=ckpts[-1])

    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def _extract_model_state_dict(raw_state: Any) -> dict[str, Tensor]:
    state = raw_state
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        # Lightning checkpoint format
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(state).__name__}")

    out: dict[str, Tensor] = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        key = k[6:] if k.startswith("model.") else k
        out[key] = v
    return out


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = _extract_model_state_dict(raw)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)


class CEMController:
    """Cross-Entropy Method controller on latent RSSM dynamics."""

    def __init__(
        self,
        model: torch.nn.Module,
        horizon: int,
        action_dim: int,
        exogenous_dim: int,
        population: int = 1000,
        iterations: int = 5,
        elite_frac: float = 0.1,
        init_std: float = 0.5,
        min_std: float = 1e-3,
        device: str | torch.device = "cpu",
        sample_latent: bool = False,
        action_low: Optional[Tensor | np.ndarray | list[float]] = None,
        action_high: Optional[Tensor | np.ndarray | list[float]] = None,
        momentum: float = 0.0,
    ):
        self.model = model
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.exogenous_dim = int(exogenous_dim)
        self.population = int(population)
        self.iterations = int(iterations)
        self.elite_frac = float(elite_frac)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.device = torch.device(device)
        self.sample_latent = bool(sample_latent)
        self.momentum = float(momentum)

        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if self.exogenous_dim < 0:
            raise ValueError(f"exogenous_dim must be >= 0, got {self.exogenous_dim}")
        if self.population <= 0:
            raise ValueError(f"population must be > 0, got {self.population}")
        if self.iterations <= 0:
            raise ValueError(f"iterations must be > 0, got {self.iterations}")
        if not (0.0 < self.elite_frac <= 1.0):
            raise ValueError(f"elite_frac must be in (0,1], got {self.elite_frac}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be > 0, got {self.min_std}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"momentum must be in [0,1), got {self.momentum}")

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.action_low = self._format_action_bound(action_low, "action_low")
        self.action_high = self._format_action_bound(action_high, "action_high")
        if self.action_low is not None and self.action_high is not None:
            if torch.any(self.action_low > self.action_high):
                raise ValueError("action_low must be <= action_high for all dimensions")

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        horizon: int,
        model_type: str = "latent_ssm",
        device: str | torch.device = "cpu",
        population: int = 1000,
        iterations: int = 5,
        elite_frac: float = 0.1,
        init_std: float = 0.5,
        min_std: float = 1e-3,
        sample_latent: bool = False,
        action_low: Optional[Tensor | np.ndarray | list[float]] = None,
        action_high: Optional[Tensor | np.ndarray | list[float]] = None,
        momentum: float = 0.0,
    ) -> "CEMController":
        """Build controller from a trained run directory.

        ``run_dir`` can point either to ``.../<run>/<model_type>`` or ``.../<run>``.
        """
        run_dir = Path(run_dir)
        model_dir = run_dir / model_type if (run_dir / model_type).exists() else run_dir
        model_cfg_path = model_dir / "model_config.yaml"
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Missing model config: {model_cfg_path}")

        model_cfg = _load_yaml(model_cfg_path)
        resolved = model_cfg.get("resolved", {}) if isinstance(model_cfg.get("resolved"), dict) else {}
        data_resolved = resolved.get("data", {}) if isinstance(resolved.get("data"), dict) else {}
        model_params = resolved.get("model_params", {}) if isinstance(resolved.get("model_params"), dict) else {}

        input_dim = data_resolved.get("input_dim")
        output_dim = data_resolved.get("output_dim")
        seq_len = data_resolved.get("seq_len")
        pred_len = data_resolved.get("pred_len")
        if any(v is None for v in (input_dim, output_dim, seq_len, pred_len)):
            raise ValueError(
                f"model_config.yaml at {model_cfg_path} is missing resolved data dimensions"
            )

        control_dim = model_params.get("control_dim")
        exogenous_dim = model_params.get("exogenous_dim")
        # fallback from schema when explicit dims were not persisted
        if control_dim is None:
            var_schema = data_resolved.get("variable_schema", {})
            if isinstance(var_schema, dict):
                control_dim = len(var_schema.get("control", []) or [])
        if exogenous_dim is None:
            var_schema = data_resolved.get("variable_schema", {})
            if isinstance(var_schema, dict):
                exogenous_dim = len(var_schema.get("exogenous", []) or [])
        if exogenous_dim is None:
            raise ValueError("Could not infer exogenous_dim from model_config.yaml")

        model = build_model(
            model_type=model_type,
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            control_dim=(None if control_dim is None else int(control_dim)),
            exogenous_dim=int(exogenous_dim),
            per_model_cfg={"type": model_type, **model_params},
            model_defaults_cfg={},
            overrides=None,
        )

        selected = _resolve_best_checkpoint(model_dir)
        _load_model_state(model, selected.path, torch.device(device))
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        inferred_action_dim = int(getattr(model, "control_dim", control_dim if control_dim is not None else 0))
        inferred_exo_dim = int(getattr(model, "exogenous_dim", exogenous_dim))
        if inferred_action_dim <= 0:
            raise ValueError("Could not infer positive action_dim from loaded model/config")

        return cls(
            model=model,
            horizon=int(horizon),
            action_dim=inferred_action_dim,
            exogenous_dim=inferred_exo_dim,
            population=population,
            iterations=iterations,
            elite_frac=elite_frac,
            init_std=init_std,
            min_std=min_std,
            device=device,
            sample_latent=sample_latent,
            action_low=action_low,
            action_high=action_high,
            momentum=momentum,
        )

    def _format_action_bound(
        self, value: Optional[Tensor | np.ndarray | list[float]], name: str
    ) -> Optional[Tensor]:
        if value is None:
            return None
        t = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if t.ndim == 0:
            t = t.expand(self.action_dim)
        if t.ndim != 1 or t.shape[0] != self.action_dim:
            raise ValueError(
                f"{name} must be scalar or shape [{self.action_dim}], got {tuple(t.shape)}"
            )
        return t

    def _to_rssm_state(self, initial_state: Any) -> RSSMState:
        if isinstance(initial_state, RSSMState):
            h, z = initial_state.h, initial_state.z
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            h, z = initial_state
        elif isinstance(initial_state, dict) and "h" in initial_state and "z" in initial_state:
            h, z = initial_state["h"], initial_state["z"]
        else:
            raise TypeError(
                "initial_state must be RSSMState, (h, z) tuple, or {'h':..., 'z':...} dict"
            )

        h_t = torch.as_tensor(h, device=self.device)
        z_t = torch.as_tensor(z, device=self.device)
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if h_t.ndim != 2 or z_t.ndim != 2:
            raise ValueError(f"h and z must be rank-2 after batching, got {h_t.ndim} and {z_t.ndim}")
        if h_t.shape[0] != 1 or z_t.shape[0] != 1:
            raise ValueError(
                f"CEMController expects batch size 1 initial state, got h={tuple(h_t.shape)}, z={tuple(z_t.shape)}"
            )
        return RSSMState(h=h_t, z=z_t)

    def _clip_actions(self, actions: Tensor) -> Tensor:
        if self.action_low is not None:
            actions = torch.maximum(actions, self.action_low.view(1, 1, -1))
        if self.action_high is not None:
            actions = torch.minimum(actions, self.action_high.view(1, 1, -1))
        return actions

    def optimize(
        self,
        initial_state: Any,
        future_exogenous: Tensor | np.ndarray,
        cost_function: Callable[[Tensor, Tensor], Tensor],
    ) -> dict[str, Tensor | float | list[float]]:
        if not callable(cost_function):
            raise TypeError("cost_function must be callable")

        exo = torch.as_tensor(future_exogenous, device=self.device)
        if exo.ndim != 2:
            raise ValueError(
                f"future_exogenous must have shape [horizon, exogenous_dim], got {tuple(exo.shape)}"
            )
        if exo.shape[0] != self.horizon:
            raise ValueError(f"Expected horizon={self.horizon}, got {exo.shape[0]}")
        if exo.shape[1] != self.exogenous_dim:
            raise ValueError(f"Expected exogenous_dim={self.exogenous_dim}, got {exo.shape[1]}")

        state = self._to_rssm_state(initial_state)
        dtype = state.h.dtype
        exo = exo.to(dtype=dtype)

        mean = torch.zeros(
            (self.horizon, self.action_dim),
            device=self.device,
            dtype=dtype,
        )
        std = torch.full_like(mean, fill_value=self.init_std).clamp_min(self.min_std)

        elite_count = max(1, int(self.elite_frac * self.population))
        cost_history: list[float] = []
        best_cost = float("inf")
        best_actions: Optional[Tensor] = None
        best_pred: Optional[Tensor] = None
        best_elites: Optional[Tensor] = None

        with torch.no_grad():
            for _ in range(self.iterations):
                eps = torch.randn(
                    (self.population, self.horizon, self.action_dim),
                    device=self.device,
                    dtype=dtype,
                )
                actions = mean.unsqueeze(0) + std.unsqueeze(0) * eps
                actions = self._clip_actions(actions)

                exo_batch = exo.unsqueeze(0).expand(self.population, -1, -1)
                init = RSSMState(
                    h=state.h.expand(self.population, -1),
                    z=state.z.expand(self.population, -1),
                )

                rollout = self.model.imagine(
                    initial_state=init,
                    future_controls=actions,
                    future_exogenous=exo_batch,
                    n_steps=self.horizon,
                    n_samples=1,
                    sample_latent=self.sample_latent,
                )
                if "predictions" not in rollout:
                    raise KeyError("model.imagine(...) output missing 'predictions'")
                y_preds = rollout["predictions"]
                if not isinstance(y_preds, torch.Tensor):
                    raise TypeError("model.imagine(...) 'predictions' must be a torch.Tensor")
                if y_preds.ndim != 3 or y_preds.shape[:2] != (self.population, self.horizon):
                    raise ValueError(
                        "Expected predictions shape [N, H, dim_y], "
                        f"got {tuple(y_preds.shape)}"
                    )

                costs = cost_function(y_preds, actions)
                if not isinstance(costs, torch.Tensor):
                    costs = torch.as_tensor(costs, device=self.device, dtype=dtype)
                else:
                    costs = costs.to(device=self.device, dtype=dtype)
                if costs.ndim != 1 or costs.shape[0] != self.population:
                    raise ValueError(
                        "cost_function must return shape [N], "
                        f"got {tuple(costs.shape)}"
                    )
                if not torch.all(torch.isfinite(costs)):
                    raise ValueError("cost_function returned non-finite values")

                min_cost, min_idx = torch.min(costs, dim=0)
                min_cost_f = float(min_cost.item())
                cost_history.append(min_cost_f)
                if min_cost_f < best_cost:
                    best_cost = min_cost_f
                    best_actions = actions[min_idx].clone()
                    best_pred = y_preds[min_idx].clone()

                elite_idx = torch.topk(costs, k=elite_count, largest=False).indices
                elite_actions = actions.index_select(0, elite_idx)
                elite_mean = elite_actions.mean(dim=0)
                elite_std = elite_actions.std(dim=0, unbiased=False).clamp_min(self.min_std)

                m = self.momentum
                mean = m * mean + (1.0 - m) * elite_mean
                std = (m * std + (1.0 - m) * elite_std).clamp_min(self.min_std)
                best_elites = elite_idx

        if best_actions is None or best_pred is None:
            raise RuntimeError("CEM optimization failed to produce any candidate")

        return {
            "action_t0": best_actions[0].detach(),
            "trajectory": best_pred.detach(),
            "best_actions": best_actions.detach(),
            "best_cost": float(best_cost),
            "mean": mean.detach(),
            "std": std.detach(),
            "cost_history": cost_history,
            "elite_indices": best_elites.detach() if best_elites is not None else None,
        }


__all__ = ["CEMController"]
