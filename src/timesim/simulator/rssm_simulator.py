"""Stateful simulator wrapper for RSSM world models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.dataset import GroupedTimeSeriesDataset
from ..data.schema import VariableSchema


@dataclass
class InputStats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray


class RSSMSimulator:
    """Stateful simulator over a trained latent RSSM model.

    Required API:
    - ``reset(history_df) -> self``
    - ``step(control_dict, exogenous_dict) -> {objective: {mean, std}}``
    - ``rollout(control_df, exogenous_df) -> pd.DataFrame``
    """

    def __init__(
        self,
        model,
        feature_columns: Sequence[str],
        input_columns: Sequence[str],
        output_columns: Sequence[str],
        in_idx: Sequence[int],
        out_idx: Sequence[int],
        control_positions: Sequence[int],
        known_exo_positions: Sequence[int],
        scaler=None,
        variable_schema: Optional[VariableSchema] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
        sigma_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.feature_columns = list(feature_columns)
        self.input_columns = list(input_columns)
        self.output_columns = list(output_columns)
        self.in_idx = list(in_idx)
        self.out_idx = list(out_idx)
        self.control_positions = list(control_positions)
        self.known_exo_positions = list(known_exo_positions)
        self.scaler = scaler
        self.normalization_stats = scaler
        self.variable_schema = variable_schema
        self.input_stats = input_stats or {}
        self.sigma_scale = float(max(1e-6, sigma_scale))

        self.feature_idx = {c: i for i, c in enumerate(self.feature_columns)}
        self.control_columns = [self.input_columns[i] for i in self.control_positions]
        self.exogenous_columns = [self.input_columns[i] for i in self.known_exo_positions]
        self.objective_columns = list(self.output_columns)

        self._state = None
        self._last_exogenous_raw: Optional[np.ndarray] = None
        self._last_warnings: list[str] = []
        self._last_latency_ms: float = 0.0
        self._last_reset_history_len: int = 0

    @classmethod
    def from_dataset(
        cls,
        model,
        dataset: GroupedTimeSeriesDataset,
        sigma_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> "RSSMSimulator":
        scaler = getattr(dataset, "scaler", None)
        values = dataset.values
        raw_values = scaler.inverse_transform(values) if scaler is not None else values
        feature_cols = list(getattr(dataset, "feature_cols", []))
        if not feature_cols:
            raise ValueError("Dataset is missing feature column metadata (feature_cols)")
        feature_idx = {c: i for i, c in enumerate(feature_cols)}

        control_cols = [dataset.input_cols[i] for i in dataset.control_positions]
        exo_cols = [dataset.input_cols[i] for i in dataset.known_exo_positions]

        def _stats_for(cols: List[str]) -> InputStats:
            if not cols:
                empty = np.zeros((0,), dtype=np.float32)
                return InputStats(mean=empty, std=np.ones((0,), dtype=np.float32), min=empty, max=empty)
            idx = [feature_idx[c] for c in cols]
            arr = np.asarray(raw_values[:, idx], dtype=np.float32)
            std = np.std(arr, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            return InputStats(
                mean=np.mean(arr, axis=0),
                std=std,
                min=np.min(arr, axis=0),
                max=np.max(arr, axis=0),
            )

        input_stats = {
            "control": _stats_for(control_cols),
            "exogenous": _stats_for(exo_cols),
        }

        return cls(
            model=model,
            feature_columns=dataset.feature_cols,
            input_columns=dataset.input_cols,
            output_columns=dataset.output_cols,
            in_idx=dataset.in_idx,
            out_idx=dataset.out_idx,
            control_positions=dataset.control_positions,
            known_exo_positions=dataset.known_exo_positions,
            scaler=scaler,
            variable_schema=getattr(dataset, "variable_schema", None),
            input_stats=input_stats,
            sigma_scale=sigma_scale,
            device=device,
        )

    def clone_empty(self) -> "RSSMSimulator":
        """Clone wrapper metadata while clearing latent state."""
        return RSSMSimulator(
            model=self.model,
            feature_columns=self.feature_columns,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
            in_idx=self.in_idx,
            out_idx=self.out_idx,
            control_positions=self.control_positions,
            known_exo_positions=self.known_exo_positions,
            scaler=self.scaler,
            variable_schema=self.variable_schema,
            input_stats=self.input_stats,
            sigma_scale=self.sigma_scale,
            device=self.device,
        )

    def schema(self) -> Dict[str, Any]:
        groups = (
            self.variable_schema.to_groups()
            if self.variable_schema is not None
            else {
                "control": list(self.control_columns),
                "exogenous": list(self.exogenous_columns),
                "objective": list(self.objective_columns),
            }
        )
        return {
            "groups": groups,
            "feature_columns": list(self.feature_columns),
            "input_columns": list(self.input_columns),
            "output_columns": list(self.output_columns),
            "device": str(self.device),
            "is_conditioned": bool(self._state is not None),
        }

    @property
    def last_warnings(self) -> list[str]:
        return list(self._last_warnings)

    @property
    def last_latency_ms(self) -> float:
        return float(self._last_latency_ms)

    @property
    def last_reset_history_len(self) -> int:
        return int(self._last_reset_history_len)

    def _apply_sigma_scale_to_samples(self, samples: np.ndarray) -> np.ndarray:
        if np.isclose(self.sigma_scale, 1.0):
            return samples.astype(np.float32, copy=False)
        mean = np.mean(samples, axis=0, keepdims=True)
        return (mean + self.sigma_scale * (samples - mean)).astype(np.float32, copy=False)

    def _to_scaled(self, df: pd.DataFrame) -> np.ndarray:
        arr = np.asarray(df[self.feature_columns].values, dtype=np.float32)
        if self.scaler is None:
            return arr
        return self.scaler.transform(arr).astype(np.float32, copy=False)

    def _inverse_outputs(self, y_scaled: np.ndarray) -> np.ndarray:
        arr = np.asarray(y_scaled, dtype=np.float32)
        if self.scaler is None:
            return arr
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        full = np.zeros((arr.shape[0], len(self.feature_columns)), dtype=np.float32)
        full[:, self.out_idx] = arr
        inv = self.scaler.inverse_transform(full)[:, self.out_idx]
        return inv[0] if squeeze else inv

    def _check_history_columns(self, history_df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in history_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _dict_or_array_to_vec(self, value: Any, columns: List[str], name: str) -> np.ndarray:
        if isinstance(value, Mapping):
            missing = [c for c in columns if c not in value]
            if missing:
                raise ValueError(f"Missing {name} variables: {missing}")
            return np.asarray([value[c] for c in columns], dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D vector for {name}, got shape {tuple(arr.shape)}")
        if arr.shape[0] != len(columns):
            raise ValueError(f"Expected {name} length {len(columns)}, got {arr.shape[0]}")
        return arr

    def _coerce_rollout_frame(
        self,
        value: Any,
        columns: List[str],
        frame_name: str,
        required_len: Optional[int] = None,
    ) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            missing = [c for c in columns if c not in value.columns]
            if missing:
                raise ValueError(f"Missing {frame_name} columns: {missing}")
            df = value.loc[:, columns].copy()
        else:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"{frame_name} must be a DataFrame or rank-2 array-like with shape (H, {len(columns)})"
                )
            if arr.shape[1] != len(columns):
                raise ValueError(
                    f"{frame_name} must have {len(columns)} columns, got {arr.shape[1]}"
                )
            df = pd.DataFrame(arr, columns=columns)

        if required_len is not None and len(df) != int(required_len):
            raise ValueError(f"{frame_name} length mismatch: expected {required_len}, got {len(df)}")
        return df

    def _scale_control_exogenous(
        self,
        control_raw: np.ndarray,
        exogenous_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler is None:
            return control_raw.astype(np.float32), exogenous_raw.astype(np.float32)
        row = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        for i, col in enumerate(self.control_columns):
            row[0, self.feature_idx[col]] = control_raw[i]
        for i, col in enumerate(self.exogenous_columns):
            row[0, self.feature_idx[col]] = exogenous_raw[i]
        scaled = self.scaler.transform(row).astype(np.float32, copy=False)
        control_scaled = np.asarray([scaled[0, self.feature_idx[c]] for c in self.control_columns], dtype=np.float32)
        exo_scaled = np.asarray([scaled[0, self.feature_idx[c]] for c in self.exogenous_columns], dtype=np.float32)
        return control_scaled, exo_scaled

    def _range_warnings(
        self,
        values_raw: np.ndarray,
        columns: List[str],
        stats_key: str,
    ) -> list[str]:
        warnings: list[str] = []
        stats = self.input_stats.get(stats_key, None)
        if stats is None or stats.mean.size != values_raw.size:
            return warnings
        z = np.abs((values_raw - stats.mean) / np.maximum(stats.std, 1e-8))
        for i, col in enumerate(columns):
            if z[i] > 2.0:
                warnings.append(
                    f"{stats_key} '{col}' is {float(z[i]):.2f} sigma from training mean "
                    f"(value={float(values_raw[i]):.4g}, mean={float(stats.mean[i]):.4g}, "
                    f"std={float(stats.std[i]):.4g}, train_min={float(stats.min[i]):.4g}, "
                    f"train_max={float(stats.max[i]):.4g})"
                )
        return warnings

    def _prediction_dict(self, mean: np.ndarray, std: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for i, col in enumerate(self.objective_columns):
            out[col] = {
                "mean": float(mean[i]),
                "std": float(std[i]),
            }
        return out

    def _samples_from_imagine(self, out: Dict[str, Any]) -> np.ndarray:
        if "samples" in out:
            arr = out["samples"].detach().cpu().numpy().astype(np.float32, copy=False)
            # (N,B,H,O) -> (N,H,O) when B=1
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
            elif arr.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected samples shape: {arr.shape}")
            return arr
        pred = out["predictions"].detach().cpu().numpy().astype(np.float32, copy=False)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        return pred[None, ...]

    def reset(self, historical_df: pd.DataFrame) -> "RSSMSimulator":
        """Condition internal state from historical data and return self."""
        self._check_history_columns(historical_df)
        scaled = self._to_scaled(historical_df)

        warmup_inputs = scaled[:, self.in_idx]
        warmup_outputs = scaled[:, self.out_idx]
        controls = (
            warmup_inputs[:, self.control_positions]
            if self.control_positions
            else np.zeros((len(warmup_inputs), 0), dtype=np.float32)
        )
        exogenous = (
            warmup_inputs[:, self.known_exo_positions]
            if self.known_exo_positions
            else np.zeros((len(warmup_inputs), 0), dtype=np.float32)
        )

        c_t = torch.from_numpy(controls).unsqueeze(0).to(self.device)
        x_t = torch.from_numpy(exogenous).unsqueeze(0).to(self.device)
        y_t = torch.from_numpy(warmup_outputs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            observed = self.model.observe(c_t, x_t, y_t, sample_posterior=False)
        self._state = observed["state"]
        if self.exogenous_columns:
            self._last_exogenous_raw = historical_df[self.exogenous_columns].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        else:
            self._last_exogenous_raw = np.zeros((0,), dtype=np.float32)
        self._last_warnings = []
        self._last_latency_ms = 0.0
        self._last_reset_history_len = int(len(historical_df))
        return self

    def step(
        self,
        control_values: Mapping[str, float] | Sequence[float] | np.ndarray,
        exogenous_values: Optional[Mapping[str, float] | Sequence[float] | np.ndarray] = None,
        n_samples: int = 50,
        return_details: bool = False,
    ) -> Dict[str, Dict[str, float]] | Dict[str, Any]:
        """Advance one step in imagine mode.

        Returns predictions in original scale:
        ``{objective_name: {"mean": float, "std": float}}``.
        """
        if self._state is None:
            raise RuntimeError("Simulator is not conditioned. Call reset() first.")

        start = perf_counter()
        control_raw = self._dict_or_array_to_vec(control_values, self.control_columns, "controls")
        if exogenous_values is None:
            if self.exogenous_columns:
                if self._last_exogenous_raw is None:
                    exogenous_raw = np.zeros((len(self.exogenous_columns),), dtype=np.float32)
                else:
                    exogenous_raw = self._last_exogenous_raw.astype(np.float32, copy=True)
            else:
                exogenous_raw = np.zeros((0,), dtype=np.float32)
        else:
            exogenous_raw = self._dict_or_array_to_vec(exogenous_values, self.exogenous_columns, "exogenous")

        warnings = []
        warnings.extend(self._range_warnings(control_raw, self.control_columns, "control"))
        warnings.extend(self._range_warnings(exogenous_raw, self.exogenous_columns, "exogenous"))

        control_scaled, exo_scaled = self._scale_control_exogenous(control_raw, exogenous_raw)
        c_t = torch.from_numpy(control_scaled).view(1, 1, -1).to(self.device)
        x_t = torch.from_numpy(exo_scaled).view(1, 1, -1).to(self.device)

        with torch.no_grad():
            out = self.model.imagine(
                initial_state=self._state,
                future_controls=c_t,
                future_exogenous=x_t,
                n_steps=1,
                n_samples=max(1, int(n_samples)),
                sample_latent=True,
            )

        self._state = out["state"]
        self._last_exogenous_raw = exogenous_raw.astype(np.float32, copy=True)

        samples_scaled = self._samples_from_imagine(out)  # (N,H=1,O)
        samples_scaled = self._apply_sigma_scale_to_samples(samples_scaled)
        samples_step = samples_scaled[:, 0, :]  # (N,O)
        samples = np.stack([self._inverse_outputs(s) for s in samples_step], axis=0)
        mean = np.mean(samples, axis=0).astype(np.float32, copy=False)
        std = np.std(samples, axis=0).astype(np.float32, copy=False)
        pred = self._prediction_dict(mean, std)

        self._last_warnings = sorted(set(warnings))
        self._last_latency_ms = (perf_counter() - start) * 1000.0
        if return_details:
            return {
                "predictions": pred,
                "warnings": self.last_warnings,
                "latency_ms": self.last_latency_ms,
            }
        return pred

    def rollout(
        self,
        control_trajectory: pd.DataFrame | Sequence[Sequence[float]] | np.ndarray,
        exogenous_trajectory: Optional[pd.DataFrame | Sequence[Sequence[float]] | np.ndarray] = None,
        n_samples: int = 50,
        return_details: bool = False,
    ) -> pd.DataFrame | Dict[str, Any]:
        """Roll forward N steps and return original-scale forecast DataFrame.

        Output columns per objective:
        ``<name>_mean``, ``<name>_std``, ``<name>_p5``, ``<name>_p95``.
        """
        if self._state is None:
            raise RuntimeError("Simulator is not conditioned. Call reset() first.")

        start = perf_counter()
        control_df = self._coerce_rollout_frame(control_trajectory, self.control_columns, "controls")
        horizon = int(len(control_df))
        if exogenous_trajectory is None:
            if self.exogenous_columns:
                if self._last_exogenous_raw is None:
                    exo_row = np.zeros((len(self.exogenous_columns),), dtype=np.float32)
                else:
                    exo_row = self._last_exogenous_raw.astype(np.float32, copy=True)
                exo_arr = np.repeat(exo_row.reshape(1, -1), horizon, axis=0)
                exogenous_df = pd.DataFrame(exo_arr, columns=self.exogenous_columns)
            else:
                exogenous_df = pd.DataFrame(np.zeros((horizon, 0), dtype=np.float32))
        else:
            exogenous_df = self._coerce_rollout_frame(
                exogenous_trajectory,
                self.exogenous_columns,
                "exogenous",
                required_len=horizon,
            )

        controls_raw = control_df[self.control_columns].to_numpy(dtype=np.float32, copy=False)
        exo_raw = exogenous_df[self.exogenous_columns].to_numpy(dtype=np.float32, copy=False)

        warnings: list[str] = []
        controls_scaled = np.zeros_like(controls_raw, dtype=np.float32)
        exo_scaled = np.zeros_like(exo_raw, dtype=np.float32)
        for t in range(horizon):
            warnings.extend(self._range_warnings(controls_raw[t], self.control_columns, "control"))
            warnings.extend(self._range_warnings(exo_raw[t], self.exogenous_columns, "exogenous"))
            c_s, x_s = self._scale_control_exogenous(controls_raw[t], exo_raw[t])
            controls_scaled[t] = c_s
            exo_scaled[t] = x_s

        c_t = torch.from_numpy(controls_scaled).unsqueeze(0).to(self.device)
        x_t = torch.from_numpy(exo_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.imagine(
                initial_state=self._state,
                future_controls=c_t,
                future_exogenous=x_t,
                n_steps=horizon,
                n_samples=max(1, int(n_samples)),
                sample_latent=True,
            )

        self._state = out["state"]
        self._last_exogenous_raw = exo_raw[-1].astype(np.float32, copy=True) if horizon > 0 else self._last_exogenous_raw

        samples_scaled = self._samples_from_imagine(out)  # (N,H,O)
        samples_scaled = self._apply_sigma_scale_to_samples(samples_scaled)
        samples = np.stack([self._inverse_outputs(s) for s in samples_scaled], axis=0)  # (N,H,O) original scale
        mean = np.mean(samples, axis=0).astype(np.float32, copy=False)  # (H,O)
        std = np.std(samples, axis=0).astype(np.float32, copy=False)  # (H,O)
        p5 = np.quantile(samples, 0.05, axis=0).astype(np.float32, copy=False)
        p95 = np.quantile(samples, 0.95, axis=0).astype(np.float32, copy=False)

        rows: Dict[str, np.ndarray] = {}
        for j, col in enumerate(self.objective_columns):
            rows[f"{col}_mean"] = mean[:, j]
            rows[f"{col}_std"] = std[:, j]
            rows[f"{col}_p5"] = p5[:, j]
            rows[f"{col}_p95"] = p95[:, j]
        result_df = pd.DataFrame(rows)

        self._last_warnings = sorted(set(warnings))
        self._last_latency_ms = (perf_counter() - start) * 1000.0
        if return_details:
            return {
                "predictions": result_df,
                "warnings": self.last_warnings,
                "latency_ms": self.last_latency_ms,
            }
        return result_df
