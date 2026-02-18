"""Stateful simulator wrapper for RSSM world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.dataset import GroupedTimeSeriesDataset


@dataclass
class InputStats:
    mean: np.ndarray
    std: np.ndarray


class RSSMSimulator:
    """Stateful simulator over a trained latent RSSM model.

    API:
    - reset(historical_df)
    - step(control_values, exogenous_values=None)
    - rollout(control_trajectory, exogenous_trajectory=None)
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
        self.input_stats = input_stats or {}
        self.sigma_scale = float(max(1e-6, sigma_scale))

        self.feature_idx = {c: i for i, c in enumerate(self.feature_columns)}
        self.control_columns = [self.input_columns[i] for i in self.control_positions]
        self.exogenous_columns = [self.input_columns[i] for i in self.known_exo_positions]

        self._state = None
        self._last_exogenous_scaled: Optional[np.ndarray] = None

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
        if scaler is not None:
            raw_values = scaler.inverse_transform(values)
        else:
            raw_values = values
        feature_cols = list(getattr(dataset, "feature_cols", []))
        if not feature_cols:
            raise ValueError("Dataset is missing feature column metadata (feature_cols)")
        feature_idx = {c: i for i, c in enumerate(feature_cols)}

        control_cols = [dataset.input_cols[i] for i in dataset.control_positions]
        exo_cols = [dataset.input_cols[i] for i in dataset.known_exo_positions]

        def _stats_for(cols: List[str]) -> InputStats:
            if not cols:
                return InputStats(mean=np.zeros((0,), dtype=np.float32), std=np.ones((0,), dtype=np.float32))
            idx = [feature_idx[c] for c in cols]
            arr = raw_values[:, idx]
            std = np.std(arr, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            return InputStats(mean=np.mean(arr, axis=0), std=std)

        input_stats = {
            "control": _stats_for(control_cols),
            "exogenous": _stats_for(exo_cols),
        }

        return cls(
            model=model,
            feature_columns=feature_cols,
            input_columns=dataset.input_cols,
            output_columns=dataset.output_cols,
            in_idx=dataset.in_idx,
            out_idx=dataset.out_idx,
            control_positions=dataset.control_positions,
            known_exo_positions=dataset.known_exo_positions,
            scaler=scaler,
            input_stats=input_stats,
            sigma_scale=sigma_scale,
            device=device,
        )

    def clone_empty(self) -> "RSSMSimulator":
        """Clone wrapper metadata while clearing latent state."""
        clone = RSSMSimulator(
            model=self.model,
            feature_columns=self.feature_columns,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
            in_idx=self.in_idx,
            out_idx=self.out_idx,
            control_positions=self.control_positions,
            known_exo_positions=self.known_exo_positions,
            scaler=self.scaler,
            input_stats=self.input_stats,
            sigma_scale=self.sigma_scale,
            device=self.device,
        )
        return clone

    def _apply_sigma_scale_to_samples(self, samples: np.ndarray) -> np.ndarray:
        """Scale sample spread around the per-step mean for post-hoc calibration."""
        if np.isclose(self.sigma_scale, 1.0):
            return samples.astype(np.float32, copy=False)
        mean = np.mean(samples, axis=0, keepdims=True)
        return (mean + self.sigma_scale * (samples - mean)).astype(np.float32, copy=False)

    def _check_columns(self, df: pd.DataFrame):
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _to_scaled(self, df: pd.DataFrame) -> np.ndarray:
        arr = df[self.feature_columns].values.astype(np.float32, copy=False)
        if self.scaler is None:
            return arr
        return self.scaler.transform(arr).astype(np.float32, copy=False)

    def _inverse_outputs(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return y_scaled
        full = np.zeros((y_scaled.shape[0], len(self.feature_columns)), dtype=np.float32)
        full[:, self.out_idx] = y_scaled
        inv = self.scaler.inverse_transform(full)
        return inv[:, self.out_idx]

    def _output_scale(self) -> np.ndarray:
        if self.scaler is None:
            return np.ones((len(self.out_idx),), dtype=np.float32)
        scale = (self.scaler.max - self.scaler.min + 1e-8).astype(np.float32, copy=False)
        return scale[self.out_idx]

    def _dict_or_array_to_vec(self, value, columns: List[str]) -> np.ndarray:
        if isinstance(value, dict):
            vec = np.asarray([value[c] for c in columns], dtype=np.float32)
            return vec
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Expected a 1D input vector")
        if arr.shape[0] != len(columns):
            raise ValueError(f"Expected vector length {len(columns)}, got {arr.shape[0]}")
        return arr

    def _scale_control_exogenous(self, control_raw: np.ndarray, exogenous_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler is None:
            return control_raw.astype(np.float32), exogenous_raw.astype(np.float32)

        row = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        for i, col in enumerate(self.control_columns):
            row[0, self.feature_idx[col]] = control_raw[i]
        for i, col in enumerate(self.exogenous_columns):
            row[0, self.feature_idx[col]] = exogenous_raw[i]

        scaled = self.scaler.transform(row)
        control_scaled = np.asarray([scaled[0, self.feature_idx[c]] for c in self.control_columns], dtype=np.float32)
        exo_scaled = np.asarray([scaled[0, self.feature_idx[c]] for c in self.exogenous_columns], dtype=np.float32)
        return control_scaled, exo_scaled

    def _range_warnings(self, control_raw: np.ndarray, exogenous_raw: np.ndarray) -> List[str]:
        warnings: List[str] = []
        cstats = self.input_stats.get("control")
        if cstats is not None and cstats.mean.size == control_raw.size:
            z = np.abs((control_raw - cstats.mean) / cstats.std)
            if np.any(z > 2.0):
                warnings.append("Control input exceeds 2σ from training distribution.")
        xstats = self.input_stats.get("exogenous")
        if xstats is not None and xstats.mean.size == exogenous_raw.size:
            z = np.abs((exogenous_raw - xstats.mean) / xstats.std)
            if np.any(z > 2.0):
                warnings.append("Exogenous input exceeds 2σ from training distribution.")
        return warnings

    def reset(self, historical_df: pd.DataFrame) -> Dict[str, int]:
        self._check_columns(historical_df)
        scaled = self._to_scaled(historical_df)

        warmup_inputs = scaled[:, self.in_idx]
        warmup_outputs = scaled[:, self.out_idx]
        controls = warmup_inputs[:, self.control_positions] if self.control_positions else np.zeros((len(warmup_inputs), 0), dtype=np.float32)
        exogenous = warmup_inputs[:, self.known_exo_positions] if self.known_exo_positions else np.zeros((len(warmup_inputs), 0), dtype=np.float32)

        c_t = torch.from_numpy(controls).unsqueeze(0).to(self.device)
        x_t = torch.from_numpy(exogenous).unsqueeze(0).to(self.device)
        y_t = torch.from_numpy(warmup_outputs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            observed = self.model.observe(c_t, x_t, y_t, sample_posterior=False)
        self._state = observed["state"]
        self._last_exogenous_scaled = exogenous[-1].astype(np.float32, copy=True) if exogenous.shape[-1] > 0 else np.zeros((0,), dtype=np.float32)
        return {"history_len": int(len(historical_df))}

    def step(
        self,
        control_values,
        exogenous_values=None,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray | List[str]]:
        if self._state is None:
            raise RuntimeError("Simulator is not conditioned. Call reset() first.")

        control_raw = self._dict_or_array_to_vec(control_values, self.control_columns)
        if exogenous_values is None:
            exogenous_raw = np.zeros((len(self.exogenous_columns),), dtype=np.float32)
            if self._last_exogenous_scaled is None:
                exo_scaled = np.zeros((len(self.exogenous_columns),), dtype=np.float32)
            else:
                exo_scaled = self._last_exogenous_scaled.astype(np.float32, copy=True)
            control_scaled, _ = self._scale_control_exogenous(control_raw, exogenous_raw)
        else:
            exogenous_raw = self._dict_or_array_to_vec(exogenous_values, self.exogenous_columns)
            control_scaled, exo_scaled = self._scale_control_exogenous(control_raw, exogenous_raw)

        warnings = self._range_warnings(control_raw, exogenous_raw)

        c_t = torch.from_numpy(control_scaled).view(1, 1, -1).to(self.device)
        x_t = torch.from_numpy(exo_scaled).view(1, 1, -1).to(self.device)

        with torch.no_grad():
            out = self.model.imagine(
                initial_state=self._state,
                future_controls=c_t,
                future_exogenous=x_t,
                n_steps=1,
                n_samples=n_samples,
                sample_latent=True,
            )

        self._state = out["state"]
        self._last_exogenous_scaled = exo_scaled

        if "samples" in out:
            samples_scaled = out["samples"].squeeze(2).squeeze(1).cpu().numpy()  # (N, O)
        else:
            samples_scaled = out["predictions"][0, 0, :].cpu().numpy()[None, :]
        samples_scaled = self._apply_sigma_scale_to_samples(samples_scaled)
        mean_scaled = np.mean(samples_scaled, axis=0)
        std_scaled = np.std(samples_scaled, axis=0)

        mean = self._inverse_outputs(mean_scaled.reshape(1, -1))[0]
        scale_vec = self._output_scale()
        std = std_scaled * scale_vec
        samples = np.stack([
            self._inverse_outputs(s.reshape(1, -1))[0] for s in samples_scaled
        ], axis=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
            "warnings": warnings,
        }

    def rollout(
        self,
        control_trajectory,
        exogenous_trajectory=None,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray | List[str]]:
        if self._state is None:
            raise RuntimeError("Simulator is not conditioned. Call reset() first.")

        controls_raw = np.asarray(control_trajectory, dtype=np.float32)
        if controls_raw.ndim != 2 or controls_raw.shape[1] != len(self.control_columns):
            raise ValueError(
                f"control_trajectory must have shape (H, {len(self.control_columns)})"
            )

        horizon = controls_raw.shape[0]
        if exogenous_trajectory is None:
            exo_raw = np.zeros((horizon, len(self.exogenous_columns)), dtype=np.float32)
        else:
            exo_raw = np.asarray(exogenous_trajectory, dtype=np.float32)
            if exo_raw.ndim != 2 or exo_raw.shape != (horizon, len(self.exogenous_columns)):
                raise ValueError(
                    f"exogenous_trajectory must have shape ({horizon}, {len(self.exogenous_columns)})"
                )

        warnings: List[str] = []
        controls_scaled = np.zeros_like(controls_raw, dtype=np.float32)
        exo_scaled = np.zeros_like(exo_raw, dtype=np.float32)
        for t in range(horizon):
            warnings.extend(self._range_warnings(controls_raw[t], exo_raw[t]))
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
                n_samples=n_samples,
                sample_latent=True,
            )

        self._state = out["state"]
        self._last_exogenous_scaled = exo_scaled[-1]

        if "samples" in out:
            samples_scaled = out["samples"].squeeze(1).cpu().numpy()  # (N, H, O)
        else:
            samples_scaled = out["predictions"].squeeze(0).cpu().numpy()[None, ...]
        samples_scaled = self._apply_sigma_scale_to_samples(samples_scaled)
        mean_scaled = np.mean(samples_scaled, axis=0)
        std_scaled = np.std(samples_scaled, axis=0)

        mean = self._inverse_outputs(mean_scaled)
        scale_vec = self._output_scale().reshape(1, -1)
        std = std_scaled * scale_vec
        samples = np.stack([
            self._inverse_outputs(s) for s in samples_scaled
        ], axis=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
            "warnings": sorted(set(warnings)),
        }
