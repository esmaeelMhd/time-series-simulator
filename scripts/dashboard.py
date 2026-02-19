#!/usr/bin/env python3
"""Streamlit demo for interactive RSSM what-if simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

from timesim.data.loader import build_grouped_dataloaders
from timesim.models.factory import build_model
from timesim.simulator import RSSMSimulator
from timesim.training import WorldModelTrainer
from timesim.data.sampling import RandomStartFixedHorizon


st.set_page_config(page_title="TimeSim RSSM Simulator", layout="wide")
st.title("TimeSim RSSM World Model Simulator")


@st.cache_data(show_spinner=False)
def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

raw_df = _read_csv(uploaded)
st.write("Rows:", len(raw_df), "Columns:", list(raw_df.columns))

all_cols = list(raw_df.columns)
control_cols = st.multiselect("Control columns", options=all_cols)
exo_cols = st.multiselect("Exogenous columns", options=[c for c in all_cols if c not in control_cols])
obj_cols = st.multiselect("Objective columns", options=[c for c in all_cols if c not in control_cols and c not in exo_cols])

if not control_cols or not obj_cols:
    st.warning("Select at least one control and one objective column.")
    st.stop()
unassigned = [c for c in all_cols if c not in control_cols and c not in exo_cols and c not in obj_cols]
if unassigned:
    st.warning(f"Assign all columns to a role. Unassigned: {unassigned}")
    st.stop()

groups = {
    "control": control_cols,
    "exogenous": exo_cols,
    "objective": obj_cols,
}

seq_len = st.slider("Warmup length", min_value=8, max_value=128, value=32, step=1)
pred_len = st.slider("Training horizon", min_value=1, max_value=64, value=12, step=1)

left, right = st.columns(2)
train_epochs = left.number_input("Train epochs (if no checkpoint)", min_value=1, max_value=50, value=3)
checkpoint_path = right.text_input("Checkpoint path (optional)", value="")

if st.button("Initialize Simulator", type="primary"):
    with st.spinner("Preparing model and simulator..."):
        train_loader, val_loader, scaler = build_grouped_dataloaders(
            raw_df,
            groups,
            input_groups=["control", "exogenous"],
            output_groups=["objective"],
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            batch_size=64,
            train_split=0.7,
            add_time=False,
            require_full_role_mapping=True,
        )
        dataset = train_loader.dataset

        input_dim = len(set(dataset.input_cols) | set(dataset.output_cols))
        output_dim = len(dataset.output_cols)
        model = build_model(
            "latent_ssm",
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            per_model_cfg={"type": "latent_ssm"},
            model_defaults_cfg={},
        )

        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt is not None and ckpt.exists():
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
        else:
            trainer = WorldModelTrainer(
                model=model,
                dataset=train_loader.dataset,
                val_dataset=val_loader.dataset,
                sampling_strategy=RandomStartFixedHorizon(horizon=int(pred_len)),
                warmup_len=int(seq_len),
                batch_size=32,
                training_mode="multi_step",
                feedback="model",
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                device="cpu",
                probabilistic_cfg={
                    "recon_weight": 1.0,
                    "kl_weight": 1.0,
                    "aux_weight": 1.0,
                    "kl_free_bits": 1.0,
                    "kl_balance": 0.8,
                    "use_kl_balancing": True,
                    "use_free_bits": True,
                    "lr_warmup_steps": 100,
                    "grad_clip_norm": 100.0,
                },
            )
            trainer.fit(epochs=int(train_epochs), steps_per_epoch=50, verbose=False)

        sim = RSSMSimulator(
            model=model,
            feature_columns=dataset.feature_cols,
            input_columns=dataset.input_cols,
            output_columns=dataset.output_cols,
            in_idx=dataset.in_idx,
            out_idx=dataset.out_idx,
            control_positions=dataset.control_positions,
            known_exo_positions=dataset.known_exo_positions,
            scaler=scaler,
            device="cpu",
        )

        hist_df = raw_df[dataset.feature_cols].tail(int(seq_len)).copy()
        sim.reset(hist_df)
        st.session_state["sim"] = sim
        st.session_state["horizon"] = int(pred_len)
        st.success("Simulator ready.")

if "sim" not in st.session_state:
    st.stop()

sim: RSSMSimulator = st.session_state["sim"]
horizon = int(st.session_state.get("horizon", pred_len))

st.subheader("Control scenario")
scenario_controls = np.zeros((horizon, len(control_cols)), dtype=np.float32)
for i, col in enumerate(control_cols):
    default_val = float(raw_df[col].iloc[-1])
    scenario_controls[:, i] = st.slider(f"{col}", float(raw_df[col].min()), float(raw_df[col].max()), default_val)

if st.button("Run rollout"):
    with st.spinner("Simulating..."):
        out = sim.rollout(scenario_controls, n_samples=50)
    mean = np.asarray(out["mean"])
    std = np.asarray(out["std"])

    for i, col in enumerate(obj_cols):
        chart_df = pd.DataFrame(
            {
                "step": np.arange(mean.shape[0]),
                "mean": mean[:, i],
                "lower": mean[:, i] - 1.96 * std[:, i],
                "upper": mean[:, i] + 1.96 * std[:, i],
            }
        ).set_index("step")
        st.line_chart(chart_df[["mean", "lower", "upper"]])

    if out.get("warnings"):
        st.warning("; ".join(out["warnings"]))
