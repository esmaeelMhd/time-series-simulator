# TimeSim RSSM Model Card

## 1. Model Summary
- Model family: Recurrent State-Space Model (RSSM) with deterministic + stochastic state.
- Primary use: Counterfactual simulation and probabilistic forecasting for control/exogenous interventions in time series.
- Not a classifier or causal proof engine; outputs are data-driven predictions with uncertainty estimates.

## 2. Input / Output Schema
- Variable roles are explicit and mutually exclusive:
  - `CONTROL`: actionable variables (interventions).
  - `EXOGENOUS`: non-controlled context/disturbances.
  - `OBJECTIVE`: predicted targets.
- Example wastewater schema:
  - Control: `IN_METAL_Q`, `T1_O2`
  - Exogenous: `METAL_Q`, `IN_Q`, `MAX_CF`, `PROCESSPHASE_INLET`, `PROCESSPHASE_OUTLET`
  - Objective: `T1_PO4`

## 3. Training Recipe
- Core losses:
  - Objective reconstruction Gaussian NLL.
  - KL(post || prior) with optional free-bits + KL balancing.
  - Optional exogenous auxiliary Gaussian NLL.
  - Optional rollout loss (NLL + optional soft-DTW term).
- Optimization:
  - AdamW (default `lr=3e-4`, `weight_decay=1e-6`).
  - LR warmup + cosine decay.
  - Gradient clipping (`global_norm=100`).
- Data handling:
  - Strict chronological split.
  - Train-only normalization stats; val/test reuse train stats.

## 4. How To Run Simulator
- Python API:
  - `sim.reset(history_df)`
  - `sim.step(control_dict, exogenous_dict)`
  - `sim.rollout(control_df, exogenous_df, n_samples=50)`
- FastAPI endpoints:
  - `GET /health`, `GET /schema`
  - `POST /reset`, `POST /step`, `POST /rollout`

## 5. Benchmarks (Example Run)
- Example artifact: `runs/wastewater/checklist_loss_eval_rollout/latent_ssm/eval_suite/summary.yaml`
- Snapshot metrics:
  - Open-loop RMSE@1: `0.7296`
  - Open-loop RMSE@20: `1.0371`
  - RMSE ratio h20/h1: `1.4214`
  - Coverage@95 mean: `0.8950`
  - Latent KL mean: `2.7505`

## 6. Known Limitations
- Interventional validity depends on correct role mapping and data coverage.
- Out-of-distribution controls/exogenous values can degrade calibration.
- Probabilistic intervals can still be over/under-confident without post-hoc calibration.
- Model does not replace domain constraints, safety interlocks, or first-principles validation.

## 7. Safety / Guardrails
- Guardrails prevent unsafe defaults:
  - Objective leakage into transition blocked unless explicit ablation flag enabled.
  - Disabling aux decoder blocked unless explicit ablation flag enabled.
  - Shared encoders and no-stochastic-path blocked unless explicit ablation flags enabled.
- Serve-time extrapolation warnings trigger when inputs are >2 sigma from train distribution.

