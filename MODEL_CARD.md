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
- Example wastewater schema (matches `configs/dataset/wastewater.yaml`):
  - Control: `IN_METAL_Q`, `T1_O2`, `METAL_Q`
  - Exogenous: `IN_Q`, `MAX_CF`, `PROCESSPHASE_INLET`, `PROCESSPHASE_OUTLET`
  - Objective: `T1_PO4`

The wastewater CSV is **not** bundled with this repository. Download it from the data paper (arXiv:2407.05346) and place it at the path in the dataset config.

## 3. Training Recipe
- Core losses:
  - Objective reconstruction Gaussian NLL (in **normalized** space; RMSE/MAE/CRPS in evaluation reports are inverse-transformed to original units).
  - KL(post || prior) with optional free-bits applied to the **per-step aggregate** KL (Dreamer convention) + optional KL balancing.
  - Optional exogenous auxiliary Gaussian NLL.
  - Optional rollout loss (NLL + optional soft-DTW term).
- Latent / decoder noise:
  - Gaussian latent stds are learned unless `prior_constant_std` / `posterior_constant_std` are set (the wastewater RSSM config uses categorical latents).
  - Decoder `min_std` is honoured as configured (shipped configs use `0.5` on min-max normalized `[0, 1]` targets).
- Optimization:
  - AdamW (default `lr=3e-4`, `weight_decay=1e-6`).
  - LR warmup + cosine decay.
  - Gradient clipping (`global_norm=100`).
- Data handling:
  - Strict chronological split.
  - Train-only normalization stats; val/test reuse train stats.
  - `scripts/eval.py` and `scripts/compare.py` report metrics on the held-out **test** split. Validation is used for early stopping, checkpoint selection, and Optuna.

## 4. How To Run Simulator
- Python API:
  - `sim.reset(history_df)`
  - `sim.step(control_dict, exogenous_dict)`
  - `sim.rollout(control_df, exogenous_df, n_samples=50)`
- FastAPI endpoints:
  - `GET /health`, `GET /schema`
  - `POST /reset`, `POST /step`, `POST /rollout`
- Uncertainty: `step` / `rollout` intervals include both latent sampling and decoder Gaussian (aleatoric) noise, matching `LatentSSMWorldModel.rollout_mc` used by the evaluation suite.

## 5. Benchmarks
Headline numbers should be regenerated with `scripts/eval.py` / `scripts/eval_rssm_suite.py` on the test split after training. This card does not ship run artifacts.

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
