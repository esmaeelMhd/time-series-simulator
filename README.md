# Time-Series World Models for Control

A general-purpose Python library for training recurrent world models (simulators) on time-series data for model-based control, reinforcement learning, and model predictive control (MPC).

## Overview

This library provides a clean, domain-agnostic framework for:

1. **Training world models** that learn system dynamics from historical data
2. **Multi-step rollout training** to reduce compounding prediction errors
3. **Flexible sampling strategies** for diverse training scenarios
4. **Gymnasium-compatible environments** for RL integration
5. **Easy deployment** for MPC and control applications

Model behavior, schema expectations, training recipe, simulator usage, limitations, and benchmark references are documented in [`MODEL_CARD.md`](MODEL_CARD.md).

## Installation

```bash
git clone https://github.com/esmaeelMhd/time-series-simulator.git
cd time-series-simulator

# Default install (CUDA 12.9 PyTorch)
pip install -r requirements.txt
pip install -e .
```

CPU-only install:

```bash
pip install -r requirements.cpu.txt
pip install -e .
```

Explicit CUDA alias (equivalent to `requirements.txt`):

```bash
pip install -r requirements.cuda.txt
pip install -e .
```

Install all optional extras (tracking, Lightning, serving, validation, dev tools):

```bash
pip install -e ".[all]"
```

### Requirements

- Python >= 3.10
- PyTorch 2.13.0 (CUDA 12.9 by default via `requirements.txt`)
- Use `requirements.cpu.txt` for CPU-only installs

## Repository Layout

```
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Composition root
│   ├── dataset/                # Data schema & paths
│   ├── experiment/             # Per-run experiment profiles
│   ├── model/                  # Model architecture defaults
│   ├── training/               # Training & eval defaults
│   └── serving/                # Serving defaults
├── scripts/                    # CLI entry-point scripts
├── src/timesim/                # Core package (src layout)
│   ├── cli/                    # Packaged CLI commands (optimize, retrain)
│   ├── config/                 # Structured config dataclasses
│   ├── data/                   # Schema, datasets, preprocessing, datamodule
│   ├── envs/                   # Gymnasium environment adapter
│   ├── evaluation/             # Rollout, interventional eval, metrics
│   ├── models/                 # RSSM, LSTM, Transformer, DLinear, TFT, XGBoost
│   ├── optimization/           # Optuna hyperparameter search
│   ├── serving/                # FastAPI + Streamlit dashboard
│   ├── simulator/              # Stateful simulator wrapper
│   ├── training/               # Losses, trainer, retrainer, rollout, Lightning module
│   └── utils/                  # Helpers (config, logging, scaling, symlog, plotting)
├── tests/                      # Pytest suite
├── Dockerfile                  # Container image
├── docker-compose.yaml         # API + dashboard services
├── pyproject.toml              # Package metadata & tool config
├── requirements.base.txt       # Shared runtime dependencies
├── requirements.txt            # Default CUDA 12.9 dependency set
├── requirements.cpu.txt        # CPU-only dependency set
└── requirements.cuda.txt       # Explicit CUDA alias
```

## Technology Stack

- **Deep learning:** PyTorch 2.x
- **Training orchestration:** native trainer + optional PyTorch Lightning module
- **Configuration:** Hydra defaults composition (`configs/experiment/*` + group profiles)
- **Experiment tracking:** TensorBoard + optional W&B / MLflow (`tracking.backend`)
- **Hyperparameter optimization:** Optuna (`scripts/optimize.py`)
- **Serving:** FastAPI + Uvicorn (`scripts/serve.py`)
- **Dashboard:** Streamlit (`scripts/dashboard.py`)
- **Containerization:** `Dockerfile`, `docker-compose.yaml`
- **Testing:** pytest + property-based tests
- **Data validation:** optional Pandera (`data.validation.enabled`)

## Full Workflow (Small Run)

The commands below walk through a complete optimize → train → evaluate → simulate → compare → serve cycle using the small config, which slices the dataset to 1 000 rows and trains for only a few epochs.

### 1. Hyperparameter Optimization (optional)

```bash
python scripts/optimize.py \
    --config configs/wastewater.small.yaml \
    --models latent_ssm \
    --n-trials 10 \
    --fast-mode
```

Runs Optuna Bayesian search. Results are saved to `runs/wastewater/small/optuna/latent_ssm/` (best params YAML, trials CSV, importance plots).

### 2. Train

```bash
# Hydra entrypoint (recommended)
python scripts/train_hydra.py --config-name=wastewater.small

# Legacy entrypoint (equivalent)
python scripts/train.py --config configs/wastewater.small.yaml
```

To apply Optuna best params from step 1:

```bash
python scripts/train.py --config configs/wastewater.small.yaml --use-optuna-best-params
```

Outputs per model go to `runs/wastewater/small/<model>/` — checkpoints (`.pth`), scaler (`.pkl`), loss curves, forecast & simulation plots.

### 3. Evaluate

```bash
python scripts/eval.py \
    --config configs/wastewater.small.yaml \
    --model latent_ssm
```

Runs multi-window rollout evaluation and recursive simulation on the validation set. Override horizon and window count:

```bash
python scripts/eval.py \
    --config configs/wastewater.small.yaml \
    --model latent_ssm \
    --eval-horizon 12 \
    --n-windows 4 \
    --sim-horizon 100
```

### 4. Simulate

```bash
python scripts/simulate.py \
    --config configs/wastewater.small.yaml \
    --model latent_ssm \
    --horizon 100 \
    --start-idx 0
```

Produces autoregressive rollout plots and CSV from a chosen starting point.

### 5. Compare Multiple Models

```bash
python scripts/compare.py \
    --config configs/wastewater.small.yaml \
    --models latent_ssm lstm transformer
```

Generates side-by-side comparison plots and summary CSV in `runs/wastewater/small/figures/`.

### 6. Serve

**FastAPI:**

```bash
python scripts/serve.py \
    --config configs/wastewater.small.yaml \
    --checkpoint runs/wastewater/small/latent_ssm/train_checkpoint.pth \
    --port 8000
```

Endpoints: `GET /health`, `GET /schema`, `POST /reset`, `POST /step`, `POST /rollout`.

**Streamlit dashboard:**

```bash
streamlit run scripts/dashboard.py
```

### Hydra Overrides

Any config value can be overridden from the command line:

```bash
python scripts/train_hydra.py --config-name=wastewater.small \
    training_rounds.0.epochs=10 \
    misc.device=cuda \
    training.steps_per_epoch=50
```

## Docker

```bash
# Default API image (CPU PyTorch)
docker compose up api

# GPU API (requires nvidia-docker)
docker compose --profile gpu up api-gpu

# Streamlit dashboard
docker compose up dashboard
```

Override config and checkpoint via environment variables:

```bash
docker build -t timesim-api .
docker run --rm -p 8000:8000 \
    -e TIMESIM_CONFIG=configs/wastewater.yaml \
    -e TIMESIM_CHECKPOINT=runs/wastewater/full_with_time/latent_ssm/train_checkpoint.pth \
    -v "$PWD:/app" \
    timesim-api
```

## Testing

```bash
# Full suite
pytest tests/ -v

# Fast CI smoke test (synthetic data, no config needed)
python scripts/ci_fast_train.py
```

Test coverage spans data pipeline, encoders, RSSM cell, training loop, evaluation, API, simulator, and end-to-end integration.

## CI Pipeline

`.github/workflows/ci.yml` runs on every push/PR:

1. Lint (`ruff check src scripts tests`)
2. Unit tests (full `pytest tests/` suite)
3. Fast training smoke test (`scripts/ci_fast_train.py`, 5 epochs on synthetic data)

The wastewater CSV used by `configs/dataset/wastewater.yaml` is not in this repository; see that config for provenance and the expected local path.

## Architecture

### Core Abstractions

**`WorldModelBase`** — abstract interface for all world models:

```python
class WorldModelBase(nn.Module):
    def init_state(self, warmup_seq):
        """Initialize hidden state from warmup sequence."""

    def step(self, state, control_t, exo_t, prev_output_t):
        """Single-step prediction."""

    def rollout(self, warmup_seq, rollout_inputs, horizon, feedback="model"):
        """Multi-step autoregressive rollout."""
```

**`SamplingStrategy`** — defines how to sample (start_index, horizon) pairs for training:

- `RandomStartRandomHorizon` — maximum diversity
- `RandomStartFixedHorizon` — fixed horizon, random starts
- `DailyFixedHorizon` — domain-specific (start at midnight, 24 h horizon)
- `GeometricHorizonSampling` — curriculum learning (1, 2, 4, 8, …)
- `StrideBasedSampling` — regular stride intervals

**`WorldModelTrainer`** — training loop with multi-step rollout support, teacher forcing, scheduled sampling, and pure autoregressive modes.

### Training Modes

```python
# Multi-step (recommended) — trains with autoregressive rollouts
trainer = WorldModelTrainer(..., training_mode="multi_step", feedback="model")

# Combined — mixes one-step teacher forcing + multi-step rollouts
trainer = WorldModelTrainer(..., training_mode="combined", one_step_weight=0.3, multi_step_weight=0.7)

# One-step — teacher forcing only
trainer = WorldModelTrainer(..., training_mode="one_step")
```

## Quick Start (Python API)

```python
import pandas as pd
from timesim.data import load_csv_dataset, GroupedTimeSeriesDataset
from timesim.models import LSTMWorldModel
from timesim.training import WorldModelTrainer
from timesim.data.sampling import RandomStartFixedHorizon

# 1. Load data
df = load_csv_dataset("data/my_system.csv", index_col="timestamp")
groups = {
    "control": ["valve_1", "pump_speed"],
    "exogenous": ["temperature", "flow_in"],
    "objective": ["quality_metric"],
}
dataset = GroupedTimeSeriesDataset(
    df, groups,
    input_groups=["control", "exogenous"],
    output_groups=["objective"],
    seq_len=24, pred_len=12,
)

# 2. Train
model = LSTMWorldModel(input_dim=4, output_dim=1, hidden_dim=64, num_layers=2)
trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    sampling_strategy=RandomStartFixedHorizon(horizon=24),
    warmup_len=24, batch_size=32,
    training_mode="multi_step",
    device="cuda",
)
trainer.fit(epochs=50)
trainer.save("checkpoint.pth")

# 3. Simulate
predictions = model.rollout(
    warmup_seq={"inputs": warmup_data},
    rollout_inputs={"controls": controls, "exogenous": exogenous},
    horizon=100, feedback="model",
)

# 4. RL integration
from timesim.envs import WorldModelEnv

env = WorldModelEnv(
    world_model=model, dataset=dataset,
    warmup_len=24, episode_len=100,
    control_dim=2, exo_dim=2, output_dim=1,
    target_output=target,
)
obs, info = env.reset()
for _ in range(100):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

## Design Principles

1. **Domain-agnostic** — no domain-specific assumptions in the core library
2. **Composable** — mix and match models, sampling strategies, and loss functions
3. **Extensible** — easy to add new models, strategies, and features
4. **Operationally careful** — error handling, logging, and checkpointing throughout
5. **Research-friendly** — clean abstractions for experimentation

## License

MIT License — see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
