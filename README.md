# Time-Series World Models for Control

A general-purpose Python library for training recurrent world models (simulators) on time-series data for model-based control, reinforcement learning, and model predictive control (MPC).

## Overview

This library provides a clean, domain-agnostic framework for:

1. **Training world models** that learn system dynamics from historical data
2. **Multi-step rollout training** to reduce compounding prediction errors
3. **Flexible sampling strategies** for diverse training scenarios
4. **Gymnasium-compatible environments** for RL integration
5. **Easy deployment** for MPC and control applications

## Model Card

Model behavior, schema expectations, training recipe, simulator usage, limitations, and benchmark references are documented in:

- `MODEL_CARD.md`

## Production Hardening Quickstart

### Docker API (CPU)

```bash
docker build -t timesim-api .
docker run --rm -p 8000:8000 \
  -e TIMESIM_CONFIG=configs/wastewater.yaml \
  -e TIMESIM_CHECKPOINT=runs/wastewater/full_with_time/latent_ssm/train_checkpoint.pth \
  -v "$PWD:/app" \
  timesim-api
```

If config/checkpoint are missing, the container still starts a minimal health API (`/health`, `/ready`).

### Docker API (GPU / nvidia-docker)

```bash
docker build -t timesim-api-gpu \
  --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 .

docker run --rm --gpus all -p 8000:8000 \
  -e TIMESIM_DEVICE=cuda \
  -e TIMESIM_CONFIG=configs/wastewater.yaml \
  -e TIMESIM_CHECKPOINT=runs/wastewater/full_with_time/latent_ssm/train_checkpoint.pth \
  -v "$PWD:/app" \
  timesim-api-gpu
```

### CI Pipeline

The repository includes GitHub Actions CI at:

- `.github/workflows/ci.yml`

On every push/PR it runs:

1. Lint (`ruff` fatal checks)
2. Unit tests for RSSM/encoders/decoders/losses/normalization/simulator
3. A fast synthetic training smoke run (`scripts/ci_fast_train.py`, 5 epochs)

### Key Features

- ✅ **Domain-agnostic**: Works across industries (wastewater, energy, manufacturing, etc.)
- ✅ **Multi-step training**: "See every possible path" approach reduces long-horizon errors
- ✅ **Flexible architectures**: LSTM, Transformer, or custom models
- ✅ **Sampling strategies**: Random, daily patterns, geometric horizons, etc.
- ✅ **RL-ready**: Gymnasium environment adapter included
- ✅ **Production-ready**: Proper scaling, checkpointing, logging, and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-simulator.git
cd time-series-simulator

# Install in editable mode
pip install -e .
```

### Optional Full-Stack Extras

```bash
# Tracking + Lightning + validation + property tests
pip install -e ".[all]"
```

## Technology Stack

The repository now supports this stack end-to-end (optionally enabled):

- Deep learning: PyTorch 2.x
- Training orchestration: native trainer + optional PyTorch Lightning (`scripts/train_lightning.py`)
- Configuration: YAML `_base` chain + optional Hydra wrapper (`scripts/train_hydra.py`)
- Experiment tracking: TensorBoard + optional W&B / MLflow (`tracking.backend`)
- Hyperparameter optimization: Optuna (`timesim.cli.optimize`, `scripts/optimize.py`)
- Serving: FastAPI + Uvicorn (`scripts/serve.py`)
- Dashboard: Streamlit (`scripts/dashboard.py`)
- Containerization: `Dockerfile`, `docker-compose.yml`
- Testing: `pytest` + property-based tests (`tests/test_property_invariants.py`)
- Data validation: optional Pandera (`data.validation.enabled`)
- Dataframe/array stack: NumPy, pandas, optional polars (`data.csv_engine: polars`)

Examples:

```bash
# Hydra wrapper (composes base config + overrides)
python scripts/train_hydra.py base_config=configs/wastewater.yaml overrides.training.epochs=5

# Lightning entrypoint (experimental)
python scripts/train_lightning.py --config configs/wastewater.yaml --model latent_ssm
```

## Repository Layout

Source code now follows `src/` layout:

- `src/timesim/data`: schema, datasets, preprocessing, datamodule
- `src/timesim/models`: RSSM/world-model components and model factory
- `src/timesim/training`: losses, trainer, Lightning wrapper
- `src/timesim/evaluation`: rollout/interventional evaluation and metrics
- `src/timesim/serving`: FastAPI + simulator wrappers + dashboard launcher
- `src/timesim/optimization`: Optuna wrapper
- `src/timesim/utils`: common helpers including `symlog`

Config groups are organized under `configs/`:

- `configs/config.yaml`
- `configs/model/*.yaml`
- `configs/training/*.yaml`
- `configs/data/*.yaml`
- `configs/serving/*.yaml`

### Requirements

- Python >= 3.10
- PyTorch >= 1.13
- NumPy, Pandas
- Gymnasium (for RL integration)
- See `requirements.txt` for full list

## Quick Start

### 1. Prepare Your Data

Organize your time-series data into semantic groups:

```python
import pandas as pd
from timesim.data import load_csv_dataset, GroupedTimeSeriesDataset

# Load your data
df = load_csv_dataset("data/my_system.csv", index_col="timestamp")

# Define semantic groups
groups = {
    "control": ["valve_1", "pump_speed"],      # Control inputs (actions)
    "exogenous": ["temperature", "flow_in"],   # Disturbances/context
    "objective": ["quality_metric"],           # Outputs to predict
}

# Create dataset
dataset = GroupedTimeSeriesDataset(
    df, groups,
    input_groups=["control", "exogenous"],
    output_groups=["objective"],
    seq_len=24,
    pred_len=12,
)
```

### 2. Train a World Model

```python
from timesim.models import LSTMWorldModel
from timesim.training import WorldModelTrainer
from timesim.data.sampling import RandomStartFixedHorizon

# Create model
model = LSTMWorldModel(
    input_dim=4,  # 2 controls + 2 exogenous
    output_dim=1,
    hidden_dim=64,
    num_layers=2,
)

# Create sampling strategy
sampling_strategy = RandomStartFixedHorizon(horizon=24)

# Train with multi-step rollouts
trainer = WorldModelTrainer(
    model=model,
    dataset=train_dataset,
    val_dataset=val_dataset,
    sampling_strategy=sampling_strategy,
    warmup_len=24,
    batch_size=32,
    training_mode="multi_step",  # Key: multi-step training!
    device="cuda",
)

train_losses, val_losses = trainer.fit(epochs=50)
trainer.save("checkpoint.pth")
```

### 3. Use for Simulation or RL

```python
# Option A: Direct simulation
from timesim.training.rollout import batch_rollout

predictions = model.rollout(
    warmup_seq={"inputs": warmup_data},
    rollout_inputs={"controls": controls, "exogenous": exogenous},
    horizon=100,
    feedback="model",
)

# Option B: Wrap as Gymnasium environment for RL
from timesim.envs import WorldModelEnv

env = WorldModelEnv(
    world_model=model,
    dataset=dataset,
    warmup_len=24,
    episode_len=100,
    control_dim=2,
    exo_dim=2,
    output_dim=1,
    target_output=target,
)

obs, info = env.reset()
for _ in range(100):
    action = policy(obs)  # Your RL policy
    obs, reward, terminated, truncated, info = env.step(action)
```

## Architecture

### Core Abstractions

#### 1. `WorldModelBase`

Abstract interface for world models:

```python
class WorldModelBase(nn.Module):
    def init_state(self, warmup_seq):
        """Initialize hidden state from warmup sequence."""
        
    def step(self, state, control_t, exo_t, prev_output_t):
        """Single-step prediction."""
        
    def rollout(self, warmup_seq, rollout_inputs, horizon, feedback="model"):
        """Multi-step autoregressive rollout."""
```

#### 2. `SamplingStrategy`

Defines how to sample (start_index, horizon) pairs for training:

- **`RandomStartRandomHorizon`**: Maximum diversity
- **`RandomStartFixedHorizon`**: Fixed horizon, random starts
- **`DailyFixedHorizon`**: Domain-specific (e.g., start at midnight, 24h horizon)
- **`GeometricHorizonSampling`**: Curriculum learning (1, 2, 4, 8, ...)
- **`StrideBasedSampling`**: Regular stride intervals

#### 3. `WorldModelTrainer`

Unified trainer with multi-step rollout training:

- Samples diverse starting points and horizons
- Performs batched multi-environment rollouts
- Computes multi-step losses to reduce compounding errors
- Supports teacher forcing, scheduled sampling, and pure autoregressive modes

### Package Structure

```
timesim/
├── data/
│   ├── dataset.py          # TimeSeriesDataset, GroupedTimeSeriesDataset
│   ├── sampling.py         # SamplingStrategy classes
│   └── loader.py           # Data loading utilities
├── models/
│   ├── base.py             # WorldModelBase abstract class
│   ├── lstm.py             # LSTMWorldModel
│   └── transformer.py      # Transformer-based models
├── training/
│   ├── losses.py           # OneStepLoss, MultiStepLoss, CombinedLoss
│   ├── rollout.py          # Multi-environment rollout utilities
│   └── trainer.py          # WorldModelTrainer
├── envs/
│   └── gym_adapter.py      # Gymnasium environment wrapper
└── utils/
    ├── metrics.py          # Evaluation metrics
    ├── plotting.py         # Visualization utilities
    └── scaler.py           # Data scaling
```

## Training Modes

### Multi-Step Training (Recommended)

Trains the model using autoregressive rollouts, which reduces compounding errors:

```python
trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    training_mode="multi_step",
    feedback="model",  # Pure autoregressive
)
```

### Combined Training

Mixes one-step teacher forcing with multi-step rollouts:

```python
trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    training_mode="combined",
    one_step_weight=0.3,
    multi_step_weight=0.7,
)
```

### Teacher Forcing

Traditional one-step ahead prediction:

```python
trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    training_mode="one_step",
)
```

## Examples

### Wastewater Treatment Plant

See `examples/wastewater/` for a complete example:

```bash
# Train
python examples/wastewater/train_world_model.py --config examples/wastewater/config.yaml

# Evaluate
python examples/wastewater/evaluate_world_model.py --run_dir runs/wastewater/lstm/...
```

### Custom Domain

1. Prepare your CSV data with timestamp index
2. Define control/exogenous/objective groups in config
3. Choose a sampling strategy appropriate for your domain
4. Train and evaluate!

## Advanced Usage

### Custom Sampling Strategy

```python
class CustomSampling:
    def sample(self, dataset_length, batch_size, warmup_len, rng=None):
        # Your custom logic here
        start_indices = ...
        horizons = ...
        return start_indices, horizons

trainer = WorldModelTrainer(
    model=model,
    dataset=dataset,
    sampling_strategy=CustomSampling(),
)
```

### Custom World Model

```python
from timesim.models.base import WorldModelBase

class MyWorldModel(WorldModelBase):
    def init_state(self, warmup_seq):
        # Initialize your model's state
        ...
    
    def step(self, state, control_t, exo_t, prev_output_t):
        # Single-step prediction
        ...
        return new_state, prediction
```

### Custom Loss Function

```python
from timesim.training.losses import MultiStepLoss

loss_fn = MultiStepLoss(
    loss_type="huber",
    weighting="linear",  # Emphasize later steps
)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Dataset handling and windowing
- Sampling strategies
- Model interfaces and rollouts
- Training loops
- End-to-end workflows

## Design Principles

1. **Domain-agnostic**: No wastewater-specific assumptions in core library
2. **Composable**: Mix and match models, sampling strategies, and loss functions
3. **Extensible**: Easy to add new models, strategies, and features
4. **Production-ready**: Proper error handling, logging, and checkpointing
5. **Research-friendly**: Clean abstractions for experimenting with new ideas

## Citation

If you use this library in your research, please cite:

```bibtex
@software{timeseries_world_models,
  title={Time-Series World Models for Control},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/time-series-simulator}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This library was developed as part of PhD research on model-based control for wastewater treatment plants, but has been refactored to be domain-agnostic and reusable across industries.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
