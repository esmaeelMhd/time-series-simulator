# Quick Reference Guide

## Installation

```bash
pip install -e .
```

## Basic Workflow

### 1. Prepare Data

```python
from timesim.data import load_csv_dataset, GroupedTimeSeriesDataset

df = load_csv_dataset("data.csv", index_col="timestamp")

groups = {
    "control": ["valve", "pump"],
    "exogenous": ["temp", "flow"],
    "objective": ["quality"],
}

dataset = GroupedTimeSeriesDataset(
    df, groups,
    input_groups=["control", "exogenous"],
    output_groups=["objective"],
    seq_len=24, pred_len=12,
)
```

### 2. Create Model

```python
from timesim.models import LSTMWorldModel

model = LSTMWorldModel(
    input_dim=4,    # controls + exogenous
    output_dim=1,   # objectives
    hidden_dim=64,
    num_layers=2,
)
```

### 3. Choose Sampling Strategy

```python
from timesim.data.sampling import RandomStartFixedHorizon

strategy = RandomStartFixedHorizon(horizon=24)
```

### 4. Train

```python
from timesim.training import WorldModelTrainer

trainer = WorldModelTrainer(
    model=model,
    dataset=train_dataset,
    val_dataset=val_dataset,
    sampling_strategy=strategy,
    warmup_len=24,
    batch_size=32,
    training_mode="multi_step",
    device="cuda",
)

trainer.fit(epochs=50)
trainer.save("checkpoint.pth")
```

### 5. Evaluate

```python
# Load model
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

# Rollout
result = model.rollout(
    warmup_seq={"inputs": warmup},
    rollout_inputs={"controls": controls, "exogenous": exo},
    horizon=100,
    feedback="model",
)

predictions = result["predictions"]
```

## Sampling Strategies

### Random Start, Random Horizon
```python
from timesim.data.sampling import RandomStartRandomHorizon
strategy = RandomStartRandomHorizon(h_min=12, h_max=48)
```

### Random Start, Fixed Horizon
```python
from timesim.data.sampling import RandomStartFixedHorizon
strategy = RandomStartFixedHorizon(horizon=24)
```

### Daily Fixed (Start-of-Day + 24h)
```python
from timesim.data.sampling import DailyFixedHorizon
strategy = DailyFixedHorizon(
    start_hour=0,
    horizon=24,
    samples_per_hour=1,  # or 30 for 2-min data
)
```

### Geometric (Curriculum Learning)
```python
from timesim.data.sampling import GeometricHorizonSampling
strategy = GeometricHorizonSampling(pred_len=1, h_max=64)
# Samples from: 1, 2, 4, 8, 16, 32, 64
```

## Training Modes

### Multi-Step (Recommended)
```python
trainer = WorldModelTrainer(
    ...,
    training_mode="multi_step",
    feedback="model",  # Pure autoregressive
)
```

### Combined (One-Step + Multi-Step)
```python
trainer = WorldModelTrainer(
    ...,
    training_mode="combined",
    one_step_weight=0.3,
    multi_step_weight=0.7,
)
```

### One-Step (Teacher Forcing)
```python
trainer = WorldModelTrainer(
    ...,
    training_mode="one_step",
)
```

## Feedback Modes

### Model Feedback (Autoregressive)
```python
result = model.rollout(..., feedback="model")
```

### Teacher Forcing
```python
result = model.rollout(..., feedback="teacher", targets=targets)
```

### Mixed (Scheduled Sampling)
```python
result = model.rollout(
    ...,
    feedback="mixed",
    teacher_forcing_ratio=0.5,
    targets=targets,
)
```

## Loss Functions

### Multi-Step Loss
```python
from timesim.training.losses import MultiStepLoss

loss_fn = MultiStepLoss(
    loss_type="mse",           # mse, mae, huber
    weighting="linear",        # uniform, linear, exponential
)
```

### Combined Loss
```python
from timesim.training.losses import CombinedLoss

loss_fn = CombinedLoss(
    one_step_weight=0.3,
    multi_step_weight=0.7,
    loss_type="mse",
)
```

## Gymnasium Environment

```python
from timesim.envs import WorldModelEnv

env = WorldModelEnv(
    world_model=model,
    dataset=dataset,
    warmup_len=24,
    episode_len=100,
    control_dim=2,
    exo_dim=2,
    output_dim=1,
    target_output=np.array([0.5]),
)

obs, info = env.reset()
for _ in range(100):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

## Configuration File Example

```yaml
dataset:
  csv: data/system.csv
  variables:
    control: [valve, pump]
    exogenous: [temp, flow]
    objective: [quality]

model:
  type: lstm
  hidden_dim: 64
  num_layers: 2

training:
  mode: multi_step
  sampling:
    strategy: random_fixed
    horizon: 24
  epochs: 50
  warmup_len: 24
```

## Common Patterns

### Load and Split Data
```python
from timesim.data.loader import build_grouped_dataloaders

train_loader, val_loader, scaler = build_grouped_dataloaders(
    df, groups, input_groups, output_groups,
    seq_len=24, pred_len=12, batch_size=32,
    train_split=0.8,
)

train_dataset = train_loader.dataset
val_dataset = val_loader.dataset
```

### Save and Load
```python
# Save
trainer.save("checkpoint.pth")
from joblib import dump
dump(scaler, "scaler.pkl")

# Load
model.load_state_dict(torch.load("checkpoint.pth"))
from joblib import load
scaler = load("scaler.pkl")
```

### Custom Sampling Strategy
```python
class MySampling:
    def sample(self, dataset_length, batch_size, warmup_len, rng=None):
        # Your logic
        start_indices = ...
        horizons = ...
        return start_indices, horizons

trainer = WorldModelTrainer(..., sampling_strategy=MySampling())
```

### Custom World Model
```python
from timesim.models.base import WorldModelBase

class MyModel(WorldModelBase):
    def init_state(self, warmup_seq):
        # Initialize state
        return state
    
    def step(self, state, control_t, exo_t, prev_output_t):
        # Single step
        return new_state, prediction
```

## Debugging Tips

1. **NaN losses**: Check data for NaNs, reduce learning rate
2. **Memory issues**: Reduce batch_size or horizon
3. **Slow training**: Use GPU, reduce steps_per_epoch
4. **Poor long-horizon**: Use multi-step training, increase warmup_len
5. **Overfitting**: Add dropout, use validation early stopping

## Performance Tips

1. Use GPU: `device="cuda"`
2. Larger batch sizes for efficiency
3. `RandomStartFixedHorizon` is faster than variable horizons
4. Adjust `steps_per_epoch` based on dataset size
5. Use mixed precision training for large models

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=timesim --cov-report=html
```

## CLI Tools

```bash
# Train
timesim-train --config config.yaml --epochs 50

# Retrain
timesim-retrain --checkpoint runs/.../checkpoint.pth --epochs 10
```

