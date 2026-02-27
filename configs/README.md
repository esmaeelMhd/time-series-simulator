# Config Layout (Hydra-Only)

Canonical config groups:

- `configs/dataset/`
- `configs/model/`
- `configs/training/`
- `configs/serving/`
- `configs/experiment/`

`configs/config.yaml` is the single training composition root and includes:

- `train_schema` (structured dataclass schema)
- one `experiment/*` profile

## Source of Truth

- Dataset/schema/splits live under `dataset/*`.
- Model architecture defaults live under `model/*`.
- Training/eval/simulation defaults live under `training/*`.
- Serving defaults live under `serving/*`.
- Per-run selections/overrides are only in `experiment/*`.

Top-level files like `configs/wastewater*.yaml` are thin Hydra aliases that select
an `experiment/*` profile. They do not duplicate settings.

## Hydra Overrides

All CLI scripts accept Hydra `key=value` overrides **after** the standard argparse
flags. Unrecognised arguments are forwarded to `hydra.compose()` as overrides.

### Examples

```bash
# Override the device and evaluation horizon from the command line:
python scripts/eval.py --config configs/wastewater.small.yaml --model latent_ssm \
    misc.device=cpu evaluation.horizon=24

# Override training epochs and seed:
python scripts/train.py --config wastewater.small \
    training.epochs=5 misc.seed=123

# Override dataset batch size during optimisation:
python scripts/optimize.py --config wastewater.small --n-trials 5 \
    dataset.batch_size=64

# Use a Hydra config name instead of a file path (equivalent):
python scripts/simulate.py --config wastewater.small --model latent_ssm \
    simulation.horizon=200
```

Override keys use the dot-separated path into the YAML tree,
e.g. `training.epochs=10` sets `training: { epochs: 10 }`.

### How it works

1. Each script parses its own `--flags` with `argparse.parse_known_args()`.
2. Any leftover `key=value` tokens are passed as Hydra overrides to
   `compose_config()`.
3. `compose_config()` calls `hydra.compose()` with those overrides, resolves
   all interpolations, and returns a plain Python dict.

This means **both** styles work side by side:

```bash
python scripts/eval.py --config wastewater.small --model latent_ssm --eval-horizon 48
python scripts/eval.py --config wastewater.small --model latent_ssm evaluation.horizon=48
```

## Notes

- `_base` inheritance is deprecated and no longer supported by the loader.
- The `--config-name` flag is accepted as an alias for `--config` in all scripts.
