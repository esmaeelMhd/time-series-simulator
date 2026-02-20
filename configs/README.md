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

## Notes

- `_base` inheritance is deprecated and no longer supported by the loader.
