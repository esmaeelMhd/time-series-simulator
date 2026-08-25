import random

import numpy as np
import pytest
import torch

from timesim.data.schema import VariableSchema
from timesim.utils.misc import resolve_device, seed_everything


def test_variable_schema_rejects_duplicate_assignments():
    with pytest.raises(ValueError, match="assigned to both"):
        VariableSchema.from_groups(
            {
                "control": ["a", "b"],
                "exogenous": ["b"],
                "objective": ["c"],
            }
        )


def test_seed_everything_reproducible_streams():
    seed_everything(123, deterministic=True)
    a1 = random.random()
    b1 = np.random.rand(3)
    c1 = torch.rand(3)

    seed_everything(123, deterministic=True)
    a2 = random.random()
    b2 = np.random.rand(3)
    c2 = torch.rand(3)

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert torch.allclose(c1, c2)


def test_resolve_device_auto_returns_valid_token():
    d = resolve_device("auto")
    assert d in {"cpu", "cuda"}
