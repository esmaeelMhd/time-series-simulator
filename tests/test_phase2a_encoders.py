import pytest
import torch

from timesim.models.encoders import (
    ControlEncoder,
    ExogenousEncoder,
    ObservationEncoder,
    assert_no_shared_encoder_params,
)
from timesim.models.latent_ssm import LatentSSMWorldModel


def test_role_encoders_output_shapes():
    batch = 5
    ctrl = ControlEncoder(input_dim=3, hidden_dim=16, embed_dim=7)
    exog = ExogenousEncoder(input_dim=4, hidden_dim=16, embed_dim=9)
    obs = ObservationEncoder(input_dim=2, hidden_dim=16, embed_dim=11)

    assert ctrl(torch.randn(batch, 3)).shape == (batch, 7)
    assert exog(torch.randn(batch, 4)).shape == (batch, 9)
    assert obs(torch.randn(batch, 2)).shape == (batch, 11)


def test_role_encoders_have_disjoint_parameter_sets():
    ctrl = ControlEncoder(input_dim=3, hidden_dim=16, embed_dim=7)
    exog = ExogenousEncoder(input_dim=4, hidden_dim=16, embed_dim=7)
    obs = ObservationEncoder(input_dim=2, hidden_dim=16, embed_dim=7)

    ctrl_params = set(id(p) for p in ctrl.parameters())
    exog_params = set(id(p) for p in exog.parameters())
    obs_params = set(id(p) for p in obs.parameters())

    assert ctrl_params & exog_params == set()
    assert ctrl_params & obs_params == set()
    assert exog_params & obs_params == set()

    assert_no_shared_encoder_params(ctrl, exog, obs)


def test_shared_encoder_detection_raises():
    ctrl = ControlEncoder(input_dim=3, hidden_dim=16, embed_dim=7)
    exog = ExogenousEncoder(input_dim=4, hidden_dim=16, embed_dim=7)
    # Intentional parameter sharing for guardrail check.
    obs = ctrl
    with pytest.raises(RuntimeError, match="separate modules"):
        assert_no_shared_encoder_params(ctrl, exog, obs)


def test_latent_ssm_uses_explicit_role_encoders():
    model = LatentSSMWorldModel(
        input_dim=6,
        output_dim=2,
        hidden_dim=32,
        latent_dim=8,
        control_dim=3,
        exogenous_dim=1,
        encoder_dim=12,
    )
    assert isinstance(model.control_encoder, ControlEncoder)
    assert isinstance(model.exogenous_encoder, ExogenousEncoder)
    assert isinstance(model.observation_encoder, ObservationEncoder)
