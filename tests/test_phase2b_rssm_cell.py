import inspect

import torch

from timesim.models.rssm import RSSMCell, RSSMState


def _make_cell(**kwargs) -> RSSMCell:
    return RSSMCell(
        dim_h=32,
        dim_z=8,
        dim_control_embed=6,
        dim_exogenous_embed=5,
        dim_obs_embed=7,
        **kwargs,
    )


def test_transition_input_uses_z_control_exogenous_only_by_default():
    cell = _make_cell(leak_objective_to_transition=False)
    assert cell.transition_input_dim == 8 + 6 + 5
    assert cell.transition_mlp[0].in_features == 8 + 6 + 5
    assert cell.transition_gru.input_size == 32


def test_deterministic_path_shape_from_gru():
    cell = _make_cell()
    state0 = cell.initial_state(batch_size=4, device=torch.device("cpu"), dtype=torch.float32)
    out = cell.imagine(
        prev_state=state0,
        control_embed=torch.randn(4, 6),
        exogenous_embed=torch.randn(4, 5),
        sample=False,
    )
    assert out.h.shape == (4, 32)
    assert out.state.h.shape == (4, 32)


def test_prior_and_posterior_distribution_shapes():
    cell = _make_cell()
    state0 = cell.initial_state(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
    imag = cell.imagine(
        prev_state=state0,
        control_embed=torch.randn(3, 6),
        exogenous_embed=torch.randn(3, 5),
        sample=False,
    )
    assert imag.prior_mean.shape == (3, 8)
    assert imag.prior_std.shape == (3, 8)
    assert imag.prior.rsample().shape == (3, 8)

    obs = cell.observe(
        prev_state=state0,
        control_embed=torch.randn(3, 6),
        exogenous_embed=torch.randn(3, 5),
        observation_embed=torch.randn(3, 7),
        sample=False,
    )
    assert obs.posterior is not None
    assert obs.posterior_mean is not None
    assert obs.posterior_std is not None
    assert obs.posterior_mean.shape == (3, 8)
    assert obs.posterior_std.shape == (3, 8)
    assert obs.posterior.rsample().shape == (3, 8)


def _observe(cell):
    state0 = cell.initial_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    return cell.observe(
        prev_state=state0,
        control_embed=torch.randn(2, 6),
        exogenous_embed=torch.randn(2, 5),
        observation_embed=torch.randn(2, 7),
        sample=False,
    )


def test_configured_min_std_is_respected_exactly():
    """min_std is a user parameter, not a suggestion.

    The cell must not silently raise a configured floor to some hidden internal
    constant; a caller asking for 1e-6 gets 1e-6. Non-negativity is the only
    constraint the cell imposes.
    """
    cell = _make_cell(min_std=1e-6)
    assert cell.min_std == 1e-6

    obs = _observe(cell)
    assert torch.all(obs.prior_std >= 1e-6)
    assert obs.posterior_std is not None
    assert torch.all(obs.posterior_std >= 1e-6)


def test_min_std_and_max_std_bound_prior_and_posterior():
    cell = _make_cell(min_std=0.05, max_std=0.9)
    assert cell.min_std == 0.05
    assert cell.max_std == 0.9

    obs = _observe(cell)
    assert torch.all(obs.prior_std >= 0.05)
    assert torch.all(obs.prior_std <= 0.9)
    assert obs.posterior_std is not None
    assert torch.all(obs.posterior_std >= 0.05)
    assert torch.all(obs.posterior_std <= 0.9)


def test_negative_min_std_is_floored_to_zero():
    cell = _make_cell(min_std=-1.0)
    assert cell.min_std == 0.0


def test_max_std_is_never_below_min_std():
    cell = _make_cell(min_std=0.4, max_std=0.1)
    assert cell.max_std >= cell.min_std


def test_observe_returns_state_sampled_from_posterior():
    cell = _make_cell()
    state0 = cell.initial_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    out = cell.observe(
        prev_state=state0,
        control_embed=torch.randn(2, 6),
        exogenous_embed=torch.randn(2, 5),
        observation_embed=torch.randn(2, 7),
        sample=False,
    )
    assert out.posterior_mean is not None
    assert torch.allclose(out.state.z, out.posterior_mean)


def test_imagine_uses_prior_only_and_has_no_y_argument():
    cell = _make_cell()
    sig = inspect.signature(cell.imagine)
    assert "observation_embed" not in sig.parameters

    state0 = cell.initial_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    out = cell.imagine(
        prev_state=state0,
        control_embed=torch.randn(2, 6),
        exogenous_embed=torch.randn(2, 5),
        sample=False,
    )
    assert out.posterior is None
    assert out.posterior_mean is None
    assert out.state.z.shape == (2, 8)


def test_initial_state_zero_initialized():
    cell = _make_cell()
    state = cell.initial_state(batch_size=5, device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(state, RSSMState)
    assert state.h.shape == (5, 32)
    assert state.z.shape == (5, 8)
    assert torch.allclose(state.h, torch.zeros_like(state.h))
    assert torch.allclose(state.z, torch.zeros_like(state.z))
