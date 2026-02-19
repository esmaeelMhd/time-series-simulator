from unittest.mock import patch

import torch

from timesim.models.latent_ssm import LatentSSMWorldModel
from timesim.models.rssm import RSSMState


def _make_model() -> LatentSSMWorldModel:
    return LatentSSMWorldModel(
        input_dim=7,
        output_dim=2,
        hidden_dim=32,
        latent_dim=8,
        control_dim=3,
        exogenous_dim=2,
        encoder_dim=16,
    )


def test_forward_observation_mode_calls_rssm_observe_each_step():
    model = _make_model().eval()
    bsz, t = 4, 9
    batch = {
        "control": torch.randn(bsz, t, 3),
        "exogenous": torch.randn(bsz, t, 2),
        "objective": torch.randn(bsz, t, 2),
    }

    with patch.object(model.rssm_cell, "observe", wraps=model.rssm_cell.observe) as observe_spy:
        out = model.forward(batch, mode="observe")

    assert observe_spy.call_count == t
    assert out["prior_mu"].shape == (bsz, t, 8)
    assert out["posterior_mu"].shape == (bsz, t, 8)
    assert out["predictions"].shape == (bsz, t, 2)
    assert len(out["objective_dists"]) == t
    assert len(out["objective_dists_latent"]) == t


def test_imagine_forward_calls_rssm_imagine_each_step():
    model = _make_model().eval()
    bsz, h = 3, 6
    state0 = RSSMState(
        h=torch.zeros(bsz, model.hidden_dim),
        z=torch.zeros(bsz, model.latent_dim),
    )
    controls = torch.randn(bsz, h, 3)
    exogenous = torch.randn(bsz, h, 2)

    with patch.object(model.rssm_cell, "imagine", wraps=model.rssm_cell.imagine) as imagine_spy:
        out = model.imagine_forward(
            initial_state=state0,
            future_controls=controls,
            future_exogenous=exogenous,
            n_steps=h,
            n_samples=1,
            sample_latent=True,
        )

    assert imagine_spy.call_count == h
    assert out["predictions"].shape == (bsz, h, 2)
    assert out["dist_scale"].shape == (bsz, h, 2)
    assert len(out["objective_dists"]) == h


def test_imagine_rollout_with_loss_splits_context_and_horizon():
    model = _make_model().eval()
    bsz, t = 2, 12
    batch = {
        "control": torch.randn(bsz, t, 3),
        "exogenous": torch.randn(bsz, t, 2),
        "objective": torch.randn(bsz, t, 2),
    }
    context_len, horizon = 8, 4
    out = model.imagine_rollout_with_loss(
        batch=batch,
        context_len=context_len,
        horizon=horizon,
        sample_posterior=True,
        sample_prior=True,
    )

    assert out["context_len"] == context_len
    assert out["horizon"] == horizon
    assert torch.isfinite(out["obs_recon_nll"])
    assert torch.isfinite(out["obs_kl"])
    assert torch.isfinite(out["obs_aux_nll"])
    assert torch.isfinite(out["rollout_nll"])
    assert torch.isfinite(out["rollout_aux_nll"])
    assert torch.isfinite(out["rollout_dtw"])

    observed = out["observed"]
    imagined = out["imagined"]
    assert observed["predictions"].shape == (bsz, context_len, 2)
    assert imagined is not None
    assert imagined["predictions"].shape == (bsz, horizon, 2)
