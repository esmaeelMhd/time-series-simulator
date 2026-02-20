import torch

from timesim.models.decoders import AuxiliaryDecoder, ObjectiveDecoder
from timesim.models.latent_ssm import LatentSSMWorldModel


def test_objective_decoder_returns_gaussian_distribution():
    dec = ObjectiveDecoder(in_dim=20, out_dim=3, hidden_dim=32, num_layers=2)
    latent = torch.randn(5, 20)
    dist, mu, sigma = dec(latent, min_scale=0.01)

    assert mu.shape == (5, 3)
    assert sigma.shape == (5, 3)
    assert torch.all(sigma >= 0.01)
    lp = dist.log_prob(torch.randn(5, 3))
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


def test_auxiliary_decoder_returns_gaussian_distribution():
    dec = AuxiliaryDecoder(in_dim=16, out_dim=4, hidden_dim=24, num_layers=2)
    latent = torch.randn(7, 16)
    dist, mu, sigma = dec(latent, min_scale=0.01)

    assert mu.shape == (7, 4)
    assert sigma.shape == (7, 4)
    assert torch.all(sigma >= 0.01)
    lp = dist.log_prob(torch.randn(7, 4))
    assert lp.shape == (7,)
    assert torch.isfinite(lp).all()


def test_latent_ssm_decoder_outputs_include_distributions_and_variance():
    model = LatentSSMWorldModel(
        input_dim=6,
        output_dim=2,
        hidden_dim=32,
        latent_dim=8,
        control_dim=3,
        exogenous_dim=2,
        use_aux_decoder=True,
    )
    model.eval()

    h = torch.randn(4, model.hidden_dim)
    z = torch.randn(4, model.latent_dim)
    y_dist, y_dist_latent, y_mu, y_sigma, _ = model._decode_obs(h, z)  # pylint: disable=protected-access
    x_dist, x_mu, x_sigma = model._decode_exogenous(h, z, exogenous_dim=2)  # pylint: disable=protected-access

    assert y_mu.shape == (4, 2)
    assert y_sigma.shape == (4, 2)
    assert torch.all(y_sigma >= 0.01)
    assert y_dist.log_prob(torch.randn(4, 2)).shape == (4,)
    assert y_dist_latent.log_prob(torch.randn(4, 2)).shape == (4,)

    assert x_dist is not None
    assert x_mu.shape == (4, 2)
    assert x_sigma.shape == (4, 2)
    assert torch.all(x_sigma >= 0.01)
    assert x_dist.log_prob(torch.randn(4, 2)).shape == (4,)
