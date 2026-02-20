import torch

from timesim.training.losses import ProbabilisticRolloutLoss


def _diag_normal(loc: torch.Tensor, scale: torch.Tensor) -> torch.distributions.Distribution:
    return torch.distributions.Independent(torch.distributions.Normal(loc=loc, scale=scale), 1)


def _sum_time_mean_batch(step_losses: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return step_losses.sum(dim=1).mean()
    return (step_losses * mask.to(step_losses.dtype)).sum(dim=1).mean()


def test_recon_nll_uses_distribution_log_prob_sum_time_mean_batch() -> None:
    loss_fn = ProbabilisticRolloutLoss(use_symlog=False)
    bsz, horizon, dim_y = 3, 5, 2
    targets = torch.randn(bsz, horizon, dim_y)
    loc = torch.randn(bsz, horizon, dim_y)
    scale = torch.rand(bsz, horizon, dim_y) + 0.2
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )

    dist = _diag_normal(loc, scale)
    expected = _sum_time_mean_batch(-dist.log_prob(targets), mask)

    got_from_params = loss_fn.compute_recon_nll(
        targets=targets,
        dist_loc_latent=loc,
        dist_scale=scale,
        mask=mask,
    )
    got_from_dist = loss_fn.compute_recon_nll(
        targets=targets,
        dist_loc_latent=None,
        dist_scale=None,
        recon_dist=dist,
        mask=mask,
    )

    assert torch.allclose(got_from_params, expected, atol=1e-6)
    assert torch.allclose(got_from_dist, expected, atol=1e-6)


def test_kl_divergence_matches_torch_kl_without_balancing_or_freebits() -> None:
    loss_fn = ProbabilisticRolloutLoss(use_kl_balancing=False, use_free_bits=False)
    bsz, horizon, dim_y, dim_z = 2, 4, 1, 3
    targets = torch.randn(bsz, horizon, dim_y)
    loc = torch.randn(bsz, horizon, dim_y)
    scale = torch.rand(bsz, horizon, dim_y) + 0.2
    prior_mu = torch.randn(bsz, horizon, dim_z)
    prior_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    post_mu = torch.randn(bsz, horizon, dim_z)
    post_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    _, kl, _ = loss_fn.compute_terms(
        targets=targets,
        dist_loc_latent=loc,
        dist_scale=scale,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=post_mu,
        posterior_logvar=post_logvar,
        mask=mask,
    )

    prior = torch.distributions.Normal(prior_mu, torch.exp(0.5 * prior_logvar).clamp_min(1e-6))
    post = torch.distributions.Normal(post_mu, torch.exp(0.5 * post_logvar).clamp_min(1e-6))
    kl_elem = torch.distributions.kl_divergence(post, prior)
    expected = _sum_time_mean_batch(kl_elem.sum(dim=-1), mask)
    assert torch.allclose(kl, expected, atol=1e-6)


def test_kl_balancing_and_free_bits_match_formula() -> None:
    alpha = 0.8
    free_nats = 0.7
    loss_fn = ProbabilisticRolloutLoss(
        kl_balance=alpha,
        kl_free_bits=free_nats,
        use_kl_balancing=True,
        use_free_bits=True,
    )
    bsz, horizon, dim_y, dim_z = 2, 3, 1, 4
    targets = torch.randn(bsz, horizon, dim_y)
    loc = torch.randn(bsz, horizon, dim_y)
    scale = torch.rand(bsz, horizon, dim_y) + 0.2
    prior_mu = torch.randn(bsz, horizon, dim_z)
    prior_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    post_mu = torch.randn(bsz, horizon, dim_z)
    post_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    _, kl, _ = loss_fn.compute_terms(
        targets=targets,
        dist_loc_latent=loc,
        dist_scale=scale,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=post_mu,
        posterior_logvar=post_logvar,
        mask=mask,
    )

    prior_std = torch.exp(0.5 * prior_logvar).clamp_min(1e-6)
    post_std = torch.exp(0.5 * post_logvar).clamp_min(1e-6)
    prior = torch.distributions.Normal(prior_mu, prior_std)
    post = torch.distributions.Normal(post_mu, post_std)
    prior_sg = torch.distributions.Normal(prior_mu.detach(), prior_std.detach())
    post_sg = torch.distributions.Normal(post_mu.detach(), post_std.detach())
    kl_balanced = alpha * torch.distributions.kl_divergence(post_sg, prior) + (
        1.0 - alpha
    ) * torch.distributions.kl_divergence(post, prior_sg)
    kl_fb = torch.maximum(kl_balanced, torch.full_like(kl_balanced, free_nats))
    expected = _sum_time_mean_batch(kl_fb.sum(dim=-1), mask)
    assert torch.allclose(kl, expected, atol=1e-6)


def test_aux_nll_uses_distribution_log_prob_sum_time_mean_batch() -> None:
    loss_fn = ProbabilisticRolloutLoss()
    bsz, horizon, dim_x = 3, 4, 2
    targets = torch.randn(bsz, horizon, dim_x)
    loc = torch.randn(bsz, horizon, dim_x)
    scale = torch.rand(bsz, horizon, dim_x) + 0.2
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    dist = _diag_normal(loc, scale)
    expected = _sum_time_mean_batch(-dist.log_prob(targets), mask)
    got = loss_fn.compute_aux_nll(
        targets=targets,
        aux_loc=loc,
        aux_scale=scale,
        mask=mask,
    )
    got_from_dist = loss_fn.compute_aux_nll(
        targets=targets,
        aux_loc=None,
        aux_scale=None,
        aux_dist=dist,
        mask=mask,
    )
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got_from_dist, expected, atol=1e-6)


def test_total_standard_loss_weighted_sum() -> None:
    loss_fn = ProbabilisticRolloutLoss(
        recon_weight=1.3,
        kl_weight=0.4,
        aux_weight=2.1,
        use_kl_balancing=True,
        use_free_bits=True,
    )
    bsz, horizon, dim_y, dim_x, dim_z = 2, 3, 1, 2, 4
    targets = torch.randn(bsz, horizon, dim_y)
    dist_loc_latent = torch.randn(bsz, horizon, dim_y)
    dist_scale = torch.rand(bsz, horizon, dim_y) + 0.2
    prior_mu = torch.randn(bsz, horizon, dim_z)
    prior_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    post_mu = torch.randn(bsz, horizon, dim_z)
    post_logvar = torch.randn(bsz, horizon, dim_z).clamp(-3.0, 3.0)
    exogenous_targets = torch.randn(bsz, horizon, dim_x)
    aux_loc = torch.randn(bsz, horizon, dim_x)
    aux_scale = torch.rand(bsz, horizon, dim_x) + 0.2
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    kl_beta = 0.5

    total, info = loss_fn(
        targets=targets,
        dist_loc_latent=dist_loc_latent,
        dist_scale=dist_scale,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=post_mu,
        posterior_logvar=post_logvar,
        exogenous_targets=exogenous_targets,
        aux_loc=aux_loc,
        aux_scale=aux_scale,
        mask=mask,
        kl_beta=kl_beta,
    )
    recon, kl, aux = loss_fn.compute_terms(
        targets=targets,
        dist_loc_latent=dist_loc_latent,
        dist_scale=dist_scale,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        posterior_mu=post_mu,
        posterior_logvar=post_logvar,
        exogenous_targets=exogenous_targets,
        aux_loc=aux_loc,
        aux_scale=aux_scale,
        mask=mask,
    )
    expected = loss_fn.recon_weight * recon + loss_fn.kl_weight * kl_beta * kl + loss_fn.aux_weight * aux

    assert torch.allclose(total, expected, atol=1e-6)
    assert abs(info["recon_nll"] - float(recon.detach().item())) < 1e-8
    assert abs(info["kl"] - float(kl.detach().item())) < 1e-8
    assert abs(info["aux_nll"] - float(aux.detach().item())) < 1e-8
