"""Training-time guardrails for RSSM safety constraints."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    return bool(v)


def merged_latent_ssm_params(
    model_defaults_cfg: Mapping[str, Any],
    per_model_cfg: Mapping[str, Any],
    model_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(dict(model_defaults_cfg.get("latent_ssm", {}) or {}))
    merged.update({k: v for k, v in dict(per_model_cfg or {}).items() if k != "type"})
    if model_overrides:
        merged.update(dict(model_overrides))
    return merged


def merged_probabilistic_cfg(
    training_cfg: Mapping[str, Any],
    training_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(training_cfg.get("probabilistic", {}) or {})
    if training_overrides:
        for k, v in dict(training_overrides).items():
            cfg[k] = v
    return cfg


def validate_latent_ssm_do_not(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    prob_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    context: str = "",
) -> None:
    """Raise ValueError when an RSSM "DO NOT" constraint is violated."""

    where = f" ({context})" if context else ""
    leak_y = _as_bool(model_params.get("leak_objective_to_transition", False))
    allow_leak = _as_bool(model_params.get("allow_objective_leak_for_ablation", False))
    use_aux = _as_bool(model_params.get("use_aux_decoder", True), default=True)
    allow_no_aux = _as_bool(model_params.get("allow_disable_aux_decoder_for_ablation", False))
    share_enc = _as_bool(model_params.get("share_encoder_weights", False))
    allow_share_enc = _as_bool(model_params.get("allow_shared_encoder_for_ablation", False))
    use_stochastic = _as_bool(model_params.get("use_stochastic_path", True), default=True)
    allow_no_stochastic = _as_bool(
        model_params.get("allow_disable_stochastic_for_ablation", False)
    )

    if leak_y and not allow_leak:
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: objective leakage into transition is disabled by policy. "
            "Set `allow_objective_leak_for_ablation: true` only for explicit ablation runs."
        )
    if (not use_aux) and (not allow_no_aux):
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: auxiliary exogenous decoder cannot be disabled by default. "
            "Set `allow_disable_aux_decoder_for_ablation: true` only for explicit ablation runs."
        )
    if share_enc and not allow_share_enc:
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: shared encoder weights across roles are blocked by default. "
            "Set `allow_shared_encoder_for_ablation: true` only for explicit ablation runs."
        )
    if (not use_stochastic) and (not allow_no_stochastic):
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: stochastic latent path cannot be disabled by default. "
            "Set `allow_disable_stochastic_for_ablation: true` only for explicit ablation runs."
        )

    prob_enabled = _as_bool(prob_cfg.get("enabled", True), default=True)
    if not prob_enabled:
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: probabilistic RSSM training must remain enabled "
            "(Gaussian NLL + KL), not point-estimate mode."
        )

    objective = str(prob_cfg.get("objective", "rssm")).lower()
    if objective != "rssm":
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: expected `training.probabilistic.objective: rssm`, got '{objective}'."
        )

    checkpoint_metric = str(prob_cfg.get("checkpoint_metric", "open_loop_crps")).lower()
    allow_recon_only_eval = _as_bool(
        prob_cfg.get("allow_reconstruction_only_eval_for_ablation", False),
        default=False,
    )
    if checkpoint_metric != "open_loop_crps" and not allow_recon_only_eval:
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: checkpoint selection must use open-loop quality "
            "(`checkpoint_metric: open_loop_crps`), not reconstruction-only metrics."
        )

    if _as_bool(data_cfg.get("shuffle_within_window", False), default=False):
        raise ValueError(
            f"[{model_name}{where}] DO-NOT violation: `data.shuffle_within_window` must stay false "
            "(never permute timesteps inside a window)."
        )
