import pytest

from timesim.training.safety import validate_latent_ssm_do_not


def _base_prob_cfg():
    return {
        "enabled": True,
        "objective": "rssm",
        "checkpoint_metric": "open_loop_crps",
    }


def _base_data_cfg():
    return {"shuffle_within_window": False}


def test_guardrails_allow_standard_rssm_settings():
    validate_latent_ssm_do_not(
        model_name="latent_ssm",
        model_params={
            "use_aux_decoder": True,
            "leak_objective_to_transition": False,
        },
        prob_cfg=_base_prob_cfg(),
        data_cfg=_base_data_cfg(),
        context="unit_test",
    )


def test_guardrails_block_objective_leak_without_ablation_flag():
    with pytest.raises(ValueError, match="objective leakage"):
        validate_latent_ssm_do_not(
            model_name="latent_ssm",
            model_params={
                "use_aux_decoder": True,
                "leak_objective_to_transition": True,
                "allow_objective_leak_for_ablation": False,
            },
            prob_cfg=_base_prob_cfg(),
            data_cfg=_base_data_cfg(),
            context="unit_test",
        )


def test_guardrails_block_aux_decoder_disable_without_ablation_flag():
    with pytest.raises(ValueError, match="auxiliary exogenous decoder"):
        validate_latent_ssm_do_not(
            model_name="latent_ssm",
            model_params={
                "use_aux_decoder": False,
                "allow_disable_aux_decoder_for_ablation": False,
                "leak_objective_to_transition": False,
            },
            prob_cfg=_base_prob_cfg(),
            data_cfg=_base_data_cfg(),
            context="unit_test",
        )


def test_guardrails_block_reconstruction_only_checkpointing():
    with pytest.raises(ValueError, match="checkpoint selection"):
        validate_latent_ssm_do_not(
            model_name="latent_ssm",
            model_params={
                "use_aux_decoder": True,
                "leak_objective_to_transition": False,
            },
            prob_cfg={
                "enabled": True,
                "objective": "rssm",
                "checkpoint_metric": "val_loss",
            },
            data_cfg=_base_data_cfg(),
            context="unit_test",
        )


def test_guardrails_block_shared_encoder_without_ablation_flag():
    with pytest.raises(ValueError, match="shared encoder weights"):
        validate_latent_ssm_do_not(
            model_name="latent_ssm",
            model_params={
                "use_aux_decoder": True,
                "leak_objective_to_transition": False,
                "share_encoder_weights": True,
                "allow_shared_encoder_for_ablation": False,
            },
            prob_cfg=_base_prob_cfg(),
            data_cfg=_base_data_cfg(),
            context="unit_test",
        )


def test_guardrails_block_no_stochastic_without_ablation_flag():
    with pytest.raises(ValueError, match="stochastic latent path"):
        validate_latent_ssm_do_not(
            model_name="latent_ssm",
            model_params={
                "use_aux_decoder": True,
                "leak_objective_to_transition": False,
                "use_stochastic_path": False,
                "allow_disable_stochastic_for_ablation": False,
            },
            prob_cfg=_base_prob_cfg(),
            data_cfg=_base_data_cfg(),
            context="unit_test",
        )
