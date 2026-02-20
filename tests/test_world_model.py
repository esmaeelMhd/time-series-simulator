from timesim.models.world_model import WorldModel


def test_world_model_alias_points_to_latent_ssm():
    assert WorldModel.__name__ == "LatentSSMWorldModel"
