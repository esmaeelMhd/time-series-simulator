from timesim.models.rssm import RSSMState


def test_rssm_state_has_expected_fields():
    assert hasattr(RSSMState, "__annotations__")
    assert "h" in RSSMState.__annotations__
    assert "z" in RSSMState.__annotations__
