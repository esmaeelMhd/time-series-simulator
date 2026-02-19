from timesim.serving.api import create_app


def test_api_factory_is_callable():
    assert callable(create_app)
