import math

import pytest

from semif_serve import Service, Settings, StubEngine


@pytest.fixture
def settings():
    return Settings(port=0)


@pytest.fixture
def service(settings):
    return Service(settings, StubEngine(settings))


def assert_jev_choice(answer, option_ids):
    """Mirror of jev_ultrafast.model.validate_choice, the strictest real Jev client.

    Kept as an explicit copy rather than an import: the point is to prove the response
    satisfies that contract without depending on the agent package.
    """
    probabilities = answer["probabilities"]
    numbers = [*probabilities.values(), answer["confidence"]]
    assert answer["choice"] in option_ids
    assert set(probabilities) == set(option_ids)
    assert all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in numbers)
    assert abs(sum(probabilities.values()) - 1) < 0.02
    assert probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
