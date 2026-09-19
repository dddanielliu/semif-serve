import json
import urllib.error
import urllib.request

import pytest
from conftest import assert_jev_choice

from semif_serve import Service, Settings, StubEngine
from semif_serve.errors import InvalidRequest, Overloaded, Unauthorized
from semif_serve.server import serve


def flights_request(targets=40):
    """The shape jev-ultrafast actually sends: one state, three heads, object criteria."""
    return {
        "model": "jev-latest",
        "state": {
            "page": {"url": "https://flights.test", "title": "Flights", "text": "Search flights"},
            "elements": [{"index": str(i + 1), "label": f"Control {i + 1}"} for i in range(targets)],
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Fly to London", "rules": "Advance the goal."},
                "criteria": {"CLICK": "Click an element.", "TYPE_TEXT": "Enter text.", "DONE": "All done."},
            },
            "click_target": {
                "type": "choice",
                "instructions": {"goal": "Fly to London", "operation": "CLICK"},
                "criteria": {
                    str(i + 1): {"element": f"[{i + 1}] Control {i + 1}", "role": "button"} for i in range(targets)
                },
            },
        },
    }


def test_full_request_answers_every_question(service):
    response = service.systemone(flights_request())
    assert set(response["answers"]) == {"operation", "click_target"}
    assert response["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert response["model"] == "semif-stub"


def test_response_satisfies_the_strictest_client_contract(service):
    payload = flights_request(targets=250)
    response = service.systemone(payload)
    assert_jev_choice(response["answers"]["operation"], ["CLICK", "TYPE_TEXT", "DONE"])
    assert_jev_choice(response["answers"]["click_target"], [str(i + 1) for i in range(250)])


def test_single_option_head_is_answerable(service):
    """jev-ultrafast can offer exactly one target; a scorer needs two, so this must short-circuit."""
    response = service.systemone(
        {
            "model": "m",
            "state": "s",
            "questions": {"only": {"type": "choice", "instructions": "Pick", "criteria": {"e1": "The one field"}}},
        }
    )
    assert response["answers"]["only"] == {
        "type": "choice",
        "choice": "e1",
        "probabilities": {"e1": 1.0},
        "confidence": 1.0,
    }


def test_mixed_primitives_return_their_own_shapes(service):
    response = service.systemone(
        {
            "model": "m",
            "state": "A customer cannot sign in after a reset.",
            "questions": {
                "department": {"type": "choice", "instructions": "Route it", "criteria": {"a": "Access", "b": "Bill"}},
                "frustration": {"type": "score", "instructions": "How upset?", "criteria": ["Calm", "Mad", "Livid"]},
                "is_urgent": {"type": "noul", "instructions": "Urgent?"},
            },
        }
    )
    answers = response["answers"]
    assert answers["department"]["type"] == "choice" and "choice" in answers["department"]
    assert 0.0 <= answers["frustration"]["score"] <= 2.0
    assert set(answers["frustration"]["legend"]) == {"0", "1", "2"}
    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0
    assert set(answers["is_urgent"]) == {"type", "noul"}


def test_authorization_is_enforced_only_when_a_key_is_configured():
    open_service = Service(Settings(port=0), StubEngine())
    open_service.authorize(None)

    guarded = Service(Settings(port=0, api_key="secret"), StubEngine())
    guarded.authorize("Bearer secret")
    for header in (None, "", "secret", "Bearer wrong", "Basic secret"):
        with pytest.raises(Unauthorized):
            guarded.authorize(header)


def test_engine_failures_map_onto_retryable_and_fatal_statuses(settings):
    class Failing(StubEngine):
        name = "failing"

        def __init__(self, error):
            super().__init__()
            self.error = error

        def score(self, state, decisions):
            raise self.error

    payload = {"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    for error, status in ((Overloaded("busy"), 529), (InvalidRequest("too long"), 422)):
        with pytest.raises(type(error)) as caught:
            Service(settings, Failing(error)).systemone(payload)
        assert caught.value.status == status


def call(base, path, payload=None, headers=None):
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    if data:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def live():
    settings = Settings(port=0, api_key="topsecret")
    server = serve(settings, StubEngine(settings))
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_health_needs_no_key_and_reports_the_model(live):
    status, body = call(live, "/health")
    assert status == 200 and body["status"] == "ok" and body["model"] == "semif-stub"


def test_endpoint_over_http_round_trips(live):
    status, body = call(live, "/v1/systemone", flights_request(), {"Authorization": "Bearer topsecret"})
    assert status == 200
    assert_jev_choice(body["answers"]["click_target"], [str(i + 1) for i in range(40)])


@pytest.mark.parametrize(
    "headers,status",
    [({}, 401), ({"Authorization": "Bearer nope"}, 401), ({"Authorization": "Bearer topsecret"}, 422)],
)
def test_auth_precedes_validation(live, headers, status):
    code, body = call(live, "/v1/systemone", {"model": "m"}, headers)
    assert code == status
    assert "error" in body and body["error"]["message"]


def test_unknown_paths_are_404(live):
    assert call(live, "/v1/nope", {"model": "m"}, {"Authorization": "Bearer topsecret"})[0] == 404
    assert call(live, "/nope")[0] == 404


def test_malformed_json_is_422_not_a_crash(live):
    request = urllib.request.Request(
        f"{live}/v1/systemone",
        data=b"{not json",
        headers={"Authorization": "Bearer topsecret", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10)
        raise AssertionError("expected an error status")
    except urllib.error.HTTPError as error:
        assert error.code == 422


def test_models_listing_matches_the_published_shape(service):
    body = service.models()
    assert set(body) == {"models"}
    for entry in body["models"]:
        assert set(entry) == {"name", "description", "release_date"}
        assert entry["name"] and entry["description"]
        assert len(entry["release_date"]) == 10 and entry["release_date"][4] == "-"


def test_models_advertises_the_served_model_and_the_jev_aliases(service):
    names = [entry["name"] for entry in service.models()["models"]]
    assert names[0] == "semif-stub", "the real model is named first"
    # TypeSafe SDKs default to jev-latest; the docs also publish jev-preview.
    assert "jev-latest" in names and "jev-preview" in names


def test_models_is_reachable_without_a_key(live):
    """Listing is how a client discovers what to ask for, so it must not be gated."""
    status, body = call(live, "/v1/models")
    assert status == 200
    assert "jev-latest" in [entry["name"] for entry in body["models"]]


def test_an_unrecognised_model_name_is_still_answered(service):
    """Jev does not document rejecting one, and refusing would break the clients we serve."""
    response = service.systemone(
        {"model": "jev-1.13.0", "state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    )
    assert 0.0 <= response["answers"]["q"]["noul"] <= 1.0
