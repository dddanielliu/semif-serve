import pytest

from semif_serve.errors import InvalidRequest
from semif_serve.protocol import parse_request, render


def body(**questions):
    return {"model": "jev-latest", "state": "A customer cannot sign in.", "questions": questions}


def test_parses_all_three_primitives_in_one_request():
    request = parse_request(
        body(
            queue={"type": "choice", "instructions": "Which queue?", "criteria": {"a": "Access", "b": "Billing"}},
            severity={"type": "score", "instructions": "How severe?", "criteria": ["Low", "Mid", "High"]},
            urgent={"type": "noul", "instructions": "Is it urgent?"},
        )
    )
    assert request.model == "jev-latest"
    assert {question.name: question.type for question in request.questions} == {
        "queue": "choice",
        "severity": "score",
        "urgent": "noul",
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"model": "", "state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        {"model": "m", "state": "", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        {"model": "m", "state": "s", "questions": {}},
        {"model": "m", "state": "s", "questions": {"q": {"type": "vibes", "instructions": "?"}}},
        {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}},
        {"model": "m", "state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}},
        {"model": "m", "state": "s", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["a"]}}},
        {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["x"] * 11}},
        },
        {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "hm"}}},
        },
    ],
)
def test_schema_violations_are_client_errors(payload):
    with pytest.raises(InvalidRequest):
        parse_request(payload)


def test_structured_state_and_instructions_survive():
    """jev-ultrafast sends an object state and an object instructions block."""
    request = parse_request(
        {
            "model": "jev-latest",
            "state": {"page": {"url": "https://example.test"}, "elements": [{"index": "1"}]},
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": {"goal": "Find a book", "rules": ["one", "two"]},
                    "criteria": {"CLICK": {"element": "[1] Go"}, "WAIT": None},
                }
            },
        }
    )
    (question,) = request.questions
    assert request.state["page"]["url"] == "https://example.test"
    assert question.instructions["goal"] == "Find a book"
    assert question.criteria["WAIT"] is None


def test_render_flattens_structure_and_rejects_the_unserialisable():
    assert render("plain", "f") == "plain"
    assert render({"b": 1, "a": 2}, "f") == '{"b": 1, "a": 2}'
    assert render(["x", 1], "f") == '["x", 1]'
    with pytest.raises(InvalidRequest):
        render(None, "f")
    with pytest.raises(InvalidRequest):
        render(float("nan"), "f")
    with pytest.raises(InvalidRequest):
        render(object(), "f")


def test_non_finite_state_is_rejected():
    with pytest.raises(InvalidRequest):
        parse_request(
            {
                "model": "m",
                "state": {"score": float("inf")},
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            }
        )
