import pytest

from semif_serve.protocol import Question
from semif_serve.translate import answer_for, confidence, decision_for, options_for


def question(kind, criteria, instructions="Why?"):
    return Question(name="q", type=kind, instructions=instructions, criteria=criteria)


def test_choice_options_follow_criteria_keys():
    decision = decision_for(question("choice", {"access": "Account access.", "billing": "Billing."}))
    assert decision["id"] == "q"
    assert [option["id"] for option in decision["options"]] == ["access", "billing"]
    assert decision["options"][0]["description"] == "Account access."


def test_null_description_falls_back_to_the_option_name():
    options = options_for(question("choice", {"WAIT": None}))
    assert options == [{"id": "WAIT", "description": "WAIT"}]


def test_object_descriptions_are_serialised_for_the_model():
    options = options_for(question("choice", {"1": {"element": "[1] Go", "role": "button"}}))
    assert options[0]["description"] == '{"element": "[1] Go", "role": "button"}'


def test_structured_instructions_become_the_question_text():
    decision = decision_for(question("choice", {"a": "A", "b": "B"}, instructions={"goal": "Book a flight"}))
    assert decision["question"] == '{"goal": "Book a flight"}'


def test_score_levels_become_indexed_options():
    options = options_for(question("score", ["Cosmetic", "Degraded", "Blocking"]))
    assert [option["id"] for option in options] == ["0", "1", "2"]
    assert options[2]["description"] == "Blocking"


def test_noul_defaults_to_yes_and_no_but_honours_criteria():
    assert options_for(question("noul", None)) == [
        {"id": "true", "description": "Yes"},
        {"id": "false", "description": "No"},
    ]
    options = options_for(question("noul", {"true": "Wants a human"}))
    assert options[0]["description"] == "Wants a human"
    assert options[1]["description"] == "No"


def test_choice_answer_matches_the_published_shape():
    answer = answer_for(question("choice", {"a": "A", "b": "B"}), ["a", "b"], [0.25, 0.75])
    assert answer == {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.25, "b": 0.75},
        "confidence": 0.75,
    }


def test_score_answer_is_the_probability_weighted_mean_with_a_legend():
    """The documented example: levels 0/1/2 at 0.0/0.7/0.3 gives a score of 1.3."""
    spec = question("score", ["Cosmetic", "Degraded", "Blocking"])
    answer = answer_for(spec, ["0", "1", "2"], [0.0, 0.7, 0.3])
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(1.3)
    assert answer["legend"] == {"0": "Cosmetic", "1": "Degraded", "2": "Blocking"}
    assert answer["probabilities"] == {"0": 0.0, "1": 0.7, "2": 0.3}
    assert answer["confidence"] == pytest.approx(0.7)


def test_noul_answer_is_only_the_probability_of_yes():
    answer = answer_for(question("noul", None), ["true", "false"], [0.99, 0.01])
    assert answer == {"type": "noul", "noul": 0.99}
    assert "confidence" not in answer and "probabilities" not in answer


def test_confidence_tracks_concentration():
    assert confidence([0.5, 0.5]) == 0.5
    assert confidence([0.9, 0.1]) == 0.9
    assert confidence([]) == 0.0


def test_misaligned_scores_are_a_server_fault_not_a_silent_answer():
    with pytest.raises(RuntimeError):
        answer_for(question("choice", {"a": "A", "b": "B"}), ["a", "b"], [1.0])
    with pytest.raises(RuntimeError):
        answer_for(question("choice", {"a": "A", "b": "B"}), ["a", "b"], [float("nan"), 1.0])
