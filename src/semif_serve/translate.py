"""Mapping between Jev questions/answers and SemIf option-scoring decisions.

All three Jev primitives reduce to one readout: score a fixed option set and read the
distribution back. Choice options are the criteria keys, score options are the rubric level
indices, and noul is a two-option yes/no whose `noul` value is P(yes).
"""

from __future__ import annotations

import math

from .errors import InvalidRequest
from .protocol import Question, render

NOUL_DEFAULTS = {"true": "Yes", "false": "No"}


def score_confidence(probabilities: list[float]) -> float:
    """How peaked an ordered level distribution is: 1 - sigma / sigma_max.

    Recovered from Jev's own published score example. Levels 0/1/2 at 0.0/0.7/0.3 give
    sigma = 0.4583 against a maximum of (n-1)/2 = 1.0, so 1 - 0.4583 = 0.5417, and Jev
    publishes 0.54. Max-probability (0.70), normalised entropy (0.44) and top-two margin
    (0.40) all miss it, so this is the statistic Jev uses for score.
    """
    count = len(probabilities)
    if count < 2:
        return 1.0
    mean = sum(index * value for index, value in enumerate(probabilities))
    variance = sum(index * index * value for index, value in enumerate(probabilities)) - mean * mean
    deviation = math.sqrt(max(variance, 0.0))
    # Spread is greatest with the mass split between the two end levels.
    return _clamp(1.0 - deviation / ((count - 1) / 2))


def choice_confidence(probabilities: list[float]) -> float:
    """How peaked an unordered distribution is: 1 - H / log(n).

    Choice options have no ordering, so score's standard deviation has no meaning here and
    Jev does not publish this one. Normalised entropy is the usual dispersion measure for a
    categorical distribution and matches the documented behaviour: a flat shape scores 0, a
    single peak scores 1. Inferred, not verified against a published value.
    """
    count = len(probabilities)
    if count < 2:
        return 1.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return _clamp(1.0 - entropy / math.log(count))


def _clamp(value: float) -> float:
    """Keep float noise inside the [0, 1] range every Jev client validates against."""
    return min(1.0, max(0.0, value))


def options_for(question: Question) -> list[dict]:
    """The option set a question is scored over, in a stable order."""
    if question.type == "choice":
        options = []
        for key, description in question.criteria.items():
            # A null description is documented; the option name is then all the model gets.
            text = key if description is None else render(description, f"questions.{question.name}.criteria.{key}")
            options.append({"id": key, "description": text})
        return options
    if question.type == "score":
        return [
            {"id": str(index), "description": render(level, f"questions.{question.name}.criteria[{index}]")}
            for index, level in enumerate(question.criteria)
        ]
    criteria = question.criteria or {}
    return [
        {
            "id": key,
            "description": render(criteria.get(key, NOUL_DEFAULTS[key]), f"questions.{question.name}.criteria.{key}"),
        }
        for key in ("true", "false")
    ]


def decision_for(question: Question) -> dict:
    options = options_for(question)
    if not options:
        raise InvalidRequest(f"questions.{question.name} has no options to score")
    return {
        "id": question.name,
        "question": render(question.instructions, f"questions.{question.name}.instructions"),
        "options": options,
    }


def answer_for(question: Question, option_ids: list[str], probabilities: list[float]) -> dict:
    if len(option_ids) != len(probabilities):
        raise RuntimeError(f"questions.{question.name} scored {len(probabilities)} of {len(option_ids)} options")
    if not all(math.isfinite(value) for value in probabilities):
        raise RuntimeError(f"questions.{question.name} produced a non-finite probability")
    distribution = dict(zip(option_ids, probabilities))

    if question.type == "noul":
        # A noul answer is the probability of yes, with no confidence or distribution.
        return {"type": "noul", "noul": distribution["true"]}

    if question.type == "score":
        legend = {
            str(index): render(level, f"questions.{question.name}.criteria[{index}]")
            for index, level in enumerate(question.criteria)
        }
        return {
            "type": "score",
            "score": sum(int(index) * value for index, value in distribution.items()),
            "confidence": score_confidence(probabilities),
            "legend": legend,
            "probabilities": distribution,
        }

    best = max(distribution, key=lambda key: distribution[key])
    return {
        "type": "choice",
        "choice": best,
        "probabilities": distribution,
        "confidence": choice_confidence(probabilities),
    }
