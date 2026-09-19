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


def confidence(probabilities: list[float]) -> float:
    """How concentrated the distribution is, as the mass on its strongest option.

    Jev publishes a `confidence` field but not how it is derived, and SemIf states plainly
    that its probabilities are uncalibrated. This is a declared, monotone stand-in: it moves
    the right way, it is always in [0, 1], and it is not comparable to Jev's number.
    """
    return max(probabilities) if probabilities else 0.0


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
            "confidence": confidence(probabilities),
            "legend": legend,
            "probabilities": distribution,
        }

    best = max(distribution, key=lambda key: distribution[key])
    return {
        "type": "choice",
        "choice": best,
        "probabilities": distribution,
        "confidence": confidence(probabilities),
    }
