"""Parsing and validation of the Jev /v1/systemone request shape.

The published schema types choice descriptions as `string | null` and instructions as
`string | object | array`. Real clients are looser than that: jev-ultrafast sends whole
objects as option descriptions. Anything JSON-serialisable is accepted in both positions and
rendered to text for the model, because rejecting a body Jev accepts would not be drop-in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import InvalidRequest

QUESTION_TYPES = ("choice", "score", "noul")
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class Question:
    name: str
    type: str
    instructions: Any
    criteria: Any


@dataclass(frozen=True)
class Request:
    model: str
    state: Any
    questions: tuple[Question, ...]


def render(value: Any, field: str) -> str:
    """Flatten a string-or-structured field into the single string the scorer consumes."""
    if isinstance(value, str):
        return value
    if value is None:
        raise InvalidRequest(f"{field} must not be null")
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidRequest(f"{field} must be JSON-serialisable") from None


def _require_state(payload: dict) -> Any:
    state = payload.get("state")
    if not isinstance(state, (str, dict, list)) or not state:
        raise InvalidRequest("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidRequest("state must be finite JSON-compatible data") from None
    return state


def _question(name: str, spec: Any) -> Question:
    if not isinstance(spec, dict):
        raise InvalidRequest(f"questions.{name} must be an object")
    kind = spec.get("type")
    if kind not in QUESTION_TYPES:
        raise InvalidRequest(f"questions.{name}.type must be one of {', '.join(QUESTION_TYPES)}")
    if "instructions" not in spec:
        raise InvalidRequest(f"questions.{name}.instructions is required")
    criteria = spec.get("criteria")

    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise InvalidRequest(f"questions.{name}.criteria must be a nonempty object")
        for key in criteria:
            if not isinstance(key, str) or not key:
                raise InvalidRequest(f"questions.{name}.criteria keys must be nonempty strings")
    elif kind == "score":
        if not isinstance(criteria, list):
            raise InvalidRequest(f"questions.{name}.criteria must be an array of levels")
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise InvalidRequest(
                f"questions.{name}.criteria must hold {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels"
            )
    elif criteria is not None:
        if not isinstance(criteria, dict):
            raise InvalidRequest(f"questions.{name}.criteria must be an object with true/false")
        unknown = set(criteria) - {"true", "false"}
        if unknown:
            raise InvalidRequest(f"questions.{name}.criteria may only hold true and false")

    return Question(name=name, type=kind, instructions=spec["instructions"], criteria=criteria)


def parse_request(payload: Any) -> Request:
    if not isinstance(payload, dict):
        raise InvalidRequest("Request body must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("model must be a nonempty string")
    state = _require_state(payload)
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise InvalidRequest("questions must be a nonempty object")
    return Request(
        model=model,
        state=state,
        questions=tuple(_question(name, spec) for name, spec in questions.items()),
    )
