"""Scoring backends behind the HTTP layer.

`SemIfEngine` owns the model and holds the only lock that touches the GPU. `StubEngine`
answers with a deterministic distribution and no model at all, so the wire format can be
exercised in tests and on machines without CUDA.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Protocol

from .config import Settings
from .errors import InvalidRequest, Overloaded


class Engine(Protocol):
    name: str

    def score(self, state: Any, decisions: list[dict]) -> tuple[list[dict], dict]:
        """Return one result per decision, aligned by `id`, plus a timing/usage report."""


class StubEngine:
    """Deterministic scores derived from the option text. No model, no GPU, no network."""

    name = "semif-stub"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def score(self, state: Any, decisions: list[dict]) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        results = []
        for decision in decisions:
            weights = []
            for option in decision["options"]:
                seed = f"{decision['question']}\x1f{option['description']}".encode()
                weights.append(int.from_bytes(hashlib.sha256(seed).digest()[:4], "big") / 2**32 + 1e-9)
            total = sum(weights)
            results.append(
                {
                    "id": decision["id"],
                    "option_ids": [option["id"] for option in decision["options"]],
                    "probabilities": [weight / total for weight in weights],
                    "rounds": 1,
                }
            )
        return results, {
            "total_seconds": time.perf_counter() - started,
            "prefill_tokens": 0,
            "suffix_tokens": 0,
            "rounds": 1,
            "batches": 0,
        }


class SemIfEngine:
    """Load one pinned SemIf model and score every request through it, one at a time."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.name = settings.model
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._metadata: dict = {}

    def load(self) -> dict:
        # Imported here so the package stays importable, and testable, without torch.
        from semif_phase1.core import load_causal_model

        self._model, self._tokenizer, self._metadata = load_causal_model(
            self.settings.model, self.settings.revision
        )
        return self._metadata

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def warmup(self) -> float:
        """Score one throwaway decision so the first real request does not pay for JIT.

        Triton compiles the linear-attention kernels on first use, which cost ~10s of a
        ~1.3s request when it lands on a client instead of on startup.
        """
        started = time.perf_counter()
        self.score(
            "Warmup state for kernel compilation.",
            [
                {
                    "id": "warmup",
                    "question": "Is this a warmup?",
                    "options": [{"id": "true", "description": "Yes"}, {"id": "false", "description": "No"}],
                }
            ],
        )
        return time.perf_counter() - started

    def score(self, state: Any, decisions: list[dict]) -> tuple[list[dict], dict]:
        from semif_phase1.runoff import score_options

        if self._model is None:
            raise RuntimeError("Engine.load() must run before scoring")
        # One GPU, one forward pass at a time. Queue briefly, then shed load with a retryable 529.
        if not self._lock.acquire(timeout=self.settings.queue_seconds):
            raise Overloaded("Scoring queue is full; retry with backoff.")
        try:
            return score_options(
                self._model,
                self._tokenizer,
                state,
                decisions,
                self._metadata,
                max_tokens=self.settings.max_tokens,
                max_batch=self.settings.max_batch,
            )
        except ValueError as error:
            # SemIf raises ValueError for prompts that exceed max_tokens, and it refuses to
            # truncate. Surface that as a client error instead of a server fault.
            raise InvalidRequest(str(error)) from None
        finally:
            self._lock.release()
