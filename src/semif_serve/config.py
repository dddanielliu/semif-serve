"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Qwen3.5-4B is SemIf's direct_option_logits model. MiniCPM5-2B is deliberately not a default:
# its vocabulary merges `]}` into one token, which defeats the single-token trim SemIf uses to
# find the shared state prefix, on exactly the states this server is sent. See the README.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# Measured on a 12GB RTX 3080 Ti with a 2918-token prefix: batch 12 peaks at 10.7GiB, 16 OOMs.
DEFAULT_MAX_BATCH = 12

LOOPBACK = ("127.0.0.1", "::1", "localhost")

# TypeSafe's SDKs default to `jev-latest`, and their docs also publish `jev-preview`. Both are
# listed and accepted so those clients work unchanged. The request's `model` field is not
# otherwise validated: Jev does not document rejecting an unknown name, and refusing one would
# only break callers this server exists to support.
JEV_ALIASES = ("jev-latest", "jev-preview")
MODEL_RELEASE_DATE = "2026-09-19"


class ConfigError(Exception):
    """Refuse to start rather than serve on an unsafe or impossible configuration."""


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8077
    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    max_tokens: int = 16384
    max_batch: int = DEFAULT_MAX_BATCH
    api_key: str | None = None
    queue_seconds: float = 20.0

    @classmethod
    def from_env(cls, environ=None) -> "Settings":
        source = os.environ if environ is None else environ

        def number(name, fallback, cast=int):
            raw = source.get(name)
            if raw is None or raw == "":
                return fallback
            try:
                return cast(raw)
            except ValueError:
                raise ConfigError(f"{name} must be a number, got {raw!r}") from None

        settings = cls(
            host=source.get("SEMIF_HOST", "127.0.0.1"),
            port=number("SEMIF_PORT", 8077),
            model=source.get("SEMIF_MODEL", DEFAULT_MODEL),
            revision=source.get("SEMIF_REVISION", DEFAULT_REVISION),
            max_tokens=number("SEMIF_MAX_TOKENS", 16384),
            max_batch=number("SEMIF_MAX_BATCH", DEFAULT_MAX_BATCH),
            api_key=source.get("SEMIF_API_KEY") or None,
            queue_seconds=number("SEMIF_QUEUE_SECONDS", 20.0, float),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        # 0 is allowed: it asks the OS for an ephemeral port, which is how tests bind.
        if not 0 <= self.port <= 65535:
            raise ConfigError(f"SEMIF_PORT must be a valid port, got {self.port}")
        if self.max_tokens < 1:
            raise ConfigError("SEMIF_MAX_TOKENS must be positive")
        if self.max_batch < 0:
            raise ConfigError("SEMIF_MAX_BATCH cannot be negative")
        if self.queue_seconds <= 0:
            raise ConfigError("SEMIF_QUEUE_SECONDS must be positive")
        if self.host not in LOOPBACK and not self.api_key:
            # The decision engine answers anything it is asked; do not expose it unauthenticated.
            raise ConfigError(
                f"Refusing to bind {self.host} without SEMIF_API_KEY. "
                "Set a key, or bind 127.0.0.1 and reach it over an SSH tunnel."
            )
