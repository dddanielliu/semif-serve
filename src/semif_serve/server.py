"""The Jev-compatible HTTP surface.

`Service` holds the whole request path and knows nothing about sockets, so the wire contract
is testable without binding a port. The handler below is a thin adapter over it.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import JEV_ALIASES, MODEL_RELEASE_DATE, Settings
from .engine import Engine
from .errors import InvalidRequest, ProtocolError, Unauthorized
from .protocol import parse_request
from .translate import answer_for, decision_for

ENDPOINT = "/v1/systemone"
MODELS = "/v1/models"
HEALTH = "/health"
MAX_BODY_BYTES = 32 * 1024 * 1024

logger = logging.getLogger("semif_serve")


class Service:
    def __init__(self, settings: Settings, engine: Engine):
        self.settings = settings
        self.engine = engine

    def authorize(self, header: str | None) -> None:
        expected = self.settings.api_key
        if not expected:
            return
        supplied = ""
        if header and header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        if not secrets.compare_digest(supplied, expected):
            raise Unauthorized("Invalid or missing API key")

    def models(self) -> dict:
        """The documented `{name, description, release_date}` listing.

        The served model is named first, then the Jev aliases, so a TypeSafe SDK listing
        models sees something it recognises and its default `jev-latest` resolves here.
        """
        served = self.engine.name
        entries = [
            {
                "name": served,
                "description": f"SemIf option-logit readout on {served}, served over the Jev protocol.",
                "release_date": MODEL_RELEASE_DATE,
            }
        ]
        entries += [
            {
                "name": alias,
                "description": f"Accepted for TypeSafe SDK compatibility. Resolves to {served}.",
                "release_date": MODEL_RELEASE_DATE,
            }
            for alias in JEV_ALIASES
        ]
        return {"models": entries}

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.engine.name,
            "max_tokens": self.settings.max_tokens,
            "max_batch": self.settings.max_batch,
        }

    def systemone(self, payload: Any) -> dict:
        request = parse_request(payload)
        decisions = [decision_for(question) for question in request.questions]
        results, timing = self.engine.score(request.state, decisions)

        by_id = {result["id"]: result for result in results}
        missing = [question.name for question in request.questions if question.name not in by_id]
        if missing:
            raise RuntimeError(f"Engine returned no result for: {', '.join(missing)}")

        answers = {}
        for question in request.questions:
            result = by_id[question.name]
            answers[question.name] = answer_for(question, result["option_ids"], result["probabilities"])
        return {
            "model": self.engine.name,
            "answers": answers,
            # Nothing is sampled: the readout is a logit lookup, so there are no output tokens.
            "usage": {
                "input_tokens": timing.get("prefill_tokens", 0) + timing.get("suffix_tokens", 0),
                "output_tokens": 0,
            },
        }


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "semif-serve"

        def log_message(self, fmt, *args):
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, error: ProtocolError) -> None:
            self._send(error.status, error.body())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == HEALTH:
                self._send(200, service.health())
                return
            if path == MODELS:
                # Listing models needs no key: it is how a client discovers what to ask for.
                self._send(200, service.models())
                return
            self._send(404, {"error": {"type": "not_found", "message": "Unknown path"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path != ENDPOINT:
                self._send(404, {"error": {"type": "not_found", "message": "Unknown path"}})
                return
            try:
                service.authorize(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise InvalidRequest("Request body is required")
                if length > MAX_BODY_BYTES:
                    raise InvalidRequest("Request body is too large")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise InvalidRequest("Request body must be valid JSON") from None
                self._send(200, service.systemone(payload))
            except ProtocolError as error:
                self._fail(error)
            except Exception:
                logger.exception("Unhandled error while scoring")
                self._send(500, {"error": {"type": "internal_error", "message": "Scoring failed"}})

    return Handler


def build_server(settings: Settings, engine: Engine) -> ThreadingHTTPServer:
    """A bound, not-yet-serving server. Call `serve_forever()` to run it in the foreground."""
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(Service(settings, engine)))
    server.daemon_threads = True
    return server


def serve(settings: Settings, engine: Engine) -> ThreadingHTTPServer:
    """Serve in a background thread. Callers own shutdown; tests use this."""
    server = build_server(settings, engine)
    threading.Thread(target=server.serve_forever, name="semif-serve", daemon=True).start()
    return server
