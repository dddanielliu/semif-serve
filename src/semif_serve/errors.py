"""Error types mapped onto the status codes Jev clients already handle."""

from __future__ import annotations


class ProtocolError(Exception):
    """An error that must reach the client as a Jev-shaped JSON body."""

    status = 500
    kind = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def body(self) -> dict:
        return {"error": {"type": self.kind, "message": self.message}}


class Unauthorized(ProtocolError):
    status = 401
    kind = "authentication_error"


class InvalidRequest(ProtocolError):
    """422, the status Jev returns for a schema violation."""

    status = 422
    kind = "invalid_request_error"


class Overloaded(ProtocolError):
    """529, which Jev clients retry with backoff rather than treating as fatal."""

    status = 529
    kind = "overloaded_error"
