"""The errors this SDK raises. Catch ``AgentError`` for all of them.

Every failure that comes from the server keeps the server's own message in
``str(err)`` and in ``err.details``, and its gRPC status in ``err.code``.
"""

from __future__ import annotations

import asyncio

import grpc

__all__ = [
    "AgentError",
    "AmbiguousParticipantError",
    "CallEndedError",
    "CallLostError",
    "CanceledError",
    "DeadlineError",
    "EventsInUseError",
    "InvalidRequestError",
    "NotFoundError",
    "NotSupportedError",
    "RoomFullError",
    "RoomNotFoundError",
    "SessionClosedError",
    "UnauthorizedError",
    "UnavailableError",
    "ValidationError",
]


class AgentError(Exception):
    """The base of every error this SDK raises."""

    summary = "agent error"

    code: grpc.StatusCode | None
    """The gRPC status the server answered with, when the error came from it."""
    details: str
    """The server's own message, when the error came from it."""

    def __init__(
        self,
        message: str = "",
        *,
        code: grpc.StatusCode | None = None,
        details: str = "",
    ) -> None:
        super().__init__(message or self.summary)
        self.code = code
        self.details = details


class ValidationError(AgentError, ValueError):
    """Something passed to the SDK cannot be sent: ``problems`` lists every one."""

    summary = "invalid"

    problems: list[str]

    def __init__(self, problems: list[str] | str) -> None:
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        if len(self.problems) == 1:
            message = self.problems[0]
        else:
            lines = "\n".join(f"  - {p}" for p in self.problems)
            message = f"{len(self.problems)} problems:\n{lines}"
        super().__init__(message)


class UnauthorizedError(AgentError):
    """The API key was missing or wrong."""

    summary = "unauthorized"


class UnavailableError(AgentError):
    """The server could not be reached or is not taking work. Worth a retry."""

    summary = "server unavailable"


class InvalidRequestError(AgentError):
    """The server rejected the request as malformed."""

    summary = "invalid request"


class NotFoundError(AgentError):
    """The thing named does not exist: a session that has ended, for instance."""

    summary = "not found"


class NotSupportedError(AgentError):
    """This server does not serve that: a build that runs no calls of its own,
    or a server on another protocol version."""

    summary = "not supported by this server"


class DeadlineError(AgentError, TimeoutError):
    """The deadline passed before an answer came back."""

    summary = "timed out"


class CanceledError(AgentError, asyncio.CancelledError):
    """The work was cancelled.

    It is also an ``asyncio.CancelledError``, so asyncio's own handling of
    cancellation (``asyncio.timeout``, task groups, Ctrl-C under
    ``asyncio.run``) keeps working when it passes through.
    """

    summary = "canceled"


class CallEndedError(AgentError):
    """A command was sent to a call that has already ended."""

    summary = "the call has ended"


class CallLostError(AgentError):
    """The server dropped the call without it ending: most often its model
    failed to connect (check ``Agent.cloud``), or the call was torn down."""

    summary = "the call was lost before it ended"


class SessionClosedError(AgentError):
    """A command was sent after ``Session.close``."""

    summary = "the session is closed"


class EventsInUseError(AgentError):
    """A second, concurrent reader of ``Session.events``."""

    summary = "the session's events are already being read"


class RoomNotFoundError(AgentError):
    """The room does not exist, or has ended."""

    summary = "room not found"


class RoomFullError(AgentError):
    """The room is at its participant limit. Retrying may work once someone leaves."""

    summary = "room is full"


class AmbiguousParticipantError(AgentError):
    """The room holds several agents and leave did not say which one: pass
    ``participant_id``."""

    summary = "several agents in this room"


_BY_CODE: dict[grpc.StatusCode, type[AgentError]] = {
    grpc.StatusCode.UNAUTHENTICATED: UnauthorizedError,
    grpc.StatusCode.PERMISSION_DENIED: UnauthorizedError,
    grpc.StatusCode.UNAVAILABLE: UnavailableError,
    grpc.StatusCode.INVALID_ARGUMENT: InvalidRequestError,
    grpc.StatusCode.NOT_FOUND: NotFoundError,
    grpc.StatusCode.UNIMPLEMENTED: NotSupportedError,
    grpc.StatusCode.DEADLINE_EXCEEDED: DeadlineError,
    grpc.StatusCode.CANCELLED: CanceledError,
}

# Join and Leave, where three codes mean something specific about the room.
_ROOM_BY_CODE: dict[grpc.StatusCode, type[AgentError]] = {
    grpc.StatusCode.NOT_FOUND: RoomNotFoundError,
    grpc.StatusCode.RESOURCE_EXHAUSTED: RoomFullError,
    grpc.StatusCode.FAILED_PRECONDITION: AmbiguousParticipantError,
}


def status_error(op: str, err: grpc.aio.AioRpcError, *, room: bool = False) -> AgentError:
    """Turn a gRPC failure into one of the errors above, keeping the server's message."""
    code = err.code()
    details = err.details() or ""
    cls = _ROOM_BY_CODE.get(code) if room else None
    cls = cls or _BY_CODE.get(code)
    if cls is None:
        name = code.name if code is not None else "UNKNOWN"
        return AgentError(f"{op}: {name}: {details}", code=code, details=details)
    message = f"{op}: {cls.summary}"
    if details:
        message += f" ({details})"
    return cls(message, code=code, details=details)
