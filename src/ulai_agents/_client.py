"""Client: a connection to an agent server, and everything done over it."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import grpc
from ulai.agent.v1 import agent_pb2 as pb
from ulai.agent.v1 import agent_pb2_grpc as rpc

from ._agent import Agent, Room, opening_pb, tool_problems
from ._errors import (
    AgentError,
    CallEndedError,
    CanceledError,
    UnavailableError,
    ValidationError,
    status_error,
)
from ._events import ms
from ._session import Session
from ._tools import Tool
from ._types import CallInfo

__all__ = ["Client", "ErrorHandler", "Handler", "Job", "LeaveResult", "Seat"]

log = logging.getLogger("ulai_agents")

# Where the server looks for the key. Lower-case because gRPC normalises
# metadata keys.
API_KEY_HEADER = "x-api-key"

# How long run waits to take the agent out of the room once the call is over.
LEAVE_TIMEOUT = 10.0

# How serve backs off while the server is unavailable.
BACKOFF_START = 1.0
BACKOFF_MAX = 30.0

# Patched in tests to watch serve's backoff without waiting it out.
_sleep = asyncio.sleep

Handler = Callable[[Session], Awaitable[None] | None]
"""An agent's own logic for one call: ``async def handle(session) -> None``."""

ErrorHandler = Callable[["Job | None", BaseException], Awaitable[None] | None]
"""Receives what serve could not do: ``on_error(job, error)``; job is None for
a lost connection to the server."""


@dataclass(frozen=True)
class Seat:
    """An agent's place in a room."""

    room: str
    agent_id: str
    participant_id: str
    connection_id: str
    """Changes on every rejoin, which is why leave keys on participant_id."""
    node: str
    """The media server the room is on, for log correlation."""
    _client: Client = field(repr=False, compare=False)

    async def leave(self, reason: str = "") -> LeaveResult:
        """Take this agent out of its room."""
        return await self._client.leave(self.room, participant_id=self.participant_id, reason=reason)


@dataclass(frozen=True)
class LeaveResult:
    """What an agent's time in a room amounted to."""

    present: bool
    """False when the agent was already gone. Leave still succeeds."""
    duration: timedelta
    heard: timedelta
    """How much agent audio the caller heard."""


@dataclass(frozen=True)
class Job:
    """A call the server offers this client: it is already running on the
    agent's instructions, and is yours once you attach."""

    session_id: str
    call: CallInfo
    attach_by: timedelta
    """How long the call waits for a client before carrying on without one."""


class Client:
    """A connection to an agent server. One client serves every call a process
    makes::

        async with agent.Client("agent-server:443", api_key=key, tls=True) as client:
            await client.run(room, support, handle)

    It does not dial until first used, so a server that is down is reported by
    the first call, as UnavailableError.

    Without ``tls`` the connection is unencrypted, which is allowed only to
    this machine (localhost, 127.0.0.1, ::1, a unix socket, an SSH tunnel).
    Anything else needs ``tls=True``, or ``insecure=True`` on a network you
    trust end to end: an agent's cloud key and room token travel with every
    call. ``root_certificates`` (PEM) verifies the server against a private
    issuer, and implies TLS.
    """

    def __init__(
        self,
        address: str,
        *,
        api_key: str | None = None,
        tls: bool = False,
        insecure: bool = False,
        root_certificates: bytes | str | None = None,
    ) -> None:
        if not address:
            raise ValidationError("Client needs the server's address, as host:port")
        if root_certificates is not None:
            pem = root_certificates.encode() if isinstance(root_certificates, str) else bytes(root_certificates)
            if b"-----BEGIN CERTIFICATE-----" not in pem:
                raise ValidationError("root_certificates: no certificates in the PEM given")
            root_certificates, tls = pem, True
        if not tls and not insecure and not _is_local(address):
            raise ValidationError(
                f"{address} is not this machine, and an unencrypted connection would carry the agent's "
                "credentials in the clear: pass tls=True, or insecure=True on a network you trust"
            )
        self._address = address
        self._credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates) if tls else None
        self._metadata: tuple[tuple[str, str], ...] | None = ((API_KEY_HEADER, api_key),) if api_key else None
        self._channel: grpc.aio.Channel | None = None
        self._rooms: rpc.RoomStub
        self._sessions: rpc.SessionStub
        self._dispatch: rpc.DispatchStub

    def __repr__(self) -> str:
        return f"<Client {self._address} tls={self._credentials is not None}>"

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the connection. Calls in progress on the server carry on."""
        channel, self._channel = self._channel, None
        if channel is not None:
            await channel.close()

    def _connect(self) -> None:
        # Made on first use, inside the event loop the client is used from.
        if self._channel is not None:
            return
        if self._credentials is not None:
            self._channel = grpc.aio.secure_channel(self._address, self._credentials)
        else:
            self._channel = grpc.aio.insecure_channel(self._address)
        self._rooms = rpc.RoomStub(self._channel)
        self._sessions = rpc.SessionStub(self._channel)
        self._dispatch = rpc.DispatchStub(self._channel)

    # ── Rooms ───────────────────────────────────────────────────────────────

    async def join(
        self,
        room: Room,
        agent: Agent,
        *,
        participant_id: str = "",
        fields: Mapping[str, Any] | None = None,
        allow_unlisted_values: bool = False,
    ) -> Seat:
        """Put an agent into a room. Returns once it has a seat, not once it is
        talking: the model connects a second or two later.

        The call's session id is ``room.id``, for attach. ``fields`` are custom
        fields for the call, such as your own call id; they come back in
        ``CallInfo.fields``. ``allow_unlisted_values`` sends models, voices and
        languages this SDK does not know and leaves the server to judge them:
        for a platform forwarding agents stored in its own database.
        """
        if not isinstance(agent, Agent):
            raise ValidationError(f"join needs an agent.Agent, not {type(agent).__name__}")
        problems = agent._problems(lists=not allow_unlisted_values)
        if not isinstance(room, Room):
            problems.append(f"join needs an agent.Room, not {type(room).__name__}")
        else:
            if not room.id:
                problems.append("join needs Room.id: the server never creates rooms")
            if not room.url:
                problems.append("join needs Room.url, the control plane holding the room")
        custom = ""
        if fields:
            if not isinstance(fields, Mapping):
                problems.append(f"fields must be a dict, not {type(fields).__name__}")
            else:
                try:
                    custom = json.dumps(dict(fields), separators=(",", ":"))
                except (TypeError, ValueError) as e:
                    problems.append(f"fields cannot be encoded as JSON: {e}")
        if problems:
            raise ValidationError(problems)

        self._connect()
        request = pb.JoinRoomRequest(
            agent_id=agent.id,
            room_id=room.id,
            participant_id=participant_id,
            custom_fields_json=custom,
            opening=opening_pb(agent.opening),
            room_url=room.url,
            room_token=room.token,
            profile=agent._profile(),
        )
        try:
            resp = await self._rooms.Join(request, metadata=self._metadata)
        except grpc.aio.AioRpcError as e:
            raise status_error("join", e, room=True) from e
        return Seat(
            room=resp.room_id,
            agent_id=resp.agent_id,
            participant_id=resp.participant_id,
            connection_id=resp.connection_id,
            node=resp.node_address,
            _client=self,
        )

    async def leave(
        self,
        room_id: str,
        *,
        participant_id: str = "",
        reason: str = "",
        force: bool = False,
    ) -> LeaveResult:
        """Take an agent out of a room, whoever put it there.

        ``participant_id`` is needed when the room holds several agents.
        ``force`` takes the agent out without its closing line.
        """
        if not room_id:
            raise ValidationError("leave needs a room id")
        self._connect()
        request = pb.LeaveRequest(room_id=room_id, participant_id=participant_id, reason=reason, force=force)
        try:
            resp = await self._rooms.Leave(request, metadata=self._metadata)
        except grpc.aio.AioRpcError as e:
            raise status_error("leave", e, room=True) from e
        return LeaveResult(present=resp.was_connected, duration=ms(resp.duration_ms), heard=ms(resp.played_ms))

    # ── Sessions ────────────────────────────────────────────────────────────

    async def attach(
        self,
        session_id: str,
        *,
        tools: Sequence[Tool] = (),
        history: bool = False,
    ) -> Session:
        """Follow and steer a running call. The session id of a call started
        with join is its room id.

        ``tools`` with a handler are answered by this client. ``history``
        replays what the call has already said before live events, for a
        client attaching mid-call; it cannot be combined with tools, because
        the replay includes past tool calls, which would be answered again.

        Close the session when you are done. Closing detaches; the call
        carries on.
        """
        if not session_id:
            raise ValidationError("attach needs a session id")
        problems = tool_problems(tools, "tools") if tools else []
        if problems:
            raise ValidationError(problems)
        handlers = {t.name: t for t in tools if t.handler is not None}
        if history and handlers:
            raise ValidationError(
                "history=True replays past tool calls, which tools would answer again; "
                "attach without history to answer tools"
            )

        self._connect()
        call = self._sessions.Run(metadata=self._metadata)
        try:
            try:
                await call.write(pb.ClientEvent(attach=pb.Attach(session_id=session_id, replay_history=history)))
            except (asyncio.InvalidStateError, grpc.aio.AioRpcError):
                pass  # the stream failed: the read below says why
            # The server answers an Attach with SessionStarted, or with an
            # error such as NotFound for a call that has ended.
            first = await call.read()
        except grpc.aio.AioRpcError as e:
            call.cancel()
            raise status_error("attach", e) from e
        except BaseException:
            call.cancel()
            raise
        if first is grpc.aio.EOF:
            call.cancel()
            raise AgentError(f"attach {session_id}: the server ended the stream before the session started")
        kind = first.WhichOneof("event")
        if kind != "session_started":
            call.cancel()
            raise AgentError(f"attach {session_id}: expected the session to start, got {kind or 'an empty event'}")

        session = Session(call, first.session_started, handlers)
        # Claim the tools this client answers, so the server waits for our
        # answer instead of answering them itself. A call that ended as we
        # attached is not an attach failure: events() reports how it ended.
        if handlers:
            try:
                await session._send("handled_tools", pb.ClientEvent(
                    handled_tools=pb.HandledTools(names=sorted(handlers))
                ))
            except CallEndedError:
                pass
            except BaseException:
                call.cancel()
                raise
        session._start()
        return session

    # ── Outbound ────────────────────────────────────────────────────────────

    async def run(
        self,
        room: Room,
        agent: Agent,
        handler: Handler | None = None,
        *,
        participant_id: str = "",
        fields: Mapping[str, Any] | None = None,
        allow_unlisted_values: bool = False,
    ) -> None:
        """The whole outbound call in one step: put the agent in the room,
        attach with its tools, run handler, and take the agent out of the room
        when handler returns, fails or is cancelled.

        A None handler runs the call on the agent's instructions alone, until
        it ends. Raises CanceledError when cancelled, and CallLostError when
        the server dropped the call.
        """
        try:
            seat = await self.join(
                room, agent, participant_id=participant_id, fields=fields, allow_unlisted_values=allow_unlisted_values
            )
        except CanceledError:
            raise
        except asyncio.CancelledError as e:
            raise CanceledError("run: canceled") from e
        try:
            session = await self.attach(room.id, tools=agent.tools)
            try:
                await _call_handler(handler or _until_ended, session)
            finally:
                await session.close()
        except CanceledError:
            raise
        except asyncio.CancelledError as e:
            raise CanceledError("run: canceled") from e
        finally:
            # Leave even when cancelled, or the agent stays in the room.
            await _leave_quietly(seat)
        if _cancelling():
            # The handler swallowed a cancellation and returned.
            raise CanceledError("run: canceled")

    # ── Inbound ─────────────────────────────────────────────────────────────

    async def serve(
        self,
        agent_id: str,
        tools: Sequence[Tool] = (),
        handler: Handler | None = None,
        *,
        name: str = "",
        on_error: ErrorHandler | None = None,
    ) -> None:
        """The inbound side: run handler on every call for agent_id as each one
        starts, each as its own task, answering tools. Runs until cancelled.

        An inbound call already runs on the configuration it was started with,
        so serve takes only the agent's id and the tools this client answers.
        Run one serve per agent: a call is offered to every client serving its
        agent, and the one that claimed tools last would answer them.

        A handler's exceptions, calls that could not be attached to, and lost
        connections to the server (with job None) go to on_error, by default
        logged, and serving carries on. When the server is unavailable, serve
        reconnects with backoff; other errors end it. Cancelling serve cancels
        the running handlers and waits for them.
        """
        if not agent_id:
            raise ValidationError("serve needs an agent id")
        problems = tool_problems(tools, "tools")
        if problems:
            raise ValidationError(problems)
        handler = handler or _until_ended
        report = _reporter(on_error)

        running: set[asyncio.Task[None]] = set()
        try:
            backoff = BACKOFF_START
            while True:
                connected = False
                try:
                    async for job in self.jobs([agent_id], name=name):
                        connected, backoff = True, BACKOFF_START
                        task = asyncio.ensure_future(self._serve_one(tools, handler, job, report))
                        running.add(task)
                        task.add_done_callback(running.discard)
                except UnavailableError as e:
                    await report(None, UnavailableError(
                        f"{e}; reconnecting in {_seconds(backoff)}", code=e.code, details=e.details
                    ))
                if not connected:
                    await _sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX)
        except asyncio.CancelledError:
            for t in running:
                t.cancel()
            raise
        finally:
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    async def _serve_one(
        self,
        tools: Sequence[Tool],
        handler: Handler,
        job: Job,
        report: Callable[[Job | None, BaseException], Awaitable[None]],
    ) -> None:
        # No history: the job is offered as the call starts, and a replay
        # would answer its tool calls a second time.
        try:
            session = await self.attach(job.session_id, tools=tools)
        except Exception as e:
            if _cancelling():
                raise
            await report(job, e)
            return
        try:
            await _call_handler(handler, session)
        except Exception as e:
            if _cancelling():
                raise
            await report(job, e)
        finally:
            await session.close()

    def jobs(self, agent_ids: Sequence[str] | str, *, name: str = "") -> AsyncIterator[Job]:
        """Calls for these agents as the server starts them, for a client that
        dispatches them itself (its own queue, a concurrency limit). Attach to
        a job's session_id to take it. ``name`` names this client in the
        server's logs.
        """
        ids = [agent_ids] if isinstance(agent_ids, str) else list(agent_ids or ())
        if not ids or not all(ids):
            # The server reads an empty list as every agent, which on a shared
            # server means taking other people's calls.
            raise ValidationError("jobs needs at least one agent id")
        return self._jobs(ids, name)

    async def _jobs(self, ids: list[str], name: str) -> AsyncIterator[Job]:
        self._connect()
        call = self._dispatch.Subscribe(pb.SubscribeRequest(agent_ids=ids, client_name=name), metadata=self._metadata)
        try:
            while True:
                try:
                    j = await call.read()
                except grpc.aio.AioRpcError as e:
                    raise status_error("subscribe", e) from e
                if j is grpc.aio.EOF:
                    return
                yield Job(session_id=j.session_id, call=CallInfo._from_pb(j.call), attach_by=ms(j.attach_deadline_ms))
        finally:
            call.cancel()


async def _call_handler(handler: Handler, session: Session) -> None:
    result = handler(session)
    if inspect.isawaitable(result):
        await result


async def _until_ended(session: Session) -> None:
    """Wait for the call to end, raising for a call that was lost."""
    async for _ in session.events():
        pass


async def _leave_quietly(seat: Seat) -> None:
    leave = asyncio.ensure_future(asyncio.wait_for(seat.leave("done"), LEAVE_TIMEOUT))
    leave.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        await asyncio.shield(leave)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning("ulai_agents: leaving room %s: %s", seat.room, e)


def _cancelling() -> bool:
    """Whether the current task is being cancelled."""
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)  # Python 3.11+
    return bool(cancelling and cancelling())


def _seconds(s: float) -> str:
    return f"{s:g}s"


def _reporter(on_error: ErrorHandler | None) -> Callable[[Job | None, BaseException], Awaitable[None]]:
    async def report(job: Job | None, err: BaseException) -> None:
        try:
            if on_error is None:
                _log_error(job, err)
                return
            result = on_error(job, err)
            if inspect.isawaitable(result):
                await result
        except Exception:
            log.exception("ulai_agents: serve: on_error failed")

    return report


def _log_error(job: Job | None, err: BaseException) -> None:
    exc_info = None if isinstance(err, AgentError) else err
    if job is None:
        log.error("ulai_agents: serve: %s", err, exc_info=exc_info)
    else:
        log.error("ulai_agents: call %s: %s", job.session_id, err, exc_info=exc_info)


def _is_local(address: str) -> bool:
    """Whether address is this machine."""
    if address.startswith("unix:"):
        return True
    for scheme in ("dns:///", "passthrough:///"):
        if address.startswith(scheme):
            address = address[len(scheme):]
    host = address
    if address.startswith("["):
        end = address.find("]")
        if end != -1:
            host = address[1:end]
    elif address.count(":") == 1:
        host = address.split(":", 1)[0]
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
