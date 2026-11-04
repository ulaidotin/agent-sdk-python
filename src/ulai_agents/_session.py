"""Session: one live call, attached."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import weakref
from collections import deque
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

import grpc
from ulai.agent.v1 import agent_pb2 as pb

from ._errors import (
    AgentError,
    CallEndedError,
    CallLostError,
    EventsInUseError,
    SessionClosedError,
    status_error,
)
from ._events import Ended, Event, Rejected, ToolCall, event_from
from ._models import Role, wire
from ._tools import Tool, ToolError
from ._types import BackendInfo, CallInfo

__all__ = ["Session"]

log = logging.getLogger("ulai_agents")

# How many recent commands a Rejected event can be matched to.
REJECT_WINDOW = 256

_ENCODE_FAILED = '{"error":"the tool\'s result could not be encoded"}'


class Session:
    """One live call, attached: you see what happens on it and steer it.

    From the moment ``attach`` returns, the session receives in the background
    and answers tool calls as they arrive, whether or not anything is reading
    ``events()``.

    Close it when you are done (or use it as ``async with``). Closing detaches;
    the call carries on.
    """

    def __init__(self, call: Any, started: pb.SessionStarted, tools: dict[str, Tool]) -> None:
        self._call = call
        self._id = started.session_id
        self._info = CallInfo._from_pb(started.call)
        self._backend = BackendInfo._from_pb(started.backend)
        self._tools = tools

        # The lock serialises writes. _seq numbers each event sent after the
        # Attach, as the server numbers them, so a Rejected event can name the
        # command it refuses.
        self._send_lock = asyncio.Lock()
        self._seq = 0
        self._sent: dict[int, str] = {}
        self._closed = False

        # Unbounded, so a client that is slow to read never stops tool calls
        # being answered.
        self._items: deque[Event] = deque()
        self._finished = False
        self._end_error: BaseException | None = None
        self._wake = asyncio.Event()
        self._reader: weakref.ref[_Events] | None = None
        self._reader_task: asyncio.Task[Any] | None = None

        self._receiver: asyncio.Task[None] | None = None
        self._tool_tasks: set[asyncio.Task[None]] = set()

    def __repr__(self) -> str:
        return f"<Session {self._id!r} closed={self._closed}>"

    @property
    def id(self) -> str:
        """The session's id. For a call started with join, the room id."""
        return self._id

    @property
    def call(self) -> CallInfo:
        """Describes the call."""
        return self._info

    @property
    def backend(self) -> BackendInfo:
        """Describes what is running the call."""
        return self._backend

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Events ──────────────────────────────────────────────────────────────

    def events(self) -> AsyncIterator[Event]:
        """Everything that happens on the call, in order, until it ends::

            async for event in session.events():
                match event:
                    case agent.Transcript(final=True): ...
                    case agent.Ended(): ...

        Read it with one ``async for`` at a time; a second, concurrent reader
        gets EventsInUseError. Breaking out and reading again later is fine:
        nothing is lost in between. Steer the call from anywhere, including
        from inside the loop.

        It raises CallLostError when the server dropped the call without it
        ending (a call whose model failed to connect, for instance), or a
        connection error.
        """
        return _Events(self)

    async def _next(self) -> Event | None:
        """The next event, or None at the end. Raises how the call was lost."""
        while True:
            if self._items:
                return self._items.popleft()
            if self._finished:
                if self._end_error is not None:
                    raise self._end_error.with_traceback(None)
                return None
            self._wake.clear()
            await self._wake.wait()

    def _push(self, ev: Event) -> None:
        self._items.append(ev)
        self._wake.set()

    def _finish(self, err: BaseException | None) -> None:
        if not self._finished:
            self._finished, self._end_error = True, err
        self._wake.set()

    # ── Receiving ───────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._receiver = asyncio.ensure_future(self._receive())

    async def _receive(self) -> None:
        """Read the stream until it ends, answering tools as they arrive and
        queueing every event for events()."""
        ended = False
        try:
            while True:
                msg = await self._call.read()
                if msg is grpc.aio.EOF:
                    if ended or self._closed:
                        self._finish(None)
                    else:
                        # The server finished the stream without saying the
                        # call ended: it failed to start, or it was dropped.
                        self._finish(CallLostError(
                            "the call was lost before it ended: the server closed the stream without an "
                            "Ended event (most often the model failed to connect: check Agent.cloud)"
                        ))
                    return
                ev = event_from(msg)
                if ev is None:
                    continue
                if isinstance(ev, ToolCall):
                    self._answer(ev)
                elif isinstance(ev, Rejected):
                    ev = dataclasses.replace(ev, command=self._sent.get(ev.sequence_ref, ""))
                elif isinstance(ev, Ended):
                    ended = True
                self._push(ev)
        except asyncio.CancelledError:
            self._finish(None)
            raise
        except grpc.aio.AioRpcError as e:
            self._finish(None if ended or self._closed else status_error("receive", e))
        except Exception as e:  # a bug here must still end events() rather than hang it
            log.exception("ulai_agents: session %s: receiving failed", self._id)
            self._finish(None if ended or self._closed else AgentError(f"receive: {e}"))

    def _answer(self, call: ToolCall) -> None:
        """Run a tool's handler on its own task, so a slow lookup does not hold
        up the events behind it."""
        t = self._tools.get(call.name)
        if t is None:
            return
        task = asyncio.ensure_future(self._run_tool(t, call))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool(self, t: Tool, call: ToolCall) -> None:
        try:
            out = await t._invoke(call.args_json)
        except ToolError as e:
            out = {"error": str(e)}
        except asyncio.CancelledError:
            raise
        except Exception:
            # Reported to the model as a failed tool rather than taking the
            # session, and every call on it, down.
            log.exception("ulai_agents: tool %s failed", call.name)
            out = {"error": "the tool failed"}
        if out is None:
            out = {"ok": True}
        try:
            body = json.dumps(out, separators=(",", ":"), ensure_ascii=False, default=_encode)
        except (TypeError, ValueError):
            log.exception("ulai_agents: tool %s returned a result that is not JSON", call.name)
            body = _ENCODE_FAILED
        try:
            await self._send("tool_result", pb.ClientEvent(
                tool_result=pb.ToolResult(id=call.id, name=call.name, result_json=body)
            ))
        except AgentError as e:
            log.debug("ulai_agents: tool %s: result not sent: %s", call.name, e)

    # ── Commands ────────────────────────────────────────────────────────────
    #
    # Each returns once the command is sent; the server applies it
    # asynchronously, and reports one it cannot apply as a Rejected event
    # naming the command. Commands sent in the first second or two of a call,
    # before the model is connected, are rejected.

    async def reply(self, instructions: str = "") -> None:
        """Ask the model to speak now, optionally with instructions for what to say."""
        await self._send("reply", pb.ClientEvent(generate_reply=pb.GenerateReply(instructions=instructions)))

    async def inject_context(self, role: Role | str, text: str) -> None:
        """Tell the model something without it being spoken: a lookup that
        finished mid-call, an action taken elsewhere."""
        await self._send("inject_context", pb.ClientEvent(
            inject_context=pb.InjectContext(text=text, role=wire(role))
        ))

    async def interrupt(self) -> None:
        """Stop the agent mid-sentence."""
        await self._send("interrupt", pb.ClientEvent(interrupt=pb.Interrupt()))

    async def hangup(self, reason: str = "") -> None:
        """End the call now."""
        await self._send("hangup", pb.ClientEvent(hangup=pb.Hangup(reason=reason)))

    async def _send(self, command: str, ev: pb.ClientEvent) -> None:
        if self._closed:
            raise SessionClosedError()
        # Shielded: a caller cancelled mid-write must not cancel the stream
        # under it, which would take the whole session down.
        await asyncio.shield(self._write(command, ev))

    async def _write(self, command: str, ev: pb.ClientEvent) -> None:
        async with self._send_lock:
            if self._closed:
                raise SessionClosedError()
            try:
                await send_message(self._call, ev)
            except asyncio.InvalidStateError:
                # The stream is finished: for a session that means the call is over.
                if self._closed:
                    raise SessionClosedError() from None
                raise CallEndedError(f"{command}: the call has ended") from None
            except grpc.aio.AioRpcError as e:
                if self._closed:
                    raise SessionClosedError() from None
                details = e.details() or ""
                message = f"{command}: the call has ended" + (f" ({details})" if details else "")
                raise CallEndedError(message, code=e.code(), details=details) from None
            except asyncio.CancelledError:
                if self._closed or self._call.cancelled():
                    raise SessionClosedError() from None
                raise
            self._seq += 1
            self._sent[self._seq] = command
            if self._seq > REJECT_WINDOW:
                self._sent.pop(self._seq - REJECT_WINDOW, None)

    async def close(self) -> None:
        """Detach from the call. It does not hang up: the call carries on."""
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [t for t in (self._receiver, *self._tool_tasks) if t is not None and t is not current]
        try:
            # Say we are going, then go. The server reads either as a detach.
            if not self._send_lock.locked() and not self._call.done():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._call.done_writing(), 1.0)
        finally:
            self._call.cancel()
            for t in tasks:
                if t is not self._receiver:
                    t.cancel()
            self._finish(None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class _Events:
    """The iterator events() returns.

    One reader at a time: the reader is the iterator that last read, for as
    long as it is alive and its task running. An `async for` that breaks drops
    its iterator, so reading again afterwards just works.
    """

    def __init__(self, session: Session) -> None:
        self._s = session
        self._done = False

    def __aiter__(self) -> _Events:
        return self

    async def __anext__(self) -> Event:
        s = self._s
        if self._done:
            raise StopAsyncIteration
        self._acquire()
        try:
            ev = await s._next()
        except asyncio.CancelledError:
            self._release()
            raise
        except BaseException:
            self._done = True
            self._release()
            raise
        if ev is None:
            self._done = True
            self._release()
            raise StopAsyncIteration
        return ev

    async def aclose(self) -> None:
        self._done = True
        self._release()

    def _acquire(self) -> None:
        s = self._s
        owner = s._reader() if s._reader is not None else None
        task = asyncio.current_task()
        if owner is not None and owner is not self:
            owner_task = s._reader_task
            if owner_task is not None and owner_task is not task and not owner_task.done():
                raise EventsInUseError()
        s._reader = weakref.ref(self)
        s._reader_task = task

    def _release(self) -> None:
        s = self._s
        if s._reader is not None and s._reader() is self:
            s._reader = None
            s._reader_task = None


async def send_message(call: Any, ev: pb.ClientEvent) -> None:
    """Write ev on a stream whose first message has been sent. A stream the
    server has finished raises asyncio.InvalidStateError and stays readable.

    grpc.aio's call.write marks the whole call failed (INTERNAL, "Internal
    error from Core") when a write races the server finishing the stream. That
    throws away the events still waiting to be read, the Ended among them, and
    the status the server actually sent. Go's Send reports io.EOF and leaves
    Recv alone; this does the same by sending through the call's Cython layer,
    falling back to call.write where that layer is not there.
    """
    send = getattr(getattr(call, "_cython_call", None), "send_serialized_message", None)
    if send is None:
        await call.write(ev)
        return
    if call.done():
        raise asyncio.InvalidStateError("the stream is finished")
    try:
        await send(ev.SerializeToString())
    except asyncio.CancelledError:
        if not call.cancelled():
            call.cancel()
        raise
    except Exception as e:  # cygrpc.ExecuteBatchError: the server finished the stream
        raise asyncio.InvalidStateError(f"the stream is finished: {e}") from None


def _encode(v: Any) -> Any:
    """JSON for what json does not know: dataclasses and enums."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return dataclasses.asdict(v)
    if isinstance(v, Enum):
        return v.value
    raise TypeError(f"{type(v).__name__} is not JSON serializable")
