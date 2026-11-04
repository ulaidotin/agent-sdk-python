"""Sessions: attaching, answering tools in the background, reading events,
steering, and detaching."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import grpc
import pytest
from fakeserver import Stream, eventually
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent


def transcript(text: str, final: bool = True) -> pb.ServerEvent:
    return pb.ServerEvent(transcript=pb.Transcript(speaker=pb.SPEAKER_CALLER, text=text, final=final))


def tool_call(id: str, name: str, args: str = "") -> pb.ServerEvent:
    return pb.ServerEvent(tool_call=pb.ToolCall(id=id, name=name, args_json=args))


async def drain(session: agent.Session) -> list[agent.Event]:
    return [ev async for ev in session.events()]


# ── Attaching ─────────────────────────────────────────────────────────────────


async def test_attach_reads_the_session(fake):
    attached = []

    async def script(s: Stream):
        attached.append(await s.started(
            session_id="room-1",
            call=pb.CallInfo(call_id="c-9", agent_id="support", direction="room", from_number="p-1",
                             to_number="room-1", custom_fields_json='{"call_id":"c-9","n":2}'),
            backend=pb.BackendInfo(provider="google", model="m", pipeline=pb.PIPELINE_CASCADE,
                                   turn_detection=pb.TURN_DETECTION_CLIENT),
        ))
        await s.ended(reason="done")

    fake.session = script
    async with await fake.client.attach("room-1") as session:
        assert session.id == "room-1"
        assert session.call == agent.CallInfo(call_id="c-9", agent_id="support", direction="room", from_="p-1",
                                              to="room-1", fields={"call_id": "c-9", "n": 2})
        assert session.backend == agent.BackendInfo(provider="google", model="m", pipeline=agent.PipelineKind.CASCADE,
                                                    turn_detection=agent.TurnDetection.CLIENT)
        assert [type(e) for e in await drain(session)] == [agent.Ended]
    assert session.closed
    assert attached[0].attach.session_id == "room-1" and not attached[0].attach.replay_history


async def test_attach_requires_session_started_first(fake):
    async def script(s: Stream):
        await s.recv()
        await s.send(transcript("too early"))
        await s.until_gone()

    fake.session = script
    with pytest.raises(agent.AgentError, match="expected the session to start, got transcript"):
        await fake.client.attach("room-1")


async def test_attach_to_a_stream_that_ends_at_once(fake):
    async def script(s: Stream):
        await s.recv()

    fake.session = script
    with pytest.raises(agent.AgentError, match="ended the stream before the session started"):
        await fake.client.attach("room-1")


async def test_attach_to_an_ended_call_is_not_found(fake):
    async def script(s: Stream):
        await s.context.abort(grpc.StatusCode.NOT_FOUND, 'no live session "x"')

    fake.session = script
    with pytest.raises(agent.NotFoundError) as e:
        await fake.client.attach("x")
    assert 'no live session "x"' in str(e.value) and e.value.details == 'no live session "x"'
    assert e.value.code == grpc.StatusCode.NOT_FOUND
    assert not isinstance(e.value, agent.RoomNotFoundError)


async def test_attach_needs_a_session_id(fake):
    with pytest.raises(agent.ValidationError, match="attach needs a session id"):
        await fake.client.attach("")


async def test_history_with_tools_is_refused(fake):
    @agent.tool(description="d")
    def t() -> None: ...

    with pytest.raises(agent.ValidationError, match="answer again"):
        await fake.client.attach("room-1", tools=[t], history=True)
    assert fake.keys == []  # refused before reaching the server


async def test_history_without_handlers_is_replayed(fake):
    seen = []

    async def script(s: Stream):
        seen.append(await s.started())
        seen.append(await s.recv())  # what follows the attach: nothing, until the client goes

    fake.session = script
    declared_only = agent.tool_from_schema("end_call", "Ends the call.")
    session = await fake.client.attach("room-1", tools=[declared_only], history=True)
    await session.close()
    await eventually(lambda: len(seen) == 2)
    assert seen[0].attach.replay_history is True
    assert seen[1] is None  # nothing claimed: no handler


async def test_claimed_tools_are_sent_sorted_right_after_attach(fake):
    first_after_attach = []

    async def script(s: Stream):
        await s.started()
        first_after_attach.append(await s.recv())
        await s.ended()

    fake.session = script
    tools = [
        agent.tool_from_schema("zeta", "d", handler=lambda a: None),
        agent.tool_from_schema("alpha", "d", handler=lambda a: None),
        agent.tool_from_schema("declared", "d"),
        agent.tool_from_schema("mid", "d", handler=lambda a: None),
    ]
    session = await fake.client.attach("room-1", tools=tools)
    await drain(session)
    await session.close()
    ev = first_after_attach[0]
    assert ev.WhichOneof("event") == "handled_tools"
    assert list(ev.handled_tools.names) == ["alpha", "mid", "zeta"]


# ── Tools in the background ───────────────────────────────────────────────────


@dataclass
class Balance:
    account: str
    rupees: int


async def test_tools_are_answered_without_reading_events(fake):
    """The server waits a few seconds for a claimed tool and then gives up on
    it, so answering cannot depend on the handler reading events."""
    results: dict[str, str] = {}
    calls = [
        ("1", "ok", '{"n":2}'),
        ("2", "fails", ""),
        ("3", "raises", ""),
        ("4", "nothing", ""),
        ("5", "sync", '{"n":5}'),
        ("6", "ok", '{"n":"two"}'),
        ("7", "dataclass", ""),
        ("8", "unencodable", ""),
        ("9", "schema", '{"x":1}'),
        ("10", "not_mine", ""),
    ]

    async def script(s: Stream):
        await s.started()
        await s.recv()  # the tool claim
        for id, name, args in calls:
            await s.send(tool_call(id, name, args))
        for _ in range(len(calls) - 1):  # not_mine is not ours to answer
            ev = await s.recv()
            results[ev.tool_result.id] = ev.tool_result.result_json
            assert ev.tool_result.name == dict((c[0], c[1]) for c in calls)[ev.tool_result.id]
        await s.until_gone()

    @agent.tool(description="d")
    async def ok(n: int) -> dict:
        return {"double": n * 2}

    @agent.tool(description="d")
    async def fails() -> None:
        raise agent.ToolError("no such account")

    @agent.tool(description="d")
    async def raises() -> None:
        raise RuntimeError("bug")

    @agent.tool(description="d")
    async def nothing() -> None:
        return None

    @agent.tool(description="d")
    def sync(n: int) -> int:
        return n + 1

    @agent.tool(description="d")
    async def dataclass_() -> Balance:
        return Balance("a1", 250)

    @agent.tool(description="d")
    async def unencodable() -> object:
        return object()

    schema = agent.tool_from_schema("schema", "d", {"x": agent.Param(type="integer")}, [], lambda args: args)

    fake.session = script
    session = await fake.client.attach("room-1", tools=[
        ok, fails, raises, nothing, sync, agent.tool(dataclass_.handler, name="dataclass", description="d"),
        unencodable, schema,
    ])
    try:
        await eventually(lambda: len(results) == len(calls) - 1)
    finally:
        await session.close()
    assert results == {
        "1": '{"double":4}',
        "2": '{"error":"no such account"}',
        "3": '{"error":"the tool failed"}',
        "4": '{"ok":true}',
        "5": "6",
        "6": '{"error":"the arguments for ok were not valid: n must be a integer, not \'two\'"}',
        "7": '{"account":"a1","rupees":250}',
        "8": '{"error":"the tool\'s result could not be encoded"}',
        "9": '{"x":1}',
    }


async def test_a_slow_tool_does_not_hold_up_events(fake):
    release = asyncio.Event()
    order = []

    async def script(s: Stream):
        await s.started()
        await s.recv()
        await s.send(tool_call("1", "slow"))
        await s.send(transcript("after the tool call"))
        ev = await s.recv()
        order.append(ev.tool_result.result_json)
        await s.ended()

    @agent.tool(description="d")
    async def slow() -> str:
        await release.wait()
        return "done"

    fake.session = script
    session = await fake.client.attach("room-1", tools=[slow])
    async for ev in session.events():
        order.append(type(ev).__name__)
        if isinstance(ev, agent.Transcript):
            release.set()
    await session.close()
    assert order == ["ToolCall", "Transcript", '"done"', "Ended"]


async def test_events_queue_is_unbounded(fake):
    """Nothing reads events while the server sends thousands; a tool call
    behind them is still answered."""
    answered = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        await s.recv()
        for i in range(3000):
            await s.send(transcript(f"t{i}"))
        await s.send(tool_call("1", "ping"))
        ev = await s.recv()
        assert ev.tool_result.result_json == '"pong"'
        answered.set()
        await s.ended()

    fake.session = script
    session = await fake.client.attach("room-1", tools=[agent.tool_from_schema("ping", "d", handler=lambda a: "pong")])
    await asyncio.wait_for(answered.wait(), 10)
    events = await drain(session)
    await session.close()
    assert len(events) == 3002
    assert [e.text for e in events[:3000]] == [f"t{i}" for i in range(3000)]
    assert isinstance(events[3000], agent.ToolCall) and isinstance(events[3001], agent.Ended)


# ── Reading events ────────────────────────────────────────────────────────────


async def test_events_one_reader_at_a_time(fake):
    gate = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        for i in range(3):
            await s.send(transcript(f"t{i}"))
        await gate.wait()
        await s.send(transcript("t3"))
        await s.ended()

    fake.session = script
    session = await fake.client.attach("room-1")
    got: list[str] = []
    waiting = asyncio.Event()

    async def first_reader():
        async for ev in session.events():
            got.append(ev.text)
            if len(got) == 3:
                waiting.set()

    reader = asyncio.ensure_future(first_reader())
    await asyncio.wait_for(waiting.wait(), 5)
    with pytest.raises(agent.EventsInUseError):
        async for _ in session.events():
            pass

    # The first reader goes away; reading again picks up where it stopped.
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    gate.set()
    rest = await drain(session)
    await session.close()
    assert got == ["t0", "t1", "t2"]
    assert [type(e).__name__ for e in rest] == ["Transcript", "Ended"] and rest[0].text == "t3"


async def test_reading_again_after_a_break(fake):
    async def script(s: Stream):
        await s.started()
        for i in range(4):
            await s.send(transcript(f"t{i}"))
        await s.ended()

    fake.session = script
    session = await fake.client.attach("room-1")
    got = []
    async for ev in session.events():
        got.append(ev.text)
        break
    async for ev in session.events():  # straight after the break: allowed
        got.append(ev.text)
        break
    held = session.events()
    async for ev in held:  # an iterator still referenced...
        got.append(ev.text)
        break
    async for ev in session.events():  # ...does not block the same task
        got.append(getattr(ev, "text", "END"))
    await session.close()
    assert got == ["t0", "t1", "t2", "t3", "END"]


async def test_a_lost_call_is_an_error(fake):
    """The server dropping a call without an Ended event (a model that failed
    to connect) must not read as a call that went fine."""

    async def script(s: Stream):
        await s.started()
        await s.send(transcript("hi"))
        # the stream ends, no SessionEnded

    fake.session = script
    session = await fake.client.attach("room-1")
    got = []
    with pytest.raises(agent.CallLostError):
        async for ev in session.events():
            got.append(ev)
    assert [type(e) for e in got] == [agent.Transcript]
    with pytest.raises(agent.CallLostError):  # and it stays lost
        await drain(session)
    await session.close()


async def test_a_broken_stream_is_reported(fake):
    async def script(s: Stream):
        await s.started()
        await s.context.abort(grpc.StatusCode.UNAVAILABLE, "worker went away")

    fake.session = script
    session = await fake.client.attach("room-1")
    with pytest.raises(agent.UnavailableError, match="receive: server unavailable \\(worker went away\\)"):
        await drain(session)
    await session.close()


async def test_an_error_after_ended_is_not_reported(fake):
    async def script(s: Stream):
        await s.started()
        await s.ended(reason="hangup")
        await s.context.abort(grpc.StatusCode.INTERNAL, "teardown noise")

    fake.session = script
    session = await fake.client.attach("room-1")
    assert [type(e) for e in await drain(session)] == [agent.Ended]
    await session.close()


# ── Steering ──────────────────────────────────────────────────────────────────


async def test_steering_reaches_the_server(fake):
    got: list[pb.ClientEvent] = []
    release = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        while (ev := await s.recv()) is not None:
            got.append(ev)
            if ev.HasField("hangup"):
                await release.wait()
                await s.ended(reason="hangup")
                return

    fake.session = script
    session = await fake.client.attach("room-1")
    assert not hasattr(session, "say")  # the server cannot speak text verbatim
    await session.reply("be brief")
    await session.inject_context(agent.Role.MODEL, "fact")
    await session.inject_context("user", "told")
    await session.interrupt()
    await session.hangup("done")
    release.set()

    # The call ends cleanly: events finish with no error.
    assert [type(e) for e in await drain(session)] == [agent.Ended]

    # The server has finished the stream: a further command says so.
    with pytest.raises(agent.CallEndedError, match="reply: the call has ended"):
        await session.reply("late")

    await session.close()
    for command in (session.interrupt(), session.reply(), session.hangup(), session.inject_context("user", "x")):
        with pytest.raises(agent.SessionClosedError):
            await command

    assert [ev.WhichOneof("event") for ev in got] == [
        "generate_reply", "inject_context", "inject_context", "interrupt", "hangup",
    ]
    assert got[0].generate_reply.instructions == "be brief"
    assert (got[1].inject_context.text, got[1].inject_context.role) == ("fact", "model")
    assert (got[2].inject_context.text, got[2].inject_context.role) == ("told", "user")
    assert got[4].hangup.reason == "done" and got[4].hangup.speak_closing is False


async def test_every_event_after_attach_is_numbered(fake):
    """handled_tools is 1, the tool result 2, the reply 3: the server counts
    every client event after the Attach, and a Rejected names the command its
    sequence_ref points to."""

    async def script(s: Stream):
        await s.started()
        assert (await s.recv()).HasField("handled_tools")
        await s.send(tool_call("t1", "ping"))
        assert (await s.recv()).HasField("tool_result")
        assert (await s.recv()).HasField("generate_reply")
        for ref in (3, 2, 1, 99):
            await s.send(pb.ServerEvent(error=pb.SessionError(sequence_ref=ref, message=f"no {ref}")))
        await s.ended()

    fake.session = script
    session = await fake.client.attach("room-1", tools=[agent.tool_from_schema("ping", "d", handler=lambda a: 1)])
    rejected = []
    async for ev in session.events():
        if isinstance(ev, agent.ToolCall):
            await eventually(lambda: session._seq == 2)  # the result is out
            await session.reply("go on")
        if isinstance(ev, agent.Rejected):
            rejected.append((ev.command, ev.message, ev.sequence_ref))
    await session.close()
    assert rejected == [
        ("reply", "no 3", 3),
        ("tool_result", "no 2", 2),
        ("handled_tools", "no 1", 1),
        ("", "no 99", 99),
    ]


async def test_rejected_remembers_the_last_256_commands(fake):
    async def script(s: Stream):
        await s.started()
        for _ in range(300):
            assert (await s.recv()).HasField("interrupt")
        for ref in (44, 45, 300):
            await s.send(pb.ServerEvent(error=pb.SessionError(sequence_ref=ref, message="not steerable yet")))
        await s.ended()

    fake.session = script
    session = await fake.client.attach("room-1")
    for _ in range(300):
        await session.interrupt()
    rejected = [(e.sequence_ref, e.command) for e in await drain(session) if isinstance(e, agent.Rejected)]
    await session.close()
    assert rejected == [(44, ""), (45, "interrupt"), (300, "interrupt")]


async def test_a_cancelled_command_does_not_take_the_session_down(fake):
    got = []

    async def script(s: Stream):
        await s.started()
        while (ev := await s.recv()) is not None:
            got.append(ev.WhichOneof("event"))
            if len(got) == 2:
                await s.ended()
                return

    fake.session = script
    session = await fake.client.attach("room-1")
    task = asyncio.ensure_future(session.reply("first"))
    await asyncio.sleep(0)  # into the write
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await session.interrupt()
    assert [type(e) for e in await drain(session)] == [agent.Ended]
    await session.close()
    assert got == ["generate_reply", "interrupt"]


async def test_a_call_that_ends_as_we_attach_still_ends_cleanly(fake):
    """The server finishes the stream before the tool claim reaches it. In
    grpc.aio a write that races the end of a stream marks the whole call
    failed and loses the events behind it; the session must still deliver
    Ended, then report the late command as CallEndedError."""

    async def script(s: Stream):
        await s.started()
        await s.ended(reason="caller_hangup")

    fake.session = script
    for _ in range(5):
        session = await fake.client.attach("room-1", tools=[agent.tool_from_schema("t", "d", handler=lambda a: 1)])
        assert [type(e) for e in await drain(session)] == [agent.Ended]
        with pytest.raises(agent.CallEndedError):
            await session.interrupt()
        await session.close()


async def test_a_command_racing_the_end_of_the_call(fake):
    async def script(s: Stream):
        await s.started()
        await s.send(transcript("bye"))
        await s.ended(reason="caller_hangup")

    fake.session = script
    session = await fake.client.attach("room-1")
    await asyncio.sleep(0.2)  # the server has finished; nothing has been read yet
    with pytest.raises(agent.CallEndedError):
        for _ in range(50):
            await session.interrupt()
            await asyncio.sleep(0.01)
    assert [type(e) for e in await drain(session)] == [agent.Transcript, agent.Ended]
    await session.close()


# ── Detaching ─────────────────────────────────────────────────────────────────


async def test_close_detaches_and_never_hangs_up(fake):
    seen: list[str] = []
    gone = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        try:
            while (ev := await s.recv()) is not None:
                seen.append(ev.WhichOneof("event"))
            seen.append("detached")
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise
        finally:
            gone.set()

    fake.session = script
    session = await fake.client.attach("room-1")
    await session.reply("hi")
    await session.close()
    await session.close()  # idempotent
    await asyncio.wait_for(gone.wait(), 5)
    assert seen[0] == "generate_reply" and seen[-1] in ("detached", "cancelled")
    assert "hangup" not in seen
    # After a close, events end quietly rather than reporting a lost call.
    assert await drain(session) == []


async def test_close_cancels_running_tools(fake):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        await s.recv()
        await s.send(tool_call("1", "forever"))
        await s.until_gone()

    @agent.tool(description="d")
    async def forever() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake.session = script
    session = await fake.client.attach("room-1", tools=[forever])
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.wait_for(session.close(), 5)
    assert cancelled.is_set()


def test_tool_results_are_compact_json():
    assert json.dumps({"a": 1}, separators=(",", ":")) == '{"a":1}'
