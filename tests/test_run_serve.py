"""run (outbound) and serve (inbound)."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import timedelta

import grpc
import pytest
from fakeserver import ROOM, Stream, eventually, support
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent
from ulai_agents import _client

needs_cancelling = pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.cancelling is Python 3.11+")


async def wait_for_detach(s: Stream) -> None:
    """Hold the call open until the client goes."""
    try:
        await s.until_gone()
    except asyncio.CancelledError:
        pass


# ── 5. run ────────────────────────────────────────────────────────────────────


async def test_run_joins_answers_tools_and_leaves(fake):
    seen: dict = {}

    async def script(s: Stream):
        attach = await s.started(
            session_id="room-1",
            call=pb.CallInfo(call_id="c-9", agent_id="support", custom_fields_json='{"call_id":"c-9"}'),
            backend=pb.BackendInfo(provider="test", pipeline=pb.PIPELINE_REALTIME),
        )
        seen["attach"] = attach.attach
        seen["handled"] = list((await s.recv()).handled_tools.names)
        await s.send(pb.ServerEvent(
            tool_call=pb.ToolCall(id="t1", name="check_balance", args_json='{"account":"a1"}'),
        ))
        seen["result"] = (await s.recv()).tool_result.result_json
        await s.send(pb.ServerEvent(transcript=pb.Transcript(speaker=pb.SPEAKER_CALLER, text="hello", final=True)))
        seen["said"] = (await s.recv()).generate_reply.instructions
        await s.send(pb.ServerEvent(
            elapsed_ms=3000, session_ended=pb.SessionEnded(reason="caller_hangup", played_ms=900),
        ))

    fake.session = script
    ended: list[agent.Ended] = []

    async def handle(session: agent.Session) -> None:
        assert session.call.fields["call_id"] == "c-9"
        assert session.backend.pipeline == agent.PipelineKind.REALTIME
        async for event in session.events():
            match event:
                case agent.Transcript(final=True, text=text):
                    await session.reply("greet " + text)
                case agent.Ended():
                    ended.append(event)

    await fake.client.run(ROOM, support(), handle, participant_id="agent-p", fields={"call_id": "c-9"})

    assert seen["attach"].session_id == "room-1" and seen["attach"].replay_history is False
    assert seen["handled"] == ["check_balance"]
    assert seen["result"] == '{"account":"a1","balance":250}'
    assert seen["said"] == "greet hello"
    assert ended == [
        agent.Ended(reason="caller_hangup", heard=timedelta(milliseconds=900), elapsed=timedelta(seconds=3)),
    ]
    assert len(fake.joins) == 1
    assert (fake.joins[0].participant_id, fake.joins[0].custom_fields_json) == ("agent-p", '{"call_id":"c-9"}')
    assert len(fake.leaves) == 1
    leave = fake.leaves[0]
    assert (leave.room_id, leave.participant_id, leave.reason) == ("room-1", "p-1", "done")


async def test_run_without_a_handler_waits_until_the_call_ends(fake):
    async def script(s: Stream):
        await s.started()
        await s.recv()  # the tool claim
        await asyncio.sleep(0.1)
        await s.ended(reason="caller_hangup")

    fake.session = script
    await asyncio.wait_for(fake.client.run(ROOM, support()), 5)
    assert len(fake.leaves) == 1


async def test_run_takes_a_sync_handler(fake):
    calls = []

    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    fake.session = script
    await fake.client.run(ROOM, support(), lambda session: calls.append(session.id))
    assert calls == ["room-1"] and len(fake.leaves) == 1


async def test_a_lost_call_is_an_error_and_still_leaves(fake):
    async def script(s: Stream):
        await s.started()
        # the stream ends, no SessionEnded: a model that failed to connect

    fake.session = script
    with pytest.raises(agent.CallLostError):
        await fake.client.run(ROOM, support())
    assert len(fake.leaves) == 1


async def test_run_leaves_when_the_handler_fails(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    async def handle(session: agent.Session) -> None:
        raise LookupError("no such customer")

    fake.session = script
    with pytest.raises(LookupError, match="no such customer"):
        await fake.client.run(ROOM, support(), handle)
    assert len(fake.leaves) == 1


async def test_run_reports_cancellation_and_leaves(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    async def handle(session: agent.Session) -> None:
        await asyncio.sleep(60)

    fake.session = script
    task = asyncio.ensure_future(fake.client.run(ROOM, support(), handle))
    await eventually(lambda: len(fake.keys) == 2)  # joined and attached
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(agent.CanceledError, match="run: canceled"):
        await task
    assert task.cancelled()  # still cancellation, as far as asyncio is concerned
    assert len(fake.leaves) == 1


@needs_cancelling
async def test_run_reports_cancellation_a_handler_swallowed(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    async def handle(session: agent.Session) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return  # swallowed

    fake.session = script
    task = asyncio.ensure_future(fake.client.run(ROOM, support(), handle))
    await eventually(lambda: len(fake.keys) == 2)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(agent.CanceledError):
        await task
    assert len(fake.leaves) == 1


async def test_run_under_wait_for(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    fake.session = script
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(fake.client.run(ROOM, support()), 0.3)
    assert len(fake.leaves) == 1


# Python 3.11's asyncio.timeout recognises only CancelledError itself, not a
# subclass such as CanceledError; 3.12 and later recognise both.
@pytest.mark.skipif(sys.version_info < (3, 12), reason="asyncio.timeout accepts CancelledError subclasses from 3.12")
async def test_run_under_asyncio_timeout(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    fake.session = script
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.3):
            await fake.client.run(ROOM, support())
    assert len(fake.leaves) == 1


async def test_run_waits_at_most_a_while_to_leave(fake, monkeypatch):
    assert _client.LEAVE_TIMEOUT == 10
    monkeypatch.setattr(_client, "LEAVE_TIMEOUT", 0.2)

    async def script(s: Stream):
        await s.started()
        await s.ended()

    fake.session = script
    fake.leave_delay = 5
    loop = asyncio.get_running_loop()
    start = loop.time()
    await fake.client.run(ROOM, support())
    assert loop.time() - start < 2
    assert len(fake.leaves) == 1


async def test_a_failed_leave_does_not_fail_the_run(fake, caplog):
    async def script(s: Stream):
        await s.started()
        await s.ended()

    fake.session = script
    fake.leave_error = (grpc.StatusCode.UNAVAILABLE, "control plane down")
    with caplog.at_level(logging.WARNING, logger="ulai_agents"):
        await fake.client.run(ROOM, support())
    assert "control plane down" in caplog.text


async def test_run_does_not_attach_when_join_fails(fake):
    fake.join_error = (grpc.StatusCode.NOT_FOUND, "no such room")
    with pytest.raises(agent.RoomNotFoundError, match="no such room"):
        await fake.client.run(ROOM, support())
    assert [k[0] for k in fake.keys] == ["Join"] and fake.leaves == []


# ── 6. serve ──────────────────────────────────────────────────────────────────


async def test_serve_attaches_to_each_job(fake):
    attaches: list[pb.Attach] = []

    async def script(s: Stream):
        first = await s.recv()
        attaches.append(first.attach)
        await s.send(pb.ServerEvent(session_started=pb.SessionStarted(
            session_id=first.attach.session_id, call=pb.CallInfo(agent_id="support", from_number="+91"),
        )))
        await s.ended(reason="done")

    fake.session = script
    calls: asyncio.Queue[agent.CallInfo] = asyncio.Queue()

    async def handle(session: agent.Session) -> None:
        await calls.put(session.call)
        async for _ in session.events():
            pass

    a = support()
    serving = asyncio.ensure_future(fake.client.serve(a.id, a.tools, handle, name="test"))
    await fake.jobs.put(pb.Job(session_id="s-1", call=pb.CallInfo(agent_id="support"), attach_deadline_ms=5000))
    call = await asyncio.wait_for(calls.get(), 5)
    assert call.from_ == "+91"

    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(serving, 5)
    assert len(fake.subs) == 1
    assert list(fake.subs[0].agent_ids) == ["support"] and fake.subs[0].client_name == "test"
    # No history: a replay would answer the call's tool calls a second time.
    assert attaches[0].session_id == "s-1" and attaches[0].replay_history is False


async def test_serve_runs_each_call_on_its_own_task(fake):
    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)

    fake.session = script
    started: list[str] = []
    release = asyncio.Event()

    async def handle(session: agent.Session) -> None:
        started.append(session.id)
        await release.wait()

    serving = asyncio.ensure_future(fake.client.serve("support", [], handle))
    await fake.jobs.put(pb.Job(session_id="a"))
    await fake.jobs.put(pb.Job(session_id="b"))
    await eventually(lambda: len(started) == 2)  # both running at once
    release.set()
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving


async def test_serve_survives_failing_handlers(fake):
    async def script(s: Stream):
        first = await s.recv()
        if first.attach.session_id == "gone":
            await s.context.abort(grpc.StatusCode.NOT_FOUND, 'no live session "gone"')
        await s.send(pb.ServerEvent(session_started=pb.SessionStarted(session_id=first.attach.session_id)))
        await wait_for_detach(s)

    fake.session = script
    failures: list[tuple[str | None, BaseException]] = []
    served: list[str] = []

    async def handle(session: agent.Session) -> None:
        served.append(session.id)
        if session.id != "fine":
            raise RuntimeError(f"handler bug on {session.id}")

    async def on_error(job: agent.Job | None, err: BaseException) -> None:
        failures.append((job.session_id if job else None, err))

    serving = asyncio.ensure_future(fake.client.serve("support", [], handle, on_error=on_error))
    for sid in ("a", "gone", "b", "fine"):
        await fake.jobs.put(pb.Job(session_id=sid))
    await eventually(lambda: len(served) == 3 and len(failures) == 3)
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    by_job = {sid: err for sid, err in failures}
    assert set(by_job) == {"a", "b", "gone"}
    assert isinstance(by_job["a"], RuntimeError) and "handler bug on a" in str(by_job["a"])
    assert isinstance(by_job["gone"], agent.NotFoundError)
    assert sorted(served) == ["a", "b", "fine"]


async def test_serve_logs_failures_by_default(fake, caplog):
    async def script(s: Stream):
        await s.started(session_id="s-9")
        await wait_for_detach(s)

    async def handle(session: agent.Session) -> None:
        raise RuntimeError("handler bug")

    fake.session = script
    serving = asyncio.ensure_future(fake.client.serve("support", [], handle))
    with caplog.at_level(logging.ERROR, logger="ulai_agents"):
        await fake.jobs.put(pb.Job(session_id="s-9"))
        await eventually(lambda: "handler bug" in caplog.text)
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    assert "call s-9: handler bug" in caplog.text


async def test_serve_reconnects_with_backoff(fake, monkeypatch):
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(_client, "_sleep", sleep)
    unavailable = (grpc.StatusCode.UNAVAILABLE, "restarting")
    fake.subscribe_errors = [unavailable] * 7
    fake.session = None
    lost: list[tuple[agent.Job | None, BaseException]] = []

    serving = asyncio.ensure_future(fake.client.serve(
        "support", [], lambda s: None, on_error=lambda job, err: lost.append((job, err)),
    ))
    await eventually(lambda: len(fake.subs) == 8)  # seven failures, then connected
    assert sleeps == [1, 2, 4, 8, 16, 30, 30]
    assert all(job is None and isinstance(err, agent.UnavailableError) for job, err in lost)
    assert [str(err).rsplit("; ", 1)[1] for _, err in lost] == [
        "reconnecting in 1s", "reconnecting in 2s", "reconnecting in 4s", "reconnecting in 8s",
        "reconnecting in 16s", "reconnecting in 30s", "reconnecting in 30s",
    ]
    assert "restarting" in str(lost[0][1]) and lost[0][1].code == grpc.StatusCode.UNAVAILABLE

    # Once connected, a drop reconnects at once, and the backoff starts over.
    await fake.jobs.put(pb.Job(session_id="x"))  # a job: connected (no session script: reported, not fatal)
    await eventually(lambda: len(lost) == 8)  # the attach failure
    fake.subscribe_errors = [unavailable]
    await fake.jobs.put(unavailable)
    await eventually(lambda: len(fake.subs) == 10)
    assert sleeps == [1, 2, 4, 8, 16, 30, 30, 1]
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving


async def test_serve_ends_on_other_errors(fake):
    fake.subscribe_errors = [(grpc.StatusCode.UNAUTHENTICATED, "invalid api key")]
    reported = []
    with pytest.raises(agent.UnauthorizedError, match="invalid api key"):
        await asyncio.wait_for(fake.client.serve("support", on_error=lambda j, e: reported.append(e)), 5)
    assert reported == []


async def test_cancelling_serve_waits_for_running_handlers(fake):
    detached = asyncio.Event()

    async def script(s: Stream):
        await s.started()
        await wait_for_detach(s)
        detached.set()

    fake.session = script
    in_call = asyncio.Event()
    cleaned_up = []

    async def handle(session: agent.Session) -> None:
        in_call.set()
        try:
            async for _ in session.events():
                pass
        finally:
            await asyncio.sleep(0.1)  # cleanup that takes a moment
            cleaned_up.append(session.id)

    serving = asyncio.ensure_future(fake.client.serve("support", [], handle))
    await fake.jobs.put(pb.Job(session_id="s-1"))
    await asyncio.wait_for(in_call.wait(), 5)
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    assert cleaned_up == ["room-1"]  # finished before serve returned
    await asyncio.wait_for(detached.wait(), 5)  # and the session detached


async def test_serve_and_jobs_need_an_agent(fake):
    with pytest.raises(agent.ValidationError, match="at least one agent id"):
        fake.client.jobs([])
    with pytest.raises(agent.ValidationError, match="at least one agent id"):
        fake.client.jobs(["support", ""])
    with pytest.raises(agent.ValidationError, match="serve needs an agent id"):
        await fake.client.serve("")
    dup = agent.tool_from_schema("t", "d", handler=lambda a: None)
    with pytest.raises(agent.ValidationError, match="declared twice"):
        await fake.client.serve("support", [dup, dup])
    assert fake.subs == []


async def test_jobs(fake):
    await fake.jobs.put(pb.Job(session_id="s-1", call=pb.CallInfo(agent_id="support", from_number="p-1",
                                                                   to_number="room-1", direction="room"),
                               attach_deadline_ms=5000))
    await fake.jobs.put(None)  # the server ends the subscription
    jobs = [job async for job in fake.client.jobs("support", name="dispatcher")]
    assert jobs == [agent.Job(
        session_id="s-1",
        call=agent.CallInfo(agent_id="support", from_="p-1", to="room-1", direction="room"),
        attach_by=timedelta(seconds=5),
    )]
    assert list(fake.subs[0].agent_ids) == ["support"] and fake.subs[0].client_name == "dispatcher"
