"""Events: each server event arrives as the SDK type a handler matches on,
with its fields intact. A field mapped to the wrong one reads as a plausible
number, so each is checked by value."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest
from fakeserver import Stream
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent
from ulai_agents._events import event_from


def ms(v: float) -> timedelta:
    return timedelta(milliseconds=v)


EVERY_KIND = [
    pb.ServerEvent(elapsed_ms=1200, transcript=pb.Transcript(speaker=pb.SPEAKER_AGENT, text="hi", final=True)),
    pb.ServerEvent(elapsed_ms=1300, turn_started=pb.TurnStarted(speaker=pb.SPEAKER_CALLER, turn_id="t1")),
    pb.ServerEvent(elapsed_ms=1400, turn_ended=pb.TurnEnded(
        speaker=pb.SPEAKER_AGENT, turn_id="t2", duration_ms=1500, transcript="Let me", interrupted=True,
        heard_ms=800, generated="Let me check", audio_bytes=9600, tts_characters=12,
        tool_calls=[pb.ToolCall(id="c1", name="check_balance", args_json='{"a":1}')],
        tokens=pb.TurnTokens(prompt_total=100, prompt_cached=40, response_total=20, total=130, reasoning=7,
                             tool_prompt=3, prompt=[pb.ModalityTokens(modality="audio", tokens=60)],
                             response=[pb.ModalityTokens(modality="text", tokens=20)]),
    )),
    pb.ServerEvent(elapsed_ms=1500, agent_state=pb.AgentStateChanged(state=pb.AGENT_STATE_THINKING)),
    pb.ServerEvent(elapsed_ms=1600, tool_call=pb.ToolCall(id="c2", name="lookup", args_json='{"q":"x"}')),
    pb.ServerEvent(elapsed_ms=1700, tool_outcome=pb.ToolOutcome(
        id="o1", name="end_call", args_json="{}", result_json='{"ok":true}', error="e", error_code="X")),
    pb.ServerEvent(elapsed_ms=1800, usage=pb.UsageReport(
        prompt_total=500, prompt_cached=100, response_total=50,
        prompt=[pb.ModalityTokens(modality="text", tokens=500)],
        response=[pb.ModalityTokens(modality="audio", tokens=50)],
        models=[pb.ModelUsage(role="realtime", provider="p", model="m", unit="tokens",
                              audio=pb.UsageIO(input=1, cached=2, output=3), text=pb.UsageIO(input=4))],
    )),
    pb.ServerEvent(elapsed_ms=1900, error=pb.SessionError(sequence_ref=3, message="no")),
    pb.ServerEvent(elapsed_ms=60000, session_ended=pb.SessionEnded(
        reason="agent_ended", disposition="COMPLETED", duration_ms=60000, played_ms=30000, end_reason="max_duration",
        pipeline_metrics=pb.PipelineMetrics(
            ttfb_avg_ms=700.25, ttfb_p50_ms=650, ttfb_p95_ms=812.5, ttfb_p99_ms=900, ttfb_min_ms=300, ttfb_max_ms=950,
            user_turns=6, agent_turns=7, interrupted_turns=2, user_speaking_ms=12000.5, agent_speaking_ms=20000,
            inbound_frames_total=3000, inbound_avg_us=120.5, inbound_max_us=950, outbound_chunks_total=1500,
            outbound_avg_us=80, outbound_max_us=400,
            drops=pb.PipelineDrops(input_filter=1, to_model=3, from_model=4, model_write=5, provider_send=6,
                                   provider_receive=7, recording=8),
        ),
    )),
]

WANT = [
    agent.Transcript(speaker=agent.Speaker.AGENT, text="hi", final=True, elapsed=ms(1200)),
    agent.TurnStarted(speaker=agent.Speaker.CALLER, turn_id="t1", elapsed=ms(1300)),
    agent.TurnEnded(
        speaker=agent.Speaker.AGENT, turn_id="t2", text="Let me", duration=ms(1500), interrupted=True,
        heard=ms(800), generated="Let me check", audio_bytes=9600, tts_chars=12,
        tool_calls=(agent.ToolCall(id="c1", name="check_balance", args={"a": 1}, args_json='{"a":1}',
                                   elapsed=ms(1400)),),
        tokens=agent.TurnTokens(prompt_total=100, prompt_cached=40, prompt={"audio": 60}, response_total=20,
                                response={"text": 20}, total=130, reasoning=7, tool_prompt=3),
        elapsed=ms(1400),
    ),
    agent.StateChanged(state=agent.AgentState.THINKING, elapsed=ms(1500)),
    agent.ToolCall(id="c2", name="lookup", args={"q": "x"}, args_json='{"q":"x"}', elapsed=ms(1600)),
    agent.ToolOutcome(id="o1", name="end_call", args={}, args_json="{}", result={"ok": True},
                      result_json='{"ok":true}', error="e", error_code="X", elapsed=ms(1700)),
    agent.Usage(
        prompt_total=500, prompt_cached=100, response_total=50, prompt={"text": 500}, response={"audio": 50},
        models=(agent.ModelUsage(role="realtime", provider="p", model="m", unit="tokens",
                                 audio=agent.UsageIO(1, 2, 3), text=agent.UsageIO(input=4)),),
        elapsed=ms(1800),
    ),
    agent.Rejected(command="", message="no", sequence_ref=3, elapsed=ms(1900)),
    agent.Ended(
        reason="agent_ended", disposition="COMPLETED", duration=timedelta(minutes=1), heard=timedelta(seconds=30),
        end_reason="max_duration",
        pipeline=agent.PipelineMetrics(
            ttfb_avg=timedelta(microseconds=700250), ttfb_p50=ms(650), ttfb_p95=timedelta(microseconds=812500),
            ttfb_p99=ms(900), ttfb_min=ms(300), ttfb_max=ms(950), user_turns=6, agent_turns=7, interrupted_turns=2,
            user_speaking=timedelta(microseconds=12000500), agent_speaking=ms(20000), inbound_frames=3000,
            inbound_avg=timedelta(microseconds=120.5), inbound_max=timedelta(microseconds=950),
            outbound_chunks=1500, outbound_avg=timedelta(microseconds=80), outbound_max=timedelta(microseconds=400),
            dropped_by_stage={"input_filter": 1, "to_model": 3, "from_model": 4, "model_write": 5,
                              "provider_send": 6, "provider_receive": 7, "recording": 8},
        ),
        elapsed=timedelta(minutes=1),
    ),
]


@pytest.mark.parametrize("msg, want", list(zip(EVERY_KIND, WANT, strict=True)), ids=lambda v: type(v).__name__)
def test_every_kind_maps_by_value(msg, want):
    got = event_from(msg)
    assert got == want
    assert isinstance(got, agent.Event)
    assert dataclasses.is_dataclass(got)
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.elapsed = timedelta(0)


def test_defaults_when_the_server_sends_less():
    # A call with no figures sends none, not zeros.
    e = event_from(pb.ServerEvent(session_ended=pb.SessionEnded()))
    assert e == agent.Ended() and e.pipeline is None
    # Figures without drops: an empty dict.
    e = event_from(pb.ServerEvent(session_ended=pb.SessionEnded(pipeline_metrics=pb.PipelineMetrics(agent_turns=1))))
    assert e.pipeline.agent_turns == 1 and e.pipeline.dropped_by_stage == {}
    # A turn without a token report, or a heard figure.
    e = event_from(pb.ServerEvent(turn_ended=pb.TurnEnded(turn_id="t")))
    assert e.tokens is None and e.heard == timedelta(0) and e.speaker is None and e.tool_calls == ()
    # Unknown speaker and state values.
    assert event_from(pb.ServerEvent(transcript=pb.Transcript())).speaker is None
    assert event_from(pb.ServerEvent(agent_state=pb.AgentStateChanged())).state is None
    # Tool arguments that are not an object, and a result that is not JSON.
    e = event_from(pb.ServerEvent(tool_call=pb.ToolCall(args_json="[1]")))
    assert e.args == {} and e.args_json == "[1]"
    e = event_from(pb.ServerEvent(tool_outcome=pb.ToolOutcome(result_json="nope")))
    assert e.result is None and e.result_json == "nope"
    # Events the SDK does not deliver.
    assert event_from(pb.ServerEvent(session_started=pb.SessionStarted())) is None
    assert event_from(pb.ServerEvent()) is None


def test_events_work_with_match():
    seen = []
    for ev in WANT:
        match ev:
            case agent.Transcript(final=True, speaker=agent.Speaker.AGENT, text=text):
                seen.append(f"said {text}")
            case agent.TurnEnded(interrupted=True, heard=heard):
                seen.append(f"cut at {heard.total_seconds()}")
            case agent.ToolCall(name=name, args={"q": q}):
                seen.append(f"{name}({q})")
            case agent.Rejected(command=command, message=message):
                seen.append(f"rejected {command!r}: {message}")
            case agent.Ended(pipeline=agent.PipelineMetrics(agent_turns=n)):
                seen.append(f"ended after {n} turns")
            case agent.Event(elapsed=elapsed):
                seen.append(f"{type(ev).__name__} at {elapsed.total_seconds()}")
    assert seen == [
        "said hi",
        "TurnStarted at 1.3",
        "cut at 0.8",
        "StateChanged at 1.5",
        "lookup(x)",
        "ToolOutcome at 1.7",
        "Usage at 1.8",
        "rejected '': no",
        "ended after 7 turns",
    ]


async def test_every_kind_arrives_over_the_wire(fake):
    async def script(s: Stream):
        await s.started()
        for ev in EVERY_KIND:
            await s.send(ev)

    fake.session = script
    session = await fake.client.attach("room-1")
    got = [ev async for ev in session.events()]
    await session.close()
    assert got == WANT
