"""Events: what happens on a call, as frozen dataclasses to ``match`` on.

==============  ==============================================================
Transcript      something was said
TurnStarted     someone started speaking
TurnEnded       someone finished, with what they said and what it cost
StateChanged    the agent started listening, thinking or speaking
ToolCall        the model called a tool
ToolOutcome     the server answered one of its own tools
Usage           running token counts
Rejected        a command this client sent could not be applied
Ended           the call is over; no events follow
==============  ==============================================================

Every event has ``elapsed``: how far into the call it happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ulai.agent.v1 import agent_pb2 as pb

from ._models import AgentState, Speaker

__all__ = [
    "Ended",
    "Event",
    "ModelUsage",
    "PipelineMetrics",
    "Rejected",
    "StateChanged",
    "ToolCall",
    "ToolOutcome",
    "Transcript",
    "TurnEnded",
    "TurnStarted",
    "TurnTokens",
    "Usage",
    "UsageIO",
]

_ZERO = timedelta(0)


class Event:
    """The base of every event."""

    __slots__ = ()

    elapsed: timedelta


@dataclass(frozen=True)
class Transcript(Event):
    """Something said on the call. Partial transcripts arrive while someone is
    still speaking; the final one has ``final`` set."""

    speaker: Speaker | None = None
    text: str = ""
    final: bool = False
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class TurnStarted(Event):
    """Someone starting to speak."""

    speaker: Speaker | None = None
    turn_id: str = ""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class TurnTokens:
    """One turn's token usage, as the model reported it."""

    prompt_total: int = 0
    prompt_cached: int = 0
    prompt: dict[str, int] = field(default_factory=dict)
    """By modality: "audio", "text"."""
    response_total: int = 0
    response: dict[str, int] = field(default_factory=dict)
    total: int = 0
    reasoning: int = 0
    tool_prompt: int = 0


@dataclass(frozen=True)
class ToolCall(Event):
    """The model calling a tool. A tool with a handler is answered for you; the
    event is still delivered so you can see it."""

    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    """The arguments, decoded. Empty when there were none, or they were not a
    JSON object (``args_json`` has them as sent)."""
    args_json: str = ""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class TurnEnded(Event):
    """Someone finishing a turn."""

    speaker: Speaker | None = None
    turn_id: str = ""
    text: str = ""
    """What the caller heard."""
    duration: timedelta = _ZERO
    interrupted: bool = False
    heard: timedelta = _ZERO
    """How much of an agent turn the caller actually heard, which is less than
    ``duration`` when they interrupted it."""
    generated: str = ""
    """What the model wrote for the turn."""
    audio_bytes: int = 0
    tts_chars: int = 0
    """How many characters were sent to text-to-speech."""
    tool_calls: tuple[ToolCall, ...] = ()
    tokens: TurnTokens | None = None
    """The model's usage report for the turn, when it gave one."""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class StateChanged(Event):
    """The agent starting to listen, think or speak."""

    state: AgentState | None = None
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class ToolOutcome(Event):
    """The server answering one of its own tools, such as end_call."""

    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    args_json: str = ""
    result: Any = None
    """The result, decoded; None when there was none."""
    result_json: str = ""
    error: str = ""
    error_code: str = ""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class UsageIO:
    """Input, cached input and output, in ``ModelUsage.unit``."""

    input: int = 0
    cached: int = 0
    output: int = 0


@dataclass(frozen=True)
class ModelUsage:
    """One model's share of a call's usage."""

    role: str = ""
    provider: str = ""
    model: str = ""
    unit: str = ""
    audio: UsageIO = UsageIO()
    text: UsageIO = UsageIO()


@dataclass(frozen=True)
class Usage(Event):
    """The call's running token counts."""

    prompt_total: int = 0
    prompt_cached: int = 0
    response_total: int = 0
    prompt: dict[str, int] = field(default_factory=dict)
    response: dict[str, int] = field(default_factory=dict)
    models: tuple[ModelUsage, ...] = ()
    """Usage per model: one for a realtime call, one per stage for a cascade,
    plus answering-machine detection when it ran."""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class Rejected(Event):
    """The server refusing a command this client sent. The call goes on."""

    command: str = ""
    """The command refused: "reply", "inject_context", "interrupt", "hangup",
    "tool_result" or "handled_tools". Empty when it cannot be told."""
    message: str = ""
    sequence_ref: int = 0
    """The refused command's number, counting from 1 after the attach."""
    elapsed: timedelta = _ZERO


@dataclass(frozen=True)
class PipelineMetrics:
    """A call's latency and backpressure."""

    ttfb_avg: timedelta = _ZERO
    ttfb_p50: timedelta = _ZERO
    ttfb_p95: timedelta = _ZERO
    ttfb_p99: timedelta = _ZERO
    ttfb_min: timedelta = _ZERO
    ttfb_max: timedelta = _ZERO
    user_turns: int = 0
    agent_turns: int = 0
    interrupted_turns: int = 0
    user_speaking: timedelta = _ZERO
    agent_speaking: timedelta = _ZERO
    inbound_frames: int = 0
    inbound_avg: timedelta = _ZERO
    inbound_max: timedelta = _ZERO
    outbound_chunks: int = 0
    outbound_avg: timedelta = _ZERO
    outbound_max: timedelta = _ZERO
    dropped_by_stage: dict[str, int] = field(default_factory=dict)
    """Frames dropped at each stage: input_filter, to_model, from_model,
    model_write, provider_send, provider_receive, recording."""


@dataclass(frozen=True)
class Ended(Event):
    """The call ending. It is the last event."""

    reason: str = ""
    """Which side ended the call."""
    disposition: str = ""
    duration: timedelta = _ZERO
    heard: timedelta = _ZERO
    """How much agent audio the caller heard."""
    end_reason: str = ""
    """Why, in the server's words: ended, terminated, max_duration,
    no_response, voicemail, CALL_FAILED."""
    pipeline: PipelineMetrics | None = None
    """The call's latency figures, when the server kept them."""
    elapsed: timedelta = _ZERO


_SPEAKERS = {pb.SPEAKER_CALLER: Speaker.CALLER, pb.SPEAKER_AGENT: Speaker.AGENT}
_STATES = {
    pb.AGENT_STATE_LISTENING: AgentState.LISTENING,
    pb.AGENT_STATE_THINKING: AgentState.THINKING,
    pb.AGENT_STATE_SPEAKING: AgentState.SPEAKING,
}


def ms(v: float) -> timedelta:
    return timedelta(milliseconds=v)


def _us(v: float) -> timedelta:
    return timedelta(microseconds=v)


def _object(raw: str) -> dict[str, Any]:
    try:
        v = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


def _json(raw: str) -> Any:
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None


def _modalities(ms_: Any) -> dict[str, int]:
    return {m.modality: m.tokens for m in ms_}


def _usage_io(u: pb.UsageIO) -> UsageIO:
    return UsageIO(input=u.input, cached=u.cached, output=u.output)


def _tool_call(c: pb.ToolCall, at: timedelta) -> ToolCall:
    return ToolCall(id=c.id, name=c.name, args=_object(c.args_json), args_json=c.args_json, elapsed=at)


def event_from(ev: pb.ServerEvent) -> Event | None:
    """A server event as the SDK type a handler matches on, or None for one
    this SDK does not deliver."""
    at = ms(ev.elapsed_ms)
    kind = ev.WhichOneof("event")

    if kind == "transcript":
        t = ev.transcript
        return Transcript(speaker=_SPEAKERS.get(t.speaker), text=t.text, final=t.final, elapsed=at)

    if kind == "turn_started":
        t = ev.turn_started
        return TurnStarted(speaker=_SPEAKERS.get(t.speaker), turn_id=t.turn_id, elapsed=at)

    if kind == "turn_ended":
        t = ev.turn_ended
        tokens = None
        if t.HasField("tokens"):
            k = t.tokens
            tokens = TurnTokens(
                prompt_total=k.prompt_total,
                prompt_cached=k.prompt_cached,
                prompt=_modalities(k.prompt),
                response_total=k.response_total,
                response=_modalities(k.response),
                total=k.total,
                reasoning=k.reasoning,
                tool_prompt=k.tool_prompt,
            )
        return TurnEnded(
            speaker=_SPEAKERS.get(t.speaker),
            turn_id=t.turn_id,
            text=t.transcript,
            duration=ms(t.duration_ms),
            interrupted=t.interrupted,
            heard=ms(t.heard_ms),
            generated=t.generated,
            audio_bytes=t.audio_bytes,
            tts_chars=t.tts_characters,
            tool_calls=tuple(_tool_call(c, at) for c in t.tool_calls),
            tokens=tokens,
            elapsed=at,
        )

    if kind == "agent_state":
        return StateChanged(state=_STATES.get(ev.agent_state.state), elapsed=at)

    if kind == "tool_call":
        return _tool_call(ev.tool_call, at)

    if kind == "tool_outcome":
        o = ev.tool_outcome
        return ToolOutcome(
            id=o.id,
            name=o.name,
            args=_object(o.args_json),
            args_json=o.args_json,
            result=_json(o.result_json),
            result_json=o.result_json,
            error=o.error,
            error_code=o.error_code,
            elapsed=at,
        )

    if kind == "usage":
        u = ev.usage
        return Usage(
            prompt_total=u.prompt_total,
            prompt_cached=u.prompt_cached,
            response_total=u.response_total,
            prompt=_modalities(u.prompt),
            response=_modalities(u.response),
            models=tuple(
                ModelUsage(
                    role=m.role,
                    provider=m.provider,
                    model=m.model,
                    unit=m.unit,
                    audio=_usage_io(m.audio),
                    text=_usage_io(m.text),
                )
                for m in u.models
            ),
            elapsed=at,
        )

    if kind == "error":
        return Rejected(message=ev.error.message, sequence_ref=ev.error.sequence_ref, elapsed=at)

    if kind == "session_ended":
        s = ev.session_ended
        return Ended(
            reason=s.reason,
            disposition=s.disposition,
            duration=ms(s.duration_ms),
            heard=ms(s.played_ms),
            end_reason=s.end_reason,
            pipeline=_pipeline(s.pipeline_metrics) if s.HasField("pipeline_metrics") else None,
            elapsed=at,
        )

    return None


def _pipeline(m: pb.PipelineMetrics) -> PipelineMetrics:
    drops: dict[str, int] = {}
    if m.HasField("drops"):
        d = m.drops
        drops = {
            "input_filter": d.input_filter,
            "to_model": d.to_model,
            "from_model": d.from_model,
            "model_write": d.model_write,
            "provider_send": d.provider_send,
            "provider_receive": d.provider_receive,
            "recording": d.recording,
        }
    return PipelineMetrics(
        ttfb_avg=ms(m.ttfb_avg_ms),
        ttfb_p50=ms(m.ttfb_p50_ms),
        ttfb_p95=ms(m.ttfb_p95_ms),
        ttfb_p99=ms(m.ttfb_p99_ms),
        ttfb_min=ms(m.ttfb_min_ms),
        ttfb_max=ms(m.ttfb_max_ms),
        user_turns=m.user_turns,
        agent_turns=m.agent_turns,
        interrupted_turns=m.interrupted_turns,
        user_speaking=ms(m.user_speaking_ms),
        agent_speaking=ms(m.agent_speaking_ms),
        inbound_frames=m.inbound_frames_total,
        inbound_avg=_us(m.inbound_avg_us),
        inbound_max=_us(m.inbound_max_us),
        outbound_chunks=m.outbound_chunks_total,
        outbound_avg=_us(m.outbound_avg_us),
        outbound_max=_us(m.outbound_max_us),
        dropped_by_stage=drops,
    )
