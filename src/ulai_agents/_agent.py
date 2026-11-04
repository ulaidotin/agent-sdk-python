"""The agent: who it is, how it sounds, how it hears and speaks, what it can do."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ulai.agent.v1 import agent_pb2 as pb

from ._errors import ValidationError
from ._models import ClassifierBackend, Language, Opening, Voice, check_listed, wire
from ._pipeline import PIPELINES, Pipeline, UlaiLive, check_duration, check_range
from ._tools import Tool

__all__ = ["Agent", "CloudAccount", "Options", "Room"]

_SECOND = timedelta(seconds=1)
_MS = timedelta(milliseconds=1)


def _redacted(secret: str) -> str:
    return "[redacted]" if secret else "''"


@dataclass(frozen=True)
class CloudAccount:
    """The Google Cloud project an agent's models run in, and its key.

    It is sent with every call, so connect with TLS anywhere but your own
    machine. Its repr never shows the key.
    """

    project: str
    """The project ID. Required."""
    region: str = ""
    """Where a Realtime model runs. Default: us-central1. A Cascade sets its
    regions per stage instead."""
    credentials_json: str = field(default="", repr=False)
    """A service-account key for the project. Required."""

    @classmethod
    def from_file(cls, project: str, region: str, key_file: str | os.PathLike[str]) -> CloudAccount:
        """Read a service-account key file into a CloudAccount."""
        try:
            with open(key_file, encoding="utf-8") as f:
                raw = f.read()
        except OSError as e:
            raise ValidationError(f"reading the service-account key: {e}") from e
        try:
            key = json.loads(raw)
        except json.JSONDecodeError:
            key = None
        if not isinstance(key, dict) or not key.get("type"):
            raise ValidationError(f"{os.fspath(key_file)} is not a service-account key file")
        return cls(project=project, region=region, credentials_json=raw)

    def __repr__(self) -> str:
        return (
            f"CloudAccount(project={self.project!r}, region={self.region!r}, "
            f"credentials_json={_redacted(self.credentials_json)})"
        )

    __str__ = __repr__


@dataclass(frozen=True)
class Room:
    """A room that already exists, and where it lives.

    The server never creates rooms: an agent alone in a room nobody was told
    about would talk to no one. Its repr never shows the token.
    """

    id: str
    """The room. Required."""
    url: str
    """The control plane that holds the room. Required."""
    token: str = field(default="", repr=False)
    """Authenticates to that control plane."""

    def __repr__(self) -> str:
        return f"Room(id={self.id!r}, url={self.url!r}, token={_redacted(self.token)})"

    __str__ = __repr__


@dataclass(kw_only=True)
class Options:
    """Fine-tuning for a call. Every field is optional: None, zero and empty
    all mean the server's default."""

    interruptible: bool | None = None
    """Let the caller talk over the agent. Default: yes."""
    greeting_interruptible: bool | None = None
    """Let the caller talk over the greeting. Default: yes."""
    first_words_timeout: timedelta | None = None
    """How long to wait for the caller's first words."""
    backchannel_gate: bool | None = None
    """Keep short acknowledgements ("hmm", "okay") from interrupting the agent.
    Default: on."""
    silence_monitor: bool | None = None
    """Nudge a caller who goes quiet, and hang up after a long silence.
    Default: decided by the pipeline."""
    end_call_tool: bool | None = None
    """Let the model hang up by itself with the server's end_call tool.
    Default: on. Declaring your own end_call tool replaces the server's."""
    voicemail_detection: bool | None = None
    """End outbound calls that reach a voicemail greeting."""
    replay_lost_utterances: bool | None = None
    """Re-send what the caller said while the model was reconnecting."""
    background_noise: bool = False
    """Mix an ambient bed under the agent's voice."""
    live_transcription: bool = False
    """Publish the conversation into the room as captions while it happens."""
    context_compression: bool = False
    """Let a long call drop its oldest turns instead of running out of context."""
    minimal_audio_processing: bool | None = None
    """Turn off all caller-audio processing except the denoiser. A diagnostic."""
    classifier: ClassifierBackend | str | None = None
    """What judges who answered a dialled call."""
    workspace_id: str = ""
    """The account the call belongs to, passed through to the server untouched."""


@dataclass(kw_only=True)
class Agent:
    """Everything a voice agent is. The server keeps no agents of its own, so
    this travels with every call. None (or empty) in any field is the server's
    default."""

    id: str = ""
    """Names the agent. Required."""
    instructions: str = ""
    """Who the agent is and how it behaves: the system prompt. Required."""
    greeting: str = ""
    """The agent's opening line."""
    closing: str = ""
    """The agent's last line."""
    persona: str = ""
    """Extra character, given to the model silently after the greeting rather
    than in the instructions, which keeps the first turn short."""
    language: Language | str | None = None
    """The language a Cascade listens and speaks in. Default: en-IN. A Realtime
    model takes its language from the conversation: say which in instructions."""
    voice: Voice | str | None = None
    """The agent's voice, for Realtime and a Cascade's speech. UlaiLive has its
    own: UlaiLive.voice."""
    pipeline: Pipeline | None = None
    """How the agent hears and speaks: Realtime, Cascade or UlaiLive. None is
    the server's default, Realtime."""
    tools: Sequence[Tool] = field(default_factory=list)
    """What the agent can do. Each is shown to the model and, when it has a
    handler, answered by this client."""
    opening: Opening | str | None = None
    """Who speaks first. Default: the agent."""
    max_duration: timedelta | None = None
    """End the call after this long. None or zero is the server's limit."""
    temperature: float | None = None
    """The model's sampling temperature, 0 to 2."""
    cloud: CloudAccount | None = None
    """The Google Cloud project the models run in. Required for Realtime and
    Cascade: the server holds no model credentials of its own."""
    options: Options = field(default_factory=Options)

    def validate(self, *, allow_unlisted_values: bool = False) -> None:
        """Raise ValidationError listing every problem with this agent.

        join and run call it for you; call it yourself to check an agent at
        startup. allow_unlisted_values skips only the checks against this SDK's
        lists of models, voices and languages.
        """
        problems = self._problems(lists=not allow_unlisted_values)
        if problems:
            raise ValidationError(problems)

    def _problems(self, lists: bool) -> list[str]:
        out: list[str] = []
        if not self.id or not isinstance(self.id, str):
            out.append("Agent.id is required")
        if not self.instructions or not isinstance(self.instructions, str):
            out.append("Agent.instructions is required: the server keeps no agents, so they travel with every call")
        for name in ("greeting", "closing", "persona"):
            if not isinstance(getattr(self, name), str):
                out.append(f"Agent.{name} must be a string")
        if self.cloud is not None and not isinstance(self.cloud, CloudAccount):
            out.append(f"Agent.cloud must be an agent.CloudAccount, not {type(self.cloud).__name__}")
        elif self._needs_cloud() and (
            self.cloud is None or not self.cloud.project or not self.cloud.credentials_json
        ):
            out.append(
                "Agent.cloud needs a project and credentials_json: the server holds no model "
                "credentials of its own (load a key with agent.CloudAccount.from_file)"
            )
        check_listed(out, "Agent.language", self.language, Language, lists, "use one of the agent.Language members")
        check_listed(out, "Agent.voice", self.voice, Voice, lists, "use one of the agent.Voice members")
        # Not a list check: there are only two sides to a call.
        check_listed(out, "Agent.opening", self.opening, Opening, True,
                     "use agent.Opening.AGENT or agent.Opening.CALLER")
        check_range(out, "Agent.temperature", self.temperature, 0, 2)
        check_duration(out, "Agent.max_duration", self.max_duration, negative_ok=False)
        d = self.max_duration
        if isinstance(d, timedelta) and timedelta(0) < d < _SECOND:
            out.append(f"Agent.max_duration {d} must be zero (the server's limit) or at least a second")
        if self.pipeline is not None:
            if isinstance(self.pipeline, PIPELINES):
                out.extend(self.pipeline._problems(lists))
            else:
                out.append(
                    f"Agent.pipeline is {type(self.pipeline).__name__}: use agent.Realtime, "
                    "agent.Cascade or agent.UlaiLive"
                )
        o = self.options
        if not isinstance(o, Options):
            out.append(f"Agent.options must be an agent.Options, not {type(o).__name__}")
        else:
            check_listed(out, "Options.classifier", o.classifier, ClassifierBackend, lists,
                         "use one of the agent.ClassifierBackend members")
            check_duration(out, "Options.first_words_timeout", o.first_words_timeout, negative_ok=False)
        out.extend(tool_problems(self.tools, "Agent.tools"))
        return out

    def _needs_cloud(self) -> bool:
        """Whether this agent's calls reach Google models, which the server can
        only do with the agent's own credentials."""
        if isinstance(self.pipeline, UlaiLive):
            # Voicemail detection still asks a Google model.
            return isinstance(self.options, Options) and self.options.voicemail_detection is True
        return True

    def _profile(self) -> pb.AgentProfile:
        """This agent on the wire."""
        o = self.options
        p = pb.AgentProfile(
            prompt=self.instructions,
            greeting=self.greeting,
            closing=self.closing,
            persona_text=self.persona,
            voice=wire(self.voice),
            language=wire(self.language),
            max_call_duration_seconds=(self.max_duration // _SECOND) if self.max_duration else 0,
            opening=opening_pb(self.opening),
            initial_engagement_deadline_ms=(o.first_words_timeout // _MS) if o.first_words_timeout else 0,
            background_noise=o.background_noise,
            live_transcription=o.live_transcription,
            context_compression=o.context_compression,
            ai_classify_backend=wire(o.classifier),
            workspace_id=o.workspace_id,
        )
        for name, value in (
            ("temperature", self.temperature),
            ("allow_interruptions", o.interruptible),
            ("greeting_interruptible", o.greeting_interruptible),
            ("backchannel_gate", o.backchannel_gate),
            ("silence_monitor", o.silence_monitor),
            ("end_call_enabled", o.end_call_tool),
            ("amd_enabled", o.voicemail_detection),
            ("lost_utterance_replay", o.replay_lost_utterances),
            ("minimal_audio_processing", o.minimal_audio_processing),
        ):
            if value is not None:
                setattr(p, name, value)
        if self.cloud is not None:
            p.provider_project = self.cloud.project
            p.provider_region = self.cloud.region
            p.provider_credentials = self.cloud.credentials_json
        for t in self.tools:
            p.tools.append(t._spec())
        if self.pipeline is not None:
            self.pipeline._apply(p)
        return p


def opening_pb(opening: Any) -> int:
    v = wire(opening)
    if v == Opening.AGENT.value:
        return pb.OPENING_TURN_AGENT
    if v == Opening.CALLER.value:
        return pb.OPENING_TURN_CALLER
    return pb.OPENING_TURN_UNSPECIFIED


def tool_problems(tools: Any, label: str) -> list[str]:
    """Every problem with a list of tools: bad declarations and duplicates."""
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        return [f"{label} must be a list of tools"]
    out: list[str] = []
    seen: set[str] = set()
    for i, t in enumerate(tools):
        if not isinstance(t, Tool):
            out.append(
                f"{label}[{i}] is a {type(t).__name__}, not a tool: declare it with @agent.tool "
                "or agent.tool_from_schema"
            )
            continue
        out.extend(t._problems())
        if t.name in seen:
            out.append(f'tool "{t.name}" is declared twice')
        seen.add(t.name)
    return out
