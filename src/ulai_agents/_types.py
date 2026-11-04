"""What the server says about a call and what runs it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ulai.agent.v1 import agent_pb2 as pb

from ._models import PipelineKind, TurnDetection

__all__ = ["BackendInfo", "CallInfo"]


@dataclass(frozen=True)
class CallInfo:
    """Describes a call."""

    call_id: str = ""
    agent_id: str = ""
    direction: str = ""
    """How the call reached the agent: "room" for a call started with join."""
    from_: str = ""
    """One end. For a room call, the agent's participant id."""
    to: str = ""
    """The other end. For a room call, the room id."""
    fields: dict[str, Any] = field(default_factory=dict)
    """The custom fields the call was started with (``fields=`` on join/run)."""

    @classmethod
    def _from_pb(cls, p: pb.CallInfo) -> CallInfo:
        fields: dict[str, Any] = {}
        if p.custom_fields_json:
            try:
                decoded = json.loads(p.custom_fields_json)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                fields = decoded
        return cls(
            call_id=p.call_id,
            agent_id=p.agent_id,
            direction=p.direction,
            from_=p.from_number,
            to=p.to_number,
            fields=fields,
        )


_PIPELINES = {
    pb.PIPELINE_REALTIME: PipelineKind.REALTIME,
    pb.PIPELINE_SEMI_CASCADE: PipelineKind.SEMI_CASCADE,
    pb.PIPELINE_CASCADE: PipelineKind.CASCADE,
}

_TURN_DETECTION = {
    pb.TURN_DETECTION_SERVER: TurnDetection.SERVER,
    pb.TURN_DETECTION_CLIENT: TurnDetection.CLIENT,
}


@dataclass(frozen=True)
class BackendInfo:
    """Describes what is running a call."""

    provider: str = ""
    model: str = ""
    pipeline: PipelineKind | None = None
    turn_detection: TurnDetection | None = None

    @classmethod
    def _from_pb(cls, p: pb.BackendInfo) -> BackendInfo:
        return cls(
            provider=p.provider,
            model=p.model,
            pipeline=_PIPELINES.get(p.pipeline),
            turn_detection=_TURN_DETECTION.get(p.turn_detection),
        )
