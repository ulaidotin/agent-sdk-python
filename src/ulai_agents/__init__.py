"""Build voice agents on the Ulai agent server.

Describe an agent once::

    import ulai_agents as agent

    @agent.tool(description="Look up the balance of an account.")
    async def check_balance(account: Annotated[str, "The account number."]) -> dict:
        return {"balance_rupees": 250}

    support = agent.Agent(
        id="support",
        instructions="You are a bank's support agent. Be brief.",
        greeting="Hello, how can I help?",
        voice=agent.Voice.KORE,
        pipeline=agent.Realtime(model=agent.RealtimeModel.GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO),
        cloud=agent.CloudAccount.from_file(project, "", "service-account.json"),
        tools=[check_balance],
    )

then connect, and either start a call yourself or take calls as they arrive::

    async with agent.Client("agent-server:443", api_key=key, tls=True) as client:
        # Outbound: put the agent in a room and run the call to its end.
        await client.run(agent.Room(id=room_id, url=room_url, token=token), support, handle)

        # Inbound: run handle on every call for this agent, as each one starts.
        await client.serve(support.id, support.tools, handle)

``handle`` is the agent's own logic for one call. Tool calls are answered for
you, in the background. Events arrive in order::

    async def handle(session: agent.Session) -> None:
        async for event in session.events():
            match event:
                case agent.Transcript(final=True):
                    ...  # what was said
                case agent.Ended():
                    ...  # the last event

Choosing models
    Models, voices, languages and the rest are str enums: RealtimeModel,
    LLMModel, STTModel, Voice, Language, VAD and so on. Agent.validate rejects
    any value the server does not know before a call is made, so a typo fails
    on your machine rather than on a live call.

    The pipeline decides how the agent hears and speaks: Realtime (one
    speech-to-speech model), Cascade (speech-to-text, a text model, then
    text-to-speech) or UlaiLive (the Ulai Live API). Realtime and Cascade run
    on the agent's own Google Cloud project (Agent.cloud): the server holds no
    model credentials of its own.

Detaching is not hanging up
    Closing a session, or crashing, detaches from the call: the server carries
    on without you. A call ends when the caller hangs up, when the model ends
    it with its end_call tool, or when you call Session.hangup. run is the
    exception: when its handler returns or it is cancelled, it takes the agent
    out of the room.

Connections
    Without tls=True the connection is unencrypted, which Client allows only
    to this machine (localhost, or an SSH tunnel to the server). An agent's
    cloud key travels with every call.
"""

from ._agent import Agent, CloudAccount, Options, Room
from ._client import Client, ErrorHandler, Handler, Job, LeaveResult, Seat
from ._errors import (
    AgentError,
    AmbiguousParticipantError,
    CallEndedError,
    CallLostError,
    CanceledError,
    DeadlineError,
    EventsInUseError,
    InvalidRequestError,
    NotFoundError,
    NotSupportedError,
    RoomFullError,
    RoomNotFoundError,
    SessionClosedError,
    UnauthorizedError,
    UnavailableError,
    ValidationError,
)
from ._events import (
    Ended,
    Event,
    ModelUsage,
    PipelineMetrics,
    Rejected,
    StateChanged,
    ToolCall,
    ToolOutcome,
    Transcript,
    TurnEnded,
    TurnStarted,
    TurnTokens,
    Usage,
    UsageIO,
)
from ._models import (
    AgentState,
    CascadeMode,
    ClassifierBackend,
    Language,
    LLMModel,
    Opening,
    ParamType,
    PipelineKind,
    RealtimeModel,
    Role,
    Speaker,
    STTModel,
    Transport,
    TurnDetection,
    UlaiLiveVoice,
    VAD,
    Voice,
)
from ._pipeline import Cascade, Pipeline, Realtime, UlaiLive
from ._session import Session
from ._tools import Param, Tool, ToolError, tool, tool_from_schema
from ._types import BackendInfo, CallInfo

__version__ = "0.1.0"

__all__ = [
    # The agent
    "Agent",
    "CloudAccount",
    "Options",
    "Room",
    # Pipelines
    "Cascade",
    "Pipeline",
    "Realtime",
    "UlaiLive",
    # Tools
    "Param",
    "Tool",
    "ToolError",
    "tool",
    "tool_from_schema",
    # Connecting
    "Client",
    "ErrorHandler",
    "Handler",
    "Job",
    "LeaveResult",
    "Seat",
    "Session",
    "BackendInfo",
    "CallInfo",
    # Events
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
    # Values
    "AgentState",
    "CascadeMode",
    "ClassifierBackend",
    "Language",
    "LLMModel",
    "Opening",
    "Transport",
    "ParamType",
    "PipelineKind",
    "RealtimeModel",
    "Role",
    "Speaker",
    "STTModel",
    "TurnDetection",
    "UlaiLiveVoice",
    "VAD",
    "Voice",
    # Errors
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
