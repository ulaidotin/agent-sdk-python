"""A fake agent server: it records what the SDK sends and answers from a
script. The round trips are real: a real listener, real streams, real status
codes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Annotated, Any, Literal

import grpc
import pytest
from ulai.agent.v1 import agent_pb2 as pb
from ulai.agent.v1 import agent_pb2_grpc as rpc

import ulai_agents as agent


class Stream:
    """The server's side of one Session.Run stream."""

    def __init__(self, context: grpc.aio.ServicerContext) -> None:
        self.context = context

    async def recv(self) -> pb.ClientEvent | None:
        """The next client event, or None when the client stopped sending."""
        msg = await self.context.read()
        return None if msg is grpc.aio.EOF else msg

    async def send(self, ev: pb.ServerEvent) -> None:
        await self.context.write(ev)

    async def started(self, session_id: str = "room-1", **kw: Any) -> pb.ClientEvent:
        """Answer an Attach the way the server does. Returns the Attach."""
        first = await self.recv()
        assert first is not None and first.WhichOneof("event") == "attach", first
        await self.send(pb.ServerEvent(sequence=1, session_started=pb.SessionStarted(session_id=session_id, **kw)))
        return first

    async def ended(self, **kw: Any) -> None:
        await self.send(pb.ServerEvent(session_ended=pb.SessionEnded(**kw)))

    async def until_gone(self) -> None:
        """Read until the client detaches."""
        while await self.recv() is not None:
            pass


Script = Callable[[Stream], Awaitable[None]]


class FakeServer(rpc.RoomServicer, rpc.SessionServicer, rpc.DispatchServicer):
    def __init__(self) -> None:
        self.joins: list[pb.JoinRoomRequest] = []
        self.leaves: list[pb.LeaveRequest] = []
        self.subs: list[pb.SubscribeRequest] = []
        self.keys: list[tuple[str, str]] = []  # (method, x-api-key)
        self.join_error: tuple[grpc.StatusCode, str] | None = None
        self.leave_error: tuple[grpc.StatusCode, str] | None = None
        self.leave_delay = 0.0
        # Errors to answer Subscribe with, one per attempt, before serving jobs.
        self.subscribe_errors: list[tuple[grpc.StatusCode, str]] = []
        self.session: Script | None = None
        self.jobs: asyncio.Queue[pb.Job | tuple[grpc.StatusCode, str] | None] = asyncio.Queue()
        self.address = ""
        self.client: agent.Client

    def _key(self, method: str, context: grpc.aio.ServicerContext) -> None:
        md = dict(context.invocation_metadata() or ())
        self.keys.append((method, md.get("x-api-key", "")))

    async def Join(self, request, context):
        self._key("Join", context)
        self.joins.append(request)
        if self.join_error:
            await context.abort(*self.join_error)
        return pb.JoinRoomResponse(
            room_id=request.room_id,
            agent_id=request.agent_id,
            participant_id="p-1",
            connection_id="c-1",
            node_address="node:443",
        )

    async def Leave(self, request, context):
        self._key("Leave", context)
        self.leaves.append(request)
        if self.leave_delay:
            await asyncio.sleep(self.leave_delay)
        if self.leave_error:
            await context.abort(*self.leave_error)
        return pb.LeaveResponse(was_connected=True, duration_ms=1500, played_ms=900)

    async def Run(self, request_iterator, context):
        self._key("Run", context)
        if self.session is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "no session script")
        await self.session(Stream(context))

    async def Subscribe(self, request, context):
        self._key("Subscribe", context)
        self.subs.append(request)
        if self.subscribe_errors:
            await context.abort(*self.subscribe_errors.pop(0))
        while True:
            job = await self.jobs.get()
            if job is None:  # end this subscription cleanly
                return
            if isinstance(job, tuple):  # break it: (code, message)
                await context.abort(*job)
            await context.write(job)


async def start_server(
    fake: FakeServer, *, credentials: grpc.ServerCredentials | None = None
) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server()
    rpc.add_RoomServicer_to_server(fake, server)
    rpc.add_SessionServicer_to_server(fake, server)
    rpc.add_DispatchServicer_to_server(fake, server)
    if credentials is None:
        port = server.add_insecure_port("127.0.0.1:0")
    else:
        port = server.add_secure_port("127.0.0.1:0", credentials)
    await server.start()
    return server, port


@pytest.fixture
async def fake():
    f = FakeServer()
    server, port = await start_server(f)
    f.address = f"127.0.0.1:{port}"
    f.client = agent.Client(f.address, api_key="k")
    yield f
    await f.client.close()
    await server.stop(None)


ROOM = agent.Room(id="room-1", url="https://cp.test", token="tok")


@agent.tool(description="Look up a balance.")
async def check_balance(
    account: Annotated[str, "The account number."],
    kind: Annotated[Literal["savings", "current"], "Which balance."] | None = None,
    limit: int | None = None,
) -> dict:
    return {"account": account, "balance": 250}


def support(**changes: Any) -> agent.Agent:
    """A fully populated agent, like the Go tests' support()."""
    a = agent.Agent(
        id="support",
        instructions="You are a support agent.",
        greeting="Hello",
        language=agent.Language.HINDI,
        voice=agent.Voice.KORE,
        pipeline=agent.Realtime(model=agent.RealtimeModel.GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO),
        tools=[check_balance],
        opening=agent.Opening.CALLER,
        max_duration=timedelta(minutes=10),
        temperature=0.3,
        cloud=agent.CloudAccount(project="proj", region="us-central1", credentials_json="{}"),
        options=agent.Options(
            interruptible=False,
            first_words_timeout=timedelta(seconds=4),
            voicemail_detection=True,
            context_compression=True,
            classifier=agent.ClassifierBackend.GEMINI_LIVE,
        ),
    )
    for k, v in changes.items():
        setattr(a, k, v)
    return a


async def eventually(check: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait for check to hold."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.01)
