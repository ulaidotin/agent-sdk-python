"""The connection: TLS rules, the API key, and how server failures map to errors."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
from datetime import timedelta

import grpc
import pytest
from fakeserver import ROOM, FakeServer, Stream, start_server, support
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent

# ── 1. The connection ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1:50052", "localhost:50052", "LOCALHOST:1", "[::1]:50052", "::1", "127.8.9.10:1",
     "dns:///localhost:50052", "passthrough:///127.0.0.1:5", "unix:/tmp/agent.sock", "unix:///tmp/agent.sock",
     "localhost"],
)
def test_plaintext_to_this_machine(address):
    agent.Client(address)


@pytest.mark.parametrize(
    "address", ["agents.example.com:443", "10.0.0.5:50052", "[2001:db8::1]:443", "dns:///agents.example.com:443",
                "localhost.example.com:1"],
)
def test_plaintext_elsewhere_needs_tls_or_insecure(address):
    with pytest.raises(agent.ValidationError, match="tls=True"):
        agent.Client(address)
    agent.Client(address, tls=True)
    agent.Client(address, insecure=True)


def test_client_needs_an_address():
    with pytest.raises(agent.ValidationError, match="address"):
        agent.Client("")


def test_root_certificates_need_a_certificate():
    with pytest.raises(agent.ValidationError, match="no certificates in the PEM"):
        agent.Client("agents.example.com:443", root_certificates=b"not a pem")


async def test_the_api_key_goes_with_every_rpc(fake):
    async def script(s: Stream):
        await s.started()
        await s.ended()

    fake.session = script
    seat = await fake.client.join(ROOM, support())
    await seat.leave("bye")
    session = await fake.client.attach("room-1")
    await session.close()
    await fake.jobs.put(pb.Job(session_id="s-1"))
    async for job in fake.client.jobs(["support"]):
        assert job.session_id == "s-1"
        break
    assert sorted(fake.keys) == [("Join", "k"), ("Leave", "k"), ("Run", "k"), ("Subscribe", "k")]


async def test_no_api_key_sends_none(fake):
    client = agent.Client(fake.address)
    try:
        await client.join(ROOM, support())
    finally:
        await client.close()
    assert fake.keys == [("Join", "")]


async def test_a_server_that_is_down_is_unavailable():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    async with agent.Client(f"127.0.0.1:{port}", api_key="k") as client:
        with pytest.raises(agent.UnavailableError):
            await client.join(ROOM, support())
        with pytest.raises(agent.UnavailableError):
            await client.attach("room-1")
        with pytest.raises(agent.UnavailableError):
            async for _ in client.jobs(["support"]):
                pass


@pytest.fixture
def certificate(tmp_path):
    """A self-signed certificate for 127.0.0.1, made with openssl."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
         "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
        check=True, capture_output=True,
    )
    return key.read_bytes(), cert.read_bytes()


async def test_tls_with_a_private_ca(certificate):
    key, cert = certificate
    fake = FakeServer()
    server, port = await start_server(fake, credentials=grpc.ssl_server_credentials([(key, cert)]))
    try:
        # Trusting the private issuer: it works.
        async with agent.Client(f"127.0.0.1:{port}", api_key="k", root_certificates=cert) as client:
            seat = await client.join(ROOM, support())
            assert seat.participant_id == "p-1"
        # The system roots do not know it: the handshake fails.
        async with agent.Client(f"127.0.0.1:{port}", api_key="k", tls=True) as client:
            with pytest.raises(agent.UnavailableError):
                await client.join(ROOM, support())
        # Plaintext to a TLS server fails too.
        async with agent.Client(f"127.0.0.1:{port}", api_key="k") as client:
            with pytest.raises(agent.UnavailableError):
                await client.join(ROOM, support())
        assert len(fake.joins) == 1
    finally:
        await server.stop(None)


# ── Errors ────────────────────────────────────────────────────────────────────

ROOM_CODES = [
    (grpc.StatusCode.NOT_FOUND, agent.RoomNotFoundError),
    (grpc.StatusCode.RESOURCE_EXHAUSTED, agent.RoomFullError),
    (grpc.StatusCode.FAILED_PRECONDITION, agent.AmbiguousParticipantError),
]

API_CODES = [
    (grpc.StatusCode.UNAUTHENTICATED, agent.UnauthorizedError),
    (grpc.StatusCode.PERMISSION_DENIED, agent.UnauthorizedError),
    (grpc.StatusCode.UNAVAILABLE, agent.UnavailableError),
    (grpc.StatusCode.INVALID_ARGUMENT, agent.InvalidRequestError),
    (grpc.StatusCode.NOT_FOUND, agent.NotFoundError),
    (grpc.StatusCode.UNIMPLEMENTED, agent.NotSupportedError),
    (grpc.StatusCode.DEADLINE_EXCEEDED, agent.DeadlineError),
    (grpc.StatusCode.CANCELLED, agent.CanceledError),
]


@pytest.mark.parametrize(
    "code, want",
    ROOM_CODES + [c for c in API_CODES if c[0] != grpc.StatusCode.NOT_FOUND],
    ids=lambda v: getattr(v, "name", getattr(v, "__name__", "")),
)
async def test_join_and_leave_errors(fake, code, want):
    fake.join_error = (code, "from the server")
    with pytest.raises(want) as e:
        await fake.client.join(ROOM, support())
    assert type(e.value) is want
    assert "from the server" in str(e.value) and str(e.value).startswith("join: ")
    assert (e.value.code, e.value.details) == (code, "from the server")
    assert isinstance(e.value, agent.AgentError)

    fake.leave_error = (code, "from the server")
    with pytest.raises(want) as e:
        await fake.client.leave("room-1", participant_id="p-1")
    assert "from the server" in str(e.value) and str(e.value).startswith("leave: ")


@pytest.mark.parametrize("code, want", API_CODES, ids=lambda v: getattr(v, "name", getattr(v, "__name__", "")))
async def test_session_errors(fake, code, want):
    async def script(s: Stream):
        await s.context.abort(code, "from the server")

    fake.session = script
    with pytest.raises(want) as e:
        await fake.client.attach("room-1")
    assert type(e.value) is want and str(e.value) == f"attach: {want.summary} (from the server)"


@pytest.mark.parametrize("code, want", API_CODES, ids=lambda v: getattr(v, "name", getattr(v, "__name__", "")))
async def test_subscribe_errors(fake, code, want):
    fake.subscribe_errors = [(code, "from the server")]
    with pytest.raises(want, match="from the server"):
        async for _ in fake.client.jobs(["support"]):
            pass


async def test_unmapped_codes_keep_theirs(fake):
    fake.join_error = (grpc.StatusCode.INTERNAL, "boom")
    with pytest.raises(agent.AgentError) as e:
        await fake.client.join(ROOM, support())
    assert type(e.value) is agent.AgentError
    assert str(e.value) == "join: INTERNAL: boom" and e.value.code == grpc.StatusCode.INTERNAL


def test_error_classes():
    for cls in (agent.UnauthorizedError, agent.UnavailableError, agent.InvalidRequestError, agent.NotFoundError,
                agent.NotSupportedError, agent.DeadlineError, agent.CanceledError, agent.CallEndedError,
                agent.CallLostError, agent.SessionClosedError, agent.EventsInUseError, agent.RoomNotFoundError,
                agent.RoomFullError, agent.AmbiguousParticipantError, agent.ValidationError):
        assert issubclass(cls, agent.AgentError) and issubclass(cls, Exception), cls
    assert str(agent.SessionClosedError()) == "the session is closed"
    assert issubclass(agent.DeadlineError, TimeoutError)
    # Cancellation stays cancellation for asyncio.
    assert issubclass(agent.CanceledError, asyncio.CancelledError)
    assert issubclass(agent.ValidationError, ValueError)


async def test_leave(fake):
    result = await fake.client.leave("room-1", participant_id="p-2", reason="done", force=True)
    assert result == agent.LeaveResult(present=True, duration=timedelta(seconds=1.5), heard=timedelta(milliseconds=900))
    req = fake.leaves[0]
    assert (req.room_id, req.participant_id, req.reason, req.force) == ("room-1", "p-2", "done", True)
    with pytest.raises(agent.ValidationError, match="leave needs a room id"):
        await fake.client.leave("")

