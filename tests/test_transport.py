"""The room's media transport: the client picks how the agent server carries
the audio; unset leaves the server's default, WebSocket."""

import dataclasses

import pytest
from fakeserver import ROOM, support
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent


@pytest.mark.parametrize(
    ("transport", "want"),
    [
        (None, pb.MEDIA_TRANSPORT_UNSPECIFIED),
        (agent.Transport.WEBSOCKET, pb.MEDIA_TRANSPORT_WEBSOCKET),
        (agent.Transport.WEBRTC, pb.MEDIA_TRANSPORT_WEBRTC),
        ("webrtc", pb.MEDIA_TRANSPORT_WEBRTC),
    ],
)
async def test_room_transport_is_sent(fake, transport, want):
    await fake.client.join(dataclasses.replace(ROOM, transport=transport), support())
    assert fake.joins[-1].transport == want


async def test_unknown_transport_is_refused_before_sending(fake):
    room = dataclasses.replace(ROOM, transport="quic")
    with pytest.raises(agent.ValidationError, match="Room.transport"):
        await fake.client.join(room, support(), allow_unlisted_values=True)
    assert not fake.joins


def test_room_repr_shows_transport_not_token():
    r = dataclasses.replace(ROOM, transport=agent.Transport.WEBRTC)
    assert "transport='webrtc'" in repr(r)
    assert "'tok'" not in repr(r) and "token=[redacted]" in repr(r)
