from __future__ import annotations

import asyncio
import sys

import pytest
from aiohttp import WSMsgType, web

from meshagent.api import Participant
from meshagent.api.messaging import pack_message

from .channel_process import ExternalChannelBridgeSupervisor, dispatch_main
from .external_process_channel import ExternalProcessChannel
from .messages import (
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_PARTICIPANT_CONNECT,
    ModelsRequest,
    ParticipantConnect,
)
from .process import Channel, Message


class _FakeConnection:
    def __init__(self, inbound: list[Message]) -> None:
        self.inbound: asyncio.Queue[Message | None] = asyncio.Queue()
        for message in inbound:
            self.inbound.put_nowait(message)
        self.inbound.put_nowait(None)
        self.sent: list[Message] = []

    async def send(
        self,
        *,
        payload,
        sender: Participant | None = None,
        to_participant_id: str | None = None,
    ) -> None:
        self.sent.append(
            Message(
                data=payload,
                sender=sender,
                to_participant_id=to_participant_id,
            )
        )

    async def receive(self) -> Message:
        message = await self.inbound.get()
        if message is None:
            raise EOFError
        return message


class _RecordingChannel(Channel):
    def __init__(self) -> None:
        super().__init__()
        self.direct_messages: list[tuple[Participant, object]] = []

    async def on_start(self) -> None:
        participant = Participant(id="room-user", attributes={"name": "Room User"})
        self.emit(
            sender=participant,
            payload=ParticipantConnect(
                type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                participant_id=participant.id,
            ),
        )

    async def send_agent_message_to_participant_and_wait(
        self,
        *,
        participant: Participant,
        payload,
    ) -> bool:
        self.direct_messages.append((participant, payload))
        return True


@pytest.mark.asyncio
async def test_bridge_runs_existing_channel_with_bidirectional_agent_messages() -> None:
    target = Participant(id="room-user", attributes={"name": "Room User"})
    request = ModelsRequest(type=AGENT_MESSAGE_MODELS_REQUEST)
    connection = _FakeConnection(
        [
            Message(
                data=request,
                sender=target,
                to_participant_id=target.id,
            )
        ]
    )
    channel = _RecordingChannel()
    bridge = ExternalChannelBridgeSupervisor(connection=connection)  # type: ignore[arg-type]

    await bridge.run_channel(channel=channel)

    assert channel.state == "stopped"
    assert channel.direct_messages == [(target, request)]
    assert len(connection.sent) == 1
    connected = connection.sent[0]
    assert isinstance(connected.data, ParticipantConnect)
    assert connected.sender is not None
    assert connected.sender.id == target.id
    assert connected.sender.attributes == target.attributes


def test_dispatch_main_ignores_regular_meshagent_executable() -> None:
    assert dispatch_main(executable_name="meshagent", argv=[]) is False


def _protocol_packet_frames(
    *, message_id: int, message_type: str, data: bytes
) -> list[bytes]:
    packet_size = 1024
    packet_count = (len(data) + packet_size - 1) // packet_size
    frames = [
        message_id.to_bytes(8)
        + (0).to_bytes(4)
        + packet_count.to_bytes(4)
        + message_type.encode()
    ]
    for index in range(packet_count):
        chunk = data[index * packet_size : (index + 1) * packet_size]
        frames.append(message_id.to_bytes(8) + (index + 1).to_bytes(4) + chunk)
    return frames


@pytest.mark.asyncio
async def test_bundled_toolkit_process_joins_room_over_external_channel() -> None:
    room_ready = asyncio.Event()

    async def room_handler(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        for message_id, message_type, payload in [
            (
                1,
                "room_ready",
                pack_message(
                    {
                        "room_name": "room-1",
                        "room_url": "ws://room.test/rooms/room-1",
                        "session_id": "session-1",
                    }
                ),
            ),
            (
                2,
                "connected",
                pack_message(
                    {
                        "type": "init",
                        "participantId": "channel-process",
                        "attributes": {"name": "Channel Process"},
                    }
                ),
            ),
        ]:
            for frame in _protocol_packet_frames(
                message_id=message_id,
                message_type=message_type,
                data=payload,
            ):
                await websocket.send_bytes(frame)
        room_ready.set()
        async for message in websocket:
            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break
        return websocket

    app = web.Application()
    app.router.add_get("/rooms/room-1", room_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]

    channel = ExternalProcessChannel(
        command=[
            sys.executable,
            "-c",
            "from meshagent.agents.channel_process import toolkit_main; toolkit_main()",
            "assistant",
        ],
        environment={
            "MESHAGENT_ROOM": "room-1",
            "MESHAGENT_TOKEN": "room-token",
            "MESHAGENT_ROOM_URL": f"ws://127.0.0.1:{port}",
        },
    )
    supervisor = _FakeParentSupervisor()
    try:
        await channel.start(supervisor)  # type: ignore[arg-type]
        await asyncio.wait_for(room_ready.wait(), timeout=5)
        await channel.stop(supervisor)  # type: ignore[arg-type]
    finally:
        if channel.state == "started":
            await channel.stop(supervisor)  # type: ignore[arg-type]
        await runner.cleanup()


class _FakeParentSupervisor:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def register_channel(self, _channel: Channel) -> int:
        return 1

    def unregister_channel(self, _channel_id: int, _channel: Channel) -> None:
        return None

    def send(self, message: Message) -> None:
        self.messages.append(message)
