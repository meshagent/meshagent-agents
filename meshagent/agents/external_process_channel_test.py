from __future__ import annotations

import asyncio
import sys

import pytest

from meshagent.api import Participant

from .external_process_channel import ExternalProcessChannel
from .messages import (
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_PARTICIPANT_CONNECT,
    ModelsRequest,
    ParticipantConnect,
)
from .process import Message


_ECHO_CHANNEL = """
import sys
import msgpack
import os
import socket
from urllib.parse import urlparse

transport = urlparse(os.environ["MESHAGENT_CHANNEL_URL"])
if transport.scheme == "unix":
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(transport.path)
else:
    connection = socket.create_connection((transport.hostname, transport.port))
connection.sendall(msgpack.packb({
    "capability": os.environ["MESHAGENT_CHANNEL_CAPABILITY"],
}, use_bin_type=True))

participant = {
    "id": "child-user",
    "attributes": {"name": os.environ.get("MESHAGENT_TEST_NAME", "Child User")},
}
connected = {
    "data": {
        "type": "meshagent.agent.participant.connect",
        "participant_id": "child-user",
    },
    "sender": participant,
    "source": {"channel_id": 999999},
}
connection.sendall(msgpack.packb(connected, use_bin_type=True))

unpacker = msgpack.Unpacker(raw=False)
while chunk := connection.recv(65536):
    unpacker.feed(chunk)
    for message in unpacker:
        connection.sendall(msgpack.packb(message, use_bin_type=True))
"""


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.message_event = asyncio.Event()
        self.unregistered_channel_ids: list[int] = []

    def register_channel(self, _channel: ExternalProcessChannel) -> int:
        return 42

    def unregister_channel(
        self, channel_id: int, _channel: ExternalProcessChannel
    ) -> None:
        self.unregistered_channel_ids.append(channel_id)

    def send(self, message: Message) -> None:
        self.messages.append(message)
        self.message_event.set()


async def _wait_for_messages(supervisor: _RecordingSupervisor, count: int) -> None:
    while len(supervisor.messages) < count:
        supervisor.message_event.clear()
        await asyncio.wait_for(supervisor.message_event.wait(), timeout=2)


def test_external_process_channel_rejects_empty_command_arguments() -> None:
    with pytest.raises(ValueError, match="command cannot be empty"):
        ExternalProcessChannel(command=[sys.executable, ""])


@pytest.mark.asyncio
async def test_external_process_channel_round_trips_existing_msgpack_message_envelope() -> (
    None
):
    channel = ExternalProcessChannel(
        command=[sys.executable, "-c", _ECHO_CHANNEL],
        environment={"MESHAGENT_TEST_NAME": "Environment User"},
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await _wait_for_messages(supervisor, 1)
        connected = supervisor.messages[0]
        assert isinstance(connected.data, ParticipantConnect)
        assert connected.data.type == AGENT_MESSAGE_PARTICIPANT_CONNECT
        assert connected.sender is not None
        assert connected.sender.id == "child-user"
        assert connected.sender.attributes == {"name": "Environment User"}
        assert connected.source is channel
        assert channel.channel_id == 42

        participant = Participant(
            id="child-user",
            attributes={"name": "Environment User"},
        )
        assert channel.send_agent_message_to_participant(
            participant=participant,
            payload=ModelsRequest(type=AGENT_MESSAGE_MODELS_REQUEST),
        )
        await _wait_for_messages(supervisor, 2)

        echoed = supervisor.messages[1]
        assert isinstance(echoed.data, ModelsRequest)
        assert echoed.sender is not None
        assert echoed.sender.id == participant.id
        assert echoed.sender.attributes == participant.attributes
        assert echoed.to_participant_id == "child-user"
        assert echoed.source is channel
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]

    assert supervisor.unregistered_channel_ids == [42]
