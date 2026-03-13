import asyncio

import pytest

from meshagent.agents.messages import TurnStart
from meshagent.agents.process import Message
from meshagent.agents.queue_channel import QueueChannel
from meshagent.api import Participant


class _FakeLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


class _FakeQueues:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self.receive_calls: list[dict[str, object]] = []

    async def receive(self, *, name: str, create: bool, wait: bool):
        self.receive_calls.append({"name": name, "create": create, "wait": wait})
        return await self._queue.get()

    async def push(self, payload: object) -> None:
        await self._queue.put(payload)


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.queues = _FakeQueues()


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


async def _drain() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_queue_channel_emits_turn_start_from_prompt_and_path() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = QueueChannel(room=room, queue_name="jobs")
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await room.queues.push(
            {
                "prompt": "Process webhook payload",
                "path": ".threads/jobs/webhook.thread",
                "model": "gpt-5.4",
                "instructions": "Be concise",
                "tools": [{"name": "search"}],
                "sender_name": "Webhook",
            }
        )
        await _drain()

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert outbound.data.thread_id == ".threads/jobs/webhook.thread"
        assert outbound.data.model == "gpt-5.4"
        assert outbound.data.instructions == "Be concise"
        assert outbound.data.toolkits == [{"name": "search"}]
        assert outbound.data.content[0].text == "Process webhook payload"
        assert outbound.sender is not None
        assert outbound.sender.get_attribute("name") == "Webhook"
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_queue_channel_generates_thread_path_and_uses_payload_json_when_prompt_missing() -> (
    None
):
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = QueueChannel(room=room, queue_name="jobs")
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await room.queues.push({"body": "hello", "value": 3})
        await _drain()

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert outbound.data.thread_id.startswith(".threads/assistant/")
        assert outbound.data.thread_id.endswith(".thread")
        assert '"body": "hello"' in outbound.data.content[0].text
        assert outbound.sender is None
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_queue_channel_supports_string_messages() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = QueueChannel(room=room, queue_name="jobs", thread_dir=".threads/queue")
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await room.queues.push("Do the thing")
        await _drain()

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert outbound.data.content[0].text == "Do the thing"
        assert outbound.data.thread_id.startswith(".threads/queue/")
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]
