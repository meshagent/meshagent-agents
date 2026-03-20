import asyncio
import uuid

import pytest

from meshagent.agents.messages import TurnStart
from meshagent.agents.process import Message
from meshagent.agents.queue_channel import QueueChannel
from meshagent.agents.thread_schema import thread_list_schema
from meshagent.api import Participant


class _FakeLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})
        self.set_attribute_calls: list[tuple[str, object]] = []

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value
        self.set_attribute_calls.append((name, value))


class _FakeQueues:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self.receive_calls: list[dict[str, object]] = []

    async def receive(self, *, name: str, create: bool, wait: bool):
        self.receive_calls.append({"name": name, "create": create, "wait": wait})
        return await self._queue.get()

    async def push(self, payload: object) -> None:
        await self._queue.put(payload)


class _FakeThreadListElement:
    def __init__(self, *, tag_name: str, attributes: dict[str, str]) -> None:
        self.tag_name = tag_name
        self._attributes = dict(attributes)

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


class _FakeThreadListRoot:
    def __init__(self) -> None:
        self._children: list[_FakeThreadListElement] = []

    def get_children(self) -> list[_FakeThreadListElement]:
        return [*self._children]

    def append_child(
        self,
        *,
        tag_name: str,
        attributes: dict[str, str],
    ) -> _FakeThreadListElement:
        element = _FakeThreadListElement(tag_name=tag_name, attributes=attributes)
        self._children.append(element)
        return element


class _FakeThreadListDocument:
    def __init__(self) -> None:
        self.root = _FakeThreadListRoot()


class _FakeSync:
    def __init__(self) -> None:
        self.document = _FakeThreadListDocument()
        self.open_calls: list[dict[str, object]] = []
        self.close_calls: list[str] = []

    async def open(self, *, path: str, schema=None) -> _FakeThreadListDocument:
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)


class _FakeStorage:
    def __init__(self, *, existing_paths: set[str] | None = None) -> None:
        self._existing_paths = set(existing_paths or [])
        self.exists_calls: list[str] = []

    async def exists(self, *, path: str) -> bool:
        self.exists_calls.append(path)
        return path in self._existing_paths


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.queues = _FakeQueues()
        self.sync = _FakeSync()
        self.storage = _FakeStorage()


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


@pytest.mark.asyncio
async def test_queue_channel_default_new_indexes_new_threads_from_prompt() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = QueueChannel(
        room=room,
        queue_name="jobs",
        threading_mode="default-new",
    )
    expected_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

    original_uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: expected_uuid
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await room.queues.push("follow up on billing issue")
        await _drain()

        assert room.local_participant.set_attribute_calls == [
            ("meshagent.chatbot.threading", "default-new"),
            ("meshagent.chatbot.thread-dir", ".threads/assistant"),
            ("meshagent.chatbot.thread-list", ".threads/assistant/index.threadl"),
        ]
        assert room.sync.open_calls == [
            {
                "path": ".threads/assistant/index.threadl",
                "schema": thread_list_schema,
            }
        ]
        assert room.storage.exists_calls == [
            ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        ]

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert (
            outbound.data.thread_id
            == ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        )

        entries = room.sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Follow Up On Billing Issue"
        assert (
            entries[0].get_attribute("path")
            == ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        )
    finally:
        uuid.uuid4 = original_uuid4
        await channel.stop(supervisor)  # type: ignore[arg-type]

    assert room.sync.close_calls == [".threads/assistant/index.threadl"]
