import asyncio
import uuid
from unittest import mock

import pytest

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.chat_channel import ChatChannel
from meshagent.agents.thread_schema import thread_list_schema
from meshagent.agents.messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentFileContent,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentThreadStatus,
    AgentUsageUpdated,
    ApproveAgentToolCall,
    CapabilitiesRequest,
    ClearThread,
    CloseThread,
    AgentContextWindowUsage,
    OpenThread,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnSteered,
    TurnStart,
    TurnStarted,
    TurnSteer,
)
from meshagent.agents.process import AgentSupervisor, Message
from meshagent.api import Participant, RoomException, RoomMessage
from meshagent.api.messaging import EmptyContent, JsonContent
from meshagent.tools import ToolContext


class _FakeParticipant(Participant):
    def __init__(self, *, name: str, participant_id: str) -> None:
        super().__init__(id=participant_id, attributes={"name": name})


class _FakeLocalParticipant(_FakeParticipant):
    def __init__(self) -> None:
        super().__init__(name="assistant", participant_id="assistant-id")
        self.set_attribute_calls: list[tuple[str, object]] = []

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value
        self.set_attribute_calls.append((name, value))


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

    async def open(
        self,
        *,
        path: str,
        schema=None,
    ) -> _FakeThreadListDocument:
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


class _FakeMessaging:
    def __init__(
        self,
        *,
        participants: list[Participant] | None = None,
        is_enabled: bool = False,
    ) -> None:
        self._participants = {
            participant.id: participant for participant in participants or []
        }
        self._handlers: dict[str, list] = {}
        self._is_enabled = is_enabled
        self.enable_calls = 0
        self.sent_messages: list[dict] = []

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    async def enable(self) -> None:
        self.enable_calls += 1
        self._is_enabled = True

    def on(self, event_name: str, func) -> None:
        handlers = self._handlers.setdefault(event_name, [])
        handlers.append(func)

    def off(self, event_name: str, func) -> None:
        handlers = self._handlers.get(event_name)
        if handlers is None:
            return
        handlers.remove(func)

    def get_participant(self, participant_id: str) -> Participant | None:
        return self._participants.get(participant_id)

    def send_message_nowait(
        self,
        *,
        to: Participant,
        type: str,
        message: dict,
        attachment=None,
    ) -> None:
        del attachment
        self.sent_messages.append({"to": to, "type": type, "message": message})

    def emit_message(self, message: RoomMessage) -> None:
        for handler in self._handlers.get("message", []):
            handler(message=message)

    def remove_participant(self, participant_id: str) -> None:
        self._participants.pop(participant_id, None)


class _FakeRoom:
    def __init__(
        self,
        *,
        participants: list[Participant] | None = None,
        messaging_enabled: bool = False,
        sync: _FakeSync | None = None,
        storage: _FakeStorage | None = None,
    ) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.messaging = _FakeMessaging(
            participants=participants,
            is_enabled=messaging_enabled,
        )
        self.sync = sync if sync is not None else _FakeSync()
        self.storage = storage if storage is not None else _FakeStorage()


class _RecordingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


class _FakeThreadNameAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str) -> None:
        self.generated_thread_name = generated_thread_name
        self.prompts: list[str] = []

    def default_model(self) -> str:
        return "thread-name-model"

    async def next(
        self,
        *,
        context,
        caller,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        options=None,
    ):
        del caller
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        self.prompts = [
            message["content"]
            for message in context.messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        return {"thread_name": self.generated_thread_name}


async def _drain_background_tasks() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _assert_uuid_thread_path(*, path: str, prefix: str) -> None:
    assert path.startswith(prefix)
    assert path.endswith(".thread")
    basename = path[len(prefix) : -len(".thread")]
    parsed = uuid.UUID(basename)
    assert str(parsed) == basename


def _assert_uuid_thread_table_url(*, path: str, prefix: str) -> None:
    assert path.startswith(prefix)
    basename = path[len(prefix) :]
    parsed = uuid.UUID(basename)
    assert str(parsed) == basename


@pytest.mark.asyncio
async def test_chat_channel_exposes_chat_toolkits_and_new_thread_emits_turn_start() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        agent_toolkits = channel.get_agent_toolkits()
        assert len(agent_toolkits) == 1
        agent_tool_names = {tool.name for tool in agent_toolkits[0].tools}
        assert agent_tool_names == {
            "new_thread",
            "attach_file",
            "list_threads",
            "grep_thread_list",
        }

        remote_toolkit = channel.make_toolkit()
        assert {tool.name for tool in remote_toolkit.tools} == agent_tool_names
        assert remote_toolkit.validation_mode == "content_types"

        new_thread_tool = next(
            tool for tool in agent_toolkits[0].tools if tool.name == "new_thread"
        )
        context = ToolContext(caller=caller)
        result = await new_thread_tool.execute(
            context=context,
            message={
                "text": "Plan this friendly thread",
                "attachments": [{"path": "uploads/plan.md"}],
            },
        )

        assert isinstance(result, JsonContent)
        result_path = result.json["path"]
        result_message_id = result.json["message_id"]
        assert isinstance(result_message_id, str)
        assert result_message_id != ""
        assert isinstance(result_path, str)
        _assert_uuid_thread_path(path=result_path, prefix="/threads/chat/")

        assert len(supervisor.sent) == 1
        sent = supervisor.sent[0]
        assert sent.sender is caller
        assert sent.source is channel
        turn = sent.data
        assert isinstance(turn, TurnStart)
        assert turn.message_id == result_message_id
        assert turn.type == AGENT_MESSAGE_TURN_START
        assert turn.thread_id == result_path
        assert turn.content == [
            AgentTextContent(type="text", text="Plan this friendly thread"),
            AgentFileContent(type="file", url="room:///uploads/plan.md"),
        ]

        entries = sync.document.root.get_children()
        assert entries == []
        await channel._wait_for_thread_list_background_tasks()
        await _drain_background_tasks()
        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("path") == result_path
        assert entries[0].get_attribute("name") == "Plan this friendly thread"

        list_threads_tool = next(
            tool for tool in agent_toolkits[0].tools if tool.name == "list_threads"
        )
        list_result = await list_threads_tool.execute(
            context=context,
            limit=20,
            offset=0,
        )
        assert isinstance(list_result, JsonContent)
        assert list_result.json["total"] == 1
        assert list_result.json["threads"][0]["path"] == result_path
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_start_thread_message_allocates_thread_and_routes_turn() -> (
    None
):
    caller = _FakeParticipant(name="Jesse", participant_id="caller-id")
    sync = _FakeSync()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_START,
                        "message_id": "start-thread-1",
                        "content": [
                            {"type": "text", "text": "Plan this friendly thread"},
                            {"type": "file", "url": "uploads/plan.md"},
                        ],
                    }
                },
            )
        )

        await channel._wait_for_thread_list_background_tasks()
        await _drain_background_tasks()

        assert len(room.messaging.sent_messages) == 1
        response_payload = room.messaging.sent_messages[0]["message"]["payload"]
        assert response_payload["type"] == AGENT_EVENT_THREAD_STARTED
        assert response_payload["source_message_id"] == "start-thread-1"
        result_path = response_payload["thread_id"]
        assert isinstance(result_path, str)
        _assert_uuid_thread_path(path=result_path, prefix="/threads/chat/")

        assert len(supervisor.sent) == 1
        sent = supervisor.sent[0]
        assert sent.sender is caller
        assert sent.source is channel
        turn = sent.data
        assert isinstance(turn, TurnStart)
        assert turn.message_id == "start-thread-1"
        assert turn.thread_id == result_path
        assert turn.content == [
            AgentTextContent(type="text", text="Plan this friendly thread"),
            AgentFileContent(type="file", url="room:///uploads/plan.md"),
        ]

        assert channel._open_participant_ids_by_thread == {result_path: {caller.id}}
        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("path") == result_path
        assert entries[0].get_attribute("name") == "Plan this friendly thread"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_chat_channel_dataset_thread_urls_are_canonical() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    storage = _FakeStorage()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
        storage=storage,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/agents/dataset/threads",
        thread_url_scheme="dataset",
        thread_path_extension="",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        new_thread_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "new_thread"
        )
        result = await new_thread_tool.execute(
            context=ToolContext(caller=caller),
            message={"text": "hello"},
        )

        assert isinstance(result, JsonContent)
        result_path = result.json["path"]
        assert isinstance(result_path, str)
        result_message_id = result.json["message_id"]
        assert isinstance(result_message_id, str)
        assert result_message_id != ""
        assert len(supervisor.sent) == 1
        assert supervisor.sent[0].data.message_id == result_message_id
        _assert_uuid_thread_table_url(
            path=result_path,
            prefix="dataset://agents/dataset/threads/",
        )
        assert "dataset:///" not in result_path
        assert all(not path.startswith("dataset://") for path in storage.exists_calls)
        assert storage.exists_calls[0].startswith("/agents/dataset/threads/")

        await channel._wait_for_thread_list_background_tasks()
        await _drain_background_tasks()
        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("path") == result_path
        assert (
            room.local_participant.get_attribute("meshagent.chatbot.thread-dir")
            == "dataset://agents/dataset/threads"
        )
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_dataset_non_threading_publishes_thread_path() -> None:
    sync = _FakeSync()
    room = _FakeRoom(
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode=None,
        thread_dir="/agents/dataset/threads",
        thread_url_scheme="dataset",
        thread_path_extension="",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        assert (
            room.local_participant.get_attribute("meshagent.chatbot.thread-path")
            == "dataset://agents/dataset/threads/main"
        )
        assert (
            room.local_participant.get_attribute("meshagent.chatbot.thread-dir") is None
        )
        assert (
            room.local_participant.get_attribute("meshagent.chatbot.thread-list")
            is None
        )
        assert sync.open_calls == []
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_chat_channel_tmp_thread_urls_are_canonical() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    storage = _FakeStorage()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
        storage=storage,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/agents/tmp/threads",
        thread_url_scheme="tmp",
        thread_path_extension="",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        new_thread_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "new_thread"
        )
        result = await new_thread_tool.execute(
            context=ToolContext(caller=caller),
            message={"text": "hello"},
        )

        assert isinstance(result, JsonContent)
        result_path = result.json["path"]
        assert isinstance(result_path, str)
        _assert_uuid_thread_table_url(
            path=result_path,
            prefix="tmp://agents/tmp/threads/",
        )
        assert "tmp:///" not in result_path
        assert all(not path.startswith("tmp://") for path in storage.exists_calls)
        assert storage.exists_calls[0].startswith("/agents/tmp/threads/")

        await channel._wait_for_thread_list_background_tasks()
        await _drain_background_tasks()
        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("path") == result_path
        assert (
            room.local_participant.get_attribute("meshagent.chatbot.thread-dir")
            == "tmp://agents/tmp/threads"
        )
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_new_thread_uses_message_text_and_attachment_names_for_llm_naming() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    adapter = _FakeThreadNameAdapter(generated_thread_name="Release Plan")
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
        llm_adapter=adapter,
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        new_thread_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "new_thread"
        )
        result = await new_thread_tool.execute(
            context=ToolContext(caller=caller),
            message={
                "text": "Plan the release work",
                "attachments": [
                    {"path": "uploads/release-plan.md"},
                    {"path": "uploads/screenshot.png"},
                ],
            },
        )

        assert isinstance(result, JsonContent)
        assert isinstance(result.json["path"], str)
        assert isinstance(result.json["message_id"], str)
        await channel._wait_for_thread_list_background_tasks()
        assert adapter.prompts == [
            "Message:\nPlan the release work\n\nAttachments:\n- release-plan.md\n- screenshot.png"
        ]

        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Release Plan"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_new_thread_rejects_empty_message_without_attachments() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        new_thread_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "new_thread"
        )

        with pytest.raises(
            RoomException,
            match="requires non-empty text or at least one attachment",
        ):
            await new_thread_tool.execute(
                context=ToolContext(caller=caller),
                message={"text": "   ", "attachments": [{"path": "  "}]},
            )

        assert supervisor.sent == []
        assert sync.document.root.get_children() == []
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_attach_file_emits_file_content_events() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        storage=_FakeStorage(existing_paths={"docs/report.pdf"}),
    )
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        attach_file_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "attach_file"
        )

        result = await attach_file_tool.execute(
            context=ToolContext(
                caller=caller,
                caller_context={
                    "thread_id": "/threads/test.thread",
                    "turn_id": "turn-1",
                },
            ),
            path="docs/report.pdf",
        )

        assert isinstance(result, EmptyContent)
        assert room.storage.exists_calls == ["docs/report.pdf"]
        assert len(supervisor.sent) == 3

        started = supervisor.sent[0]
        delta = supervisor.sent[1]
        ended = supervisor.sent[2]
        assert started.sender is caller
        assert delta.sender is caller
        assert ended.sender is caller

        started_payload = started.data
        delta_payload = delta.data
        ended_payload = ended.data
        assert isinstance(started_payload, AgentFileContentStarted)
        assert started_payload.type == AGENT_EVENT_FILE_CONTENT_STARTED
        assert started_payload.thread_id == "/threads/test.thread"
        assert started_payload.turn_id == "turn-1"
        assert isinstance(delta_payload, AgentFileContentDelta)
        assert delta_payload.type == AGENT_EVENT_FILE_CONTENT_DELTA
        assert delta_payload.thread_id == "/threads/test.thread"
        assert delta_payload.turn_id == "turn-1"
        assert delta_payload.url == "room:///docs/report.pdf"
        assert isinstance(ended_payload, AgentFileContentEnded)
        assert ended_payload.type == AGENT_EVENT_FILE_CONTENT_ENDED
        assert ended_payload.thread_id == "/threads/test.thread"
        assert ended_payload.turn_id == "turn-1"
        assert started_payload.item_id == delta_payload.item_id
        assert delta_payload.item_id == ended_payload.item_id
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_attach_file_publishes_file_content_to_open_participants() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        storage=_FakeStorage(existing_paths={"docs/report.pdf"}),
    )
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        attach_file_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "attach_file"
        )

        await attach_file_tool.execute(
            context=ToolContext(
                caller=caller,
                caller_context={
                    "thread_id": "/threads/test.thread",
                    "turn_id": "turn-1",
                },
            ),
            path="docs/report.pdf",
        )

        payloads = [
            sent["message"]["payload"]
            for sent in room.messaging.sent_messages
            if sent["type"] == "agent-message"
        ]
        assert [payload["type"] for payload in payloads] == [
            AGENT_EVENT_FILE_CONTENT_STARTED,
            AGENT_EVENT_FILE_CONTENT_DELTA,
            AGENT_EVENT_FILE_CONTENT_ENDED,
        ]
        assert payloads[1]["url"] == "room:///docs/report.pdf"
        assert payloads[0]["item_id"] == payloads[1]["item_id"]
        assert payloads[1]["item_id"] == payloads[2]["item_id"]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_attach_file_raises_for_missing_room_file() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        attach_file_tool = next(
            tool
            for tool in channel.get_agent_toolkits()[0].tools
            if tool.name == "attach_file"
        )

        with pytest.raises(
            RoomException,
            match=r"attach_file could not find a room file at docs/missing\.pdf",
        ):
            await attach_file_tool.execute(
                context=ToolContext(
                    caller=caller,
                    caller_context={
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                    },
                ),
                path="docs/missing.pdf",
            )

        assert room.storage.exists_calls == ["docs/missing.pdf"]
        assert supervisor.sent == []
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_default_new_exposes_thread_list_tools_without_explicit_thread_dir() -> (
    None
):
    sync = _FakeSync()
    room = _FakeRoom(messaging_enabled=True, sync=sync)
    channel = ChatChannel(room=room, threading_mode="default-new")
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        tool_names = {tool.name for tool in channel.get_agent_toolkits()[0].tools}
        assert tool_names == {
            "new_thread",
            "attach_file",
            "list_threads",
            "grep_thread_list",
        }
        assert channel.make_toolkit().validation_mode == "content_types"
        assert (
            "meshagent.chatbot.thread-list",
            ".threads/assistant/index.threadl",
        ) in room.local_participant.set_attribute_calls
        assert (
            "meshagent.chatbot.thread-dir",
            ".threads/assistant",
        ) in room.local_participant.set_attribute_calls
        assert sync.open_calls == [
            {
                "path": ".threads/assistant/index.threadl",
                "schema": thread_list_schema,
            }
        ]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_ignores_legacy_chat_messages() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller])
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        assert room.messaging.enable_calls == 1

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="chat",
                message={
                    "path": "/threads/test.thread",
                    "attachments": [
                        {"path": "room://docs/report.pdf"},
                        {"path": "https://example.com/image.png"},
                    ],
                },
            )
        )

        assert supervisor.sent == []
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_chat_channel_translates_capabilities_request_agent_message() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_CAPABILITIES_REQUEST,
                        "thread_id": "/threads/test.thread",
                        "message_id": "capabilities-1",
                    }
                },
            )
        )

        assert len(supervisor.sent) == 1
        sent = supervisor.sent[0]
        assert sent.sender is caller
        assert sent.source is channel

        request = sent.data
        assert isinstance(request, CapabilitiesRequest)
        assert request.type == AGENT_MESSAGE_CAPABILITIES_REQUEST
        assert request.thread_id == "/threads/test.thread"
        assert request.message_id == "capabilities-1"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_chat_channel_does_not_bump_thread_index_on_capabilities_request() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    existing_entry = sync.document.root.append_child(
        tag_name="thread",
        attributes={
            "path": "/threads/chat/example.thread",
            "name": "Example",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_CAPABILITIES_REQUEST,
                        "thread_id": "/threads/chat/example.thread",
                        "message_id": "capabilities-1",
                    }
                },
            )
        )

        assert existing_entry.get_attribute("modified_at") == "2024-01-01T00:00:00Z"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_chat_channel_bumps_thread_index_on_turn_start() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    existing_entry = sync.document.root.append_child(
        tag_name="thread",
        attributes={
            "path": "/threads/chat/example.thread",
            "name": "Example",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_START,
                        "thread_id": "/threads/chat/example.thread",
                        "message_id": "turn-1",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                },
            )
        )

        assert existing_entry.get_attribute("modified_at") != "2024-01-01T00:00:00Z"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_translates_agent_steer_cancel_and_approval() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        assert room.messaging.enable_calls == 0

        await channel.on_message(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    source_message_id="source-1",
                )
            )
        )
        assert channel._active_turn_ids_by_thread == {"/threads/test.thread": "turn-1"}

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_STEER,
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                        "message_id": "steer-1",
                        "content": [{"type": "text", "text": "keep going"}],
                    }
                },
            )
        )

        steer_message = supervisor.sent[0].data
        assert isinstance(steer_message, TurnSteer)
        assert steer_message.type == AGENT_MESSAGE_TURN_STEER
        assert steer_message.thread_id == "/threads/test.thread"
        assert steer_message.turn_id == "turn-1"
        assert steer_message.content == [
            AgentTextContent(type="text", text="keep going")
        ]

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TOOL_CALL_APPROVE,
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                        "item_id": "approval-1",
                        "message_id": "approve-1",
                    }
                },
            )
        )

        approval_message = supervisor.sent[1].data
        assert isinstance(approval_message, ApproveAgentToolCall)
        assert approval_message.type == AGENT_MESSAGE_TOOL_CALL_APPROVE
        assert approval_message.thread_id == "/threads/test.thread"
        assert approval_message.turn_id == "turn-1"
        assert approval_message.item_id == "approval-1"

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_INTERRUPT,
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                        "message_id": "interrupt-1",
                    }
                },
            )
        )

        interrupt_message = supervisor.sent[2].data
        assert isinstance(interrupt_message, TurnInterrupt)
        assert interrupt_message.type == AGENT_MESSAGE_TURN_INTERRUPT
        assert interrupt_message.thread_id == "/threads/test.thread"
        assert interrupt_message.turn_id == "turn-1"

        await channel.on_message(
            Message(
                data=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    error=None,
                )
            )
        )

        assert len(supervisor.sent) == 3
        assert channel._active_turn_ids_by_thread == {}
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sends_agent_text_events_to_open_participants() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="opened",
                message={"path": "/threads/test.thread"},
            )
        )

        assert supervisor.sent == []

        await channel.on_message(
            Message(
                data=AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                    text="hello",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                )
            )
        )

        assert [sent["type"] for sent in room.messaging.sent_messages] == [
            "agent-message",
            "agent-message",
            "agent-message",
        ]
        assert [
            sent["message"]["payload"]["type"] for sent in room.messaging.sent_messages
        ] == [
            AGENT_EVENT_TEXT_CONTENT_STARTED,
            AGENT_EVENT_TEXT_CONTENT_DELTA,
            AGENT_EVENT_TEXT_CONTENT_ENDED,
        ]
        assert room.messaging.sent_messages[1]["message"]["payload"]["text"] == "hello"

        room.messaging.remove_participant(caller.id)

        await channel.on_message(
            Message(
                data=AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-2",
                    item_id="assistant-2",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-2",
                    item_id="assistant-2",
                    text="goodbye",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-2",
                    item_id="assistant-2",
                )
            )
        )

        assert len(room.messaging.sent_messages) == 3
        assert channel._open_participant_ids_by_thread == {}
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sends_usage_updates_to_open_participants() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(supervisor.sent) == 1
        assert isinstance(supervisor.sent[0].data, OpenThread)
        assert room.messaging.sent_messages == []

        await channel.on_message(
            Message(
                data=AgentUsageUpdated(
                    type=AGENT_EVENT_USAGE_UPDATED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    usage={
                        "input_tokens": 120,
                        "output_tokens": 30,
                    },
                    context_window=AgentContextWindowUsage(
                        used_tokens=480,
                        total_tokens=128000,
                        compaction_mode="auto",
                        compaction_threshold=64000,
                    ),
                )
            )
        )

        assert len(room.messaging.sent_messages) == 1
        sent = room.messaging.sent_messages[0]
        assert sent["to"] is caller
        assert sent["type"] == "agent-message"
        assert sent["message"] == {
            "payload": {
                "type": AGENT_EVENT_USAGE_UPDATED,
                "thread_id": "/threads/test.thread",
                "turn_id": "turn-1",
                "message_id": sent["message"]["payload"]["message_id"],
                "usage": {
                    "input_tokens": 120.0,
                    "output_tokens": 30.0,
                },
                "context_window": {
                    "used_tokens": 480,
                    "total_tokens": 128000,
                    "compaction_mode": "auto",
                    "compaction_threshold": 64000,
                },
            }
        }
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_agent_open_replays_buffered_events_and_close_unsubscribes() -> (
    None
):
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        await channel.on_message(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                    text="hello",
                )
            )
        )
        assert room.messaging.sent_messages == []

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 1
        assert room.messaging.sent_messages[0]["message"]["payload"]["type"] == (
            AGENT_EVENT_TEXT_CONTENT_DELTA
        )
        assert room.messaging.sent_messages[0]["message"]["payload"]["text"] == "hello"
        assert len(supervisor.sent) == 1
        assert supervisor.sent[0].sender is caller
        assert supervisor.sent[0].source is channel
        assert isinstance(supervisor.sent[0].data, OpenThread)
        assert supervisor.sent[0].data.thread_id == "/threads/test.thread"

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_CLOSE,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )
        assert len(supervisor.sent) == 2
        assert isinstance(supervisor.sent[1].data, CloseThread)
        assert supervisor.sent[1].data.thread_id == "/threads/test.thread"

        await channel.on_message(
            Message(
                data=AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    error=None,
                )
            )
        )
        assert len(room.messaging.sent_messages) == 1

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )
        assert len(room.messaging.sent_messages) == 1
        assert len(supervisor.sent) == 3
        assert isinstance(supervisor.sent[2].data, OpenThread)
        assert supervisor.sent[2].data.thread_id == "/threads/test.thread"

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_CLOSE,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )
        assert len(supervisor.sent) == 4
        assert isinstance(supervisor.sent[3].data, CloseThread)
        assert supervisor.sent[3].data.thread_id == "/threads/test.thread"

        await channel.on_message(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-2",
                    item_id="assistant-2",
                    text="ignored",
                )
            )
        )

        assert len(room.messaging.sent_messages) == 1
        assert channel._open_participant_ids_by_thread == {}
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_agent_open_replays_current_thread_status() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        await channel.on_message(
            Message(
                data=AgentThreadStatus(
                    type=AGENT_EVENT_THREAD_STATUS,
                    thread_id="/threads/test.thread",
                    status="Thinking",
                    mode="steerable",
                    started_at="2026-05-07T16:00:00Z",
                    turn_id="turn-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentThreadStatus(
                    type=AGENT_EVENT_THREAD_STATUS,
                    thread_id="/threads/test.thread",
                    status="Generating image",
                    mode="steerable",
                    started_at="2026-05-07T16:00:05Z",
                    turn_id="turn-1",
                )
            )
        )
        assert room.messaging.sent_messages == []

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 1
        payload = room.messaging.sent_messages[0]["message"]["payload"]
        assert payload == {
            "type": AGENT_EVENT_THREAD_STATUS,
            "thread_id": "/threads/test.thread",
            "message_id": payload["message_id"],
            "status": "Generating image",
            "mode": "steerable",
            "started_at": "2026-05-07T16:00:05Z",
            "turn_id": "turn-1",
        }

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_CLOSE,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        await channel.on_message(
            Message(
                data=AgentThreadStatus(
                    type=AGENT_EVENT_THREAD_STATUS,
                    thread_id="/threads/test.thread",
                    status=None,
                    mode=None,
                    started_at=None,
                    turn_id=None,
                )
            )
        )

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 1
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_publishes_turn_started_with_tracked_input_to_open_participants() -> (
    None
):
    sender = _FakeParticipant(name="sender", participant_id="sender-id")
    viewer = _FakeParticipant(name="viewer", participant_id="viewer-id")
    late_viewer = _FakeParticipant(name="late", participant_id="late-id")
    room = _FakeRoom(participants=[sender, viewer, late_viewer], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=viewer.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=sender.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_START,
                        "thread_id": "/threads/test.thread",
                        "message_id": "user-message-1",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                },
            )
        )

        assert len(supervisor.sent) == 2
        assert isinstance(supervisor.sent[0].data, OpenThread)
        assert isinstance(supervisor.sent[1].data, TurnStart)
        assert len(room.messaging.sent_messages) == 1
        input_payload = room.messaging.sent_messages[0]["message"]["payload"]
        assert room.messaging.sent_messages[0]["to"] == viewer
        assert input_payload["type"] == AGENT_MESSAGE_TURN_START
        assert input_payload["message_id"] == "user-message-1"
        assert input_payload["content"] == [{"type": "text", "text": "hello"}]
        assert input_payload["sender_name"] == "sender"

        await channel.on_message(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    source_message_id="user-message-1",
                )
            )
        )

        assert len(room.messaging.sent_messages) == 2
        sent_payload = room.messaging.sent_messages[1]["message"]["payload"]
        assert room.messaging.sent_messages[1]["to"] == viewer
        assert sent_payload["type"] == AGENT_EVENT_TURN_STARTED
        assert sent_payload["source_message_id"] == "user-message-1"
        assert sent_payload["content"] == [{"type": "text", "text": "hello"}]
        assert sent_payload["sender_name"] == "sender"

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=late_viewer.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 4
        replayed_input_payload = room.messaging.sent_messages[2]["message"]["payload"]
        replayed_payload = room.messaging.sent_messages[3]["message"]["payload"]
        assert room.messaging.sent_messages[2]["to"] == late_viewer
        assert replayed_input_payload["type"] == AGENT_MESSAGE_TURN_START
        assert replayed_input_payload["message_id"] == "user-message-1"
        assert replayed_input_payload["content"] == [{"type": "text", "text": "hello"}]
        assert room.messaging.sent_messages[3]["to"] == late_viewer
        assert replayed_payload["type"] == AGENT_EVENT_TURN_STARTED
        assert replayed_payload["source_message_id"] == "user-message-1"
        assert replayed_payload["content"] == [{"type": "text", "text": "hello"}]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_agent_open_replays_pending_turn_start_as_accepted() -> None:
    sender = _FakeParticipant(name="sender", participant_id="sender-id")
    viewer = _FakeParticipant(name="viewer", participant_id="viewer-id")
    room = _FakeRoom(participants=[sender, viewer], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=sender.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_START,
                        "thread_id": "/threads/test.thread",
                        "message_id": "user-message-1",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                },
            )
        )

        assert len(supervisor.sent) == 1
        assert isinstance(supervisor.sent[0].data, TurnStart)

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=viewer.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 2
        input_payload = room.messaging.sent_messages[0]["message"]["payload"]
        payload = room.messaging.sent_messages[1]["message"]["payload"]
        assert room.messaging.sent_messages[0]["to"] == viewer
        assert input_payload["type"] == AGENT_MESSAGE_TURN_START
        assert input_payload["message_id"] == "user-message-1"
        assert input_payload["content"] == [{"type": "text", "text": "hello"}]
        assert input_payload["sender_name"] == "sender"
        assert room.messaging.sent_messages[1]["to"] == viewer
        assert payload["type"] == AGENT_EVENT_TURN_START_ACCEPTED
        assert payload["source_message_id"] == "user-message-1"
        assert "content" not in payload
        assert "sender_name" not in payload or payload["sender_name"] is None
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_agent_open_replays_pending_turn_steer_as_accepted() -> None:
    sender = _FakeParticipant(name="sender", participant_id="sender-id")
    viewer = _FakeParticipant(name="viewer", participant_id="viewer-id")
    room = _FakeRoom(participants=[sender, viewer], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=sender.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_STEER,
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                        "message_id": "user-message-2",
                        "content": [{"type": "text", "text": "wait"}],
                    }
                },
            )
        )

        assert len(supervisor.sent) == 1
        assert isinstance(supervisor.sent[0].data, TurnSteer)

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=viewer.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        assert len(room.messaging.sent_messages) == 2
        input_payload = room.messaging.sent_messages[0]["message"]["payload"]
        payload = room.messaging.sent_messages[1]["message"]["payload"]
        assert room.messaging.sent_messages[0]["to"] == viewer
        assert input_payload["type"] == AGENT_MESSAGE_TURN_STEER
        assert input_payload["message_id"] == "user-message-2"
        assert input_payload["turn_id"] == "turn-1"
        assert input_payload["content"] == [{"type": "text", "text": "wait"}]
        assert input_payload["sender_name"] == "sender"
        assert room.messaging.sent_messages[1]["to"] == viewer
        assert payload["type"] == AGENT_EVENT_TURN_STEER_ACCEPTED
        assert payload["source_message_id"] == "user-message-2"
        assert payload["turn_id"] == "turn-1"
        assert "content" not in payload
        assert "sender_name" not in payload or payload["sender_name"] is None
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_publishes_turn_steered_with_tracked_input_to_open_participants() -> (
    None
):
    sender = _FakeParticipant(name="sender", participant_id="sender-id")
    viewer = _FakeParticipant(name="viewer", participant_id="viewer-id")
    room = _FakeRoom(participants=[sender, viewer], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=viewer.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_OPEN,
                        "thread_id": "/threads/test.thread",
                    }
                },
            )
        )

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=sender.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_TURN_STEER,
                        "thread_id": "/threads/test.thread",
                        "turn_id": "turn-1",
                        "message_id": "user-message-2",
                        "content": [{"type": "text", "text": "wait"}],
                    }
                },
            )
        )

        assert len(supervisor.sent) == 2
        assert isinstance(supervisor.sent[0].data, OpenThread)
        assert isinstance(supervisor.sent[1].data, TurnSteer)
        assert len(room.messaging.sent_messages) == 1
        input_payload = room.messaging.sent_messages[0]["message"]["payload"]
        assert room.messaging.sent_messages[0]["to"] == viewer
        assert input_payload["type"] == AGENT_MESSAGE_TURN_STEER
        assert input_payload["message_id"] == "user-message-2"
        assert input_payload["content"] == [{"type": "text", "text": "wait"}]
        assert input_payload["sender_name"] == "sender"

        await channel.on_message(
            Message(
                data=TurnSteered(
                    type=AGENT_EVENT_TURN_STEERED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    source_message_id="user-message-2",
                )
            )
        )

        assert len(room.messaging.sent_messages) == 2
        sent_payload = room.messaging.sent_messages[1]["message"]["payload"]
        assert room.messaging.sent_messages[1]["to"] == viewer
        assert sent_payload["type"] == AGENT_EVENT_TURN_STEERED
        assert sent_payload["source_message_id"] == "user-message-2"
        assert sent_payload["content"] == [{"type": "text", "text": "wait"}]
        assert sent_payload["sender_name"] == "sender"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sends_agent_file_events_to_open_participants() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="opened",
                message={"path": "/threads/test.thread"},
            )
        )

        await channel.on_message(
            Message(
                data=AgentFileContentStarted(
                    type=AGENT_EVENT_FILE_CONTENT_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-file-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentFileContentDelta(
                    type=AGENT_EVENT_FILE_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-file-1",
                    url="room:///docs/report.pdf",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentFileContentEnded(
                    type=AGENT_EVENT_FILE_CONTENT_ENDED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-file-1",
                )
            )
        )

        assert [sent["type"] for sent in room.messaging.sent_messages] == [
            "agent-message",
            "agent-message",
            "agent-message",
        ]
        assert [
            sent["message"]["payload"]["type"] for sent in room.messaging.sent_messages
        ] == [
            AGENT_EVENT_FILE_CONTENT_STARTED,
            AGENT_EVENT_FILE_CONTENT_DELTA,
            AGENT_EVENT_FILE_CONTENT_ENDED,
        ]
        assert (
            room.messaging.sent_messages[1]["message"]["payload"]["url"]
            == "room:///docs/report.pdf"
        )
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_translates_agent_clear_messages() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="agent-message",
                message={
                    "payload": {
                        "type": AGENT_MESSAGE_THREAD_CLEAR,
                        "thread_id": "/threads/test.thread",
                        "message_id": "clear-1",
                    }
                },
            )
        )

        assert len(supervisor.sent) == 1
        clear_message = supervisor.sent[0].data
        assert isinstance(clear_message, ClearThread)
        assert clear_message.type == AGENT_MESSAGE_THREAD_CLEAR
        assert clear_message.thread_id == "/threads/test.thread"
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_notifies_open_participants_when_thread_is_cleared() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="opened",
                message={"path": "/threads/test.thread"},
            )
        )

        await channel.on_message(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    source_message_id="source-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                )
            )
        )
        await channel.on_message(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="assistant-1",
                    text="partial",
                )
            )
        )
        room.messaging.sent_messages.clear()

        await channel.on_message(
            Message(
                data=ThreadCleared(
                    type=AGENT_EVENT_THREAD_CLEARED,
                    thread_id="/threads/test.thread",
                    source_message_id="clear-source-1",
                )
            )
        )

        assert room.messaging.sent_messages == [
            {
                "to": caller,
                "type": "agent-message",
                "message": {
                    "payload": {
                        "type": AGENT_EVENT_THREAD_CLEARED,
                        "thread_id": "/threads/test.thread",
                        "message_id": room.messaging.sent_messages[0]["message"][
                            "payload"
                        ]["message_id"],
                        "source_message_id": "clear-source-1",
                    }
                },
            }
        ]
        assert channel._active_turn_ids_by_thread == {}
        assert channel._open_participant_ids_by_thread == {
            "/threads/test.thread": {caller.id}
        }
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sets_threading_attributes_and_tracks_thread_list() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        assert room.local_participant.set_attribute_calls == [
            ("meshagent.chatbot.threading", "default-new"),
            ("meshagent.chatbot.thread-dir", "/threads/chat"),
            ("meshagent.chatbot.thread-list", "/threads/chat/index.threadl"),
            ("empty_state_title", "How can I help you?"),
            ("supports_agent_messages", True),
        ]
        assert sync.open_calls == [
            {
                "path": "/threads/chat/index.threadl",
                "schema": thread_list_schema,
            }
        ]

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="opened",
                message={"path": "/threads/chat/example.thread"},
            )
        )

        assert sync.document.root.get_children() == []
        assert room.messaging.sent_messages == []
    finally:
        await channel.stop(supervisor)

    assert sync.close_calls == ["/threads/chat/index.threadl"]


@pytest.mark.asyncio
async def test_chat_channel_does_not_bump_thread_index_on_open() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    sync = _FakeSync()
    existing_entry = sync.document.root.append_child(
        tag_name="thread",
        attributes={
            "path": "/threads/chat/example.thread",
            "name": "Example",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    room = _FakeRoom(
        participants=[caller],
        messaging_enabled=True,
        sync=sync,
    )
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="opened",
                message={"path": "/threads/chat/example.thread"},
            )
        )
        assert existing_entry.get_attribute("modified_at") == "2024-01-01T00:00:00Z"
    finally:
        await channel.stop(supervisor)


def test_chat_channel_bump_thread_forces_modified_at_forward_when_clock_does_not_advance() -> (
    None
):
    existing_modified_at = "2024-01-01T00:00:00Z"
    sync = _FakeSync()
    existing_entry = sync.document.root.append_child(
        tag_name="thread",
        attributes={
            "path": "/threads/chat/example.thread",
            "name": "Example",
            "created_at": existing_modified_at,
            "modified_at": existing_modified_at,
        },
    )
    room = _FakeRoom(sync=sync)
    channel = ChatChannel(
        room=room,
        threading_mode="default-new",
        thread_dir="/threads/chat",
    )
    channel._thread_list_document = sync.document

    with mock.patch.object(channel, "_utc_now_iso", return_value=existing_modified_at):
        channel.bump_thread(path="/threads/chat/example.thread")

    updated_modified_at = existing_entry.get_attribute("modified_at")
    assert isinstance(updated_modified_at, str)
    assert channel._parse_iso_datetime(
        value=updated_modified_at
    ) > channel._parse_iso_datetime(value=existing_modified_at)
