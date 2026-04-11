import uuid
from unittest import mock

import pytest
from pydantic import BaseModel

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.legacy_chat_channel import LegacyChatChannel as ChatChannel
from meshagent.agents.thread_schema import thread_list_schema
from meshagent.agents.messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_THREAD_CLEAR,
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
    AgentToolCallApprovalRequested,
    ApproveAgentToolCall,
    ClearThread,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnStart,
    TurnStarted,
    TurnSteer,
)
from meshagent.agents.process import AgentSupervisor, Message
from meshagent.api import Participant, RoomException, RoomMessage
from meshagent.api.messaging import EmptyContent, JsonContent
from meshagent.tools import ToolContext, ToolkitBuilder


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


class _FakeToolkitConfig(BaseModel):
    name: str


class _FakeToolkitBuilder(ToolkitBuilder):
    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, type=_FakeToolkitConfig)

    async def make(self, *, model: str, config: _FakeToolkitConfig):
        del model
        del config
        raise AssertionError(
            "toolkit builder should not be called in chat channel test"
        )


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
        room,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        options=None,
    ):
        del room
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


def _assert_uuid_thread_path(*, path: str, prefix: str) -> None:
    assert path.startswith(prefix)
    assert path.endswith(".thread")
    basename = path[len(prefix) : -len(".thread")]
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
        thread_dir="/threads/chat",
        toolkit_builders=[_FakeToolkitBuilder(name="search")],
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
        tools_schema = new_thread_tool.input_schema["properties"]["message"][
            "properties"
        ]["tools"]
        assert len(tools_schema["anyOf"]) == 2
        assert tools_schema["anyOf"][0]["type"] == "array"
        assert tools_schema["anyOf"][0]["items"]["type"] == "object"
        assert (
            tools_schema["anyOf"][0]["items"]["properties"]["name"]["type"] == "string"
        )
        assert tools_schema["anyOf"][1] == {"type": "null"}
        context = ToolContext(room=room, caller=caller)
        result = await new_thread_tool.execute(
            context=context,
            message={
                "text": "Plan this friendly thread",
                "attachments": [{"path": "uploads/plan.md"}],
                "tools": [{"name": "search"}],
            },
        )

        assert isinstance(result, JsonContent)
        result_path = result.json["path"]
        assert result.json["name"] == "Plan this friendly thread"
        assert isinstance(result_path, str)
        _assert_uuid_thread_path(path=result_path, prefix="/threads/chat/")

        assert len(supervisor.sent) == 1
        sent = supervisor.sent[0]
        assert sent.sender is caller
        assert sent.source is channel
        turn = sent.data
        assert isinstance(turn, TurnStart)
        assert turn.type == AGENT_MESSAGE_TURN_START
        assert turn.thread_id == result_path
        assert turn.toolkits == [{"name": "search"}]
        assert turn.content == [
            AgentTextContent(type="text", text="Plan this friendly thread"),
            AgentFileContent(type="file", url="room:///uploads/plan.md"),
        ]

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
            context=ToolContext(room=room, caller=caller),
            message={
                "text": "Plan the release work",
                "attachments": [
                    {"path": "uploads/release-plan.md"},
                    {"path": "uploads/screenshot.png"},
                ],
            },
        )

        assert isinstance(result, JsonContent)
        assert result.json["name"] == "Release Plan"
        assert adapter.prompts == [
            "Message:\nPlan the release work\n\nAttachments:\n- release-plan.md\n- screenshot.png"
        ]

        entries = sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Release Plan"
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
                room=room,
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
                    room=room,
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
async def test_chat_channel_enables_messaging_and_translates_chat_messages() -> None:
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
                    "text": "hello",
                    "attachments": [{"path": "docs/report.pdf"}],
                    "tools": [{"name": "search"}],
                    "model": "gpt-5",
                },
            )
        )

        assert len(supervisor.sent) == 1
        sent = supervisor.sent[0]
        assert sent.sender is caller
        assert sent.source is channel

        turn = sent.data
        assert isinstance(turn, TurnStart)
        assert turn.type == AGENT_MESSAGE_TURN_START
        assert turn.thread_id == "/threads/test.thread"
        assert turn.model == "gpt-5"
        assert turn.toolkits == [{"name": "search"}]
        assert turn.content == [
            AgentTextContent(type="text", text="hello"),
            AgentFileContent(type="file", url="room:///docs/report.pdf"),
        ]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_preserves_existing_attachment_urls() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
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

        assert len(supervisor.sent) == 1
        turn = supervisor.sent[0].data
        assert isinstance(turn, TurnStart)
        assert turn.content == [
            AgentFileContent(type="file", url="room://docs/report.pdf"),
            AgentFileContent(type="file", url="https://example.com/image.png"),
        ]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_tracks_turn_state_for_steer_cancel_and_approval() -> None:
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

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="steer",
                message={
                    "path": "/threads/test.thread",
                    "text": "keep going",
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

        await channel.on_message(
            Message(
                data=AgentToolCallApprovalRequested(
                    type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="approval-1",
                    toolkit="filesystem",
                    tool="delete",
                    arguments={"path": "/tmp/file"},
                )
            )
        )

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="approved",
                message={
                    "path": "/threads/test.thread",
                    "approval_id": "approval-1",
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
                type="cancel",
                message={"path": "/threads/test.thread"},
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

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="steer",
                message={
                    "path": "/threads/test.thread",
                    "text": "one more thing",
                },
            )
        )

        assert len(supervisor.sent) == 3
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sends_completed_text_to_open_participants() -> None:
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

        assert room.messaging.sent_messages == [
            {
                "to": caller,
                "type": "chat",
                "message": {
                    "path": "/threads/test.thread",
                    "text": "hello",
                },
            }
        ]

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

        assert len(room.messaging.sent_messages) == 1
        assert channel._open_participant_ids_by_thread == {}
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sends_completed_file_to_open_participants() -> None:
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

        assert room.messaging.sent_messages == [
            {
                "to": caller,
                "type": "chat",
                "message": {
                    "path": "/threads/test.thread",
                    "attachments": [{"path": "room:///docs/report.pdf"}],
                },
            }
        ]
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_translates_clear_messages() -> None:
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    room = _FakeRoom(participants=[caller], messaging_enabled=True)
    channel = ChatChannel(room=room)
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="clear",
                message={"path": "/threads/test.thread"},
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
                data=AgentToolCallApprovalRequested(
                    type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
                    thread_id="/threads/test.thread",
                    turn_id="turn-1",
                    item_id="approval-1",
                    toolkit="filesystem",
                    tool="delete",
                    arguments={"path": "/tmp/file"},
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
                "type": "cleared",
                "message": {
                    "path": "/threads/test.thread",
                },
            }
        ]
        assert channel._active_turn_ids_by_thread == {}
        assert channel._pending_approval_turn_ids_by_thread == {}
        assert channel._active_text_by_thread == {}
        assert channel._open_participant_ids_by_thread == {
            "/threads/test.thread": {caller.id}
        }
    finally:
        await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_chat_channel_sets_threading_attributes_tracks_thread_list_and_reports_tool_providers() -> (
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
        toolkit_builders=[
            _FakeToolkitBuilder(name="search"),
            _FakeToolkitBuilder(name="shell"),
        ],
    )
    supervisor = _RecordingSupervisor()

    await channel.start(supervisor)
    try:
        assert room.local_participant.set_attribute_calls == [
            ("meshagent.chatbot.threading", "default-new"),
            ("meshagent.chatbot.thread-dir", "/threads/chat"),
            ("meshagent.chatbot.thread-list", "/threads/chat/index.threadl"),
            ("empty_state_title", "How can I help you?"),
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

        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="get_thread_toolkit_builders",
                message={"path": "/threads/chat/example.thread"},
            )
        )

        assert room.messaging.sent_messages == [
            {
                "to": caller,
                "type": "set_thread_tool_providers",
                "message": {
                    "path": "/threads/chat/example.thread",
                    "tool_providers": [
                        {"name": "search"},
                        {"name": "shell"},
                    ],
                },
            }
        ]
    finally:
        await channel.stop(supervisor)

    assert sync.close_calls == ["/threads/chat/index.threadl"]


@pytest.mark.asyncio
async def test_chat_channel_does_not_bump_thread_index_on_open_or_tool_provider_request() -> (
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
        toolkit_builders=[_FakeToolkitBuilder(name="search")],
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
        room.messaging.emit_message(
            RoomMessage(
                from_participant_id=caller.id,
                type="get_thread_toolkit_builders",
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
