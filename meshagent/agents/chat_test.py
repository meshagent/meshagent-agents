import asyncio
from typing import Optional, Literal
from unittest import mock
import uuid

import aiohttp
import pytest
from aiohttp.client_reqrep import RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import BaseModel
from yarl import URL

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.chat import (
    ChatBot,
    ChatBotClient,
    ChatThreadContext,
)
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.thread_schema import thread_list_schema, thread_schema
from meshagent.api import RoomException
from meshagent.api.chan import Chan
from meshagent.api.messaging import JsonContent
from meshagent.api.participant import Participant
from meshagent.openai.tools.responses_adapter import MCPToolkitBuilder
from meshagent.tools import ToolContext, ToolkitBuilder, Toolkit


class _FakeStorage:
    def __init__(self, *, existing_paths: Optional[set[str]] = None):
        self._existing_paths = existing_paths or set()

    async def exists(self, *, path: str) -> bool:
        return path in self._existing_paths

    async def list(self, *, path: str):
        del path
        return []


class _FakeMessaging:
    def __init__(self):
        self.sent_messages: list[dict] = []
        self.handlers: dict[str, object] = {}
        self.enabled = False

    def get_participants(self):
        return []

    def on(self, event: str, handler) -> None:
        self.handlers[event] = handler

    def off(self, event: str, handler) -> None:
        existing = self.handlers.get(event)
        if existing is handler:
            self.handlers.pop(event, None)

    async def enable(self) -> None:
        self.enabled = True

    def send_message_nowait(self, *, to, type, message):
        self.sent_messages.append({"to": to, "type": type, "message": message})

    async def send_message(
        self,
        *,
        to,
        type,
        message,
        attachment=None,
    ) -> None:
        del attachment
        self.send_message_nowait(to=to, type=type, message=message)


class _FakeAgents:
    async def list_toolkits(
        self,
        *,
        participant_id: Optional[str] = None,
        participant_name: Optional[str] = None,
    ):
        del participant_id
        del participant_name
        return []


class _FakeThreadListElement:
    def __init__(self, *, tag_name: str, attributes: dict[str, str]):
        self.tag_name = tag_name
        self._attributes = dict(attributes)

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value):
        self._attributes[name] = value


class _FakeThreadElement:
    def __init__(
        self,
        *,
        tag_name: str,
        attributes: Optional[dict[str, object]] = None,
    ):
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_FakeThreadElement"] = []

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value

    def get_children(self) -> list["_FakeThreadElement"]:
        return [*self._children]

    def get_children_by_tag_name(self, tag_name: str) -> list["_FakeThreadElement"]:
        return [child for child in self._children if child.tag_name == tag_name]

    def append_child(
        self,
        *,
        tag_name: str,
        attributes: Optional[dict[str, object]] = None,
    ) -> "_FakeThreadElement":
        element = _FakeThreadElement(tag_name=tag_name, attributes=attributes)
        self._children.append(element)
        return element


class _FakeThreadDocument:
    def __init__(self):
        self.root = _FakeThreadElement(tag_name="thread")


class _FakeThreadListRoot:
    def __init__(self):
        self._children: list[_FakeThreadListElement] = []

    def get_children(self) -> list[_FakeThreadListElement]:
        return [*self._children]

    def append_child(
        self, *, tag_name: str, attributes: dict[str, str]
    ) -> _FakeThreadListElement:
        element = _FakeThreadListElement(tag_name=tag_name, attributes=attributes)
        self._children.append(element)
        return element


class _FakeThreadListDocument:
    def __init__(self):
        self.root = _FakeThreadListRoot()


class _FakeSync:
    def __init__(self, *, document: Optional[_FakeThreadListDocument] = None):
        self.document = document or _FakeThreadListDocument()
        self.open_calls: list[dict] = []
        self.close_calls: list[str] = []

    async def open(
        self,
        *,
        path: str,
        create: bool = True,
        initial_json: Optional[dict] = None,
        schema=None,
    ):
        del create
        del initial_json
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)


class _FakeLocalParticipant(Participant):
    def __init__(self):
        super().__init__(
            id="assistant-id",
            attributes={"name": "assistant"},
        )
        self.set_attribute_calls: list[tuple[str, object]] = []

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value
        self.set_attribute_calls.append((name, value))


class _FakeRoom:
    def __init__(
        self,
        *,
        existing_paths: Optional[set[str]] = None,
        sync: Optional[_FakeSync] = None,
    ):
        self.local_participant = _FakeLocalParticipant()
        self.storage = _FakeStorage(existing_paths=existing_paths)
        self.messaging = _FakeMessaging()
        self.sync = sync or _FakeSync()
        self.agents = _FakeAgents()


class _FakeQueue:
    def __init__(self):
        self.items = []

    def send_nowait(self, item) -> None:
        self.items.append(item)


class _ExplodingMessageChannel:
    def __init__(self, *, error: Exception):
        self._error = error

    async def recv(self):
        raise self._error


class _BlockingMessageChannel:
    async def recv(self):
        await asyncio.Future()


class _TrackedThreadContext:
    def __init__(self, *, path: str):
        self.path = path
        self.closed = False

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        self.closed = True


def _make_ws_handshake_error(
    *, status: int, headers: dict[str, str] | None = None
) -> aiohttp.WSServerHandshakeError:
    request_headers = CIMultiDictProxy(CIMultiDict())
    url = URL("ws://localhost:8080/openai/v1/responses")
    return aiohttp.WSServerHandshakeError(
        request_info=RequestInfo(
            url=url,
            method="GET",
            headers=request_headers,
            real_url=url,
        ),
        history=(),
        status=status,
        message="Invalid response status",
        headers=CIMultiDictProxy(CIMultiDict(headers or {})),
    )


class _FakeOpenThreadAdapter:
    def make_toolkit(self) -> Toolkit:
        return Toolkit(name="open thread", tools=[])


class _FakeThreadNameAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str):
        self.generated_thread_name = generated_thread_name
        self.calls: list[dict] = []

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
        options: Optional[dict] = None,
    ):
        del context
        del room
        del toolkits
        del event_handler
        del steering_callback
        del on_behalf_of
        del options
        self.calls.append({"output_schema": output_schema, "model": model})
        return {"thread_name": self.generated_thread_name}


class _SessionRequiredContext(AgentSessionContext):
    pass


class _SessionRequiredThreadNameAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str):
        self.generated_thread_name = generated_thread_name
        self.last_messages: list[dict] = []
        self.last_context_type: type[AgentSessionContext] | None = None

    def default_model(self) -> str:
        return "thread-name-model"

    def create_session(self) -> AgentSessionContext:
        return _SessionRequiredContext(system_role=None)

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
        options: Optional[dict] = None,
    ):
        del room
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        self.last_context_type = type(context)
        self.last_messages = [*context.messages]
        if not isinstance(context, _SessionRequiredContext):
            raise RuntimeError("expected adapter-created session context")
        return {"thread_name": self.generated_thread_name}


class _CaptureChatAdapter(LLMAdapter):
    def __init__(self):
        self.last_messages: list[dict] = []

    def default_model(self) -> str:
        return "chat-model"

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
        options: Optional[dict] = None,
    ):
        del room
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        self.last_messages = [*context.messages]
        return "ok"


class _ShouldReplyAdapter(LLMAdapter):
    def __init__(self, *, response):
        self.response = response

    def default_model(self) -> str:
        return "decision-model"

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
        options: Optional[dict] = None,
    ):
        del context
        del room
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        return self.response


class _ExampleToolkitConfig(BaseModel):
    name: Literal["example"] = "example"
    enabled: bool = False


class _ExampleToolkitBuilder(ToolkitBuilder):
    def __init__(self):
        super().__init__(name="example", type=_ExampleToolkitConfig)

    async def make(self, *, room, model: str, config: _ExampleToolkitConfig) -> Toolkit:
        del room
        del model
        del config
        return Toolkit(name="example", tools=[])


class _ChatBotWithToolBuilders(ChatBot):
    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return [_ExampleToolkitBuilder()]


class _ChatBotWithMCPToolkitBuilder(ChatBot):
    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return [MCPToolkitBuilder()]


class _ChatBotAlwaysReplies(ChatBot):
    async def should_reply(
        self,
        *,
        has_more_than_one_other_user: bool,
        online: list[Participant],
        context: ChatThreadContext,
        toolkits: list[Toolkit],
        from_user: Participant,
    ) -> bool:
        del has_more_than_one_other_user
        del online
        del context
        del toolkits
        del from_user
        return True


@pytest.mark.asyncio
async def test_chatbot_client_send_requests_server_side_store_without_touching_doc():
    class _ExplodingDocument:
        @property
        def root(self):
            raise AssertionError("chat client should not write directly to the thread")

    room = _FakeRoom()
    client = ChatBotClient(
        room=room,
        participant_name="assistant",
        thread_path="/threads/test.thread",
    )
    client._participant = Participant(  # type: ignore[assignment]
        id="assistant-id",
        attributes={"name": "assistant"},
    )
    client._doc = _ExplodingDocument()

    await client.send(
        text="hello",
        attachments=[{"path": "uploads/report.pdf"}],
    )

    assert room.messaging.sent_messages == [
        {
            "to": client._participant,
            "type": "chat",
            "message": {
                "text": "hello",
                "path": "/threads/test.thread",
                "tools": [],
                "attachments": [{"path": "uploads/report.pdf"}],
                "store": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_send_and_save_chat_uses_thread_adapter_write_text_message() -> None:
    class _RecordingThreadAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.path = "/threads/test.thread"

        def write_text_message(
            self,
            *,
            text: str,
            participant,
            attachments=None,
        ) -> None:
            self.calls.append(
                {
                    "text": text,
                    "participant": participant,
                    "attachments": attachments,
                }
            )

    room = _FakeRoom()
    bot = ChatBot(llm_adapter=_CaptureChatAdapter())
    bot._room = room
    adapter = _RecordingThreadAdapter()
    recipient = Participant(id="caller-id", attributes={"name": "alice"})

    await bot._send_and_save_chat(
        thread_adapter=adapter,  # type: ignore[arg-type]
        to=recipient,  # type: ignore[arg-type]
        id="message-1",
        text="hello",
        thread_attributes={"path": "/threads/test.thread"},
    )

    assert room.messaging.sent_messages == [
        {
            "to": recipient,
            "type": "chat",
            "message": {"path": "/threads/test.thread", "text": "hello"},
        }
    ]
    assert adapter.calls == [
        {
            "text": "hello",
            "participant": room.local_participant,
            "attachments": None,
        }
    ]

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: Participant
    ) -> list[Toolkit]:
        del thread_context
        del participant
        return []


async def _new_thread_tool(bot: ChatBot):
    toolkits = await bot.get_exposed_toolkits()
    chatbot_toolkit = next(
        toolkit
        for toolkit in toolkits
        if any(tool.name == "new_thread" for tool in toolkit.tools)
    )
    return next(tool for tool in chatbot_toolkit.tools if tool.name == "new_thread")


def _stub_new_thread_member_seed(bot: ChatBot) -> mock.AsyncMock:
    stub = mock.AsyncMock()
    bot._seed_new_thread_members = stub  # type: ignore[method-assign]
    return stub


async def _chat_toolkit(bot: ChatBot):
    toolkits = await bot.get_exposed_toolkits()
    return next(toolkit for toolkit in toolkits if toolkit.name == "chat")


def _assert_uuid_thread_path(*, path: str, prefix: str) -> None:
    assert path.startswith(prefix)
    assert path.endswith(".thread")
    basename = path[len(prefix) : -len(".thread")]
    parsed = uuid.UUID(basename)
    assert str(parsed) == basename


@pytest.mark.asyncio
async def test_new_thread_tool_creates_named_thread_and_queues_message() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Release Planning / Q1")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    seeded_members = _stub_new_thread_member_seed(bot)

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    tools_schema = tool.input_schema["properties"]["message"]["properties"]["tools"]
    assert len(tools_schema["anyOf"]) == 2
    assert tools_schema["anyOf"][0]["type"] == "array"
    assert tools_schema["anyOf"][0]["items"]["type"] == "object"
    assert tools_schema["anyOf"][0]["items"]["additionalProperties"] is False
    assert tools_schema["anyOf"][0]["items"]["properties"] == {}
    assert tools_schema["anyOf"][0]["items"]["required"] == []
    assert tools_schema["anyOf"][1] == {"type": "null"}
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={
            "text": "Plan the Q1 release milestones",
            "attachments": [{"path": "uploads/plan.md"}],
        },
    )

    assert isinstance(result, JsonContent)
    result_path = result.json["path"]
    assert result.json["name"] == "Release Planning Q1"
    _assert_uuid_thread_path(path=result_path, prefix=".threads/assistant/")

    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.type == "chat"
    assert queued.message["path"] == result_path
    assert queued.message["text"] == "Plan the Q1 release milestones"
    assert queued.message["attachments"] == [{"path": "uploads/plan.md"}]
    assert queued.message["store"] is True
    assert not queued.result.done()

    seeded_members.assert_awaited_once()
    assert seeded_members.await_args.kwargs["path"] == result_path
    assert seeded_members.await_args.kwargs["members"][0] is room.local_participant
    assert seeded_members.await_args.kwargs["members"][1] is context.caller

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["output_schema"] is not None


@pytest.mark.asyncio
async def test_new_thread_tool_uses_thread_dir_for_guid_path() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Release Planning")
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room
    _stub_new_thread_member_seed(bot)

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    tools_schema = tool.input_schema["properties"]["message"]["properties"]["tools"]
    assert len(tools_schema["anyOf"]) == 2
    assert tools_schema["anyOf"][0]["type"] == "array"
    assert tools_schema["anyOf"][0]["items"]["type"] == "object"
    assert tools_schema["anyOf"][0]["items"]["additionalProperties"] is False
    assert tools_schema["anyOf"][0]["items"]["properties"] == {}
    assert tools_schema["anyOf"][0]["items"]["required"] == []
    assert tools_schema["anyOf"][1] == {"type": "null"}
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={"text": "Plan the release"},
    )

    assert isinstance(result, JsonContent)
    result_path = result.json["path"]
    _assert_uuid_thread_path(path=result_path, prefix="custom/")
    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.message["path"] == result_path
    assert queued.message["text"] == "Plan the release"
    assert queued.message["attachments"] == []
    assert queued.message["store"] is True


@pytest.mark.asyncio
async def test_new_thread_tool_accepts_empty_tools_without_builder_schema() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="No Builder Tools")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    _stub_new_thread_member_seed(bot)

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    tools_schema = tool.input_schema["properties"]["message"]["properties"]["tools"]
    assert len(tools_schema["anyOf"]) == 2
    assert tools_schema["anyOf"][0]["type"] == "array"
    assert tools_schema["anyOf"][0]["items"]["type"] == "object"
    assert tools_schema["anyOf"][0]["items"]["additionalProperties"] is False
    assert tools_schema["anyOf"][0]["items"]["properties"] == {}
    assert tools_schema["anyOf"][0]["items"]["required"] == []
    assert tools_schema["anyOf"][1] == {"type": "null"}
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={
            "text": "Thread with empty tools",
            "attachments": [],
            "tools": [],
        },
    )

    assert isinstance(result, JsonContent)
    _assert_uuid_thread_path(
        path=result.json["path"],
        prefix=".threads/assistant/",
    )
    assert result.json["name"] == "No Builder Tools"
    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.message["tools"] == []
    assert queued.message["store"] is True


@pytest.mark.asyncio
async def test_new_thread_tool_schema_supports_mcp_toolkit_builder() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="MCP")
    bot = _ChatBotWithMCPToolkitBuilder(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    tool = await _new_thread_tool(bot)
    tools_schema = tool.input_schema["properties"]["message"]["properties"]["tools"]
    items_schema = tools_schema["anyOf"][0]["items"]
    headers_schema = items_schema["$defs"]["MCPServer"]["properties"]["headers"][
        "anyOf"
    ][0]

    assert headers_schema["type"] == "array"
    assert headers_schema["items"]["$ref"] == "#/$defs/Header"
    assert items_schema["$defs"]["Header"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_new_thread_tool_accepts_tools_with_toolkit_builder_schema() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Builder Thread")
    bot = _ChatBotWithToolBuilders(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    _stub_new_thread_member_seed(bot)

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    tools_schema = tool.input_schema["properties"]["message"]["properties"]["tools"]
    assert len(tools_schema["anyOf"]) == 2
    assert tools_schema["anyOf"][0]["type"] == "array"
    assert tools_schema["anyOf"][0]["items"]["type"] == "object"
    assert tools_schema["anyOf"][0]["items"]["properties"]["name"]["type"] == "string"
    assert tools_schema["anyOf"][1] == {"type": "null"}
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={
            "text": "Thread with tool config",
            "tools": [{"name": "example", "enabled": True}],
        },
    )

    assert isinstance(result, JsonContent)
    _assert_uuid_thread_path(
        path=result.json["path"],
        prefix=".threads/assistant/",
    )
    assert result.json["name"] == "Builder Thread"
    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.message["tools"] == [{"name": "example", "enabled": True}]
    assert queued.message["store"] is True


@pytest.mark.asyncio
async def test_new_thread_tool_uses_adapter_created_context_for_thread_naming() -> None:
    adapter = _SessionRequiredThreadNameAdapter(generated_thread_name="Adapter Context")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    _stub_new_thread_member_seed(bot)

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
        caller_context={
            "chat": {
                "messages": [{"role": "user", "content": "prior context"}],
                "system_role": None,
                "previous_messages": [],
                "previous_response_id": None,
            }
        },
    )
    result = await tool.execute(
        context=context,
        message={"text": "Name this from adapter context"},
    )

    assert isinstance(result, JsonContent)
    _assert_uuid_thread_path(
        path=result.json["path"],
        prefix=".threads/assistant/",
    )
    assert result.json["name"] == "Adapter Context"
    assert adapter.last_context_type is _SessionRequiredContext
    assert [m.get("content") for m in adapter.last_messages] == [
        "prior context",
        "Name this from adapter context",
    ]


@pytest.mark.asyncio
async def test_new_thread_tool_records_friendly_name_in_thread_list() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Friendly Plan Name")
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room
    _stub_new_thread_member_seed(bot)

    doc = _FakeThreadListDocument()
    bot._thread_list_document = doc
    bot._thread_list_path = "custom/index.threadl"

    queue = _FakeQueue()

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={"text": "Plan this friendly thread"},
    )

    path = result.json["path"]
    assert result.json["name"] == "Friendly Plan Name"
    _assert_uuid_thread_path(path=path, prefix="custom/")
    entries = doc.root.get_children()
    assert len(entries) == 1
    assert entries[0].get_attribute("name") == "Friendly Plan Name"
    assert entries[0].get_attribute("path") == path


@pytest.mark.asyncio
async def test_seed_new_thread_members_adds_agent_and_sender_without_duplicates() -> (
    None
):
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    thread = _FakeThreadDocument()
    members = thread.root.append_child(tag_name="members")
    members.append_child(tag_name="member", attributes={"name": "assistant"})

    sync = mock.Mock()
    sync.open = mock.AsyncMock(return_value=thread)
    sync.close = mock.AsyncMock()

    room = _FakeRoom(sync=sync)
    bot._room = room

    sender = Participant(id="caller-id", attributes={"name": "alice"})

    await bot._seed_new_thread_members(
        path=".threads/assistant/test.thread",
        members=[room.local_participant, sender, sender],
    )

    assert sync.open.await_args.kwargs == {
        "path": ".threads/assistant/test.thread",
        "schema": thread_schema,
    }
    sync.close.assert_awaited_once_with(path=".threads/assistant/test.thread")
    assert [child.get_attribute("name") for child in members.get_children()] == [
        "assistant",
        "alice",
    ]


@pytest.mark.asyncio
async def test_thread_list_tools_only_present_when_thread_dir_is_set() -> None:
    adapter = _CaptureChatAdapter()

    bot_without_thread_dir = ChatBot(llm_adapter=adapter)
    room_without_thread_dir = _FakeRoom()
    bot_without_thread_dir._room = room_without_thread_dir
    toolkit_without_thread_dir = await _chat_toolkit(bot_without_thread_dir)
    names_without = {tool.name for tool in toolkit_without_thread_dir.tools}
    assert "list_threads" not in names_without
    assert "grep_thread_list" not in names_without

    bot_with_thread_dir = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room_with_thread_dir = _FakeRoom()
    bot_with_thread_dir._room = room_with_thread_dir
    toolkit_with_thread_dir = await _chat_toolkit(bot_with_thread_dir)
    names_with = {tool.name for tool in toolkit_with_thread_dir.tools}
    assert "list_threads" in names_with
    assert "grep_thread_list" in names_with


@pytest.mark.asyncio
async def test_default_new_threading_mode_enables_thread_list_tools_and_relaxed_remote_validation() -> (
    None
):
    adapter = _CaptureChatAdapter()
    sync = _FakeSync()
    room = _FakeRoom(sync=sync)
    bot = ChatBot(llm_adapter=adapter, threading_mode="default-new")
    bot._room = room

    toolkit = await _chat_toolkit(bot)
    tool_names = {tool.name for tool in toolkit.tools}

    assert toolkit.validation_mode == "content_types"
    assert "list_threads" in tool_names
    assert "grep_thread_list" in tool_names

    await bot._open_thread_list_document()

    assert sync.open_calls == [
        {
            "path": ".threads/assistant/index.threadl",
            "schema": thread_list_schema,
        }
    ]


@pytest.mark.asyncio
async def test_get_thread_toolkits_includes_thread_list_tools_only() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room

    path = "custom/main.thread"
    bot._open_threads[path] = _FakeOpenThreadAdapter()

    thread = mock.Mock()
    thread.root = mock.Mock()
    thread.root.get_elements_by_tag_name.return_value = [mock.Mock()]

    thread_context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=path,
    )
    participant = Participant(id="caller-id", attributes={"name": "alice"})

    toolkits = await bot.get_thread_toolkits(
        thread_context=thread_context,
        participant=participant,
    )
    tool_names = {tool.name for toolkit in toolkits for tool in toolkit.tools}

    assert "list_threads" in tool_names
    assert "grep_thread_list" in tool_names
    assert "ask" not in tool_names
    assert "new_thread" not in tool_names


@pytest.mark.asyncio
async def test_get_thread_toolkits_omits_thread_list_tools_without_thread_dir() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    path = ".threads/assistant/main.thread"
    bot._open_threads[path] = _FakeOpenThreadAdapter()

    thread = mock.Mock()
    thread.root = mock.Mock()
    thread.root.get_elements_by_tag_name.return_value = [mock.Mock()]

    thread_context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=path,
    )
    participant = Participant(id="caller-id", attributes={"name": "alice"})

    toolkits = await bot.get_thread_toolkits(
        thread_context=thread_context,
        participant=participant,
    )
    tool_names = {tool.name for toolkit in toolkits for tool in toolkit.tools}

    assert "list_threads" not in tool_names
    assert "grep_thread_list" not in tool_names


@pytest.mark.asyncio
async def test_list_threads_sorts_by_modified_desc_and_supports_offset_limit() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room

    doc = _FakeThreadListDocument()
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "first",
            "path": "custom/a.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "second",
            "path": "custom/b.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2025-01-01T00:00:00Z",
        },
    )
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "third",
            "path": "custom/c.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2023-01-01T00:00:00Z",
        },
    )
    bot._thread_list_document = doc

    toolkit = await _chat_toolkit(bot)
    list_tool = next(tool for tool in toolkit.tools if tool.name == "list_threads")
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )

    result = await list_tool.execute(context=context, limit=2, offset=1)
    assert isinstance(result, JsonContent)
    assert result.json["sort"] == "modified_at_desc"
    assert result.json["total"] == 3
    assert result.json["offset"] == 1
    assert result.json["limit"] == 2
    names = [entry["name"] for entry in result.json["threads"]]
    assert names == ["first", "third"]
    assert "Use read_file with a thread path" in result.json["read_file_hint"]


@pytest.mark.asyncio
async def test_grep_thread_list_finds_matches_and_mentions_read_file() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room

    doc = _FakeThreadListDocument()
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "Release Plan",
            "path": "custom/release.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2025-01-01T00:00:00Z",
        },
    )
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "Unrelated",
            "path": "custom/other.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-02T00:00:00Z",
        },
    )
    bot._thread_list_document = doc

    toolkit = await _chat_toolkit(bot)
    grep_tool = next(tool for tool in toolkit.tools if tool.name == "grep_thread_list")
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )

    result = await grep_tool.execute(
        context=context, pattern="release", ignore_case=True
    )
    assert isinstance(result, JsonContent)
    assert result.json["total_matches"] == 1
    assert result.json["pattern"] == "release"
    assert result.json["ignore_case"] is True
    assert [entry["name"] for entry in result.json["threads"]] == ["Release Plan"]
    assert "Use read_file with a thread path" in result.json["read_file_hint"]


@pytest.mark.asyncio
async def test_on_chat_received_adds_current_file_context_message() -> None:
    adapter = _CaptureChatAdapter()
    bot = _ChatBotAlwaysReplies(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    bot._open_threads[".threads/main.thread"] = _FakeOpenThreadAdapter()

    thread = mock.Mock()
    thread.root = mock.Mock()
    thread.root.get_children.return_value = []

    thread_context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=".threads/main.thread",
    )
    from_participant = Participant(
        id="caller-id",
        attributes={"name": "alice", "current_file": "docs/plan.md"},
    )

    result = await bot.on_chat_received(
        thread_context=thread_context,
        from_participant=from_participant,
        message={"text": "Summarize this"},
    )

    assert result == "ok"
    assert any(
        message.get("role") == "assistant"
        and message.get("content")
        == "alice is currently viewing the file at the path: docs/plan.md"
        for message in adapter.last_messages
    )


@pytest.mark.asyncio
async def test_on_chat_received_adds_not_viewing_message_when_file_is_closed() -> None:
    adapter = _CaptureChatAdapter()
    bot = _ChatBotAlwaysReplies(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    bot._open_threads[".threads/main.thread"] = _FakeOpenThreadAdapter()

    thread = mock.Mock()
    thread.root = mock.Mock()
    thread.root.get_children.return_value = []

    thread_context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=".threads/main.thread",
    )

    first_participant = Participant(
        id="caller-id",
        attributes={"name": "alice", "current_file": "docs/plan.md"},
    )
    second_participant = Participant(
        id="caller-id",
        attributes={"name": "alice"},
    )

    await bot.on_chat_received(
        thread_context=thread_context,
        from_participant=first_participant,
        message={"text": "First"},
    )
    await bot.on_chat_received(
        thread_context=thread_context,
        from_participant=second_participant,
        message={"text": "Second"},
    )

    assert any(
        message.get("role") == "assistant"
        and message.get("content") == "alice is not currently viewing any files."
        for message in adapter.last_messages
    )


@pytest.mark.asyncio
async def test_on_chat_received_updates_thread_index_modified_at() -> None:
    adapter = _CaptureChatAdapter()
    bot = _ChatBotAlwaysReplies(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom()
    bot._room = room
    bot._open_threads["custom/main.thread"] = _FakeOpenThreadAdapter()

    thread = mock.Mock()
    thread.root = mock.Mock()
    thread.root.get_children.return_value = []

    thread_context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path="custom/main.thread",
    )
    from_participant = Participant(
        id="caller-id",
        attributes={"name": "alice"},
    )

    doc = _FakeThreadListDocument()
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "main",
            "path": "custom/main.thread",
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    bot._thread_list_document = doc

    result = await bot.on_chat_received(
        thread_context=thread_context,
        from_participant=from_participant,
        message={"text": "hello"},
    )

    assert result == "ok"
    entry = doc.root.get_children()[0]
    assert entry.get_attribute("created_at") == "2024-01-01T00:00:00Z"
    assert entry.get_attribute("modified_at") != "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_should_reply_defaults_true_for_unstructured_decision_response() -> None:
    adapter = _ShouldReplyAdapter(response="Indexed reply")
    bot = ChatBot(llm_adapter=adapter)
    bot._room = _FakeRoom()

    thread = _FakeThreadDocument()
    members = thread.root.append_child(tag_name="members")
    members.append_child(tag_name="member", attributes={"name": "assistant"})
    members.append_child(tag_name="member", attributes={"name": "alice"})
    members.append_child(tag_name="member", attributes={"name": "bob"})

    context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=".threads/main.thread",
    )
    from_user = Participant(id="alice-id", attributes={"name": "alice"})
    online = [
        from_user,
        Participant(id="bob-id", attributes={"name": "bob"}),
    ]

    should_reply = await bot.should_reply(
        context=context,
        has_more_than_one_other_user=True,
        toolkits=[],
        from_user=from_user,
        online=online,
    )

    assert should_reply is True


@pytest.mark.asyncio
async def test_should_reply_parses_json_string_decision_response() -> None:
    adapter = _ShouldReplyAdapter(
        response='{"reasoning":"waiting on bob","expecting_assistant_reply":false,"next_user":"bob"}'
    )
    bot = ChatBot(llm_adapter=adapter)
    bot._room = _FakeRoom()

    thread = _FakeThreadDocument()
    members = thread.root.append_child(tag_name="members")
    members.append_child(tag_name="member", attributes={"name": "assistant"})
    members.append_child(tag_name="member", attributes={"name": "alice"})
    members.append_child(tag_name="member", attributes={"name": "bob"})

    context = ChatThreadContext(
        session=AgentSessionContext(),
        thread=thread,
        path=".threads/main.thread",
    )
    from_user = Participant(id="alice-id", attributes={"name": "alice"})
    online = [
        from_user,
        Participant(id="bob-id", attributes={"name": "bob"}),
    ]

    should_reply = await bot.should_reply(
        context=context,
        has_more_than_one_other_user=True,
        toolkits=[],
        from_user=from_user,
        online=online,
    )

    assert should_reply is False


def test_record_new_thread_in_index_adds_entry() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    doc = _FakeThreadListDocument()
    bot._thread_list_document = doc

    path = "custom/release-plan.thread"
    bot._record_new_thread_in_index(path=path)

    children = doc.root.get_children()
    assert len(children) == 1
    entry = children[0]
    assert entry.tag_name == "thread"
    assert entry.get_attribute("name") == "Release Plan"
    assert entry.get_attribute("path") == path
    created_at = entry.get_attribute("created_at")
    modified_at = entry.get_attribute("modified_at")
    assert isinstance(created_at, str)
    assert isinstance(modified_at, str)
    assert created_at.endswith("Z")
    assert modified_at.endswith("Z")
    assert created_at == modified_at


def test_record_new_thread_in_index_uses_new_chat_for_guid_path() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    doc = _FakeThreadListDocument()
    bot._thread_list_document = doc

    path = "custom/123e4567-e89b-12d3-a456-426614174000.thread"
    bot._record_new_thread_in_index(path=path)

    entry = doc.root.get_children()[0]
    assert entry.get_attribute("name") == "New Chat"


def test_touch_thread_in_index_updates_modified_at() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    doc = _FakeThreadListDocument()
    path = "custom/main.thread"
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "main",
            "path": path,
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        },
    )
    bot._thread_list_document = doc

    bot._touch_thread_in_index(path=path)

    entry = doc.root.get_children()[0]
    assert entry.get_attribute("name") == "main"
    assert entry.get_attribute("created_at") == "2024-01-01T00:00:00Z"
    assert entry.get_attribute("modified_at") != "2024-01-01T00:00:00Z"


def test_touch_thread_in_index_forces_modified_at_forward_when_clock_does_not_advance() -> (
    None
):
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    doc = _FakeThreadListDocument()
    path = "custom/main.thread"
    existing_modified_at = "2024-01-01T00:00:00Z"
    doc.root.append_child(
        tag_name="thread",
        attributes={
            "name": "main",
            "path": path,
            "created_at": existing_modified_at,
            "modified_at": existing_modified_at,
        },
    )
    bot._thread_list_document = doc

    with mock.patch.object(bot, "_utc_now_iso", return_value=existing_modified_at):
        bot._touch_thread_in_index(path=path)

    entry = doc.root.get_children()[0]
    updated_modified_at = entry.get_attribute("modified_at")
    assert isinstance(updated_modified_at, str)
    assert bot._parse_iso_datetime(value=updated_modified_at) > bot._parse_iso_datetime(
        value=existing_modified_at
    )


def test_record_new_thread_in_index_is_noop_without_explicit_thread_dir() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    doc = _FakeThreadListDocument()
    bot._thread_list_document = doc

    bot._record_new_thread_in_index(path=".threads/assistant/test.thread")

    assert len(doc.root.get_children()) == 0


def test_chat_error_message_uses_room_exception_text() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    assert (
        bot._chat_error_message(error=RoomException("Your account is out of credits"))
        == "Your account is out of credits"
    )


def test_chat_error_message_uses_generic_for_non_room_exception() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    assert (
        bot._chat_error_message(error=RuntimeError("boom"))
        == "An unexpected error occured. Please try again later."
    )


def test_chat_error_message_uses_handshake_error_message() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    error = _make_ws_handshake_error(
        status=402,
        headers={
            "X-Meshagent-Error-Message": "Your account is out of credits. Add credits to continue.",
        },
    )
    assert (
        bot._chat_error_message(error=error)
        == "Your account is out of credits. Add credits to continue."
    )


@pytest.mark.asyncio
async def test_thread_list_document_open_close_uses_thread_dir_index_path() -> None:
    adapter = _CaptureChatAdapter()
    sync = _FakeSync()
    room = _FakeRoom(sync=sync)
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    bot._room = room

    await bot._open_thread_list_document()

    assert len(sync.open_calls) == 1
    assert sync.open_calls[0]["path"] == "custom/index.threadl"
    assert sync.open_calls[0]["schema"] is thread_list_schema
    assert bot._thread_list_document is sync.document

    await bot._close_thread_list_document(room=room)

    assert sync.close_calls == ["custom/index.threadl"]
    assert bot._thread_list_document is None
    assert bot._thread_list_path is None


@pytest.mark.asyncio
async def test_chatbot_start_sets_thread_attributes_on_participant() -> None:
    adapter = _CaptureChatAdapter()
    sync = _FakeSync()
    room = _FakeRoom(sync=sync)
    bot = ChatBot(
        llm_adapter=adapter,
        thread_dir="custom",
        threading_mode="default-new",
    )

    with mock.patch.object(
        bot,
        "get_exposed_toolkits",
        new=mock.AsyncMock(return_value=[]),
    ):
        await bot.start(room=room)

    assert (
        room.local_participant.get_attribute("meshagent.chatbot.thread-dir") == "custom"
    )
    assert (
        "meshagent.chatbot.thread-dir",
        "custom",
    ) in room.local_participant.set_attribute_calls
    assert (
        room.local_participant.get_attribute("meshagent.chatbot.thread-list")
        == "custom/index.threadl"
    )
    assert (
        "meshagent.chatbot.thread-list",
        "custom/index.threadl",
    ) in room.local_participant.set_attribute_calls

    await bot.stop()


@pytest.mark.asyncio
async def test_thread_list_document_not_opened_without_explicit_thread_dir() -> None:
    adapter = _CaptureChatAdapter()
    sync = _FakeSync()
    room = _FakeRoom(sync=sync)
    bot = ChatBot(llm_adapter=adapter)
    bot._room = room

    await bot._open_thread_list_document()

    assert sync.open_calls == []


@pytest.mark.asyncio
async def test_spawn_thread_waits_for_cleanup_before_returning() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    path = ".threads/assistant/main.thread"
    tracked_context = _TrackedThreadContext(path=path)
    bot._thread_contexts[path] = tracked_context  # type: ignore[assignment]

    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def _close_thread(*, path: str) -> None:
        del path
        close_started.set()
        await allow_close.wait()

    bot.close_thread = mock.AsyncMock(side_effect=_close_thread)  # type: ignore[method-assign]
    bot._safe_invoke_thread_event = mock.AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(
        bot._spawn_thread(
            path=path,
            messages=_ExplodingMessageChannel(error=RuntimeError("thread ended")),  # type: ignore[arg-type]
        )
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)

    assert close_started.is_set()
    assert tracked_context.closed is True
    assert path not in bot._thread_contexts

    allow_close.set()

    with pytest.raises(RuntimeError, match="thread ended"):
        await task


@pytest.mark.asyncio
async def test_stop_waits_for_thread_cleanup_before_returning() -> None:
    adapter = _CaptureChatAdapter()
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room
    bot._exposed_toolkits = []

    path = ".threads/assistant/main.thread"
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def _close_thread(*, path: str) -> None:
        del path
        close_started.set()
        await allow_close.wait()

    bot.close_thread = mock.AsyncMock(side_effect=_close_thread)  # type: ignore[method-assign]
    bot._close_thread_list_document = mock.AsyncMock()  # type: ignore[method-assign]
    bot._clear_all_thread_statuses = mock.AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(
        bot._spawn_thread(
            path=path,
            messages=_BlockingMessageChannel(),  # type: ignore[arg-type]
        )
    )
    bot._thread_tasks[path] = task
    bot._message_channels[path] = mock.Mock(spec=Chan)

    stop_task = asyncio.create_task(bot.stop())

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=0.05)

    assert close_started.is_set()

    allow_close.set()
    await stop_task
    assert task.done()
