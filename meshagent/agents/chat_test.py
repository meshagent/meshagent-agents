from typing import Optional, Literal

import pytest
from pydantic import BaseModel

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.chat import ChatBot
from meshagent.agents.context import AgentSessionContext
from meshagent.api.messaging import JsonContent
from meshagent.api.participant import Participant
from meshagent.tools import ToolContext, ToolkitBuilder, Toolkit


class _FakeStorage:
    def __init__(self, *, existing_paths: Optional[set[str]] = None):
        self._existing_paths = existing_paths or set()

    async def exists(self, *, path: str) -> bool:
        return path in self._existing_paths


class _FakeRoom:
    def __init__(self, *, existing_paths: Optional[set[str]] = None):
        self.local_participant = Participant(
            id="assistant-id",
            attributes={"name": "assistant"},
        )
        self.storage = _FakeStorage(existing_paths=existing_paths)


class _FakeThreadAdapter:
    def __init__(self):
        self.writes: list[tuple[str, str, list[dict[str, str]]]] = []
        self.paths: list[str] = []

    def write_text_message(
        self,
        *,
        text: str,
        participant: Participant | str,
        attachments: Optional[list[dict]] = None,
    ) -> None:
        if isinstance(participant, str):
            participant_name = participant
        else:
            participant_name = participant.get_attribute("name") or ""
        normalized_attachments: list[dict[str, str]] = []
        if attachments is not None:
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                path = attachment.get("path")
                if isinstance(path, str) and path.strip() != "":
                    normalized_attachments.append({"path": path.strip()})
        self.writes.append((text, participant_name, normalized_attachments))


class _FakeQueue:
    def __init__(self):
        self.items = []

    def send_nowait(self, item) -> None:
        self.items.append(item)


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
        model=None,
        on_behalf_of=None,
        options: Optional[dict] = None,
    ):
        del context
        del room
        del toolkits
        del event_handler
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
        model=None,
        on_behalf_of=None,
        options: Optional[dict] = None,
    ):
        del room
        del toolkits
        del output_schema
        del event_handler
        del model
        del on_behalf_of
        del options
        self.last_context_type = type(context)
        self.last_messages = [*context.messages]
        if not isinstance(context, _SessionRequiredContext):
            raise RuntimeError("expected adapter-created session context")
        return {"thread_name": self.generated_thread_name}


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


async def _new_thread_tool(bot: ChatBot):
    toolkits = await bot.get_exposed_toolkits()
    chatbot_toolkit = next(
        toolkit
        for toolkit in toolkits
        if any(tool.name == "new_thread" for tool in toolkit.tools)
    )
    return next(tool for tool in chatbot_toolkit.tools if tool.name == "new_thread")


@pytest.mark.asyncio
async def test_new_thread_tool_creates_named_thread_and_queues_message() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Release Planning / Q1")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    fake_thread = _FakeThreadAdapter()
    queue = _FakeQueue()

    async def _open_thread(*, path: str):
        fake_thread.paths.append(path)
        return fake_thread

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot.open_thread = _open_thread  # type: ignore[method-assign]
    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
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
    assert result.json == {"path": ".threads/assistant/release-planning-q1.thread"}
    assert fake_thread.paths == [".threads/assistant/release-planning-q1.thread"]
    assert fake_thread.writes == [
        (
            "Plan the Q1 release milestones",
            "alice",
            [{"path": "uploads/plan.md"}],
        )
    ]

    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.type == "chat"
    assert queued.message["path"] == ".threads/assistant/release-planning-q1.thread"
    assert queued.message["text"] == "Plan the Q1 release milestones"
    assert queued.message["attachments"] == [{"path": "uploads/plan.md"}]
    assert not queued.result.done()

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["output_schema"] is not None


@pytest.mark.asyncio
async def test_new_thread_tool_uses_thread_dir_and_suffixes_existing_path() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Release Planning")
    bot = ChatBot(llm_adapter=adapter, thread_dir="custom")
    room = _FakeRoom(existing_paths={"custom/release-planning.thread"})
    bot._room = room

    fake_thread = _FakeThreadAdapter()
    queue = _FakeQueue()

    async def _open_thread(*, path: str):
        fake_thread.paths.append(path)
        return fake_thread

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot.open_thread = _open_thread  # type: ignore[method-assign]
    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
    context = ToolContext(
        room=room,
        caller=Participant(id="caller-id", attributes={"name": "alice"}),
    )
    result = await tool.execute(
        context=context,
        message={"text": "Plan the release"},
    )

    assert isinstance(result, JsonContent)
    assert result.json == {"path": "custom/release-planning 2.thread"}
    assert fake_thread.paths == ["custom/release-planning 2.thread"]
    assert fake_thread.writes == [("Plan the release", "alice", [])]


@pytest.mark.asyncio
async def test_new_thread_tool_accepts_empty_tools_without_builder_schema() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="No Builder Tools")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    fake_thread = _FakeThreadAdapter()
    queue = _FakeQueue()

    async def _open_thread(*, path: str):
        fake_thread.paths.append(path)
        return fake_thread

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot.open_thread = _open_thread  # type: ignore[method-assign]
    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
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
    assert result.json == {"path": ".threads/assistant/no-builder-tools.thread"}
    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.message["tools"] == []


@pytest.mark.asyncio
async def test_new_thread_tool_accepts_tools_with_toolkit_builder_schema() -> None:
    adapter = _FakeThreadNameAdapter(generated_thread_name="Builder Thread")
    bot = _ChatBotWithToolBuilders(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    fake_thread = _FakeThreadAdapter()
    queue = _FakeQueue()

    async def _open_thread(*, path: str):
        fake_thread.paths.append(path)
        return fake_thread

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot.open_thread = _open_thread  # type: ignore[method-assign]
    bot._ensure_thread = _ensure_thread  # type: ignore[method-assign]

    tool = await _new_thread_tool(bot)
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
    assert result.json == {"path": ".threads/assistant/builder-thread.thread"}
    assert len(queue.items) == 1
    queued = queue.items[0]
    assert queued.message["tools"] == [{"name": "example", "enabled": True}]


@pytest.mark.asyncio
async def test_new_thread_tool_uses_adapter_created_context_for_thread_naming() -> None:
    adapter = _SessionRequiredThreadNameAdapter(generated_thread_name="Adapter Context")
    bot = ChatBot(llm_adapter=adapter)
    room = _FakeRoom()
    bot._room = room

    fake_thread = _FakeThreadAdapter()
    queue = _FakeQueue()

    async def _open_thread(*, path: str):
        fake_thread.paths.append(path)
        return fake_thread

    def _ensure_thread(*, path: str):
        del path
        return queue

    bot.open_thread = _open_thread  # type: ignore[method-assign]
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
    assert result.json == {"path": ".threads/assistant/adapter-context.thread"}
    assert adapter.last_context_type is _SessionRequiredContext
    assert [m.get("content") for m in adapter.last_messages] == [
        "prior context",
        "Name this from adapter context",
    ]
