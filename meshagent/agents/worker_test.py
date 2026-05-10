from typing import Optional

import pytest

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.worker import SubmitWork
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.worker import Worker
from meshagent.openai.tools.responses_adapter import OpenAIResponsesAdapter


class _FakeParticipant:
    def __init__(self, *, name: str, participant_id: str):
        self._name = name
        self.id = participant_id

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: Optional[dict] = None):
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_FakeElement"] = []

    def get_attribute(self, key: str):
        return self._attributes.get(key)

    def set_attribute(self, key: str, value) -> None:
        self._attributes[key] = value

    def get_children(self):
        return self._children

    def append_child(self, tag_name: str, attributes: Optional[dict] = None):
        child = _FakeElement(tag_name=tag_name, attributes=attributes)
        self._children.append(child)
        return child


class _FakeThreadRoot:
    def __init__(self):
        self._members = _FakeElement(tag_name="members")

    def get_children_by_tag_name(self, tag_name: str):
        if tag_name == "members":
            return [self._members]
        return []

    @property
    def members(self) -> _FakeElement:
        return self._members


class _FakeThreadDocument:
    def __init__(self):
        self.root = _FakeThreadRoot()

    @property
    def member_names(self) -> list[str]:
        names: list[str] = []
        for child in self.root.members.get_children():
            if child.tag_name != "member":
                continue
            name = child.get_attribute("name")
            if isinstance(name, str):
                names.append(name)
        return names


class _FakeThreadListDocument:
    def __init__(self):
        self.root = _FakeElement(tag_name="thread_list")

    def get_state(self) -> bytes:
        return b"thread-list-state"


class _FakeSync:
    def __init__(self):
        self.document = _FakeThreadListDocument()
        self.open_calls: list[dict] = []
        self.close_calls: list[str] = []
        self.sync_calls: list[dict] = []

    async def open(self, *, path: str, schema=None):
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str):
        self.close_calls.append(path)

    async def sync(self, *, path: str, data: bytes):
        self.sync_calls.append({"path": path, "data": data})


class _FakeStorage:
    def __init__(self, *, existing_paths: Optional[set[str]] = None):
        self._existing_paths = set(existing_paths or set())
        self.exists_calls: list[str] = []

    async def exists(self, *, path: str) -> bool:
        self.exists_calls.append(path)
        return path in self._existing_paths


class _FakeProtocol:
    def __init__(self, *, token: str | None = None):
        self.token = token


class _FakeRoom:
    def __init__(
        self,
        *,
        existing_paths: Optional[set[str]] = None,
        token: str | None = None,
    ):
        self.local_participant = _FakeParticipant(
            name="assistant",
            participant_id="assistant-id",
        )
        self.sync = _FakeSync()
        self.storage = _FakeStorage(existing_paths=existing_paths)
        self.protocol = _FakeProtocol(token=token)


class _FakeThreadAdapter:
    instances: list["_FakeThreadAdapter"] = []

    def __init__(self, *, room, path: str):
        del room
        self.path = path
        self.thread = _FakeThreadDocument()
        self.started = False
        self.stopped = False
        self.appended = False
        self.writes: list[tuple[str, str]] = []
        self.events: list[dict] = []
        _FakeThreadAdapter.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def append_messages(self, *, context: AgentSessionContext) -> None:
        del context
        self.appended = True

    def write_text_message(self, *, text: str, participant) -> None:
        self.writes.append((text, participant))

    def push(self, *, event: dict) -> None:
        self.events.append(event)


class _FakeLLMAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str = "release planning"):
        self.generated_thread_name = generated_thread_name
        self.calls: list[dict] = []

    def default_model(self) -> str:
        return "test-model"

    async def create_response(
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
        options: Optional[dict] = None,
    ):
        del context, caller, toolkits, steering_callback, on_behalf_of, options
        self.calls.append(
            {
                "output_schema": output_schema,
                "event_handler": event_handler,
                "model": model,
            }
        )

        if output_schema is not None:
            return {"thread_name": self.generated_thread_name}

        if event_handler is not None:
            event_handler({"type": "response.output_text.done", "text": "done"})

        return "assistant response"


class _FakeDecisionLLMAdapter(LLMAdapter):
    def __init__(self, *, summary: str):
        self._summary = summary
        self.calls: list[dict] = []

    def default_model(self) -> str:
        return "decision-model"

    async def create_response(
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
        options: Optional[dict] = None,
    ):
        del (
            context,
            caller,
            toolkits,
            event_handler,
            steering_callback,
            on_behalf_of,
            options,
        )
        self.calls.append({"output_schema": output_schema, "model": model})
        return {"summary": self._summary}


class _FakeHostedToolkit:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_worker_auto_threading_creates_thread_and_index_entry() -> None:
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter(generated_thread_name="Release Planning")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="auto",
        thread_dir="/threads",
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    context = AgentSessionContext(system_role=None)
    result = await worker.process_message(
        chat_context=context,
        message={"prompt": "Plan the release"},
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 2
    assert adapter.calls[0]["output_schema"] is not None
    assert adapter.calls[1]["output_schema"] is None
    assert adapter.calls[1]["event_handler"] is not None

    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/release-planning.thread"
    assert thread_adapter.started
    assert thread_adapter.appended
    assert thread_adapter.stopped
    assert thread_adapter.thread.member_names == ["assistant"]
    assert thread_adapter.writes == [("```text\nPlan the release\n```", "worker")]
    assert [event["type"] for event in thread_adapter.events] == [
        "response.output_text.done"
    ]

    assert room.sync.open_calls[0]["path"] == "/threads/index.threadl"
    assert room.sync.sync_calls[0]["path"] == "/threads/index.threadl"
    assert room.sync.sync_calls[0]["data"] == b"dGhyZWFkLWxpc3Qtc3RhdGU="
    assert room.sync.close_calls == ["/threads/index.threadl"]
    entries = room.sync.document.root.get_children()
    assert len(entries) == 1
    assert entries[0].get_attribute("path") == "/threads/release-planning.thread"


@pytest.mark.asyncio
async def test_worker_auto_threading_uses_next_available_path_when_base_exists():
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter(generated_thread_name="Release Planning")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="auto",
        thread_dir="/threads",
    )
    room = _FakeRoom(existing_paths={"/threads/release-planning.thread"})
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={"prompt": "Plan the release"},
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/release-planning 2.thread"
    assert room.sync.open_calls[0]["path"] == "/threads/index.threadl"
    entries = room.sync.document.root.get_children()
    assert len(entries) == 1
    assert entries[0].get_attribute("path") == "/threads/release-planning 2.thread"


@pytest.mark.asyncio
async def test_worker_manual_threading_uses_message_path_without_index() -> None:
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter()
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="manual",
        thread_dir="/threads",
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={"prompt": "Manual task", "path": "/threads/manual.thread"},
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 1
    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/manual.thread"
    assert thread_adapter.thread.member_names == ["assistant"]
    assert thread_adapter.writes == [("```text\nManual task\n```", "worker")]
    assert room.sync.open_calls == []
    assert room.sync.close_calls == []


@pytest.mark.asyncio
async def test_worker_initial_message_summary_mode_uses_decision_adapter() -> None:
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter()
    decision_adapter = _FakeDecisionLLMAdapter(summary="Incoming webhook event summary")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="manual",
        thread_dir="/threads",
        initial_message_mode="summary",
        initial_message_from="Webhook",
        decision_model="gpt-5.2-mini",
        decision_llm_adapter=decision_adapter,
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={
            "prompt": "Process webhook payload",
            "path": "/threads/manual.thread",
        },
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(decision_adapter.calls) == 1
    assert decision_adapter.calls[0]["output_schema"] is not None
    assert decision_adapter.calls[0]["model"] == "gpt-5.2-mini"
    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.writes == [("Incoming webhook event summary", "Webhook")]


@pytest.mark.asyncio
async def test_worker_initial_message_none_mode_skips_thread_write() -> None:
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter()
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="manual",
        thread_dir="/threads",
        initial_message_mode="none",
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={
            "prompt": "Manual task",
            "path": "/threads/manual.thread",
        },
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.writes == []


@pytest.mark.asyncio
async def test_worker_none_threading_does_not_open_thread_adapter() -> None:
    _FakeThreadAdapter.instances.clear()
    adapter = _FakeLLMAdapter()
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        threading_mode="none",
        thread_dir="/threads",
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={"prompt": "No persistence"},
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 1
    assert _FakeThreadAdapter.instances == []
    assert room.sync.open_calls == []
    assert room.sync.close_calls == []


@pytest.mark.asyncio
async def test_worker_start_builds_room_bound_toolkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosted_toolkit = _FakeHostedToolkit()
    started: dict[str, object] = {}

    async def fake_single_room_start(self, *, room) -> None:
        self._room = room

    async def fake_single_room_stop(self) -> None:
        self._room = None

    async def fake_start_hosted_toolkit(*, room, toolkit):
        started["room"] = room
        started["toolkit"] = toolkit
        return hosted_toolkit

    async def fake_run(self, *, room) -> None:
        del room

    monkeypatch.setattr(
        "meshagent.agents.agent.SingleRoomAgent.start",
        fake_single_room_start,
    )
    monkeypatch.setattr(
        "meshagent.agents.agent.SingleRoomAgent.stop",
        fake_single_room_stop,
    )
    monkeypatch.setattr(
        "meshagent.agents.worker._start_hosted_toolkit",
        fake_start_hosted_toolkit,
    )
    monkeypatch.setattr(Worker, "run", fake_run)

    room = _FakeRoom()
    worker = Worker(
        queue="tasks",
        llm_adapter=_FakeLLMAdapter(),
        toolkit_name="assistant_tools",
    )

    await worker.start(room=room)  # type: ignore[arg-type]

    toolkit = started["toolkit"]
    assert started["room"] is room
    assert worker._worker_toolkit is toolkit
    assert toolkit.name == "assistant_tools"
    assert len(toolkit.tools) == 1
    tool = toolkit.tools[0]
    assert isinstance(tool, SubmitWork)
    assert tool.room is room
    assert tool.name == "queue_assistant_task"

    await worker.stop()

    assert hosted_toolkit.stopped is True
    assert worker._worker_toolkit is None


def test_worker_bind_runtime_credentials_swaps_distinct_adapters() -> None:
    adapter = OpenAIResponsesAdapter(model="gpt-4o")
    decision_adapter = OpenAIResponsesAdapter(model="gpt-4o-mini")
    thread_name_adapter = OpenAIResponsesAdapter(model="gpt-4.1")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        decision_llm_adapter=decision_adapter,
        threading_mode="auto",
        thread_name_adapter=thread_name_adapter,
    )

    worker.bind_runtime_credentials(room=_FakeRoom(token="service-token"))

    assert worker._llm_adapter is not adapter
    assert worker._decision_llm_adapter is not decision_adapter
    assert worker._threading_helper._thread_name_adapter is not thread_name_adapter
    assert worker._llm_adapter._api_key == "service-token"
    assert worker._decision_llm_adapter is not None
    assert worker._decision_llm_adapter._api_key == "service-token"
    assert worker._threading_helper._thread_name_adapter is not None
    assert worker._threading_helper._thread_name_adapter._api_key == "service-token"


def test_worker_bind_runtime_credentials_preserves_main_adapter_aliases() -> None:
    adapter = OpenAIResponsesAdapter(model="gpt-4o")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        decision_llm_adapter=adapter,
        threading_mode="auto",
    )

    worker.bind_runtime_credentials(room=_FakeRoom(token="service-token"))

    assert worker._llm_adapter is not adapter
    assert worker._decision_llm_adapter is worker._llm_adapter
    assert worker._threading_helper._thread_name_adapter is worker._llm_adapter
    assert worker._llm_adapter._api_key == "service-token"


def test_worker_bind_runtime_credentials_preserves_decision_thread_alias() -> None:
    adapter = OpenAIResponsesAdapter(model="gpt-4o")
    decision_adapter = OpenAIResponsesAdapter(model="gpt-4o-mini")
    worker = Worker(
        queue="tasks",
        llm_adapter=adapter,
        decision_llm_adapter=decision_adapter,
        threading_mode="auto",
        thread_name_adapter=decision_adapter,
    )

    worker.bind_runtime_credentials(room=_FakeRoom(token="service-token"))

    assert worker._llm_adapter is not adapter
    assert worker._decision_llm_adapter is not decision_adapter
    assert worker._decision_llm_adapter is not None
    assert worker._threading_helper._thread_name_adapter is worker._decision_llm_adapter
    assert worker._decision_llm_adapter._api_key == "service-token"
