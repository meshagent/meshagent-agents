import re

import pytest
from typing import Optional
import meshagent.agents.llmrunner as llmrunner_module
from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.llmrunner import LLMTaskRunner
from meshagent.agents.task_runner import TaskContext


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


class _FakeRoom:
    def __init__(self, *, existing_paths: Optional[set[str]] = None):
        self.local_participant = _FakeParticipant(
            name="assistant",
            participant_id="assistant-id",
        )
        self.sync = _FakeSync()
        self.storage = _FakeStorage(existing_paths=existing_paths)


class _FakeThreadAdapter:
    instances: list["_FakeThreadAdapter"] = []

    def __init__(self, *, room, path: str):
        del room
        self.path = path
        self.started = False
        self.stopped = False
        self.appended = False
        self.writes: list[tuple[str, str]] = []
        self.events: list[dict] = []
        self.thread = _FakeThreadDocument()
        _FakeThreadAdapter.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def append_messages(self, *, context: AgentSessionContext) -> None:
        del context
        self.appended = True

    def write_text_message(self, *, text: str, participant) -> None:
        if isinstance(participant, str):
            participant_name = participant
        else:
            participant_name = participant.get_attribute("name") or ""

        self.writes.append((text, participant_name))

    def push(self, *, event: dict) -> None:
        self.events.append(event)


class _FakeLLMAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str = "release planning"):
        self.generated_thread_name = generated_thread_name
        self.calls: list[dict] = []

    def default_model(self) -> str:
        return "test-model"

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
        del steering_callback
        self.calls.append(
            {
                "context": context,
                "room": room,
                "toolkits": toolkits,
                "output_schema": output_schema,
                "model": model,
                "on_behalf_of": on_behalf_of,
            }
        )

        if output_schema is not None:
            return {"thread_name": self.generated_thread_name}

        context.turn_count += 1

        if event_handler is not None:
            event_handler({"type": "response.content_part.added"})
            event_handler(
                {"type": "response.output_text.done", "text": "assistant response"}
            )

        return "assistant response"


def _make_context() -> TaskContext:
    room = _FakeRoom()
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    return TaskContext(
        session=AgentSessionContext(system_role=None),
        room=room,
        caller=caller,
        on_behalf_of=None,
        toolkits=[],
    )


def test_llm_task_runner_input_schema_path_only_for_manual_mode() -> None:
    adapter = _FakeLLMAdapter()
    manual_runner = LLMTaskRunner(llm_adapter=adapter, threading_mode="manual")
    auto_runner = LLMTaskRunner(llm_adapter=adapter, threading_mode="auto")
    none_runner = LLMTaskRunner(llm_adapter=adapter, threading_mode="none")

    assert "path" in manual_runner.input_schema["properties"]
    assert "path" not in auto_runner.input_schema["properties"]
    assert "path" not in none_runner.input_schema["properties"]


def test_llm_task_runner_fallback_thread_name_uses_timestamp() -> None:
    adapter = _FakeLLMAdapter()
    runner = LLMTaskRunner(llm_adapter=adapter, threading_mode="auto")
    thread_name = runner._fallback_thread_name(prompt="any prompt")

    assert re.fullmatch(r"thread-\d{8}-\d{6}", thread_name) is not None


@pytest.mark.asyncio
async def test_llm_task_runner_manual_threading_uses_input_path(monkeypatch) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(llmrunner_module, "ThreadAdapter", _FakeThreadAdapter)

    adapter = _FakeLLMAdapter()
    runner = LLMTaskRunner(llm_adapter=adapter, threading_mode="manual")
    context = _make_context()

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    result = await runner.ask(
        context=context,
        arguments={"prompt": "hello", "path": "/threads/manual"},
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 1

    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/manual"
    assert thread_adapter.started
    assert thread_adapter.appended
    assert thread_adapter.stopped
    assert thread_adapter.thread.member_names == ["assistant"]
    assert thread_adapter.writes == [("hello", "caller")]
    assert [event["type"] for event in thread_adapter.events] == [
        "response.content_part.added",
        "response.output_text.done",
    ]
    assert context.room.sync.open_calls == []
    assert context.room.sync.close_calls == []


@pytest.mark.asyncio
async def test_llm_task_runner_auto_threading_generates_path(monkeypatch) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(llmrunner_module, "ThreadAdapter", _FakeThreadAdapter)

    adapter = _FakeLLMAdapter(generated_thread_name="Release Planning / Q1")
    runner = LLMTaskRunner(
        llm_adapter=adapter,
        threading_mode="auto",
        thread_dir="/threads",
    )
    context = _make_context()

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    result = await runner.ask(
        context=context,
        arguments={"prompt": "Plan the Q1 release milestones"},
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 2

    thread_name_call = adapter.calls[0]
    assert thread_name_call["output_schema"] is not None
    assert thread_name_call["output_schema"]["required"] == ["thread_name"]

    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/release-planning-q1.thread"
    assert thread_adapter.thread.member_names == ["assistant"]
    assert thread_adapter.writes == [("Plan the Q1 release milestones", "caller")]
    assert context.room.sync.open_calls[0]["path"] == "/threads/index.threadl"
    assert context.room.sync.sync_calls[0]["path"] == "/threads/index.threadl"
    assert context.room.sync.sync_calls[0]["data"] == b"dGhyZWFkLWxpc3Qtc3RhdGU="
    assert context.room.sync.close_calls == ["/threads/index.threadl"]
    thread_entries = context.room.sync.document.root.get_children()
    assert len(thread_entries) == 1
    assert thread_entries[0].tag_name == "thread"
    assert (
        thread_entries[0].get_attribute("path") == "/threads/release-planning-q1.thread"
    )


@pytest.mark.asyncio
async def test_llm_task_runner_auto_threading_uses_custom_name_rules(
    monkeypatch,
) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(llmrunner_module, "ThreadAdapter", _FakeThreadAdapter)

    adapter = _FakeLLMAdapter(generated_thread_name="custom thread")
    runner = LLMTaskRunner(
        llm_adapter=adapter,
        threading_mode="auto",
        thread_dir="/threads",
        thread_name_rules=["pick a kebab-case name from this task prompt"],
    )
    context = _make_context()

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    result = await runner.ask(
        context=context,
        arguments={"prompt": "Do custom naming"},
    )

    assert result == "assistant response"
    assert len(adapter.calls) == 2
    thread_name_call = adapter.calls[0]
    assert thread_name_call["context"].instructions is not None
    assert (
        "pick a kebab-case name from this task prompt"
        in thread_name_call["context"].instructions
    )
