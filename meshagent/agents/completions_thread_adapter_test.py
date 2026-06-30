import pytest
import asyncio

from meshagent.api import RoomException
from meshagent.agents import (
    completions_thread_adapter as completions_thread_adapter_module,
)
from meshagent.agents.completions_thread_adapter import CompletionsThreadAdapter
from meshagent.agents.thread_adapter import ThreadAdapter


class _FakeElement:
    def __init__(self, tag_name: str) -> None:
        self.tag_name = tag_name
        self.attributes: dict[str, str] = {}
        self.children: list[_FakeElement] = []
        self.values: dict[str, str] = {}

    def append_child(
        self,
        *,
        tag_name: str,
        attributes: dict[str, str],
    ) -> "_FakeElement":
        child = _FakeElement(tag_name)
        child.attributes.update(attributes)
        self.children.append(child)
        return child

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str) -> str | None:
        return self.attributes.get(key)

    def __setitem__(self, key: str, value: str) -> None:
        self.values[key] = value

    def __getitem__(self, key: str) -> str | None:
        return self.values.get(key)


class _FakeRoot:
    def __init__(self, children: list[_FakeElement]) -> None:
        self._children = children

    def get_children(self) -> list[_FakeElement]:
        return self._children

    def get_children_by_tag_name(self, tag_name: str) -> list[_FakeElement]:
        return [child for child in self._children if child.tag_name == tag_name]


class _FakeThread:
    def __init__(self, children: list[_FakeElement]) -> None:
        self.root = _FakeRoot(children)


class _FakeParticipant:
    def get_attribute(self, key: str) -> str | None:
        if key == "name":
            return "assistant"
        return None


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeParticipant()


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_as_current_span(self, name: str) -> _FakeSpan:
        span = _FakeSpan(name)
        self.spans.append(span)
        return span


def test_completions_init_runs_base_thread_adapter_initialization() -> None:
    room = object()
    adapter = CompletionsThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]

    assert adapter._room is room
    assert adapter.path == "/threads/test"
    assert adapter.thread is None
    assert adapter._active_events_by_key == {}


@pytest.mark.asyncio
async def test_completions_stop_awaits_base_stop_before_clearing_active_events(
    monkeypatch,
) -> None:
    adapter = object.__new__(CompletionsThreadAdapter)
    active_event = object()
    adapter._active_events_by_key = {"tool-1": active_event}
    calls: list[dict] = []

    async def _fake_base_stop(self):
        calls.append({"active_events": dict(self._active_events_by_key)})

    monkeypatch.setattr(ThreadAdapter, "stop", _fake_base_stop)

    await adapter.stop()

    assert calls == [{"active_events": {"tool-1": active_event}}]
    assert adapter._active_events_by_key == {}


@pytest.mark.asyncio
async def test_completions_handle_custom_event_resolves_messages_element() -> None:
    adapter = object.__new__(CompletionsThreadAdapter)
    messages = _FakeElement("messages")
    adapter._thread = _FakeThread([_FakeElement("members"), messages])
    adapter._active_events_by_key = {}

    calls: list[dict] = []

    async def _fake_handle_custom_event_for_messages(*, messages, event):
        calls.append({"messages": messages, "event": event})

    adapter._handle_custom_event_for_messages = (  # type: ignore[method-assign]
        _fake_handle_custom_event_for_messages
    )

    event = {"type": "agent.event", "kind": "tool"}
    await adapter.handle_custom_event(event=event)

    assert calls == [{"messages": messages, "event": event}]


@pytest.mark.asyncio
async def test_completions_handle_custom_event_requires_messages_element() -> None:
    adapter = object.__new__(CompletionsThreadAdapter)
    adapter._thread = _FakeThread([_FakeElement("members")])
    adapter._active_events_by_key = {}

    with pytest.raises(
        RoomException,
        match="messages element is missing from thread document",
    ):
        await adapter.handle_custom_event(event={"type": "agent.event", "kind": "tool"})


@pytest.mark.asyncio
async def test_completions_custom_event_keeps_active_element_metadata_current() -> None:
    adapter = object.__new__(CompletionsThreadAdapter)
    adapter._active_events_by_key = {}
    messages = _FakeElement("messages")

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={
            "type": "agent.event",
            "kind": "tool",
            "state": "running",
            "name": "tool.run",
            "correlation_key": "tool-1",
            "preview": "initial preview",
            "details": "initial details",
        },
    )
    event_element = messages.children[0]
    assert adapter._active_events_by_key["tool-1"] is event_element

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={
            "type": "agent.event",
            "kind": "tool",
            "state": "info",
            "name": "tool.info",
            "correlation_key": "tool-1",
            "preview": "updated preview",
            "details": "updated details",
        },
    )
    assert adapter._active_events_by_key["tool-1"] is event_element
    assert event_element.get_attribute("preview") == "updated preview"
    assert event_element.get_attribute("details") == "updated details"

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={
            "type": "agent.event",
            "kind": "tool",
            "state": "info",
            "name": "tool.info",
            "correlation_key": "tool-1",
            "preview": "",
            "details": "",
        },
    )
    assert adapter._active_events_by_key["tool-1"] is event_element
    assert event_element.get_attribute("preview") == "updated preview"
    assert event_element.get_attribute("details") == "updated details"

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={
            "type": "agent.event",
            "kind": "tool",
            "state": "completed",
            "name": "tool.done",
            "correlation_key": "tool-1",
            "preview": "",
            "details": "",
        },
    )
    assert "tool-1" not in adapter._active_events_by_key


@pytest.mark.asyncio
async def test_completions_custom_event_appends_with_python_normalization() -> None:
    adapter = object.__new__(CompletionsThreadAdapter)
    adapter._active_events_by_key = {}
    messages = _FakeElement("messages")

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={"type": "unknown.event", "kind": "tool"},
    )
    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={"type": "agent.event", "kind": "unsupported"},
    )
    assert messages.children == []

    await adapter._handle_custom_event_for_messages(
        messages=messages,
        event={
            "type": "codex.event",
            "kind": "diff",
            "state": "running",
            "method": " apply_patch ",
            "summary": "",
            "headline": " Patch ready ",
            "details": [" line one ", "", 7, "line two"],
            "data": "diff --git a/file b/file",
            "event_key": "patch-1",
        },
    )

    event_element = messages.children[0]
    assert adapter._active_events_by_key["patch-1"] is event_element
    assert event_element.tag_name == "event"
    assert event_element.get_attribute("source") == "codex"
    assert event_element.get_attribute("name") == "codex.event"
    assert event_element.get_attribute("kind") == "diff"
    assert event_element.get_attribute("state") == "running"
    assert event_element.get_attribute("method") == "apply_patch"
    assert event_element.get_attribute("summary") == "apply_patch"
    assert event_element.get_attribute("headline") == "Patch ready"
    assert event_element.get_attribute("details") == "line one\nline two"
    assert event_element.get_attribute("data") == "diff --git a/file b/file"
    assert event_element.get_attribute("created_at") is not None
    assert event_element.get_attribute("updated_at") is not None


@pytest.mark.asyncio
async def test_completions_process_llm_events_mutates_thread_and_exits_on_queue_shutdown(
    monkeypatch,
) -> None:
    tracer = _FakeTracer()
    monkeypatch.setattr(completions_thread_adapter_module, "tracer", tracer)

    adapter = object.__new__(CompletionsThreadAdapter)
    messages = _FakeElement("messages")
    adapter._thread = _FakeThread([messages])
    adapter._room = _FakeRoom()
    adapter._active_events_by_key = {}
    adapter._llm_messages = asyncio.Queue()

    adapter._llm_messages.put_nowait(
        {
            "type": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hel"},
                    "finish_reason": None,
                }
            ],
        }
    )
    adapter._llm_messages.put_nowait(
        {
            "type": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    adapter._llm_messages.put_nowait(
        {
            "type": "agent.event",
            "kind": "tool",
            "state": "completed",
            "name": "tool.done",
            "summary": "finished",
        }
    )
    adapter._llm_messages.shutdown()

    await adapter._process_llm_events()

    message = messages.children[0]
    event = messages.children[1]
    assert message.tag_name == "message"
    assert message.get_attribute("author_name") == "assistant"
    assert message.values["text"] == "hello"
    assert event.tag_name == "event"
    assert event.get_attribute("kind") == "tool"
    assert event.get_attribute("state") == "completed"
    assert [(span.name, span.attributes) for span in tracer.spans] == [
        (
            "chatbot.thread.message",
            {
                "from_participant_name": "assistant",
                "role": "assistant",
                "text": "hello",
            },
        )
    ]


@pytest.mark.asyncio
async def test_completions_process_llm_events_handles_multiple_choices(
    monkeypatch,
) -> None:
    tracer = _FakeTracer()
    monkeypatch.setattr(completions_thread_adapter_module, "tracer", tracer)

    adapter = object.__new__(CompletionsThreadAdapter)
    messages = _FakeElement("messages")
    adapter._thread = _FakeThread([messages])
    adapter._room = _FakeRoom()
    adapter._active_events_by_key = {}
    adapter._llm_messages = asyncio.Queue()

    adapter._llm_messages.put_nowait("ignored non-dict queue entry")
    adapter._llm_messages.put_nowait(
        {
            "type": "chat.completion.chunk",
            "choices": [
                {
                    "index": 1,
                    "delta": {"content": "hel"},
                    "finish_reason": None,
                },
                {
                    "index": 2,
                    "delta": {"content": [{"text": "wor"}]},
                    "finish_reason": None,
                },
            ],
        }
    )
    adapter._llm_messages.put_nowait(
        {
            "type": "chat.completion.chunk",
            "choices": [
                {
                    "index": 1,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "delta": {"content": {"text": "ld"}},
                    "finish_reason": "stop",
                },
            ],
        }
    )
    adapter._llm_messages.shutdown()

    await adapter._process_llm_events()

    assert [message.tag_name for message in messages.children] == [
        "message",
        "message",
    ]
    assert [message.values["text"] for message in messages.children] == [
        "hello",
        "world",
    ]
    assert [message.get_attribute("author_name") for message in messages.children] == [
        "assistant",
        "assistant",
    ]
    assert [span.attributes["text"] for span in tracer.spans] == [
        "hello",
        "world",
    ]
