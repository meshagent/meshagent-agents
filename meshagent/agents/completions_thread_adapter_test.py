import pytest

from meshagent.api import RoomException
from meshagent.agents.completions_thread_adapter import CompletionsThreadAdapter


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


class _FakeRoot:
    def __init__(self, children: list[_FakeElement]) -> None:
        self._children = children

    def get_children(self) -> list[_FakeElement]:
        return self._children


class _FakeThread:
    def __init__(self, children: list[_FakeElement]) -> None:
        self.root = _FakeRoot(children)


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
