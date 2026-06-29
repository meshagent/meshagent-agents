import pytest

from meshagent.api import RoomException
from meshagent.agents.completions_thread_adapter import CompletionsThreadAdapter


class _FakeElement:
    def __init__(self, tag_name: str) -> None:
        self.tag_name = tag_name


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
