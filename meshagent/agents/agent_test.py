from typing import Any

import pytest

from meshagent.agents.agent import RoomTool
from meshagent.api import RoomException
from meshagent.api.messaging import TextContent
from meshagent.tools import ToolContext


class _FakeParticipant:
    def __init__(self, *, participant_id: str, name: str):
        self.id = participant_id
        self._name = name

    def get_attribute(self, key: str) -> Any:
        if key == "name":
            return self._name
        return None


class _FakeIterableResult:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeAgentsClient:
    def __init__(self, *, response: Any):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke_tool(
        self,
        *,
        toolkit: str,
        tool: str,
        input: Any,
        participant_id: str | None = None,
        on_behalf_of_id: str | None = None,
        caller_context: dict | None = None,
    ) -> Any:
        self.calls.append(
            {
                "toolkit": toolkit,
                "tool": tool,
                "input": input,
                "participant_id": participant_id,
                "on_behalf_of_id": on_behalf_of_id,
                "caller_context": caller_context,
            }
        )
        return self._response


class _FakeRoom:
    def __init__(self, *, agents: _FakeAgentsClient):
        self.agents = agents


@pytest.mark.asyncio
async def test_room_tool_raises_if_remote_tool_returns_iterable() -> None:
    fake_agents = _FakeAgentsClient(response=_FakeIterableResult())
    room = _FakeRoom(agents=fake_agents)
    caller = _FakeParticipant(participant_id="caller-id", name="caller")
    context = ToolContext(
        room=room,
        caller=caller,
        caller_context={"chat": {"id": "chat-1"}},
    )
    tool = RoomTool(
        toolkit_name="remote_tools",
        name="computer_call",
        input_schema={
            "type": "object",
            "required": [],
            "properties": {},
            "additionalProperties": False,
        },
    )

    with pytest.raises(RoomException, match="returned an iterable stream"):
        await tool.execute(context=context)

    assert len(fake_agents.calls) == 1
    assert fake_agents.calls[0]["caller_context"] == {"chat": {"id": "chat-1"}}


@pytest.mark.asyncio
async def test_room_tool_uses_non_stream_call_without_event_handler() -> None:
    response = TextContent(text="ok")
    fake_agents = _FakeAgentsClient(response=response)
    room = _FakeRoom(agents=fake_agents)
    caller = _FakeParticipant(participant_id="caller-id", name="caller")
    context = ToolContext(room=room, caller=caller)
    tool = RoomTool(
        toolkit_name="remote_tools",
        name="computer_call",
        input_schema={
            "type": "object",
            "required": [],
            "properties": {},
            "additionalProperties": False,
        },
    )

    result = await tool.execute(context=context)

    assert result == response
    assert len(fake_agents.calls) == 1


def test_room_tool_defaults_to_strict_when_metadata_is_missing() -> None:
    tool = RoomTool(
        toolkit_name="remote_tools",
        name="computer_call",
        input_schema={
            "type": "object",
            "required": [],
            "properties": {},
            "additionalProperties": False,
        },
    )

    assert tool.strict is True


def test_room_tool_preserves_explicit_non_strict_metadata() -> None:
    tool = RoomTool(
        toolkit_name="remote_tools",
        name="computer_call",
        input_schema={
            "type": "object",
            "required": [],
            "properties": {},
            "additionalProperties": False,
        },
        strict=False,
    )

    assert tool.strict is False
