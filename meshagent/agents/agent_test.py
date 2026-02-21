from typing import Any

import pytest

from meshagent.agents.agent import RoomTool
from meshagent.api.messaging import TextResponse
from meshagent.api.room_server_client import ToolCallStreamItem
from meshagent.tools import ToolContext


class _FakeParticipant:
    def __init__(self, *, participant_id: str, name: str):
        self.id = participant_id
        self._name = name

    def get_attribute(self, key: str) -> Any:
        if key == "name":
            return self._name
        return None


class _FakeToolCallStream:
    def __init__(self, *, items: list[ToolCallStreamItem]):
        self._items = items

    def __aiter__(self):
        return self._run()

    async def _run(self):
        for item in self._items:
            yield item


class _FakeAgentsClient:
    def __init__(self, *, response: Any):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke_tool(
        self,
        *,
        toolkit: str,
        tool: str,
        arguments: dict,
        participant_id: str | None = None,
        on_behalf_of_id: str | None = None,
        caller_context: dict | None = None,
        attachment: bytes | None = None,
        stream: bool = False,
        tool_call_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "toolkit": toolkit,
                "tool": tool,
                "arguments": arguments,
                "participant_id": participant_id,
                "on_behalf_of_id": on_behalf_of_id,
                "caller_context": caller_context,
                "attachment": attachment,
                "stream": stream,
                "tool_call_id": tool_call_id,
            }
        )
        return self._response


class _FakeRoom:
    def __init__(self, *, agents: _FakeAgentsClient):
        self.agents = agents


@pytest.mark.asyncio
async def test_room_tool_streams_events_when_context_has_event_handler() -> None:
    response = TextResponse(text="ok")
    stream = _FakeToolCallStream(
        items=[
            ToolCallStreamItem(
                type="event",
                tool_call_id="ignored",
                toolkit="remote_tools",
                tool="computer_call",
                event={
                    "headline": "Starting Playwright container",
                    "state": "in_progress",
                },
            ),
            ToolCallStreamItem(
                type="result",
                tool_call_id="ignored",
                result=response,
            ),
        ]
    )
    fake_agents = _FakeAgentsClient(response=stream)
    room = _FakeRoom(agents=fake_agents)
    caller = _FakeParticipant(participant_id="caller-id", name="caller")
    emitted: list[dict[str, Any]] = []
    context = ToolContext(
        room=room,
        caller=caller,
        caller_context={"chat": {"id": "chat-1"}},
        event_handler=lambda event: emitted.append(event),
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

    result = await tool.execute(context=context)

    assert result == response
    assert len(fake_agents.calls) == 1
    assert fake_agents.calls[0]["stream"] is True
    assert fake_agents.calls[0]["caller_context"] == {"chat": {"id": "chat-1"}}
    assert isinstance(fake_agents.calls[0]["tool_call_id"], str)
    assert fake_agents.calls[0]["tool_call_id"] != ""
    assert len(emitted) == 1
    assert emitted[0]["type"] == "agent.event"
    assert emitted[0]["kind"] == "tool"
    assert emitted[0]["state"] == "in_progress"
    assert emitted[0]["headline"] == "Starting Playwright container"
    assert emitted[0]["correlation_key"].startswith("tool_call:")


@pytest.mark.asyncio
async def test_room_tool_uses_non_stream_call_without_event_handler() -> None:
    response = TextResponse(text="ok")
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
    assert fake_agents.calls[0]["stream"] is False
    assert fake_agents.calls[0]["tool_call_id"] is None
