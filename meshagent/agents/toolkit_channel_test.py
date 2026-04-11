import asyncio

import pytest

from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AgentError,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallApprovalRequested,
    RejectAgentToolCall,
    TurnEnded,
    TurnStart,
    TurnStarted,
)
from meshagent.agents.process import Message
from meshagent.agents.toolkit_channel import ToolkitChannel
from meshagent.api import Participant
from meshagent.api.messaging import JsonContent, TextContent
from meshagent.api.room_server_client import RoomException
from meshagent.tools import ToolContext


class _FakeLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


class _FakeStorage:
    async def exists(self, *, path: str) -> bool:
        del path
        return False


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.storage = _FakeStorage()


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


class _FakeParticipant(Participant):
    def __init__(self, *, name: str, participant_id: str = "caller-id") -> None:
        super().__init__(id=participant_id, attributes={"name": name})


async def _drain() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_toolkit_channel_exposes_remote_tool_and_returns_final_text() -> None:
    room = _FakeRoom()
    caller = _FakeParticipant(name="Caller")
    supervisor = _RecordingSupervisor()
    channel = ToolkitChannel(
        room=room,
        toolkit_name="assistant",
        thread_dir="/threads/tasks",
    )
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        remote_toolkit = channel.make_toolkit()
        assert remote_toolkit.name == "assistant"
        assert {tool.name for tool in remote_toolkit.tools} == {"run_assistant_task"}

        result_task = asyncio.create_task(
            remote_toolkit.invoke(
                context=ToolContext(caller=caller),
                name="run_assistant_task",
                input=JsonContent(
                    json={
                        "prompt": "Plan the work",
                        "model": "gpt-5.4",
                        "instructions": "Be brief",
                    }
                ),
            )
        )
        await _drain()

        assert len(supervisor.sent) == 1
        request = supervisor.sent[0]
        assert isinstance(request.data, TurnStart)
        assert request.sender is caller
        assert request.data.thread_id == "/threads/tasks/Plan the work.thread"
        assert request.data.content[0].text == "Plan the work"
        assert request.data.model == "gpt-5.4"
        assert request.data.instructions == "Be brief"

        channel.send(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    source_message_id=request.data.message_id,
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    item_id="text-1",
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    item_id="text-1",
                    text="Hello",
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    item_id="text-1",
                    text=" world",
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    item_id="text-1",
                )
            )
        )
        channel.send(
            Message(
                data=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-1",
                    error=None,
                )
            )
        )

        response = await result_task
        assert isinstance(response, TextContent)
        assert response.text == "Hello world"
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_toolkit_channel_respects_explicit_thread_id_and_surfaces_turn_errors() -> (
    None
):
    room = _FakeRoom()
    caller = _FakeParticipant(name="Caller")
    supervisor = _RecordingSupervisor()
    channel = ToolkitChannel(room=room, toolkit_name="assistant")
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        remote_toolkit = channel.make_toolkit()
        result_task = asyncio.create_task(
            remote_toolkit.invoke(
                context=ToolContext(caller=caller),
                name="run_assistant_task",
                input=JsonContent(
                    json={
                        "prompt": "Continue",
                        "path": "/threads/shared.thread",
                        "thread_id": "/threads/shared.thread",
                    }
                ),
            )
        )
        await _drain()

        assert len(supervisor.sent) == 1
        request = supervisor.sent[0]
        assert isinstance(request.data, TurnStart)
        assert request.data.thread_id == "/threads/shared.thread"

        channel.send(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-2",
                    source_message_id=request.data.message_id,
                )
            )
        )
        channel.send(
            Message(
                data=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-2",
                    error=AgentError(message="turn failed", code="RoomException"),
                )
            )
        )

        with pytest.raises(RoomException, match="turn failed"):
            await result_task
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_toolkit_channel_auto_rejects_tool_approvals() -> None:
    room = _FakeRoom()
    caller = _FakeParticipant(name="Caller")
    supervisor = _RecordingSupervisor()
    channel = ToolkitChannel(room=room, toolkit_name="assistant")
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        remote_toolkit = channel.make_toolkit()
        result_task = asyncio.create_task(
            remote_toolkit.invoke(
                context=ToolContext(caller=caller),
                name="run_assistant_task",
                input=JsonContent(json={"prompt": "Do the work"}),
            )
        )
        await _drain()

        request = supervisor.sent[0]
        assert isinstance(request.data, TurnStart)
        channel.send(
            Message(
                data=TurnStarted(
                    type=AGENT_EVENT_TURN_STARTED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-3",
                    source_message_id=request.data.message_id,
                )
            )
        )
        channel.send(
            Message(
                data=AgentToolCallApprovalRequested(
                    type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-3",
                    item_id="approval-1",
                    toolkit="shell",
                    tool="shell",
                    arguments={"command": "echo hi"},
                )
            )
        )
        await _drain()

        assert len(supervisor.sent) == 2
        rejection = supervisor.sent[1]
        assert isinstance(rejection.data, RejectAgentToolCall)
        assert rejection.data.type == AGENT_MESSAGE_TOOL_CALL_REJECT
        assert rejection.data.turn_id == "turn-3"
        assert rejection.data.item_id == "approval-1"

        channel.send(
            Message(
                data=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id=request.data.thread_id,
                    turn_id="turn-3",
                    error=AgentError(message="tool rejected", code="rejected"),
                )
            )
        )

        with pytest.raises(RoomException, match="tool rejected"):
            await result_task
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]
