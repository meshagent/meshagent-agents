from typing import Any

import pytest

from meshagent.agents.agent import RemoteRoomTool, SingleRoomAgent
from meshagent.agents.web_participant import WebParticipant
from meshagent.api import (
    Participant,
    RequiredToolkit,
    TOOL_SEARCH_ANNOTATION,
    RoomException,
    ToolContentSpec,
    ToolDescription,
    ToolkitDescription,
)
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
    def __init__(
        self,
        *,
        response: Any,
        toolkits: list[Any] | None = None,
        toolkits_by_participant_id: dict[str, list[Any]] | None = None,
    ):
        self._response = response
        self._toolkits = list(toolkits or [])
        self._toolkits_by_participant_id = dict(toolkits_by_participant_id or {})
        self.calls: list[dict[str, Any]] = []
        self.list_toolkit_calls: list[dict[str, str | int | None]] = []

    async def invoke_tool(
        self,
        *,
        toolkit: str,
        tool: str,
        input: Any,
        participant_id: str | None = None,
        on_behalf_of_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "toolkit": toolkit,
                "tool": tool,
                "input": input,
                "participant_id": participant_id,
                "on_behalf_of_id": on_behalf_of_id,
            }
        )
        return self._response

    async def list_toolkits(
        self,
        *,
        participant_id: str | None = None,
        participant_name: str | None = None,
        timeout: int | None = None,
    ) -> list[Any]:
        self.list_toolkit_calls.append(
            {
                "participant_id": participant_id,
                "participant_name": participant_name,
                "timeout": timeout,
            }
        )
        if participant_id is not None:
            return self._toolkits_by_participant_id.get(participant_id, [])
        return self._toolkits


class _FakeRoom:
    def __init__(self, *, agents: _FakeAgentsClient, token: str | None = None):
        self.agents = agents
        self.protocol = _FakeProtocol(token=token)


class _FakeProtocol:
    def __init__(self, *, token: str | None = None):
        self.token = token


class _CredentialBindingAgent(SingleRoomAgent):
    def __init__(self) -> None:
        super().__init__(name="helper")
        self.bound_room = None
        self.bound_api_key = None

    async def install_requirements(self, participant_id: str | None = None):
        del participant_id

    async def get_exposed_toolkits(self) -> list:
        return []

    def bind_runtime_credentials(self, *, room) -> None:
        self.bound_room = room
        self.bound_api_key = self.resolve_runtime_api_key(room=room)


@pytest.mark.asyncio
async def test_remote_room_tool_raises_if_remote_room_tool_returns_iterable() -> None:
    fake_agents = _FakeAgentsClient(response=_FakeIterableResult())
    room = _FakeRoom(agents=fake_agents)
    caller = _FakeParticipant(participant_id="caller-id", name="caller")
    context = ToolContext(caller=caller)
    tool = RemoteRoomTool(
        room=room,
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


@pytest.mark.asyncio
async def test_remote_room_tool_uses_non_stream_call_without_event_handler() -> None:
    response = TextContent(text="ok")
    fake_agents = _FakeAgentsClient(response=response)
    room = _FakeRoom(agents=fake_agents)
    caller = _FakeParticipant(participant_id="caller-id", name="caller")
    context = ToolContext(caller=caller)
    tool = RemoteRoomTool(
        room=room,
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


def test_remote_room_tool_defaults_to_strict_when_metadata_is_missing() -> None:
    room = _FakeRoom(agents=_FakeAgentsClient(response=None))
    tool = RemoteRoomTool(
        room=room,
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


def test_remote_room_tool_preserves_explicit_non_strict_metadata() -> None:
    room = _FakeRoom(agents=_FakeAgentsClient(response=None))
    tool = RemoteRoomTool(
        room=room,
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


@pytest.mark.asyncio
async def test_single_room_agent_start_binds_runtime_credentials() -> None:
    room = _FakeRoom(
        agents=_FakeAgentsClient(response=None),
        token="  room-token  ",
    )
    agent = _CredentialBindingAgent()

    await agent.start(room=room)

    assert agent.bound_room is room
    assert agent.bound_api_key == "room-token"

    await agent.stop()


@pytest.mark.asyncio
async def test_single_room_agent_missing_required_toolkit_error_names_participants() -> (
    None
):
    fake_agents = _FakeAgentsClient(response=None)
    room = _FakeRoom(agents=fake_agents)
    agent = SingleRoomAgent(
        name="helper",
        requires=[RequiredToolkit(name="PropertyAssistant")],
    )
    agent._room = room
    caller = Participant(id="caller-id", attributes={"name": "Caller Agent"})
    on_behalf_of = Participant(id="user-id", attributes={"name": "Ada"})
    context = ToolContext(caller=caller, on_behalf_of=on_behalf_of)

    with pytest.raises(RoomException) as exc_info:
        await agent.get_required_toolkits(context)

    assert str(exc_info.value) == (
        "unable to get toolkit PropertyAssistant "
        "on behalf of participant(id=user-id, name=Ada) "
        "for caller participant(id=caller-id, name=Caller Agent)"
    )
    assert fake_agents.list_toolkit_calls == [
        {"participant_id": None, "participant_name": None, "timeout": None},
        {"participant_id": "user-id", "participant_name": None, "timeout": 0},
    ]


@pytest.mark.asyncio
async def test_single_room_agent_required_toolkit_finds_public_room_toolkit() -> None:
    tool_description = ToolDescription(
        name="lookup",
        title="",
        description="",
        input_spec=ToolContentSpec(
            types=["json"],
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    toolkit_description = ToolkitDescription(
        name="PersonalAssistant",
        title="",
        description="",
        tools=[tool_description],
    )
    fake_agents = _FakeAgentsClient(response=None, toolkits=[toolkit_description])
    room = _FakeRoom(agents=fake_agents)
    agent = SingleRoomAgent(
        name="helper",
        requires=[RequiredToolkit(name="PersonalAssistant", tools=["lookup"])],
    )
    agent._room = room
    caller = Participant(id="caller-id", attributes={"name": "Caller Agent"})
    on_behalf_of = Participant(id="user-id", attributes={"name": "Ada"})
    context = ToolContext(caller=caller, on_behalf_of=on_behalf_of)

    toolkits = await agent.get_required_toolkits(context)

    assert len(toolkits) == 1
    await toolkits[0].tools[0].execute(context=context)
    assert fake_agents.list_toolkit_calls == [
        {"participant_id": None, "participant_name": None, "timeout": None},
        {"participant_id": "user-id", "participant_name": None, "timeout": 0},
    ]
    assert fake_agents.calls[0]["toolkit"] == "PersonalAssistant"
    assert fake_agents.calls[0]["tool"] == "lookup"
    assert fake_agents.calls[0]["participant_id"] is None
    assert fake_agents.calls[0]["on_behalf_of_id"] == "user-id"


@pytest.mark.asyncio
async def test_single_room_agent_tool_search_toolkits_filter_on_annotation() -> None:
    annotated_tool = ToolDescription(
        name="lookup",
        title="",
        description="",
        input_spec=ToolContentSpec(
            types=["json"],
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    annotated_toolkit = ToolkitDescription(
        name="SearchableAssistant",
        title="",
        description="",
        annotations={TOOL_SEARCH_ANNOTATION: "true"},
        tools=[annotated_tool],
    )
    hidden_toolkit = ToolkitDescription(
        name="HiddenAssistant",
        title="",
        description="",
        annotations={TOOL_SEARCH_ANNOTATION: "false"},
        tools=[annotated_tool],
    )
    fake_agents = _FakeAgentsClient(
        response=None,
        toolkits=[annotated_toolkit, hidden_toolkit],
    )
    room = _FakeRoom(agents=fake_agents)
    agent = SingleRoomAgent(name="helper")
    agent._room = room
    caller = Participant(id="caller-id", attributes={"name": "Caller Agent"})
    context = ToolContext(caller=caller)

    toolkits = await agent.get_tool_search_toolkits(context)

    assert [toolkit.name for toolkit in toolkits] == ["SearchableAssistant"]
    assert toolkits[0].annotations == {TOOL_SEARCH_ANNOTATION: "true"}


@pytest.mark.asyncio
async def test_single_room_agent_tool_search_does_not_wait_for_participant_toolkits() -> (
    None
):
    annotated_tool = ToolDescription(
        name="lookup",
        title="",
        description="",
        input_spec=ToolContentSpec(
            types=["json"],
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    participant_toolkit = ToolkitDescription(
        name="ParticipantAssistant",
        title="",
        description="",
        annotations={TOOL_SEARCH_ANNOTATION: "true"},
        tools=[annotated_tool],
        participant_id="user-id",
    )
    fake_agents = _FakeAgentsClient(
        response=None,
        toolkits_by_participant_id={"user-id": [participant_toolkit]},
    )
    room = _FakeRoom(agents=fake_agents)
    agent = SingleRoomAgent(name="helper")
    agent._room = room
    caller = Participant(id="caller-id", attributes={"name": "Caller Agent"})
    on_behalf_of = Participant(id="user-id", attributes={"name": "Ada"})
    context = ToolContext(caller=caller, on_behalf_of=on_behalf_of)

    toolkits = await agent.get_tool_search_toolkits(context)

    assert [toolkit.name for toolkit in toolkits] == ["ParticipantAssistant"]
    assert fake_agents.list_toolkit_calls == [
        {"participant_id": None, "participant_name": None, "timeout": None},
        {"participant_id": "user-id", "participant_name": None, "timeout": 0},
    ]


@pytest.mark.asyncio
async def test_single_room_agent_forwards_web_participant_on_behalf_of_id() -> None:
    tool_description = ToolDescription(
        name="lookup",
        title="",
        description="",
        input_spec=ToolContentSpec(
            types=["json"],
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    )
    toolkit_description = ToolkitDescription(
        name="PersonalAssistant",
        title="",
        description="",
        tools=[tool_description],
    )
    fake_agents = _FakeAgentsClient(response=None, toolkits=[toolkit_description])
    room = _FakeRoom(agents=fake_agents)
    agent = SingleRoomAgent(
        name="helper",
        requires=[RequiredToolkit(name="PersonalAssistant", tools=["lookup"])],
    )
    agent._room = room
    caller = Participant(id="caller-id", attributes={"name": "Caller Agent"})
    web_participant = WebParticipant(
        participant=Participant(id="user-id", attributes={"name": "Ada"}),
        connection_id="connection-id",
    )
    context = ToolContext(caller=caller, on_behalf_of=web_participant)

    toolkits = await agent.get_required_toolkits(context)

    assert len(toolkits) == 1
    await toolkits[0].tools[0].execute(context=context)
    assert fake_agents.list_toolkit_calls == [
        {"participant_id": None, "participant_name": None, "timeout": None},
    ]
    assert fake_agents.calls[0]["toolkit"] == "PersonalAssistant"
    assert fake_agents.calls[0]["tool"] == "lookup"
    assert fake_agents.calls[0]["participant_id"] is None
    assert fake_agents.calls[0]["on_behalf_of_id"] == "user-id"
