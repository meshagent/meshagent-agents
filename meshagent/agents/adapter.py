from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, TypeVar

from jsonschema import validate

from meshagent.api import RoomClient, RoomException, RemoteParticipant
from meshagent.tools import Content, ToolContext, Toolkit, ToolkitBuilder, ToolkitConfig

from .agent import AgentSessionContext
from .messages import AgentMessage

TEvent = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolCallApprovalRequest:
    item_id: str
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None = None


ToolCallApprovalHandler = Callable[
    [ToolContext, ToolCallApprovalRequest],
    Awaitable[bool],
]
SteeringCallback = Callable[[], Awaitable[bool]]


class ToolResponseAdapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def to_plain_text(self, *, room: RoomClient, response: Content):
        pass

    @abstractmethod
    async def create_messages(
        self,
        *,
        context: AgentSessionContext,
        tool_call: Any,
        room: RoomClient,
        response: Content,
    ) -> list:
        pass


class LLMAdapter(Generic[TEvent]):
    outputTokenMax: float = float("inf")

    @abstractmethod
    def default_model(self) -> str: ...

    def create_session(self) -> AgentSessionContext:
        return AgentSessionContext()

    def context_window_size(self, model: str) -> float:
        return float("inf")

    def needs_compaction(self, *, context: AgentSessionContext) -> bool:
        return False

    async def compact(
        self,
        *,
        context: AgentSessionContext,
        room: RoomClient,
        model: Optional[str] = None,
    ) -> None:
        return None

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        room: Optional[RoomClient] = None,
        toolkits: Optional[list] = None,
        output_schema: Optional[dict] = None,
    ) -> int:
        return 0

    async def check_for_termination(
        self, *, context: AgentSessionContext, room: RoomClient
    ):
        return True

    def set_tool_call_approval_handler(
        self, handler: ToolCallApprovalHandler | None
    ) -> None:
        del handler

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback: Callable[[AgentMessage], None],
    ) -> Callable[[TEvent], None]:
        del turn_id
        del thread_id

        def publish(event: TEvent) -> None:
            if isinstance(event, AgentMessage):
                callback(event)

        return publish

    def tool_providers(self, *, model: str) -> list[ToolkitBuilder]:
        return []

    async def make_toolkit(
        self, *, room: RoomClient, model: str, config: ToolkitConfig
    ) -> Toolkit:
        for tool in self.tool_providers(model=model):
            if tool.name == config.name:
                return Toolkit(
                    name=config.name,
                    tools=[await tool.make(room=room, model=model, config=config)],
                )

        raise RoomException(f"Unexpected tool: {config.name} for model {model}")

    @abstractmethod
    async def next(
        self,
        *,
        context: AgentSessionContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[TEvent], None]] = None,
        steering_callback: SteeringCallback | None = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
        options: Optional[dict] = None,
    ) -> Any:
        pass

    def validate(response: dict, output_schema: dict):
        validate(response, output_schema)


class MessageStreamLLMAdapter(LLMAdapter):
    def __init__(
        self, *, participant_name: str, context_mode: Literal["diff", "full"] = "diff"
    ):
        self.participant_name = participant_name
        self.context_mode = context_mode

    def default_model(self) -> str:
        return "toolkit"

    def create_session(self) -> AgentSessionContext:
        return AgentSessionContext()

    async def check_for_termination(
        self, *, context: AgentSessionContext, room: RoomClient
    ):
        return True

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[TEvent], None]] = None,
        steering_callback: SteeringCallback | None = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
        options: Optional[dict] = None,
    ) -> Any:
        del steering_callback
        participant = room.messaging.get_participant_by_name(self.participant_name)
        if participant is None:
            raise RoomException("participant is not currently connected")

        stream = await room.messaging.create_stream(
            to=participant,
            header={
                "context": context.to_json(),
                "model": model,
                "output_schema": output_schema,
                "on_behalf_of_id": on_behalf_of.id if on_behalf_of else None,
                "metadata": context.metadata,
            },
        )

        error = None
        output = None
        try:
            async for chunk in stream.read_chunks():
                event = chunk.header.get("event")
                if event is not None and event_handler is not None:
                    event_handler(event)

                output = chunk.header.get("output")
                if output is not None:
                    output.append(output)

                if chunk.header.get("done"):
                    break

        except Exception as ex:
            error = ex

        await stream.close()

        if self.context_mode == "diff":
            context.messages.clear()

        if error:
            raise error

        return output

    def validate(response: dict, output_schema: dict):
        validate(response, output_schema)
