from abc import ABC, abstractmethod
from .agent import AgentChatContext
from jsonschema import validate
from meshagent.tools import Response, Toolkit, ToolkitBuilder, ToolkitConfig
from meshagent.api import RoomClient, RoomException, RemoteParticipant
from typing import Any, Optional, Callable, TypeVar, Generic, Literal

TEvent = TypeVar("T")


class ToolResponseAdapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def to_plain_text(self, *, room: RoomClient, response: Response):
        pass

    @abstractmethod
    async def create_messages(
        self,
        *,
        context: AgentChatContext,
        tool_call: Any,
        room: RoomClient,
        response: Response,
    ) -> list:
        pass


class LLMAdapter(Generic[TEvent]):
    @abstractmethod
    def default_model(self) -> str: ...

    def create_chat_context(self) -> AgentChatContext:
        return AgentChatContext()

    async def check_for_termination(
        self, *, context: AgentChatContext, room: RoomClient
    ):
        return True

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
        context: AgentChatContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[TEvent], None]] = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
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

    def create_chat_context(self) -> AgentChatContext:
        return AgentChatContext()

    async def check_for_termination(
        self, *, context: AgentChatContext, room: RoomClient
    ):
        return True

    async def next(
        self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[TEvent], None]] = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
    ) -> Any:
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
