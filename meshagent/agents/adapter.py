from abc import ABC, abstractmethod
from .agent import AgentChatContext
from jsonschema import validate
from meshagent.tools.provider import ToolProvider, ToolConfig
from meshagent.tools import Response, Toolkit, Tool
from meshagent.api import RoomClient, RoomException
from typing import Any, Optional, Callable, TypeVar, Generic

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

    @abstractmethod
    async def check_for_termination(
        self, *, context: AgentChatContext, room: RoomClient
    ):
        return True

    def tool_providers(self, *, model: str) -> list[ToolProvider]:
        return []

    def make_tool(self, *, model: str, config: ToolConfig, **kwargs) -> Tool:
        for tool in self.tool_providers(model=model):
            if tool.name == config.name:
                return tool.make(model=model, config=config, **kwargs)

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
    ) -> Any:
        pass

    def validate(response: dict, output_schema: dict):
        validate(response, output_schema)
