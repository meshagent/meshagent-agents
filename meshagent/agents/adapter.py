from abc import ABC, abstractmethod
from .agent import AgentChatContext
from jsonschema import validate
from meshagent.tools.toolkit import Response, Toolkit, Tool
from meshagent.api import RoomClient, RoomException
from typing import Any, Optional, Callable, TypeVar, Generic
from pydantic import BaseModel

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


class LLMToolConfig(BaseModel):
    name: str


class LLMTool:
    def __init__(self, *, name: str, type: type):
        self.name = name
        self.type = type

    def make(self, *, model: str, config: LLMToolConfig, **kwargs) -> Tool: ...


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

    def llm_tools(self, *, model: str) -> list[LLMTool]:
        return []

    def make_tool(self, *, model: str, config: LLMToolConfig, **kwargs) -> Tool:
        for tool in self.llm_tools(model=model):
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
