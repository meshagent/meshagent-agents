from abc import ABC, abstractmethod
from .agent import AgentChatContext
from jsonschema import validate
from meshagent.tools.toolkit import Response, Toolkit
from meshagent.api import RoomClient
from typing import Any, Optional

class ToolResponseAdapter(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    async def to_plain_text(self, *, room: RoomClient, response: Response):
        pass

    @abstractmethod
    async def append_messages(self, *, context: AgentChatContext, tool_call: Any, room: RoomClient, response: Response):
        pass


class LLMAdapter(ABC):

    def create_chat_context(self) -> AgentChatContext:
        return AgentChatContext()
    
    @abstractmethod
    async def next(self, 
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: Toolkit,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
    ) -> Any:  
        pass

    def validate(response: dict, output_schema: dict):
        validate(response, output_schema)

