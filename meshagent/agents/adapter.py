import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, TypeVar

from jsonschema import validate

from meshagent.api.messaging import FileContent, JsonContent, TextContent
from meshagent.api import RoomClient, RoomException, RemoteParticipant
from meshagent.tools import Content, ToolContext, Toolkit, ToolkitBuilder, ToolkitConfig

from .agent import AgentSessionContext
from .messages import AgentMessage

TEvent = TypeVar("T")

DEFAULT_MAX_TOOL_CALL_LINES = 2000
DEFAULT_MAX_TOOL_CALL_LENGTH = 50 * 1024
_TEXTUAL_APPLICATION_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-sh",
    "application/yaml",
    "application/x-yaml",
    "application/xhtml+xml",
}


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
    def __init__(
        self,
        *,
        max_tool_call_length: int = DEFAULT_MAX_TOOL_CALL_LENGTH,
        max_tool_call_lines: int = DEFAULT_MAX_TOOL_CALL_LINES,
    ):
        if max_tool_call_length <= 0:
            raise ValueError("max_tool_call_length must be greater than 0")
        if max_tool_call_lines <= 0:
            raise ValueError("max_tool_call_lines must be greater than 0")
        self.max_tool_call_length = max_tool_call_length
        self.max_tool_call_lines = max_tool_call_lines

    @staticmethod
    def _normalize_mime_type(mime_type: str | None) -> str:
        if mime_type is None:
            return ""
        return mime_type.partition(";")[0].strip().lower()

    @staticmethod
    def _looks_like_text(*, data: bytes, decoded: str) -> bool:
        if b"\x00" in data:
            return False
        return all(ord(ch) >= 32 or ch in "\n\r\t\f\b" for ch in decoded)

    async def file_content_to_text_content(
        self,
        *,
        room: RoomClient,
        content: Content,
    ) -> TextContent | None:
        del room
        if not isinstance(content, FileContent):
            return None

        normalized_mime_type = self._normalize_mime_type(content.mime_type)
        is_declared_text = normalized_mime_type.startswith("text/") or (
            normalized_mime_type in _TEXTUAL_APPLICATION_MIME_TYPES
        )

        try:
            decoded = content.data.decode("utf-8")
        except UnicodeDecodeError:
            return None

        if is_declared_text or self._looks_like_text(
            data=content.data,
            decoded=decoded,
        ):
            return TextContent(text=decoded)

        return None

    def truncate(self, *, room: RoomClient, content: Content) -> Content:
        del room
        text: str | None = None
        if isinstance(content, TextContent):
            text = content.text
        elif isinstance(content, JsonContent):
            text = json.dumps(content.json, ensure_ascii=False)

        if text is None:
            return content

        original_line_count = len(text.splitlines()) if text != "" else 0
        limited_text = text
        if original_line_count > self.max_tool_call_lines:
            limited_text = "\n".join(text.splitlines()[: self.max_tool_call_lines])

        original_bytes = text.encode("utf-8")
        limited_bytes = limited_text.encode("utf-8")
        if len(limited_bytes) > self.max_tool_call_length:
            limited_text = limited_bytes[: self.max_tool_call_length].decode(
                "utf-8", errors="ignore"
            )

        if (
            original_line_count <= self.max_tool_call_lines
            and len(original_bytes) <= self.max_tool_call_length
        ):
            return content

        truncated_text = limited_text.rstrip()
        notice = (
            "The tool call returned too much data and was truncated. "
            f"Showing at most {self.max_tool_call_lines} lines and "
            f"{self.max_tool_call_length} bytes."
        )
        if truncated_text == "":
            return TextContent(text=notice)

        return TextContent(text=f"{truncated_text}\n\n{notice}")

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

    def get_additional_instructions(self) -> str | None:
        return None

    def on_turn_steer(self, *, context: AgentSessionContext, interrupted: bool) -> None:
        del context
        del interrupted

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
        custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Callable[[TEvent], None]:
        del turn_id
        del thread_id

        def publish(event: TEvent) -> None:
            if isinstance(event, AgentMessage):
                callback(event)
                return

            if not isinstance(event, dict) or custom_event_callback is None:
                return

            event_type = event.get("type")
            if event_type in ("agent.event", "codex.event"):
                custom_event_callback(event)

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
        del context
        del room
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        raise RoomException(
            "MessageStreamLLMAdapter has been removed; use streaming toolkits instead"
        )

    def validate(response: dict, output_schema: dict):
        validate(response, output_schema)
