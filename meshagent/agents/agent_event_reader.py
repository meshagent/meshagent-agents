from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from meshagent.api.agent_content import AgentFileContent, AgentTextContent
from meshagent.api.messaging import Content

from .context import AgentSessionContext
from .messages import (
    AgentContextCompacted,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentImageGenerationCompleted,
    AgentImageGenerationFailed,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentThreadEvent,
    AgentToolCallEnded,
    AgentToolCallLogDelta,
    AgentToolCallPending,
    AgentToolCallInProgress,
    AgentToolCallStarted,
    TurnStart,
    TurnSteer,
)


class AgentEventReader(Protocol):
    def consume(self, message: AgentMessage) -> None: ...

    def finalize(self) -> None: ...


@dataclass(slots=True)
class _BufferedTextItem:
    role: str
    kind: str
    text: str = ""


@dataclass(slots=True)
class _BufferedFileItem:
    urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _BufferedToolCall:
    namespace: str
    call_id: str | None
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None
    logs: list[dict[str, str]] = field(default_factory=list)


def _image_generation_status(
    *,
    message: AgentMessage,
) -> str:
    if isinstance(message, AgentImageGenerationCompleted):
        return "completed"
    if isinstance(message, AgentImageGenerationFailed):
        return "failed"
    if isinstance(message, AgentImageGenerationPartial):
        return "in_progress"
    return "pending"


class DefaultAgentEventReader:
    def __init__(self, *, context: AgentSessionContext) -> None:
        self._context = context
        self._text_by_item_id: dict[str, _BufferedTextItem] = {}
        self._files_by_item_id: dict[str, _BufferedFileItem] = {}
        self._tool_calls_by_item_id: dict[str, _BufferedToolCall] = {}
        context.metadata.setdefault("agent_events", [])

    def consume(self, message: AgentMessage) -> None:
        self._record_event(message=message)

        if isinstance(message, (TurnStart, TurnSteer)):
            self._append_user_turn(message=message)
            return

        if isinstance(message, AgentTextContentDelta):
            self._buffer_text(
                item_id=message.item_id,
                role="assistant",
                kind="text",
                text=message.text,
            )
            return

        if isinstance(message, AgentTextContentEnded):
            self._flush_text_item(item_id=message.item_id)
            return

        if isinstance(message, AgentReasoningContentDelta):
            self._buffer_text(
                item_id=message.item_id,
                role="assistant",
                kind="reasoning",
                text=message.text,
            )
            return

        if isinstance(message, AgentReasoningContentEnded):
            self._flush_text_item(item_id=message.item_id)
            return

        if isinstance(message, AgentFileContentDelta):
            item = self._files_by_item_id.setdefault(
                message.item_id,
                _BufferedFileItem(),
            )
            item.urls.append(message.url)
            return

        if isinstance(message, AgentFileContentEnded):
            self._flush_file_item(item_id=message.item_id)
            return

        if isinstance(
            message,
            (AgentToolCallPending, AgentToolCallInProgress, AgentToolCallStarted),
        ):
            self._tool_calls_by_item_id[message.item_id] = _BufferedToolCall(
                namespace=message.namespace,
                call_id=message.call_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
            )
            return

        if isinstance(message, AgentToolCallLogDelta):
            item = self._tool_calls_by_item_id.get(message.item_id)
            if item is None:
                item = _BufferedToolCall(
                    namespace=message.namespace,
                    call_id=message.call_id,
                    toolkit="tool",
                    tool="tool",
                    arguments=None,
                )
                self._tool_calls_by_item_id[message.item_id] = item
            item.logs.extend([line.model_dump(mode="json") for line in message.lines])
            return

        if isinstance(message, AgentToolCallEnded):
            self._flush_tool_call(message=message)
            return

        if isinstance(message, AgentThreadEvent):
            self._append_context_item(
                role="assistant",
                item={
                    "type": "event",
                    "event": message.event,
                },
            )
            return

        if isinstance(
            message,
            (
                AgentImageGenerationStarted,
                AgentImageGenerationPartial,
                AgentImageGenerationCompleted,
                AgentImageGenerationFailed,
            ),
        ):
            self._append_context_item(
                role="assistant",
                item={
                    "type": "image_generation",
                    "event_type": message.type,
                    "item_id": message.item_id,
                    "call_id": message.call_id,
                    "toolkit": message.toolkit,
                    "tool": message.tool,
                    "arguments": message.arguments,
                    "images": [
                        image.model_dump(mode="json") for image in message.images
                    ]
                    if isinstance(message, AgentImageGenerationCompleted)
                    else [],
                    "status": _image_generation_status(message=message),
                    "status_detail": message.status_detail,
                },
            )
            return

        if isinstance(message, AgentContextCompacted):
            self._context.metadata["last_compaction"] = {
                "checkpoint_id": message.checkpoint_id,
                "path": message.path,
                "through_sequence": message.through_sequence,
                "created_at": message.created_at,
            }
            return

    def finalize(self) -> None:
        for item_id in list(self._text_by_item_id):
            self._flush_text_item(item_id=item_id)
        for item_id in list(self._files_by_item_id):
            self._flush_file_item(item_id=item_id)

    def _record_event(self, *, message: AgentMessage) -> None:
        events = self._context.metadata.get("agent_events")
        if not isinstance(events, list):
            events = []
            self._context.metadata["agent_events"] = events
        events.append(message.model_dump(mode="json"))

    def _append_user_turn(self, *, message: TurnStart | TurnSteer) -> None:
        content: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in message.content:
            item_json = item.model_dump(mode="json")
            content.append(item_json)
            if isinstance(item, AgentTextContent):
                text_parts.append(item.text)
            elif isinstance(item, AgentFileContent):
                text_parts.append(f"attached file: {item.url}")

        if len(content) == 1 and len(text_parts) == 1:
            self._context.messages.append({"role": "user", "content": text_parts[0]})
            return

        self._context.messages.append({"role": "user", "content": content})

    def _buffer_text(
        self,
        *,
        item_id: str,
        role: str,
        kind: str,
        text: str,
    ) -> None:
        item = self._text_by_item_id.get(item_id)
        if item is None:
            item = _BufferedTextItem(role=role, kind=kind)
            self._text_by_item_id[item_id] = item
        item.text += text

    def _flush_text_item(self, *, item_id: str) -> None:
        item = self._text_by_item_id.pop(item_id, None)
        if item is None or item.text == "":
            return
        if item.kind == "text":
            self._context.messages.append({"role": item.role, "content": item.text})
            return
        self._append_context_item(
            role=item.role,
            item={"type": item.kind, "text": item.text},
        )

    def _flush_file_item(self, *, item_id: str) -> None:
        item = self._files_by_item_id.pop(item_id, None)
        if item is None:
            return
        for url in item.urls:
            self._append_context_item(
                role="assistant",
                item={"type": "file", "url": url},
            )

    def _flush_tool_call(self, *, message: AgentToolCallEnded) -> None:
        item = self._tool_calls_by_item_id.pop(message.item_id, None)
        if item is None:
            item = _BufferedToolCall(
                namespace=message.namespace,
                call_id=message.call_id,
                toolkit="tool",
                tool="tool",
                arguments=None,
            )

        self._append_context_item(
            role="assistant",
            item={
                "type": "tool_call",
                "item_id": message.item_id,
                "namespace": item.namespace,
                "call_id": item.call_id,
                "toolkit": item.toolkit,
                "tool": item.tool,
                "arguments": item.arguments,
                "result": self._content_to_json(content=message.result),
                "error": None
                if message.error is None
                else message.error.model_dump(mode="json"),
                "logs": item.logs,
            },
        )

    def _append_context_item(self, *, role: str, item: dict[str, Any]) -> None:
        self._context.messages.append({"role": role, "content": [item]})

    @staticmethod
    def _content_to_json(*, content: Content | None) -> dict[str, Any] | None:
        if content is None:
            return None
        return content.to_json()
