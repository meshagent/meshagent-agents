from __future__ import annotations

from abc import ABC
import base64
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Awaitable, Callable, Protocol

from meshagent.api.agent_content import AgentFileContent, AgentTextContent
from meshagent.api.messaging import Content, FileContent

from .messages import (
    AgentAudioGenerationCompleted,
    AgentAudioGenerationDelta,
    AgentAudioGenerationFailed,
    AgentAudioGenerationStarted,
    AgentAudioTranscriptionCompleted,
    AgentAudioTranscriptionDelta,
    AgentAudioTranscriptionFailed,
    AgentAudioTranscriptionStarted,
    AgentContextCompacted,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentImageGenerationCompleted,
    AgentImageGenerationFailed,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentThreadEvent,
    AgentToolCallArgumentsDelta,
    AgentToolCallEnded,
    AgentToolCallLogDelta,
    AgentToolCallPending,
    AgentToolCallInProgress,
    AgentToolCallStarted,
    AgentUsageUpdated,
    TurnStart,
    TurnStartAccepted,
    TurnSteer,
    TurnSteerAccepted,
)
from .stream_content_accumulator import FileContentAccumulator, TextContentAccumulator


AgentFileReader = Callable[[str], Awaitable[FileContent | None]]


class AgentEventReader(Protocol):
    def __call__(self, message: AgentMessage) -> None: ...

    def consume(self, message: AgentMessage) -> None: ...

    async def finalize(self, *, file_reader: AgentFileReader | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentEventReaderCallbacks:
    record_event: Callable[[AgentMessage], None]
    update_usage: Callable[[AgentUsageUpdated], None]
    restore_compacted_context: Callable[[AgentContextCompacted], None]


@dataclass(slots=True)
class _BufferedToolCall:
    item_id: str
    namespace: str
    call_id: str | None
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None
    provider: str | None = None
    model: str | None = None
    argument_deltas: list[str] = field(default_factory=list)
    logs: list[dict[str, str]] = field(default_factory=list)

    def arguments_json(self) -> str:
        if self.argument_deltas:
            return "".join(self.argument_deltas)
        if self.arguments is None:
            return "{}"
        return json.dumps(self.arguments, separators=(",", ":"), ensure_ascii=False)

    def arguments_dict(self) -> dict[str, Any]:
        if self.arguments is not None:
            return deepcopy(self.arguments)
        raw_arguments = self.arguments_json()
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"input": raw_arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"input": parsed}


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


def _noop_agent_event_reader_callbacks() -> AgentEventReaderCallbacks:
    return AgentEventReaderCallbacks(
        record_event=lambda message: None,
        update_usage=lambda message: None,
        restore_compacted_context=lambda message: None,
    )


class AccumulatingAgentEventReader(ABC):
    def __init__(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
    ) -> None:
        self._emit_message = emit_message
        self._callbacks = callbacks or _noop_agent_event_reader_callbacks()
        self._text = TextContentAccumulator()
        self._reasoning = TextContentAccumulator()
        self._files = FileContentAccumulator()
        self._tool_calls_by_item_id: dict[str, _BufferedToolCall] = {}
        self._reasoning_metadata_by_item_id: dict[str, dict[str, Any]] = {}
        self._pending_file_resolutions: list[
            tuple[str, Callable[[FileContent], None]]
        ] = []

    def _emit_context_message(self, message: dict[str, Any]) -> dict[str, Any]:
        emitted_message = deepcopy(message)
        self._emit_message(emitted_message)
        return emitted_message

    def _defer_file_resolution(
        self,
        *,
        url: str,
        resolve: Callable[[FileContent], None],
    ) -> None:
        self._pending_file_resolutions.append((url, resolve))

    def __call__(self, message: AgentMessage) -> None:
        self.consume(message)

    def consume(self, message: AgentMessage) -> None:
        self._record_event(message=message)

        if (
            isinstance(message, (TurnStartAccepted, TurnSteerAccepted))
            and not message.content
        ):
            return

        if isinstance(
            message, (TurnStart, TurnSteer, TurnStartAccepted, TurnSteerAccepted)
        ):
            self._append_user_turn(message=message)
            return

        if isinstance(message, AgentTextContentStarted):
            self._text.upsert(
                item_id=message.item_id,
                turn_id=message.turn_id,
                sender_name=message.sender_name,
                phase=message.phase,
            )
            return

        if isinstance(message, AgentTextContentDelta):
            self._text.append_delta(
                item_id=message.item_id,
                delta=message.text,
                turn_id=message.turn_id,
                sender_name=message.sender_name,
                phase=message.phase,
            )
            return

        if isinstance(message, AgentTextContentEnded):
            self._flush_text_item(item_id=message.item_id)
            return

        if isinstance(message, AgentReasoningContentStarted):
            self._merge_reasoning_metadata(message=message)
            self._reasoning.upsert(
                item_id=message.item_id,
                turn_id=message.turn_id,
                sender_name=message.sender_name,
            )
            return

        if isinstance(message, AgentReasoningContentDelta):
            self._merge_reasoning_metadata(message=message)
            self._reasoning.append_delta(
                item_id=message.item_id,
                delta=message.text,
                turn_id=message.turn_id,
                sender_name=message.sender_name,
            )
            return

        if isinstance(message, AgentReasoningContentEnded):
            self._merge_reasoning_metadata(message=message)
            self._flush_reasoning_item(item_id=message.item_id)
            return

        if isinstance(message, AgentFileContentStarted):
            self._files.upsert(
                item_id=message.item_id,
                turn_id=message.turn_id,
            )
            return

        if isinstance(message, AgentFileContentDelta):
            self._files.append_url(
                item_id=message.item_id,
                url=message.url,
                turn_id=message.turn_id,
                sender_name=message.sender_name,
            )
            return

        if isinstance(message, AgentFileContentEnded):
            self._flush_file_item(item_id=message.item_id)
            return

        if isinstance(
            message,
            (AgentToolCallPending, AgentToolCallInProgress, AgentToolCallStarted),
        ):
            existing = self._tool_calls_by_item_id.get(message.item_id)
            self._tool_calls_by_item_id[message.item_id] = _BufferedToolCall(
                item_id=message.item_id,
                namespace=message.namespace,
                call_id=message.call_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                provider=message.provider,
                model=message.model,
                argument_deltas=[] if existing is None else existing.argument_deltas,
                logs=[] if existing is None else existing.logs,
            )
            return

        if isinstance(message, AgentToolCallArgumentsDelta):
            item = self._tool_calls_by_item_id.get(message.item_id)
            if item is None:
                item = _BufferedToolCall(
                    item_id=message.item_id,
                    namespace=message.namespace,
                    call_id=message.call_id,
                    toolkit="tool",
                    tool="tool",
                    arguments=None,
                    provider=message.provider,
                    model=message.model,
                )
                self._tool_calls_by_item_id[message.item_id] = item
            item.argument_deltas.append(message.delta)
            return

        if isinstance(message, AgentToolCallLogDelta):
            item = self._tool_calls_by_item_id.get(message.item_id)
            if item is None:
                item = _BufferedToolCall(
                    item_id=message.item_id,
                    namespace=message.namespace,
                    call_id=message.call_id,
                    toolkit="tool",
                    tool="tool",
                    arguments=None,
                    provider=message.provider,
                    model=message.model,
                )
                self._tool_calls_by_item_id[message.item_id] = item
            item.logs.extend([line.model_dump(mode="json") for line in message.lines])
            return

        if isinstance(message, AgentToolCallEnded):
            self._flush_tool_call(message=message)
            return

        if isinstance(message, AgentThreadEvent):
            self._append_thread_event(event=message.event)
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
            self._append_image_generation_event(
                event_type=message.type,
                turn_id=message.turn_id,
                item_id=message.item_id,
                call_id=message.call_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                images=[image.model_dump(mode="json") for image in message.images]
                if isinstance(message, AgentImageGenerationCompleted)
                else [],
                status=_image_generation_status(message=message),
            )
            return

        if isinstance(
            message,
            (
                AgentAudioGenerationStarted,
                AgentAudioGenerationDelta,
                AgentAudioGenerationCompleted,
                AgentAudioGenerationFailed,
            ),
        ):
            self._append_audio_generation_event(message=message)
            return

        if isinstance(
            message,
            (
                AgentAudioTranscriptionStarted,
                AgentAudioTranscriptionDelta,
                AgentAudioTranscriptionCompleted,
                AgentAudioTranscriptionFailed,
            ),
        ):
            self._append_audio_transcription_event(message=message)
            return

        if isinstance(message, AgentContextCompacted):
            self._flush_pending()
            if message.messages is not None:
                self._text.clear()
                self._reasoning.clear()
                self._reasoning_metadata_by_item_id.clear()
                self._files.clear()
                self._tool_calls_by_item_id.clear()
                self._pending_file_resolutions.clear()
            self._callbacks.restore_compacted_context(message)
            if message.messages is not None:
                self._restore_compacted_messages(messages=message.messages)
            return

        if isinstance(message, AgentUsageUpdated):
            self._callbacks.update_usage(message)
            return

    def _flush_pending(self) -> None:
        for item_id in list(self._text.item_ids()):
            self._flush_text_item(item_id=item_id)
        for item_id in list(self._reasoning.item_ids()):
            self._flush_reasoning_item(item_id=item_id)
        for item_id in list(self._files.item_ids()):
            self._flush_file_item(item_id=item_id)
        for item_id in list(self._tool_calls_by_item_id):
            item = self._tool_calls_by_item_id.pop(item_id)
            self._append_tool_call(
                tool_call=item,
                result=None,
                error={
                    "message": "tool call was cancelled before completion",
                    "code": "cancelled",
                },
            )

    async def finalize(self, *, file_reader: AgentFileReader | None = None) -> None:
        self._flush_pending()
        pending_file_resolutions = self._pending_file_resolutions
        self._pending_file_resolutions = []
        if file_reader is None:
            return
        for url, resolve in pending_file_resolutions:
            content = await file_reader(url)
            if content is not None:
                resolve(content)

    def _record_event(self, *, message: AgentMessage) -> None:
        self._callbacks.record_event(message)

    def _append_user_turn(
        self, *, message: TurnStart | TurnSteer | TurnStartAccepted | TurnSteerAccepted
    ) -> None:
        content: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in message.content:
            item_json = item.model_dump(mode="json")
            if isinstance(item, AgentTextContent):
                text_parts.append(item.text)
            elif isinstance(item, AgentFileContent):
                text_parts.append(f"attached file: {item.url}")
            content.append(item_json)

        if (
            len(content) == 1
            and len(text_parts) == 1
            and isinstance(message.content[0], AgentTextContent)
        ):
            self._append_user_text(text_parts[0])
            return

        self._append_user_content(content)

    def _flush_text_item(self, *, item_id: str) -> None:
        item = self._text.remove(item_id)
        if item is None or item.text == "":
            return
        self._append_assistant_text(text=item.text, phase=item.phase)

    def _flush_reasoning_item(self, *, item_id: str) -> None:
        item = self._reasoning.remove(item_id)
        metadata = self._reasoning_metadata_by_item_id.pop(item_id, {})
        if item is None:
            if len(metadata) > 0:
                self._append_assistant_reasoning(text="", metadata=metadata)
            return
        if item.text == "" and len(metadata) == 0:
            return
        self._append_assistant_reasoning(text=item.text, metadata=metadata)

    def _merge_reasoning_metadata(
        self,
        *,
        message: AgentReasoningContentStarted
        | AgentReasoningContentDelta
        | AgentReasoningContentEnded,
    ) -> None:
        if len(message.metadata) == 0:
            return
        metadata = self._reasoning_metadata_by_item_id.setdefault(message.item_id, {})
        metadata.update(deepcopy(message.metadata))

    def _flush_file_item(self, *, item_id: str) -> None:
        item = self._files.remove(item_id)
        if item is None:
            return
        for url in item.urls:
            self._append_assistant_file(url=url)

    def _flush_tool_call(self, *, message: AgentToolCallEnded) -> None:
        item = self._tool_calls_by_item_id.pop(message.item_id, None)
        if item is None:
            item = _BufferedToolCall(
                item_id=message.item_id,
                namespace=message.namespace,
                call_id=message.call_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=None,
                provider=message.provider,
                model=message.model,
            )

        self._append_tool_call(
            tool_call=item,
            result=self._content_to_json(content=message.result),
            error=None
            if message.error is None
            else message.error.model_dump(mode="json"),
        )

    @staticmethod
    def _content_to_json(*, content: Content | None) -> dict[str, Any] | None:
        if content is None:
            return None
        return content.to_json()

    @staticmethod
    def _result_text(
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        logs: list[dict[str, str]],
    ) -> str:
        if error is not None:
            return json.dumps(
                {"error": error}, separators=(",", ":"), ensure_ascii=False
            )
        if result is not None:
            result_type = result.get("type")
            if result_type == "text":
                text = result.get("text")
                if isinstance(text, str):
                    return text
            if result_type == "empty":
                return "ok"
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        if logs:
            return "\n".join(
                line["text"] for line in logs if isinstance(line.get("text"), str)
            )
        return ""

    def _append_user_text(self, text: str) -> None:
        self._emit_context_message({"role": "user", "content": text})

    def _append_user_content(self, content: list[dict[str, Any]]) -> None:
        emitted_message = self._emit_context_message(
            {"role": "user", "content": content}
        )
        emitted_content = emitted_message["content"]
        for item in emitted_content:
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            url = item.get("url")
            if not isinstance(url, str) or url.startswith("data:"):
                continue

            def resolve_file(
                file_content: FileContent,
                *,
                item: dict[str, Any] = item,
            ) -> None:
                mime_type = file_content.mime_type or "application/octet-stream"
                item["url"] = (
                    f"data:{mime_type};base64,"
                    f"{base64.b64encode(file_content.data).decode('ascii')}"
                )
                item["name"] = file_content.name
                item["mime_type"] = mime_type

            self._defer_file_resolution(url=url, resolve=resolve_file)

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if phase is not None:
            message["phase"] = phase
        self._emit_context_message(message)

    def _append_assistant_reasoning(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        self._emit_context_message({"role": "assistant", "reasoning": text})

    def _append_assistant_file(self, *, url: str) -> None:
        self._emit_context_message({"role": "assistant", "file": url})

    def _append_thread_event(self, *, event: dict[str, Any]) -> None:
        self._emit_context_message({"role": "assistant", "event": event})

    def _append_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "tool_call": {
                    "item_id": tool_call.item_id,
                    "namespace": tool_call.namespace,
                    "call_id": tool_call.call_id,
                    "toolkit": tool_call.toolkit,
                    "tool": tool_call.tool,
                    "arguments_json": tool_call.arguments_json(),
                    "arguments": tool_call.arguments_dict(),
                    "result": result,
                    "error": error,
                    "logs": tool_call.logs,
                },
            }
        )

    def _append_image_generation_event(
        self,
        *,
        event_type: str,
        turn_id: str,
        item_id: str,
        call_id: str | None,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        images: list[dict[str, Any]],
        status: str,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "image_generation": {
                    "event_type": event_type,
                    "turn_id": turn_id,
                    "item_id": item_id,
                    "call_id": call_id,
                    "toolkit": toolkit,
                    "tool": tool,
                    "arguments": arguments,
                    "images": images,
                    "status": status,
                },
            }
        )

    def _append_audio_generation_event(
        self,
        *,
        message: AgentAudioGenerationStarted
        | AgentAudioGenerationDelta
        | AgentAudioGenerationCompleted
        | AgentAudioGenerationFailed,
    ) -> None:
        self._emit_context_message(
            {"role": "assistant", "audio_generation": message.model_dump(mode="json")}
        )

    def _append_audio_transcription_event(
        self,
        *,
        message: AgentAudioTranscriptionStarted
        | AgentAudioTranscriptionDelta
        | AgentAudioTranscriptionCompleted
        | AgentAudioTranscriptionFailed,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "audio_transcription": message.model_dump(mode="json"),
            }
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)
