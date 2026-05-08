from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from meshagent.api.messaging import BinaryContent, Content, JsonContent, TextContent

from .messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_REASONING_CONTENT_DELTA,
    AGENT_EVENT_REASONING_CONTENT_ENDED,
    AGENT_EVENT_REASONING_CONTENT_STARTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_TOOL_CALL_PENDING,
    AGENT_EVENT_TOOL_CALL_IN_PROGRESS,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
    AGENT_EVENT_IMAGE_GENERATION_FAILED,
    AGENT_EVENT_IMAGE_GENERATION_PARTIAL,
    AGENT_EVENT_IMAGE_GENERATION_STARTED,
    AgentError,
    AgentContextCompacted,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentGeneratedImage,
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
    AgentToolCallPending,
    AgentToolCallInProgress,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallEnded,
    AgentToolCallStarted,
)

AgentEventCallback = Callable[[AgentMessage], None]
FunctionToolNameResolver = Callable[[str], tuple[str, str] | None]
_ContentKind = Literal["file", "reasoning", "text"]
_MessagePhase = Literal["commentary", "final_answer"]
_MESHAGENT_TOOL_NAMESPACE = "meshagent"
_OPENAI_RESPONSES_TOOL_NAMESPACE = "openai.responses"
_ANTHROPIC_MESSAGES_TOOL_NAMESPACE = "anthropic.messages"
_DEFERRED_HANDLER_RESULT_TOOL_TYPES = {
    "apply_patch_call",
    "computer_call",
    "local_shell_call",
    "shell_call",
}
_HANDLER_RESULT_TOOL_TYPES = _DEFERRED_HANDLER_RESULT_TOOL_TYPES | {"function_call"}


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped != "" else None
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _parse_tool_arguments(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": stripped}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return None


def _parse_tool_log_lines(value: Any) -> list[AgentToolCallLogLine]:
    if not isinstance(value, list):
        return []

    parsed_lines: list[AgentToolCallLogLine] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue

        source = _as_str(entry.get("source"))
        text = _as_text(entry.get("text"))
        if source not in {"stdout", "stderr"} or text is None:
            continue

        parsed_lines.append(AgentToolCallLogLine(source=source, text=text))

    return parsed_lines


def _data_url(*, mime_type: str, data: str) -> str:
    return f"data:{mime_type};base64,{data}"


def _file_url_from_payload(
    *,
    payload: dict[str, Any],
    provider: Literal["anthropic", "openai"],
) -> str | None:
    direct_url_keys = ("url", "file_url")
    for key in direct_url_keys:
        url = _as_str(payload.get(key))
        if url is not None:
            return url

    nested_url_keys = ("image_url", "file", "source")
    for key in nested_url_keys:
        nested = _as_dict(payload.get(key))
        if nested is None:
            continue
        url = _as_str(nested.get("url"))
        if url is not None:
            return url
        nested_type = _as_str(nested.get("type"))
        if nested_type == "base64":
            media_type = _as_str(nested.get("media_type")) or _as_str(
                nested.get("mime_type")
            )
            data = _as_str(nested.get("data"))
            if media_type is not None and data is not None:
                return _data_url(mime_type=media_type, data=data)

    direct_base64_keys = (
        "partial_image_b64",
        "image_base64",
        "image_b64",
        "b64_json",
        "data",
    )
    media_type = _as_str(payload.get("mime_type")) or _as_str(payload.get("media_type"))
    for key in direct_base64_keys:
        data = _as_str(payload.get(key))
        if data is not None and media_type is not None:
            return _data_url(mime_type=media_type, data=data)

    file_id = _as_str(payload.get("file_id"))
    if file_id is not None:
        return f"{provider}://file/{file_id}"

    return None


def _part_text(part: dict[str, Any]) -> str:
    text = _as_text(part.get("text"))
    if text is not None:
        return text

    refusal = _as_text(part.get("refusal"))
    if refusal is not None:
        return refusal

    return ""


def _content_kind_from_part(part: dict[str, Any]) -> _ContentKind | None:
    part_type = (_as_str(part.get("type")) or "").lower()

    if part_type in {"output_text", "refusal", "text"}:
        return "text"

    if part_type in {"reasoning_text", "summary_text"} or part_type.startswith(
        "reasoning"
    ):
        return "reasoning"

    if part_type in {
        "document",
        "file",
        "image",
        "input_image",
        "output_file",
        "output_image",
    }:
        return "file"

    if _file_url_from_payload(payload=part, provider="openai") is not None:
        return "file"

    return None


def _message_phase_from_value(value: Any) -> _MessagePhase | None:
    phase = _as_str(value)
    if phase is None:
        return None
    normalized = phase.strip().lower()
    if normalized == "commentary":
        return "commentary"
    if normalized in {"final_answer", "answer"}:
        return "final_answer"
    return None


def _message_phase_from_event(*, event: dict[str, Any]) -> _MessagePhase | None:
    phase = _message_phase_from_value(event.get("phase"))
    if phase is not None:
        return phase

    item = _as_dict(event.get("item"))
    if item is not None:
        phase = _message_phase_from_value(item.get("phase"))
        if phase is not None:
            return phase

    response = _as_dict(event.get("response"))
    if response is not None:
        phase = _message_phase_from_value(response.get("phase"))
        if phase is not None:
            return phase

    return None


def _mime_type_from_image_output_format(output_format: Any) -> str:
    normalized = _as_str(output_format)
    if normalized is None:
        return "image/png"

    normalized = normalized.lower().lstrip(".")
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized == "":
        return "image/png"
    return f"image/{normalized}"


def _content_from_image_generation_item(item: dict[str, Any]) -> BinaryContent | None:
    headers: dict[str, str] = {
        "mime_type": _mime_type_from_image_output_format(item.get("output_format"))
    }
    for key in ("background", "output_format", "quality", "size", "status"):
        value = _as_str(item.get(key))
        if value is not None:
            headers[key] = value

    for key in ("result", "image_base64", "image_b64", "b64_json", "data"):
        value = item.get(key)
        if isinstance(value, bytes):
            return BinaryContent(data=value, headers=headers)
        if isinstance(value, bytearray):
            return BinaryContent(data=bytes(value), headers=headers)
        if isinstance(value, str) and value.strip() != "":
            try:
                return BinaryContent(
                    data=base64.b64decode(value),
                    headers=headers,
                )
            except Exception:
                return None
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, str) or entry.strip() == "":
                    continue
                try:
                    return BinaryContent(
                        data=base64.b64decode(entry),
                        headers=headers,
                    )
                except Exception:
                    return None

    return None


def _content_from_tool_item(item: dict[str, Any]):
    item_type = _as_str(item.get("type"))
    if item_type == "image_generation_call":
        return _content_from_image_generation_item(item)

    result = item.get("result")
    if isinstance(result, dict):
        return JsonContent(json=result)
    if isinstance(result, list):
        return JsonContent(json={"result": result})
    if isinstance(result, str) and result.strip() != "":
        return TextContent(text=result)

    output = item.get("output")
    if isinstance(output, dict):
        return JsonContent(json=output)
    if isinstance(output, list):
        return JsonContent(json={"output": output})
    if isinstance(output, str) and output.strip() != "":
        return TextContent(text=output)

    results = item.get("results")
    if isinstance(results, dict):
        return JsonContent(json=results)
    if isinstance(results, list):
        return JsonContent(json={"results": results})

    return None


def _content_from_handler_result(value: Any):
    if isinstance(value, dict):
        return JsonContent(json=value)
    if isinstance(value, list):
        return JsonContent(json={"result": value})
    if isinstance(value, str) and value.strip() != "":
        return TextContent(text=value)
    return None


def _generated_image_from_value(value: Any) -> AgentGeneratedImage | None:
    if not isinstance(value, dict):
        return None
    uri = _as_str(value.get("uri"))
    if uri is None:
        return None
    return AgentGeneratedImage(
        uri=uri,
        mime_type=_as_str(value.get("mime_type")),
        created_at=_as_str(value.get("created_at")),
        created_by=_as_str(value.get("created_by")),
        width=value.get("width")
        if isinstance(value.get("width"), (int, float))
        else None,
        height=value.get("height")
        if isinstance(value.get("height"), (int, float))
        else None,
        status=_as_str(value.get("status")),
        status_detail=_as_str(value.get("status_detail")),
    )


def _mime_type_from_image_generation_item(item: dict[str, Any]) -> str:
    output_format = _as_str(item.get("output_format"))
    if output_format is None:
        return "image/png"
    normalized = output_format.strip().lower()
    if normalized == "":
        return "image/png"
    if normalized == "jpg":
        normalized = "jpeg"
    return f"image/{normalized}"


def _image_generation_dimensions(item: dict[str, Any]) -> tuple[int | None, int | None]:
    size = _as_str(item.get("size"))
    if size is None:
        return None, None
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", size)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _image_generation_arguments(item: dict[str, Any]) -> dict[str, Any] | None:
    arguments = _filtered_tool_arguments(
        item=item,
        excluded_keys={
            "id",
            "images",
            "partial_image_b64",
            "partial_image_index",
            "result",
            "type",
        },
    )
    return arguments if arguments is not None and len(arguments) > 0 else None


def _generated_image_from_result_item(
    item: dict[str, Any],
) -> AgentGeneratedImage | None:
    result = _as_str(item.get("result"))
    if result is None or result.strip() == "":
        return None
    mime_type = _mime_type_from_image_generation_item(item)
    width, height = _image_generation_dimensions(item)
    uri = result.strip()
    if re.match(r"^[a-z][a-z0-9+.-]*:", uri) is None:
        uri = f"data:{mime_type};base64,{uri}"
    return AgentGeneratedImage(
        uri=uri,
        mime_type=mime_type,
        width=width,
        height=height,
        status=_as_str(item.get("status")) or "completed",
    )


def _generated_images_from_item(item: dict[str, Any]) -> list[AgentGeneratedImage]:
    images = item.get("images")
    if not isinstance(images, list):
        generated_image = _generated_image_from_result_item(item)
        return [generated_image] if generated_image is not None else []
    generated: list[AgentGeneratedImage] = []
    for image in images:
        generated_image = _generated_image_from_value(image)
        if generated_image is not None:
            generated.append(generated_image)
    return generated


@dataclass(frozen=True, slots=True)
class _ToolCallInfo:
    item_id: str
    namespace: str
    toolkit: str
    tool: str
    item_type: str | None = None
    call_id: str | None = None
    arguments: dict[str, Any] | None = None
    error: AgentError | None = None
    result: Content | None = None


def _filtered_tool_arguments(
    *,
    item: dict[str, Any],
    excluded_keys: set[str],
) -> dict[str, Any] | None:
    arguments: dict[str, Any] = {}
    for key, value in item.items():
        if key in excluded_keys or value is None:
            continue
        arguments[key] = value
    return arguments or None


def _openai_tool_call_info(
    item: dict[str, Any],
    *,
    provider_tool_namespace: str = _OPENAI_RESPONSES_TOOL_NAMESPACE,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
) -> _ToolCallInfo | None:
    item_type = _as_str(item.get("type"))
    item_id = _as_str(item.get("id"))
    if item_type is None or item_id is None:
        return None

    status = _as_str(item.get("status"))
    error_text = _as_str(item.get("error"))
    error = None
    if error_text is not None:
        error = AgentError(message=error_text, code=status or "tool_call_failed")

    if item_type == "function_call":
        safe_name = _as_str(item.get("name"))
        toolkit_name = "function"
        tool_name = safe_name or "function"
        if function_tool_name_resolver is not None and safe_name is not None:
            resolved = function_tool_name_resolver(safe_name)
            if resolved is not None:
                toolkit_name, tool_name = resolved
        return _ToolCallInfo(
            item_id=item_id,
            namespace=_MESHAGENT_TOOL_NAMESPACE,
            toolkit=toolkit_name,
            tool=tool_name,
            item_type=item_type,
            call_id=_as_str(item.get("call_id")),
            arguments=_parse_tool_arguments(item.get("arguments")),
            error=error,
            result=_content_from_tool_item(item),
        )

    if item_type == "mcp_call":
        return _ToolCallInfo(
            item_id=item_id,
            namespace=provider_tool_namespace,
            toolkit=_as_str(item.get("server_label")) or "mcp",
            tool=_as_str(item.get("name")) or "call",
            item_type=item_type,
            call_id=_as_str(item.get("call_id")),
            arguments=_parse_tool_arguments(item.get("arguments")),
            error=error,
            result=_content_from_tool_item(item),
        )

    if item_type == "mcp_list_tools":
        return _ToolCallInfo(
            item_id=item_id,
            namespace=provider_tool_namespace,
            toolkit=_as_str(item.get("server_label")) or "mcp",
            tool="list_tools",
            item_type=item_type,
            call_id=_as_str(item.get("call_id")),
            arguments=_filtered_tool_arguments(
                item=item,
                excluded_keys={"error", "id", "output", "status", "type"},
            ),
            error=error,
            result=_content_from_tool_item(item),
        )

    if not item_type.endswith("_call"):
        return None

    return _ToolCallInfo(
        item_id=item_id,
        namespace=provider_tool_namespace,
        toolkit="openai",
        tool=item_type.removesuffix("_call"),
        item_type=item_type,
        call_id=_as_str(item.get("call_id")),
        arguments=_filtered_tool_arguments(
            item=item,
            excluded_keys={
                "call_id",
                "error",
                "id",
                "output",
                "result",
                "results",
                "status",
                "type",
            },
        ),
        error=error,
        result=_content_from_tool_item(item),
    )


def _anthropic_tool_call_info(
    *,
    block: dict[str, Any],
    item_id: str,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
) -> _ToolCallInfo | None:
    block_type = _as_str(block.get("type"))
    if block_type is None:
        return None

    if block_type == "tool_use":
        safe_name = _as_str(block.get("name"))
        toolkit_name = "function"
        tool_name = safe_name or "tool"
        if function_tool_name_resolver is not None and safe_name is not None:
            resolved = function_tool_name_resolver(safe_name)
            if resolved is not None:
                toolkit_name, tool_name = resolved
        return _ToolCallInfo(
            item_id=item_id,
            namespace=_MESHAGENT_TOOL_NAMESPACE,
            toolkit=toolkit_name,
            tool=tool_name,
            item_type=block_type,
            call_id=_as_str(block.get("id")),
            arguments=_parse_tool_arguments(block.get("input")),
        )

    if block_type == "mcp_tool_use":
        return _ToolCallInfo(
            item_id=item_id,
            namespace=_ANTHROPIC_MESSAGES_TOOL_NAMESPACE,
            toolkit=(
                _as_str(block.get("server_name"))
                or _as_str(block.get("server_label"))
                or "mcp"
            ),
            tool=_as_str(block.get("name")) or "tool",
            item_type=block_type,
            call_id=_as_str(block.get("id")),
            arguments=_parse_tool_arguments(
                block.get("input")
                if block.get("input") is not None
                else block.get("arguments")
            ),
        )

    if block_type.endswith("_tool_use"):
        return _ToolCallInfo(
            item_id=item_id,
            namespace=_ANTHROPIC_MESSAGES_TOOL_NAMESPACE,
            toolkit="anthropic",
            tool=block_type.removesuffix("_tool_use"),
            item_type=block_type,
            call_id=_as_str(block.get("id")),
            arguments=_filtered_tool_arguments(
                item=block,
                excluded_keys={"id", "name", "type"},
            ),
        )

    return None


@dataclass(slots=True)
class _AgentMessageEmitter:
    turn_id: str
    thread_id: str
    callback: AgentEventCallback
    _started_content: set[tuple[_ContentKind, str]] = field(default_factory=set)
    _ended_content: set[tuple[_ContentKind, str]] = field(default_factory=set)
    _content_with_data: set[tuple[_ContentKind, str]] = field(default_factory=set)
    _pending_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)
    _in_progress_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)
    _started_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)
    _ended_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)

    def _content_key(
        self, *, kind: _ContentKind, item_id: str
    ) -> tuple[_ContentKind, str]:
        return kind, item_id

    def _ensure_content_started(self, *, kind: _ContentKind, item_id: str) -> None:
        key = self._content_key(kind=kind, item_id=item_id)
        if key in self._started_content:
            return

        self._started_content.add(key)
        if kind == "text":
            self.callback(
                AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id=self.thread_id,
                    turn_id=self.turn_id,
                    item_id=item_id,
                )
            )
            return

        if kind == "reasoning":
            self.callback(
                AgentReasoningContentStarted(
                    type=AGENT_EVENT_REASONING_CONTENT_STARTED,
                    thread_id=self.thread_id,
                    turn_id=self.turn_id,
                    item_id=item_id,
                )
            )
            return

        self.callback(
            AgentFileContentStarted(
                type=AGENT_EVENT_FILE_CONTENT_STARTED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
            )
        )

    def _mark_content_has_data(self, *, kind: _ContentKind, item_id: str) -> None:
        self._content_with_data.add(self._content_key(kind=kind, item_id=item_id))

    def has_content_data(self, *, kind: _ContentKind, item_id: str) -> bool:
        return self._content_key(kind=kind, item_id=item_id) in self._content_with_data

    def emit_text_delta(
        self, *, item_id: str, text: str, phase: _MessagePhase | None = None
    ) -> None:
        if text == "":
            return
        self.emit_text_started(item_id=item_id, phase=phase)
        self._mark_content_has_data(kind="text", item_id=item_id)
        self.callback(
            AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                text=text,
                phase=phase,
            )
        )

    def emit_reasoning_delta(self, *, item_id: str, text: str) -> None:
        if text == "":
            return
        self._ensure_content_started(kind="reasoning", item_id=item_id)
        self._mark_content_has_data(kind="reasoning", item_id=item_id)
        self.callback(
            AgentReasoningContentDelta(
                type=AGENT_EVENT_REASONING_CONTENT_DELTA,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                text=text,
            )
        )

    def emit_file_delta(self, *, item_id: str, url: str | None) -> None:
        if url is None:
            return
        self._ensure_content_started(kind="file", item_id=item_id)
        self._mark_content_has_data(kind="file", item_id=item_id)
        self.callback(
            AgentFileContentDelta(
                type=AGENT_EVENT_FILE_CONTENT_DELTA,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                url=url,
            )
        )

    def emit_text_started(
        self, *, item_id: str, phase: _MessagePhase | None = None
    ) -> None:
        key = self._content_key(kind="text", item_id=item_id)
        if key in self._started_content:
            return

        self._started_content.add(key)
        self.callback(
            AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                phase=phase,
            )
        )

    def emit_reasoning_started(self, *, item_id: str) -> None:
        self._ensure_content_started(kind="reasoning", item_id=item_id)

    def emit_file_started(self, *, item_id: str) -> None:
        self._ensure_content_started(kind="file", item_id=item_id)

    def emit_text_ended(
        self, *, item_id: str, phase: _MessagePhase | None = None
    ) -> None:
        key = self._content_key(kind="text", item_id=item_id)
        if key in self._ended_content:
            return
        self.emit_text_started(item_id=item_id, phase=phase)
        self._ended_content.add(key)
        self.callback(
            AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                phase=phase,
            )
        )

    def emit_reasoning_ended(self, *, item_id: str) -> None:
        key = self._content_key(kind="reasoning", item_id=item_id)
        if key in self._ended_content:
            return
        self._ensure_content_started(kind="reasoning", item_id=item_id)
        self._ended_content.add(key)
        self.callback(
            AgentReasoningContentEnded(
                type=AGENT_EVENT_REASONING_CONTENT_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
            )
        )

    def emit_file_ended(self, *, item_id: str) -> None:
        key = self._content_key(kind="file", item_id=item_id)
        if key in self._ended_content:
            return
        self._ensure_content_started(kind="file", item_id=item_id)
        self._ended_content.add(key)
        self.callback(
            AgentFileContentEnded(
                type=AGENT_EVENT_FILE_CONTENT_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
            )
        )

    def emit_tool_pending(self, *, info: _ToolCallInfo) -> None:
        existing = self._pending_tool_calls.get(info.item_id)
        if (
            existing is not None
            and existing.namespace == info.namespace
            and existing.call_id == info.call_id
            and existing.toolkit == info.toolkit
            and existing.tool == info.tool
            and existing.arguments == info.arguments
        ):
            return
        self._pending_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallPending(
                type=AGENT_EVENT_TOOL_CALL_PENDING,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                namespace=info.namespace,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments,
            )
        )

    def emit_tool_in_progress(self, *, info: _ToolCallInfo) -> None:
        existing = self._in_progress_tool_calls.get(info.item_id)
        if (
            existing is not None
            and existing.namespace == info.namespace
            and existing.call_id == info.call_id
            and existing.toolkit == info.toolkit
            and existing.tool == info.tool
            and existing.arguments == info.arguments
        ):
            return
        self._pending_tool_calls.pop(info.item_id, None)
        self._in_progress_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallInProgress(
                type=AGENT_EVENT_TOOL_CALL_IN_PROGRESS,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                namespace=info.namespace,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments,
            )
        )

    def emit_tool_started(self, *, info: _ToolCallInfo) -> None:
        existing = self._started_tool_calls.get(info.item_id)
        if (
            existing is not None
            and existing.namespace == info.namespace
            and existing.call_id == info.call_id
            and existing.toolkit == info.toolkit
            and existing.tool == info.tool
            and existing.arguments == info.arguments
        ):
            return
        self._pending_tool_calls.pop(info.item_id, None)
        self._in_progress_tool_calls.pop(info.item_id, None)
        self._started_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                namespace=info.namespace,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments,
            )
        )

    def emit_tool_log_delta(
        self,
        *,
        item_id: str,
        lines: list[AgentToolCallLogLine],
    ) -> None:
        if len(lines) == 0:
            return
        info = (
            self._started_tool_calls.get(item_id)
            or self._in_progress_tool_calls.get(item_id)
            or self._pending_tool_calls.get(item_id)
        )
        self.callback(
            AgentToolCallLogDelta(
                type=AGENT_EVENT_TOOL_CALL_LOG_DELTA,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                namespace=(
                    info.namespace if info is not None else _MESHAGENT_TOOL_NAMESPACE
                ),
                call_id=info.call_id if info is not None else None,
                lines=lines,
            )
        )

    def emit_tool_ended(self, *, info: _ToolCallInfo) -> None:
        self.emit_tool_started(info=info)
        existing = self._ended_tool_calls.get(info.item_id)
        if (
            existing is not None
            and existing.result == info.result
            and existing.error == info.error
        ):
            return
        self._pending_tool_calls.pop(info.item_id, None)
        self._in_progress_tool_calls.pop(info.item_id, None)
        self._started_tool_calls.pop(info.item_id, None)
        self._ended_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                namespace=info.namespace,
                call_id=info.call_id,
                result=info.result,
                error=info.error,
            )
        )


@dataclass(slots=True)
class _OpenAIAgentEventPublisher:
    emitter: _AgentMessageEmitter
    provider_tool_namespace: str = _OPENAI_RESPONSES_TOOL_NAMESPACE
    function_tool_name_resolver: FunctionToolNameResolver | None = None
    custom_event_callback: Callable[[dict[str, Any]], None] | None = None
    _active_response_id: str | None = None
    _output_item_ids: dict[int, str] = field(default_factory=dict)
    _pending_handler_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)
    _finished_handler_tool_call_ids: set[str] = field(default_factory=set)
    _started_image_generation_ids: set[str] = field(default_factory=set)
    _emitted_compaction_ids: set[str] = field(default_factory=set)
    _synthetic_item_counter: int = 0

    def set_function_tool_name_resolver(
        self,
        resolver: FunctionToolNameResolver | None,
    ) -> None:
        self.function_tool_name_resolver = resolver

    def _track_response_boundary(self, *, event: dict[str, Any]) -> None:
        response = _as_dict(event.get("response"))
        if response is None:
            return

        response_id = _as_str(response.get("id"))
        if response_id is None:
            return

        if self._active_response_id is None:
            self._active_response_id = response_id
            return

        if self._active_response_id == response_id:
            return

        self._active_response_id = response_id
        self._output_item_ids.clear()
        self._emitted_compaction_ids.clear()

    def _response_id_from_event(self, *, event: dict[str, Any]) -> str | None:
        response_id = _as_str(event.get("response_id"))
        if response_id is not None:
            return response_id
        response = _as_dict(event.get("response"))
        if response is None:
            return None
        return _as_str(response.get("id"))

    def _item_id_from_event(self, *, event: dict[str, Any]) -> str | None:
        output_index = _as_int(event.get("output_index"))
        item_id = _as_str(event.get("item_id"))

        item = _as_dict(event.get("item"))
        if item is not None and item_id is None:
            item_id = _as_str(item.get("id"))

        if output_index is not None:
            return self._mapped_output_item_id(
                output_index=output_index,
                item_id=item_id,
            )

        return item_id

    def _synthetic_item_id(self, *, event: dict[str, Any], kind: str) -> str:
        response_id = self._active_response_id
        if response_id is None:
            response_id = self._response_id_from_event(event=event)
        sequence_number = _as_int(event.get("sequence_number"))
        if sequence_number is not None:
            return f"{kind}:{response_id or self.emitter.turn_id}:{sequence_number}"
        self._synthetic_item_counter += 1
        return f"{kind}:{response_id or self.emitter.turn_id}:{self._synthetic_item_counter}"

    def _mapped_output_item_id(
        self,
        *,
        output_index: int,
        item_id: str | None = None,
    ) -> str:
        existing = self._output_item_ids.get(output_index)
        if existing is not None:
            return existing

        if item_id is not None:
            self._output_item_ids[output_index] = item_id
            return item_id

        synthesized = f"output:{output_index}"
        self._output_item_ids[output_index] = synthesized
        return synthesized

    def _record_output_item(
        self, *, event: dict[str, Any], item: dict[str, Any]
    ) -> None:
        output_index = _as_int(event.get("output_index"))
        item_id = _as_str(item.get("id"))
        if output_index is None or item_id is None:
            return
        self._mapped_output_item_id(output_index=output_index, item_id=item_id)

    def _finish_part_from_snapshot(
        self,
        *,
        item_id: str,
        part: dict[str, Any],
        phase: _MessagePhase | None = None,
    ) -> None:
        content_kind = _content_kind_from_part(part)
        if content_kind == "text":
            text = _part_text(part)
            if text != "" and not self.emitter.has_content_data(
                kind="text", item_id=item_id
            ):
                self.emitter.emit_text_delta(item_id=item_id, text=text, phase=phase)
            self.emitter.emit_text_ended(item_id=item_id, phase=phase)
            return

        if content_kind == "reasoning":
            text = _part_text(part)
            if text != "" and not self.emitter.has_content_data(
                kind="reasoning", item_id=item_id
            ):
                self.emitter.emit_reasoning_delta(item_id=item_id, text=text)
            self.emitter.emit_reasoning_ended(item_id=item_id)
            return

        if content_kind == "file":
            url = _file_url_from_payload(payload=part, provider="openai")
            if not self.emitter.has_content_data(kind="file", item_id=item_id):
                self.emitter.emit_file_delta(item_id=item_id, url=url)
            self.emitter.emit_file_ended(item_id=item_id)

    def _on_output_item(self, *, event: dict[str, Any], completed: bool) -> None:
        item = _as_dict(event.get("item"))
        if item is None:
            return

        self._record_output_item(event=event, item=item)

        tool_info = _openai_tool_call_info(
            item,
            provider_tool_namespace=self.provider_tool_namespace,
            function_tool_name_resolver=self.function_tool_name_resolver,
        )
        if tool_info is not None:
            if tool_info.item_id in self._finished_handler_tool_call_ids:
                return

            if tool_info.item_type == "image_generation_call":
                if completed:
                    self._emit_image_generation_completed(info=tool_info, item=item)
                else:
                    self._emit_image_generation_started(info=tool_info, item=item)
                return

            if tool_info.item_type in _HANDLER_RESULT_TOOL_TYPES:
                self._pending_handler_tool_calls[tool_info.item_id] = tool_info
                self.emitter.emit_tool_pending(info=tool_info)
                return

            if completed:
                self.emitter.emit_tool_ended(info=tool_info)
            else:
                self.emitter.emit_tool_started(info=tool_info)
            return

        item_type = _as_str(item.get("type"))
        item_id = self._item_id_from_event(event=event)
        message_phase = _message_phase_from_event(event=event)
        if item_type == "reasoning" and item_id is not None:
            if not completed:
                self.emitter.emit_reasoning_started(item_id=item_id)
                return

            if not self.emitter.has_content_data(kind="reasoning", item_id=item_id):
                for bucket_name in ("summary", "content"):
                    bucket = item.get(bucket_name)
                    if not isinstance(bucket, list):
                        continue
                    for part in bucket:
                        if not isinstance(part, dict):
                            continue
                        text = _part_text(part)
                        if text != "":
                            self.emitter.emit_reasoning_delta(
                                item_id=item_id, text=text
                            )
            self.emitter.emit_reasoning_ended(item_id=item_id)
            return

        if item_type == "compaction":
            if item_id is None:
                item_id = self._synthetic_item_id(event=event, kind="compaction")
                item = {**item, "id": item_id}
            elif _as_str(item.get("id")) is None:
                item = {**item, "id": item_id}
            if completed and item_id in self._emitted_compaction_ids:
                return
            if not completed:
                self.emitter.callback(
                    AgentThreadEvent(
                        type=AGENT_EVENT_THREAD_EVENT,
                        thread_id=self.emitter.thread_id,
                        event={
                            "type": "agent.event",
                            "source": "openai",
                            "name": "openai.context_compaction",
                            "kind": "message",
                            "state": "in_progress",
                            "headline": "Compacting context",
                            "summary": "Compacting context",
                            "item_id": item_id,
                        },
                    )
                )
                return
            self._emitted_compaction_ids.add(item_id)
            self.emitter.callback(
                AgentContextCompacted(
                    type=AGENT_EVENT_CONTEXT_COMPACTED,
                    thread_id=self.emitter.thread_id,
                    checkpoint_id=item_id,
                    path=self.emitter.thread_id,
                    through_sequence=0,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    messages=[item],
                )
            )
            self.emitter.callback(
                AgentThreadEvent(
                    type=AGENT_EVENT_THREAD_EVENT,
                    thread_id=self.emitter.thread_id,
                    event={
                        "type": "agent.event",
                        "source": "openai",
                        "name": "openai.context_compaction",
                        "kind": "message",
                        "state": "completed",
                        "headline": "Compacted context",
                        "summary": "Compacted context",
                        "item_id": item_id,
                    },
                )
            )
            return

        if item_type == "message" and completed and item_id is not None:
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        self._finish_part_from_snapshot(
                            item_id=item_id, part=part, phase=message_phase
                        )

    def _image_generation_info_from_event(
        self,
        *,
        event: dict[str, Any],
    ) -> _ToolCallInfo | None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return None
        item = dict(event)
        item["type"] = "image_generation_call"
        item["id"] = item_id
        return _openai_tool_call_info(
            item,
            provider_tool_namespace=self.provider_tool_namespace,
            function_tool_name_resolver=self.function_tool_name_resolver,
        )

    def _emit_image_generation_started(
        self,
        *,
        info: _ToolCallInfo,
        item: dict[str, Any],
    ) -> None:
        if info.item_id in self._started_image_generation_ids:
            return
        self._started_image_generation_ids.add(info.item_id)
        self.emitter.callback(
            AgentImageGenerationStarted(
                type=AGENT_EVENT_IMAGE_GENERATION_STARTED,
                thread_id=self.emitter.thread_id,
                turn_id=self.emitter.turn_id,
                item_id=info.item_id,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments or _image_generation_arguments(item),
                status_detail="Generating image",
            )
        )

    def _emit_image_generation_partial(
        self,
        *,
        info: _ToolCallInfo,
        event: dict[str, Any],
    ) -> None:
        self._started_image_generation_ids.add(info.item_id)
        partial_image = _as_str(event.get("partial_image_b64"))
        mime_type = _mime_type_from_image_generation_item(event)
        width, height = _image_generation_dimensions(event)
        image = None
        if partial_image is not None and partial_image.strip() != "":
            image = AgentGeneratedImage(
                uri=f"data:{mime_type};base64,{partial_image.strip()}",
                mime_type=mime_type,
                width=width,
                height=height,
                status="in_progress",
            )
        self.emitter.callback(
            AgentImageGenerationPartial(
                type=AGENT_EVENT_IMAGE_GENERATION_PARTIAL,
                thread_id=self.emitter.thread_id,
                turn_id=self.emitter.turn_id,
                item_id=info.item_id,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments or _image_generation_arguments(event),
                image=image,
                partial_index=_as_int(event.get("partial_image_index")),
                status_detail="Generating image",
            )
        )

    def _emit_image_generation_completed(
        self,
        *,
        info: _ToolCallInfo,
        item: dict[str, Any],
    ) -> None:
        self._started_image_generation_ids.discard(info.item_id)
        if info.error is not None:
            self.emitter.callback(
                AgentImageGenerationFailed(
                    type=AGENT_EVENT_IMAGE_GENERATION_FAILED,
                    thread_id=self.emitter.thread_id,
                    turn_id=self.emitter.turn_id,
                    item_id=info.item_id,
                    call_id=info.call_id,
                    toolkit=info.toolkit,
                    tool=info.tool,
                    arguments=info.arguments or _image_generation_arguments(item),
                    error=info.error,
                    status_detail=info.error.message,
                )
            )
            return

        images = _generated_images_from_item(item)
        self.emitter.callback(
            AgentImageGenerationCompleted(
                type=AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
                thread_id=self.emitter.thread_id,
                turn_id=self.emitter.turn_id,
                item_id=info.item_id,
                call_id=info.call_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments or _image_generation_arguments(item),
                images=images,
                status_detail="Image saved" if len(images) > 0 else None,
            )
        )

    def _on_content_part_added(self, *, event: dict[str, Any]) -> None:
        part = _as_dict(event.get("part"))
        item_id = self._item_id_from_event(event=event)
        if part is None or item_id is None:
            return

        content_kind = _content_kind_from_part(part)
        phase = _message_phase_from_event(event=event)
        if content_kind == "text":
            self.emitter.emit_text_started(item_id=item_id, phase=phase)
            self.emitter.emit_text_delta(
                item_id=item_id, text=_part_text(part), phase=phase
            )
            return

        if content_kind == "reasoning":
            self.emitter.emit_reasoning_started(item_id=item_id)
            self.emitter.emit_reasoning_delta(item_id=item_id, text=_part_text(part))
            return

        if content_kind == "file":
            self.emitter.emit_file_started(item_id=item_id)
            self.emitter.emit_file_delta(
                item_id=item_id,
                url=_file_url_from_payload(payload=part, provider="openai"),
            )

    def _on_content_part_done(self, *, event: dict[str, Any]) -> None:
        part = _as_dict(event.get("part"))
        item_id = self._item_id_from_event(event=event)
        if part is None or item_id is None:
            return

        self._finish_part_from_snapshot(
            item_id=item_id,
            part=part,
            phase=_message_phase_from_event(event=event),
        )

    def _on_text_delta(self, *, event: dict[str, Any]) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return
        phase = _message_phase_from_event(event=event)
        self.emitter.emit_text_delta(
            item_id=item_id,
            text=_as_text(event.get("delta")) or "",
            phase=phase,
        )

    def _on_text_done(self, *, event: dict[str, Any], field_name: str) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return

        phase = _message_phase_from_event(event=event)
        final_text = _as_text(event.get(field_name)) or ""
        if final_text != "" and not self.emitter.has_content_data(
            kind="text", item_id=item_id
        ):
            self.emitter.emit_text_delta(item_id=item_id, text=final_text, phase=phase)
        self.emitter.emit_text_ended(item_id=item_id, phase=phase)

    def _on_reasoning_delta(self, *, event: dict[str, Any], field_name: str) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return
        self.emitter.emit_reasoning_delta(
            item_id=item_id,
            text=_as_text(event.get(field_name)) or "",
        )

    def _on_reasoning_done(
        self, *, event: dict[str, Any], field_name: str | None = None
    ) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return

        if field_name is not None:
            final_text = _as_text(event.get(field_name)) or ""
            if final_text != "" and not self.emitter.has_content_data(
                kind="reasoning", item_id=item_id
            ):
                self.emitter.emit_reasoning_delta(item_id=item_id, text=final_text)
        self.emitter.emit_reasoning_ended(item_id=item_id)

    def __call__(self, event: dict[str, Any]) -> None:
        event_type = _as_str(event.get("type"))
        if event_type is None:
            return

        if event_type in ("agent.event", "codex.event"):
            if self.custom_event_callback is not None:
                self.custom_event_callback(event)
            return

        if event_type.startswith("response."):
            self._track_response_boundary(event=event)

        if event_type == "response.output_item.added":
            self._on_output_item(event=event, completed=False)
            return

        if event_type == "response.output_item.done":
            self._on_output_item(event=event, completed=True)
            return

        if event_type in {
            "response.completed",
            "response.done",
            "response.incomplete",
        }:
            response = _as_dict(event.get("response"))
            if response is None:
                return
            outputs = response.get("output")
            if not isinstance(outputs, list):
                return
            for output_index, item in enumerate(outputs):
                if not isinstance(item, dict):
                    continue
                if item.get("type") not in {"compaction", "message"}:
                    continue
                self._on_output_item(
                    event={**event, "item": item, "output_index": output_index},
                    completed=True,
                )
            return

        if event_type in {
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
        }:
            info = self._image_generation_info_from_event(event=event)
            if info is None:
                return
            self._emit_image_generation_started(info=info, item=event)
            return

        if event_type == "response.image_generation_call.partial_image":
            info = self._image_generation_info_from_event(event=event)
            if info is None:
                return
            self._emit_image_generation_partial(info=info, event=event)
            return

        if event_type == "response.content_part.added":
            self._on_content_part_added(event=event)
            return

        if event_type == "response.content_part.done":
            self._on_content_part_done(event=event)
            return

        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            self._on_text_delta(event=event)
            return

        if event_type == "response.output_text.done":
            self._on_text_done(event=event, field_name="text")
            return

        if event_type == "response.refusal.done":
            self._on_text_done(event=event, field_name="refusal")
            return

        if event_type in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            self._on_reasoning_delta(event=event, field_name="delta")
            return

        if event_type == "response.reasoning_summary_part.added":
            part = _as_dict(event.get("part"))
            item_id = self._item_id_from_event(event=event)
            if part is None or item_id is None:
                return
            self.emitter.emit_reasoning_started(item_id=item_id)
            self.emitter.emit_reasoning_delta(item_id=item_id, text=_part_text(part))
            return

        if event_type == "response.reasoning_summary_part.done":
            self._on_reasoning_done(event=event)
            return

        if event_type == "response.reasoning_text.done":
            self._on_reasoning_done(event=event, field_name="text")
            return

        if event_type == "response.reasoning_summary_text.done":
            self._on_reasoning_done(event=event)
            return

        if event_type == "meshagent.handler.added":
            item = _as_dict(event.get("item"))
            if item is None:
                return
            tool_info = _openai_tool_call_info(
                item,
                provider_tool_namespace=self.provider_tool_namespace,
                function_tool_name_resolver=self.function_tool_name_resolver,
            )
            if tool_info is None:
                return
            self._pending_handler_tool_calls[tool_info.item_id] = tool_info
            self.emitter.emit_tool_started(info=tool_info)
            return

        if event_type == "meshagent.handler.output":
            item_id = _as_str(event.get("item_id"))
            if item_id is None:
                return
            self.emitter.emit_tool_log_delta(
                item_id=item_id,
                lines=_parse_tool_log_lines(event.get("lines")),
            )
            return

        if event_type == "meshagent.handler.done":
            result_item = _as_dict(event.get("item"))
            item_id = _as_str(event.get("item_id"))
            if item_id is None and result_item is not None:
                item_id = _as_str(result_item.get("id"))

            pending_tool_call = (
                self._pending_handler_tool_calls.pop(item_id, None)
                if item_id is not None
                else None
            )
            if pending_tool_call is None and result_item is not None:
                result_call_id = _as_str(result_item.get("call_id"))
                if result_call_id is not None:
                    for pending_item_id, pending_info in list(
                        self._pending_handler_tool_calls.items()
                    ):
                        if pending_info.call_id != result_call_id:
                            continue
                        pending_tool_call = pending_info
                        self._pending_handler_tool_calls.pop(pending_item_id, None)
                        break
            if pending_tool_call is None and item_id is None:
                if len(self._pending_handler_tool_calls) == 1:
                    pending_item_id, pending_info = next(
                        iter(self._pending_handler_tool_calls.items())
                    )
                    pending_tool_call = pending_info
                    self._pending_handler_tool_calls.pop(pending_item_id, None)
            if pending_tool_call is None and result_item is not None:
                pending_tool_call = _openai_tool_call_info(
                    result_item,
                    provider_tool_namespace=self.provider_tool_namespace,
                    function_tool_name_resolver=self.function_tool_name_resolver,
                )
            if pending_tool_call is None:
                return

            result = pending_tool_call.result
            if result_item is not None:
                result = (
                    _content_from_tool_item(result_item)
                    or _content_from_handler_result(result_item.get("result"))
                    or result
                )
            else:
                result = (
                    _content_from_handler_result(event.get("result"))
                    or pending_tool_call.result
                )

            error_text = _as_str(event.get("error"))
            self._finished_handler_tool_call_ids.add(pending_tool_call.item_id)
            self.emitter.emit_tool_ended(
                info=replace(
                    pending_tool_call,
                    result=result,
                    error=(
                        AgentError(
                            message=error_text,
                            code="tool_call_failed",
                        )
                        if error_text is not None
                        else pending_tool_call.error
                    ),
                )
            )


@dataclass(slots=True)
class _AnthropicBlockState:
    kind: Literal["file", "reasoning", "text", "tool"]
    item_id: str
    block: dict[str, Any]
    arguments_text: str = ""


@dataclass(slots=True)
class _AnthropicAgentEventPublisher:
    emitter: _AgentMessageEmitter
    function_tool_name_resolver: FunctionToolNameResolver | None = None
    custom_event_callback: Callable[[dict[str, Any]], None] | None = None
    _openai_publisher: _OpenAIAgentEventPublisher = field(init=False)
    _message_id: str | None = None
    _blocks: dict[int, _AnthropicBlockState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._openai_publisher = _OpenAIAgentEventPublisher(
            emitter=self.emitter,
            provider_tool_namespace=_ANTHROPIC_MESSAGES_TOOL_NAMESPACE,
            function_tool_name_resolver=self.function_tool_name_resolver,
            custom_event_callback=self.custom_event_callback,
        )

    def set_function_tool_name_resolver(
        self,
        resolver: FunctionToolNameResolver | None,
    ) -> None:
        self.function_tool_name_resolver = resolver
        self._openai_publisher.set_function_tool_name_resolver(resolver)

    def _synthesized_item_id(self, *, index: int) -> str:
        base = self._message_id or "anthropic"
        return f"{base}:content:{index}"

    def __call__(self, event: dict[str, Any]) -> None:
        event_type = _as_str(event.get("type"))
        if event_type is None:
            return

        if event_type in ("agent.event", "codex.event"):
            if self.custom_event_callback is not None:
                self.custom_event_callback(event)
            return

        if event_type.startswith("response."):
            self._openai_publisher(event)
            return

        if event_type.startswith("meshagent.handler."):
            self._openai_publisher(event)
            return

        normalized_type = event_type.replace(".", "_")
        payload = _as_dict(event.get("event")) or {}

        if normalized_type == "message_start":
            message = _as_dict(payload.get("message"))
            if message is None:
                return
            self._message_id = _as_str(message.get("id"))
            return

        if normalized_type == "content_block_start":
            index = _as_int(payload.get("index"))
            block = _as_dict(payload.get("content_block"))
            if index is None or block is None:
                return

            block_type = _as_str(block.get("type"))
            if block_type is None:
                return

            if block_type == "text":
                item_id = self._synthesized_item_id(index=index)
                self._blocks[index] = _AnthropicBlockState(
                    kind="text",
                    item_id=item_id,
                    block=block,
                )
                self.emitter.emit_text_started(item_id=item_id)
                self.emitter.emit_text_delta(
                    item_id=item_id,
                    text=_as_text(block.get("text")) or "",
                )
                return

            if block_type == "thinking":
                item_id = self._synthesized_item_id(index=index)
                self._blocks[index] = _AnthropicBlockState(
                    kind="reasoning",
                    item_id=item_id,
                    block=block,
                )
                self.emitter.emit_reasoning_started(item_id=item_id)
                self.emitter.emit_reasoning_delta(
                    item_id=item_id,
                    text=_as_text(block.get("thinking")) or "",
                )
                return

            tool_item_id = _as_str(block.get("id"))
            tool_info = (
                _anthropic_tool_call_info(
                    block=block,
                    item_id=tool_item_id,
                    function_tool_name_resolver=self.function_tool_name_resolver,
                )
                if tool_item_id is not None
                else None
            )
            if tool_info is not None:
                self._blocks[index] = _AnthropicBlockState(
                    kind="tool",
                    item_id=tool_info.item_id,
                    block=block,
                )
                self.emitter.emit_tool_pending(info=tool_info)
                return

            if block_type in {"document", "image"}:
                item_id = self._synthesized_item_id(index=index)
                self._blocks[index] = _AnthropicBlockState(
                    kind="file",
                    item_id=item_id,
                    block=block,
                )
                self.emitter.emit_file_started(item_id=item_id)
                self.emitter.emit_file_delta(
                    item_id=item_id,
                    url=_file_url_from_payload(payload=block, provider="anthropic"),
                )
            return

        if normalized_type == "content_block_delta":
            index = _as_int(payload.get("index"))
            delta = _as_dict(payload.get("delta"))
            if index is None or delta is None:
                return

            state = self._blocks.get(index)
            if state is None:
                return

            delta_type = _as_str(delta.get("type"))
            if state.kind == "text" and delta_type == "text_delta":
                self.emitter.emit_text_delta(
                    item_id=state.item_id,
                    text=_as_text(delta.get("text")) or "",
                )
                return

            if state.kind == "reasoning" and delta_type == "thinking_delta":
                self.emitter.emit_reasoning_delta(
                    item_id=state.item_id,
                    text=_as_text(delta.get("thinking")) or "",
                )
                return

            if state.kind == "tool" and delta_type == "input_json_delta":
                state.arguments_text += _as_text(delta.get("partial_json")) or ""
                return

            if state.kind == "file":
                self.emitter.emit_file_delta(
                    item_id=state.item_id,
                    url=_file_url_from_payload(payload=delta, provider="anthropic"),
                )
            return

        if normalized_type == "content_block_stop":
            index = _as_int(payload.get("index"))
            if index is None:
                return

            state = self._blocks.pop(index, None)
            if state is None:
                return

            if state.kind == "text":
                self.emitter.emit_text_ended(item_id=state.item_id)
                return

            if state.kind == "reasoning":
                self.emitter.emit_reasoning_ended(item_id=state.item_id)
                return

            if state.kind == "file":
                if not self.emitter.has_content_data(
                    kind="file", item_id=state.item_id
                ):
                    self.emitter.emit_file_delta(
                        item_id=state.item_id,
                        url=_file_url_from_payload(
                            payload=state.block,
                            provider="anthropic",
                        ),
                    )
                self.emitter.emit_file_ended(item_id=state.item_id)
                return

            tool_info = _anthropic_tool_call_info(
                block=state.block,
                item_id=state.item_id,
                function_tool_name_resolver=self.function_tool_name_resolver,
            )
            if tool_info is not None:
                parsed_arguments = (
                    _parse_tool_arguments(state.arguments_text)
                    if state.arguments_text != ""
                    else None
                )
                updated_arguments = parsed_arguments
                if updated_arguments is None:
                    updated_arguments = tool_info.arguments
                self.emitter.emit_tool_pending(
                    info=replace(tool_info, arguments=updated_arguments)
                )


def make_openai_agent_event_publisher(
    *,
    turn_id: str,
    thread_id: str,
    callback: AgentEventCallback,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
    custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
    provider_tool_namespace: str = _OPENAI_RESPONSES_TOOL_NAMESPACE,
) -> Callable[[dict[str, Any]], None]:
    emitter = _AgentMessageEmitter(
        turn_id=turn_id,
        thread_id=thread_id,
        callback=callback,
    )
    publisher = _OpenAIAgentEventPublisher(
        emitter=emitter,
        provider_tool_namespace=provider_tool_namespace,
        function_tool_name_resolver=function_tool_name_resolver,
        custom_event_callback=custom_event_callback,
    )
    return publisher


def make_anthropic_agent_event_publisher(
    *,
    turn_id: str,
    thread_id: str,
    callback: AgentEventCallback,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
    custom_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    emitter = _AgentMessageEmitter(
        turn_id=turn_id,
        thread_id=thread_id,
        callback=callback,
    )
    publisher = _AnthropicAgentEventPublisher(
        emitter=emitter,
        function_tool_name_resolver=function_tool_name_resolver,
        custom_event_callback=custom_event_callback,
    )
    return publisher
