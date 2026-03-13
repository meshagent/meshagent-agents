from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from meshagent.api.messaging import JsonContent, TextContent

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
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AgentError,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallEnded,
    AgentToolCallStarted,
)

AgentEventCallback = Callable[[AgentMessage], None]
FunctionToolNameResolver = Callable[[str], tuple[str, str] | None]
_ContentKind = Literal["file", "reasoning", "text"]
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


def _content_from_tool_item(item: dict[str, Any]):
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


@dataclass(frozen=True, slots=True)
class _ToolCallInfo:
    item_id: str
    toolkit: str
    tool: str
    item_type: str | None = None
    call_id: str | None = None
    arguments: dict[str, Any] | None = None
    error: AgentError | None = None
    result: TextContent | JsonContent | None = None


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
            toolkit=toolkit_name,
            tool=tool_name,
            arguments=_parse_tool_arguments(block.get("input")),
        )

    if block_type == "mcp_tool_use":
        return _ToolCallInfo(
            item_id=item_id,
            toolkit=(
                _as_str(block.get("server_name"))
                or _as_str(block.get("server_label"))
                or "mcp"
            ),
            tool=_as_str(block.get("name")) or "tool",
            arguments=_parse_tool_arguments(
                block.get("input")
                if block.get("input") is not None
                else block.get("arguments")
            ),
        )

    if block_type.endswith("_tool_use"):
        return _ToolCallInfo(
            item_id=item_id,
            toolkit="anthropic",
            tool=block_type.removesuffix("_tool_use"),
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

    def emit_text_delta(self, *, item_id: str, text: str) -> None:
        if text == "":
            return
        self._ensure_content_started(kind="text", item_id=item_id)
        self._mark_content_has_data(kind="text", item_id=item_id)
        self.callback(
            AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
                text=text,
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

    def emit_text_started(self, *, item_id: str) -> None:
        self._ensure_content_started(kind="text", item_id=item_id)

    def emit_reasoning_started(self, *, item_id: str) -> None:
        self._ensure_content_started(kind="reasoning", item_id=item_id)

    def emit_file_started(self, *, item_id: str) -> None:
        self._ensure_content_started(kind="file", item_id=item_id)

    def emit_text_ended(self, *, item_id: str) -> None:
        key = self._content_key(kind="text", item_id=item_id)
        if key in self._ended_content:
            return
        self._ensure_content_started(kind="text", item_id=item_id)
        self._ended_content.add(key)
        self.callback(
            AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=item_id,
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

    def emit_tool_started(self, *, info: _ToolCallInfo) -> None:
        existing = self._started_tool_calls.get(info.item_id)
        if (
            existing is not None
            and existing.toolkit == info.toolkit
            and existing.tool == info.tool
            and existing.arguments == info.arguments
        ):
            return
        self._started_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                toolkit=info.toolkit,
                tool=info.tool,
                arguments=info.arguments,
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
        self._ended_tool_calls[info.item_id] = info
        self.callback(
            AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                item_id=info.item_id,
                result=info.result,
                error=info.error,
            )
        )


@dataclass(slots=True)
class _OpenAIAgentEventPublisher:
    emitter: _AgentMessageEmitter
    function_tool_name_resolver: FunctionToolNameResolver | None = None
    _output_item_ids: dict[int, str] = field(default_factory=dict)
    _pending_handler_tool_calls: dict[str, _ToolCallInfo] = field(default_factory=dict)

    def set_function_tool_name_resolver(
        self,
        resolver: FunctionToolNameResolver | None,
    ) -> None:
        self.function_tool_name_resolver = resolver

    def _item_id_from_event(self, *, event: dict[str, Any]) -> str | None:
        item_id = _as_str(event.get("item_id"))
        if item_id is not None:
            return item_id

        item = _as_dict(event.get("item"))
        if item is not None:
            item_id = _as_str(item.get("id"))
            if item_id is not None:
                return item_id

        output_index = _as_int(event.get("output_index"))
        if output_index is None:
            return None

        existing = self._output_item_ids.get(output_index)
        if existing is not None:
            return existing

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
        self._output_item_ids[output_index] = item_id

    def _finish_part_from_snapshot(self, *, item_id: str, part: dict[str, Any]) -> None:
        content_kind = _content_kind_from_part(part)
        if content_kind == "text":
            text = _part_text(part)
            if text != "" and not self.emitter.has_content_data(
                kind="text", item_id=item_id
            ):
                self.emitter.emit_text_delta(item_id=item_id, text=text)
            self.emitter.emit_text_ended(item_id=item_id)
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
            function_tool_name_resolver=self.function_tool_name_resolver,
        )
        if tool_info is not None:
            if completed and tool_info.item_type in _HANDLER_RESULT_TOOL_TYPES:
                self._pending_handler_tool_calls[tool_info.item_id] = tool_info
                return

            if completed:
                self.emitter.emit_tool_ended(info=tool_info)
            else:
                self.emitter.emit_tool_started(info=tool_info)
            return

        item_type = _as_str(item.get("type"))
        item_id = _as_str(item.get("id"))
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

        if item_type == "message" and completed and item_id is not None:
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        self._finish_part_from_snapshot(item_id=item_id, part=part)

    def _on_content_part_added(self, *, event: dict[str, Any]) -> None:
        part = _as_dict(event.get("part"))
        item_id = self._item_id_from_event(event=event)
        if part is None or item_id is None:
            return

        content_kind = _content_kind_from_part(part)
        if content_kind == "text":
            self.emitter.emit_text_started(item_id=item_id)
            self.emitter.emit_text_delta(item_id=item_id, text=_part_text(part))
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

        self._finish_part_from_snapshot(item_id=item_id, part=part)

    def _on_text_delta(self, *, event: dict[str, Any]) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return
        self.emitter.emit_text_delta(
            item_id=item_id, text=_as_text(event.get("delta")) or ""
        )

    def _on_text_done(self, *, event: dict[str, Any], field_name: str) -> None:
        item_id = self._item_id_from_event(event=event)
        if item_id is None:
            return

        final_text = _as_text(event.get(field_name)) or ""
        if final_text != "" and not self.emitter.has_content_data(
            kind="text", item_id=item_id
        ):
            self.emitter.emit_text_delta(item_id=item_id, text=final_text)
        self.emitter.emit_text_ended(item_id=item_id)

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

        if event_type == "response.output_item.added":
            self._on_output_item(event=event, completed=False)
            return

        if event_type == "response.output_item.done":
            self._on_output_item(event=event, completed=True)
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
                function_tool_name_resolver=self.function_tool_name_resolver,
            )
            if tool_info is None:
                return
            self._pending_handler_tool_calls[tool_info.item_id] = tool_info
            self.emitter.emit_tool_started(info=tool_info)
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
    _openai_publisher: _OpenAIAgentEventPublisher = field(init=False)
    _message_id: str | None = None
    _blocks: dict[int, _AnthropicBlockState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._openai_publisher = _OpenAIAgentEventPublisher(
            emitter=self.emitter,
            function_tool_name_resolver=self.function_tool_name_resolver,
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
                self.emitter.emit_tool_started(info=tool_info)
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
                if parsed_arguments is not None and tool_info.arguments is None:
                    self.emitter.emit_tool_started(
                        info=replace(tool_info, arguments=parsed_arguments)
                    )


def make_openai_agent_event_publisher(
    *,
    turn_id: str,
    thread_id: str,
    callback: AgentEventCallback,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
) -> Callable[[dict[str, Any]], None]:
    emitter = _AgentMessageEmitter(
        turn_id=turn_id,
        thread_id=thread_id,
        callback=callback,
    )
    publisher = _OpenAIAgentEventPublisher(
        emitter=emitter,
        function_tool_name_resolver=function_tool_name_resolver,
    )
    return publisher


def make_anthropic_agent_event_publisher(
    *,
    turn_id: str,
    thread_id: str,
    callback: AgentEventCallback,
    function_tool_name_resolver: FunctionToolNameResolver | None = None,
) -> Callable[[dict[str, Any]], None]:
    emitter = _AgentMessageEmitter(
        turn_id=turn_id,
        thread_id=thread_id,
        callback=callback,
    )
    publisher = _AnthropicAgentEventPublisher(
        emitter=emitter,
        function_tool_name_resolver=function_tool_name_resolver,
    )
    return publisher
