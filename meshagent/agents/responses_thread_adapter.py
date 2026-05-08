import asyncio
import base64
import json
import logging
import re
import shlex
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from meshagent.api import Element, RoomException

from .images_dataset import ImagesDataset
from .image_captioner import ImageCaptioner
from .thread_adapter import ThreadAdapter, tracer

logger = logging.getLogger("thread_adapter")

_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_RESPONSE_NOISE_TYPES = {
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.custom_tool_call_input.delta",
    "response.custom_tool_call_input.done",
    "response.image_generation_call.partial_image",
}
_RESPONSE_TURN_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.completed",
    "response.failed",
    "response.cancelled",
    "response.canceled",
    "response.queued",
}
_SUPPORTED_EVENT_KINDS = {
    "exec",
    "tool",
    "web",
    "search",
    "diff",
    "image",
    "approval",
    "collab",
    "plan",
}
_IMAGE_DB_SAVE_TIMEOUT_SECONDS = 20.0
_IMAGE_STAGE_PARTIAL = "partial"
_IMAGE_STAGE_FINAL = "final"
_PARTIAL_IMAGE_SOURCE = "response.image_generation_call.partial_image"
_IMAGE_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _to_text(item).strip()
            if text != "":
                parts.append(text)
        return " ".join(parts)
    if isinstance(value, dict):
        for key in ("text", "value", "name", "description"):
            text = value.get(key)
            if isinstance(text, str) and text.strip() != "":
                return text
    return str(value)


def _normalize_positive_dimension(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        as_int = int(value)
        return as_int if as_int > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _parse_image_dimensions_from_size(
    value: Any,
) -> tuple[Optional[int], Optional[int]]:
    if isinstance(value, str):
        match = _IMAGE_SIZE_RE.match(value)
        if match is None:
            return (None, None)
        width = int(match.group(1))
        height = int(match.group(2))
        return (
            width if width > 0 else None,
            height if height > 0 else None,
        )

    if isinstance(value, dict):
        return (
            _normalize_positive_dimension(value.get("width")),
            _normalize_positive_dimension(value.get("height")),
        )

    if isinstance(value, list) and len(value) >= 2:
        return (
            _normalize_positive_dimension(value[0]),
            _normalize_positive_dimension(value[1]),
        )

    return (None, None)


def _extract_image_dimensions(
    *,
    item: Optional[dict] = None,
    event: Optional[dict] = None,
) -> tuple[Optional[int], Optional[int]]:
    width = None
    height = None

    if isinstance(item, dict):
        width = _normalize_positive_dimension(item.get("width"))
        height = _normalize_positive_dimension(item.get("height"))

    if isinstance(event, dict):
        if width is None:
            width = _normalize_positive_dimension(event.get("width"))
        if height is None:
            height = _normalize_positive_dimension(event.get("height"))

    for source in (item, event):
        if not isinstance(source, dict):
            continue
        if width is None or height is None:
            parsed_width, parsed_height = _parse_image_dimensions_from_size(
                source.get("size")
            )
            if width is None and parsed_width is not None:
                width = parsed_width
            if height is None and parsed_height is not None:
                height = parsed_height

        if width is None or height is None:
            parsed_width, parsed_height = _parse_image_dimensions_from_size(
                source.get("resolution")
            )
            if width is None and parsed_width is not None:
                width = parsed_width
            if height is None and parsed_height is not None:
                height = parsed_height

    return (width, height)


def _image_status_from_state(*, state: str) -> str:
    normalized = state.strip().lower()
    if normalized in ("queued", "pending", "searching"):
        return "queued"
    if normalized in ("running", "in_progress"):
        return "generating"
    if normalized in ("completed",):
        return "completed"
    if normalized in ("failed",):
        return "failed"
    if normalized in ("cancelled",):
        return "cancelled"
    return "info"


def _first_nested_text(*, value: Any, keys: tuple[str, ...]) -> str:
    key_set = {key.lower() for key in keys}

    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in key_set:
                text = _to_text(nested).strip()
                if text != "":
                    return text

        for nested in value.values():
            text = _first_nested_text(value=nested, keys=keys)
            if text != "":
                return text

    elif isinstance(value, list):
        for nested in value:
            text = _first_nested_text(value=nested, keys=keys)
            if text != "":
                return text

    return ""


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_status_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized == "":
        return None

    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized == "inprogress":
        normalized = "in_progress"

    if normalized in ("failed", "error", "errored", "rejected"):
        return "failed"
    if normalized in ("cancelled", "canceled", "interrupted", "aborted", "stopped"):
        return "cancelled"
    if normalized in ("queued", "pending", "waiting"):
        return "queued"
    if normalized in ("running", "started", "starting", "in_progress"):
        return "in_progress"
    if normalized in (
        "completed",
        "complete",
        "done",
        "succeeded",
        "success",
        "finished",
    ):
        return "completed"

    if "fail" in normalized or "error" in normalized:
        return "failed"
    if "cancel" in normalized or "interrupt" in normalized or "abort" in normalized:
        return "cancelled"
    if "queue" in normalized or "pending" in normalized or "wait" in normalized:
        return "queued"
    if "complete" in normalized or "success" in normalized or "done" in normalized:
        return "completed"
    if "progress" in normalized or "running" in normalized or "start" in normalized:
        return "in_progress"

    return None


def _kind_from_item_type(*, item_type: str) -> Optional[str]:
    normalized = _normalize_name(item_type)
    if normalized == "":
        return None
    if normalized in (
        "mcpcall",
        "mcplisttools",
        "functioncall",
        "functioncalloutput",
        "customtoolcall",
        "hostedtoolcall",
        "toolcall",
        "computercall",
    ):
        return "tool"
    if normalized in ("websearchcall",):
        return "web"
    if normalized in ("filesearchcall",):
        return "search"
    if normalized in ("applypatchcall",):
        return "diff"
    if normalized in ("codeinterpretercall", "shellcall"):
        return "exec"
    if normalized in ("imagegenerationcall",):
        return "image"
    if normalized in ("reasoning",):
        return "reasoning"
    if normalized in ("message", "agentmessage"):
        return "message"
    return None


def _normalize_state_from_response_type(*, event_type: str) -> str:
    lower = event_type.lower()
    if lower.endswith(".failed"):
        return "failed"
    if lower.endswith(".cancelled") or lower.endswith(".canceled"):
        return "cancelled"
    if lower.endswith(".completed") or lower == "response.completed":
        return "completed"
    if (
        lower.endswith(".queued")
        or lower.endswith(".pending")
        or lower == "response.queued"
    ):
        return "queued"
    if (
        lower.endswith(".in_progress")
        or lower.endswith(".searching")
        or lower.endswith(".started")
        or lower.endswith(".generating")
        or lower.endswith(".added")
        or lower == "response.in_progress"
    ):
        return "in_progress"
    if lower.endswith(".done"):
        return "completed"
    return "info"


def _normalize_kind_from_response_type(*, event_type: str) -> str:
    lower = event_type.lower()
    if ".web_search_call." in lower:
        return "web"
    if ".file_search_call." in lower:
        return "search"
    if (
        ".mcp_call." in lower
        or ".mcp_list_tools." in lower
        or ".function_call." in lower
        or ".function_call_arguments." in lower
        or ".computer_call." in lower
    ):
        return "tool"
    if ".apply_patch_call." in lower:
        return "diff"
    if ".code_interpreter_call." in lower:
        return "exec"
    if ".image_generation_call." in lower:
        return "image"
    if lower.startswith("response.reasoning"):
        return "reasoning"
    if lower.startswith("response.output_item"):
        return "item"
    if lower.startswith("response."):
        return "turn"
    return "event"


def _response_identity(*, event: dict) -> tuple[Optional[str], Optional[str]]:
    item_id = event.get("item_id")
    if not isinstance(item_id, str) or item_id.strip() == "":
        item = event.get("item")
        if isinstance(item, dict):
            candidate = item.get("id")
            if isinstance(candidate, str) and candidate.strip() != "":
                item_id = candidate
            else:
                item_id = None
        else:
            item_id = None
    else:
        item_id = item_id.strip()

    response_id = None
    response = event.get("response")
    if isinstance(response, dict):
        candidate = response.get("id")
        if isinstance(candidate, str) and candidate.strip() != "":
            response_id = candidate.strip()

    return item_id, response_id


def _response_base_name(*, event_type: str) -> str:
    suffixes = (
        ".in_progress",
        ".searching",
        ".generating",
        ".completed",
        ".failed",
        ".cancelled",
        ".canceled",
        ".queued",
        ".pending",
        ".started",
    )
    for suffix in suffixes:
        if event_type.endswith(suffix):
            return event_type[: -len(suffix)]
    return event_type


def _computer_action_payload(*, event: dict) -> Optional[dict]:
    item = event.get("item")
    if isinstance(item, dict):
        action = item.get("action")
        if isinstance(action, dict):
            return action

    action = event.get("action")
    if isinstance(action, dict):
        return action

    return None


def _computer_action_type(*, event: dict) -> str:
    action = _computer_action_payload(event=event)
    if not isinstance(action, dict):
        return ""

    action_type = action.get("type")
    if not isinstance(action_type, str):
        return ""

    normalized = action_type.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized


def _is_computer_call_event(*, event: dict) -> bool:
    event_name = event.get("name")
    if event_name == "computer.startup":
        return True

    item = event.get("item")
    if isinstance(item, dict):
        item_type = item.get("type")
        if isinstance(item_type, str) and _normalize_name(item_type) == "computercall":
            return True

    event_type = event.get("type")
    if isinstance(event_type, str) and ".computer_call." in event_type.lower():
        return True

    return False


def _computer_headline_for_state(*, event: dict, state: str) -> str:
    event_name = event.get("name")
    if event_name == "computer.startup":
        if state in _ACTIVE_STATES:
            return "Starting computer..."
        if state == "completed":
            return "Computer ready"
        if state == "failed":
            return "Failed to start computer"
        if state == "cancelled":
            return "Computer startup cancelled"
        return "Computer startup update"

    action_type = _computer_action_type(event=event)
    active_headlines = {
        "click": "Clicking on page",
        "double_click": "Double-clicking on page",
        "scroll": "Scrolling page",
        "type": "Typing text",
        "keypress": "Pressing keys",
        "move": "Moving cursor",
        "drag": "Dragging on page",
        "wait": "Waiting",
        "goto": "Opening page",
        "back": "Navigating back",
        "forward": "Navigating forward",
        "screenshot": "Capturing screenshot",
    }
    completed_headlines = {
        "click": "Clicked on page",
        "double_click": "Double-clicked on page",
        "scroll": "Scrolled page",
        "type": "Typed text",
        "keypress": "Pressed keys",
        "move": "Moved cursor",
        "drag": "Dragged on page",
        "wait": "Wait complete",
        "goto": "Opened page",
        "back": "Went back",
        "forward": "Went forward",
        "screenshot": "Captured screenshot",
    }

    if state in _ACTIVE_STATES:
        return active_headlines.get(action_type, "Using computer")
    if state == "completed":
        return completed_headlines.get(action_type, "Computer action complete")
    if state == "failed":
        return "Computer action failed"
    if state == "cancelled":
        return "Computer action cancelled"
    return "Computer action"


def _computer_scroll_direction_detail(*, action_payload: dict) -> Optional[str]:
    raw_scroll_x = action_payload.get("scroll_x")
    raw_scroll_y = action_payload.get("scroll_y")

    scroll_x = int(raw_scroll_x) if isinstance(raw_scroll_x, (int, float)) else 0
    scroll_y = int(raw_scroll_y) if isinstance(raw_scroll_y, (int, float)) else 0
    if scroll_x == 0 and scroll_y == 0:
        return None

    directions: list[str] = []
    if scroll_y != 0:
        directions.append("down" if scroll_y > 0 else "up")
    if scroll_x != 0:
        directions.append("right" if scroll_x > 0 else "left")

    if len(directions) == 1:
        return f"Direction: {directions[0]}"
    return f"Direction: {', '.join(directions)}"


def _headline_for_response_event(*, event: dict, kind: str, state: str) -> str:
    if kind == "turn":
        if state in _ACTIVE_STATES:
            return "Thinking"
        if state == "completed":
            return "Response Ready"
        if state == "failed":
            return "Response Failed"
        if state == "cancelled":
            return "Response Cancelled"
        return "Response Update"

    if kind == "web":
        if state in _ACTIVE_STATES:
            return "Searching Web"
        if state == "completed":
            return "Searched Web"
        if state == "failed":
            return "Web Search Failed"
        if state == "cancelled":
            return "Web Search Cancelled"
        return "Web Search"

    if kind == "search":
        if state in _ACTIVE_STATES:
            return "Searching Files"
        if state == "completed":
            return "Searched Files"
        if state == "failed":
            return "File Search Failed"
        if state == "cancelled":
            return "File Search Cancelled"
        return "File Search"

    if kind == "tool":
        if _is_computer_call_event(event=event):
            return _computer_headline_for_state(event=event, state=state)

        if state in _ACTIVE_STATES:
            return "Calling Tool"
        if state == "completed":
            return "Called Tool"
        if state == "failed":
            return "Tool Failed"
        if state == "cancelled":
            return "Tool Cancelled"
        return "Tool"

    if kind == "diff":
        if state in _ACTIVE_STATES:
            return "Applying Patch"
        if state == "completed":
            return "Applied Patch"
        if state == "failed":
            return "Patch Failed"
        if state == "cancelled":
            return "Patch Cancelled"
        return "Patch"

    if kind == "exec":
        if state in _ACTIVE_STATES:
            return "Running Command"
        if state == "completed":
            return "Ran Command"
        if state == "failed":
            return "Command Failed"
        if state == "cancelled":
            return "Command Cancelled"
        return "Command"

    if kind == "image":
        if state in _ACTIVE_STATES:
            return "Generating Image"
        if state == "completed":
            return "Generated Image"
        if state == "failed":
            return "Image Generation Failed"
        return "Image Generation"

    return "Event Update"


def _details_for_response_event(*, event: dict, kind: str) -> list[str]:
    details: list[str] = []
    seen: set[str] = set()

    def append_detail(text: str) -> None:
        normalized = " ".join(text.strip().lower().split())
        if normalized == "" or normalized in seen:
            return
        seen.add(normalized)
        details.append(text.strip())

    payload = event
    item = event.get("item")
    if isinstance(item, dict):
        payload = item

    action_payload = _computer_action_payload(event=event)
    action_type = _computer_action_type(event=event)

    if kind == "exec":
        command = _first_nested_text(
            value=payload,
            keys=("command", "cmd", "shell_command", "raw_command"),
        )
        if command != "":
            append_detail(command)

    if kind in ("web", "search"):
        query = _first_nested_text(value=payload, keys=("query", "pattern"))
        if query != "":
            append_detail(query)

    if kind == "tool":
        if _is_computer_call_event(event=event):
            if isinstance(action_payload, dict):
                if action_type == "scroll":
                    direction = _computer_scroll_direction_detail(
                        action_payload=action_payload
                    )
                    if isinstance(direction, str):
                        append_detail(direction)

                url = action_payload.get("url")
                if isinstance(url, str) and url.strip() != "":
                    append_detail(f"Opening: {url.strip()}")
        else:
            tool_name = _first_nested_text(
                value=payload,
                keys=("tool_name", "name", "server_label", "server", "tool"),
            )
            if tool_name != "":
                append_detail(f"Tool: {tool_name}")

    if kind == "diff":
        path = _first_nested_text(value=payload, keys=("path", "file", "filename"))
        if path != "":
            append_detail(path)

    return details


def _headline_with_primary_detail(
    *,
    headline: str,
    kind: str,
    details: list[str],
) -> str:
    if kind not in ("web", "search"):
        return headline

    if len(details) == 0:
        return headline

    detail = details[0].strip()
    if detail == "":
        return headline

    if detail.lower() in headline.lower():
        return headline

    max_chars = 120
    display = detail
    if len(display) > max_chars:
        display = display[:max_chars].rstrip() + "..."

    return f"{headline}: {display}"


def _item_id_for_message_event(*, event: dict) -> Optional[str]:
    item_id, _ = _response_identity(event=event)
    return item_id


def _content_index_for_message_event(*, event: dict) -> int:
    content_index = event.get("content_index")
    if isinstance(content_index, int) and content_index >= 0:
        return content_index
    return 0


def _message_text_from_content_part(*, part: Any) -> str:
    if not isinstance(part, dict):
        return ""

    part_type = part.get("type")
    if part_type == "output_text":
        text = part.get("text")
        if isinstance(text, str):
            return text
    elif part_type == "refusal":
        refusal = part.get("refusal")
        if isinstance(refusal, str):
            return refusal

    return ""


def _message_text_from_output_item(*, item: Any) -> str:
    if not isinstance(item, dict):
        return ""

    item_type = item.get("type")
    if item_type != "message":
        return ""

    content = item.get("content")
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        text = _message_text_from_content_part(part=part)
        if text != "":
            parts.append(text)

    return "".join(parts)


def _mime_type_from_output_format(*, output_format: Optional[str]) -> str:
    if not isinstance(output_format, str):
        return "image/png"

    normalized = output_format.strip().lower().lstrip(".")
    if normalized == "":
        return "image/png"

    alias = {
        "jpg": "jpeg",
    }
    normalized = alias.get(normalized, normalized)
    return f"image/{normalized}"


def _image_bytes_from_output_item(*, item: dict) -> Optional[bytes]:
    if not isinstance(item, dict):
        return None

    for key in ("result", "image_base64", "image_b64", "b64_json", "data"):
        value = item.get(key)
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, str) and value.strip() != "":
            try:
                return base64.b64decode(value)
            except Exception:
                logger.warning("unable to decode image payload from key '%s'", key)
                return None
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str) and entry.strip() != "":
                    try:
                        return base64.b64decode(entry)
                    except Exception:
                        logger.warning(
                            "unable to decode image payload from key '%s' list entry",
                            key,
                        )
                        return None

    return None


def response_event_to_agent_event(event: dict) -> Optional[dict]:
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type.startswith("response."):
        return None

    if event_type in _RESPONSE_NOISE_TYPES:
        return None
    if event_type in _RESPONSE_TURN_EVENTS:
        return None

    item = event.get("item")
    if not isinstance(item, dict):
        item = {}

    item_type = item.get("type")
    if not isinstance(item_type, str):
        item_type = ""

    state = _normalize_state_from_response_type(event_type=event_type)
    if state == "info":
        state = _normalize_status_value(item.get("status")) or "info"

    lower_event_type = event_type.lower()
    if state == "info" and lower_event_type == "response.output_item.added":
        state = "in_progress"
    elif state == "info" and lower_event_type == "response.output_item.done":
        state = "completed"

    if state == "info":
        return None

    kind = _kind_from_item_type(
        item_type=item_type
    ) or _normalize_kind_from_response_type(event_type=event_type)
    if kind in ("turn", "message", "reasoning", "item", "event"):
        return None
    if kind not in _SUPPORTED_EVENT_KINDS:
        return None

    item_id, response_id = _response_identity(event=event)
    base_name = _response_base_name(event_type=event_type)
    correlation_key = (
        f"item:{item_id}"
        if item_id is not None
        else (
            f"response:{base_name}:{response_id}"
            if response_id is not None
            else f"response:{base_name}"
        )
    )

    details = _details_for_response_event(event=event, kind=kind)
    headline = _headline_for_response_event(event=event, kind=kind, state=state)
    headline = _headline_with_primary_detail(
        headline=headline,
        kind=kind,
        details=details,
    )
    data = json.dumps(event, ensure_ascii=False, default=str)
    if len(data) > 8000:
        data = data[:8000] + "..."

    return {
        "type": "agent.event",
        "source": "openai",
        "name": event_type,
        "kind": kind,
        "state": state,
        "method": event_type,
        "correlation_key": correlation_key,
        "item_id": item_id,
        "item_type": None,
        "headline": headline,
        "details": details,
        "summary": headline,
        "data": data,
    }


class ResponsesThreadAdapter(ThreadAdapter):
    def __init__(
        self,
        *,
        image_captioner: Optional[ImageCaptioner] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._active_events_by_key: dict[str, Element] = {}
        self._images_db = ImagesDataset(room=self._room)
        self._image_captioner = image_captioner
        self._saved_image_ids_by_item_id: dict[str, str] = {}
        self._saved_image_stage_by_item_id: dict[str, str] = {}

    async def stop(self) -> None:
        await super().stop()
        self._active_events_by_key.clear()
        self._saved_image_ids_by_item_id.clear()
        self._saved_image_stage_by_item_id.clear()

    async def handle_custom_event(
        self,
        *,
        event: dict,
    ) -> None:
        await self._handle_custom_event_for_messages(
            messages=self._messages_element(),
            event=event,
        )

    async def _handle_custom_event_for_messages(
        self,
        *,
        messages: Element,
        event: dict,
    ) -> None:
        normalized_event = None

        event_type = event.get("type")
        if event_type in ("agent.event", "codex.event"):
            normalized_event = event
        else:
            normalized_event = response_event_to_agent_event(event)

        if not isinstance(normalized_event, dict):
            return

        source = normalized_event.get("source")
        if not isinstance(source, str) or source.strip() == "":
            source = "agent"
        source = source.strip()

        name = normalized_event.get("name")
        if not isinstance(name, str) or name.strip() == "":
            name = "agent.event"
        name = name.strip()

        kind = normalized_event.get("kind")
        if not isinstance(kind, str) or kind.strip() == "":
            return
        kind = kind.strip().lower()
        if kind not in _SUPPORTED_EVENT_KINDS:
            return

        state = normalized_event.get("state")
        if not isinstance(state, str) or state.strip() == "":
            state = "info"
        state = state.strip().lower()

        method = normalized_event.get("method")
        if not isinstance(method, str) or method.strip() == "":
            method = name
        method = method.strip()

        summary = normalized_event.get("summary")
        if not isinstance(summary, str) or summary.strip() == "":
            summary = method
        summary = summary.strip()

        headline = normalized_event.get("headline")
        if not isinstance(headline, str):
            headline = ""
        headline = headline.strip()

        item_id = normalized_event.get("item_id")
        if not isinstance(item_id, str):
            item_id = ""

        item_type = normalized_event.get("item_type")
        if not isinstance(item_type, str):
            item_type = ""
        event_path = normalized_event.get("path")
        if not isinstance(event_path, str):
            event_path = ""
        preview = normalized_event.get("preview")
        if not isinstance(preview, str):
            preview = ""

        details_value = normalized_event.get("details")
        if isinstance(details_value, list):
            lines = [line.strip() for line in details_value if isinstance(line, str)]
            details = "\n".join(line for line in lines if line != "")
        elif isinstance(details_value, str):
            details = details_value.strip()
        else:
            details = ""

        data = normalized_event.get("data")
        if not isinstance(data, str):
            data = json.dumps(normalized_event, ensure_ascii=False, default=str)
        persisted_data = data if kind == "diff" and data != "" else ""

        correlation_key = normalized_event.get("correlation_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = normalized_event.get("event_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = None
        else:
            correlation_key = correlation_key.strip()

        in_progress = state in _ACTIVE_STATES
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        event_element: Element | None = None
        if correlation_key is not None:
            event_element = self._active_events_by_key.get(correlation_key)

        if event_element is None:
            attributes = {
                "id": str(uuid.uuid4()),
                "source": source,
                "name": name,
                "kind": kind,
                "state": state,
                "method": method,
                "item_id": item_id,
                "item_type": item_type,
                "path": event_path,
                "preview": preview,
                "summary": summary,
                "headline": headline,
                "details": details,
                "created_at": now,
                "updated_at": now,
            }
            if persisted_data != "":
                attributes["data"] = persisted_data
            event_element = messages.append_child(
                tag_name="event",
                attributes=attributes,
            )
        else:
            event_element.set_attribute("source", source)
            event_element.set_attribute("name", name)
            event_element.set_attribute("kind", kind)
            event_element.set_attribute("state", state)
            event_element.set_attribute("method", method)
            event_element.set_attribute("item_id", item_id)
            event_element.set_attribute("item_type", item_type)
            event_element.set_attribute("path", event_path)
            if preview != "" or event_element.get_attribute("preview") in (None, ""):
                event_element.set_attribute("preview", preview)
            event_element.set_attribute("summary", summary)
            event_element.set_attribute("headline", headline)
            if details != "" or event_element.get_attribute("details") in (None, ""):
                event_element.set_attribute("details", details)
            if persisted_data != "":
                event_element.set_attribute("data", persisted_data)
            event_element.set_attribute("updated_at", now)

        if correlation_key is not None:
            if in_progress:
                self._active_events_by_key[correlation_key] = event_element
            elif state in _TERMINAL_STATES:
                self._active_events_by_key.pop(correlation_key, None)

    async def _process_llm_events(self) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        doc_messages = None
        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop
                break

        if doc_messages is None:
            raise RoomException("messages element is missing from thread document")

        updates = asyncio.Queue()
        update_thread_stop = object()
        message_elements_by_key: dict[str, Element] = {}
        message_parts_by_key: dict[str, dict[int, str]] = {}
        latest_message_key: Optional[str] = None

        local_participant_name = self._room.local_participant.get_attribute("name")
        if not isinstance(local_participant_name, str):
            local_participant_name = ""

        # throttle updates so we don't send too many syncs over the wire at once
        async def update_thread() -> None:
            while True:
                entry = await updates.get()
                if entry is update_thread_stop:
                    break

                element, partial_text = entry
                changes: dict[Element, str] = {element: partial_text}
                should_stop = False
                while True:
                    try:
                        pending = updates.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if pending is update_thread_stop:
                        should_stop = True
                        break

                    pending_element, pending_text = pending
                    changes[pending_element] = pending_text

                for changed_element, changed_text in changes.items():
                    changed_element["text"] = changed_text

                if should_stop:
                    break

        def resolve_message_key(*, event: dict) -> Optional[str]:
            item_id = _item_id_for_message_event(event=event)
            if item_id is not None:
                return f"item:{item_id}"
            return None

        def ensure_message_element(*, key: str, item_id: Optional[str]) -> Element:
            existing = message_elements_by_key.get(key)
            if existing is not None:
                return existing

            attributes: dict[str, str] = {
                "text": "",
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": local_participant_name,
            }
            if item_id is not None:
                attributes["id"] = item_id

            element = doc_messages.append_child(
                tag_name="message", attributes=attributes
            )
            message_elements_by_key[key] = element
            return element

        def queue_text_update(*, key: str, text: str) -> None:
            element = message_elements_by_key.get(key)
            if element is None:
                return
            updates.put_nowait((element, text))

        def set_part_text(*, key: str, content_index: int, text: str) -> None:
            by_index = message_parts_by_key.setdefault(key, {})
            by_index[content_index] = text
            merged = "".join(by_index[i] for i in sorted(by_index))
            queue_text_update(key=key, text=merged)

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await self._llm_messages.get()

                event_type = evt.get("type")
                if not isinstance(event_type, str):
                    await self._handle_custom_event_for_messages(
                        messages=doc_messages,
                        event=evt,
                    )
                    continue

                if event_type == "response.content_part.added":
                    part = evt.get("part")
                    part_type = part.get("type") if isinstance(part, dict) else None
                    if part_type == "reasoning_text":
                        continue

                    key = resolve_message_key(event=evt)
                    if key is None:
                        key = f"message:{uuid.uuid4()}"

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    part_text = _message_text_from_content_part(part=part)
                    if part_text != "":
                        set_part_text(
                            key=key,
                            content_index=_content_index_for_message_event(event=evt),
                            text=part_text,
                        )

                elif event_type == "response.output_text.delta":
                    key = resolve_message_key(event=evt) or latest_message_key
                    if key is None:
                        continue

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    content_index = _content_index_for_message_event(event=evt)
                    current = message_parts_by_key.setdefault(key, {}).get(
                        content_index, ""
                    )
                    delta = evt.get("delta")
                    if not isinstance(delta, str):
                        delta = ""
                    set_part_text(
                        key=key, content_index=content_index, text=current + delta
                    )

                elif event_type == "response.output_text.done":
                    key = resolve_message_key(event=evt) or latest_message_key
                    if key is None:
                        continue

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    text = evt.get("text")
                    if not isinstance(text, str):
                        text = ""
                    set_part_text(
                        key=key,
                        content_index=_content_index_for_message_event(event=evt),
                        text=text,
                    )
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attribute(
                            "from_participant_name", local_participant_name
                        )
                        span.set_attribute("role", "assistant")
                        span.set_attribute("text", text)

                elif event_type == "response.refusal.delta":
                    key = resolve_message_key(event=evt) or latest_message_key
                    if key is None:
                        continue

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    content_index = _content_index_for_message_event(event=evt)
                    current = message_parts_by_key.setdefault(key, {}).get(
                        content_index, ""
                    )
                    delta = evt.get("delta")
                    if not isinstance(delta, str):
                        delta = ""
                    set_part_text(
                        key=key, content_index=content_index, text=current + delta
                    )

                elif event_type == "response.refusal.done":
                    key = resolve_message_key(event=evt) or latest_message_key
                    if key is None:
                        continue

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    refusal = evt.get("refusal")
                    if not isinstance(refusal, str):
                        refusal = ""
                    set_part_text(
                        key=key,
                        content_index=_content_index_for_message_event(event=evt),
                        text=refusal,
                    )

                elif event_type == "response.content_part.done":
                    part = evt.get("part")
                    part_type = part.get("type") if isinstance(part, dict) else None
                    if part_type not in ("output_text", "refusal"):
                        continue

                    key = resolve_message_key(event=evt) or latest_message_key
                    if key is None:
                        continue

                    item_id = _item_id_for_message_event(event=evt)
                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    part_text = _message_text_from_content_part(part=part)
                    set_part_text(
                        key=key,
                        content_index=_content_index_for_message_event(event=evt),
                        text=part_text,
                    )

                elif event_type in (
                    "response.output_item.added",
                    "response.output_item.done",
                ):
                    item = evt.get("item")
                    item_id = _item_id_for_message_event(event=evt)
                    if not isinstance(item, dict):
                        await self._handle_custom_event_for_messages(
                            messages=doc_messages,
                            event=evt,
                        )
                        continue

                    item_type = item.get("type")
                    if item_type == "image_generation_call":
                        if event_type == "response.output_item.done":
                            await self.handle_image_generation_output_item(
                                event=evt,
                                item=item,
                                source="response.output_item.done",
                            )
                        else:
                            await self.handle_image_generation_started(
                                event=evt,
                                item=item,
                            )
                        continue

                    if item_type != "message":
                        await self._handle_custom_event_for_messages(
                            messages=doc_messages,
                            event=evt,
                        )
                        continue

                    if item_id is None:
                        key = latest_message_key
                    else:
                        key = f"item:{item_id}"

                    if key is None:
                        continue

                    ensure_message_element(key=key, item_id=item_id)
                    latest_message_key = key

                    message_text = _message_text_from_output_item(item=item)
                    if message_text != "":
                        set_part_text(key=key, content_index=0, text=message_text)

                elif event_type == "response.image_generation_call.partial_image":
                    await self.handle_image_generation_partial(event=evt)

                elif event_type == "meshagent.handler.added":
                    item = evt.get("item")
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "shell_call":
                        await self.handle_shell_call_output(item=item)

                    elif item_type == "local_shell_call":
                        await self.handle_local_shell_call_output(item=item)

                    elif item_type == "image_generation_call":
                        await self.handle_image_generation_output_item(
                            event=evt,
                            item=item,
                            source="meshagent.handler.added",
                        )

                else:
                    await self._handle_custom_event_for_messages(
                        messages=doc_messages,
                        event=evt,
                    )

        except asyncio.QueueShutDown:
            pass
        finally:
            updates.put_nowait(update_thread_stop)

        await update_thread_task

    def _messages_element(self) -> Element:
        if self._thread is None:
            raise RoomException("thread was not opened")

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                return prop

        raise RoomException("messages element is missing from thread document")

    def _resolve_image_item_id(
        self,
        *,
        event: dict,
        item: Optional[dict] = None,
    ) -> str:
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.strip() != "":
                return item_id

        event_item_id = event.get("item_id")
        if isinstance(event_item_id, str) and event_item_id.strip() != "":
            return event_item_id

        return str(uuid.uuid4())

    def _upsert_image_status(
        self,
        *,
        item_id: str,
        state: str,
        headline: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        created_by = self._room.local_participant.get_attribute("name")
        if not isinstance(created_by, str):
            created_by = ""

        self.write_image(
            message_id=item_id,
            width=width,
            height=height,
            created_by=created_by,
            status=_image_status_from_state(state=state),
            status_detail=headline,
        )

    async def _emit_image_status_event(
        self,
        *,
        messages: Element,
        item_id: str,
        state: str,
        headline: str,
        details: Optional[list[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        normalized_details = [line for line in (details or []) if isinstance(line, str)]
        await self._handle_custom_event_for_messages(
            messages=messages,
            event={
                "type": "agent.event",
                "source": "openai",
                "name": "response.image_generation_call.partial_image",
                "kind": "image",
                "state": state,
                "method": "response.image_generation_call.partial_image",
                # Match response_event_to_agent_event correlation semantics so
                # custom image status updates mutate the existing image event line.
                "correlation_key": f"item:{item_id}",
                "item_id": item_id,
                "item_type": "image_generation_call",
                "summary": headline,
                "headline": headline,
                "details": normalized_details,
                "data": json.dumps(
                    {
                        "item_id": item_id,
                        "state": state,
                        "headline": headline,
                        "details": normalized_details,
                        "width": width,
                        "height": height,
                    }
                ),
            },
        )
        self._upsert_image_status(
            item_id=item_id,
            state=state,
            headline=headline,
            width=width,
            height=height,
        )

    async def _persist_generated_image(
        self,
        *,
        item_id: str,
        image_bytes: bytes,
        mime_type: str,
        created_by: str,
        source: str,
        annotations: Optional[dict[str, str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        is_partial_source = source == _PARTIAL_IMAGE_SOURCE
        incoming_stage = (
            _IMAGE_STAGE_PARTIAL if is_partial_source else _IMAGE_STAGE_FINAL
        )

        existing_stage = self._saved_image_stage_by_item_id.get(item_id)
        if existing_stage == _IMAGE_STAGE_FINAL:
            return
        if (
            existing_stage == _IMAGE_STAGE_PARTIAL
            and incoming_stage == _IMAGE_STAGE_PARTIAL
        ):
            return

        messages = self._messages_element()
        progress_headline = (
            "Saving final image"
            if existing_stage == _IMAGE_STAGE_PARTIAL
            and incoming_stage == _IMAGE_STAGE_FINAL
            else "Saving image"
        )
        await self._emit_image_status_event(
            messages=messages,
            item_id=item_id,
            state="in_progress",
            headline=progress_headline,
            width=width,
            height=height,
        )

        payload_annotations = {
            "source": source,
            "item_id": item_id,
            "stage": incoming_stage,
        }
        if isinstance(annotations, dict):
            for key, value in annotations.items():
                payload_annotations[str(key)] = str(value)
        if width is not None:
            payload_annotations["width"] = str(width)
        if height is not None:
            payload_annotations["height"] = str(height)

        try:
            saved_image = await asyncio.wait_for(
                self._images_db.save(
                    data=image_bytes,
                    mime_type=mime_type,
                    created_by=created_by,
                    annotations=payload_annotations,
                ),
                timeout=_IMAGE_DB_SAVE_TIMEOUT_SECONDS,
            )
        except Exception as ex:
            logger.error("failed to save generated image to dataset", exc_info=ex)
            await self._emit_image_status_event(
                messages=messages,
                item_id=item_id,
                state="failed",
                headline="Image save failed",
                details=[str(ex)],
                width=width,
                height=height,
            )
            return

        try:
            self.write_image(
                message_id=item_id,
                image_id=saved_image.id,
                mime_type=saved_image.mime_type,
                created_at=saved_image.created_at,
                created_by=saved_image.created_by,
                width=width,
                height=height,
                status=_image_status_from_state(state="completed"),
                status_detail="Image saved",
            )
        except Exception as ex:
            logger.error("failed to attach saved image to thread", exc_info=ex)
            await self._emit_image_status_event(
                messages=messages,
                item_id=item_id,
                state="failed",
                headline="Image attach failed",
                details=[str(ex)],
                width=width,
                height=height,
            )
            return

        if incoming_stage == _IMAGE_STAGE_PARTIAL:
            await self._emit_image_status_event(
                messages=messages,
                item_id=item_id,
                state="completed",
                headline="Image saved",
                width=width,
                height=height,
            )
        else:
            await self._emit_image_status_event(
                messages=messages,
                item_id=item_id,
                state="completed",
                headline=(
                    "Final image saved"
                    if existing_stage == _IMAGE_STAGE_PARTIAL
                    else "Image saved"
                ),
                width=width,
                height=height,
            )

        self._saved_image_ids_by_item_id[item_id] = saved_image.id
        self._saved_image_stage_by_item_id[item_id] = incoming_stage
        logger.info("Saved generated image %s to images dataset", saved_image.id)

    async def handle_image_generation_started(
        self,
        *,
        event: dict,
        item: dict,
    ) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        if not isinstance(item, dict):
            return

        item_id = self._resolve_image_item_id(event=event, item=item)
        width, height = _extract_image_dimensions(item=item, event=event)

        normalized_state = _normalize_status_value(item.get("status")) or "in_progress"
        if normalized_state == "failed":
            headline = "Image generation failed"
        elif normalized_state == "cancelled":
            headline = "Image generation cancelled"
        elif normalized_state == "queued":
            headline = "Image generation queued"
        else:
            normalized_state = "in_progress"
            headline = "Generating image"

        messages = self._messages_element()
        await self._emit_image_status_event(
            messages=messages,
            item_id=item_id,
            state=normalized_state,
            headline=headline,
            width=width,
            height=height,
        )

    async def handle_image_generation_output_item(
        self,
        *,
        event: dict,
        item: dict,
        source: str,
    ) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        if not isinstance(item, dict):
            return

        item_id = self._resolve_image_item_id(event=event, item=item)
        width, height = _extract_image_dimensions(item=item, event=event)

        image_bytes = _image_bytes_from_output_item(item=item)
        if image_bytes is None:
            messages = self._messages_element()
            normalized_state = _normalize_status_value(
                item.get("status")
            ) or _normalize_status_value(event.get("status"))

            if normalized_state in ("failed", "cancelled"):
                details: list[str] = []
                detail = _first_nested_text(
                    value=item,
                    keys=("error", "errors", "message", "reason", "detail"),
                ).strip()
                if detail != "":
                    details.append(detail)
                await self._emit_image_status_event(
                    messages=messages,
                    item_id=item_id,
                    state=normalized_state,
                    headline=(
                        "Image generation failed"
                        if normalized_state == "failed"
                        else "Image generation cancelled"
                    ),
                    details=details,
                    width=width,
                    height=height,
                )
            elif normalized_state in ("queued", "in_progress"):
                await self._emit_image_status_event(
                    messages=messages,
                    item_id=item_id,
                    state=normalized_state,
                    headline=(
                        "Image generation queued"
                        if normalized_state == "queued"
                        else "Generating image"
                    ),
                    width=width,
                    height=height,
                )
            return

        output_format = item.get("output_format")
        if not isinstance(output_format, str) or output_format.strip() == "":
            output_format = event.get("output_format")
        mime_type = _mime_type_from_output_format(output_format=output_format)

        created_by = self._room.local_participant.get_attribute("name")
        if not isinstance(created_by, str):
            created_by = ""

        item_annotations: dict[str, str] = {}
        for key in ("status", "size", "quality", "background", "output_format"):
            value = item.get(key)
            if isinstance(value, str) and value.strip() != "":
                item_annotations[key] = value.strip()

        await self._persist_generated_image(
            item_id=item_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            created_by=created_by,
            source=source,
            annotations=item_annotations,
            width=width,
            height=height,
        )

    async def handle_image_generation_partial(
        self,
        *,
        event: dict,
    ) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        item_id = self._resolve_image_item_id(event=event)
        width, height = _extract_image_dimensions(event=event)

        messages = self._messages_element()
        await self._emit_image_status_event(
            messages=messages,
            item_id=item_id,
            state="in_progress",
            headline="Generating image",
            width=width,
            height=height,
        )

    async def handle_local_shell_call_output(
        self,
        *,
        item: dict,
    ) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        command = action["command"]
        working_dir = action["working_dir"]

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        exec_element = messages.append_child(
            tag_name="exec",
            attributes={"command": shlex.join(command), "pwd": working_dir},
        )

        evt = await self._llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = evt.get("error")
        item = evt.get("item")

        if error is not None:
            pass

        if item is not None:
            if item["type"] != "local_shell_call_output":
                raise RoomException("expected local_shell_call_output")

            exec_element.set_attribute("result", item["output"])

    async def handle_shell_call_output(
        self,
        *,
        item: dict,
    ) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        commands = action["commands"]

        exec_elements = []
        for command in commands:
            exec_element = messages.append_child(
                tag_name="exec",
                attributes={"command": command},
            )
            exec_elements.append(exec_element)

        evt = await self._llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = evt.get("error")
        item = evt.get("item")

        if error is not None:
            pass

        if item is not None:
            if item["type"] != "shell_call_output":
                raise RoomException("expected shell_call_output")

            results = item["output"]

            for i in range(0, len(results)):
                result = results[i]
                exec_element = exec_elements[i]
                if "exit_code" in result["outcome"]:
                    exec_element.set_attribute(
                        "exit_code", result["outcome"]["exit_code"]
                    )

                exec_element.set_attribute("outcome", result["outcome"]["type"])
                exec_element.set_attribute("stdout", result["stdout"])
                exec_element.set_attribute("stderr", result["stderr"])
