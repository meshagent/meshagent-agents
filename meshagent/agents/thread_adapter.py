import asyncio
import base64
import json
import logging
import shlex
import uuid
from meshagent.tools import tool, Toolkit
from datetime import datetime, timezone
from typing import Any, Optional, Callable
from meshagent.api import RemoteParticipant

from opentelemetry import trace

from meshagent.agents.agent import AgentChatContext
from meshagent.agents.thread_schema import thread_schema
from meshagent.api import (
    RoomClient,
    Element,
    MeshDocument,
    RoomException,
)


tracer = trace.get_tracer("meshagent.thread_adapter")

logger = logging.getLogger("thread_adapter")

_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_RESPONSE_NOISE_TYPES = {
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
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
    ):
        return "tool"
    if normalized in ("websearchcall",):
        return "web"
    if normalized in ("filesearchcall",):
        return "search"
    if normalized in ("applypatchcall",):
        return "diff"
    if normalized in ("codeinterpretercall", "computercall", "shellcall"):
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


def _headline_for_response_event(*, kind: str, state: str) -> str:
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

    headline = _headline_for_response_event(kind=kind, state=state)
    details = _details_for_response_event(event=event, kind=kind)
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


def default_format_message(self, *, user_name: str, message: str, iso_timestamp: str):
    return f"{user_name} said at {iso_timestamp}: {message}"


class ThreadAdapter:
    def __init__(
        self,
        *,
        room: RoomClient,
        path: str,
        format_message: Optional[Callable] = None,
        max_append_message_count: int = 25,
    ):
        self._room = room
        self._thread_path = path
        self._processor_task: Optional[asyncio.Task] = None
        self._llm_messages: asyncio.Queue = asyncio.Queue()
        self._thread: Optional[MeshDocument] = None
        self._format_message = format_message or default_format_message
        self._max_append_message_count = max_append_message_count
        self._active_events_by_key: dict[str, Element] = {}

    async def start(self) -> None:
        self._thread = await self._room.sync.open(
            path=self._thread_path, schema=thread_schema
        )

        self._processor_task = asyncio.create_task(self._process_llm_events())

    @property
    def thread(self) -> Optional[MeshDocument]:
        return self._thread

    async def stop(self) -> None:
        self._llm_messages.shutdown()
        if self._processor_task is not None:
            await self._processor_task
            self._processor_task = None
        self._active_events_by_key.clear()

        if self._thread is not None:
            await self._room.sync.close(path=self._thread_path)
            self._thread = None

    def push(self, *, event: dict) -> None:
        self._llm_messages.put_nowait(event)

    async def handle_custom_event(
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
            event_element = messages.append_child(
                tag_name="event",
                attributes={
                    "id": str(uuid.uuid4()),
                    "source": source,
                    "name": name,
                    "kind": kind,
                    "state": state,
                    "method": method,
                    "item_id": item_id,
                    "item_type": item_type,
                    "summary": summary,
                    "headline": headline,
                    "details": details,
                    "data": data,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        else:
            event_element.set_attribute("source", source)
            event_element.set_attribute("name", name)
            event_element.set_attribute("kind", kind)
            event_element.set_attribute("state", state)
            event_element.set_attribute("method", method)
            event_element.set_attribute("item_id", item_id)
            event_element.set_attribute("item_type", item_type)
            event_element.set_attribute("summary", summary)
            event_element.set_attribute("headline", headline)
            if details != "" or event_element.get_attribute("details") in (None, ""):
                event_element.set_attribute("details", details)
            event_element.set_attribute("data", data)
            event_element.set_attribute("updated_at", now)

        if correlation_key is not None:
            if in_progress:
                self._active_events_by_key[correlation_key] = event_element
            elif state in _TERMINAL_STATES:
                self._active_events_by_key.pop(correlation_key, None)

    def make_toolkit(self):
        toolkit = Toolkit(
            name="search",
            description="tools for searching conversation history",
            tools=[
                self.grep_tool,
                self.get_message_range,
                self.count_tool,
            ],
        )
        return toolkit

    @tool(
        name="get_message_range",
        description="gets a range of messages, index 0 is the first message in the conversation",
    )
    def get_message_range(self, *, start: int, end: int) -> str:
        all_items = self._thread.root.get_children_by_tag_name("messages")[
            0
        ].get_children()
        messages = [
            item
            for item in all_items
            if isinstance(item, Element) and item.tag_name == "message"
        ]

        elements = messages[start:end]

        if len(elements) == 0:
            return "no messages were found within the specified range"

        response = "matching messages:\n"

        for element in elements:
            response = response + self._format_message(
                user_name=element["author_name"],
                message=element["text"],
                iso_timestamp=element["created_at"],
            )

        return response

    @tool(
        name="count_current_thread_messages",
        description="return the number of messages in the current thread (including those outside the context window)",
    )
    def count_tool(
        self,
        *,
        pattern: str,
        ignore_case: bool,
        messages_before: int,
        messages_after: int,
    ) -> str:
        messages = self._thread.root.get_children_by_tag_name("messages")[0]
        count = len(
            [
                item
                for item in messages.get_children()
                if isinstance(item, Element) and item.tag_name == "message"
            ]
        )
        return f"{count}"

    @tool(
        name="grep_current_thread",
        description="search the current thread for text, includes messages outside the current context window",
    )
    def grep_tool(
        self,
        *,
        pattern: str,
        ignore_case: bool,
        messages_before: int,
        messages_after: int,
    ) -> str:
        messages = self._thread.root.get_children_by_tag_name("messages")[0]
        elements = [
            item
            for item in messages.grep(
                pattern,
                ignore_case=ignore_case,
                before=messages_before,
                after=messages_after,
            )
            if isinstance(item, Element) and item.tag_name == "message"
        ]
        if len(elements) == 0:
            return "no messages were found with the specified pattern"

        response = "matching messages:\n"

        for element in elements:
            response = response + self._format_message(
                user_name=element["author_name"],
                message=element["text"],
                iso_timestamp=element["created_at"],
            )

        return response

    def append_messages(self, *, context: AgentChatContext) -> None:
        doc_messages = None

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop

                messages = [
                    element
                    for element in doc_messages.get_children()
                    if isinstance(element, Element) and element.tag_name == "message"
                ]
                if len(messages) > self._max_append_message_count:
                    first_message = len(messages) - self._max_append_message_count
                    messages = messages[first_message:]
                    context.append_assistant_message(
                        f"there are more messages outside the current context window, the index of the first message loaded is {first_message}"
                    )

                for element in messages:
                    if isinstance(element, Element) and element.tag_name == "message":
                        msg = element["text"]
                        if element[
                            "author_name"
                        ] == self._room.local_participant.get_attribute("name"):
                            context.append_assistant_message(msg)
                        else:
                            context.append_user_message(
                                self._format_message(
                                    user_name=element["author_name"],
                                    message=element["text"],
                                    iso_timestamp=element["created_at"],
                                )
                            )

                        for child in element.get_children():
                            if child.tag_name == "file":
                                context.append_assistant_message(
                                    f"the user attached a file with the path '{child.get_attribute('path')}'"
                                )

                break

        if doc_messages is None:
            raise Exception("thread was not properly initialized")

    def write_text_message(
        self, *, text: str, participant: RemoteParticipant | str
    ) -> None:
        doc_messages: Element = self._thread.root.get_children_by_tag_name("messages")[
            0
        ]

        doc_messages.append_child(
            tag_name="message",
            attributes={
                "text": text,
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": participant.get_attribute("name")
                if isinstance(participant, RemoteParticipant)
                else participant,
            },
        )

    async def _process_llm_events(
        self,
    ):
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
        content_element = None
        partial = ""

        # throttle updates so we don't send too many syncs over the wire at once
        async def update_thread():
            changes = {}
            try:
                while True:
                    try:
                        element, partial_text = updates.get_nowait()
                        changes[element] = partial_text

                    except asyncio.QueueEmpty:
                        for element, partial_text in changes.items():
                            element["text"] = partial_text

                        changes.clear()

                        element, partial_text = await updates.get()
                        changes[element] = partial_text

                        # await asyncio.sleep(0.1)

            except asyncio.QueueShutDown:
                # flush any pending changes
                for element, partial_text in changes.items():
                    element["text"] = partial_text

                changes.clear()
                pass

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await self._llm_messages.get()

                if evt["type"] == "response.content_part.added":
                    partial = ""

                    content_element = doc_messages.append_child(
                        tag_name="message",
                        attributes={
                            "text": "",
                            "created_at": datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "author_name": self._room.local_participant.get_attribute(
                                "name"
                            ),
                        },
                    )

                elif evt["type"] == "response.output_text.delta":
                    partial += evt["delta"]
                    updates.put_nowait((content_element, partial))

                elif evt["type"] == "response.output_text.done":
                    content_element = None
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attribute(
                            "from_participant_name",
                            self._room.local_participant.get_attribute("name"),
                        )
                        span.set_attribute("role", "assistant")
                        span.set_attributes({"text": evt["text"]})

                elif evt["type"] == "response.image_generation_call.partial_image":
                    await self.handle_image_generation_partial(event=evt)

                elif evt["type"] == "meshagent.handler.added":
                    item = evt["item"]
                    if item["type"] == "shell_call":
                        await self.handle_shell_call_output(item=item)

                    elif item["type"] == "local_shell_call":
                        await self.handle_local_shell_call_output(item=item)

                else:
                    await self.handle_custom_event(messages=doc_messages, event=evt)

        except asyncio.QueueShutDown:
            pass
        finally:
            updates.shutdown()

        await update_thread_task

    async def handle_image_generation_partial(
        self,
        *,
        event: dict,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        item_id = event["item_id"]
        partial_image_b64 = event["partial_image_b64"]
        output_format = event["output_format"]

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        if output_format is None:
            output_format = "png"

        image_name = f"{str(uuid.uuid4())}.{output_format}"

        handle = await self._room.storage.open(path=image_name)
        await self._room.storage.write(
            handle=handle, data=base64.b64decode(partial_image_b64)
        )
        await self._room.storage.close(handle=handle)

        messages = None

        logger.info(f"A partial was saved at the path {image_name}")

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        for child in messages.get_children():
            if child.get_attribute("id") == item_id:
                for file in child.get_children():
                    file.set_attribute("path", image_name)

                return

        message_element = messages.append_child(
            tag_name="message",
            attributes={
                "id": item_id,
                "text": "",
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": self._room.local_participant.get_attribute("name"),
            },
        )
        message_element.append_child(tag_name="file", attributes={"path": image_name})

    async def handle_local_shell_call_output(
        self,
        *,
        item: dict,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        command = action["command"]
        working_directory = action["working_directory"]

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        exec_element = messages.append_child(
            tag_name="exec",
            attributes={"command": shlex.join(command), "pwd": working_directory},
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
    ):
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
