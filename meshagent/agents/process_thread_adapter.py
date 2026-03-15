from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shlex
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from meshagent.api import Element, Participant, RoomException
from meshagent.api.chan import ChanClosed
from meshagent.api.messaging import (
    Content,
    JsonContent,
)

from .context import AgentSessionContext
from .messages import (
    AGENT_EVENT_TURN_INTERRUPTED,
    AgentError,
    AgentFileContent,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallApprovalRequested,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallPending,
    AgentToolCallStarted,
    ThreadCleared,
    TurnEnded,
    TurnInterrupted,
    TurnStart,
    TurnStarted,
    TurnSteer,
    TurnSteerAccepted,
    TurnSteered,
    TurnSteerRejected,
)
from .shell_semantics import analyze_shell_command
from .thread_adapter import ThreadAdapter

logger = logging.getLogger("agent.process_thread_adapter")

ThreadStatusMode = Literal["busy", "steerable"]
_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
EXEC_SEARCH_DETAIL_LIMIT = 2000
DIFF_PREVIEW_LIMIT = 12000
EVENT_LOG_LINE_LIMIT = 10
_APPLY_PATCH_PATH_RES = (
    re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (?P<path>.+)$", re.MULTILINE),
    re.compile(r"^(?:\+\+\+ b/|--- a/)(?P<path>.+)$", re.MULTILINE),
)


@dataclass(frozen=True, slots=True)
class _ActiveToolCall:
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _ClearThreadRequest:
    future: asyncio.Future[None]


@dataclass(slots=True)
class ThreadAdapterMessage:
    message: AgentMessage
    sender: Participant | None = None


@dataclass(frozen=True, slots=True)
class _ExecSearchPreview:
    query: str
    paths: tuple[str, ...]

    @property
    def path(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0]
        return ""


@dataclass(frozen=True, slots=True)
class _StorageReadToolCallDisplay:
    path: str


@dataclass(frozen=True, slots=True)
class _StorageWriteToolCallDisplay:
    path: str


@dataclass(frozen=True, slots=True)
class _NewThreadToolCallDisplay:
    path: str
    name: str


def _storage_tool_call_display(
    *,
    toolkit: str,
    tool: str,
    arguments: dict[str, Any] | None,
) -> _StorageReadToolCallDisplay | _StorageWriteToolCallDisplay | None:
    if arguments is None:
        return None

    if toolkit.strip().lower() != "storage":
        return None

    normalized_tool = tool.strip().lower()

    if normalized_tool == "read_file":
        path = arguments.get("path")
        if isinstance(path, str):
            normalized_path = path.strip()
            if normalized_path != "":
                return _StorageReadToolCallDisplay(path=normalized_path)
        return None

    if normalized_tool == "write_file":
        path = arguments.get("path")
        if isinstance(path, str):
            normalized_path = path.strip()
            if normalized_path != "":
                return _StorageWriteToolCallDisplay(
                    path=normalized_path,
                )
        return None

    return None


def _new_thread_tool_call_display(
    *,
    toolkit: str,
    tool: str,
    result: Content | None,
) -> _NewThreadToolCallDisplay | None:
    if toolkit.strip().lower() != "chat" or tool.strip().lower() != "new_thread":
        return None

    if not isinstance(result, JsonContent):
        return None

    raw_json = result.json
    if not isinstance(raw_json, dict):
        return None

    raw_path = raw_json.get("path")
    if not isinstance(raw_path, str):
        return None

    path = raw_path.strip()
    if path == "":
        return None

    raw_name = raw_json.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    if name == "":
        fallback_name = Path(path).stem.strip()
        name = fallback_name if fallback_name != "" else path

    return _NewThreadToolCallDisplay(path=path, name=name)


def _storage_grep_preview(
    *,
    toolkit: str,
    tool: str,
    arguments: dict[str, Any] | None,
) -> _ExecSearchPreview | None:
    if arguments is None:
        return None

    if toolkit.strip().lower() != "storage" or tool.strip().lower() != "grep_file":
        return None

    path = arguments.get("path")
    pattern = arguments.get("pattern")
    if not isinstance(path, str) or not isinstance(pattern, str):
        return None

    normalized_path = path.strip()
    normalized_pattern = pattern.strip()
    if normalized_path == "" or normalized_pattern == "":
        return None

    return _ExecSearchPreview(
        query=normalized_pattern,
        paths=(normalized_path,),
    )


@dataclass(frozen=True, slots=True)
class _NormalizedThreadEvent:
    source: str
    name: str
    kind: str
    state: str
    method: str
    turn_id: str = ""
    item_id: str = ""
    item_type: str = ""
    summary: str = ""
    headline: str = ""
    details: tuple[str, ...] = ()
    preview: str = ""
    path: str = ""
    data: str = ""
    correlation_key: str | None = None
    retain_correlation: bool = False
    drop_correlation_keys: tuple[str, ...] = ()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _humanize_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _terminal_state_from_error(error: AgentError | None) -> str:
    if error is None:
        return "completed"

    if error.code == "cancelled":
        return "cancelled"

    return "failed"


def _truncate_text(
    text: str,
    *,
    max_length: int | None = None,
    limit: int | None = None,
) -> str:
    resolved_limit = limit if limit is not None else max_length
    if resolved_limit is None:
        resolved_limit = 8000
    if len(text) <= resolved_limit:
        return text
    return text[:resolved_limit] + "..."


class AgentProcessThreadAdapter(ThreadAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_message_elements_by_key: dict[str, Element] = {}
        self._active_reasoning_elements_by_item_id: dict[str, Element] = {}
        self._active_event_elements_by_key: dict[str, Element] = {}
        self._active_event_elements_by_item_id: dict[str, Element] = {}
        self._active_tool_calls_by_item_id: dict[str, _ActiveToolCall] = {}
        self._pending_turn_ids_by_message_id: dict[str, str] = {}
        self._thread_status_lock = asyncio.Lock()
        self._thread_status_generation = 0
        self._thread_status_key: str | None = None
        self._thread_status_value: str | None = None
        self._thread_status_mode_value: ThreadStatusMode | None = None
        self._thread_status_started_at_value: str | None = None
        self._thread_status_turn_id_value: str | None = None
        self._thread_status_pending_messages_value: list[dict[str, Any]] = []
        self._thread_status_pending_item_id_value: str | None = None

    async def start(self) -> None:
        await super().start()
        self._ensure_local_member_on_thread()

    async def stop(self) -> None:
        await self.set_thread_turn_id(turn_id=None)
        await self.set_pending_messages(pending_messages=[])
        await self.clear_thread_status()
        await super().stop()
        self._active_message_elements_by_key.clear()
        self._active_reasoning_elements_by_item_id.clear()
        self._active_event_elements_by_key.clear()
        self._active_event_elements_by_item_id.clear()
        self._active_tool_calls_by_item_id.clear()
        self._pending_turn_ids_by_message_id.clear()

    async def clear_thread(self) -> None:
        if self._processor_task is None or self._processor_task.done():
            await self._clear_thread_contents(messages=self._messages_element())
            return

        clear_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        try:
            self._llm_messages.put_nowait(_ClearThreadRequest(future=clear_future))
        except asyncio.QueueShutDown:
            logger.debug("clearing thread directly after queue shutdown")
            await self._clear_thread_contents(messages=self._messages_element())
            return

        await clear_future

    def push_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant | None = None,
    ) -> None:
        try:
            self._llm_messages.put_nowait(
                ThreadAdapterMessage(message=message, sender=sender)
            )
        except asyncio.QueueShutDown:
            logger.debug("dropping thread adapter message after queue shutdown")

    def restore_session_context(self, *, context: AgentSessionContext) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._message_elements()
        if len(messages) > self._max_append_message_count:
            first_message = len(messages) - self._max_append_message_count
            messages = messages[first_message:]
            context.append_assistant_message(
                "there are more messages outside the current context window, "
                f"the index of the first message loaded is {first_message}"
            )

        local_name = self._local_participant_name()
        for message in messages:
            author_name = self._attribute_as_str(message, "author_name")
            created_at = self._attribute_as_str(message, "created_at")
            text = self._attribute_as_str(message, "text")
            is_local_author = author_name == local_name and author_name != ""

            if text != "":
                if is_local_author:
                    context.append_assistant_message(text)
                else:
                    context.append_user_message(
                        self._format_message(
                            user_name=author_name or "user",
                            message=text,
                            iso_timestamp=created_at,
                        )
                    )

            for child in message.get_children():
                if child.tag_name != "file":
                    continue

                path = self._attribute_as_str(child, "path")
                if path == "":
                    continue

                if is_local_author:
                    context.append_assistant_message(
                        f"assistant attached a file available at {path}"
                    )
                else:
                    context.append_user_message(
                        f"{author_name or 'a user'} attached a file available at {path}"
                    )

    async def handle_custom_event(
        self,
        *,
        messages: Element,
        event: dict,
    ) -> None:
        normalized_event = self._normalized_event_from_dict(event)
        if normalized_event is None:
            return

        self._upsert_event(messages=messages, event=normalized_event)
        await self._update_thread_status_from_event(event=normalized_event)

    async def _process_llm_events(self) -> None:
        messages = self._messages_element()

        while True:
            try:
                queued_entry = await self._llm_messages.get()
            except asyncio.QueueShutDown:
                break

            if isinstance(queued_entry, _ClearThreadRequest):
                try:
                    await self._clear_thread_contents(messages=messages)
                except Exception as exc:
                    if not queued_entry.future.done():
                        queued_entry.future.set_exception(exc)
                else:
                    if not queued_entry.future.done():
                        queued_entry.future.set_result(None)
                continue

            if isinstance(queued_entry, ThreadAdapterMessage):
                await self._handle_agent_message(
                    messages=messages,
                    message=queued_entry.message,
                    sender=queued_entry.sender,
                )
                continue

            if isinstance(queued_entry, dict):
                await self.handle_custom_event(
                    messages=messages,
                    event=queued_entry,
                )

    async def _handle_agent_message(
        self,
        *,
        messages: Element,
        message: AgentMessage,
        sender: Participant | None,
    ) -> None:
        normalized_event: _NormalizedThreadEvent | None = None
        if isinstance(message, ThreadCleared):
            await self.set_thread_turn_id(turn_id=None)
            await self.set_pending_messages(pending_messages=[])

        if isinstance(message, (TurnStart, TurnSteer)):
            self._write_turn_message(sender=sender, turn=message)
        elif isinstance(message, AgentTextContentStarted):
            self._ensure_assistant_message(
                messages=messages,
                key=self._content_message_key(kind="text", item_id=message.item_id),
                turn_id=message.turn_id,
            )
        elif isinstance(message, AgentTextContentDelta):
            assistant_message = self._ensure_assistant_message(
                messages=messages,
                key=self._content_message_key(kind="text", item_id=message.item_id),
                turn_id=message.turn_id,
            )
            current_text = self._attribute_as_str(assistant_message, "text")
            assistant_message.set_attribute("text", current_text + message.text)
        elif isinstance(message, AgentTextContentEnded):
            self._active_message_elements_by_key.pop(
                self._content_message_key(kind="text", item_id=message.item_id),
                None,
            )
        elif isinstance(message, AgentFileContentStarted):
            self._ensure_assistant_message(
                messages=messages,
                key=self._content_message_key(kind="file", item_id=message.item_id),
                turn_id=message.turn_id,
            )
        elif isinstance(message, AgentFileContentDelta):
            assistant_message = self._ensure_assistant_message(
                messages=messages,
                key=self._content_message_key(kind="file", item_id=message.item_id),
                turn_id=message.turn_id,
            )
            file_element = self._ensure_file_element(message=assistant_message)
            file_element.set_attribute("path", message.url)
        elif isinstance(message, AgentFileContentEnded):
            self._active_message_elements_by_key.pop(
                self._content_message_key(kind="file", item_id=message.item_id),
                None,
            )
        elif isinstance(message, AgentReasoningContentStarted):
            self._ensure_reasoning_element(
                messages=messages,
                item_id=message.item_id,
                turn_id=message.turn_id,
            )
        elif isinstance(message, AgentReasoningContentDelta):
            reasoning = self._ensure_reasoning_element(
                messages=messages,
                item_id=message.item_id,
                turn_id=message.turn_id,
            )
            current_summary = self._attribute_as_str(reasoning, "summary")
            reasoning.set_attribute("summary", current_summary + message.text)
        elif isinstance(message, AgentReasoningContentEnded):
            self._active_reasoning_elements_by_item_id.pop(message.item_id, None)
        elif isinstance(message, AgentToolCallLogDelta):
            self._append_event_logs(
                messages=messages,
                item_id=message.item_id,
                lines=message.lines,
            )
        else:
            normalized_event = self._normalized_event_from_message(message)
            if normalized_event is not None:
                self._upsert_event(messages=messages, event=normalized_event)

        if isinstance(message, TurnStarted):
            if not self.set_message_turn_id(
                message_id=message.source_message_id,
                turn_id=message.turn_id,
            ):
                self._pending_turn_ids_by_message_id[message.source_message_id] = (
                    message.turn_id
                )
            await self.set_thread_turn_id(turn_id=message.turn_id)
            if normalized_event is not None:
                await self._update_thread_status_from_event(event=normalized_event)
        elif isinstance(message, TurnEnded):
            if self._thread_status_turn_id_value == message.turn_id:
                await self.set_thread_turn_id(turn_id=None)
            self._clear_active_turn_state()
            await self.clear_thread_status()
        else:
            if normalized_event is not None:
                await self._update_thread_status_from_event(event=normalized_event)
            if isinstance(message, AgentToolCallEnded):
                self._active_tool_calls_by_item_id.pop(message.item_id, None)

    def _write_turn_message(
        self,
        *,
        sender: Participant | None,
        turn: TurnStart | TurnSteer,
    ) -> None:
        text_parts: list[str] = []
        attachments: list[dict[str, str]] = []

        for item in turn.content:
            if isinstance(item, AgentTextContent):
                normalized_text = item.text.strip()
                if normalized_text != "":
                    text_parts.append(normalized_text)
            elif isinstance(item, AgentFileContent):
                normalized_path = self._thread_attachment_path(url=item.url)
                if normalized_path != "":
                    attachments.append({"path": normalized_path})

        if len(text_parts) == 0 and len(attachments) == 0:
            return

        participant: Participant | str = sender if sender is not None else "user"
        turn_id: str | None = turn.turn_id if isinstance(turn, TurnSteer) else None
        if turn_id is None:
            turn_id = self._pending_turn_ids_by_message_id.pop(turn.message_id, None)
        else:
            self._pending_turn_ids_by_message_id.pop(turn.message_id, None)
        self.write_text_message(
            text="\n\n".join(text_parts),
            participant=participant,
            message_id=turn.message_id,
            turn_id=turn_id,
            attachments=attachments if len(attachments) > 0 else None,
        )

    @staticmethod
    def _thread_attachment_path(*, url: str) -> str:
        normalized_url = url.strip()
        if normalized_url == "":
            return ""

        parsed = urlparse(normalized_url)
        if parsed.scheme != "room":
            return normalized_url

        room_path = f"{parsed.netloc}{parsed.path}".lstrip("/")
        if room_path == "":
            return normalized_url

        return room_path

    def _messages_element(self) -> Element:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            raise RoomException("messages element is missing from thread document")

        return messages[0]

    def _message_elements(self) -> list[Element]:
        messages = self._messages_element()
        return [
            child for child in messages.get_children() if child.tag_name == "message"
        ]

    def _ensure_local_member_on_thread(self) -> None:
        self.ensure_member(participant=self._room.local_participant)

    def _local_participant_name(self) -> str:
        value = self._room.local_participant.get_attribute("name")
        if isinstance(value, str):
            return value
        return ""

    @staticmethod
    def _attribute_as_str(element: Element, name: str) -> str:
        value = element.get_attribute(name)
        if isinstance(value, str):
            return value
        return ""

    def _content_message_key(
        self, *, kind: Literal["file", "text"], item_id: str
    ) -> str:
        return f"{kind}:{item_id}"

    def _ensure_assistant_message(
        self,
        *,
        messages: Element,
        key: str,
        turn_id: str | None = None,
    ) -> Element:
        existing = self._active_message_elements_by_key.get(key)
        if existing is not None:
            if isinstance(turn_id, str) and turn_id.strip() != "":
                existing.set_attribute("turn_id", turn_id.strip())
            return existing

        existing_id = key
        for child in messages.get_children():
            if child.tag_name != "message":
                continue

            if self._attribute_as_str(child, "id") == existing_id:
                if isinstance(turn_id, str) and turn_id.strip() != "":
                    child.set_attribute("turn_id", turn_id.strip())
                self._active_message_elements_by_key[key] = child
                return child

        assistant_message_attributes: dict[str, Any] = {
            "id": existing_id,
            "text": "",
            "created_at": _now_iso(),
            "author_name": self._local_participant_name(),
        }
        if isinstance(turn_id, str) and turn_id.strip() != "":
            assistant_message_attributes["turn_id"] = turn_id.strip()

        assistant_message = messages.append_child(
            tag_name="message",
            attributes=assistant_message_attributes,
        )
        self._active_message_elements_by_key[key] = assistant_message
        return assistant_message

    def _ensure_file_element(self, *, message: Element) -> Element:
        for child in message.get_children():
            if child.tag_name == "file":
                return child

        return message.append_child(tag_name="file", attributes={"path": ""})

    def _ensure_reasoning_element(
        self,
        *,
        messages: Element,
        item_id: str,
        turn_id: str | None = None,
    ) -> Element:
        existing = self._active_reasoning_elements_by_item_id.get(item_id)
        if existing is not None:
            if isinstance(turn_id, str) and turn_id.strip() != "":
                existing.set_attribute("turn_id", turn_id.strip())
            return existing

        reasoning_attributes: dict[str, str] = {
            "summary": "",
            "created_at": _now_iso(),
        }
        if isinstance(turn_id, str) and turn_id.strip() != "":
            reasoning_attributes["turn_id"] = turn_id.strip()

        reasoning = messages.append_child(
            tag_name="reasoning",
            attributes=reasoning_attributes,
        )
        self._active_reasoning_elements_by_item_id[item_id] = reasoning
        return reasoning

    def _clear_active_turn_state(self) -> None:
        self._active_message_elements_by_key.clear()
        self._active_reasoning_elements_by_item_id.clear()
        self._active_tool_calls_by_item_id.clear()
        self._pending_turn_ids_by_message_id.clear()

    async def _clear_thread_contents(self, *, messages: Element) -> None:
        self._clear_active_turn_state()
        self._active_event_elements_by_key.clear()
        self._active_event_elements_by_item_id.clear()
        for child in list(messages.get_children()):
            child.delete()
        await self.clear_thread_status()

    def _normalized_event_from_dict(
        self, event: dict[str, Any]
    ) -> _NormalizedThreadEvent | None:
        event_type = event.get("type")
        if event_type != "agent.event":
            return None

        kind = event.get("kind")
        state = event.get("state")
        name = event.get("name")
        if (
            not isinstance(kind, str)
            or not isinstance(state, str)
            or not isinstance(name, str)
        ):
            return None

        raw_drop_keys = event.get("drop_correlation_keys")
        drop_correlation_keys: tuple[str, ...]
        if isinstance(raw_drop_keys, list):
            drop_correlation_keys = tuple(
                key.strip()
                for key in raw_drop_keys
                if isinstance(key, str) and key.strip() != ""
            )
        else:
            drop_correlation_keys = ()

        correlation_key = event.get("correlation_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = event.get("event_key")
        if isinstance(correlation_key, str):
            correlation_key = correlation_key.strip() or None
        else:
            correlation_key = None

        path = event.get("path")
        if not isinstance(path, str):
            path = ""

        preview = event.get("preview")
        if not isinstance(preview, str):
            preview = ""

        data = event.get("data")
        if not isinstance(data, str):
            data = ""

        return _NormalizedThreadEvent(
            source=event.get("source")
            if isinstance(event.get("source"), str)
            else "agent",
            name=name,
            kind=kind.strip().lower(),
            state=state.strip().lower(),
            method=event.get("method")
            if isinstance(event.get("method"), str)
            else name,
            turn_id=event.get("turn_id")
            if isinstance(event.get("turn_id"), str)
            else "",
            item_id=event.get("item_id")
            if isinstance(event.get("item_id"), str)
            else "",
            item_type=event.get("item_type")
            if isinstance(event.get("item_type"), str)
            else "",
            summary=event.get("summary")
            if isinstance(event.get("summary"), str)
            else name,
            headline=event.get("headline")
            if isinstance(event.get("headline"), str)
            else "",
            details=self._event_detail_lines(event.get("details")),
            preview=preview,
            path=path,
            data=data,
            correlation_key=correlation_key,
            retain_correlation=event.get("retain_correlation") is True,
            drop_correlation_keys=drop_correlation_keys,
        )

    @staticmethod
    def _normalize_name(*, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    @staticmethod
    def _event_detail_lines(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized == "":
                return ()
            return tuple(line for line in normalized.splitlines() if line.strip() != "")

        if isinstance(value, list):
            lines: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                normalized = item.strip()
                if normalized == "":
                    continue
                lines.append(normalized)
            return tuple(lines)

        return ()

    @staticmethod
    def _details_text(*, details: tuple[str, ...]) -> str:
        return "\n".join(details)

    @staticmethod
    def _stringify_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _message_event_data(self, *, message: AgentMessage) -> str:
        return _truncate_text(
            self._stringify_json(message.model_dump(mode="json")),
        )

    def _status_event_details(
        self,
        *,
        event: _NormalizedThreadEvent,
    ) -> tuple[str | None, str | None, str | None]:
        if event.name == AGENT_EVENT_TURN_INTERRUPTED:
            text = event.headline.strip()
            if text == "":
                text = event.summary.strip()
            if text == "":
                text = event.name.strip()
            return None, event.state, text or None

        key = event.correlation_key
        if key is None and event.item_id.strip() != "":
            key = event.item_id.strip()
        if key is None:
            normalized_name = event.name.strip()
            if normalized_name != "":
                key = normalized_name

        text = event.headline.strip()
        if text == "":
            text = event.summary.strip()
        if text == "":
            text = event.name.strip()

        return key, event.state, text or None

    def _command_text(self, *, value: Any, multiline: bool = False) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            string_items = [item.strip() for item in value if isinstance(item, str)]
            string_items = [item for item in string_items if item != ""]
            if len(string_items) == len(value) and len(string_items) > 0:
                if multiline:
                    return "\n".join(string_items)
                with contextlib.suppress(ValueError, TypeError):
                    return shlex.join(string_items)
                return " ".join(string_items)

            parts = [
                self._command_text(value=item, multiline=multiline) for item in value
            ]
            parts = [part for part in parts if part != ""]
            if len(parts) == 0:
                return ""
            return "\n".join(parts) if multiline else " ".join(parts)

        if isinstance(value, dict):
            for key in ("command", "commands", "cmd", "code", "text", "value"):
                if key not in value:
                    continue
                text = self._command_text(
                    value=value[key],
                    multiline=multiline or key == "commands",
                )
                if text != "":
                    return text
            nested = value.get("content")
            if nested is not None:
                return self._command_text(value=nested, multiline=multiline)

        return ""

    def _first_nested_text(self, *, value: Any, keys: tuple[str, ...]) -> str:
        key_set = {key.lower() for key in keys}

        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() not in key_set:
                    continue
                text = self._command_text(value=nested, multiline=key.endswith("s"))
                if text != "":
                    return text

            for nested in value.values():
                text = self._first_nested_text(value=nested, keys=keys)
                if text != "":
                    return text

        elif isinstance(value, list):
            for nested in value:
                text = self._first_nested_text(value=nested, keys=keys)
                if text != "":
                    return text

        return ""

    def _extract_tool_command(
        self,
        *,
        tool: str,
        arguments: dict[str, Any] | None,
    ) -> str:
        if arguments is None:
            return ""

        action = arguments.get("action")
        if isinstance(action, dict):
            for key in ("commands", "command", "cmd"):
                if key not in action:
                    continue
                text = self._command_text(
                    value=action[key],
                    multiline=key == "commands",
                )
                if text != "":
                    return text

        for key in ("commands", "command", "cmd"):
            if key not in arguments:
                continue
            text = self._command_text(
                value=arguments[key],
                multiline=key == "commands",
            )
            if text != "":
                return text

        if tool == "code_interpreter":
            text = self._command_text(value=arguments.get("code"))
            if text != "":
                return text

        return self._first_nested_text(
            value=arguments,
            keys=("command", "commands", "cmd", "shell_command", "raw_command"),
        )

    def _extract_web_query(self, *, arguments: dict[str, Any] | None) -> str:
        if arguments is None:
            return ""

        queries = arguments.get("queries")
        if isinstance(queries, list):
            values = [item.strip() for item in queries if isinstance(item, str)]
            values = [item for item in values if item != ""]
            if len(values) == 1:
                return values[0]
            if len(values) > 1:
                return ", ".join(values)

        return self._first_nested_text(
            value=arguments,
            keys=("query", "queries", "q"),
        )

    def _extract_apply_patch_text(
        self,
        *,
        arguments: dict[str, Any] | None,
    ) -> str:
        if arguments is None:
            return ""
        return self._first_nested_text(
            value=arguments,
            keys=("patch", "input", "diff"),
        )

    def _is_active_state(self, *, state: str) -> bool:
        return state in _ACTIVE_STATES

    def _is_pending_state(self, *, state: str) -> bool:
        return state == "pending"

    def _exec_search_target(self, *, search_preview: _ExecSearchPreview) -> str:
        if search_preview.path != "":
            return search_preview.path
        if len(search_preview.paths) > 1:
            return f"{len(search_preview.paths)} paths"
        return search_preview.query

    def _exec_search_headline(
        self,
        *,
        status: str,
        search_preview: _ExecSearchPreview,
    ) -> str:
        target = self._exec_search_target(search_preview=search_preview)
        uses_query_as_target = (
            search_preview.path == "" and len(search_preview.paths) == 0
        )
        if status == "failed":
            text = (
                f"Search Failed for {target}"
                if uses_query_as_target
                else f"Search Failed: {target}"
            )
        elif status == "cancelled":
            text = (
                f"Search Cancelled for {target}"
                if uses_query_as_target
                else f"Search Cancelled: {target}"
            )
        elif self._is_active_state(state=status):
            text = (
                f"Searching for {target}"
                if uses_query_as_target
                else f"Searching {target}"
            )
        elif status == "completed":
            text = (
                f"Searched for {target}"
                if uses_query_as_target
                else f"Searched {target}"
            )
        else:
            text = (
                f"Search for {target}" if uses_query_as_target else f"Search {target}"
            )

        return _truncate_text(text=text, limit=280)

    def _exec_search_details(
        self,
        *,
        search_preview: _ExecSearchPreview,
    ) -> tuple[str, ...]:
        details: list[str] = []
        if search_preview.path != "" or len(search_preview.paths) > 1:
            details.append(
                _truncate_text(
                    text=f"Pattern: {search_preview.query}",
                    limit=EXEC_SEARCH_DETAIL_LIMIT,
                )
            )
        if len(search_preview.paths) > 1:
            details.append(
                _truncate_text(
                    text="Paths: " + ", ".join(search_preview.paths),
                    limit=EXEC_SEARCH_DETAIL_LIMIT,
                )
            )
        return tuple(details)

    def _apply_patch_path(self, *, patch: str) -> str:
        for pattern in _APPLY_PATCH_PATH_RES:
            match = pattern.search(patch)
            if match is None:
                continue
            path = match.group("path").strip()
            if path != "":
                return path
        return ""

    def _apply_patch_headline(
        self,
        *,
        status: str,
        path: str,
    ) -> str:
        if path != "":
            if self._is_pending_state(state=status):
                return f"Preparing to edit {path}"
            if status == "failed":
                return f"Patch Failed: {path}"
            if status == "cancelled":
                return f"Patch Cancelled: {path}"
            if self._is_active_state(state=status):
                return f"Editing {path}"
            return f"Edited {path}"

        if self._is_pending_state(state=status):
            return "Preparing Patch"
        if status == "failed":
            return "Patch Failed"
        if status == "cancelled":
            return "Patch Cancelled"
        if self._is_active_state(state=status):
            return "Applying Patch"
        return "Applied Patch"

    def _tool_result_preview(self, *, result: Content | None) -> str:
        del result
        return ""

    def _storage_tool_event(
        self,
        *,
        turn_id: str,
        message_type: str,
        item_id: str,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        state: str,
        data: str,
    ) -> _NormalizedThreadEvent | None:
        def _storage_tool_correlation_key(path: str) -> str:
            tool_key = f"tool:{item_id}"
            if tool_key in self._active_event_elements_by_key:
                return tool_key
            return f"turn.explore:{turn_id}:{path}"

        grep_preview = _storage_grep_preview(
            toolkit=toolkit,
            tool=tool,
            arguments=arguments,
        )
        if grep_preview is not None:
            path = grep_preview.path
            if self._is_pending_state(state=state):
                headline = f"Preparing to search {path}"
            elif self._is_active_state(state=state):
                headline = f"Searching {path}"
            elif state == "failed":
                headline = f"Attempted to search file {path}"
            elif state == "cancelled":
                headline = f"Cancelled searching file {path}"
            else:
                headline = f"Searched {path}"
            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind="exec",
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=_truncate_text(text=headline, limit=280),
                headline=headline,
                details=self._exec_search_details(search_preview=grep_preview),
                path=path,
                data=data,
                correlation_key=_storage_tool_correlation_key(path),
                retain_correlation=True,
            )

        display = _storage_tool_call_display(
            toolkit=toolkit,
            tool=tool,
            arguments=arguments,
        )
        if display is None:
            return None

        if isinstance(display, _StorageReadToolCallDisplay):
            path = display.path
            if self._is_pending_state(state=state):
                headline = f"Preparing to read {path}"
            elif self._is_active_state(state=state):
                headline = f"Reading {path}"
            elif state == "failed":
                headline = f"Attempted to read file {path}"
            elif state == "cancelled":
                headline = f"Cancelled reading file {path}"
            else:
                headline = f"Read {path}"
            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind="exec",
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=_truncate_text(text=headline, limit=280),
                headline=headline,
                path=path,
                data=data,
                correlation_key=_storage_tool_correlation_key(path),
                retain_correlation=True,
            )

        if not isinstance(display, _StorageWriteToolCallDisplay):
            return None

        path = display.path
        if self._is_pending_state(state=state):
            headline = f"Preparing to write {path}"
        elif self._is_active_state(state=state):
            headline = f"Writing {path}"
        elif state == "failed":
            headline = f"Attempted to write file {path}"
        elif state == "cancelled":
            headline = f"Cancelled writing file {path}"
        else:
            headline = f"Wrote {path}"
        return _NormalizedThreadEvent(
            source="agent",
            name=message_type,
            kind="file",
            state=state,
            method=message_type,
            turn_id=turn_id,
            item_id=item_id,
            item_type="tool_call",
            summary=_truncate_text(text=headline, limit=280),
            headline=headline,
            path=path,
            data=data,
            correlation_key=f"tool:{item_id}",
        )

    def _tool_event_kind(self, *, toolkit: str, tool: str) -> str:
        del toolkit
        normalized_tool = tool.strip().lower()
        if normalized_tool == "web_search":
            return "web"
        if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
            return "exec"
        if normalized_tool == "apply_patch":
            return "diff"
        if normalized_tool == "image_generation":
            return "image"
        return "tool"

    def _tool_detail_lines(
        self,
        *,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        result: Content | None,
        error: AgentError | None,
    ) -> tuple[str, ...]:
        del toolkit
        del tool
        del arguments
        del result
        del error
        return ()

    def _tool_event(
        self,
        *,
        turn_id: str,
        message_type: str,
        item_id: str,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        state: str,
        result: Content | None,
        error: AgentError | None,
        data: str,
    ) -> _NormalizedThreadEvent:
        new_thread_display = _new_thread_tool_call_display(
            toolkit=toolkit,
            tool=tool,
            result=result,
        )
        if new_thread_display is not None:
            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind="thread",
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=_truncate_text(text=new_thread_display.name, limit=280),
                headline=new_thread_display.name,
                path=new_thread_display.path,
                data=data,
                correlation_key=f"tool:{item_id}",
            )

        storage_event = self._storage_tool_event(
            turn_id=turn_id,
            message_type=message_type,
            item_id=item_id,
            toolkit=toolkit,
            tool=tool,
            arguments=arguments,
            state=state,
            data=data,
        )
        if storage_event is not None:
            return storage_event

        kind = self._tool_event_kind(toolkit=toolkit, tool=tool)
        correlation_key = f"tool:{item_id}"
        retain_correlation = False
        path = ""
        preview = ""

        if kind == "exec":
            command = self._extract_tool_command(tool=tool, arguments=arguments)
            shell_analysis = analyze_shell_command(command=command)
            shell_phase = shell_analysis.display.phase_for_state(state=state)

            if shell_analysis.display.event_kind == "file":
                return _NormalizedThreadEvent(
                    source="agent",
                    name=message_type,
                    kind="file",
                    state=state,
                    method=message_type,
                    turn_id=turn_id,
                    item_id=item_id,
                    item_type="tool_call",
                    summary=shell_phase.summary,
                    headline=shell_phase.headline,
                    path=shell_analysis.display.path,
                    preview=shell_analysis.display.preview,
                    data=data,
                    correlation_key=f"tool:{item_id}",
                )

            exploration_path = shell_analysis.display.coalesce_path
            if exploration_path != "":
                correlation_key = f"turn.explore:{turn_id}:{exploration_path}"
                retain_correlation = True

            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind=kind,
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=shell_phase.summary,
                headline=shell_phase.headline,
                details=shell_analysis.display.details,
                preview=shell_analysis.display.preview,
                path="",
                data=data,
                correlation_key=correlation_key,
                retain_correlation=retain_correlation,
            )

        if kind == "web":
            query = self._extract_web_query(arguments=arguments)
            if self._is_pending_state(state=state):
                headline = "Preparing web search"
            elif self._is_active_state(state=state):
                headline = "Searching the web"
            elif state == "failed":
                headline = "Web Search Failed"
            elif state == "cancelled":
                headline = "Web Search Cancelled"
            else:
                headline = "Searched the web"
            details = (query,) if query != "" else ()
            summary = (
                _truncate_text(text=query, limit=280)
                if query != ""
                else _truncate_text(text=headline, limit=280)
            )
            preview = self._tool_result_preview(result=result)
            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind=kind,
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=summary,
                headline=headline,
                details=details,
                preview=preview,
                data=data,
                correlation_key=correlation_key,
            )

        if kind == "diff":
            patch = self._extract_apply_patch_text(arguments=arguments)
            path = self._apply_patch_path(patch=patch)
            headline = self._apply_patch_headline(status=state, path=path)
            preview = (
                _truncate_text(text=patch, limit=DIFF_PREVIEW_LIMIT)
                if patch != ""
                else self._tool_result_preview(result=result)
            )
            details = ()
            if patch == "":
                details = self._tool_detail_lines(
                    toolkit=toolkit,
                    tool=tool,
                    arguments=arguments,
                    result=result,
                    error=error,
                )

            return _NormalizedThreadEvent(
                source="agent",
                name=message_type,
                kind=kind,
                state=state,
                method=message_type,
                turn_id=turn_id,
                item_id=item_id,
                item_type="tool_call",
                summary=_truncate_text(text=headline, limit=280),
                headline=headline,
                details=details,
                preview=preview,
                path=path,
                data=data,
                correlation_key=correlation_key,
            )

        if kind == "image":
            if self._is_pending_state(state=state):
                headline = "Preparing image generation"
            elif self._is_active_state(state=state):
                headline = "Generating image"
            elif state == "failed":
                headline = "Image Generation Failed"
            elif state == "cancelled":
                headline = "Image Generation Cancelled"
            else:
                headline = "Generated image"
        else:
            humanized = _humanize_name(tool)
            if self._is_pending_state(state=state):
                headline = f"Preparing {humanized}"
            elif self._is_active_state(state=state):
                headline = f"Calling {humanized}"
            elif state == "failed":
                headline = f"{humanized} Failed"
            elif state == "cancelled":
                headline = f"{humanized} Cancelled"
            else:
                headline = f"Called {humanized}"

        return _NormalizedThreadEvent(
            source="agent",
            name=message_type,
            kind=kind,
            state=state,
            method=message_type,
            turn_id=turn_id,
            item_id=item_id,
            item_type="tool_call",
            summary=_truncate_text(text=headline, limit=280),
            headline=headline,
            details=self._tool_detail_lines(
                toolkit=toolkit,
                tool=tool,
                arguments=arguments,
                result=result,
                error=error,
            ),
            preview=self._tool_result_preview(result=result),
            data=data,
            correlation_key=correlation_key,
        )

    def _normalized_event_from_message(
        self,
        message: AgentMessage,
    ) -> _NormalizedThreadEvent | None:
        if isinstance(message, ThreadCleared):
            self._clear_active_turn_state()
            self._active_event_elements_by_key.clear()
            self._active_event_elements_by_item_id.clear()
            self._clear_thread_status_nowait()
            return None

        data = self._message_event_data(message=message)

        if isinstance(message, TurnStarted):
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state="in_progress",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.turn_id,
                item_type="turn",
                summary="Thinking",
                headline="Thinking",
                data=data,
                correlation_key=f"turn:{message.turn_id}",
            )

        if isinstance(message, TurnEnded):
            details: tuple[str, ...] = ()
            summary = "Turn completed"
            if message.error is not None:
                details = (self._error_details(message.error),)
                if message.error.code == "cancelled":
                    summary = "Turn cancelled"
                else:
                    summary = "Turn failed"
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state=_terminal_state_from_error(message.error),
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.turn_id,
                item_type="turn",
                summary=summary,
                headline=summary,
                details=details,
                data=data,
                correlation_key=f"turn:{message.turn_id}",
            )

        if isinstance(message, TurnInterrupted):
            self._active_tool_calls_by_item_id.clear()
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state="cancelled",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.turn_id,
                item_type="turn",
                summary="Turn interrupted",
                headline="Turn interrupted",
                data=data,
            )

        if isinstance(message, TurnSteerAccepted):
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state="accepted",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.source_message_id,
                item_type="turn_steer",
                summary="Queued turn steering",
                headline="Queued turn steering",
                data=data,
                correlation_key=f"steer:{message.source_message_id}",
            )

        if isinstance(message, TurnSteered):
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state="completed",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.source_message_id,
                item_type="turn_steer",
                summary="Applied turn steering",
                headline="Applied turn steering",
                data=data,
                correlation_key=f"steer:{message.source_message_id}",
            )

        if isinstance(message, TurnSteerRejected):
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="turn",
                state="failed",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.source_message_id,
                item_type="turn_steer",
                summary="Rejected turn steering",
                headline="Rejected turn steering",
                details=(self._error_details(message.error),),
                data=data,
                correlation_key=f"steer:{message.source_message_id}",
            )

        if isinstance(message, AgentToolCallApprovalRequested):
            details = self._tool_detail_lines(
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                result=None,
                error=None,
            )
            return _NormalizedThreadEvent(
                source="agent",
                name=message.type,
                kind="approval",
                state="pending",
                method=message.type,
                turn_id=message.turn_id,
                item_id=message.item_id,
                item_type="tool_call",
                summary="Tool approval requested",
                headline="Waiting for approval",
                details=details,
                data=data,
                correlation_key=f"approval:{message.item_id}",
            )

        if isinstance(message, AgentToolCallPending):
            self._active_tool_calls_by_item_id[message.item_id] = _ActiveToolCall(
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
            )
            return self._tool_event(
                turn_id=message.turn_id,
                message_type=message.type,
                item_id=message.item_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                state="pending",
                result=None,
                error=None,
                data=data,
            )

        if isinstance(message, AgentToolCallInProgress):
            self._active_tool_calls_by_item_id[message.item_id] = _ActiveToolCall(
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
            )
            return self._tool_event(
                turn_id=message.turn_id,
                message_type=message.type,
                item_id=message.item_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                state="in_progress",
                result=None,
                error=None,
                data=data,
            )

        if isinstance(message, AgentToolCallStarted):
            self._active_tool_calls_by_item_id[message.item_id] = _ActiveToolCall(
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
            )
            return self._tool_event(
                turn_id=message.turn_id,
                message_type=message.type,
                item_id=message.item_id,
                toolkit=message.toolkit,
                tool=message.tool,
                arguments=message.arguments,
                state="in_progress",
                result=None,
                error=None,
                data=data,
            )

        if isinstance(message, AgentToolCallEnded):
            active_tool_call = self._active_tool_calls_by_item_id.get(message.item_id)
            if active_tool_call is None:
                active_tool_call = _ActiveToolCall(
                    toolkit="tool",
                    tool="tool",
                    arguments=None,
                )
            return self._tool_event(
                turn_id=message.turn_id,
                message_type=message.type,
                item_id=message.item_id,
                toolkit=active_tool_call.toolkit,
                tool=active_tool_call.tool,
                arguments=active_tool_call.arguments,
                state=_terminal_state_from_error(message.error),
                result=message.result,
                error=message.error,
                data=data,
            )

        return None

    def _upsert_event(
        self, *, messages: Element, event: _NormalizedThreadEvent
    ) -> None:
        key = event.correlation_key
        state = event.state
        now = _now_iso()
        item_id = event.item_id.strip()

        event_element = (
            self._active_event_elements_by_item_id.get(item_id)
            if item_id != ""
            else None
        )
        if event_element is None:
            event_element = (
                self._active_event_elements_by_key.get(key) if key is not None else None
            )
        if event_element is None:
            attributes = {
                "id": str(uuid.uuid4()),
                "source": event.source,
                "name": event.name,
                "kind": event.kind,
                "state": state,
                "method": event.method,
                "item_id": event.item_id,
                "item_type": event.item_type,
                "path": event.path,
                "summary": event.summary,
                "headline": event.headline,
                "details": self._details_text(details=event.details),
                "preview": event.preview,
                "created_at": now,
                "updated_at": now,
            }
            if event.turn_id != "":
                attributes["turn_id"] = event.turn_id
            if event.data != "":
                attributes["data"] = event.data
            event_element = messages.append_child(
                tag_name="event", attributes=attributes
            )
        else:
            event_element.set_attribute("source", event.source)
            event_element.set_attribute("name", event.name)
            event_element.set_attribute("kind", event.kind)
            event_element.set_attribute("state", state)
            event_element.set_attribute("method", event.method)
            if event.turn_id != "":
                event_element.set_attribute("turn_id", event.turn_id)
            event_element.set_attribute("item_id", event.item_id)
            event_element.set_attribute("item_type", event.item_type)
            event_element.set_attribute("path", event.path)
            event_element.set_attribute("summary", event.summary)
            event_element.set_attribute("headline", event.headline)
            details_text = self._details_text(details=event.details)
            if details_text != "" or event_element.get_attribute("details") in (
                None,
                "",
            ):
                event_element.set_attribute("details", details_text)
            if event.preview != "" or event_element.get_attribute("preview") in (
                None,
                "",
            ):
                event_element.set_attribute("preview", event.preview)
            event_element.set_attribute("updated_at", now)
            if event.data != "":
                event_element.set_attribute("data", event.data)

        self._clear_active_event_element_mappings(event_element=event_element)

        if state in _ACTIVE_STATES or event.retain_correlation:
            if key is not None:
                self._active_event_elements_by_key[key] = event_element
            if item_id != "":
                self._active_event_elements_by_item_id[item_id] = event_element
        for drop_key in event.drop_correlation_keys:
            self._active_event_elements_by_key.pop(drop_key, None)

    def _event_element_by_item_id(
        self, *, messages: Element, item_id: str
    ) -> Element | None:
        normalized_item_id = item_id.strip()
        if normalized_item_id == "":
            return None

        active_event = self._active_event_elements_by_item_id.get(normalized_item_id)
        if active_event is not None:
            return active_event

        for child in reversed(messages.get_children()):
            if not isinstance(child, Element) or child.tag_name != "event":
                continue
            if self._attribute_as_str(child, "item_id") == normalized_item_id:
                return child

        return None

    def _append_event_logs(
        self,
        *,
        messages: Element,
        item_id: str,
        lines: list[AgentToolCallLogLine],
    ) -> None:
        if len(lines) == 0:
            return

        event_element = self._event_element_by_item_id(
            messages=messages, item_id=item_id
        )
        if event_element is None:
            return

        for line in lines:
            event_element.append_child(
                "log",
                {
                    "source": line.source,
                    "text": line.text,
                    "created_at": _now_iso(),
                },
            )

        log_elements = event_element.get_children_by_tag_name("log")
        overflow = len(log_elements) - EVENT_LOG_LINE_LIMIT
        if overflow > 0:
            for log_element in log_elements[:overflow]:
                log_element.delete()

        event_element.set_attribute("updated_at", _now_iso())

    def _clear_active_event_element_mappings(self, *, event_element: Element) -> None:
        for key, mapped_element in list(self._active_event_elements_by_key.items()):
            if mapped_element is event_element:
                self._active_event_elements_by_key.pop(key, None)

        for item_id, mapped_element in list(
            self._active_event_elements_by_item_id.items()
        ):
            if mapped_element is event_element:
                self._active_event_elements_by_item_id.pop(item_id, None)

    @staticmethod
    def _error_details(error: AgentError) -> str:
        if error.code is None or error.code == "":
            return error.message
        return f"{error.code}: {error.message}"

    def _thread_status_attribute_name(self) -> str:
        return f"thread.status.{self.path}"

    def _thread_status_text_attribute_name(self) -> str:
        return f"thread.status.text.{self.path}"

    def _thread_status_mode_attribute_name(self) -> str:
        return f"thread.status.mode.{self.path}"

    def _thread_status_started_at_attribute_name(self) -> str:
        return f"thread.status.started_at.{self.path}"

    def _thread_status_pending_messages_attribute_name(self) -> str:
        return f"thread.status.pending_messages.{self.path}"

    def _thread_status_pending_item_id_attribute_name(self) -> str:
        return f"thread.status.pending_item_id.{self.path}"

    def processing_thread_status_mode(self) -> ThreadStatusMode:
        return "steerable"

    async def _set_local_participant_attribute(
        self,
        attribute_name: str,
        value: str | None,
    ) -> None:
        try:
            await self._room.local_participant.set_attribute(attribute_name, value)
        except ChanClosed:
            logger.debug(
                "room channel closed while setting thread status '%s'",
                attribute_name,
            )

    def _pending_messages_payload(self) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if self._thread_status_turn_id_value is not None:
            payload["turn_id"] = self._thread_status_turn_id_value
        if len(self._thread_status_pending_messages_value) > 0:
            payload["messages"] = self._thread_status_pending_messages_value
        if len(payload) == 0:
            return None
        return payload

    async def _write_pending_messages_attribute(self) -> None:
        serialized_pending_messages: str | None = None
        pending_messages_payload = self._pending_messages_payload()
        if pending_messages_payload is not None:
            serialized_pending_messages = json.dumps(
                pending_messages_payload,
                ensure_ascii=False,
                sort_keys=True,
            )

        await self._set_local_participant_attribute(
            self._thread_status_pending_messages_attribute_name(),
            serialized_pending_messages,
        )

    async def _write_thread_status_attributes(self) -> None:
        await self._set_local_participant_attribute(
            self._thread_status_attribute_name(),
            self._thread_status_value,
        )
        await self._set_local_participant_attribute(
            self._thread_status_text_attribute_name(),
            self._thread_status_value,
        )
        await self._set_local_participant_attribute(
            self._thread_status_mode_attribute_name(),
            self._thread_status_mode_value,
        )
        await self._set_local_participant_attribute(
            self._thread_status_started_at_attribute_name(),
            self._thread_status_started_at_value,
        )
        await self._set_local_participant_attribute(
            self._thread_status_pending_item_id_attribute_name(),
            self._thread_status_pending_item_id_value,
        )

    async def set_thread_turn_id(self, *, turn_id: str | None) -> None:
        async with self._thread_status_lock:
            if self._thread_status_turn_id_value == turn_id:
                return

            self._thread_status_turn_id_value = turn_id
            await self._write_pending_messages_attribute()

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, Any]],
    ) -> None:
        normalized_pending_messages = json.loads(
            json.dumps(pending_messages, ensure_ascii=False)
        )

        async with self._thread_status_lock:
            if (
                self._thread_status_pending_messages_value
                == normalized_pending_messages
            ):
                return

            self._thread_status_pending_messages_value = normalized_pending_messages
            await self._write_pending_messages_attribute()

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode: ThreadStatusMode | None = None,
    ) -> None:
        if status is None or status.strip() == "":
            self._thread_status_value = None
            self._thread_status_mode_value = None
            self._thread_status_started_at_value = None
            self._thread_status_pending_item_id_value = None
            await self._write_thread_status_attributes()
            return

        normalized_status = status.strip()
        normalized_mode = (
            mode if mode is not None else self.processing_thread_status_mode()
        )
        started_at = self._thread_status_started_at_value
        if (
            started_at is None
            or self._thread_status_value != normalized_status
            or self._thread_status_mode_value != normalized_mode
        ):
            started_at = _now_iso()

        if (
            self._thread_status_value == normalized_status
            and self._thread_status_mode_value == normalized_mode
            and self._thread_status_started_at_value == started_at
        ):
            return

        self._thread_status_value = normalized_status
        self._thread_status_mode_value = normalized_mode
        self._thread_status_started_at_value = started_at
        await self._write_thread_status_attributes()

    def _next_thread_status_generation(self) -> int:
        self._thread_status_generation += 1
        return self._thread_status_generation

    async def _apply_thread_status(
        self,
        *,
        status: str | None,
        generation: int | None = None,
    ) -> None:
        async with self._thread_status_lock:
            if generation is not None and generation != self._thread_status_generation:
                return

            await self.set_thread_status(status=status)

    def _set_thread_status_nowait(self, *, status: str | None) -> None:
        generation = self._next_thread_status_generation()

        async def run() -> None:
            try:
                await self._apply_thread_status(
                    status=status,
                    generation=generation,
                )
            except Exception:
                logger.exception("unable to set thread status for %s", self.path)

        asyncio.create_task(run())

    async def clear_thread_status(self) -> None:
        self._thread_status_key = None
        generation = self._next_thread_status_generation()
        await self._apply_thread_status(status=None, generation=generation)

    def _clear_thread_status_nowait(self) -> None:
        self._thread_status_key = None
        self._set_thread_status_nowait(status=None)

    async def _update_thread_status_from_event(
        self,
        *,
        event: _NormalizedThreadEvent,
    ) -> None:
        key, state, text = self._status_event_details(event=event)
        if state is None:
            return

        if state in _ACTIVE_STATES:
            if text is None:
                return
            if key is not None:
                self._thread_status_key = key
            self._thread_status_pending_item_id_value = (
                event.item_id.strip() if event.item_id.strip() != "" else None
            )
            await self.set_thread_status(status=text)
            return

        fallback_status = "Thinking"

        if key is not None:
            tracked = self._thread_status_key
            if tracked is not None and tracked == key:
                self._thread_status_key = None
                self._thread_status_pending_item_id_value = None
                await self.set_thread_status(status=fallback_status)
            return

        if state in _TERMINAL_STATES:
            self._thread_status_pending_item_id_value = None
            await self.set_thread_status(status=fallback_status)
