from __future__ import annotations

import logging
import posixpath
import re
import uuid
from datetime import datetime, timezone
from typing import Any, TypeVar
from urllib.parse import urlparse

from meshagent.api import (
    Element,
    MeshDocument,
    Participant,
    RoomClient,
    RoomException,
    RoomMessage,
)
from meshagent.api.messaging import JsonContent
from pydantic import BaseModel, ValidationError
from meshagent.tools import (
    FunctionTool,
    RemoteToolkit,
    ToolContext,
    Toolkit,
    ToolkitBuilder,
    tool,
)
from meshagent.tools.strict_schema import ensure_strict_json_schema

from .messages import (
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentFileContent,
    AgentMessage,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallApprovalRequested,
    ApproveAgentToolCall,
    ClearThread,
    RejectAgentToolCall,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnStart,
    TurnStarted,
    TurnSteer,
)
from .process import Channel, Message
from .thread_schema import thread_list_schema
from .toolkit_schema import build_tools_property_schema

logger = logging.getLogger("legacy-chat-channel")
_MessageT = TypeVar("_MessageT", bound=AgentMessage)


class _ChatAttachmentPayload(BaseModel):
    path: str


class _ChatMessagePayload(BaseModel):
    path: str
    text: str = ""
    attachments: list[_ChatAttachmentPayload] | None = None
    tools: list[dict[str, Any]] | None = None
    model: str | None = None


class _PathMessagePayload(BaseModel):
    path: str


class _ApprovalDecisionPayload(BaseModel):
    path: str
    approval_id: str


class LegacyChatChannel(Channel):
    def __init__(
        self,
        *,
        room: RoomClient,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        empty_state_title: str = "How can I help you?",
        toolkit_builders: list[ToolkitBuilder] | None = None,
    ) -> None:
        super().__init__()
        self._room = room
        self._threading_mode = self._normalize_threading_mode(
            threading_mode=threading_mode
        )
        self._thread_dir = self._normalize_thread_dir(thread_dir=thread_dir)
        self._empty_state_title = empty_state_title
        self._toolkit_builders = list(toolkit_builders or [])
        self._active_turn_ids_by_thread: dict[str, str] = {}
        self._pending_approval_turn_ids_by_thread: dict[str, dict[str, str]] = {}
        self._open_participant_ids_by_thread: dict[str, set[str]] = {}
        self._active_text_by_thread: dict[str, dict[str, str]] = {}
        self._thread_list_document: MeshDocument | None = None
        self._thread_list_path: str | None = None

    @property
    def room(self) -> RoomClient:
        return self._room

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_EVENT_TEXT_CONTENT_STARTED,
            AGENT_EVENT_TEXT_CONTENT_DELTA,
            AGENT_EVENT_TEXT_CONTENT_ENDED,
            AGENT_EVENT_THREAD_CLEARED,
            AGENT_EVENT_TURN_STARTED,
            AGENT_EVENT_TURN_ENDED,
            AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
        }

    async def on_start(self) -> None:
        self._room.messaging.on("message", self._on_room_message)
        if self._threading_mode is not None:
            await self._room.local_participant.set_attribute(
                "meshagent.chatbot.threading",
                self._threading_mode,
            )
        thread_list_path = self._thread_list_index_path()
        if thread_list_path is not None:
            await self._room.local_participant.set_attribute(
                "meshagent.chatbot.thread-list",
                thread_list_path,
            )
        await self._room.local_participant.set_attribute(
            "empty_state_title",
            self._empty_state_title,
        )
        await self._open_thread_list_document()
        if not self._room.messaging.is_enabled:
            await self._room.messaging.enable()

    async def on_stop(self) -> None:
        self._room.messaging.off("message", self._on_room_message)
        await self._close_thread_list_document()
        self._active_turn_ids_by_thread.clear()
        self._pending_approval_turn_ids_by_thread.clear()
        self._open_participant_ids_by_thread.clear()
        self._active_text_by_thread.clear()

    async def on_message(self, message: Message) -> None:
        data = message.data
        if data.type == AGENT_EVENT_TEXT_CONTENT_STARTED:
            text_started = self._coerce_message(
                data=data, model=AgentTextContentStarted
            )
            active_text = self._active_text_by_thread.setdefault(
                text_started.thread_id,
                {},
            )
            active_text[text_started.item_id] = ""
            return

        if data.type == AGENT_EVENT_TEXT_CONTENT_DELTA:
            text_delta = self._coerce_message(data=data, model=AgentTextContentDelta)
            active_text = self._active_text_by_thread.setdefault(
                text_delta.thread_id,
                {},
            )
            current_text = active_text.get(text_delta.item_id, "")
            active_text[text_delta.item_id] = current_text + text_delta.text
            return

        if data.type == AGENT_EVENT_TEXT_CONTENT_ENDED:
            text_ended = self._coerce_message(data=data, model=AgentTextContentEnded)
            active_text = self._active_text_by_thread.get(text_ended.thread_id, {})
            text = active_text.pop(text_ended.item_id, "")
            if len(active_text) == 0:
                self._active_text_by_thread.pop(text_ended.thread_id, None)
            if text.strip() != "":
                self._send_chat_to_open_participants(
                    thread_id=text_ended.thread_id,
                    text=text,
                )
            return

        if data.type == AGENT_EVENT_TURN_STARTED:
            turn_started = self._coerce_message(data=data, model=TurnStarted)
            self._active_turn_ids_by_thread[turn_started.thread_id] = (
                turn_started.turn_id
            )
            return

        if data.type == AGENT_EVENT_THREAD_CLEARED:
            thread_cleared = self._coerce_message(data=data, model=ThreadCleared)
            self._clear_tracked_thread_state(thread_id=thread_cleared.thread_id)
            self._send_cleared_to_open_participants(thread_id=thread_cleared.thread_id)
            return

        if data.type == AGENT_EVENT_TURN_ENDED:
            turn_ended = self._coerce_message(data=data, model=TurnEnded)
            tracked_turn_id = self._active_turn_ids_by_thread.get(turn_ended.thread_id)
            if tracked_turn_id == turn_ended.turn_id:
                self._active_turn_ids_by_thread.pop(turn_ended.thread_id, None)
            self._active_text_by_thread.pop(turn_ended.thread_id, None)

            pending_approvals = self._pending_approval_turn_ids_by_thread.get(
                turn_ended.thread_id
            )
            if pending_approvals is None:
                return

            remaining_approvals = {
                approval_id: pending_turn_id
                for approval_id, pending_turn_id in pending_approvals.items()
                if pending_turn_id != turn_ended.turn_id
            }
            if len(remaining_approvals) == 0:
                self._pending_approval_turn_ids_by_thread.pop(
                    turn_ended.thread_id,
                    None,
                )
            else:
                self._pending_approval_turn_ids_by_thread[turn_ended.thread_id] = (
                    remaining_approvals
                )
            return

        if data.type == AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED:
            approval = self._coerce_message(
                data=data,
                model=AgentToolCallApprovalRequested,
            )
            pending_approvals = self._pending_approval_turn_ids_by_thread.setdefault(
                approval.thread_id,
                {},
            )
            pending_approvals[approval.item_id] = approval.turn_id

    def _on_room_message(self, *, message: RoomMessage) -> None:
        sender = self._room.messaging.get_participant(message.from_participant_id)
        if sender is None:
            logger.warning(
                "ignoring chat message from unknown participant %s",
                message.from_participant_id,
            )
            return

        if self._handle_room_control_message(message=message, sender=sender):
            return

        thread_id = self._thread_id_from_room_message(message=message)
        if thread_id is not None:
            self._register_open_participant(
                thread_id=thread_id,
                participant_id=sender.id,
            )
            self._touch_thread_in_index(path=thread_id)

        try:
            agent_message = self._agent_message_from_room_message(message=message)
        except ValidationError:
            logger.exception(
                "unable to translate chat room message of type %s",
                message.type,
            )
            return

        if agent_message is None:
            return

        self.emit(sender=sender, payload=agent_message)

    @staticmethod
    def _coerce_message(
        *,
        data: AgentMessage,
        model: type[_MessageT],
    ) -> _MessageT:
        if isinstance(data, model):
            return data
        return model.model_validate(data.model_dump(mode="python"))

    @staticmethod
    def _normalize_attachment_url(*, path: str) -> str | None:
        normalized_path = path.strip()
        if normalized_path == "":
            return None

        parsed = urlparse(normalized_path)
        if parsed.scheme != "":
            return normalized_path

        room_path = normalized_path.lstrip("/")
        if room_path == "":
            return None

        return f"room:///{room_path}"

    @staticmethod
    def _content_from_chat_message(
        *,
        payload: _ChatMessagePayload,
    ) -> list[AgentTextContent | AgentFileContent]:
        content: list[AgentTextContent | AgentFileContent] = []

        if payload.text.strip() != "":
            content.append(
                AgentTextContent(
                    type="text",
                    text=payload.text,
                )
            )

        if payload.attachments is not None:
            for attachment in payload.attachments:
                normalized_url = LegacyChatChannel._normalize_attachment_url(
                    path=attachment.path
                )
                if normalized_url is None:
                    continue
                content.append(
                    AgentFileContent(
                        type="file",
                        url=normalized_url,
                    )
                )

        return content

    def _active_turn_id(self, *, thread_id: str) -> str | None:
        return self._active_turn_ids_by_thread.get(thread_id)

    @staticmethod
    def _thread_id_from_room_message(*, message: RoomMessage) -> str | None:
        payload = message.message
        if not isinstance(payload, dict):
            return None

        path = payload.get("path")
        if not isinstance(path, str):
            return None

        normalized_path = path.strip()
        if normalized_path == "":
            return None

        return normalized_path

    @staticmethod
    def _normalize_threading_mode(*, threading_mode: str | None) -> str | None:
        if threading_mode is None:
            return None

        normalized = threading_mode.strip()
        if normalized == "" or normalized == "none":
            return None
        return normalized

    @staticmethod
    def _normalize_thread_dir(*, thread_dir: str | None) -> str | None:
        if thread_dir is None:
            return None

        normalized = thread_dir.strip().rstrip("/")
        if normalized == "":
            raise ValueError("thread_dir must not be empty")
        return normalized

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _thread_list_index_path(self) -> str | None:
        thread_dir = self._thread_list_dir()
        if thread_dir is None:
            return None
        return posixpath.join(thread_dir, "index.threadl")

    def _thread_list_entry_name_for_path(self, *, path: str) -> str:
        filename = posixpath.basename(path.strip())
        if filename.endswith(".thread"):
            filename = filename[: -len(".thread")]
        raw_name = filename.strip()
        if raw_name != "":
            try:
                parsed_uuid = uuid.UUID(raw_name)
            except ValueError:
                parsed_uuid = None
            if parsed_uuid is not None and str(parsed_uuid) == raw_name.lower():
                return "New Chat"

        normalized = re.sub(r"[-_/]+", " ", filename)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        if normalized == "":
            return "New Chat"
        if normalized == normalized.lower() or normalized == normalized.upper():
            normalized = normalized.title()
        return normalized[:64].strip() or "New Chat"

    def _default_thread_dir(self) -> str:
        local_name = "chat"
        participant_name = self._room.local_participant.get_attribute("name")
        if isinstance(participant_name, str) and participant_name.strip() != "":
            local_name = participant_name.strip()

        return self._normalize_thread_dir(
            thread_dir=posixpath.join(".threads", local_name)
        )

    def _thread_list_dir(self) -> str | None:
        if self._thread_dir is not None:
            return self._thread_dir
        if self._threading_mode == "default-new":
            return self._default_thread_dir()
        return None

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir
        return self._default_thread_dir()

    @staticmethod
    def _sanitize_thread_name(*, value: str) -> str:
        normalized = value.strip()
        if normalized.endswith(".thread"):
            normalized = normalized[: -len(".thread")]

        normalized = re.sub(r"[-_/]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        normalized = re.sub(r"[^A-Za-z0-9 .,!?':()&]+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        if normalized == "":
            return "New Chat"
        if normalized == normalized.lower() or normalized == normalized.upper():
            normalized = normalized.title()
        return normalized[:64].strip() or "New Chat"

    def _fallback_thread_name(self, *, text: str) -> str:
        normalized_text = text.strip()
        if normalized_text != "":
            return self._sanitize_thread_name(value=normalized_text)
        return "New Chat"

    @staticmethod
    def _thread_path_for_name(*, thread_name: str, thread_dir: str) -> str:
        return posixpath.join(thread_dir, f"{thread_name}.thread")

    def _find_thread_list_entry(self, *, path: str) -> Element | None:
        if self._thread_list_document is None:
            return None

        for child in self._thread_list_document.root.get_children():
            if child.tag_name != "thread":
                continue
            if child.get_attribute("path") == path:
                return child

        return None

    def _is_index_managed_path(self, *, path: str) -> bool:
        thread_dir = self._thread_list_dir()
        if thread_dir is None:
            return False

        normalized_path = path.strip().strip("/")
        normalized_dir = thread_dir.strip().strip("/")
        if normalized_path == "" or normalized_dir == "":
            return False
        return normalized_path == normalized_dir or normalized_path.startswith(
            f"{normalized_dir}/"
        )

    def _upsert_thread_list_entry(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> None:
        if self._thread_list_document is None or not self._is_index_managed_path(
            path=path
        ):
            return

        now = self._utc_now_iso()
        provided_name = name.strip() if isinstance(name, str) else ""
        entry = self._find_thread_list_entry(path=path)
        if entry is None:
            resolved_created_at = (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
            resolved_modified_at = (
                modified_at.strip()
                if isinstance(modified_at, str) and modified_at.strip() != ""
                else resolved_created_at
            )
            self._thread_list_document.root.append_child(
                tag_name="thread",
                attributes={
                    "name": (
                        provided_name
                        if provided_name != ""
                        else self._thread_list_entry_name_for_path(path=path)
                    ),
                    "path": path,
                    "created_at": resolved_created_at,
                    "modified_at": resolved_modified_at,
                },
            )
            return

        entry.set_attribute("path", path)
        if provided_name != "":
            entry.set_attribute("name", provided_name)
        else:
            existing_name = entry.get_attribute("name")
            if isinstance(existing_name, str) and existing_name.strip() != "":
                pass
            else:
                entry.set_attribute(
                    "name", self._thread_list_entry_name_for_path(path=path)
                )

        existing_created_at = entry.get_attribute("created_at")
        resolved_created_at = (
            existing_created_at.strip()
            if isinstance(existing_created_at, str)
            and existing_created_at.strip() != ""
            else (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
        )
        entry.set_attribute("created_at", resolved_created_at)
        resolved_modified_at = (
            modified_at.strip()
            if isinstance(modified_at, str) and modified_at.strip() != ""
            else now
        )
        entry.set_attribute("modified_at", resolved_modified_at)

    def _record_new_thread_in_index(
        self,
        *,
        path: str,
        name: str | None = None,
    ) -> None:
        now = self._utc_now_iso()
        self._upsert_thread_list_entry(
            path=path,
            name=name,
            created_at=now,
            modified_at=now,
        )

    def _thread_list_entries(self) -> list[Element]:
        if self._thread_list_dir() is None or self._thread_list_document is None:
            return []

        return [
            child
            for child in self._thread_list_document.root.get_children()
            if child.tag_name == "thread"
        ]

    @staticmethod
    def _parse_iso_datetime(*, value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if raw == "":
            return None

        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _thread_sort_datetime(self, *, entry: Element) -> datetime:
        modified_at = self._parse_iso_datetime(value=entry.get_attribute("modified_at"))
        if modified_at is not None:
            return modified_at
        created_at = self._parse_iso_datetime(value=entry.get_attribute("created_at"))
        if created_at is not None:
            return created_at
        return datetime.min.replace(tzinfo=timezone.utc)

    def _sorted_thread_list_entries(self) -> list[Element]:
        return sorted(
            self._thread_list_entries(),
            key=lambda entry: self._thread_sort_datetime(entry=entry),
            reverse=True,
        )

    @staticmethod
    def _thread_list_slice(
        *,
        entries: list[Element],
        limit: int,
        offset: int,
    ) -> list[Element]:
        normalized_offset = max(0, int(offset))
        normalized_limit = max(1, min(200, int(limit)))
        return entries[normalized_offset : normalized_offset + normalized_limit]

    async def _next_available_thread_path(self, *, base_path: str) -> str:
        try:
            exists = await self._room.storage.exists(path=base_path)
        except Exception:
            return base_path

        if not exists:
            return base_path

        thread_dir, filename = posixpath.split(base_path)
        if filename.endswith(".thread"):
            base_name = filename[: -len(".thread")]
        else:
            base_name = filename

        for index in range(2, 1000):
            candidate = posixpath.join(thread_dir, f"{base_name} {index}.thread")
            try:
                if not await self._room.storage.exists(path=candidate):
                    return candidate
            except Exception:
                return candidate

        return posixpath.join(thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}.thread")

    def _build_thread_list_tools(self) -> list[FunctionTool]:
        if self._thread_list_dir() is None:
            return []

        read_file_hint = (
            "Use read_file with a thread path to read that thread's contents."
        )
        outer = self

        def to_json_entry(entry: Element) -> dict[str, str]:
            return {
                "name": str(entry.get_attribute("name") or ""),
                "path": str(entry.get_attribute("path") or ""),
                "modified_at": str(entry.get_attribute("modified_at") or ""),
                "created_at": str(entry.get_attribute("created_at") or ""),
            }

        @tool(
            name="list_threads",
            description="lists recent threads sorted by last modified date (newest first). Use read_file with a thread path to read that thread's contents.",
        )
        def list_threads(*, limit: int = 20, offset: int = 0) -> JsonContent:
            normalized_offset = max(0, int(offset))
            normalized_limit = max(1, min(200, int(limit)))

            entries = outer._sorted_thread_list_entries()
            if len(entries) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": 0,
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "no threads were found in the thread list",
                        "read_file_hint": read_file_hint,
                    }
                )

            selected = outer._thread_list_slice(
                entries=entries,
                limit=limit,
                offset=offset,
            )
            if len(selected) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": len(entries),
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "no threads were found for the requested limit/offset",
                        "read_file_hint": read_file_hint,
                    }
                )

            return JsonContent(
                json={
                    "threads": [to_json_entry(entry) for entry in selected],
                    "total": len(entries),
                    "offset": normalized_offset,
                    "limit": normalized_limit,
                    "sort": "modified_at_desc",
                    "read_file_hint": read_file_hint,
                }
            )

        @tool(
            name="grep_thread_list",
            description="searches the thread list for matching thread names and paths. Use read_file with a thread path to read that thread's contents.",
        )
        def grep_thread_list(*, pattern: str, ignore_case: bool = True) -> JsonContent:
            needle = pattern.strip()
            if needle == "":
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "message": "pattern is required",
                        "read_file_hint": read_file_hint,
                    }
                )

            flags = re.IGNORECASE if ignore_case else 0
            try:
                matcher = re.compile(needle, flags)
            except re.error as ex:
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "error": "invalid_regex_pattern",
                        "message": f"invalid regex pattern: {ex}",
                        "read_file_hint": read_file_hint,
                    }
                )

            matches: list[dict[str, str]] = []
            for entry in outer._sorted_thread_list_entries():
                name = entry.get_attribute("name")
                path = entry.get_attribute("path")
                created_at = entry.get_attribute("created_at")
                modified_at = entry.get_attribute("modified_at")
                haystack = f"{name}\n{path}\n{created_at}\n{modified_at}"
                if matcher.search(haystack) is None:
                    continue
                matches.append(to_json_entry(entry))

            if len(matches) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "message": "no matching threads were found",
                        "read_file_hint": read_file_hint,
                    }
                )

            return JsonContent(
                json={
                    "threads": matches,
                    "total_matches": len(matches),
                    "pattern": needle,
                    "ignore_case": ignore_case,
                    "read_file_hint": read_file_hint,
                }
            )

        return [list_threads, grep_thread_list]

    def _local_participant_name(self) -> str:
        local_name = self._room.local_participant.get_attribute("name")
        if not isinstance(local_name, str) or local_name.strip() == "":
            return "assistant"
        return local_name.strip()

    def _build_new_thread_tool_schema(self) -> dict[str, Any]:
        tools_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["message"],
            "properties": {
                "message": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "attachments": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/ChatAttachment"},
                                },
                                {"type": "null"},
                            ]
                        },
                    },
                }
            },
            "$defs": {
                "ChatAttachment": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string"},
                    },
                }
            },
        }

        message_tools_schema, defs = build_tools_property_schema(
            toolkit_builders=self._toolkit_builders
        )
        if message_tools_schema is not None:
            tools_schema["properties"]["message"]["properties"]["tools"] = (
                message_tools_schema
            )
        else:
            tools_schema["properties"]["message"]["properties"]["tools"] = {
                "anyOf": [
                    {
                        "type": "array",
                        "items": ensure_strict_json_schema({}),
                    },
                    {"type": "null"},
                ],
            }

        for key, value in defs.items():
            tools_schema["$defs"][key] = value
        return ensure_strict_json_schema(tools_schema)

    def _make_new_thread_tool(self) -> FunctionTool:
        local_name = self._local_participant_name()
        tools_schema = self._build_new_thread_tool_schema()

        outer = self

        class NewThreadTool(FunctionTool):
            def __init__(self) -> None:
                super().__init__(
                    name="new_thread",
                    description=f"starts a new thread for {local_name}, posts a message to the thread, and then returns the path and name of the new thread. Since work will continue asynchronously on that thread, an agent should not invoke this if it will check the result of the work, it should generally be invoked as fire and forget.",
                    input_schema=tools_schema,
                    supports_context=True,
                )

            async def execute(
                self,
                context: ToolContext,
                *,
                message: dict[str, Any],
            ) -> JsonContent:
                if outer.supervisor is None:
                    raise RoomException(
                        "chat channel must be attached to a supervisor before using chat.new_thread"
                    )

                text_value = message.get("text")
                text = text_value if isinstance(text_value, str) else ""
                payload = {**message, "text": text}

                path = outer._thread_path_for_name(
                    thread_name=str(uuid.uuid4()),
                    thread_dir=outer._get_thread_dir(),
                )
                path = await outer._next_available_thread_path(base_path=path)
                friendly_name = outer._fallback_thread_name(text=text)
                outer._record_new_thread_in_index(
                    path=path,
                    name=friendly_name,
                )

                chat_message = _ChatMessagePayload.model_validate(
                    {
                        "path": path,
                        "text": text,
                        "attachments": payload.get("attachments"),
                        "tools": payload.get("tools"),
                    }
                )
                outer.emit(
                    sender=context.on_behalf_of or context.caller,
                    payload=TurnStart(
                        type=AGENT_MESSAGE_TURN_START,
                        thread_id=path,
                        content=outer._content_from_chat_message(payload=chat_message),
                        toolkits=chat_message.tools,
                    ),
                )
                return JsonContent(json={"path": path, "name": friendly_name})

        return NewThreadTool()

    def _build_chat_tools(self) -> list[FunctionTool]:
        return [self._make_new_thread_tool(), *self._build_thread_list_tools()]

    def get_agent_toolkits(self) -> list[Toolkit]:
        return [
            Toolkit(
                name="chat",
                tools=self._build_chat_tools(),
            )
        ]

    def make_remote_toolkit(self) -> RemoteToolkit:
        local_name = self._local_participant_name()
        return RemoteToolkit(
            name="chat",
            description=f"tools for interacting with {local_name}",
            public=False,
            tools=self._build_chat_tools(),
            validation_mode="content_types",
        )

    def _touch_thread_in_index(self, *, path: str) -> None:
        self._upsert_thread_list_entry(
            path=path,
            modified_at=self._utc_now_iso(),
        )

    async def _open_thread_list_document(self) -> None:
        index_path = self._thread_list_index_path()
        if index_path is None:
            return

        if (
            self._thread_list_document is not None
            and self._thread_list_path == index_path
        ):
            return

        self._thread_list_document = await self._room.sync.open(
            path=index_path,
            schema=thread_list_schema,
        )
        self._thread_list_path = index_path

    async def _close_thread_list_document(self) -> None:
        thread_list_path = self._thread_list_path
        if self._thread_list_document is None or thread_list_path is None:
            return

        self._thread_list_document = None
        self._thread_list_path = None
        await self._room.sync.close(path=thread_list_path)

    def _send_thread_tool_providers(
        self,
        *,
        to: Participant,
        path: str,
    ) -> None:
        self._room.messaging.send_message_nowait(
            to=to,
            type="set_thread_tool_providers",
            message={
                "path": path,
                "tool_providers": [
                    {"name": toolkit_builder.name}
                    for toolkit_builder in self._toolkit_builders
                ],
            },
        )

    def _handle_room_control_message(
        self,
        *,
        message: RoomMessage,
        sender: Participant,
    ) -> bool:
        if message.type == "get_thread_toolkit_builders":
            thread_id = self._thread_id_from_room_message(message=message)
            if thread_id is None:
                return True
            self._register_open_participant(
                thread_id=thread_id,
                participant_id=sender.id,
            )
            self._touch_thread_in_index(path=thread_id)
            self._send_thread_tool_providers(
                to=sender,
                path=thread_id,
            )
            return True

        if message.type == "typing":
            return True

        return False

    def _register_open_participant(
        self,
        *,
        thread_id: str,
        participant_id: str,
    ) -> None:
        participant_ids = self._open_participant_ids_by_thread.setdefault(
            thread_id,
            set(),
        )
        participant_ids.add(participant_id)

    def _open_participants(self, *, thread_id: str) -> list[Participant]:
        participant_ids = self._open_participant_ids_by_thread.get(thread_id)
        if participant_ids is None:
            return []

        online_participants: list[Participant] = []
        stale_participant_ids: list[str] = []
        for participant_id in participant_ids:
            participant = self._room.messaging.get_participant(participant_id)
            if participant is None:
                stale_participant_ids.append(participant_id)
                continue
            online_participants.append(participant)

        for participant_id in stale_participant_ids:
            participant_ids.discard(participant_id)

        if len(participant_ids) == 0:
            self._open_participant_ids_by_thread.pop(thread_id, None)

        return online_participants

    def _approval_turn_id(self, *, thread_id: str, approval_id: str) -> str | None:
        pending_approvals = self._pending_approval_turn_ids_by_thread.get(thread_id)
        if pending_approvals is None:
            return None

        turn_id = pending_approvals.pop(approval_id, None)
        if len(pending_approvals) == 0:
            self._pending_approval_turn_ids_by_thread.pop(thread_id, None)
        return turn_id

    def _clear_tracked_thread_state(self, *, thread_id: str) -> None:
        self._active_turn_ids_by_thread.pop(thread_id, None)
        self._pending_approval_turn_ids_by_thread.pop(thread_id, None)
        self._active_text_by_thread.pop(thread_id, None)

    def _send_chat_to_open_participants(self, *, thread_id: str, text: str) -> None:
        for participant in self._open_participants(thread_id=thread_id):
            if participant.id == self._room.local_participant.id:
                continue

            self._room.messaging.send_message_nowait(
                to=participant,
                type="chat",
                message={
                    "path": thread_id,
                    "text": text,
                },
            )

    def _send_cleared_to_open_participants(self, *, thread_id: str) -> None:
        for participant in self._open_participants(thread_id=thread_id):
            if participant.id == self._room.local_participant.id:
                continue

            self._room.messaging.send_message_nowait(
                to=participant,
                type="cleared",
                message={
                    "path": thread_id,
                },
            )

    def _agent_message_from_room_message(
        self,
        *,
        message: RoomMessage,
    ) -> AgentMessage | None:
        message_type = message.type

        if message_type == "chat":
            payload = _ChatMessagePayload.model_validate(message.message)
            return TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=payload.path,
                content=self._content_from_chat_message(payload=payload),
                toolkits=payload.tools,
                model=payload.model,
            )

        if message_type == "steer":
            payload = _ChatMessagePayload.model_validate(message.message)
            turn_id = self._active_turn_id(thread_id=payload.path)
            if turn_id is None:
                logger.debug(
                    "ignoring steer for thread %s without an active turn",
                    payload.path,
                )
                return None

            return TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                thread_id=payload.path,
                turn_id=turn_id,
                content=self._content_from_chat_message(payload=payload),
                toolkits=payload.tools,
            )

        if message_type == "cancel":
            payload = _PathMessagePayload.model_validate(message.message)
            turn_id = self._active_turn_id(thread_id=payload.path)
            if turn_id is None:
                logger.debug(
                    "ignoring cancel for thread %s without an active turn",
                    payload.path,
                )
                return None

            return TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                thread_id=payload.path,
                turn_id=turn_id,
            )

        if message_type in {"approved", "rejected"}:
            payload = _ApprovalDecisionPayload.model_validate(message.message)
            turn_id = self._approval_turn_id(
                thread_id=payload.path,
                approval_id=payload.approval_id,
            )
            if turn_id is None:
                logger.debug(
                    "ignoring %s for thread %s without a pending approval",
                    message_type,
                    payload.path,
                )
                return None

            if message_type == "approved":
                return ApproveAgentToolCall(
                    type=AGENT_MESSAGE_TOOL_CALL_APPROVE,
                    thread_id=payload.path,
                    turn_id=turn_id,
                    item_id=payload.approval_id,
                )

            return RejectAgentToolCall(
                type=AGENT_MESSAGE_TOOL_CALL_REJECT,
                thread_id=payload.path,
                turn_id=turn_id,
                item_id=payload.approval_id,
            )

        if message_type == "clear":
            payload = _PathMessagePayload.model_validate(message.message)
            return ClearThread(
                type=AGENT_MESSAGE_THREAD_CLEAR,
                thread_id=payload.path,
            )

        if message_type in {"opened", "cleared"}:
            logger.debug(
                "ignoring chat room message of type %s because it has no agent equivalent",
                message_type,
            )
            return None

        return None
