from __future__ import annotations

import logging
import re
import uuid
from pathlib import PurePosixPath
from typing import Any, TypeVar
from urllib.parse import urlparse

from meshagent.api import (
    Element,
    Participant,
    RoomClient,
    RoomException,
    RoomMessage,
)
from meshagent.api.messaging import JsonContent
from pydantic import BaseModel, ValidationError
from meshagent.tools import (
    FunctionTool,
    ToolContext,
    Toolkit,
    tool,
)
from meshagent.tools.strict_schema import ensure_strict_json_schema

from .adapter import LLMAdapter
from .messages import (
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
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
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
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
from .process import Message
from .threaded_channel import ThreadedChannel

logger = logging.getLogger("legacy-chat-channel")
_MessageT = TypeVar("_MessageT", bound=AgentMessage)


class _ChatAttachmentPayload(BaseModel):
    path: str


class _ChatMessagePayload(BaseModel):
    path: str
    text: str = ""
    attachments: list[_ChatAttachmentPayload] | None = None
    model: str | None = None


class _PathMessagePayload(BaseModel):
    path: str


class _ApprovalDecisionPayload(BaseModel):
    path: str
    approval_id: str


class LegacyChatChannel(ThreadedChannel):
    def __init__(
        self,
        *,
        room: RoomClient,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        llm_adapter: LLMAdapter | None = None,
        empty_state_title: str = "How can I help you?",
    ) -> None:
        super().__init__(
            room=room,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            llm_adapter=llm_adapter,
        )
        self._empty_state_title = empty_state_title
        self._active_turn_ids_by_thread: dict[str, str] = {}
        self._pending_approval_turn_ids_by_thread: dict[str, dict[str, str]] = {}
        self._open_participant_ids_by_thread: dict[str, set[str]] = {}
        self._active_text_by_thread: dict[str, dict[str, str]] = {}
        self._active_files_by_thread: dict[str, dict[str, str]] = {}

    def _uses_explicit_thread_dir_for_thread_list(self) -> bool:
        return True

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_EVENT_FILE_CONTENT_STARTED,
            AGENT_EVENT_FILE_CONTENT_DELTA,
            AGENT_EVENT_FILE_CONTENT_ENDED,
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
        await self.publish_thread_attributes()
        await self._room.local_participant.set_attribute(
            "empty_state_title",
            self._empty_state_title,
        )
        await self.open_thread_list_document()
        if not self._room.messaging.is_enabled:
            await self._room.messaging.enable()

    async def on_stop(self) -> None:
        self._room.messaging.off("message", self._on_room_message)
        await self._cancel_thread_list_background_tasks()
        await self.close_thread_list_document()
        self._active_turn_ids_by_thread.clear()
        self._pending_approval_turn_ids_by_thread.clear()
        self._open_participant_ids_by_thread.clear()
        self._active_text_by_thread.clear()
        self._active_files_by_thread.clear()

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

        if data.type == AGENT_EVENT_FILE_CONTENT_STARTED:
            file_started = self._coerce_message(
                data=data, model=AgentFileContentStarted
            )
            active_files = self._active_files_by_thread.setdefault(
                file_started.thread_id,
                {},
            )
            active_files[file_started.item_id] = ""
            return

        if data.type == AGENT_EVENT_FILE_CONTENT_DELTA:
            file_delta = self._coerce_message(data=data, model=AgentFileContentDelta)
            active_files = self._active_files_by_thread.setdefault(
                file_delta.thread_id,
                {},
            )
            active_files[file_delta.item_id] = file_delta.url
            return

        if data.type == AGENT_EVENT_FILE_CONTENT_ENDED:
            file_ended = self._coerce_message(data=data, model=AgentFileContentEnded)
            active_files = self._active_files_by_thread.get(file_ended.thread_id, {})
            path = active_files.pop(file_ended.item_id, "")
            if len(active_files) == 0:
                self._active_files_by_thread.pop(file_ended.thread_id, None)
            if path.strip() != "":
                self._send_attachments_to_open_participants(
                    thread_id=file_ended.thread_id,
                    attachments=[path],
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
            self._active_files_by_thread.pop(turn_ended.thread_id, None)

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
            if self._should_touch_thread_index_for_room_message(message=message):
                self.bump_thread(path=thread_id)

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
    def _room_storage_path_from_attachment_url(*, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme == "":
            raw_path = url.strip()
        elif parsed.scheme == "room":
            raw_path = f"{parsed.netloc}{parsed.path}"
        else:
            return None

        normalized = PurePosixPath("/" + raw_path).as_posix().strip("/")
        if normalized == "":
            return None

        if any(part in {".", ".."} for part in PurePosixPath(normalized).parts):
            return None

        return normalized

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

    def _thread_and_turn_id_from_tool_context(
        self, *, context: ToolContext
    ) -> tuple[str, str]:
        caller_context = context.caller_context
        if not isinstance(caller_context, dict):
            raise RoomException(
                "chat tool requires thread_id and turn_id in caller_context"
            )

        raw_thread_id = caller_context.get("thread_id")
        if not isinstance(raw_thread_id, str) or raw_thread_id.strip() == "":
            raise RoomException("chat tool requires a non-empty thread_id")
        thread_id = raw_thread_id.strip()

        raw_turn_id = caller_context.get("turn_id")
        if isinstance(raw_turn_id, str) and raw_turn_id.strip() != "":
            return thread_id, raw_turn_id.strip()

        turn_id = self._active_turn_id(thread_id=thread_id)
        if turn_id is None:
            raise RoomException("attach_file requires an active turn")

        return thread_id, turn_id

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

        return ensure_strict_json_schema(tools_schema)

    def _make_new_thread_tool(self) -> FunctionTool:
        local_name = self._local_participant_name()
        tools_schema = self._build_new_thread_tool_schema()

        outer = self

        class NewThreadTool(FunctionTool):
            def __init__(self) -> None:
                super().__init__(
                    name="new_thread",
                    description=f"starts a new thread for {local_name}, posts a message to the thread, and then returns the new thread path. The thread list entry is named and added asynchronously, so an agent should invoke this as fire and forget.",
                    input_schema=tools_schema,
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
                attachment_paths = [
                    attachment.path
                    for attachment in payload.get("attachments") or []
                    if isinstance(attachment, _ChatAttachmentPayload)
                ]
                if len(attachment_paths) == 0:
                    raw_attachments = payload.get("attachments")
                    if isinstance(raw_attachments, list):
                        attachment_paths = [
                            attachment_value["path"]
                            for attachment_value in raw_attachments
                            if isinstance(attachment_value, dict)
                            and isinstance(attachment_value.get("path"), str)
                            and attachment_value.get("path", "").strip() != ""
                        ]

                if text.strip() == "" and len(attachment_paths) == 0:
                    raise RoomException(
                        "chat.new_thread requires non-empty text or at least one attachment"
                    )

                path = await outer._new_thread_path()

                chat_message = _ChatMessagePayload.model_validate(
                    {
                        "path": path,
                        "text": text,
                        "attachments": payload.get("attachments"),
                    }
                )
                outer._begin_pending_thread_list_entry(path=path)
                outer.emit(
                    sender=context.on_behalf_of or context.caller,
                    payload=TurnStart(
                        type=AGENT_MESSAGE_TURN_START,
                        thread_id=path,
                        content=outer._content_from_chat_message(payload=chat_message),
                    ),
                )
                outer._schedule_pending_thread_list_entry(
                    path=path,
                    message_text=text,
                    attachments=attachment_paths,
                    on_behalf_of=context.on_behalf_of or context.caller,
                )
                return JsonContent(json={"path": path})

        return NewThreadTool()

    def _make_attach_file_tool(self) -> FunctionTool:
        outer = self

        @tool(
            name="attach_file",
            description="attach a room file path or URL to the current thread so the user can see it",
        )
        async def attach_file(context: ToolContext, path: str) -> None:
            thread_id, turn_id = outer._thread_and_turn_id_from_tool_context(
                context=context
            )
            normalized_url = outer._normalize_attachment_url(path=path)
            if normalized_url is None:
                raise RoomException("attach_file requires a non-empty path")

            room_storage_path = outer._room_storage_path_from_attachment_url(
                url=normalized_url
            )
            if room_storage_path is not None:
                try:
                    exists = await outer.room.storage.exists(path=room_storage_path)
                except Exception as exc:
                    raise RoomException(
                        f"attach_file could not verify room file {room_storage_path}: {exc}"
                    ) from exc
                if not exists:
                    raise RoomException(
                        f"attach_file could not find a room file at {room_storage_path}"
                    )

            item_id = str(uuid.uuid4())
            sender = context.on_behalf_of or context.caller
            outer.emit(
                sender=sender,
                payload=AgentFileContentStarted(
                    type=AGENT_EVENT_FILE_CONTENT_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                ),
            )
            outer.emit(
                sender=sender,
                payload=AgentFileContentDelta(
                    type=AGENT_EVENT_FILE_CONTENT_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    url=normalized_url,
                ),
            )
            outer.emit(
                sender=sender,
                payload=AgentFileContentEnded(
                    type=AGENT_EVENT_FILE_CONTENT_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                ),
            )

        return attach_file

    def _build_chat_tools(self) -> list[FunctionTool]:
        return [
            self._make_new_thread_tool(),
            self._make_attach_file_tool(),
            *self._build_thread_list_tools(),
        ]

    def get_agent_toolkits(self) -> list[Toolkit]:
        return [
            Toolkit(
                name="chat",
                tools=self._build_chat_tools(),
            )
        ]

    def make_toolkit(self) -> Toolkit:
        local_name = self._local_participant_name()
        return Toolkit(
            name="chat",
            description=f"tools for interacting with {local_name}",
            public=False,
            tools=self._build_chat_tools(),
            validation_mode="content_types",
        )

    def _handle_room_control_message(
        self,
        *,
        message: RoomMessage,
        sender: Participant,
    ) -> bool:
        if message.type == "typing":
            return True

        return False

    @classmethod
    def _should_touch_thread_index_for_room_message(
        cls,
        *,
        message: RoomMessage,
    ) -> bool:
        del cls
        return message.type not in {
            "opened",
            "cleared",
            "typing",
        }

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
        self._active_files_by_thread.pop(thread_id, None)

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

    def _send_attachments_to_open_participants(
        self,
        *,
        thread_id: str,
        attachments: list[str],
    ) -> None:
        normalized_attachments = [
            {"path": attachment}
            for attachment in attachments
            if isinstance(attachment, str) and attachment.strip() != ""
        ]
        if len(normalized_attachments) == 0:
            return

        for participant in self._open_participants(thread_id=thread_id):
            if participant.id == self._room.local_participant.id:
                continue

            self._room.messaging.send_message_nowait(
                to=participant,
                type="chat",
                message={
                    "path": thread_id,
                    "attachments": normalized_attachments,
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
