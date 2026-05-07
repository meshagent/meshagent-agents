from __future__ import annotations

import logging
import re
import uuid
from pathlib import PurePosixPath
from typing import Any, TypeVar
from urllib.parse import urlparse

from meshagent.api import Element, Participant, RoomClient, RoomException, RoomMessage
from meshagent.api.messaging import JsonContent
from pydantic import BaseModel, ValidationError
from meshagent.tools import FunctionTool, ToolContext, Toolkit, tool
from meshagent.tools.strict_schema import ensure_strict_json_schema

from .adapter import LLMAdapter
from .messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_THREAD_OPEN,
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
    ApproveAgentToolCall,
    CapabilitiesRequest,
    ClearThread,
    CloseThread,
    AgentThreadStatus,
    OpenThread,
    RejectAgentToolCall,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnSteered,
    TurnStart,
    TurnStarted,
    TurnSteer,
)
from .process import Message
from .threaded_channel import ThreadedChannel

logger = logging.getLogger("chat-channel")
_MessageT = TypeVar("_MessageT", bound=AgentMessage)


class _AgentMessageEnvelope(BaseModel):
    payload: dict[str, Any]


class _ChatAttachmentPayload(BaseModel):
    path: str


class _ChatMessagePayload(BaseModel):
    path: str
    text: str = ""
    attachments: list[_ChatAttachmentPayload] | None = None


class _PathMessagePayload(BaseModel):
    path: str


_INBOUND_AGENT_MESSAGE_MODELS: dict[str, type[AgentMessage]] = {
    AGENT_MESSAGE_CAPABILITIES_REQUEST: CapabilitiesRequest,
    AGENT_MESSAGE_TURN_START: TurnStart,
    AGENT_MESSAGE_TURN_STEER: TurnSteer,
    AGENT_MESSAGE_TURN_INTERRUPT: TurnInterrupt,
    AGENT_MESSAGE_THREAD_CLEAR: ClearThread,
    AGENT_MESSAGE_THREAD_OPEN: OpenThread,
    AGENT_MESSAGE_THREAD_CLOSE: CloseThread,
    AGENT_MESSAGE_TOOL_CALL_APPROVE: ApproveAgentToolCall,
    AGENT_MESSAGE_TOOL_CALL_REJECT: RejectAgentToolCall,
}

_THREAD_CONTROL_AGENT_MESSAGE_MODELS: dict[str, type[AgentMessage]] = {
    AGENT_MESSAGE_THREAD_OPEN: OpenThread,
    AGENT_MESSAGE_THREAD_CLOSE: CloseThread,
}


class ChatChannel(ThreadedChannel):
    _THREAD_INDEX_BUMP_AGENT_MESSAGE_TYPES = {
        AGENT_MESSAGE_TURN_START,
        AGENT_MESSAGE_TURN_STEER,
    }

    def __init__(
        self,
        *,
        room: RoomClient,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        thread_url_scheme: str | None = None,
        thread_path_extension: str = ".thread",
        llm_adapter: LLMAdapter | None = None,
        empty_state_title: str = "How can I help you?",
    ) -> None:
        super().__init__(
            room=room,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_url_scheme=thread_url_scheme,
            thread_path_extension=thread_path_extension,
            llm_adapter=llm_adapter,
        )
        self._empty_state_title = empty_state_title
        self._active_turn_ids_by_thread: dict[str, str] = {}
        self._open_participant_ids_by_thread: dict[str, set[str]] = {}
        self._event_buffer_by_thread: dict[str, list[dict[str, Any]]] = {}
        self._thread_status_by_thread: dict[str, dict[str, Any]] = {}
        self._turn_input_payloads_by_message_id: dict[str, dict[str, Any]] = {}
        self._max_event_buffer_size = 512

    def _uses_explicit_thread_dir_for_thread_list(self) -> bool:
        return True

    def get_agent_toolkits(self) -> list[Toolkit]:
        return [
            Toolkit(
                name="chat",
                tools=self._build_chat_tools(),
            )
        ]

    def get_exposed_toolkits(self) -> list[Toolkit]:
        return [self.make_toolkit()]

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        if message_type in _INBOUND_AGENT_MESSAGE_MODELS:
            return False
        return message_type.startswith("meshagent.agent.")

    async def on_start(self) -> None:
        self._room.messaging.on("message", self._on_room_message)
        await self.publish_thread_attributes()
        await self._room.local_participant.set_attribute(
            "empty_state_title",
            self._empty_state_title,
        )
        await self._room.local_participant.set_attribute(
            "supports_agent_messages", True
        )
        await self.open_thread_list_document()
        if not self._room.messaging.is_enabled:
            await self._room.messaging.enable()

    async def on_stop(self) -> None:
        await self._room.local_participant.set_attribute(
            "supports_agent_messages", None
        )
        self._room.messaging.off("message", self._on_room_message)
        await self._cancel_thread_list_background_tasks()
        await self.close_thread_list_document()
        self._active_turn_ids_by_thread.clear()
        self._open_participant_ids_by_thread.clear()
        self._event_buffer_by_thread.clear()
        self._thread_status_by_thread.clear()
        self._turn_input_payloads_by_message_id.clear()

    async def on_message(self, message: Message) -> None:
        self._track_agent_event(message=message)
        payload = self._outbound_agent_message_payload(message=message)
        if self._should_buffer_agent_event(payload=payload):
            self._buffer_agent_event(payload=payload)
        for participant in self._open_participants(thread_id=message.data.thread_id):
            if participant.id == self.room.local_participant.id:
                continue

            self._send_agent_payload_nowait(participant=participant, payload=payload)

    def _outbound_agent_message_payload(self, *, message: Message) -> dict[str, Any]:
        payload = message.data.model_dump(mode="json")
        if message.data.type == AGENT_EVENT_TURN_START_ACCEPTED:
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=payload.get("source_message_id"),
                remove=False,
            )

        if message.data.type == AGENT_EVENT_TURN_STEER_ACCEPTED:
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=payload.get("source_message_id"),
                remove=False,
            )

        if message.data.type == AGENT_EVENT_TURN_STARTED:
            source_message_id = self._coerce_message(
                data=message.data, model=TurnStarted
            ).source_message_id
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=source_message_id,
                remove=True,
            )

        if message.data.type == AGENT_EVENT_TURN_STEERED:
            source_message_id = self._coerce_message(
                data=message.data, model=TurnSteered
            ).source_message_id
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=source_message_id,
                remove=True,
            )

        return payload

    def _outbound_turn_input_payload(
        self,
        *,
        payload: dict[str, Any],
        source_message_id: Any,
        remove: bool,
    ) -> dict[str, Any]:
        if not isinstance(source_message_id, str):
            return payload

        if remove:
            input_payload = self._turn_input_payloads_by_message_id.pop(
                source_message_id,
                None,
            )
        else:
            input_payload = self._turn_input_payloads_by_message_id.get(
                source_message_id
            )
        if input_payload is None:
            return payload

        input_content = input_payload.get("content")
        if isinstance(input_content, list):
            payload["content"] = input_content

        sender_name = input_payload.get("sender_name")
        if isinstance(sender_name, str) and sender_name.strip() != "":
            payload["sender_name"] = sender_name.strip()

        return payload

    def _send_agent_payload_nowait(
        self,
        *,
        participant: Participant,
        payload: dict[str, Any],
    ) -> None:
        try:
            self.room.messaging.send_message_nowait(
                to=participant,
                type="agent-message",
                message={"payload": payload},
            )
        except Exception:
            logger.debug(
                "failed to send agent message to participant %s",
                participant.id,
                exc_info=True,
            )

    def _buffer_agent_event(self, *, payload: dict[str, Any]) -> None:
        raw_thread_id = payload.get("thread_id")
        if not isinstance(raw_thread_id, str) or raw_thread_id.strip() == "":
            return

        thread_id = raw_thread_id.strip()
        buffer = self._event_buffer_by_thread.setdefault(thread_id, [])
        buffer.append(dict(payload))
        if len(buffer) > self._max_event_buffer_size:
            del buffer[: len(buffer) - self._max_event_buffer_size]

    @staticmethod
    def _should_buffer_agent_event(*, payload: dict[str, Any]) -> bool:
        message_type = payload.get("type")
        if message_type == AGENT_EVENT_THREAD_STATUS:
            return False

        return message_type not in {
            AGENT_EVENT_THREAD_CLEARED,
            AGENT_EVENT_TURN_ENDED,
        }

    def _track_agent_event(self, *, message: Message) -> None:
        data = message.data
        if data.type == AGENT_EVENT_THREAD_STATUS:
            thread_status = self._coerce_message(data=data, model=AgentThreadStatus)
            if thread_status.status is None:
                self._thread_status_by_thread.pop(thread_status.thread_id, None)
            else:
                self._thread_status_by_thread[thread_status.thread_id] = (
                    thread_status.model_dump(mode="json")
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
            self._event_buffer_by_thread.pop(thread_cleared.thread_id, None)
            self._thread_status_by_thread.pop(thread_cleared.thread_id, None)
            return

        if data.type == AGENT_EVENT_TURN_ENDED:
            turn_ended = self._coerce_message(data=data, model=TurnEnded)
            tracked_turn_id = self._active_turn_ids_by_thread.get(turn_ended.thread_id)
            if tracked_turn_id == turn_ended.turn_id:
                self._active_turn_ids_by_thread.pop(turn_ended.thread_id, None)
            self._event_buffer_by_thread.pop(turn_ended.thread_id, None)
            self._thread_status_by_thread.pop(turn_ended.thread_id, None)

    def _on_room_message(self, *, message: RoomMessage) -> None:
        sender = self._room.messaging.get_participant(message.from_participant_id)
        if sender is None:
            logger.warning(
                "ignoring chat message from unknown participant %s",
                message.from_participant_id,
            )
            return

        if message.type == "typing":
            return

        thread_id = self._thread_id_from_room_message(message=message)
        if thread_id is not None:
            if self._should_touch_thread_index_for_room_message(message=message):
                self.bump_thread(path=thread_id)

        if message.type != "agent-message":
            if thread_id is not None and message.type == "opened":
                self._register_open_participant(
                    thread_id=thread_id,
                    participant_id=sender.id,
                )
            logger.debug(
                "ignoring unsupported chat room message of type %s",
                message.type,
            )
            return

        try:
            if self._handle_agent_control_message(message=message, sender=sender):
                return
            agent_message = self._agent_message_from_room_message(message=message)
        except (ValidationError, ValueError):
            logger.exception(
                "unable to translate chat room message of type %s",
                message.type,
            )
            return

        self._track_inbound_agent_message(message=agent_message, sender=sender)
        self.emit(sender=sender, payload=agent_message)

    def _track_inbound_agent_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant,
    ) -> None:
        if message.type not in {AGENT_MESSAGE_TURN_START, AGENT_MESSAGE_TURN_STEER}:
            return

        payload = message.model_dump(mode="json")
        sender_name = sender.get_attribute("name")
        if isinstance(sender_name, str) and sender_name.strip() != "":
            payload["sender_name"] = sender_name.strip()

        self._turn_input_payloads_by_message_id[message.message_id] = payload

    def _handle_agent_control_message(
        self,
        *,
        message: RoomMessage,
        sender: Participant,
    ) -> bool:
        envelope = _AgentMessageEnvelope.model_validate(message.message)
        payload = self._normalize_agent_message_payload(payload=envelope.payload)
        message_type = payload.get("type")
        if not isinstance(message_type, str):
            return False

        model = _THREAD_CONTROL_AGENT_MESSAGE_MODELS.get(message_type)
        if model is None:
            return False

        control_message = model.model_validate(payload)
        if isinstance(control_message, OpenThread):
            self._register_open_participant(
                thread_id=control_message.thread_id,
                participant_id=sender.id,
            )
            self._send_buffered_agent_events(
                thread_id=control_message.thread_id,
                participant=sender,
            )
            return False

        if isinstance(control_message, CloseThread):
            self._remove_open_participant(
                thread_id=control_message.thread_id,
                participant_id=sender.id,
            )
            return False

        return False

    def _send_buffered_agent_events(
        self,
        *,
        thread_id: str,
        participant: Participant,
    ) -> None:
        buffered_accepted_source_message_ids: set[str] = set()
        for payload in self._event_buffer_by_thread.get(thread_id, []):
            if payload.get("type") in {
                AGENT_EVENT_TURN_START_ACCEPTED,
                AGENT_EVENT_TURN_STEER_ACCEPTED,
            }:
                source_message_id = payload.get("source_message_id")
                if isinstance(source_message_id, str):
                    buffered_accepted_source_message_ids.add(source_message_id)
            self._send_agent_payload_nowait(participant=participant, payload=payload)

        for payload in self._pending_accepted_turn_payloads(thread_id=thread_id):
            source_message_id = payload.get("source_message_id")
            if (
                isinstance(source_message_id, str)
                and source_message_id in buffered_accepted_source_message_ids
            ):
                continue
            self._send_agent_payload_nowait(participant=participant, payload=payload)

        status_payload = self._thread_status_by_thread.get(thread_id)
        if status_payload is not None:
            self._send_agent_payload_nowait(
                participant=participant,
                payload=status_payload,
            )

    def _pending_accepted_turn_payloads(
        self, *, thread_id: str
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for (
            source_message_id,
            input_payload,
        ) in self._turn_input_payloads_by_message_id.items():
            if input_payload.get("thread_id") != thread_id:
                continue

            input_type = input_payload.get("type")
            if input_type == AGENT_MESSAGE_TURN_START:
                accepted_type = AGENT_EVENT_TURN_START_ACCEPTED
            elif input_type == AGENT_MESSAGE_TURN_STEER:
                accepted_type = AGENT_EVENT_TURN_STEER_ACCEPTED
            else:
                continue

            payload = {
                "type": accepted_type,
                "thread_id": thread_id,
                "source_message_id": source_message_id,
            }
            turn_id = input_payload.get("turn_id")
            if accepted_type == AGENT_EVENT_TURN_STEER_ACCEPTED and isinstance(
                turn_id, str
            ):
                payload["turn_id"] = turn_id

            payloads.append(
                self._outbound_turn_input_payload(
                    payload=payload,
                    source_message_id=source_message_id,
                    remove=False,
                )
            )

        return payloads

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

    @classmethod
    def _content_from_chat_message(
        cls,
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
                normalized_url = cls._normalize_attachment_url(path=attachment.path)
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

    @classmethod
    def _normalize_agent_message_payload(
        cls,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_payload = dict(payload)
        message_type = normalized_payload.get("type")
        if message_type not in {
            AGENT_MESSAGE_TURN_START,
            AGENT_MESSAGE_TURN_STEER,
        }:
            return normalized_payload

        raw_content = normalized_payload.get("content")
        if not isinstance(raw_content, list):
            return normalized_payload

        normalized_content: list[Any] = []
        content_changed = False
        for item in raw_content:
            if not isinstance(item, dict):
                normalized_content.append(item)
                continue

            if item.get("type") != "file":
                normalized_content.append(item)
                continue

            url = item.get("url")
            if not isinstance(url, str):
                normalized_content.append(item)
                continue

            normalized_url = cls._normalize_attachment_url(path=url)
            if normalized_url is None or normalized_url == url:
                normalized_content.append(item)
                continue

            normalized_content.append({**item, "url": normalized_url})
            content_changed = True

        if content_changed:
            normalized_payload["content"] = normalized_content

        return normalized_payload

    @staticmethod
    def _thread_id_from_room_message(*, message: RoomMessage) -> str | None:
        if message.type == "agent-message":
            try:
                envelope = _AgentMessageEnvelope.model_validate(message.message)
            except Exception:
                return None

            raw_thread_id = envelope.payload.get("thread_id")
            if not isinstance(raw_thread_id, str):
                return None

            thread_id = raw_thread_id.strip()
            if thread_id == "":
                return None

            return thread_id

        if message.type != "opened":
            return None

        try:
            payload = _PathMessagePayload.model_validate(message.message)
        except ValidationError:
            return None

        thread_id = payload.path.strip()
        if thread_id == "":
            return None
        return thread_id

    @classmethod
    def _should_touch_thread_index_for_room_message(
        cls,
        *,
        message: RoomMessage,
    ) -> bool:
        if message.type != "agent-message":
            return False

        try:
            envelope = _AgentMessageEnvelope.model_validate(message.message)
        except Exception:
            return False

        payload_type = envelope.payload.get("type")
        if not isinstance(payload_type, str):
            return False

        return payload_type in cls._THREAD_INDEX_BUMP_AGENT_MESSAGE_TYPES

    def _agent_message_from_room_message(
        self,
        *,
        message: RoomMessage,
    ) -> AgentMessage:
        envelope = _AgentMessageEnvelope.model_validate(message.message)
        payload = self._normalize_agent_message_payload(payload=envelope.payload)
        message_type = payload.get("type")
        if not isinstance(message_type, str):
            raise ValueError("agent-message payload must include a string type")

        model = _INBOUND_AGENT_MESSAGE_MODELS.get(message_type)
        if model is None:
            raise ValueError(f"unsupported agent-message payload type: {message_type}")

        return model.model_validate(payload)

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

                turn_start = TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id=path,
                    content=outer._content_from_chat_message(payload=chat_message),
                )
                outer._begin_pending_thread_list_entry(path=path)
                await outer.supervisor.route(
                    Message(
                        sender=context.on_behalf_of or context.caller,
                        source=outer,
                        data=turn_start,
                    )
                )
                outer._schedule_pending_thread_list_entry(
                    path=path,
                    message_text=text,
                    attachments=attachment_paths,
                    on_behalf_of=context.on_behalf_of or context.caller,
                )
                return JsonContent(
                    json={"path": path, "message_id": turn_start.message_id}
                )

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

    def make_toolkit(self) -> Toolkit:
        local_name = self._local_participant_name()
        return Toolkit(
            name="chat",
            description=f"tools for interacting with {local_name}",
            public=False,
            tools=self._build_chat_tools(),
            validation_mode="content_types",
        )

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

    def _remove_open_participant(
        self,
        *,
        thread_id: str,
        participant_id: str,
    ) -> None:
        participant_ids = self._open_participant_ids_by_thread.get(thread_id)
        if participant_ids is None:
            return
        participant_ids.discard(participant_id)
        if len(participant_ids) == 0:
            self._open_participant_ids_by_thread.pop(thread_id, None)

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

    def _clear_tracked_thread_state(self, *, thread_id: str) -> None:
        self._active_turn_ids_by_thread.pop(thread_id, None)
        self._turn_input_payloads_by_message_id = {
            message_id: payload
            for message_id, payload in self._turn_input_payloads_by_message_id.items()
            if payload.get("thread_id") != thread_id
        }
