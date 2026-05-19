from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable, Protocol, TypeVar
from urllib.parse import urlparse

import aiohttp
import msgpack
from aiohttp import web
from meshagent.api import Participant, RoomClient, RoomException, RoomMessage
from meshagent.api.messaging import JsonContent
from pydantic import BaseModel, ValidationError
from meshagent.tools import FunctionTool, ToolContext, Toolkit, tool
from meshagent.tools.strict_schema import ensure_strict_json_schema

from .adapter import LLMAdapter
from .messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_AUDIO_GENERATION_DELTA,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
    AGENT_EVENT_IMAGE_GENERATION_FAILED,
    AGENT_EVENT_IMAGE_GENERATION_PARTIAL,
    AGENT_EVENT_IMAGE_GENERATION_STARTED,
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_PARTICIPANT_CONNECT,
    AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentError,
    AgentFileContent,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentAudioGenerationDelta,
    AgentMessage,
    AgentRealtimeAudioChunk,
    AgentRealtimeAudioCommit,
    AgentRealtimeConnectionInfo,
    AgentTextContent,
    AgentThreadMessage,
    CloseThread,
    AgentThreadStatus,
    AgentToolCallEnded,
    OpenThread,
    ParticipantConnect,
    ParticipantDisconnect,
    StartThread,
    ThreadCleared,
    ThreadStarted,
    TurnEnded,
    TurnSteered,
    TurnStart,
    TurnStartAccepted,
    TurnStartRejected,
    TurnStarted,
    TurnSteer,
    parse_agent_message,
)
from .process import AgentSupervisor, Message
from .threaded_channel import ThreadedChannel

DEFAULT_WEBSOCKET_MAX_MSG_SIZE = int(
    os.getenv("MESHAGENT_AGENT_WEBSOCKET_MAX_MSG_SIZE", str(64 * 1024 * 1024))
)
_LARGE_AGENT_WEBSOCKET_MESSAGE_LOG_BYTES = int(
    os.getenv("MESHAGENT_AGENT_WEBSOCKET_LARGE_MESSAGE_LOG_BYTES", str(1024 * 1024))
)
_IMAGE_GENERATION_AGENT_EVENT_TYPES = {
    AGENT_EVENT_IMAGE_GENERATION_STARTED,
    AGENT_EVENT_IMAGE_GENERATION_PARTIAL,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
    AGENT_EVENT_IMAGE_GENERATION_FAILED,
}

logger = logging.getLogger("chat-channel")
_MessageT = TypeVar("_MessageT", bound=AgentMessage)
WebSocketAuthorizeHook = Callable[
    [web.Request],
    Awaitable[Participant | web.StreamResponse | None]
    | Participant
    | web.StreamResponse
    | None,
]


class WebSocketChatEncoding(Protocol):
    def encode(self, message: AgentMessage) -> str | bytes: ...

    def decode(self, message: aiohttp.WSMessage) -> AgentMessage: ...


class MsgpackWebSocketChatEncoding:
    def encode(self, message: AgentMessage) -> bytes:
        return msgpack.packb(
            message.model_dump(mode="json", exclude_none=True),
            use_bin_type=True,
        )

    def decode(self, message: aiohttp.WSMessage) -> AgentMessage:
        if message.type != aiohttp.WSMsgType.BINARY:
            raise ValueError("msgpack websocket encoding requires binary messages")
        payload = msgpack.unpackb(message.data, raw=False)
        if not isinstance(payload, dict):
            raise ValueError("msgpack websocket message must decode to an object")
        return parse_agent_message(payload)


class _ChatAttachmentPayload(BaseModel):
    path: str
    name: str | None = None


class _ChatMessagePayload(BaseModel):
    path: str
    text: str = ""
    attachments: list[_ChatAttachmentPayload] | None = None


class _PathMessagePayload(BaseModel):
    path: str


_THREAD_CONTROL_AGENT_MESSAGE_MODELS: dict[str, type[AgentMessage]] = {
    AGENT_MESSAGE_THREAD_START: StartThread,
    AGENT_MESSAGE_THREAD_OPEN: OpenThread,
    AGENT_MESSAGE_THREAD_CLOSE: CloseThread,
}


class BaseChatChannel(ThreadedChannel):
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
        self._tool_argument_delta_buffer_by_thread_item: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        self._thread_status_by_thread: dict[str, dict[str, Any]] = {}
        self._turn_input_payloads_by_message_id: dict[str, dict[str, Any]] = {}
        self._max_event_buffer_size = 512

    def _uses_explicit_thread_dir_for_thread_list(self) -> bool:
        return True

    def get_turn_toolkits(
        self,
        *,
        thread_id: str,
        turn_id: str | None = None,
    ) -> list[Toolkit]:
        return [
            Toolkit(
                name="chat",
                tools=self._build_chat_tools(thread_id=thread_id, turn_id=turn_id),
            )
        ]

    def get_exposed_toolkits(self) -> list[Toolkit]:
        return [self.make_toolkit()]

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        return message_type.startswith("meshagent.agent.")

    async def on_start(self) -> None:
        await self.publish_thread_attributes()
        await self._room.local_participant.set_attribute(
            "empty_state_title",
            self._empty_state_title,
        )
        await self._room.local_participant.set_attribute(
            "supports_agent_messages", True
        )
        await self.open_thread_list_document()

    async def on_stop(self) -> None:
        await self._room.local_participant.set_attribute(
            "supports_agent_messages", None
        )
        await self._cancel_thread_list_background_tasks()
        await self.close_thread_list_document()
        self._active_turn_ids_by_thread.clear()
        self._open_participant_ids_by_thread.clear()
        self._event_buffer_by_thread.clear()
        self._tool_argument_delta_buffer_by_thread_item.clear()
        self._thread_status_by_thread.clear()
        self._turn_input_payloads_by_message_id.clear()

    async def on_message(self, message: Message) -> None:
        self._publish_agent_event_to_open_participants(message=message)

    def _publish_agent_event_to_open_participants(self, *, message: Message) -> None:
        self._track_agent_event(message=message)
        payload = self._outbound_agent_message_payload(message=message)
        attachment = self._outbound_agent_message_attachment(message=message)
        data = message.data
        if isinstance(data, ThreadStarted) and message.sender is not None:
            self._register_open_participant(
                thread_id=data.thread_id,
                participant_id=message.sender.id,
            )
        if isinstance(data, AgentThreadMessage):
            if self._should_buffer_agent_event(payload=payload):
                self._buffer_agent_event(payload=payload)
            participants = self._open_participants(thread_id=data.thread_id)
        elif message.sender is not None:
            participants = [message.sender]
        else:
            participants = []

        for participant in participants:
            if participant.id == self.room.local_participant.id:
                continue
            self._send_agent_payload_nowait(
                participant=participant,
                payload=payload,
                attachment=attachment,
            )

    def _outbound_agent_message_payload(self, *, message: Message) -> dict[str, Any]:
        data = message.data
        if isinstance(data, (AgentAudioGenerationDelta, AgentRealtimeAudioChunk)):
            data = data.model_copy(update={"data": b""})
        payload = data.model_dump(mode="json", exclude_none=True)
        if isinstance(
            message.data, (AgentAudioGenerationDelta, AgentRealtimeAudioChunk)
        ):
            payload.pop("data", None)
        if data.type == AGENT_EVENT_TURN_STARTED:
            source_message_id = self._coerce_message(
                data=data, model=TurnStarted
            ).source_message_id
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=source_message_id,
                remove=True,
            )

        if data.type == AGENT_EVENT_TURN_STEERED:
            source_message_id = self._coerce_message(
                data=data, model=TurnSteered
            ).source_message_id
            return self._outbound_turn_input_payload(
                payload=payload,
                source_message_id=source_message_id,
                remove=True,
            )

        return payload

    @staticmethod
    def _outbound_agent_message_attachment(*, message: Message) -> bytes | None:
        data = message.data
        if isinstance(data, (AgentAudioGenerationDelta, AgentRealtimeAudioChunk)):
            return data.data
        return None

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
        attachment: bytes | None = None,
    ) -> None:
        del participant
        del payload
        del attachment
        raise NotImplementedError

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        self._send_agent_payload_nowait(
            participant=participant,
            payload=payload.model_dump(mode="json"),
        )
        return True

    async def send_agent_message_to_participant_and_wait(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        return self.send_agent_message_to_participant(
            participant=participant,
            payload=payload,
        )

    def _publish_thread_status_to_open_participants(
        self,
        *,
        thread_id: str,
        status: AgentThreadStatus,
    ) -> None:
        payload = status.model_dump(mode="json")
        self._thread_status_by_thread[thread_id] = payload
        for participant in self._open_participants(thread_id=thread_id):
            if participant.id == self.room.local_participant.id:
                continue
            self._send_agent_payload_nowait(participant=participant, payload=payload)

    def _buffer_agent_event(self, *, payload: dict[str, Any]) -> None:
        raw_thread_id = payload.get("thread_id")
        if not isinstance(raw_thread_id, str) or raw_thread_id.strip() == "":
            return

        thread_id = raw_thread_id.strip()
        if payload.get("type") == AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA:
            raw_item_id = payload.get("item_id")
            if not isinstance(raw_item_id, str) or raw_item_id.strip() == "":
                return
            key = (thread_id, raw_item_id.strip())
            existing = self._tool_argument_delta_buffer_by_thread_item.get(key)
            if existing is None:
                self._tool_argument_delta_buffer_by_thread_item[key] = dict(payload)
                return
            existing["delta"] = f"{existing.get('delta', '')}{payload.get('delta', '')}"
            existing["message_id"] = payload.get(
                "message_id", existing.get("message_id")
            )
            return
        buffer = self._event_buffer_by_thread.setdefault(thread_id, [])
        buffer.append(dict(payload))
        if len(buffer) > self._max_event_buffer_size:
            del buffer[: len(buffer) - self._max_event_buffer_size]

    def _has_buffered_agent_event(self, *, payload: dict[str, Any]) -> bool:
        thread_id = payload.get("thread_id")
        message_type = payload.get("type")
        if not isinstance(thread_id, str) or not isinstance(message_type, str):
            return False

        identity_fields = ("message_id", "item_id", "source_message_id", "turn_id")
        identity = {
            field: payload.get(field)
            for field in identity_fields
            if isinstance(payload.get(field), str)
        }
        if len(identity) == 0:
            return False

        for buffered in self._event_buffer_by_thread.get(thread_id, []):
            if buffered.get("type") != message_type:
                continue
            if all(buffered.get(field) == value for field, value in identity.items()):
                return True
        return False

    @staticmethod
    def _should_buffer_agent_event(*, payload: dict[str, Any]) -> bool:
        message_type = payload.get("type")
        if message_type == AGENT_EVENT_THREAD_STATUS:
            return False

        return message_type not in {
            AGENT_EVENT_AUDIO_GENERATION_DELTA,
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
            self._fill_turn_input_turn_id(
                thread_id=turn_started.thread_id,
                source_message_id=turn_started.source_message_id,
                turn_id=turn_started.turn_id,
            )
            self._active_turn_ids_by_thread[turn_started.thread_id] = (
                turn_started.turn_id
            )
            return

        if data.type == AGENT_EVENT_TURN_START_ACCEPTED:
            turn_start_accepted = self._coerce_message(
                data=data, model=TurnStartAccepted
            )
            if turn_start_accepted.turn_id is not None:
                self._fill_turn_input_turn_id(
                    thread_id=turn_start_accepted.thread_id,
                    source_message_id=turn_start_accepted.source_message_id,
                    turn_id=turn_start_accepted.turn_id,
                )
            return

        if data.type == AGENT_EVENT_THREAD_CLEARED:
            thread_cleared = self._coerce_message(data=data, model=ThreadCleared)
            self._clear_tracked_thread_state(thread_id=thread_cleared.thread_id)
            self._event_buffer_by_thread.pop(thread_cleared.thread_id, None)
            self._clear_tool_argument_delta_buffer(thread_id=thread_cleared.thread_id)
            self._thread_status_by_thread.pop(thread_cleared.thread_id, None)
            return

        if data.type == AGENT_EVENT_TURN_ENDED:
            turn_ended = self._coerce_message(data=data, model=TurnEnded)
            tracked_turn_id = self._active_turn_ids_by_thread.get(turn_ended.thread_id)
            if tracked_turn_id == turn_ended.turn_id:
                self._active_turn_ids_by_thread.pop(turn_ended.thread_id, None)
            self._drop_empty_open_thread_if_idle(thread_id=turn_ended.thread_id)
            self._event_buffer_by_thread.pop(turn_ended.thread_id, None)
            self._clear_tool_argument_delta_buffer(thread_id=turn_ended.thread_id)
            self._thread_status_by_thread.pop(turn_ended.thread_id, None)
            return

        if data.type == AGENT_EVENT_TOOL_CALL_ENDED:
            tool_call_ended = self._coerce_message(data=data, model=AgentToolCallEnded)
            self._tool_argument_delta_buffer_by_thread_item.pop(
                (tool_call_ended.thread_id, tool_call_ended.item_id), None
            )

    def _on_agent_message(
        self, *, agent_message: AgentMessage, sender: Participant
    ) -> None:
        agent_message = self._with_channel_sender_name(
            message=agent_message,
            sender=sender,
        )
        self._track_inbound_agent_message(message=agent_message, sender=sender)
        self._broadcast_inbound_turn_input(message=agent_message, sender=sender)
        if (
            isinstance(agent_message, AgentRealtimeAudioCommit)
            and agent_message.thread_id in self._pending_thread_list_paths
        ):
            self._schedule_pending_thread_list_entry(
                path=agent_message.thread_id,
                message_text="Audio message",
                attachments=[],
                on_behalf_of=sender,
            )
        self.emit(sender=sender, payload=agent_message)

    @staticmethod
    def _channel_sender_name(*, sender: Participant) -> str | None:
        sender_name = sender.get_attribute("name")
        if not isinstance(sender_name, str) or sender_name.strip() == "":
            return None
        return sender_name.strip()

    @classmethod
    def _with_channel_sender_name(
        cls,
        *,
        message: AgentMessage,
        sender: Participant,
    ) -> AgentMessage:
        if not isinstance(message, (StartThread, TurnStart, TurnSteer)):
            return message

        return message.model_copy(
            update={"sender_name": cls._channel_sender_name(sender=sender)}
        )

    def _track_inbound_agent_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant,
    ) -> None:
        if message.type not in {AGENT_MESSAGE_TURN_START, AGENT_MESSAGE_TURN_STEER}:
            return

        self._track_turn_input_payload(
            message=message,
            sender_name=message.sender_name,
        )

    def _track_turn_input_payload(
        self,
        *,
        message: AgentMessage,
        sender_name: Any,
    ) -> None:
        payload = message.model_dump(mode="json")
        if isinstance(sender_name, str) and sender_name.strip() != "":
            payload["sender_name"] = sender_name.strip()

        self._turn_input_payloads_by_message_id[message.message_id] = payload

    def _fill_turn_input_turn_id(
        self,
        *,
        thread_id: str,
        source_message_id: str,
        turn_id: str,
    ) -> None:
        input_payload = self._turn_input_payloads_by_message_id.get(source_message_id)
        if input_payload is not None:
            input_payload["turn_id"] = turn_id

        for payload in self._event_buffer_by_thread.get(thread_id, []):
            if (
                payload.get("type") == AGENT_MESSAGE_TURN_START
                and payload.get("message_id") == source_message_id
            ):
                payload["turn_id"] = turn_id

    def _broadcast_inbound_turn_input(
        self,
        *,
        message: AgentMessage,
        sender: Participant,
    ) -> None:
        if not isinstance(message, (TurnStart, TurnSteer)):
            return
        payload = self._outbound_turn_input_payload(
            payload=message.model_dump(mode="json"),
            source_message_id=message.message_id,
            remove=False,
        )
        if self._should_buffer_agent_event(payload=payload):
            self._buffer_agent_event(payload=payload)
        for participant in self._open_participants(thread_id=message.thread_id):
            if participant.id in {sender.id, self.room.local_participant.id}:
                continue
            self._send_agent_payload_nowait(participant=participant, payload=payload)

    def _handle_agent_control_payload(
        self,
        *,
        payload: dict[str, Any],
        sender: Participant,
    ) -> bool:
        payload = self._normalize_agent_message_payload(payload=payload)
        message_type = payload.get("type")
        if not isinstance(message_type, str):
            return False

        if message_type in {
            AGENT_MESSAGE_PARTICIPANT_CONNECT,
            AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
        }:
            logger.debug(
                "dropping client-supplied participant lifecycle message from %s",
                sender.id,
            )
            return True

        model = _THREAD_CONTROL_AGENT_MESSAGE_MODELS.get(message_type)
        if model is None:
            return False

        control_message = model.model_validate(payload)
        control_message = self._with_channel_sender_name(
            message=control_message,
            sender=sender,
        )
        if isinstance(control_message, StartThread):
            if self._supervisor_handles_start_thread():
                self.emit(
                    sender=sender,
                    payload=control_message,
                )
                return True
            task = asyncio.create_task(
                self._handle_start_thread_message(
                    start_thread=control_message,
                    sender=sender,
                )
            )
            self._track_thread_list_background_task(task=task)
            return True

        if isinstance(control_message, OpenThread):
            self._register_open_participant(
                thread_id=control_message.thread_id,
                participant_id=sender.id,
            )
            if control_message.load is not True:
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

    def _supervisor_handles_start_thread(self) -> bool:
        supervisor = self.supervisor
        if supervisor is None:
            return False
        return type(supervisor).create_thread_id is not AgentSupervisor.create_thread_id

    async def _handle_start_thread_message(
        self,
        *,
        start_thread: StartThread,
        sender: Participant,
    ) -> None:
        if self.supervisor is None:
            logger.warning(
                "ignoring start-thread message because chat channel has no supervisor"
            )
            return

        path = await self._new_thread_path()
        content = start_thread.content
        if content is None:
            realtime_connection = None
            if start_thread.realtime_protocol is not None:
                try:
                    realtime_connection = (
                        await self.supervisor.create_realtime_connection(
                            thread_id=path,
                            start_thread=start_thread,
                            sender=sender,
                        )
                    )
                except Exception as exc:
                    self._send_agent_payload_nowait(
                        participant=sender,
                        payload=TurnStartRejected(
                            type=AGENT_EVENT_TURN_START_REJECTED,
                            thread_id=path,
                            source_message_id=start_thread.message_id,
                            error=AgentError(
                                message=str(exc),
                                code="realtime_connection_failed",
                            ),
                        ).model_dump(mode="json"),
                    )
                    return
            self._begin_pending_thread_list_entry(path=path)
            self._register_open_participant(thread_id=path, participant_id=sender.id)
            self.bump_thread(
                path=path,
                name=self._sanitize_thread_name(
                    value=start_thread.name or self._default_thread_name()
                ),
            )
            self._send_agent_payload_nowait(
                participant=sender,
                payload=ThreadStarted(
                    type=AGENT_EVENT_THREAD_STARTED,
                    source_message_id=start_thread.message_id,
                    thread_id=path,
                    realtime_connection=(
                        AgentRealtimeConnectionInfo.model_validate(realtime_connection)
                        if realtime_connection is not None
                        else None
                    ),
                ).model_dump(mode="json"),
            )
            return

        turn_start = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            message_id=start_thread.message_id,
            thread_id=path,
            turn_id=str(uuid.uuid4()),
            content=content,
            sender_name=start_thread.sender_name,
            provider=start_thread.provider,
            model=start_thread.model,
            voice=start_thread.voice,
            output_modalities=start_thread.output_modalities,
            instructions=start_thread.instructions,
            toolkits=start_thread.toolkits,
            tool_choice=start_thread.tool_choice,
        )
        error = await self.supervisor.validate_turn_start(turn_start)
        if error is not None:
            self._send_agent_payload_nowait(
                participant=sender,
                payload=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=path,
                    source_message_id=start_thread.message_id,
                    error=error,
                ).model_dump(mode="json"),
            )
            return

        self._begin_pending_thread_list_entry(path=path)
        self._register_open_participant(thread_id=path, participant_id=sender.id)
        self._track_turn_input_payload(
            message=turn_start,
            sender_name=start_thread.sender_name,
        )
        self._send_agent_payload_nowait(
            participant=sender,
            payload=ThreadStarted(
                type=AGENT_EVENT_THREAD_STARTED,
                source_message_id=start_thread.message_id,
                thread_id=path,
            ).model_dump(mode="json"),
        )
        self._publish_thread_status_to_open_participants(
            thread_id=path,
            status=AgentThreadStatus(
                type=AGENT_EVENT_THREAD_STATUS,
                thread_id=path,
                turn_id=turn_start.turn_id,
                status="Starting a thread",
                mode="steerable",
                started_at=self._utc_now_iso(),
            ),
        )
        await self.supervisor.route(
            Message(sender=sender, source=self, data=turn_start)
        )
        message_text, attachments = self._text_and_attachments_from_content(
            content=content
        )
        self._schedule_pending_thread_list_entry(
            path=path,
            message_text=message_text,
            attachments=attachments,
            on_behalf_of=sender,
        )

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

        status_payload = self._thread_status_by_thread.get(thread_id)
        if status_payload is not None:
            self._send_agent_payload_nowait(
                participant=participant,
                payload=status_payload,
            )

        for payload in self._coalesced_tool_argument_delta_payloads(
            thread_id=thread_id
        ):
            self._send_agent_payload_nowait(participant=participant, payload=payload)

        for payload in self._pending_accepted_turn_payloads(thread_id=thread_id):
            source_message_id = payload.get("source_message_id")
            if (
                isinstance(source_message_id, str)
                and source_message_id in buffered_accepted_source_message_ids
            ):
                continue
            self._send_agent_payload_nowait(participant=participant, payload=payload)

    def _coalesced_tool_argument_delta_payloads(
        self, *, thread_id: str
    ) -> list[dict[str, Any]]:
        return [
            dict(payload)
            for (
                payload_thread_id,
                _,
            ), payload in self._tool_argument_delta_buffer_by_thread_item.items()
            if payload_thread_id == thread_id
        ]

    def _clear_tool_argument_delta_buffer(self, *, thread_id: str) -> None:
        self._tool_argument_delta_buffer_by_thread_item = {
            key: payload
            for key, payload in self._tool_argument_delta_buffer_by_thread_item.items()
            if key[0] != thread_id
        }

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
            if isinstance(turn_id, str):
                payload["turn_id"] = turn_id

            payloads.append(payload)

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
                        name=attachment.name,
                    )
                )

        return content

    @staticmethod
    def _text_and_attachments_from_content(
        *,
        content: list[Any],
    ) -> tuple[str, list[str]]:
        text_parts: list[str] = []
        attachments: list[str] = []
        for item in content:
            if isinstance(item, AgentTextContent):
                if item.text.strip() != "":
                    text_parts.append(item.text)
                continue

            if isinstance(item, AgentFileContent):
                if item.url.strip() != "":
                    attachments.append(item.url)
                continue

        return "\n".join(text_parts).strip(), attachments

    def _active_turn_id(self, *, thread_id: str) -> str | None:
        return self._active_turn_ids_by_thread.get(thread_id)

    @classmethod
    def _normalize_agent_message_payload(
        cls,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_payload = dict(payload)
        message_type = normalized_payload.get("type")
        if message_type == AGENT_MESSAGE_MODELS_REQUEST:
            normalized_payload.pop("thread_id", None)
            return normalized_payload

        if message_type not in {
            AGENT_MESSAGE_THREAD_START,
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
    def _payload_from_agent_room_message(
        *,
        message: RoomMessage,
    ) -> dict[str, Any] | None:
        if not isinstance(message.message, dict):
            return None

        raw_payload = message.message.get("payload")
        if "type" not in message.message and isinstance(raw_payload, dict):
            logger.warning(
                "received deprecated agent-message payload wrapper; send agent "
                "message fields at the top level"
            )
            return raw_payload

        return message.message

    @classmethod
    def _thread_id_from_room_message(cls, *, message: RoomMessage) -> str | None:
        if message.type == "agent-message":
            payload = cls._payload_from_agent_room_message(message=message)
            if payload is None:
                return None

            raw_thread_id = payload.get("thread_id")
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

        payload = cls._payload_from_agent_room_message(message=message)
        if payload is None:
            return False

        payload_type = payload.get("type")
        if not isinstance(payload_type, str):
            return False

        return payload_type in cls._THREAD_INDEX_BUMP_AGENT_MESSAGE_TYPES

    def _agent_message_from_room_message(
        self,
        *,
        message: RoomMessage,
    ) -> AgentMessage:
        raw_payload = self._payload_from_agent_room_message(message=message)
        if raw_payload is None:
            raise ValueError("agent-message payload must be a JSON object")

        payload = self._normalize_agent_message_payload(payload=raw_payload)
        message_type = payload.get("type")
        if not isinstance(message_type, str):
            raise ValueError("agent-message payload must include a string type")

        if message_type == AGENT_MESSAGE_REALTIME_AUDIO_CHUNK:
            payload["data"] = message.attachment or b""
        elif message_type == AGENT_EVENT_AUDIO_GENERATION_DELTA:
            payload["data"] = message.attachment or b""

        return parse_agent_message(payload)

    def _build_thread_list_tools(self) -> list[FunctionTool]:
        if self._thread_list_dir() is None:
            return []

        read_file_hint = (
            "Use read_file with a thread path to read that thread's contents."
        )
        outer = self

        def to_json_entry(entry) -> dict[str, str]:
            return {
                "name": str(entry.name),
                "path": str(entry.path),
                "modified_at": str(entry.modified_at),
                "created_at": str(entry.created_at),
            }

        @tool(
            name="list_threads",
            description="lists recent threads sorted by last modified date (newest first). Use read_file with a thread path to read that thread's contents.",
        )
        async def list_threads(*, limit: int = 20, offset: int = 0) -> JsonContent:
            normalized_offset = max(0, int(offset))
            normalized_limit = max(1, min(200, int(limit)))

            page = await outer.list_threads(
                limit=normalized_limit,
                offset=normalized_offset,
            )
            if page.total == 0:
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

            if len(page.threads) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": page.total,
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "no threads were found for the requested limit/offset",
                        "read_file_hint": read_file_hint,
                    }
                )

            return JsonContent(
                json={
                    "threads": [to_json_entry(entry) for entry in page.threads],
                    "total": page.total,
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
        async def grep_thread_list(
            *, pattern: str, ignore_case: bool = True
        ) -> JsonContent:
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
            page = await outer.list_threads(limit=200, offset=0)
            for entry in page.threads:
                haystack = (
                    f"{entry.name}\n{entry.path}\n"
                    f"{entry.created_at}\n{entry.modified_at}"
                )
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

    def _make_attach_file_tool(
        self,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> FunctionTool:
        outer = self

        @tool(
            name="attach_file",
            description="attach a room file path or URL to the current thread so the user can see it",
        )
        async def attach_file(context: ToolContext, path: str) -> None:
            if thread_id is None or thread_id.strip() == "":
                raise RoomException("chat tool requires a non-empty thread_id")
            resolved_thread_id = thread_id.strip()
            if turn_id is not None and turn_id.strip() != "":
                resolved_turn_id = turn_id.strip()
            else:
                active_turn_id = outer._active_turn_id(thread_id=resolved_thread_id)
                if active_turn_id is None:
                    raise RoomException("attach_file requires an active turn")
                resolved_turn_id = active_turn_id

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
            for payload in (
                AgentFileContentStarted(
                    type=AGENT_EVENT_FILE_CONTENT_STARTED,
                    thread_id=resolved_thread_id,
                    turn_id=resolved_turn_id,
                    item_id=item_id,
                ),
                AgentFileContentDelta(
                    type=AGENT_EVENT_FILE_CONTENT_DELTA,
                    thread_id=resolved_thread_id,
                    turn_id=resolved_turn_id,
                    item_id=item_id,
                    url=normalized_url,
                ),
                AgentFileContentEnded(
                    type=AGENT_EVENT_FILE_CONTENT_ENDED,
                    thread_id=resolved_thread_id,
                    turn_id=resolved_turn_id,
                    item_id=item_id,
                ),
            ):
                message = Message(sender=sender, source=outer, data=payload)
                outer._publish_agent_event_to_open_participants(message=message)
                outer.emit(sender=sender, payload=payload)

        return attach_file

    def _build_chat_tools(
        self,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> list[FunctionTool]:
        return [
            self._make_new_thread_tool(),
            self._make_attach_file_tool(thread_id=thread_id, turn_id=turn_id),
            *self._build_thread_list_tools(),
        ]

    def make_toolkit(self) -> Toolkit:
        local_name = self._local_participant_name()
        return Toolkit(
            name="chat",
            description=f"tools for interacting with {local_name}",
            public=False,
            tools=[
                self._make_new_thread_tool(),
                *self._build_thread_list_tools(),
            ],
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
        if len(participant_ids) == 0 and not self._thread_has_active_turn(
            thread_id=thread_id
        ):
            self._open_participant_ids_by_thread.pop(thread_id, None)

    def _remove_participant_from_all_open_threads(self, *, participant_id: str) -> None:
        empty_thread_ids: list[str] = []
        for thread_id, participant_ids in self._open_participant_ids_by_thread.items():
            participant_ids.discard(participant_id)
            if len(participant_ids) == 0 and not self._thread_has_active_turn(
                thread_id=thread_id
            ):
                empty_thread_ids.append(thread_id)

        for thread_id in empty_thread_ids:
            self._open_participant_ids_by_thread.pop(thread_id, None)

    def _emit_participant_connected(self, *, participant: Participant) -> None:
        logger.info(
            "agent chat participant connected: participant_id=%s participant_name=%s",
            participant.id,
            self._participant_log_name(participant),
        )
        self.emit(
            sender=participant,
            payload=ParticipantConnect(
                type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                participant_id=participant.id,
            ),
        )

    def _emit_participant_disconnected(self, *, participant: Participant) -> None:
        logger.info(
            "agent chat participant disconnected: participant_id=%s participant_name=%s",
            participant.id,
            self._participant_log_name(participant),
        )
        self.emit(
            sender=participant,
            payload=ParticipantDisconnect(
                type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                participant_id=participant.id,
            ),
        )

    @staticmethod
    def _participant_log_name(participant: Participant) -> str | None:
        name = participant.get_attribute("name")
        if not isinstance(name, str):
            return None
        name = name.strip()
        if name == "":
            return None
        return name

    def _open_participants(self, *, thread_id: str) -> list[Participant]:
        del thread_id
        raise NotImplementedError

    def _clear_tracked_thread_state(self, *, thread_id: str) -> None:
        self._active_turn_ids_by_thread.pop(thread_id, None)
        self._turn_input_payloads_by_message_id = {
            message_id: payload
            for message_id, payload in self._turn_input_payloads_by_message_id.items()
            if payload.get("thread_id") != thread_id
        }

    def _thread_has_active_turn(self, *, thread_id: str) -> bool:
        return thread_id in self._active_turn_ids_by_thread

    def _drop_empty_open_thread_if_idle(self, *, thread_id: str) -> None:
        if self._thread_has_active_turn(thread_id=thread_id):
            return
        participant_ids = self._open_participant_ids_by_thread.get(thread_id)
        if participant_ids is not None and len(participant_ids) == 0:
            self._open_participant_ids_by_thread.pop(thread_id, None)


class MessagingChatChannel(BaseChatChannel):
    async def on_start(self) -> None:
        await super().on_start()
        self._room.messaging.on("message", self._on_room_message)
        self._room.messaging.on("participant_added", self._on_participant_added)
        self._room.messaging.on("participant_removed", self._on_participant_removed)
        self._room.messaging.on("messaging_enabled", self._on_messaging_enabled)
        if not self._room.messaging.is_enabled:
            await self._room.messaging.enable()
        else:
            self._connect_current_messaging_participants()

    async def on_stop(self) -> None:
        self._disconnect_current_messaging_participants()
        self._room.messaging.off("messaging_enabled", self._on_messaging_enabled)
        self._room.messaging.off("participant_removed", self._on_participant_removed)
        self._room.messaging.off("participant_added", self._on_participant_added)
        self._room.messaging.off("message", self._on_room_message)
        await super().on_stop()

    def _connect_current_messaging_participants(self) -> None:
        for participant in self._room.messaging.get_participants():
            self._emit_participant_connected(participant=participant)

    def _disconnect_current_messaging_participants(self) -> None:
        for participant in self._room.messaging.get_participants():
            self._remove_participant_from_all_open_threads(
                participant_id=participant.id
            )
            self._emit_participant_disconnected(participant=participant)

    def _on_messaging_enabled(self) -> None:
        self._connect_current_messaging_participants()

    def _on_participant_added(self, *, participant: Participant) -> None:
        self._emit_participant_connected(participant=participant)

    def _on_participant_removed(self, *, participant: Participant) -> None:
        self._remove_participant_from_all_open_threads(participant_id=participant.id)
        self._emit_participant_disconnected(participant=participant)

    def _send_agent_payload_nowait(
        self,
        *,
        participant: Participant,
        payload: dict[str, Any],
        attachment: bytes | None = None,
    ) -> None:
        try:
            self.room.messaging.send_message_nowait(
                to=participant,
                type="agent-message",
                message=payload,
                attachment=attachment,
            )
        except Exception:
            logger.debug(
                "failed to send agent message to participant %s",
                participant.id,
                exc_info=True,
            )

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
            raw_payload = self._payload_from_agent_room_message(message=message)
            if raw_payload is None:
                raise ValueError("agent-message payload must be a JSON object")
            if self._handle_agent_control_payload(
                payload=raw_payload,
                sender=sender,
            ):
                return
            agent_message = self._agent_message_from_room_message(message=message)
        except (ValidationError, ValueError):
            logger.exception(
                "unable to translate chat room message of type %s",
                message.type,
            )
            return

        self._on_agent_message(agent_message=agent_message, sender=sender)

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

        if len(participant_ids) == 0 and not self._thread_has_active_turn(
            thread_id=thread_id
        ):
            self._open_participant_ids_by_thread.pop(thread_id, None)

        return online_participants


@dataclass(slots=True)
class _WebSocketChatConnection:
    participant: Participant
    websocket: web.WebSocketResponse


class WebSocketChatChannel(BaseChatChannel):
    def __init__(
        self,
        *,
        room: RoomClient,
        authorize: WebSocketAuthorizeHook | None = None,
        encoding: WebSocketChatEncoding | None = None,
        protocols: tuple[str, ...] = (),
        on_participant_counts_changed: Callable[[dict[str, int]], None] | None = None,
        heartbeat: float | None = 30.0,
        receive_timeout: float | None = None,
        max_msg_size: int = DEFAULT_WEBSOCKET_MAX_MSG_SIZE,
        compress: bool | int = True,
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
            empty_state_title=empty_state_title,
        )
        self._authorize = authorize
        self._encoding = encoding or MsgpackWebSocketChatEncoding()
        self._protocols = protocols
        self._on_participant_counts_changed = on_participant_counts_changed
        self._heartbeat = heartbeat
        self._receive_timeout = receive_timeout
        self._max_msg_size = max_msg_size
        self._compress = compress
        self._connections_by_participant_id: dict[
            str, list[_WebSocketChatConnection]
        ] = {}
        self._participants_by_id: dict[str, Participant] = {}
        self._send_tasks: set[asyncio.Task[None]] = set()
        self._send_tasks_by_participant_id: dict[str, asyncio.Task[None]] = {}

    async def on_stop(self) -> None:
        connections = [
            connection
            for connections in self._connections_by_participant_id.values()
            for connection in connections
        ]
        for connection in connections:
            await connection.websocket.close()
        if len(self._send_tasks) > 0:
            await asyncio.gather(*self._send_tasks, return_exceptions=True)
        self._connections_by_participant_id.clear()
        self._participants_by_id.clear()
        self._send_tasks_by_participant_id.clear()
        await super().on_stop()

    def _response_protocols(self, request: web.Request) -> tuple[str, ...]:
        protocols = [*self._protocols]
        requested_protocols = request.headers.get("Sec-WebSocket-Protocol")
        if requested_protocols is None:
            return tuple(protocols)

        for protocol in requested_protocols.split(","):
            normalized = protocol.strip()
            if normalized[:16].casefold() == "meshagent-token.":
                protocols.append(normalized)
            elif normalized[:7].casefold() == "bearer.":
                protocols.append(normalized)

        return tuple(protocols)

    def participant_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for participant in self._participants_by_id.values():
            role = participant.get_attribute("role")
            if not isinstance(role, str) or role.strip() == "":
                role = "user"
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _emit_participant_counts_changed(self) -> None:
        on_participant_counts_changed = self._on_participant_counts_changed
        if on_participant_counts_changed is None:
            return
        on_participant_counts_changed(self.participant_counts())

    @staticmethod
    def _participant_from_request_headers(request: web.Request) -> Participant | None:
        participant_id = request.headers.get("X-Meshagent-Participant-Id")
        participant_name = request.headers.get("X-Meshagent-Participant-Name")
        if participant_id is None or participant_id.strip() == "":
            participant_id = participant_name
        if participant_id is None or participant_id.strip() == "":
            return None

        attributes: dict[str, Any] = {}
        if participant_name is not None and participant_name.strip() != "":
            attributes["name"] = participant_name.strip()
        return Participant(id=participant_id.strip(), attributes=attributes)

    async def _authorize_request(
        self,
        request: web.Request,
    ) -> Participant | web.StreamResponse | None:
        if self._authorize is None:
            return self._participant_from_request_headers(request)

        result = self._authorize(request)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def websocket_handler(self, request: web.Request) -> web.StreamResponse:
        auth_result = await self._authorize_request(request)
        if auth_result is None:
            raise web.HTTPUnauthorized(text="websocket participant is not authorized")
        if isinstance(auth_result, web.StreamResponse):
            return auth_result
        if self.supervisor is None or self.state != "started":
            raise web.HTTPBadRequest(reason="chat channel is not started")

        websocket = web.WebSocketResponse(
            heartbeat=self._heartbeat,
            protocols=self._response_protocols(request),
            receive_timeout=self._receive_timeout,
            max_msg_size=self._max_msg_size,
            compress=self._compress,
        )
        await websocket.prepare(request)

        participant = self._participant_for_websocket_connection(auth_result)
        connection = _WebSocketChatConnection(
            participant=participant,
            websocket=websocket,
        )
        self._register_connection(connection=connection)
        try:
            async for message in websocket:
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                }:
                    break
                if message.type == aiohttp.WSMsgType.ERROR:
                    logger.debug(
                        "websocket chat connection failed for participant %s",
                        participant.id,
                        exc_info=websocket.exception(),
                    )
                    break
                try:
                    agent_message = self._encoding.decode(message)
                    self._handle_websocket_agent_message(
                        agent_message=agent_message,
                        sender=participant,
                    )
                except (ValidationError, ValueError):
                    logger.exception(
                        "unable to translate websocket message from participant %s",
                        participant.id,
                    )
                    await websocket.close(
                        code=aiohttp.WSCloseCode.UNSUPPORTED_DATA,
                        message=b"invalid agent message",
                    )
                    break
        finally:
            self._unregister_connection(connection=connection)

        return websocket

    @staticmethod
    def _participant_for_websocket_connection(participant: Participant) -> Participant:
        connection_id = uuid.uuid4().hex
        attributes = participant.attributes
        attributes["base_participant_id"] = participant.id
        attributes["websocket_connection_id"] = connection_id
        return Participant(
            id=f"{participant.id}:websocket:{connection_id}",
            attributes=attributes,
        )

    def _handle_websocket_agent_message(
        self,
        *,
        agent_message: AgentMessage,
        sender: Participant,
    ) -> None:
        if isinstance(agent_message, AgentThreadMessage):
            if agent_message.type in self._THREAD_INDEX_BUMP_AGENT_MESSAGE_TYPES:
                self.bump_thread(path=agent_message.thread_id)

        payload = agent_message.model_dump(mode="python", exclude_none=True)
        if self._handle_agent_control_payload(payload=payload, sender=sender):
            return
        self._on_agent_message(agent_message=agent_message, sender=sender)

    def _register_connection(self, *, connection: _WebSocketChatConnection) -> None:
        first_connection = (
            connection.participant.id not in self._connections_by_participant_id
        )
        self._participants_by_id[connection.participant.id] = connection.participant
        connections = self._connections_by_participant_id.setdefault(
            connection.participant.id,
            [],
        )
        connections.append(connection)
        if first_connection:
            self._emit_participant_connected(participant=connection.participant)
            self._emit_participant_counts_changed()

    def _unregister_connection(self, *, connection: _WebSocketChatConnection) -> None:
        connections = self._connections_by_participant_id.get(connection.participant.id)
        if connections is None:
            return
        with contextlib.suppress(ValueError):
            connections.remove(connection)
        if len(connections) == 0:
            self._connections_by_participant_id.pop(connection.participant.id, None)
            self._participants_by_id.pop(connection.participant.id, None)
            self._remove_participant_from_all_open_threads(
                participant_id=connection.participant.id
            )
            self._emit_participant_disconnected(participant=connection.participant)
            self._emit_participant_counts_changed()

    def _send_agent_payload_nowait(
        self,
        *,
        participant: Participant,
        payload: dict[str, Any],
        attachment: bytes | None = None,
    ) -> None:
        self._send_agent_payload_task(
            participant=participant,
            payload=payload,
            attachment=attachment,
        )

    def _send_agent_payload_task(
        self,
        *,
        participant: Participant,
        payload: dict[str, Any],
        attachment: bytes | None = None,
    ) -> asyncio.Task[None] | None:
        if attachment is not None:
            payload = {**payload, "data": attachment}
        try:
            agent_message = parse_agent_message(payload)
        except (ValidationError, ValueError):
            logger.debug(
                "failed to encode agent message for websocket participant %s",
                participant.id,
                exc_info=True,
            )
            return None

        participant_id = participant.id
        previous_task = self._send_tasks_by_participant_id.get(participant_id)
        task = asyncio.create_task(
            self._send_agent_message_to_websockets_after(
                previous_task=previous_task,
                participant_id=participant.id,
                message=agent_message,
            )
        )
        self._send_tasks.add(task)
        self._send_tasks_by_participant_id[participant_id] = task

        def cleanup(completed: asyncio.Task[None]) -> None:
            self._send_tasks.discard(completed)
            if self._send_tasks_by_participant_id.get(participant_id) is completed:
                self._send_tasks_by_participant_id.pop(participant_id, None)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "failed to send websocket agent message task to participant %s",
                    participant_id,
                    exc_info=True,
                )

        task.add_done_callback(cleanup)
        return task

    async def send_agent_message_to_participant_and_wait(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        task = self._send_agent_payload_task(
            participant=participant,
            payload=payload.model_dump(mode="json"),
        )
        if task is None:
            return False
        await task
        return True

    async def _send_agent_message_to_websockets_after(
        self,
        *,
        previous_task: asyncio.Task[None] | None,
        participant_id: str,
        message: AgentMessage,
    ) -> None:
        if previous_task is not None:
            await asyncio.gather(previous_task, return_exceptions=True)
        await self._send_agent_message_to_websockets(
            participant_id=participant_id,
            message=message,
        )

    async def _send_agent_message_to_websockets(
        self,
        *,
        participant_id: str,
        message: AgentMessage,
    ) -> None:
        connections = [
            *self._connections_by_participant_id.get(participant_id, []),
        ]
        if len(connections) == 0:
            return

        encoded = self._encoding.encode(message)
        encoded_size = len(encoded)
        message_type = message.type
        thread_id = (
            message.thread_id if isinstance(message, AgentThreadMessage) else None
        )
        if (
            message_type in _IMAGE_GENERATION_AGENT_EVENT_TYPES
            or encoded_size >= _LARGE_AGENT_WEBSOCKET_MESSAGE_LOG_BYTES
        ):
            logger.info(
                "sending websocket agent message: participant=%s message_type=%s "
                "thread_id=%s encoded_size=%s max_msg_size=%s connection_count=%s",
                participant_id,
                message_type,
                thread_id,
                encoded_size,
                self._max_msg_size,
                len(connections),
            )
        for connection in connections:
            if connection.websocket.closed:
                logger.warning(
                    "skipping websocket agent message send to closed connection: "
                    "participant=%s message_type=%s thread_id=%s "
                    "encoded_size=%s close_code=%s",
                    participant_id,
                    message_type,
                    thread_id,
                    encoded_size,
                    connection.websocket.close_code,
                )
                continue
            try:
                if isinstance(encoded, bytes):
                    await connection.websocket.send_bytes(encoded)
                else:
                    await connection.websocket.send_str(encoded)
            except Exception:
                logger.warning(
                    "failed to send websocket agent message to participant %s: "
                    "message_type=%s thread_id=%s encoded_size=%s "
                    "close_code=%s",
                    participant_id,
                    message_type,
                    thread_id,
                    encoded_size,
                    connection.websocket.close_code,
                    exc_info=True,
                )

    def _open_participants(self, *, thread_id: str) -> list[Participant]:
        participant_ids = self._open_participant_ids_by_thread.get(thread_id)
        if participant_ids is None:
            return []

        participants: list[Participant] = []
        stale_participant_ids: list[str] = []
        for participant_id in participant_ids:
            if participant_id not in self._connections_by_participant_id:
                stale_participant_ids.append(participant_id)
                continue
            participant = self._participants_by_id.get(participant_id)
            if participant is not None:
                participants.append(participant)

        for participant_id in stale_participant_ids:
            participant_ids.discard(participant_id)

        if len(participant_ids) == 0 and not self._thread_has_active_turn(
            thread_id=thread_id
        ):
            self._open_participant_ids_by_thread.pop(thread_id, None)

        return participants
