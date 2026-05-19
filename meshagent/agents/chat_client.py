from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import aiohttp

from meshagent.api import Participant, RemoteParticipant, RoomClient, RoomException
from meshagent.api.http import new_client_session
from meshagent.api.messaging import ErrorContent, JsonContent, ensure_content
from meshagent.tools import FunctionTool, ToolContext, Toolkit

from .chat_channel import (
    DEFAULT_WEBSOCKET_MAX_MSG_SIZE,
    MsgpackWebSocketChatEncoding,
    WebSocketChatEncoding,
)
from .messages import (
    AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED,
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_CONNECTION_STATUS,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_THREAD_CREATED,
    AGENT_EVENT_THREAD_DELETED,
    AGENT_EVENT_THREAD_LISTED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_THREAD_UPDATED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
    AGENT_MESSAGE_MODEL_CHANGE,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_MODELS_RESPONSE,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_DELETE,
    AGENT_MESSAGE_THREAD_LIST,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_RENAME,
    AgentAudioGenerationDelta,
    AgentAudioTranscriptionDelta,
    AgentClientToolCallRequested,
    AgentClientToolCallResponse,
    AgentFileContent,
    AgentFileContentDelta,
    AgentImageGenerationCompleted,
    AgentImageGenerationPartial,
    AgentMessage,
    AgentConnectionStatus,
    AgentThreadStatus,
    AgentModelInfo,
    AgentModelChanged,
    AgentReasoningContentDelta,
    AgentTextContent,
    AgentTextContentDelta,
    AgentToolCallArgumentsDelta,
    AgentToolCallLogDelta,
    ChangeModel,
    ClientToolkitDescription,
    CloseThread,
    DeleteThread,
    ListThreads,
    ModelsRequest,
    ModelsResponse,
    OpenThread,
    RenameThread,
    StartThread,
    ThreadsListed,
    TurnEnded,
    ThreadStarted,
    TurnStart,
    TurnStartAccepted,
    TurnSteer,
    TurnSteerAccepted,
    parse_agent_message,
)
from .process import Message


@dataclass(frozen=True, slots=True)
class PendingAgentInput:
    message_id: str
    role: str
    text: str
    applied: bool = False

    @property
    def label(self) -> str:
        prefix = "" if self.role == "" else f"{self.role}: "
        return f"{prefix}{self.text}".strip()


QueuedAgentInput = PendingAgentInput


@dataclass(frozen=True, slots=True)
class AcceptedAgentInput:
    message_id: str
    role: str
    text: str


def _normalized_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized == "":
        return None
    return normalized


def _agent_input_text_from_payload(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    attachment_count = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip() != "":
                text_parts.append(text.strip())
        elif item_type in ("file", "image"):
            attachment_count += 1

    if attachment_count > 0:
        noun = "attachment" if attachment_count == 1 else "attachments"
        text_parts.append(f"{attachment_count} {noun}")

    return " ".join(text_parts).strip()


def _agent_input_content_text(
    content: list[AgentTextContent | AgentFileContent],
) -> str:
    text_parts: list[str] = []
    for item in content:
        if isinstance(item, AgentTextContent) and item.text.strip() != "":
            text_parts.append(item.text)
            continue
        if isinstance(item, AgentFileContent) and item.url.strip() != "":
            text_parts.append(f"[attachment] {item.url}")

    return "\n\n".join(text_parts).strip()


def _pending_agent_message_label(payload: dict[str, Any]) -> str | None:
    text = _agent_input_text_from_payload(payload)
    sender_name = payload.get("sender_name")
    prefix = ""
    if isinstance(sender_name, str) and sender_name.strip() != "":
        prefix = f"{sender_name.strip()}: "
    label = f"{prefix}{text}".strip()
    if label == "":
        return None
    return label


def _thread_status_text(status: object) -> str | None:
    if not isinstance(status, str):
        return None
    normalized = status.strip()
    if normalized == "":
        return None
    return normalized


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        return


class BaseChatClient(ABC):
    def __init__(self, *, timeout: float = 30) -> None:
        self._timeout = timeout
        self._thread_sessions: dict[str, ChatThreadSession] = {}
        self._pending_thread_sessions: set[ChatThreadSession] = set()
        self._connection_status: AgentConnectionStatus | None = None
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []

    async def __aenter__(self) -> BaseChatClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        del exc_type, exc, exc_tb
        await self.stop()

    async def start(self) -> None:
        await self._start_transport()

    async def close(self) -> None:
        await self.stop()

    async def stop(self) -> None:
        sessions = list(self._thread_sessions.values())
        sessions.extend(self._pending_thread_sessions)
        for session in sessions:
            await session.close(close_client=False)
        self._thread_sessions.clear()
        self._pending_thread_sessions.clear()
        await self._stop_transport()

    @abstractmethod
    async def _start_transport(self) -> None: ...

    @abstractmethod
    async def _stop_transport(self) -> None: ...

    @abstractmethod
    async def _send_agent_message(self, payload: AgentMessage) -> None: ...

    def _create_thread_session(
        self,
        *,
        thread_path: str | None = None,
        local_participant_name: str | None = None,
        close_client_on_close: bool = False,
    ) -> ChatThreadSession:
        session = ChatThreadSession(
            client=self,
            thread_path=thread_path,
            local_participant_name=local_participant_name,
            close_client_on_close=close_client_on_close,
            timeout=self._timeout,
        )
        return session

    async def open_thread(
        self,
        thread_path: str,
        *,
        local_participant_name: str | None = None,
        close_client_on_close: bool = False,
        load: bool | None = None,
        since_turn: str | None = None,
    ) -> ChatThreadSession:
        session = self._create_thread_session(
            thread_path=thread_path,
            local_participant_name=local_participant_name,
            close_client_on_close=close_client_on_close,
        )
        await session.open(load=load, since_turn=since_turn)
        return session

    async def start_thread(
        self,
        payload: StartThread,
        *,
        local_participant_name: str | None = None,
        close_client_on_close: bool = False,
        on_pending_session: Callable[[ChatThreadSession], None] | None = None,
        client_toolkits: list[Toolkit] | None = None,
    ) -> ChatThreadSession:
        session = self._create_thread_session(
            local_participant_name=local_participant_name,
            close_client_on_close=close_client_on_close,
        )
        if client_toolkits is not None:
            payload = payload.model_copy(
                update={
                    "client_toolkits": session.register_client_toolkits(client_toolkits)
                }
            )
        await session.send(payload)
        if on_pending_session is not None:
            on_pending_session(session)
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    event = await session.receive()
                    if event.get("type") != AGENT_EVENT_THREAD_STARTED:
                        continue
                    thread_started = ThreadStarted.model_validate(event)
                    if thread_started.source_message_id != payload.message_id:
                        continue
                    return session
        except asyncio.TimeoutError as exc:
            raise RoomException("timed out waiting for thread to start") from exc

    async def send(self, payload: AgentMessage) -> None:
        await self._send_agent_message(payload)

    def add_event_listener(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> Callable[[], None]:
        self._event_listeners.append(callback)

        def _unsubscribe() -> None:
            try:
                self._event_listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    @property
    def connection_status(self) -> AgentConnectionStatus | None:
        return self._connection_status

    def _emit_connection_status(
        self,
        *,
        status: str,
        message: str | None = None,
        reason: str | None = None,
        retry_in_seconds: float | None = None,
    ) -> None:
        payload = AgentConnectionStatus(
            type=AGENT_EVENT_CONNECTION_STATUS,
            status=status,
            message=message,
            reason=reason,
            retry_in_seconds=retry_in_seconds,
        )
        self._connection_status = payload
        payload_json = payload.model_dump(mode="json", exclude_none=True)
        for session in self._all_sessions():
            session._handle_agent_payload(payload_json)

    def _register_thread_session(self, session: ChatThreadSession) -> None:
        self._pending_thread_sessions.discard(session)
        self._thread_sessions[session.thread_path] = session

    def _unregister_thread_session(self, session: ChatThreadSession) -> None:
        self._pending_thread_sessions.discard(session)
        if session.has_thread_path:
            existing = self._thread_sessions.get(session.thread_path)
            if existing is session:
                self._thread_sessions.pop(session.thread_path, None)

    def _handle_agent_payload(self, payload: dict[str, Any]) -> None:
        payload_type = payload.get("type")
        if payload_type in (
            AGENT_EVENT_THREAD_STARTED,
            AGENT_MESSAGE_MODELS_RESPONSE,
            AGENT_EVENT_THREAD_LISTED,
        ):
            for session in self._all_sessions():
                if session._handles_threadless_payload(payload):
                    session._handle_agent_payload(payload)
                    return
        if payload_type in (
            AGENT_EVENT_THREAD_CREATED,
            AGENT_EVENT_THREAD_UPDATED,
            AGENT_EVENT_THREAD_DELETED,
        ):
            self._emit_event(payload)
            for session in self._all_sessions():
                session._handle_agent_payload(payload)
            return

        thread_id = payload.get("thread_id")
        if isinstance(thread_id, str):
            session = self._thread_sessions.get(thread_id)
            if session is not None:
                session._handle_agent_payload(payload)
            return

    def _all_sessions(self) -> tuple[ChatThreadSession, ...]:
        return (
            *self._thread_sessions.values(),
            *self._pending_thread_sessions,
        )

    def _emit_event(self, payload: dict[str, Any]) -> None:
        for callback in tuple(self._event_listeners):
            callback(payload)


class ChatThreadSession:
    def __init__(
        self,
        *,
        client: BaseChatClient,
        thread_path: str | None,
        local_participant_name: str | None = None,
        close_client_on_close: bool = False,
        timeout: float = 30,
    ) -> None:
        self._client = client
        self._thread_path = _normalized_string(thread_path)
        self._local_participant_name = _normalized_string(local_participant_name)
        self._close_client_on_close = close_client_on_close
        self._timeout = timeout
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._thread_status_text: str | None = None
        self._thread_status: AgentThreadStatus | None = None
        self._pending_inputs: dict[str, PendingAgentInput] = {}
        self._messages: list[AgentMessage] = []
        self._message_indexes: dict[str, int] = {}
        self._accepted_input_callback: Callable[[AcceptedAgentInput], None] | None = (
            None
        )
        self._local_agent_message_ids: set[str] = set()
        self._pending_local_input_message_ids: set[str] = set()
        self._local_turn_ids: set[str] = set()
        self._remote_source_message_ids: set[str] = set()
        self._remote_turn_output_parts: dict[str, list[str]] = {}
        self._last_completed_turn_id: str | None = None
        self._current_model: AgentModelChanged | None = None
        self._models_response: ModelsResponse | None = None
        self._client_toolkits_by_tool_name: dict[str, Toolkit] = {}
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._closed = False
        if self.has_thread_path:
            self._client._register_thread_session(self)
        else:
            self._client._pending_thread_sessions.add(self)

    async def __aenter__(self) -> ChatThreadSession:
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        del exc_type, exc, exc_tb
        await self.close()

    @property
    def client(self) -> BaseChatClient:
        return self._client

    @property
    def has_thread_path(self) -> bool:
        return self._thread_path is not None

    @property
    def thread_path(self) -> str:
        if self._thread_path is None:
            raise RoomException("chat thread session not started")
        return self._thread_path

    @property
    def thread_status_text(self) -> str | None:
        return self._thread_status_text

    @property
    def thread_status(self) -> AgentThreadStatus | None:
        return self._thread_status

    @property
    def current_model(self) -> AgentModelChanged | None:
        return self._current_model

    @property
    def models_response(self) -> ModelsResponse | None:
        return self._models_response

    @property
    def local_participant_name(self) -> str | None:
        return self._local_participant_name

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def last_completed_turn_id(self) -> str | None:
        return self._last_completed_turn_id

    @property
    def pending_inputs(self) -> tuple[PendingAgentInput, ...]:
        return tuple(item for item in self._pending_inputs.values() if not item.applied)

    @property
    def queued_message_labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.pending_inputs)

    def set_accepted_input_callback(
        self, callback: Callable[[AcceptedAgentInput], None] | None
    ) -> None:
        self._accepted_input_callback = callback

    def add_event_listener(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> Callable[[], None]:
        self._event_listeners.append(callback)

        def _unsubscribe() -> None:
            try:
                self._event_listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def register_client_toolkits(
        self, client_toolkits: list[Toolkit]
    ) -> list[ClientToolkitDescription]:
        descriptions: list[ClientToolkitDescription] = []
        for toolkit in client_toolkits:
            for tool in toolkit.get_tools():
                if not isinstance(tool, FunctionTool):
                    raise RoomException(
                        "client toolkits only support FunctionTool tools"
                    )
                input_schema = tool.input_schema
                if input_schema is None:
                    raise RoomException(
                        f"client tool '{tool.name}' is missing required input schema"
                    )
                if tool.name in self._client_toolkits_by_tool_name:
                    raise RoomException(
                        f"client tool '{tool.name}' has already been registered"
                    )
                self._client_toolkits_by_tool_name[tool.name] = toolkit
                descriptions.append(
                    ClientToolkitDescription(
                        name=tool.name,
                        title=tool.title,
                        description=tool.description,
                        input_schema=input_schema,
                    )
                )
        return descriptions

    async def open(
        self,
        *,
        load: bool | None = None,
        since_turn: str | None = None,
    ) -> None:
        await self.send(
            OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id=self.thread_path,
                load=load,
                since_turn=since_turn,
            )
        )

    async def delete_thread(self, thread_path: str) -> None:
        await self.send(
            DeleteThread(
                type=AGENT_MESSAGE_THREAD_DELETE,
                thread_id=thread_path,
            )
        )

    async def rename_thread(self, thread_path: str, name: str) -> None:
        await self.send(
            RenameThread(
                type=AGENT_MESSAGE_THREAD_RENAME,
                thread_id=thread_path,
                name=name,
            )
        )

    async def list_threads(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ThreadsListed:
        payload = ListThreads(
            type=AGENT_MESSAGE_THREAD_LIST,
            limit=limit,
            offset=offset,
        )
        await self.send(payload)
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    event = await self.receive()
                    if event.get("type") != AGENT_EVENT_THREAD_LISTED:
                        continue
                    response = ThreadsListed.model_validate(event)
                    if response.source_message_id != payload.message_id:
                        continue
                    return response
        except asyncio.TimeoutError as exc:
            raise RoomException("timed out waiting for thread list") from exc

    async def close(self, *, close_client: bool | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._thread_path is not None:
                await self.send(
                    CloseThread(
                        type=AGENT_MESSAGE_THREAD_CLOSE,
                        thread_id=self.thread_path,
                    )
                )
        finally:
            self._client._unregister_thread_session(self)
            should_close_client = (
                self._close_client_on_close if close_client is None else close_client
            )
            if should_close_client:
                await self._client.stop()

    async def send(self, payload: AgentMessage) -> None:
        payload_json = payload.model_dump(mode="json")
        message_id = payload_json.get("message_id")
        if isinstance(message_id, str) and message_id.strip() != "":
            normalized_message_id = message_id.strip()
            self._local_agent_message_ids.add(normalized_message_id)
            if isinstance(payload, (StartThread, TurnStart, TurnSteer)):
                self._pending_local_input_message_ids.add(normalized_message_id)
        if isinstance(payload, (StartThread, TurnStart)):
            self.add_agent_message(payload)
        await self._client.send(payload)

    def add_agent_message(self, message: AgentMessage) -> None:
        if isinstance(message, TurnEnded):
            turn_id = _normalized_string(message.turn_id)
            if turn_id is not None:
                self._last_completed_turn_id = turn_id
        if isinstance(message, (StartThread, TurnStart, TurnSteer)):
            if _agent_input_content_text(message.content or []).strip() == "":
                return
            self._append_message(message)
            return
        if isinstance(message, (TurnStartAccepted, TurnSteerAccepted)):
            if self._is_local_source_message(message.source_message_id):
                self._pending_local_input_message_ids.discard(
                    message.source_message_id.strip()
                )
                return
            if _agent_input_content_text(message.content).strip() == "":
                return
            self._append_message(message, before_pending_local_inputs=True)
            return
        if isinstance(
            message,
            (
                AgentAudioGenerationDelta,
                AgentAudioTranscriptionDelta,
                AgentFileContentDelta,
                AgentReasoningContentDelta,
                AgentTextContentDelta,
                AgentToolCallArgumentsDelta,
                AgentToolCallLogDelta,
            ),
        ):
            self._append_or_merge_delta(message)
            return
        if isinstance(
            message,
            (
                AgentImageGenerationCompleted,
                AgentImageGenerationPartial,
            ),
        ):
            self._upsert_item_message(
                key=f"image_generation:{message.item_id}",
                message=message,
            )

    def _append_message(
        self,
        message: AgentMessage,
        *,
        before_pending_local_inputs: bool = False,
    ) -> None:
        normalized_message_id = _normalized_string(message.message_id)
        if normalized_message_id is None:
            return
        if normalized_message_id in self._message_indexes:
            return
        if before_pending_local_inputs:
            for index, existing in enumerate(self._messages):
                if existing.message_id in self._pending_local_input_message_ids:
                    self._messages.insert(index, message)
                    self._index_messages_from(index)
                    return
        self._message_indexes[normalized_message_id] = len(self._messages)
        self._messages.append(message)

    def _upsert_item_message(self, *, key: str, message: AgentMessage) -> None:
        existing_index = self._message_indexes.get(key)
        if existing_index is None:
            self._message_indexes[key] = len(self._messages)
            self._messages.append(message)
            return
        self._messages[existing_index] = message

    def _index_messages_from(self, start: int) -> None:
        for index in range(max(0, start), len(self._messages)):
            message = self._messages[index]
            key = self._message_index_key(message)
            if key is not None:
                self._message_indexes[key] = index

    @staticmethod
    def _message_index_key(message: AgentMessage) -> str | None:
        if isinstance(
            message,
            (
                AgentAudioGenerationDelta,
                AgentAudioTranscriptionDelta,
                AgentFileContentDelta,
                AgentReasoningContentDelta,
                AgentTextContentDelta,
                AgentToolCallArgumentsDelta,
                AgentToolCallLogDelta,
            ),
        ):
            return f"{message.type}:{message.item_id}"
        return _normalized_string(message.message_id)

    def _append_or_merge_delta(
        self,
        message: AgentAudioGenerationDelta
        | AgentAudioTranscriptionDelta
        | AgentFileContentDelta
        | AgentReasoningContentDelta
        | AgentTextContentDelta
        | AgentToolCallArgumentsDelta
        | AgentToolCallLogDelta,
    ) -> None:
        key = f"{message.type}:{message.item_id}"
        existing_index = self._message_indexes.get(key)
        if existing_index is None:
            self._message_indexes[key] = len(self._messages)
            self._messages.append(message)
            return

        existing = self._messages[existing_index]
        if isinstance(existing, AgentAudioGenerationDelta) and isinstance(
            message, AgentAudioGenerationDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"data": existing.data + message.data}
            )
        elif isinstance(existing, AgentAudioTranscriptionDelta) and isinstance(
            message, AgentAudioTranscriptionDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"text": existing.text + message.text}
            )
        elif isinstance(existing, AgentFileContentDelta) and isinstance(
            message, AgentFileContentDelta
        ):
            self._messages[existing_index] = message
        elif isinstance(existing, AgentReasoningContentDelta) and isinstance(
            message, AgentReasoningContentDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"text": existing.text + message.text}
            )
        elif isinstance(existing, AgentTextContentDelta) and isinstance(
            message, AgentTextContentDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"text": existing.text + message.text}
            )
        elif isinstance(existing, AgentToolCallArgumentsDelta) and isinstance(
            message, AgentToolCallArgumentsDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"delta": existing.delta + message.delta}
            )
        elif isinstance(existing, AgentToolCallLogDelta) and isinstance(
            message, AgentToolCallLogDelta
        ):
            self._messages[existing_index] = existing.model_copy(
                update={"lines": [*existing.lines, *message.lines]}
            )

    async def request_models(self) -> ModelsResponse:
        payload = ModelsRequest(
            type=AGENT_MESSAGE_MODELS_REQUEST,
        )
        await self.send(payload)
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    event = await self.receive()
                    if event.get("type") != AGENT_MESSAGE_MODELS_RESPONSE:
                        continue
                    response = ModelsResponse.model_validate(event)
                    if response.source_message_id != payload.message_id:
                        continue
                    self._apply_models_response(response)
                    return response
        except asyncio.TimeoutError as exc:
            raise RoomException("timed out waiting for model list") from exc

    def _apply_models_response(self, response: ModelsResponse) -> None:
        self._models_response = response
        if self._thread_path is None:
            return
        active_model = self._active_model_from_models_response(
            response,
            thread_id=self._thread_path,
        )
        if active_model is not None:
            self._current_model = active_model

    def apply_models_response(self, response: ModelsResponse) -> None:
        self._apply_models_response(response)

    @staticmethod
    def _active_model_from_models_response(
        response: ModelsResponse,
        *,
        thread_id: str,
    ) -> AgentModelChanged | None:
        for provider in response.providers:
            for model in provider.models:
                if not model.active:
                    continue
                return AgentModelChanged(
                    type=AGENT_EVENT_MODEL_CHANGED,
                    thread_id=thread_id,
                    source_message_id=response.source_message_id,
                    provider=provider.name,
                    model=model.name,
                    voice=model.default_output_voice,
                    input_format=model.input_format,
                    output_format=model.output_format,
                    turn_detection=model.turn_detection,
                    realtime_protocols=model.realtime_protocols,
                    supports_attachments=model.supports_attachments,
                    accepts=model.accepts,
                    output_modalities=ChatThreadSession._default_output_modalities(
                        model
                    ),
                )
        return None

    @staticmethod
    def _default_output_modalities(model: AgentModelInfo) -> list[str]:
        return [model.modalities[0]] if len(model.modalities) > 0 else ["text"]

    def select_model(self, model: AgentModelChanged) -> None:
        self._current_model = model

    async def change_model(
        self,
        *,
        provider: str | None,
        model: str | None,
        voice: str | None = None,
    ) -> AgentModelChanged:
        payload = ChangeModel(
            type=AGENT_MESSAGE_MODEL_CHANGE,
            thread_id=self.thread_path,
            provider=provider,
            model=model,
            voice=voice,
        )
        await self.send(payload)
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    event = await self.receive()
                    if event.get("type") != AGENT_EVENT_MODEL_CHANGED:
                        continue
                    changed = AgentModelChanged.model_validate(event)
                    if changed.source_message_id != payload.message_id:
                        continue
                    self._current_model = changed
                    return changed
        except asyncio.TimeoutError as exc:
            raise RoomException("timed out waiting for model change") from exc

    async def receive(self) -> dict[str, Any]:
        return await self._events.get()

    def _handles_threadless_payload(self, payload: dict[str, Any]) -> bool:
        payload_type = payload.get("type")
        source_message_id = payload.get("source_message_id")
        if payload_type in (
            AGENT_EVENT_THREAD_STARTED,
            AGENT_MESSAGE_MODELS_RESPONSE,
            AGENT_EVENT_THREAD_LISTED,
        ):
            return self._is_local_source_message(source_message_id)
        return False

    def _handle_agent_payload(self, payload: dict[str, Any]) -> None:
        payload_type = payload.get("type")
        if payload_type in (
            AGENT_EVENT_THREAD_CREATED,
            AGENT_EVENT_THREAD_UPDATED,
            AGENT_EVENT_THREAD_DELETED,
        ):
            self._emit_event(payload)
            return
        if payload_type == AGENT_EVENT_CONNECTION_STATUS:
            try:
                connection_status = AgentConnectionStatus.model_validate(payload)
            except Exception:
                return
            status = connection_status.status.strip().lower()
            if status in ("connected", "reconnected"):
                self._thread_status_text = None
                self._thread_status = None
            elif status in ("disconnected", "reconnecting"):
                self._thread_status_text = connection_status.message or (
                    "Reconnecting" if status == "reconnecting" else "Disconnected"
                )
                self._thread_status = None
            self.on_event(connection_status)
            self._events.put_nowait(payload)
            return
        if payload_type == AGENT_EVENT_THREAD_STARTED:
            if not self._is_local_source_message(payload.get("source_message_id")):
                return
            try:
                thread_started = ThreadStarted.model_validate(payload)
            except Exception:
                return
            self._thread_path = thread_started.thread_id
            self._client._register_thread_session(self)
            self._events.put_nowait(payload)
            task = asyncio.create_task(
                self.send(
                    OpenThread(
                        type=AGENT_MESSAGE_THREAD_OPEN,
                        thread_id=thread_started.thread_id,
                        load=False,
                        since_turn=None,
                    )
                )
            )
            task.add_done_callback(_consume_task_exception)
            return
        if payload_type == AGENT_MESSAGE_MODELS_RESPONSE:
            if not self._is_local_source_message(payload.get("source_message_id")):
                return
            self._events.put_nowait(payload)
            return
        if payload_type == AGENT_EVENT_THREAD_LISTED:
            if not self._is_local_source_message(payload.get("source_message_id")):
                return
            self._events.put_nowait(payload)
            return
        if self._thread_path is None:
            return
        if payload.get("thread_id") != self._thread_path:
            return
        if payload_type == AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED:
            task = asyncio.create_task(self._respond_to_client_tool_call(payload))
            task.add_done_callback(_consume_task_exception)
        try:
            agent_message = parse_agent_message(payload)
        except Exception:
            agent_message = None
        if agent_message is not None:
            self.add_agent_message(agent_message)
        if payload_type == AGENT_EVENT_THREAD_STATUS:
            try:
                thread_status = AgentThreadStatus.model_validate(payload)
            except Exception:
                thread_status = None
            self._thread_status = thread_status
            self._thread_status_text = (
                None
                if thread_status is None
                else _thread_status_text(thread_status.status)
            )
        elif payload_type == AGENT_EVENT_MODEL_CHANGED:
            try:
                self._current_model = AgentModelChanged.model_validate(payload)
            except Exception:
                return
        elif payload_type == AGENT_EVENT_TURN_START_ACCEPTED:
            if self._is_remote_agent_input(payload):
                self._track_accepted_input(payload)
            else:
                self._track_queued_agent_input(payload)
        elif payload_type == AGENT_EVENT_TURN_STEER_ACCEPTED:
            self._track_queued_agent_input(payload)
        elif payload_type == AGENT_EVENT_TURN_STARTED:
            self._mark_queued_agent_input_applied(payload.get("source_message_id"))
            self._track_local_turn_started(payload)
            self._track_remote_turn_started(payload)
        elif payload_type == AGENT_EVENT_TURN_STEERED:
            self._mark_queued_agent_input_applied(payload.get("source_message_id"))
        elif payload_type == AGENT_EVENT_TEXT_CONTENT_DELTA:
            self._track_remote_text_delta(payload)
        elif payload_type == AGENT_EVENT_TURN_STEER_REJECTED:
            self._clear_queued_agent_input(payload.get("source_message_id"))
        elif payload_type == AGENT_EVENT_TURN_START_REJECTED:
            self._clear_queued_agent_input(payload.get("source_message_id"))
        elif payload_type == AGENT_EVENT_TURN_ENDED:
            self._track_remote_turn_ended(payload)
            turn_id = _normalized_string(payload.get("turn_id"))
            if turn_id is not None:
                self._last_completed_turn_id = turn_id
            self._thread_status = None
            self._thread_status_text = None
            self.clear_applied_queued_agent_inputs()
        if agent_message is not None:
            self.on_event(agent_message)
        if self._should_enqueue_agent_event(payload):
            self._events.put_nowait(payload)
        if payload_type == AGENT_EVENT_TURN_ENDED:
            self._clear_local_turn(payload.get("turn_id"))

    def _emit_event(self, payload: dict[str, Any]) -> None:
        for callback in tuple(self._event_listeners):
            callback(payload)

    async def _respond_to_client_tool_call(self, payload: dict[str, Any]) -> None:
        try:
            request = AgentClientToolCallRequested.model_validate(payload)
        except Exception:
            return
        try:
            response = await self._invoke_client_tool(request)
        except Exception as exc:
            response = ErrorContent(text=str(exc))
        await self.send(
            AgentClientToolCallResponse(
                type=AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
                thread_id=self.thread_path,
                turn_id=request.turn_id,
                request_id=request.request_id,
                response=response,
            )
        )

    async def _invoke_client_tool(self, request: AgentClientToolCallRequested) -> Any:
        if request.toolkit != "client":
            raise RoomException(f"unsupported client toolkit proxy '{request.toolkit}'")
        toolkit = self._client_toolkits_by_tool_name.get(request.tool)
        if toolkit is None:
            raise RoomException(f"client tool '{request.tool}' is not registered")
        participant_name = self._local_participant_name or "client"
        response = await toolkit.invoke(
            context=ToolContext(
                caller=Participant(
                    id=participant_name,
                    attributes={"name": participant_name},
                )
            ),
            name=request.tool,
            input=JsonContent(json=request.arguments),
        )
        if isinstance(response, AsyncIterable):
            raise RoomException("client tools must return non-streaming responses")
        return ensure_content(response)

    def on_event(self, message: AgentMessage) -> None:
        del message

    def _is_local_source_message(self, source_message_id: object) -> bool:
        return (
            isinstance(source_message_id, str)
            and source_message_id.strip() in self._local_agent_message_ids
        )

    def _is_local_turn(self, turn_id: object) -> bool:
        return isinstance(turn_id, str) and turn_id.strip() in self._local_turn_ids

    def _clear_local_turn(self, turn_id: object) -> None:
        if not isinstance(turn_id, str):
            return
        self._local_turn_ids.discard(turn_id.strip())

    def _should_enqueue_agent_event(self, payload: dict[str, Any]) -> bool:
        payload_type = payload.get("type")
        if payload_type == AGENT_EVENT_THREAD_STATUS:
            return True
        if payload_type in (
            AGENT_EVENT_TURN_START_ACCEPTED,
            AGENT_EVENT_TURN_STEER_ACCEPTED,
            AGENT_EVENT_TURN_STEERED,
            AGENT_EVENT_TURN_STEER_REJECTED,
            AGENT_EVENT_TURN_START_REJECTED,
            AGENT_EVENT_TURN_STARTED,
        ):
            return self._is_local_source_message(payload.get("source_message_id"))
        if payload_type in (AGENT_EVENT_TEXT_CONTENT_DELTA, AGENT_EVENT_TURN_ENDED):
            return self._is_local_turn(payload.get("turn_id"))
        return True

    def _is_remote_agent_input(self, payload: dict[str, Any]) -> bool:
        source_message_id = payload.get("source_message_id")
        if self._is_local_source_message(source_message_id):
            return False

        return _agent_input_text_from_payload(payload).strip() != ""

    def _track_accepted_input(self, payload: dict[str, Any]) -> None:
        source_message_id = payload.get("source_message_id")
        if not isinstance(source_message_id, str) or source_message_id.strip() == "":
            return

        normalized_source_message_id = source_message_id.strip()
        self._remote_source_message_ids.add(normalized_source_message_id)
        text = _agent_input_text_from_payload(payload).strip()
        if text == "":
            return

        if self._accepted_input_callback is not None:
            self._accepted_input_callback(
                AcceptedAgentInput(
                    message_id=normalized_source_message_id,
                    role=self._role_for_sender(payload.get("sender_name")),
                    text=text,
                )
            )

    def _role_for_sender(self, sender_name: object) -> str:
        normalized_sender_name = _normalized_string(sender_name)
        if normalized_sender_name is None:
            return "user"
        if normalized_sender_name == self._local_participant_name:
            return "you"
        return normalized_sender_name

    def _track_local_turn_started(self, payload: dict[str, Any]) -> None:
        if not self._is_local_source_message(payload.get("source_message_id")):
            return
        turn_id = payload.get("turn_id")
        if not isinstance(turn_id, str) or turn_id.strip() == "":
            return
        self._local_turn_ids.add(turn_id.strip())

    def _track_remote_turn_started(self, payload: dict[str, Any]) -> None:
        source_message_id = payload.get("source_message_id")
        turn_id = payload.get("turn_id")
        if not isinstance(source_message_id, str) or not isinstance(turn_id, str):
            return
        if source_message_id.strip() not in self._remote_source_message_ids:
            return
        normalized_turn_id = turn_id.strip()
        if normalized_turn_id == "":
            return
        self._remote_turn_output_parts.setdefault(normalized_turn_id, [])

    def _track_remote_text_delta(self, payload: dict[str, Any]) -> None:
        turn_id = payload.get("turn_id")
        text = payload.get("text")
        if not isinstance(turn_id, str) or not isinstance(text, str):
            return
        parts = self._remote_turn_output_parts.get(turn_id.strip())
        if parts is None:
            return
        parts.append(text)

    def _track_remote_turn_ended(self, payload: dict[str, Any]) -> None:
        turn_id = payload.get("turn_id")
        if not isinstance(turn_id, str):
            return
        normalized_turn_id = turn_id.strip()
        parts = self._remote_turn_output_parts.pop(normalized_turn_id, None)
        if parts is None:
            return
        text = "".join(parts).strip()
        if text == "":
            return
        if self._accepted_input_callback is not None:
            self._accepted_input_callback(
                AcceptedAgentInput(
                    message_id=normalized_turn_id,
                    role=self._client.remote_participant_name,
                    text=text,
                )
            )

    def _track_queued_agent_input(self, payload: dict[str, Any]) -> None:
        source_message_id = payload.get("source_message_id")
        if not isinstance(source_message_id, str) or source_message_id.strip() == "":
            return
        text = _agent_input_text_from_payload(payload)
        if text == "":
            return
        message_id = source_message_id.strip()
        existing = self._pending_inputs.get(message_id)
        self._pending_inputs[message_id] = PendingAgentInput(
            message_id=message_id,
            role=self._role_for_sender(payload.get("sender_name")),
            text=text,
            applied=False if existing is None else existing.applied,
        )

    def _clear_queued_agent_input(self, source_message_id: object) -> None:
        if not isinstance(source_message_id, str):
            return
        normalized = source_message_id.strip()
        if normalized == "":
            return
        self._pending_inputs.pop(normalized, None)

    def _mark_queued_agent_input_applied(self, source_message_id: object) -> None:
        if not isinstance(source_message_id, str):
            return
        normalized = source_message_id.strip()
        if normalized == "":
            return
        existing = self._pending_inputs.get(normalized)
        if existing is None or existing.applied:
            return
        self._pending_inputs[normalized] = PendingAgentInput(
            message_id=existing.message_id,
            role=existing.role,
            text=existing.text,
            applied=True,
        )

    def clear_applied_queued_agent_inputs(self) -> None:
        self._pending_inputs = {
            message_id: item
            for message_id, item in self._pending_inputs.items()
            if not item.applied
        }


class MessagingChatClient(BaseChatClient):
    def __init__(
        self,
        *,
        room: RoomClient,
        participant_name: str,
        timeout: float = 30,
    ) -> None:
        super().__init__(timeout=timeout)
        self._room = room
        self._participant_name = participant_name
        self._participant: RemoteParticipant | None = None
        self._room_status_connected = False

    @property
    def room(self) -> RoomClient:
        return self._room

    @property
    def remote_participant_name(self) -> str:
        return self._participant_name

    async def _start_transport(self) -> None:
        self._room.on("room.status", self._on_room_status)
        self._room.on("disconnected", self._on_room_disconnected)
        self._room.on("reconnected", self._on_room_reconnected)
        self._room.messaging.on("message", self._on_message)
        if not self._room.messaging.is_enabled:
            await self._room.messaging.enable()
        await self._wait_for_participant()
        self._emit_connection_status(
            status="connected",
            message=f"connected to {self._participant_name}",
        )

    async def _stop_transport(self) -> None:
        self._room.messaging.off("message", self._on_message)
        self._room.off("room.status", self._on_room_status)
        self._room.off("disconnected", self._on_room_disconnected)
        self._room.off("reconnected", self._on_room_reconnected)
        self._emit_connection_status(
            status="disconnected",
            message="chat client stopped",
            reason="stopped",
        )

    def _on_room_status(self, **kwargs: object) -> None:
        status = kwargs.get("status")
        if not isinstance(status, str):
            return
        message_value = kwargs.get("message")
        message = message_value if isinstance(message_value, str) else None
        normalized = status.strip().lower()
        if normalized in ("connected", "ready"):
            status_name = "connected"
            self._room_status_connected = True
        elif normalized == "reconnected":
            status_name = "reconnected" if self._room_status_connected else "connected"
            self._room_status_connected = True
        elif normalized == "disconnected":
            status_name = "disconnected"
            self._room_status_connected = False
        elif normalized in ("reconnecting", "connecting"):
            status_name = "reconnecting"
        else:
            return
        self._emit_connection_status(
            status=status_name,
            message=message,
            reason=message if status_name == "disconnected" else None,
        )

    def _on_room_disconnected(self, **kwargs: object) -> None:
        reason_value = kwargs.get("reason")
        reason = reason_value if isinstance(reason_value, str) else None
        self._room_status_connected = False
        self._emit_connection_status(
            status="disconnected",
            message=reason or "room connection lost",
            reason=reason,
        )

    def _on_room_reconnected(self, **_: object) -> None:
        status = "reconnected" if self._room_status_connected else "connected"
        self._room_status_connected = True
        self._emit_connection_status(
            status=status,
            message="room connection restored",
        )

    async def _wait_for_participant(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                while self._participant is None:
                    for participant in self._room.messaging.get_participants():
                        if participant.get_attribute("name") != self._participant_name:
                            continue
                        self._participant = participant
                        return
                    await asyncio.sleep(1)
        except asyncio.TimeoutError as exc:
            raise RoomException(
                f"timed out waiting for {self._participant_name}"
            ) from exc

    def _on_message(self, message: Any) -> None:
        if self._participant is None:
            return
        if message.from_participant_id != self._participant.id:
            return
        if message.type != "agent-message":
            return
        raw_message = message.message
        if not isinstance(raw_message, dict):
            return
        raw_payload = (
            raw_message
            if isinstance(raw_message.get("type"), str)
            else raw_message.get("payload")
        )
        if not isinstance(raw_payload, dict):
            return
        self._handle_agent_payload(raw_payload)

    async def _send_agent_message(self, payload: AgentMessage) -> None:
        if self._participant is None:
            raise RoomException("chat client not started")
        await self._room.messaging.send_message(
            to=self._participant,
            type="agent-message",
            message=payload.model_dump(mode="json"),
            attachment=None,
        )


class WebSocketChatClient(BaseChatClient):
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        protocols: tuple[str, ...] = ("meshagent-msgpack",),
        encoding: WebSocketChatEncoding | None = None,
        heartbeat: float = 30.0,
        timeout: float = 30,
        max_msg_size: int = DEFAULT_WEBSOCKET_MAX_MSG_SIZE,
        reconnect: bool = True,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 10.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self._url = url
        self._headers = headers
        self._protocols = protocols
        self._encoding = encoding or MsgpackWebSocketChatEncoding()
        self._heartbeat = heartbeat
        self._max_msg_size = max_msg_size
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._close_code: int | None = None
        self._receive_exception: BaseException | None = None
        self._reconnect = reconnect
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._started = False
        self._stopping = False
        self._connecting = False
        self._reconnect_attempts = 0

    @property
    def remote_participant_name(self) -> str:
        return "assistant"

    async def _start_transport(self) -> None:
        self._started = True
        self._stopping = False
        await self._connect(is_reconnect=False)

    async def _connect(self, *, is_reconnect: bool) -> None:
        if self._connecting:
            return
        self._connecting = True
        session = new_client_session()
        try:
            websocket = await session.ws_connect(
                self._url,
                headers=self._headers,
                heartbeat=self._heartbeat,
                protocols=self._protocols,
                max_msg_size=self._max_msg_size,
            )
        except aiohttp.ClientResponseError as exc:
            await session.close()
            raise RoomException(
                f"chat websocket connection failed: {exc.status} {exc.message}"
            ) from exc
        except aiohttp.ClientError as exc:
            await session.close()
            raise RoomException(f"chat websocket connection failed: {exc}") from exc
        finally:
            self._connecting = False
        self._session = session
        self._websocket = websocket
        self._close_code = None
        self._receive_exception = None
        self._reconnect_attempts = 0
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._receive_task.add_done_callback(_consume_task_exception)
        self._emit_connection_status(
            status="reconnected" if is_reconnect else "connected",
            message=(
                "chat websocket reconnected"
                if is_reconnect
                else "chat websocket connected"
            ),
        )
        if is_reconnect:
            await self._reopen_sessions()

    async def _stop_transport(self) -> None:
        self._stopping = True
        self._started = False
        reconnect_task = self._reconnect_task
        self._reconnect_task = None
        if reconnect_task is not None:
            reconnect_task.cancel()
            await asyncio.gather(reconnect_task, return_exceptions=True)
        receive_task = self._receive_task
        self._receive_task = None
        websocket = self._websocket
        self._websocket = None
        session = self._session
        self._session = None
        if receive_task is not None:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
        if websocket is not None:
            await websocket.close()
        if session is not None:
            await session.close()
        self._emit_connection_status(
            status="disconnected",
            message="chat websocket stopped",
            reason="stopped",
        )

    async def _receive_loop(self) -> None:
        websocket = self._websocket
        if websocket is None:
            return
        try:
            async for message in websocket:
                if message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    try:
                        agent_message = self._encoding.decode(message)
                    except Exception as exc:
                        self._receive_exception = exc
                        await websocket.close()
                        return
                    self._handle_agent_payload(
                        agent_message.model_dump(mode="json", exclude_none=True)
                    )
                elif message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    self._close_code = websocket.close_code
                    if message.type == aiohttp.WSMsgType.ERROR:
                        self._receive_exception = websocket.exception()
                    return
        except BaseException as exc:
            self._receive_exception = exc
            raise
        finally:
            self._close_code = websocket.close_code
            if self._websocket is websocket:
                self._websocket = None
                session = self._session
                self._session = None
                if not websocket.closed:
                    await websocket.close()
                if session is not None:
                    await session.close()
            if not self._stopping and self._started:
                self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if not self._reconnect:
            self._emit_connection_status(
                status="disconnected",
                message="chat websocket disconnected",
                reason=self._connection_close_reason(),
            )
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        delay = min(
            self._reconnect_initial_delay * (2**self._reconnect_attempts),
            self._reconnect_max_delay,
        )
        self._reconnect_attempts += 1
        reason = self._connection_close_reason()
        self._emit_connection_status(
            status="reconnecting",
            message="chat websocket reconnecting",
            reason=reason,
            retry_in_seconds=delay,
        )
        self._reconnect_task = asyncio.create_task(self._reconnect_loop(delay=delay))
        self._reconnect_task.add_done_callback(_consume_task_exception)

    async def _reconnect_loop(self, *, delay: float) -> None:
        next_delay = delay
        while not self._stopping and self._started:
            await asyncio.sleep(next_delay)
            try:
                await self._connect(is_reconnect=True)
                return
            except RoomException as exc:
                next_delay = min(next_delay * 2, self._reconnect_max_delay)
                self._emit_connection_status(
                    status="reconnecting",
                    message="chat websocket reconnecting",
                    reason=str(exc),
                    retry_in_seconds=next_delay,
                )

    def _connection_close_reason(self) -> str | None:
        details: list[str] = []
        if self._close_code is not None:
            details.append(f"code={self._close_code}")
        if self._receive_exception is not None:
            details.append(f"error={self._receive_exception}")
        if len(details) == 0:
            return None
        return ", ".join(details)

    async def _reopen_sessions(self) -> None:
        messages: list[AgentMessage] = []
        for session in self._reopenable_sessions():
            messages.append(
                OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id=session.thread_path,
                    load=True,
                    since_turn=session.last_completed_turn_id,
                )
            )
        messages.append(ModelsRequest(type=AGENT_MESSAGE_MODELS_REQUEST))
        for message in messages:
            await self._send_agent_message(message)

    def _reopenable_sessions(self) -> Iterable[ChatThreadSession]:
        for session in self._thread_sessions.values():
            if not session._closed:
                yield session

    async def _send_agent_message(self, payload: AgentMessage) -> None:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            details: list[str] = []
            close_code = self._close_code
            if close_code is None and websocket is not None:
                close_code = websocket.close_code
            if close_code is not None:
                details.append(f"code={close_code}")
            if self._receive_exception is not None:
                details.append(f"error={self._receive_exception}")
            detail_text = "" if len(details) == 0 else f" ({', '.join(details)})"
            raise RoomException(f"chat websocket is closed{detail_text}")
        encoded = self._encoding.encode(payload)
        if isinstance(encoded, str):
            await websocket.send_str(encoded)
        else:
            await websocket.send_bytes(encoded)


class LocalChatClient(BaseChatClient):
    def __init__(
        self,
        *,
        thread_path: str | None = None,
        send_message: Callable[[Message], None],
        events: asyncio.Queue[Message],
        on_close: Callable[[], None] | None = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(timeout=timeout)
        self._thread_path = thread_path
        self._send_message = send_message
        self._events = events
        self._on_close = on_close
        self._receive_task: asyncio.Task[None] | None = None
        self._thread_session = self._create_thread_session(thread_path=thread_path)

    @property
    def thread_session(self) -> ChatThreadSession:
        return self._thread_session

    @property
    def remote_participant_name(self) -> str:
        return "assistant"

    @property
    def has_thread_path(self) -> bool:
        return self._thread_session.has_thread_path

    @property
    def thread_path(self) -> str:
        return self._thread_session.thread_path

    @property
    def thread_status_text(self) -> str | None:
        return self._thread_session.thread_status_text

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return self._thread_session.messages

    @property
    def pending_inputs(self) -> tuple[PendingAgentInput, ...]:
        return self._thread_session.pending_inputs

    @property
    def queued_message_labels(self) -> tuple[str, ...]:
        return self._thread_session.queued_message_labels

    def add_agent_message(self, message: AgentMessage) -> None:
        self._thread_session.add_agent_message(message)

    def clear_applied_queued_agent_inputs(self) -> None:
        self._thread_session.clear_applied_queued_agent_inputs()

    async def _start_transport(self) -> None:
        if self._receive_task is None:
            self._receive_task = asyncio.create_task(self._receive_loop())

    async def _stop_transport(self) -> None:
        receive_task = self._receive_task
        self._receive_task = None
        if receive_task is not None:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
        on_close = self._on_close
        self._on_close = None
        if on_close is not None:
            on_close()

    async def _receive_loop(self) -> None:
        while True:
            event = await self._events.get()
            self._handle_agent_payload(event.data.model_dump(mode="python"))

    async def _send_agent_message(self, payload: AgentMessage) -> None:
        self._send_message(Message(data=payload))

    async def receive(self) -> dict[str, Any]:
        return await self._thread_session.receive()
