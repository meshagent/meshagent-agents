from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from meshagent.agents.agent import SingleRoomAgent, AgentSessionContext
from meshagent.api.chan import Chan
from meshagent.api import (
    RoomMessage,
    RoomClient,
    RemoteParticipant,
    Participant,
    Requirement,
    Element,
    MeshDocument,
)
from meshagent.api.messaging import JsonContent
from meshagent.tools import (
    Toolkit,
    ToolContext,
    FunctionTool,
)
from meshagent.agents.adapter import LLMAdapter
from meshagent.openai.tools.responses_adapter import (
    ReasoningTool,
    OpenAIResponsesAdapter,
    OpenAIResponsesSessionContext,
)
from meshagent.openai.tools.completions_adapter import OpenAICompletionsAdapter
from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.agents.completions_thread_adapter import CompletionsThreadAdapter
from meshagent.agents.responses_thread_adapter import (
    ResponsesThreadAdapter,
    response_event_to_agent_event,
)

import uuid
import posixpath
import re
from datetime import datetime, timedelta, timezone
import asyncio
from typing import Any, Optional, Callable, AsyncIterator, Awaitable, Literal
import logging
import warnings
from asyncio import CancelledError
from meshagent.api import RoomException
from meshagent.api.chan import ChanClosed
from opentelemetry import trace
import json
import aiohttp
from pydantic import BaseModel
from meshagent.tools import tool
from meshagent.tools.strict_schema import ensure_strict_json_schema
from pathlib import Path
from meshagent.agents.skills import to_prompt
from meshagent.agents.thread_schema import thread_list_schema, thread_schema
from meshagent.tools.storage import StorageToolkit


tracer = trace.get_tracer("meshagent.chatbot")

logger = logging.getLogger("chat")
_legacy_chatbot_init_chat_context_warned: set[type] = set()
_DEFAULT_NEW_THREAD_NAME_RULES = [
    "generate a concise, friendly title for this chat thread",
    "return only a thread_name value suitable for display in a thread list",
    "thread_name should be 2-6 words and topic-focused",
    "use normal capitalization and spaces, and do not include a .thread extension",
]

ThreadStatusMode = Literal["busy", "steerable"]


class ChatAttachment(BaseModel):
    path: str


class ChatMessageInput(BaseModel):
    text: str
    attachments: Optional[list[ChatAttachment]] = None


class ChatBotReasoningTool(ReasoningTool):
    def __init__(self, *, room: RoomClient, thread_context: "ChatThreadContext"):
        super().__init__()
        self.thread_context = thread_context
        self.room = room

        self._reasoning_element = None
        self._reasoning_item = None

    def _get_messages_element(self):
        messages = self.thread_context.thread.root.get_children_by_tag_name("messages")
        if len(messages) > 0:
            return messages[0]
        return None

    async def on_reasoning_summary_part_added(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        el = self._get_messages_element()
        if el is None:
            logger.warning("missing messages element, cannot log reasoning")
        else:
            self._reasoning_element = el.append_child("reasoning", {"summary": ""})

    async def on_reasoning_summary_part_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        self._reasoning_element = None

    async def on_reasoning_summary_text_delta(
        self,
        context: ToolContext,
        *,
        delta: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        el = self._reasoning_element
        el.set_attribute("summary", el.get_attribute("summary") + delta)

    async def on_reasoning_summary_text_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass


def get_online_participants(
    *,
    room: RoomClient,
    thread: MeshDocument,
    exclude: Optional[list[Participant]] = None,
) -> list[RemoteParticipant]:
    results = list[RemoteParticipant]()

    for prop in thread.root.get_children():
        if prop.tag_name == "members":
            for member in prop.get_children():
                for online in room.messaging.get_participants():
                    if online.get_attribute("name") == member.get_attribute("name"):
                        if exclude is None or online not in exclude:
                            results.append(online)

    return results


class ChatThreadContext:
    def __init__(
        self,
        *,
        session: AgentSessionContext,
        thread: MeshDocument,
        path: str,
        participants: Optional[list[RemoteParticipant]] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
    ):
        self.thread = thread
        if participants is None:
            participants = []

        self.participants = participants
        self.session = session
        self.path = path
        self._event_handler = event_handler

    async def __aenter__(self) -> "ChatThreadContext":
        await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.session.__aexit__(exc_type, exc, tb)

    def emit(self, event: dict):
        if self._event_handler is not None:
            self._event_handler(event)

    @property
    def context_id(self) -> str:
        return self.session.id

    def to_caller_context(self) -> dict:
        return {"chat": self.session.to_json()}


class ChatBotClient:
    def __init__(
        self,
        *,
        room: RoomClient,
        participant_name: str,
        thread_path: str,
        timeout: float = 30,
    ):
        self.room = room
        self.participant_name = participant_name
        self.thread_path = thread_path
        self._messages: asyncio.Queue[str] = asyncio.Queue()
        self._doc = None
        self._participant: Optional[RemoteParticipant] = None
        self._timeout = timeout

    async def start(self) -> None:
        self._doc = await self.room.sync.open(path=self.thread_path)
        self.room.messaging.on("message", self._on_message)
        await asyncio.sleep(1)
        await self._wait_for_participant()
        if self._participant is not None:
            await self.room.messaging.send_message(
                to=self._participant,
                type="opened",
                message={"path": self.thread_path},
                attachment=None,
            )

    async def stop(self) -> None:
        await self.room.sync.close(path=self.thread_path)
        self.room.messaging.off("message", self._on_message)

    async def __aenter__(self) -> "ChatBotClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        await self.stop()

    def _on_message(self, message: RoomMessage) -> None:
        received_type = message.type
        received_message = message.message

        if received_type == "chat" and received_message["path"] == self.thread_path:
            self._messages.put_nowait(received_message["text"])

    async def _wait_for_participant(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                while self._participant is None:
                    for participant in self.room.messaging.get_participants():
                        if participant.get_attribute("name") == self.participant_name:
                            self._participant = participant
                            break

                    await asyncio.sleep(1)
        except asyncio.TimeoutError as exc:
            raise RoomException(
                f"timed out waiting for {self.participant_name}"
            ) from exc

    async def clear(self) -> None:
        if self._participant is None:
            return

        await self.room.messaging.send_message(
            to=self._participant,
            type="clear",
            message={"path": self.thread_path},
            attachment=None,
        )

    async def cancel(self) -> None:
        if self._participant is None:
            return

        await self.room.messaging.send_message(
            to=self._participant,
            type="cancel",
            message={"path": self.thread_path},
            attachment=None,
        )

    async def send_approval_decision(self, *, approval_id: str, approve: bool) -> None:
        if self._participant is None:
            return

        normalized_id = approval_id.strip()
        if normalized_id == "":
            raise RoomException("approval_id is required")

        await self.room.messaging.send_message(
            to=self._participant,
            type="approved" if approve else "rejected",
            message={
                "path": self.thread_path,
                "approval_id": normalized_id,
            },
            attachment=None,
        )

    async def send(
        self,
        *,
        text: str,
        attachments: Optional[list[dict]] = None,
        steer: bool = False,
    ) -> None:
        if self._participant is None or self._doc is None:
            raise RoomException("chat client not started")

        attachment_payload: list[dict[str, str]] = []
        if attachments is not None:
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                path = attachment.get("path")
                if not isinstance(path, str):
                    continue
                normalized_path = path.strip()
                if normalized_path == "":
                    continue
                attachment_payload.append({"path": normalized_path})

        await self.room.messaging.send_message(
            to=self._participant,
            type="steer" if steer else "chat",
            message={
                "text": text,
                "path": self.thread_path,
                "attachments": attachment_payload,
                "store": True,
            },
            attachment=None,
        )

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._messages.get()

    async def receive(self) -> str:
        return await self._messages.get()

    async def receive_nowait(self) -> str:
        return await self._messages.get_nowait()


@dataclass
class _QueuedChatMessage:
    type: str
    message: dict
    from_participant: RemoteParticipant
    result: asyncio.Future = field(default_factory=asyncio.Future)


class ChatBotBase(SingleRoomAgent, ABC):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        always_reply: Optional[bool] = None,
        skill_dirs: Optional[list[str]] = None,
        thread_dir: Optional[str] = None,
        threading_mode: Optional[str] = None,
        thread_name_rules: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            annotations=annotations,
        )

        if toolkits is None:
            toolkits = []

        if always_reply is None:
            always_reply = False

        self._always_reply = always_reply

        self._message_channels = dict[str, Chan[_QueuedChatMessage]]()

        self._room: RoomClient | None = None
        self._toolkits = toolkits
        self._client_rules = client_rules

        if rules is None:
            rules = []

        self._rules = rules
        self._is_typing = dict[str, asyncio.Task]()
        self._auto_greet_message = auto_greet_message

        if empty_state_title is None:
            empty_state_title = "How can I help you?"
        self._empty_state_title = empty_state_title

        self._thread_tasks = dict[str, asyncio.Task]()
        self._open_threads = {}
        self._thread_contexts: dict[str, ChatThreadContext] = {}
        self._thread_list_document: Optional[MeshDocument] = None
        self._thread_list_path: Optional[str] = None
        self._pending_thread_list_paths: set[str] = set()
        self._thread_list_background_tasks: set[asyncio.Task[None]] = set()

        self._skill_dirs = skill_dirs
        self._thread_status_values: dict[str, str] = {}
        self._thread_status_mode_values: dict[str, ThreadStatusMode] = {}
        self._thread_status_started_at_values: dict[str, str] = {}
        self._thread_status_keys: dict[str, str] = {}
        self._thread_status_locks: dict[str, asyncio.Lock] = {}
        self._thread_status_generations: dict[str, int] = {}
        self._threading_mode = (
            threading_mode.strip()
            if isinstance(threading_mode, str) and threading_mode.strip() != ""
            else None
        )
        self._thread_dir = (
            self._normalize_thread_dir(thread_dir=thread_dir)
            if thread_dir is not None
            else None
        )
        if thread_name_rules is not None and len(thread_name_rules) > 0:
            self._thread_name_rules = [*thread_name_rules]
        else:
            self._thread_name_rules = [*_DEFAULT_NEW_THREAD_NAME_RULES]

    def thread_name_adapter(self) -> Optional[LLMAdapter]:
        return None

    def _normalize_thread_dir(self, *, thread_dir: str) -> str:
        normalized = thread_dir.strip().rstrip("/")
        if normalized == "":
            raise RoomException("thread_dir must not be empty")
        return normalized

    def _default_thread_dir(self) -> str:
        local_name = "chat"
        if self._room is not None and self._room.local_participant is not None:
            participant_name = self._room.local_participant.get_attribute("name")
            if isinstance(participant_name, str) and participant_name.strip() != "":
                local_name = participant_name.strip()

        return self._normalize_thread_dir(
            thread_dir=posixpath.join(".threads", local_name)
        )

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir
        return self._default_thread_dir()

    def _thread_list_dir(self) -> Optional[str]:
        if self._thread_dir is not None:
            return self._thread_dir
        if self._threading_mode == "default-new":
            return self._default_thread_dir()
        return None

    def _sanitize_thread_name(self, *, value: str) -> str:
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

    def _thread_path_for_name(self, *, thread_name: str, thread_dir: str) -> str:
        return posixpath.join(thread_dir, f"{thread_name}.thread")

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _thread_list_index_path(self) -> Optional[str]:
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
                if str(parsed_uuid) == raw_name.lower():
                    return "New Chat"
            except ValueError:
                pass

        normalized = self._sanitize_thread_name(value=filename)
        return normalized

    def _find_thread_list_entry(self, *, path: str) -> Optional[Element]:
        if self._thread_list_dir() is None:
            return None

        if self._thread_list_document is None:
            return None

        for child in self._thread_list_document.root.get_children():
            if child.tag_name != "thread":
                continue
            if child.get_attribute("path") == path:
                return child

        return None

    def _upsert_thread_list_entry(
        self,
        *,
        path: str,
        name: Optional[str] = None,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
    ) -> None:
        if self._thread_list_dir() is None or self._thread_list_document is None:
            return

        now = self._utc_now_iso()
        provided_name = name.strip() if isinstance(name, str) else ""

        entry = self._find_thread_list_entry(path=path)
        if (
            entry is None
            and provided_name == ""
            and path in self._pending_thread_list_paths
        ):
            return
        if entry is None:
            resolved_name = (
                provided_name
                if provided_name != ""
                else self._thread_list_entry_name_for_path(path=path)
            )
            created_value = (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
            modified_value = (
                modified_at.strip()
                if isinstance(modified_at, str) and modified_at.strip() != ""
                else created_value
            )
            self._thread_list_document.root.append_child(
                tag_name="thread",
                attributes={
                    "name": resolved_name,
                    "path": path,
                    "created_at": created_value,
                    "modified_at": modified_value,
                },
            )
            return

        if provided_name != "":
            entry.set_attribute("name", provided_name)
        else:
            existing_name = entry.get_attribute("name")
            if not isinstance(existing_name, str) or existing_name.strip() == "":
                entry.set_attribute(
                    "name",
                    self._thread_list_entry_name_for_path(path=path),
                )

        entry.set_attribute("path", path)

        existing_created_at = entry.get_attribute("created_at")
        created_value = (
            existing_created_at.strip()
            if isinstance(existing_created_at, str)
            and existing_created_at.strip() != ""
            else (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
        )
        entry.set_attribute("created_at", created_value)

        modified_value = self._next_thread_list_modified_at(
            entry=entry,
            modified_at=modified_at,
        )
        entry.set_attribute("modified_at", modified_value)

    def _record_new_thread_in_index(
        self, *, path: str, name: Optional[str] = None
    ) -> None:
        now = self._utc_now_iso()
        self._upsert_thread_list_entry(
            path=path,
            name=name,
            created_at=now,
            modified_at=now,
        )

    def _touch_thread_in_index(self, *, path: str) -> None:
        self._upsert_thread_list_entry(
            path=path,
            modified_at=self._utc_now_iso(),
        )

    def _begin_pending_thread_list_entry(self, *, path: str) -> None:
        if self._thread_list_dir() is None:
            return
        self._pending_thread_list_paths.add(path)

    def _track_thread_list_background_task(
        self,
        *,
        task: asyncio.Task[None],
    ) -> None:
        self._thread_list_background_tasks.add(task)

        def _cleanup(done_task: asyncio.Task[None]) -> None:
            self._thread_list_background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.error(
                    "deferred chat thread list update failed",
                    exc_info=exc,
                )

        task.add_done_callback(_cleanup)

    def _schedule_pending_thread_list_entry(
        self,
        *,
        context: ToolContext,
        path: str,
        text: str,
    ) -> None:
        if self._thread_list_dir() is None:
            return

        async def run() -> None:
            try:
                try:
                    friendly_name = await self._generate_thread_friendly_name(
                        context=context,
                        text=text,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    logger.warning(
                        "unable to determine deferred chat thread name for %s, using fallback",
                        path,
                        exc_info=ex,
                    )
                    friendly_name = self._fallback_thread_name(text=text)

                self._record_new_thread_in_index(path=path, name=friendly_name)
            finally:
                self._pending_thread_list_paths.discard(path)

        self._track_thread_list_background_task(task=asyncio.create_task(run()))

    async def _cancel_thread_list_background_tasks(self) -> None:
        tasks = list(self._thread_list_background_tasks)
        for task in tasks:
            task.cancel()
        if len(tasks) > 0:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._thread_list_background_tasks.clear()
        self._pending_thread_list_paths.clear()

    async def _wait_for_thread_list_background_tasks(self) -> None:
        while len(self._thread_list_background_tasks) > 0:
            tasks = list(self._thread_list_background_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _next_thread_list_modified_at(
        self,
        *,
        entry: Element | None,
        modified_at: str | None = None,
    ) -> str:
        candidate = (
            modified_at.strip()
            if isinstance(modified_at, str) and modified_at.strip() != ""
            else self._utc_now_iso()
        )
        if entry is None:
            return candidate

        existing_modified_at = self._parse_iso_datetime(
            value=entry.get_attribute("modified_at")
        )
        candidate_modified_at = self._parse_iso_datetime(value=candidate)
        if (
            existing_modified_at is None
            or candidate_modified_at is None
            or candidate_modified_at > existing_modified_at
        ):
            return candidate

        return (
            (existing_modified_at + timedelta(microseconds=1))
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            )
        )

    def _thread_list_entries(self) -> list[Element]:
        if self._thread_list_dir() is None or self._thread_list_document is None:
            return []

        return [
            child
            for child in self._thread_list_document.root.get_children()
            if child.tag_name == "thread"
        ]

    def _parse_iso_datetime(self, *, value: Any) -> Optional[datetime]:
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

    def _thread_created_sort_datetime(self, *, entry: Element) -> datetime:
        created_at = self._parse_iso_datetime(value=entry.get_attribute("created_at"))
        if created_at is not None:
            return created_at
        return datetime.min.replace(tzinfo=timezone.utc)

    def _thread_sort_path(self, *, entry: Element) -> str:
        path_value = entry.get_attribute("path")
        if not isinstance(path_value, str):
            return ""
        return path_value.strip()

    def _sorted_thread_list_entries(self) -> list[Element]:
        entries = self._thread_list_entries()
        entries.sort(key=lambda entry: self._thread_sort_path(entry=entry))
        entries.sort(
            key=lambda entry: self._thread_created_sort_datetime(entry=entry),
            reverse=True,
        )
        entries.sort(
            key=lambda entry: self._thread_sort_datetime(entry=entry),
            reverse=True,
        )
        return entries

    def _thread_list_slice(
        self,
        *,
        entries: list[Element],
        limit: int,
        offset: int,
    ) -> list[Element]:
        normalized_offset = max(0, int(offset))
        normalized_limit = max(1, min(200, int(limit)))
        return entries[normalized_offset : normalized_offset + normalized_limit]

    async def _open_thread_list_document(self) -> None:
        index_path = self._thread_list_index_path()
        if index_path is None or self._room is None:
            return

        if (
            self._thread_list_document is not None
            and self._thread_list_path == index_path
        ):
            return

        self._thread_list_path = index_path
        try:
            self._thread_list_document = await self.room.sync.open(
                path=index_path,
                schema=thread_list_schema,
            )
        except Exception as ex:
            logger.warning(
                "unable to open thread list document at %s",
                index_path,
                exc_info=ex,
            )
            self._thread_list_document = None
            self._thread_list_path = None

    async def _close_thread_list_document(
        self,
        *,
        room: Optional[RoomClient] = None,
    ) -> None:
        if self._thread_list_document is None or self._thread_list_path is None:
            return

        close_room = room if room is not None else self._room
        close_path = self._thread_list_path
        self._thread_list_document = None
        self._thread_list_path = None

        if close_room is None:
            return

        try:
            await close_room.sync.close(path=close_path)
        except Exception as ex:
            logger.warning(
                "unable to close thread list document at %s",
                close_path,
                exc_info=ex,
            )

    async def _next_available_thread_path(self, *, base_path: str) -> str:
        if self._room is None:
            return base_path

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

    async def _generate_thread_friendly_name(
        self,
        *,
        context: ToolContext,
        text: str,
    ) -> str:
        generated_name = self._fallback_thread_name(text=text)
        adapter = self.thread_name_adapter()
        if adapter is not None:
            session = adapter.create_session()
            caller_context = context.caller_context
            if isinstance(caller_context, dict):
                raw_chat_context = caller_context.get("chat")
                if isinstance(raw_chat_context, dict):
                    session = type(session).from_json(raw_chat_context)
            cloned_context = session.copy()
            async with cloned_context:
                cloned_context.replace_rules(rules=self._thread_name_rules)
                cloned_context.append_user_message(text)
                try:
                    response = await adapter.next(
                        context=cloned_context,
                        caller=self.room.local_participant,
                        model=self.default_model(),
                        on_behalf_of=context.on_behalf_of or context.caller,
                        toolkits=[],
                        output_schema={
                            "type": "object",
                            "required": ["thread_name"],
                            "additionalProperties": False,
                            "properties": {
                                "thread_name": {
                                    "type": "string",
                                    "description": "2-6 word topic name for the task thread",
                                }
                            },
                        },
                    )
                    if isinstance(response, dict):
                        thread_name = response.get("thread_name")
                        if isinstance(thread_name, str):
                            generated_name = self._sanitize_thread_name(
                                value=thread_name
                            )
                except Exception as ex:
                    logger.warning(
                        "unable to auto-generate chat thread name, using fallback",
                        exc_info=ex,
                    )
        return generated_name

    async def _new_thread_path(self) -> str:
        guid_name = str(uuid.uuid4())
        path = self._thread_path_for_name(
            thread_name=guid_name,
            thread_dir=self._get_thread_dir(),
        )
        path = await self._next_available_thread_path(base_path=path)
        return path

    def _thread_status_attribute_name(self, *, path: str) -> str:
        return f"thread.status.{path}"

    def _thread_status_text_attribute_name(self, *, path: str) -> str:
        return f"thread.status.text.{path}"

    def _thread_status_mode_attribute_name(self, *, path: str) -> str:
        return f"thread.status.mode.{path}"

    def _thread_status_started_at_attribute_name(self, *, path: str) -> str:
        return f"thread.status.started_at.{path}"

    def _status_lock(self, *, path: str) -> asyncio.Lock:
        lock = self._thread_status_locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_status_locks[path] = lock
        return lock

    def _next_thread_status_generation(self, *, path: str) -> int:
        generation = self._thread_status_generations.get(path, 0) + 1
        self._thread_status_generations[path] = generation
        return generation

    def _normalize_thread_status_mode(
        self, *, mode: Optional[str]
    ) -> Optional[ThreadStatusMode]:
        if mode is None:
            return None
        normalized = mode.strip().lower()
        if normalized == "":
            return None
        if normalized not in ("busy", "steerable"):
            raise RoomException(f"unsupported thread status mode '{mode}'")
        if normalized == "busy":
            return "busy"
        return "steerable"

    def processing_thread_status_mode(
        self, *, path: str, thread_context: Optional["ChatThreadContext"]
    ) -> ThreadStatusMode:
        del path
        del thread_context
        return "busy"

    async def set_thread_status(
        self,
        *,
        path: str,
        status: Optional[str],
        mode: Optional[str] = None,
    ) -> None:
        if self._room is None or self._room.local_participant is None:
            return

        async def set_local_attribute(
            attribute_name: str, value: Optional[str]
        ) -> None:
            try:
                await self._room.local_participant.set_attribute(attribute_name, value)
            except ChanClosed:
                logger.debug(
                    "room channel closed while setting thread status '%s'",
                    attribute_name,
                )

        attribute_name = self._thread_status_attribute_name(path=path)
        text_attribute_name = self._thread_status_text_attribute_name(path=path)
        mode_attribute_name = self._thread_status_mode_attribute_name(path=path)
        started_at_attribute_name = self._thread_status_started_at_attribute_name(
            path=path
        )
        if status is None:
            self._thread_status_values.pop(path, None)
            self._thread_status_mode_values.pop(path, None)
            self._thread_status_started_at_values.pop(path, None)
            await set_local_attribute(attribute_name, None)
            await set_local_attribute(text_attribute_name, None)
            await set_local_attribute(mode_attribute_name, None)
            await set_local_attribute(started_at_attribute_name, None)
            return

        normalized = status.strip()
        if normalized == "":
            self._thread_status_values.pop(path, None)
            self._thread_status_mode_values.pop(path, None)
            self._thread_status_started_at_values.pop(path, None)
            await set_local_attribute(attribute_name, None)
            await set_local_attribute(text_attribute_name, None)
            await set_local_attribute(mode_attribute_name, None)
            await set_local_attribute(started_at_attribute_name, None)
            return

        normalized_mode = self._normalize_thread_status_mode(mode=mode)
        if normalized_mode is None:
            normalized_mode = self._thread_status_mode_values.get(path, "busy")

        started_at = self._thread_status_started_at_values.get(path)
        if started_at is None:
            started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        if (
            self._thread_status_values.get(path) == normalized
            and self._thread_status_mode_values.get(path) == normalized_mode
            and self._thread_status_started_at_values.get(path) == started_at
        ):
            return

        self._thread_status_values[path] = normalized
        self._thread_status_mode_values[path] = normalized_mode
        self._thread_status_started_at_values[path] = started_at
        await set_local_attribute(attribute_name, normalized)
        await set_local_attribute(text_attribute_name, normalized)
        await set_local_attribute(mode_attribute_name, normalized_mode)
        await set_local_attribute(started_at_attribute_name, started_at)

    async def _apply_thread_status(
        self,
        *,
        path: str,
        status: Optional[str],
        generation: Optional[int] = None,
    ) -> None:
        lock = self._status_lock(path=path)
        async with lock:
            if generation is not None:
                current_generation = self._thread_status_generations.get(path, 0)
                if generation != current_generation:
                    return
            await self.set_thread_status(path=path, status=status)

    def _set_thread_status_nowait(self, *, path: str, status: Optional[str]) -> None:
        generation = self._next_thread_status_generation(path=path)

        async def run() -> None:
            try:
                await self._apply_thread_status(
                    path=path,
                    status=status,
                    generation=generation,
                )
            except Exception as ex:
                logger.error(
                    f"unable to set thread status for {path}",
                    exc_info=ex,
                )

        asyncio.create_task(run())

    async def clear_thread_status(self, *, path: str) -> None:
        self._thread_status_keys.pop(path, None)
        generation = self._next_thread_status_generation(path=path)
        await self._apply_thread_status(
            path=path,
            status=None,
            generation=generation,
        )

    def _clear_thread_status_nowait(self, *, path: str) -> None:
        self._thread_status_keys.pop(path, None)
        self._set_thread_status_nowait(path=path, status=None)

    def _update_thread_status_from_event(self, *, path: str, event: dict) -> None:
        del path
        del event

    async def _clear_all_thread_statuses(self) -> None:
        paths = {
            *self._thread_status_values.keys(),
            *self._thread_status_mode_values.keys(),
            *self._thread_status_keys.keys(),
        }
        for path in paths:
            await self.set_thread_status(path=path, status=None)
        self._thread_status_keys.clear()
        self._thread_status_values.clear()
        self._thread_status_mode_values.clear()
        self._thread_status_started_at_values.clear()
        self._thread_status_locks.clear()
        self._thread_status_generations.clear()

    async def _send_and_save_chat(
        self,
        thread_adapter: ThreadAdapter,
        to: RemoteParticipant,
        id: str,
        text: str,
        thread_attributes: dict,
    ):
        with tracer.start_as_current_span("chatbot.thread.message") as span:
            span.set_attributes(thread_attributes)
            span.set_attribute("role", "assistant")
            span.set_attribute(
                "from_participant_name",
                self.room.local_participant.get_attribute("name"),
            )
            span.set_attributes({"id": id, "text": text})
            await self.room.messaging.send_message(
                to=to,
                type="chat",
                message={"path": thread_adapter.path, "text": text},
            )
            thread_adapter.write_text_message(
                text=text,
                participant=self.room.local_participant,
                role="agent",
            )

    async def _greet(
        self,
        *,
        thread_adapter: ThreadAdapter,
        thread_context: ChatThreadContext,
        participant: RemoteParticipant,
        thread_attributes: dict,
    ):
        if self._auto_greet_message is not None:
            thread_context.session.append_user_message(self._auto_greet_message)
            await self._send_and_save_chat(
                id=str(uuid.uuid4()),
                to=RemoteParticipant(id=participant.id),
                thread_adapter=thread_adapter,
                text=self._auto_greet_message,
                thread_attributes=thread_attributes,
            )

    def get_requirements(self):
        return [
            *super().get_requirements(),
        ]

    async def get_online_participants(
        self, *, thread: MeshDocument, exclude: Optional[list[Participant]] = None
    ):
        return get_online_participants(room=self._room, thread=thread, exclude=exclude)

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                caller=self.room.local_participant,
                on_behalf_of=participant,
                caller_context=thread_context.to_caller_context(),
                event_handler=thread_context.emit,
            )
        )

        @tool(
            name="attach_file",
            description="attach a file to the thread so the user can see it",
        )
        async def attach_file(path: str):
            messages = thread_context.thread.root.get_elements_by_tag_name("messages")[
                0
            ]
            message = messages.append_child(
                tag_name="message",
                attributes={
                    "id": str(uuid.uuid4()),
                    "text": "",
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "author_name": self.room.local_participant.get_attribute("name"),
                    "role": "agent",
                },
            )
            message.append_child(tag_name="file", attributes={"path": path})

        toolkits.append(
            Toolkit(
                name="thread tools",
                tools=[attach_file],
            )
        )
        thread_list_tools = self._build_thread_list_tools()
        if len(thread_list_tools) > 0:
            toolkits.append(
                Toolkit(
                    name="chat thread list",
                    tools=thread_list_tools,
                )
            )

        toolkit = self._open_threads[thread_context.path].make_toolkit()

        return [*self._toolkits, *toolkits, toolkit]

    @abstractmethod
    def default_model(self) -> str: ...

    @abstractmethod
    async def create_thread_context(
        self,
        *,
        path: str,
        thread: MeshDocument,
        participants: list[RemoteParticipant],
        event_handler: Callable[[dict], None],
    ) -> ChatThreadContext: ...

    def create_thread_adapter(self, *, path: str) -> ThreadAdapter:
        if isinstance(self._llm_adapter, OpenAICompletionsAdapter):
            return CompletionsThreadAdapter(
                room=self.room,
                path=path,
                format_message=self.format_message,
            )

        return ResponsesThreadAdapter(
            room=self.room,
            path=path,
            format_message=self.format_message,
        )

    def _should_store_received_chat_message(self, *, message: dict[str, Any]) -> bool:
        store = message.get("store")
        return isinstance(store, bool) and store

    async def open_thread(self, *, path: str) -> ThreadAdapter:
        logger.info(f"opening thread {path}")
        if path not in self._open_threads:
            adapter = self.create_thread_adapter(path=path)
            await adapter.start()
            self._open_threads[path] = adapter

        return self._open_threads[path]

    async def close_thread(self, *, path: str) -> None:
        logger.info(f"closing thread {path}")

        adapter = self._open_threads.pop(path, None)
        if adapter is not None:
            await adapter.stop()

    async def on_thread_open(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_clear(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_cancel(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_close(self, *, thread_context: ChatThreadContext):
        pass

    async def on_approved(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        pass

    async def on_rejected(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        pass

    async def on_thread_steer(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> None:
        del thread_context
        del from_participant
        del message

    async def cancel_thread_task(
        self, *, path: str, thread_context: Optional[ChatThreadContext]
    ) -> None:
        del thread_context
        if path in self._thread_tasks:
            self._thread_tasks[path].cancel()

    async def _safe_invoke_thread_event(
        self,
        *,
        event_name: str,
        thread_context: ChatThreadContext,
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        try:
            await handler(thread_context=thread_context)
        except ChanClosed:
            logger.debug(
                "chatbot thread event hook '%s' skipped because room channel is closed",
                event_name,
            )
        except Exception as e:
            logger.error(
                f"chatbot thread event hook '{event_name}' failed",
                exc_info=e,
            )

    async def _safe_invoke_chat_event(
        self,
        *,
        event_name: str,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict[str, Any],
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        try:
            await handler(
                thread_context=thread_context,
                from_participant=from_participant,
                message=message,
            )
        except Exception as e:
            logger.error(
                f"chatbot chat event hook '{event_name}' failed",
                exc_info=e,
            )

    def get_thread_members(self, *, thread: MeshDocument) -> list[str]:
        results = []

        for prop in thread.root.get_children():
            if prop.tag_name == "members":
                for member in prop.get_children():
                    results.append(member.get_attribute("name"))

        return results

    def get_skills_storage_toolkit(self) -> StorageToolkit | None:
        return None

    async def get_rules(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ):
        rules = [*self._rules]

        if self._skill_dirs is not None and len(self._skill_dirs) > 0:
            rules.append(
                "You have access to to following skills which follow the agentskills spec:"
            )
            rules.append(
                await to_prompt(
                    [*(Path(p) for p in self._skill_dirs)],
                    storage_toolkit=self.get_skills_storage_toolkit(),
                )
            )
            rules.append(
                "Use the shell or storage tool to find out more about skills and execute them when they are required"
            )

        client = participant.get_attribute("client")

        if self._client_rules is not None and client is not None:
            cr = self._client_rules.get(client)
            if cr is not None:
                rules.extend(cr)

        # Without this rule 5.2 / 5.1 like to start their messages with things like "I could say"
        rules.append("based on the previous transcript, take your turn and respond")

        return rules

    async def on_chat_received(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> Optional[str]:
        return None

    def format_message(self, *, user_name: str, message: str, iso_timestamp: str):
        return f"{user_name} said at {iso_timestamp}: {message}"

    def _chat_error_message(self, *, error: Exception) -> str:
        if isinstance(error, RoomException):
            message = str(error).strip()
            if message != "":
                return message
        if isinstance(error, aiohttp.WSServerHandshakeError):
            return OpenAIResponsesSessionContext._handshake_error_message(error)
        return "An unexpected error occured. Please try again later."

    async def _spawn_thread(self, path: str, messages: Chan[RoomMessage]):
        logger.debug("chatbot is starting a thread", extra={"path": path})
        opened = False

        current_file = None
        thread_context = None

        thread_attributes = None

        thread = None
        thread_adapter = None

        def handle_event(evt, participant: RemoteParticipant):
            if isinstance(evt, BaseModel):
                evt = evt.model_dump(mode="json")

            if evt.get("type") == "response.output_text.done" and thread is not None:
                online = get_online_participants(room=self._room, thread=thread)
                for online_participant in online:
                    if online_participant.id != self._room.local_participant.id:
                        logger.info(
                            f"replying to {online_participant.get_attribute('name')}"
                        )
                        self._room.messaging.send_message_nowait(
                            to=online_participant,
                            type="chat",
                            message={
                                "type": "chat",
                                "path": path,
                                "text": evt.get("text"),
                            },
                        )

                if participant not in online:
                    logger.info(f"replying to {participant.get_attribute('name')}")
                    self._room.messaging.send_message_nowait(
                        to=participant,
                        type="chat",
                        message={
                            "type": "chat",
                            "path": path,
                            "text": evt.get("text"),
                        },
                    )

            if thread_adapter is None:
                return

            thread_adapter.push(event=evt)
            self._update_thread_status_from_event(path=path, event=evt)

        async def _close_thread_context_session(
            context: ChatThreadContext | None,
        ) -> None:
            if context is None:
                return
            try:
                await context.__aexit__(None, None, None)
            except Exception as ex:
                logger.warning(
                    "unable to close chat session context for thread %s",
                    path,
                    exc_info=ex,
                )

        async def _activate_thread_context(
            context: ChatThreadContext,
        ) -> None:
            nonlocal thread_context

            await context.__aenter__()

            previous = self._thread_contexts.get(path)
            self._thread_contexts[path] = context
            thread_context = context

            if previous is not None and previous is not context:
                await _close_thread_context_session(previous)

        try:
            received = None

            while True:
                logger.debug(f"waiting for message on thread {path}")
                received = await messages.recv()

                logger.debug(f"received message on thread {path}: {received.type}")

                chat_with_participant = received.from_participant

                thread_attributes = {
                    "agent_name": self.name,
                    "agent_participant_id": self.room.local_participant.id,
                    "agent_participant_name": self.room.local_participant.get_attribute(
                        "name"
                    ),
                    "remote_participant_id": chat_with_participant.id,
                    "remote_participant_name": chat_with_participant.get_attribute(
                        "name"
                    ),
                    "path": path,
                }

                if current_file != chat_with_participant.get_attribute("current_file"):
                    logger.info(
                        f"{chat_with_participant.get_attribute('name')} is now looking at {chat_with_participant.get_attribute('current_file')}"
                    )
                    current_file = chat_with_participant.get_attribute("current_file")

                if thread is None:
                    with tracer.start_as_current_span("chatbot.thread.open") as span:
                        span.set_attributes(thread_attributes)

                        thread_adapter = await self.open_thread(path=path)
                        thread = thread_adapter.thread
                        if thread is None:
                            raise RoomException("thread was not opened")

                        thread_context = await self.create_thread_context(
                            path=path,
                            thread=thread,
                            participants=get_online_participants(
                                room=self.room, thread=thread
                            ),
                            event_handler=lambda evt: handle_event(
                                evt, chat_with_participant
                            ),
                        )
                        await _activate_thread_context(thread_context)

                        thread_adapter.append_messages(
                            context=thread_context.session,
                        )

                if received.type == "opened":
                    if not opened:
                        opened = True

                        if thread_context is not None:
                            await self._safe_invoke_thread_event(
                                event_name="open",
                                thread_context=thread_context,
                                handler=self.on_thread_open,
                            )

                        await self._greet(
                            thread_context=thread_context,
                            participant=chat_with_participant,
                            thread_adapter=thread_adapter,
                            thread_attributes=thread_attributes,
                        )
                elif received.type == "clear":
                    thread_context = await self.create_thread_context(
                        path=path,
                        thread=thread,
                        participants=get_online_participants(
                            room=self.room, thread=thread
                        ),
                        event_handler=lambda evt: handle_event(
                            evt, chat_with_participant
                        ),
                    )
                    await _activate_thread_context(thread_context)
                    messages_element: Element = thread.root.get_children_by_tag_name(
                        "messages"
                    )[0]
                    for child in list(messages_element.get_children()):
                        child.delete()

                    if thread_context is not None:
                        await self._safe_invoke_thread_event(
                            event_name="clear",
                            thread_context=thread_context,
                            handler=self.on_thread_clear,
                        )

                elif received.type == "chat":
                    if thread is None:
                        logger.info("thread is not open", extra={"path": path})
                        break

                    logger.debug(
                        "chatbot received a chat",
                        extra={
                            "context": (
                                thread_context.context_id
                                if thread_context is not None
                                else None
                            ),
                            "participant_id": self.room.local_participant.id,
                            "participant_name": self.room.local_participant.get_attribute(
                                "name"
                            ),
                            "text": received.message["text"],
                        },
                    )

                if received is not None and received.type == "chat":
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attributes(thread_attributes)
                        span.set_attribute("role", "user")
                        span.set_attribute(
                            "from_participant_name",
                            chat_with_participant.get_attribute("name"),
                        )

                        raw_attachments = received.message.get("attachments")
                        attachments = (
                            raw_attachments if isinstance(raw_attachments, list) else []
                        )
                        span.set_attribute("attachments", json.dumps(attachments))

                        text = received.message["text"]
                        span.set_attributes({"text": text})

                        if (
                            thread_adapter is not None
                            and self._should_store_received_chat_message(
                                message=received.message
                            )
                        ):
                            thread_adapter.write_text_message(
                                text=text,
                                participant=chat_with_participant,
                                attachments=attachments,
                            )

                        try:
                            if thread_context is None:
                                thread_context = await self.create_thread_context(
                                    path=path,
                                    thread=thread,
                                    participants=get_online_participants(
                                        room=self.room, thread=thread
                                    ),
                                    event_handler=lambda evt: handle_event(
                                        evt, chat_with_participant
                                    ),
                                )
                                await _activate_thread_context(thread_context)
                            else:
                                thread_context.participants = get_online_participants(
                                    room=self.room, thread=thread
                                )

                            await self.set_thread_status(
                                path=path,
                                status="Thinking",
                                mode=self.processing_thread_status_mode(
                                    path=path,
                                    thread_context=thread_context,
                                ),
                            )

                            result = await self.on_chat_received(
                                thread_context=thread_context,
                                from_participant=chat_with_participant,
                                message=received.message,
                            )
                            received.result.set_result(result)

                        except Exception as e:
                            handled_error: Exception = e
                            if isinstance(e, aiohttp.WSServerHandshakeError):
                                handled_error = RoomException(
                                    OpenAIResponsesSessionContext._handshake_error_message(
                                        e
                                    ),
                                    status_code=e.status,
                                )

                            if isinstance(handled_error, RoomException):
                                logger.warning(
                                    "A room error was encountered", exc_info=e
                                )
                            else:
                                logger.error("An error was encountered", exc_info=e)

                            text = self._chat_error_message(error=handled_error)
                            await self._send_and_save_chat(
                                thread_adapter=thread_adapter,
                                to=chat_with_participant,
                                id=str(uuid.uuid4()),
                                text=text,
                                thread_attributes=thread_attributes,
                            )
                            received.result.set_result(text)

                        finally:
                            await self.clear_thread_status(path=path)

        finally:

            async def cleanup():
                if self.room is not None:
                    logger.info(f"thread was ended {path}")
                    logger.info("chatbot thread ended", extra={"path": path})

                    thread_context = self._thread_contexts.pop(path, None)
                    if thread_context is not None:
                        await self._safe_invoke_thread_event(
                            event_name="close",
                            thread_context=thread_context,
                            handler=self.on_thread_close,
                        )
                        await _close_thread_context_session(thread_context)

                    await self.close_thread(path=path)

            await asyncio.shield(cleanup())

    def _get_message_channel(self, key: str) -> Chan[_QueuedChatMessage]:
        if key not in self._message_channels:
            chan = Chan[_QueuedChatMessage]()
            self._message_channels[key] = chan

        chan = self._message_channels[key]

        return chan

    async def stop(self):
        room = self._room
        message_channels = list(self._message_channels.values())
        for channel in message_channels:
            channel.close()

        thread_tasks = list(self._thread_tasks.values())
        for thread in thread_tasks:
            thread.cancel()

        if len(thread_tasks) > 0:
            await asyncio.gather(*thread_tasks, return_exceptions=True)

        self._thread_tasks.clear()
        self._message_channels.clear()

        await self._clear_all_thread_statuses()
        await self._cancel_thread_list_background_tasks()
        await self._close_thread_list_document(room=room)
        await super().stop()

    async def _queue_chat_message(
        self,
        *,
        context: ToolContext,
        path: str,
        message: dict[str, Any],
        wait_for_result: bool,
        store: bool = True,
    ) -> Optional[str]:
        text_value = message.get("text")
        text = text_value if isinstance(text_value, str) else ""
        attachments: list[dict[str, str]] = []
        raw_attachments = message.get("attachments")
        if isinstance(raw_attachments, list):
            for attachment in raw_attachments:
                if not isinstance(attachment, dict):
                    continue
                path_value = attachment.get("path")
                if not isinstance(path_value, str):
                    continue
                normalized_path = path_value.strip()
                if normalized_path == "":
                    continue
                attachments.append({"path": normalized_path})

        payload = {
            **message,
            "path": path,
            "text": text,
            "attachments": attachments,
            "store": store,
        }
        qm = _QueuedChatMessage(
            type="chat",
            message=payload,
            from_participant=context.on_behalf_of or context.caller,
        )

        messages = self._ensure_thread(path=path)
        messages.send_nowait(qm)

        if wait_for_result:
            return await qm.result
        return None

    @staticmethod
    def _ensure_thread_document_element(
        *,
        thread: MeshDocument,
        tag_name: str,
    ) -> Element:
        elements = thread.root.get_children_by_tag_name(tag_name)
        if len(elements) > 0:
            return elements[0]
        return thread.root.append_child(tag_name=tag_name)

    async def _seed_new_thread_members(
        self,
        *,
        path: str,
        members: list[Participant | str],
    ) -> None:
        thread = await self.room.sync.open(path=path, schema=thread_schema)
        try:
            members_element = self._ensure_thread_document_element(
                thread=thread,
                tag_name="members",
            )

            existing_members: set[str] = set()
            for child in members_element.get_children():
                if child.tag_name != "member":
                    continue
                member_name = child.get_attribute("name")
                if not isinstance(member_name, str):
                    continue
                normalized_name = member_name.strip()
                if normalized_name == "":
                    continue
                existing_members.add(normalized_name)

            for participant in members:
                participant_name = (
                    participant.get_attribute("name")
                    if isinstance(participant, Participant)
                    else participant
                )
                if not isinstance(participant_name, str):
                    continue

                normalized_name = participant_name.strip()
                if normalized_name == "" or normalized_name in existing_members:
                    continue

                members_element.append_child(
                    tag_name="member",
                    attributes={"name": normalized_name},
                )
                existing_members.add(normalized_name)
        finally:
            await self.room.sync.close(path=path)

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
            if outer._thread_list_dir() is None:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": 0,
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "thread list is not enabled for this chatbot",
                        "read_file_hint": read_file_hint,
                    }
                )

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
            if outer._thread_list_dir() is None:
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": pattern,
                        "ignore_case": ignore_case,
                        "message": "thread list is not enabled for this chatbot",
                        "read_file_hint": read_file_hint,
                    }
                )

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

    async def get_exposed_toolkits(self) -> list[Toolkit]:
        exposed_toolkits = await super().get_exposed_toolkits()

        @tool(
            description=f"sends a chat to {self.room.local_participant.get_attribute('name')} and gets the response"
        )
        async def ask(context: ToolContext, *, path: str, text: str) -> str:
            response = await self._queue_chat_message(
                context=context,
                path=path,
                message={"text": text},
                wait_for_result=True,
            )
            if response is None:
                raise RoomException("chat response was empty")
            return response

        tools_schema = {
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
        tools_schema = ensure_strict_json_schema(tools_schema)

        outer = self

        class NewThreadTool(FunctionTool):
            def __init__(self):
                super().__init__(
                    name="new_thread",
                    description=f"creates a new thread for {outer.room.local_participant.get_attribute('name')} and sends the first chat message, returning the new thread path. The thread list entry is named and added asynchronously.",
                    input_schema=tools_schema,
                )

            async def execute(
                self,
                context: ToolContext,
                *,
                message: dict[str, Any],
            ) -> JsonContent:
                text_value = message.get("text")
                text = text_value if isinstance(text_value, str) else ""
                attachment_paths: list[str] = []
                raw_attachments = message.get("attachments")
                if isinstance(raw_attachments, list):
                    for attachment in raw_attachments:
                        if not isinstance(attachment, dict):
                            continue
                        path_value = attachment.get("path")
                        if not isinstance(path_value, str):
                            continue
                        normalized_path = path_value.strip()
                        if normalized_path == "":
                            continue
                        attachment_paths.append(normalized_path)

                if text.strip() == "" and len(attachment_paths) == 0:
                    raise RoomException(
                        "chat.new_thread requires non-empty text or at least one attachment"
                    )

                payload = {
                    **message,
                    "text": text,
                    "attachments": [{"path": path} for path in attachment_paths],
                }

                path = await outer._new_thread_path()
                await outer._seed_new_thread_members(
                    path=path,
                    members=[
                        outer.room.local_participant,
                        context.on_behalf_of or context.caller,
                    ],
                )
                outer._begin_pending_thread_list_entry(path=path)
                await outer._queue_chat_message(
                    context=context,
                    path=path,
                    message=payload,
                    wait_for_result=False,
                )
                outer._schedule_pending_thread_list_entry(
                    context=context,
                    path=path,
                    text=text,
                )
                return JsonContent(json={"path": path})

        tools: list[FunctionTool] = [
            ask,
            NewThreadTool(),
            *self._build_thread_list_tools(),
        ]

        chatbot_toolkit = Toolkit(
            name="chat",
            description=f"tools for interacting with {self.name}",
            public=False,
            tools=tools,
            validation_mode="content_types",
        )

        exposed_toolkits.append(chatbot_toolkit)
        return exposed_toolkits

    def _ensure_thread(self, path: str) -> Chan[_QueuedChatMessage]:
        messages = self._get_message_channel(path)
        if path not in self._thread_tasks or self._thread_tasks[path].done():

            def thread_done(task: asyncio.Task):
                self._thread_tasks.pop(path, None)
                self._message_channels.pop(path, None)
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"The chat thread ended with an error {e}", exc_info=e)

            logger.debug(f"spawning chat thread for {path}")
            task = asyncio.create_task(self._spawn_thread(messages=messages, path=path))
            task.add_done_callback(thread_done)

            self._thread_tasks[path] = task

        return messages

    def on_message(
        self, message: RoomMessage, from_participant: Optional[RemoteParticipant] = None
    ) -> _QueuedChatMessage | None:
        if (
            message.type == "chat"
            or message.type == "opened"
            or message.type == "clear"
        ):
            path = message.message["path"]

            logger.debug(f"queued incoming message for thread {path}: {message.type}")

            if from_participant is None:
                for participant in self._room.messaging.get_participants():
                    if participant.id == message.from_participant_id:
                        from_participant = participant
                        break

                if from_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    return

            msg = _QueuedChatMessage(
                type=message.type,
                message=message.message,
                from_participant=from_participant,
            )

            messages = self._ensure_thread(path=path)

            messages.send_nowait(msg)

            return msg

        elif message.type == "steer":
            path = message.message["path"]

            if from_participant is None:
                for participant in self._room.messaging.get_participants():
                    if participant.id == message.from_participant_id:
                        from_participant = participant
                        break

                if from_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    return

            async def handle_steer():
                thread_context = self._thread_contexts.get(path)
                if thread_context is None:
                    logger.warning(
                        f"unable to process steer message for thread {path}: thread is not open"
                    )
                    return

                await self._safe_invoke_chat_event(
                    event_name="steer",
                    thread_context=thread_context,
                    from_participant=from_participant,
                    message=message.message,
                    handler=self.on_thread_steer,
                )

            task = asyncio.create_task(handle_steer())

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as ex:
                    logger.error(
                        f"unable to process steer message for thread {path}",
                        exc_info=ex,
                    )

            task.add_done_callback(on_done)

        elif message.type == "approved" or message.type == "rejected":
            path = message.message["path"]

            if from_participant is None:
                for participant in self._room.messaging.get_participants():
                    if participant.id == message.from_participant_id:
                        from_participant = participant
                        break

                if from_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    return

            async def handle_approval_response():
                thread_context = self._thread_contexts.get(path)
                if thread_context is None:
                    logger.warning(
                        f"unable to process {message.type} for thread {path}: thread is not open"
                    )
                    return

                if message.type == "approved":
                    await self._safe_invoke_chat_event(
                        event_name="approved",
                        thread_context=thread_context,
                        from_participant=from_participant,
                        message=message.message,
                        handler=self.on_approved,
                    )
                else:
                    await self._safe_invoke_chat_event(
                        event_name="rejected",
                        thread_context=thread_context,
                        from_participant=from_participant,
                        message=message.message,
                        handler=self.on_rejected,
                    )

            task = asyncio.create_task(handle_approval_response())

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as ex:
                    logger.error(
                        f"unable to process {message.type} message for thread {path}",
                        exc_info=ex,
                    )

            task.add_done_callback(on_done)

        elif message.type == "cancel":
            path = message.message["path"]

            async def handle_cancel():
                thread_context = self._thread_contexts.get(path)
                if thread_context is not None:
                    await self._safe_invoke_thread_event(
                        event_name="cancel",
                        thread_context=thread_context,
                        handler=self.on_thread_cancel,
                    )

                await self.cancel_thread_task(
                    path=path,
                    thread_context=thread_context,
                )

            task = asyncio.create_task(handle_cancel())

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as ex:
                    logger.error(
                        f"unable to process cancel message for thread {path}",
                        exc_info=ex,
                    )

            task.add_done_callback(on_done)

        elif message.type == "typing":

            def callback(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception:
                    pass

            async def remove_timeout(id: str):
                await asyncio.sleep(1)
                self._is_typing.pop(id)

            if message.from_participant_id in self._is_typing:
                self._is_typing[message.from_participant_id].cancel()

            timeout = asyncio.create_task(
                remove_timeout(id=message.from_participant_id)
            )
            timeout.add_done_callback(callback)

            self._is_typing[message.from_participant_id] = timeout

        return None

    async def start(self, *, room):
        await super().start(room=room)

        logger.debug("Starting chatbot")

        if self._threading_mode is not None:
            await self.room.local_participant.set_attribute(
                "meshagent.chatbot.threading", self._threading_mode
            )

        thread_dir = self._thread_list_dir()
        if thread_dir is not None:
            await self.room.local_participant.set_attribute(
                "meshagent.chatbot.thread-dir", thread_dir
            )

        thread_list_path = self._thread_list_index_path()
        if thread_list_path is not None:
            await self.room.local_participant.set_attribute(
                "meshagent.chatbot.thread-list", thread_list_path
            )

        await self.room.local_participant.set_attribute(
            "empty_state_title", self._empty_state_title
        )

        await self._open_thread_list_document()

        room.messaging.on("message", self.on_message)

        if self._auto_greet_message is not None:

            def on_participant_added(participant: RemoteParticipant):
                # will spawn the initial thread
                self._get_message_channel(participant.id)

            room.messaging.on("participant_added", on_participant_added)

        logger.debug("Enabling chatbot messaging")
        await room.messaging.enable()


class ChatBot(ChatBotBase):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires: Optional[list[Requirement]] = None,
        llm_adapter: LLMAdapter,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        decision_model: Optional[str] = None,
        decision_options: Optional[dict] = None,
        always_reply: Optional[bool] = None,
        skill_dirs: Optional[list[str]] = None,
        thread_dir: Optional[str] = None,
        threading_mode: Optional[str] = None,
        thread_name_rules: Optional[list[str]] = None,
    ):
        self._llm_adapter = llm_adapter
        if decision_model is None:
            decision_model = "gpt-5.4"
            decision_options = {"reasoning": {"effort": "none"}}

        self._decision_model = decision_model
        self._decision_options = decision_options

        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            toolkits=toolkits,
            rules=rules,
            client_rules=client_rules,
            auto_greet_message=auto_greet_message,
            empty_state_title=empty_state_title,
            annotations=annotations,
            always_reply=always_reply,
            skill_dirs=skill_dirs,
            thread_dir=thread_dir,
            threading_mode=threading_mode,
            thread_name_rules=thread_name_rules,
        )

    def default_model(self) -> str:
        return self._llm_adapter.default_model()

    def bind_runtime_credentials(self, *, room: RoomClient) -> None:
        super().bind_runtime_credentials(room=room)
        self._llm_adapter = self._llm_adapter.with_runtime_api_key(
            api_key=self.resolve_runtime_api_key(room=room)
        )

    def thread_name_adapter(self) -> Optional[LLMAdapter]:
        return self._llm_adapter

    def _status_event_details(
        self, *, event: dict
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        normalized_event = None
        event_type = event.get("type")
        if event_type in ("agent.event", "codex.event"):
            normalized_event = event
        else:
            normalized_event = response_event_to_agent_event(event)

        if not isinstance(normalized_event, dict):
            return None, None, None

        kind = normalized_event.get("kind")
        if not isinstance(kind, str):
            kind = ""
        kind = kind.strip().lower()
        if kind not in (
            "exec",
            "tool",
            "web",
            "search",
            "diff",
            "image",
        ):
            return None, None, None

        state = normalized_event.get("state")
        if not isinstance(state, str):
            state = ""
        state = state.strip().lower()

        key = None
        for candidate in (
            normalized_event.get("correlation_key"),
            normalized_event.get("event_key"),
            normalized_event.get("item_id"),
            normalized_event.get("name"),
            normalized_event.get("method"),
        ):
            if isinstance(candidate, str) and candidate.strip() != "":
                key = candidate.strip()
                break

        text = None
        for candidate in (
            normalized_event.get("headline"),
            normalized_event.get("summary"),
            normalized_event.get("name"),
            normalized_event.get("method"),
        ):
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized != "":
                    text = normalized
                    break

        return key, state, text

    def _update_thread_status_from_event(self, *, path: str, event: dict) -> None:
        key, state, text = self._status_event_details(event=event)
        if state is None:
            return

        is_active = state in (
            "queued",
            "in_progress",
            "running",
            "pending",
            "searching",
        )
        if is_active:
            if text is None:
                return
            if key is not None:
                self._thread_status_keys[path] = key
            self._set_thread_status_nowait(path=path, status=text)
            return

        if key is not None:
            tracked = self._thread_status_keys.get(path)
            if tracked is not None and tracked == key:
                self._thread_status_keys.pop(path, None)
                self._set_thread_status_nowait(path=path, status="Thinking")
            return

        if state in ("completed", "failed", "cancelled"):
            self._set_thread_status_nowait(path=path, status="Thinking")

    async def create_thread_context(
        self,
        *,
        path: str,
        thread: MeshDocument,
        participants: list[RemoteParticipant],
        event_handler: Callable[[dict], None],
    ) -> ChatThreadContext:
        return ChatThreadContext(
            path=path,
            thread=thread,
            participants=participants,
            event_handler=event_handler,
            session=await self.init_session(),
        )

    def _create_default_session(self) -> AgentSessionContext:
        context = self._llm_adapter.create_session()
        context.append_rules(self._rules)
        return context

    async def init_session(self) -> AgentSessionContext:
        legacy_initializer = type(self).init_chat_context
        if legacy_initializer is not ChatBot.init_chat_context:
            cls = type(self)
            if cls not in _legacy_chatbot_init_chat_context_warned:
                warnings.warn(
                    (
                        f"{cls.__name__}.init_chat_context() is deprecated and will be removed in a future release. "
                        "Override init_session() instead."
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                _legacy_chatbot_init_chat_context_warned.add(cls)
            return await legacy_initializer(self)
        return self._create_default_session()

    # Backwards compatibility for existing subclasses overriding init_chat_context.
    async def init_chat_context(self) -> AgentSessionContext:
        warnings.warn(
            "init_chat_context() is deprecated and will be removed in a future release. Use init_session() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._create_default_session()

    async def prepare_llm_context(self, *, thread_context: ChatThreadContext):
        """
        called prior to sending the request to the LLM in case the agent needs to modify the context prior to sending
        """
        pass

    def prepare_chat_context(self, *, chat_context: AgentSessionContext):
        pass

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ) -> list[Toolkit]:
        toolkits = await super().get_thread_toolkits(
            thread_context=thread_context, participant=participant
        )

        if isinstance(self._llm_adapter, OpenAIResponsesAdapter):
            toolkits.insert(
                0,
                Toolkit(
                    name="reasoning",
                    tools=[
                        ChatBotReasoningTool(
                            room=self._room,
                            thread_context=thread_context,
                        )
                    ],
                ),
            )

        return toolkits

    async def should_reply(
        self,
        *,
        context: ChatThreadContext,
        has_more_than_one_other_user: bool,
        toolkits: list[Toolkit],
        from_user: RemoteParticipant,
        online: list[Participant],
    ) -> bool:
        if not has_more_than_one_other_user or self._always_reply:
            return True

        online_set = {}

        all_members = []
        online_members = []

        for m in self.get_thread_members(thread=context.thread):
            all_members.append(m)

        for o in online:
            if o.get_attribute("name") not in online_set:
                online_set[o.get_attribute("name")] = True
                online_members.append(o.get_attribute("name"))

        logger.info(
            "multiple participants detected, checking whether agent should reply to conversation"
        )

        toolkits_json = ""
        for toolkit in toolkits:
            toolkits_json = (
                toolkits_json
                + f"\n{toolkit.name} ({toolkit.title}): {toolkit.description}"
            )
            for t in toolkit.tools:
                toolkits_json = (
                    toolkits_json + f"\n - {t.name} ({t.title}): {t.description}"
                )

        cloned_context = context.session.copy()
        async with cloned_context:
            cloned_context.replace_rules(
                rules=[
                    "examine the conversation so far and return whether the user is expecting a reply from you or another user as the next message in the conversation",
                    f'your name (the assistant) is "{self.room.local_participant.get_attribute("name")}"',
                    "if the user mentions a person with another name, they aren't talking to you unless they also mention you",
                    "if the user poses a question to everyone, they are talking to you",
                    "to help identify the different users in the conversation, every message in the thread will start with '{user_name} said at {time}'",
                    f"members of thread are currently {all_members}",
                    f"users online currently are {online_members}",
                    "if in doubt, reply to the user",
                    f"if the user is asking for something that these toolkits can do, they want an answer from you: {toolkits_json}",
                    "if the user they appear to be talking to is offline, then they probably are talking to you",
                ]
            )
            response = await self._llm_adapter.next(
                context=cloned_context,
                caller=self._room.local_participant,
                model=self._decision_model or self._llm_adapter.default_model(),
                options=self._decision_options,
                on_behalf_of=from_user,
                toolkits=[],
                output_schema={
                    "type": "object",
                    "required": ["reasoning", "expecting_assistant_reply", "next_user"],
                    "additionalProperties": False,
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "explain why you think the user was or was not expecting you to reply",
                        },
                        "next_user": {
                            "type": "string",
                            "description": "who would be expected to send the next message in the conversation",
                        },
                        "expecting_assistant_reply": {"type": "boolean"},
                    },
                },
            )

        logger.info(f"should reply check returned {response}")

        parsed_response = response
        if isinstance(parsed_response, str):
            try:
                parsed_response = json.loads(parsed_response)
            except json.JSONDecodeError:
                logger.warning(
                    "should reply check returned unstructured text, defaulting to reply"
                )
                return True

        if not isinstance(parsed_response, dict):
            logger.warning(
                "should reply check returned %s, defaulting to reply",
                type(parsed_response).__name__,
            )
            return True

        expecting_assistant_reply = parsed_response.get("expecting_assistant_reply")
        if isinstance(expecting_assistant_reply, bool):
            return expecting_assistant_reply

        logger.warning(
            "should reply check returned payload without a boolean expecting_assistant_reply, defaulting to reply"
        )
        return True

    async def on_thread_open(self, *, thread_context: ChatThreadContext):
        await self.clear_thread_status(path=thread_context.path)

    async def on_thread_clear(self, *, thread_context: ChatThreadContext):
        await self.clear_thread_status(path=thread_context.path)

    async def on_thread_cancel(self, *, thread_context: ChatThreadContext):
        await self.clear_thread_status(path=thread_context.path)

    async def on_thread_close(self, *, thread_context: ChatThreadContext):
        await self.clear_thread_status(path=thread_context.path)

    async def on_chat_received(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> Optional[str]:
        rules = await self.get_rules(
            thread_context=thread_context,
            participant=from_participant,
        )
        thread_context.session.replace_rules(rules)
        self._append_current_file_context(
            thread_context=thread_context,
            participant=from_participant,
        )

        attachments = message.get("attachments", [])
        for attachment in attachments:
            thread_context.session.append_assistant_message(
                message=f"the user attached a file at the path '{attachment['path']}'"
            )

        text = message["text"]
        iso_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        thread_context.session.append_user_message(
            message=self.format_message(
                user_name=from_participant.get_attribute("name"),
                message=text,
                iso_timestamp=iso_timestamp,
            )
        )

        model = message.get("model", self.default_model())

        with tracer.start_as_current_span("chatbot.llm"):
            with tracer.start_as_current_span("get_thread_toolkits"):
                thread_toolkits = await self.get_thread_toolkits(
                    thread_context=thread_context,
                    participant=from_participant,
                )

            await self.prepare_llm_context(thread_context=thread_context)

            message_toolkits = [*thread_toolkits]

        online = await self.get_online_participants(
            thread=thread_context.thread, exclude=[self.room.local_participant]
        )

        for participant in get_online_participants(
            room=self._room, thread=thread_context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": True, "path": thread_context.path},
            )

        has_more_than_one_other_user = False

        thread_participants = []

        for member_name in self.get_thread_members(thread=thread_context.thread):
            thread_participants.append(member_name)
            if member_name != self._room.local_participant.get_attribute(
                "name"
            ) and member_name != from_participant.get_attribute("name"):
                has_more_than_one_other_user = True
                break

        thread_context.session.metadata["thread_participants"] = thread_participants

        reply = await self.should_reply(
            has_more_than_one_other_user=has_more_than_one_other_user,
            online=online,
            context=thread_context,
            toolkits=message_toolkits,
            from_user=from_participant,
        )

        for participant in get_online_participants(
            room=self._room, thread=thread_context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": False, "path": thread_context.path},
            )

        if not reply:
            return None

        self.prepare_chat_context(chat_context=thread_context.session)
        self._touch_thread_in_index(path=thread_context.path)

        return await self._llm_adapter.next(
            context=thread_context.session,
            caller=self._room.local_participant,
            toolkits=message_toolkits,
            event_handler=thread_context.emit,
            model=model,
            on_behalf_of=from_participant,
        )

    def _normalize_current_file(self, *, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        trimmed = value.strip()
        if trimmed == "":
            return None
        return trimmed

    def _participant_name_for_context(self, *, participant: Participant) -> str:
        name = participant.get_attribute("name")
        if isinstance(name, str) and name.strip() != "":
            return name
        return "the user"

    def _append_current_file_context(
        self,
        *,
        thread_context: ChatThreadContext,
        participant: Participant,
    ) -> None:
        metadata = thread_context.session.metadata
        state = metadata.get("current_file_by_participant")
        if not isinstance(state, dict):
            state = {}
            metadata["current_file_by_participant"] = state

        current_file = self._normalize_current_file(
            value=participant.get_attribute("current_file")
        )
        participant_key = participant.id
        had_previous = participant_key in state
        previous_file = state.get(participant_key)

        if had_previous and previous_file == current_file:
            return
        if not had_previous and current_file is None:
            return

        participant_name = self._participant_name_for_context(participant=participant)
        if current_file is None:
            thread_context.session.append_assistant_message(
                message=f"{participant_name} is not currently viewing any files."
            )
        else:
            thread_context.session.append_assistant_message(
                message=(
                    f"{participant_name} is currently viewing the file at "
                    f"the path: {current_file}"
                )
            )

        state[participant_key] = current_file
