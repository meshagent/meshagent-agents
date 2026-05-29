from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, cast
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
from pydantic_core import from_json as pydantic_core_from_json

from meshagent.api import (
    DatasetJson,
    DatasetOptimizeConfig,
    LANCE_ZSTD_FIELD_METADATA,
    Participant,
    RoomException,
    RoomClient,
)
from meshagent.api.messaging import TextContent
from meshagent.tools import Toolkit, tool

from .agent_event_reader import AgentEventReaderCallbacks
from .context import AgentSessionContext, SessionUsage
from .images_dataset import ImagesDataset
from .messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_REASONING_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_AUDIO_GENERATION_DELTA,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_FAILED,
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
    AgentAudioGenerationDelta,
    AgentAudioTranscriptionCompleted,
    AgentAudioTranscriptionDelta,
    AgentAudioTranscriptionFailed,
    AgentAudioTranscriptionStarted,
    AgentContextCompacted,
    AgentError,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentGeneratedImage,
    AgentImageGenerationCompleted,
    AgentImageGenerationFailed,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentRealtimeAudioChunk,
    AgentRealtimeAudioCommit,
    AgentThreadMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentThreadEvent,
    AgentThreadStatus,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallArgumentsDelta,
    AgentToolCallLogDelta,
    AgentToolCallPending,
    AgentToolCallApprovalRequested,
    AgentToolCallStarted,
    AgentUsageUpdated,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnInterruptAccepted,
    TurnInterrupted,
    TurnStart,
    TurnStartAccepted,
    TurnStartRejected,
    TurnStarted,
    TurnSteer,
    TurnSteerAccepted,
    TurnSteered,
    TurnSteerRejected,
    parse_agent_message,
    scrub_agent_message_for_storage,
)
from .stream_content_accumulator import accumulate_text_delta
from .thread_storage import (
    ThreadListEntry,
    ThreadListEvent,
    ThreadListPage,
    ThreadStorage,
    thread_dir_for_namespace,
)

if TYPE_CHECKING:
    from .adapter import LLMAdapter
    from meshagent.api.room_server_client import DatasetsClient

logger = logging.getLogger("agent.dataset_thread_storage")

_DATASET_THREAD_URL_PREFIX = "dataset://"
_IMAGE_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


class _HybridDatasetThreadStorageMethod:
    def __init__(self, instance_method, class_method) -> None:
        self._instance_method = instance_method
        self._class_method = class_method

    def __get__(self, instance, owner):
        if instance is None:
            return self._class_method.__get__(None, owner)
        return self._instance_method.__get__(instance, owner)


def _is_uuid_like(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _infer_usage_model(usage: dict[str, float]) -> str:
    models: set[str] = set()
    for key in usage:
        if "." not in key:
            continue
        model, metric = key.rsplit(".", 1)
        if model != "" and metric != "":
            models.add(model)
    if len(models) == 1:
        return next(iter(models))
    return ""


_TERMINAL_REASON_TO_STATUS = {
    "completed": "completed",
    "cancelled": "cancelled",
    "failed": "failed",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_thread_list_limit_offset(
    *,
    limit: int,
    offset: int,
) -> tuple[int, int]:
    return max(1, min(200, int(limit))), max(0, int(offset))


def _parse_thread_list_datetime(*, value: str) -> datetime:
    raw = value.strip()
    if raw == "":
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sort_thread_list_entries(entries: list[ThreadListEntry]) -> list[ThreadListEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            _parse_thread_list_datetime(value=entry.modified_at),
            _parse_thread_list_datetime(value=entry.created_at),
            entry.path,
        ),
        reverse=True,
    )


def _dataset_sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_positive_dimension(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _parse_image_dimensions_from_size(value: Any) -> tuple[int | None, int | None]:
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


def _merge_tool_arguments(
    *,
    current: dict[str, Any] | None,
    update: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(current) if current is not None else {}
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_tool_arguments(current=existing, update=value)
        else:
            merged[key] = value
    return merged


def _partial_json_tool_arguments(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped == "" or stripped[0] not in "{[":
        return None

    try:
        parsed = pydantic_core_from_json(
            stripped.encode("utf-8"),
            allow_partial=True,
        )
    except ValueError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _tool_arguments_from_delta_text(
    *,
    tool: str | None,
    current: dict[str, Any] | None,
    text: str,
) -> dict[str, Any] | None:
    partial_arguments = _partial_json_tool_arguments(text)
    if partial_arguments is not None:
        return _merge_tool_arguments(current=current, update=partial_arguments)

    if tool is None:
        return None

    normalized_tool = tool.strip().lower()
    if normalized_tool == "apply_patch":
        patch = text.strip()
        if patch != "":
            return _merge_tool_arguments(current=current, update={"patch": patch})

    return None


def _mime_type_from_output_format(output_format: Any) -> str:
    if not isinstance(output_format, str):
        return "image/png"

    normalized = output_format.strip().lower().lstrip(".")
    if normalized == "":
        return "image/png"
    if normalized == "jpg":
        normalized = "jpeg"
    return f"image/{normalized}"


def _image_generation_status(
    *,
    message: AgentThreadMessage,
) -> Literal["pending", "in_progress", "completed", "failed"]:
    if isinstance(message, AgentImageGenerationCompleted):
        return "completed"
    if isinstance(message, AgentImageGenerationFailed):
        return "failed"
    if isinstance(message, AgentImageGenerationPartial):
        return "in_progress"
    return "pending"


def _normalize_path_parts(*, path: str) -> list[str]:
    parts = [part for part in path.strip().split("/") if part != ""]
    if len(parts) == 0:
        raise ValueError("dataset thread storage path must include a table name")
    return parts


@dataclass(frozen=True, slots=True)
class _DatasetThreadStoragePath:
    url: str
    table_path: str


def _normalize_dataset_thread_storage_path(*, path: str) -> _DatasetThreadStoragePath:
    normalized = path.strip()
    if not normalized.startswith(_DATASET_THREAD_URL_PREFIX):
        raise ValueError("dataset thread storage path must start with dataset://")
    table_path = normalized[len(_DATASET_THREAD_URL_PREFIX) :]
    if table_path == "" or table_path.startswith("/"):
        raise ValueError("dataset thread storage path must use dataset://path")
    if table_path.endswith(".thread"):
        raise ValueError("dataset thread storage path must not end with .thread")
    return _DatasetThreadStoragePath(
        url=f"{_DATASET_THREAD_URL_PREFIX}{table_path}",
        table_path=table_path,
    )


@dataclass(slots=True)
class _StoredThreadRow:
    turn_id: str | None
    item_id: str
    message_type: str | None
    sequence: int
    timestamp: str
    data: dict[str, Any]
    attachment: bytes | None = None


@dataclass(slots=True)
class _QueuedThreadMessage:
    message: AgentThreadMessage
    sender: Participant | None


@dataclass(slots=True)
class _StopQueue:
    future: asyncio.Future[None]


@dataclass(slots=True)
class _FlushQueue:
    future: asyncio.Future[None]


@dataclass(slots=True)
class _ActiveContent:
    kind: Literal["text", "reasoning", "file"]
    turn_id: str
    item_id: str
    message_id: str
    provider: str | None = None
    model: str | None = None
    sender_name: str | None = None
    phase: Literal["commentary", "final_answer"] | None = None
    parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ActiveAudioTranscription:
    turn_id: str
    item_id: str
    message_id: str
    role: str | None = None
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    content_index: int | None = None
    sender_name: str | None = None
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "in_progress"
    parts: list[str] = field(default_factory=list)
    error: AgentError | None = None


@dataclass(slots=True)
class _ActiveToolCall:
    turn_id: str
    item_id: str
    message_id: str
    provider: str | None = None
    model: str | None = None
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    stage: Literal["pending", "in_progress", "started"] | None = None
    argument_delta_text: str = ""
    logs: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class _ActiveImageGeneration:
    turn_id: str
    item_id: str
    message_id: str
    provider: str | None = None
    model: str | None = None
    started: AgentImageGenerationStarted | None = None
    partial: AgentImageGenerationPartial | None = None


class DatasetThreadStorage(ThreadStorage):
    @property
    def is_ephemeral(self) -> bool:
        return False

    @staticmethod
    def _client_from_room(*, room: RoomClient | None, client: DatasetsClient | None):
        if client is not None:
            return client
        if room is None:
            raise ValueError("dataset thread storage requires a dataset client or room")
        return room.datasets

    @classmethod
    def thread_list_path_for_dir(cls, *, thread_dir: str) -> str:
        normalized = thread_dir.strip().rstrip("/")
        if normalized.startswith(_DATASET_THREAD_URL_PREFIX):
            normalized = normalized[len(_DATASET_THREAD_URL_PREFIX) :]
        normalized = normalized.strip("/")
        if normalized == "":
            raise ValueError("dataset thread list directory is required")
        return f"{_DATASET_THREAD_URL_PREFIX}{normalized}/index"

    def thread_list_path(self) -> str:
        return self.thread_list_path_for_dir(thread_dir=self._thread_dir_or_raise())

    async def _list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        datasets = self._client
        thread_dir = self._thread_dir_or_raise()
        normalized_limit, normalized_offset = _normalize_thread_list_limit_offset(
            limit=limit,
            offset=offset,
        )
        entries = await self._read_thread_list_entries(
            client=datasets,
            thread_dir=thread_dir,
        )
        sorted_entries = _sort_thread_list_entries(entries)
        selected = sorted_entries[
            normalized_offset : normalized_offset + normalized_limit
        ]
        return ThreadListPage(
            threads=selected,
            total=len(sorted_entries),
            offset=normalized_offset,
            limit=normalized_limit,
        )

    @classmethod
    async def _list_threads_with_client(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
        namespace: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        resolved_thread_dir = thread_dir_for_namespace(
            thread_dir=thread_dir,
            namespace=namespace,
        )
        normalized_limit, normalized_offset = _normalize_thread_list_limit_offset(
            limit=limit,
            offset=offset,
        )
        entries = await cls._read_thread_list_entries(
            client=client,
            thread_dir=resolved_thread_dir,
        )
        sorted_entries = _sort_thread_list_entries(entries)
        selected = sorted_entries[
            normalized_offset : normalized_offset + normalized_limit
        ]
        return ThreadListPage(
            threads=selected,
            total=len(sorted_entries),
            offset=normalized_offset,
            limit=normalized_limit,
        )

    list_threads = _HybridDatasetThreadStorageMethod(
        _list_threads,
        _list_threads_with_client,
    )

    async def _upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        datasets = self._client
        thread_dir = self._thread_dir_or_raise()
        table_name, namespace = self._thread_list_table(thread_dir=thread_dir)
        await self._ensure_thread_list_table(client=datasets, thread_dir=thread_dir)
        now = _now_iso()
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
        resolved_name = self.default_thread_name(path=path, name=name)
        await datasets.merge(
            table=table_name,
            namespace=namespace,
            on="path",
            records=[
                {
                    "path": path,
                    "name": resolved_name,
                    "created_at": resolved_created_at,
                    "modified_at": resolved_modified_at,
                }
            ],
        )
        return ThreadListEntry(
            name=resolved_name,
            path=path,
            created_at=resolved_created_at,
            modified_at=resolved_modified_at,
        )

    @classmethod
    async def _upsert_thread_with_client(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
        path: str,
        name: str | None = None,
        namespace: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        repository = cls(
            client=client,
            thread_dir=thread_dir_for_namespace(
                thread_dir=thread_dir,
                namespace=namespace,
            ),
        )
        return await repository._upsert_thread(
            path=path,
            name=name,
            created_at=created_at,
            modified_at=modified_at,
        )

    upsert_thread = _HybridDatasetThreadStorageMethod(
        _upsert_thread,
        _upsert_thread_with_client,
    )

    async def _delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None:
        datasets = self._client
        thread_dir = self._thread_dir_or_raise()
        table_name, namespace = self._thread_list_table(thread_dir=thread_dir)
        await self._ensure_thread_list_table(client=datasets, thread_dir=thread_dir)
        await datasets.delete(
            table=table_name,
            namespace=namespace,
            where=f"path = {_dataset_sql_string_literal(path)}",
        )
        if delete_storage:
            normalized_path = _normalize_dataset_thread_storage_path(path=path)
            path_parts = _normalize_path_parts(path=normalized_path.table_path)
            thread_table = path_parts[-1]
            thread_namespace = path_parts[:-1] if len(path_parts) > 1 else None
            await datasets.drop_table(
                name=thread_table,
                namespace=thread_namespace,
                ignore_missing=True,
            )

    @classmethod
    async def _delete_thread_with_client(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
        path: str,
        namespace: str | None = None,
        delete_storage: bool = True,
    ) -> None:
        repository = cls(
            client=client,
            thread_dir=thread_dir_for_namespace(
                thread_dir=thread_dir,
                namespace=namespace,
            ),
        )
        await repository._delete_thread(path=path, delete_storage=delete_storage)

    delete_thread = _HybridDatasetThreadStorageMethod(
        _delete_thread,
        _delete_thread_with_client,
    )

    async def _rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None:
        resolved_name = name.strip()
        if resolved_name == "":
            raise ValueError("thread name is required")
        datasets = self._client
        thread_dir = self._thread_dir_or_raise()
        table_name, namespace = self._thread_list_table(thread_dir=thread_dir)
        await self._ensure_thread_list_table(client=datasets, thread_dir=thread_dir)
        modified_at = _now_iso()
        await datasets.update(
            table=table_name,
            namespace=namespace,
            where=f"path = {_dataset_sql_string_literal(path)}",
            values={"name": resolved_name, "modified_at": modified_at},
        )
        return ThreadListEntry(
            name=resolved_name,
            path=path,
            created_at="",
            modified_at=modified_at,
        )

    @classmethod
    async def _rename_thread_with_client(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
        path: str,
        name: str,
        namespace: str | None = None,
    ) -> ThreadListEntry | None:
        repository = cls(
            client=client,
            thread_dir=thread_dir_for_namespace(
                thread_dir=thread_dir,
                namespace=namespace,
            ),
        )
        return await repository._rename_thread(path=path, name=name)

    rename_thread = _HybridDatasetThreadStorageMethod(
        _rename_thread,
        _rename_thread_with_client,
    )

    async def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]:
        datasets = self._client
        thread_dir = self._thread_dir_or_raise()
        table_name, namespace = self._thread_list_table(thread_dir=thread_dir)
        await self._ensure_thread_list_table(client=datasets, thread_dir=thread_dir)
        previous: dict[str, ThreadListEntry] = {}
        async for event in datasets.watch_table(
            table=table_name,
            namespace=namespace,
            poll_interval_seconds=poll_interval,
        ):
            if event.kind == "ready":
                continue
            if event.table is None:
                continue
            entries = self._entries_from_records(records=event.table.to_pylist())
            if event.phase == "initial":
                for entry in entries:
                    previous[entry.path] = entry
                continue

            change_type = (event.change_type or "").casefold()
            if change_type in {"deleted", "delete", "removed", "remove"}:
                for entry in entries:
                    prior = previous.pop(entry.path, entry)
                    yield ThreadListEvent(
                        type="deleted",
                        path=entry.path,
                        entry=prior,
                    )
                continue

            for entry in entries:
                path = entry.path
                prior = previous.get(path)
                if prior is None:
                    yield ThreadListEvent(type="upserted", path=path, entry=entry)
                elif prior.name != entry.name:
                    yield ThreadListEvent(type="renamed", path=path, entry=entry)
                elif prior != entry:
                    yield ThreadListEvent(type="upserted", path=path, entry=entry)
                previous[path] = entry

    @classmethod
    async def _read_thread_list_entries(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
    ) -> list[ThreadListEntry]:
        table_name, namespace = cls._thread_list_table(thread_dir=thread_dir)
        await cls._ensure_thread_list_table(client=client, thread_dir=thread_dir)
        rows = await client.search(table=table_name, namespace=namespace)
        return cls._entries_from_records(records=rows.to_pylist())

    @classmethod
    async def _ensure_thread_list_table(
        cls,
        *,
        client: DatasetsClient,
        thread_dir: str,
    ) -> None:
        table_name, namespace = cls._thread_list_table(thread_dir=thread_dir)
        schema = pa.schema(
            [
                pa.field("path", pa.string(), nullable=False),
                pa.field("name", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("modified_at", pa.string()),
            ]
        )
        try:
            await client.inspect(table=table_name, namespace=namespace)
        except Exception:
            await client.create_table_with_schema(
                name=table_name,
                schema=schema,
                mode="create_if_not_exists",
                namespace=namespace,
            )

    @classmethod
    def _thread_list_table(cls, *, thread_dir: str) -> tuple[str, list[str] | None]:
        list_path = _normalize_dataset_thread_storage_path(
            path=cls.thread_list_path_for_dir(thread_dir=thread_dir),
        )
        path_parts = _normalize_path_parts(path=list_path.table_path)
        table_name = path_parts[-1]
        namespace = path_parts[:-1] if len(path_parts) > 1 else None
        return table_name, namespace

    @staticmethod
    def _entries_from_records(
        *, records: list[dict[str, Any]]
    ) -> list[ThreadListEntry]:
        entries: list[ThreadListEntry] = []
        for record in records:
            path = record.get("path")
            if not isinstance(path, str) or path.strip() == "":
                continue
            name = record.get("name")
            created_at = record.get("created_at")
            modified_at = record.get("modified_at")
            entries.append(
                ThreadListEntry(
                    name=name.strip() if isinstance(name, str) else "",
                    path=path.strip(),
                    created_at=created_at.strip()
                    if isinstance(created_at, str)
                    else "",
                    modified_at=(
                        modified_at.strip() if isinstance(modified_at, str) else ""
                    ),
                )
            )
        return entries

    @staticmethod
    def default_thread_name(*, path: str, name: str | None = None) -> str:
        provided_name = name.strip() if isinstance(name, str) else ""
        if provided_name != "":
            return provided_name
        return DatasetThreadStorage._default_thread_name(path=path)

    @staticmethod
    def _default_thread_name(*, path: str) -> str:
        normalized_path = path.strip()
        if normalized_path.startswith(_DATASET_THREAD_URL_PREFIX):
            normalized_path = normalized_path[len(_DATASET_THREAD_URL_PREFIX) :]
        filename = normalized_path.rstrip("/").rsplit("/", 1)[-1]
        if filename.endswith(".thread"):
            filename = filename[: -len(".thread")]
        if _is_uuid_like(filename):
            return "New Chat"
        normalized = filename.replace("-", " ").replace("_", " ").strip()
        return normalized.title() if normalized != "" else "New Chat"

    def __init__(
        self,
        *,
        room: RoomClient | None = None,
        client: DatasetsClient | None = None,
        path: str | None = None,
        thread_dir: str | None = None,
        max_append_message_count: int = 25,
        optimize_after_append_count: int = 25,
        persist_deltas: bool = False,
        persist_audio_input: bool = False,
    ) -> None:
        self._client = self._client_from_room(room=room, client=client)
        self._thread_dir = thread_dir
        if path is None:
            self._path = ""
            self._table_name = ""
            self._namespace = None
        else:
            normalized_path = _normalize_dataset_thread_storage_path(path=path)
            self._path = normalized_path.url
            path_parts = _normalize_path_parts(path=normalized_path.table_path)
            self._table_name = path_parts[-1]
            namespace = path_parts[:-1]
            self._namespace = namespace if len(namespace) > 0 else None
        self._max_append_message_count = max_append_message_count
        self._optimize_after_append_count = optimize_after_append_count
        self._persist_deltas = persist_deltas
        self._persist_audio_input = persist_audio_input
        self._appends_since_optimize = 0
        self._optimize_requested = False
        self._optimize_task: asyncio.Task[None] | None = None
        self._ready = False
        self._ready_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[_QueuedThreadMessage | _StopQueue | _FlushQueue] = (
            asyncio.Queue()
        )
        self._processor_task: asyncio.Task[None] | None = None
        self._rows: list[_StoredThreadRow] = []
        self._next_sequence = 0
        self._pending_user_turns: dict[str, _QueuedThreadMessage] = {}
        self._pending_user_turn_rows: dict[str, _StoredThreadRow] = {}
        self._pending_audio_commit_rows_by_turn_id: dict[str, _StoredThreadRow] = {}
        self._active_content_by_item_id: dict[str, _ActiveContent] = {}
        self._active_audio_transcriptions_by_item_id: dict[
            str, _ActiveAudioTranscription
        ] = {}
        self._active_tool_calls_by_item_id: dict[str, _ActiveToolCall] = {}
        self._active_image_generations_by_item_id: dict[
            str, _ActiveImageGeneration
        ] = {}
        self._pending_insert_rows: list[_StoredThreadRow] = []

    def _thread_dir_or_raise(self) -> str:
        if self._thread_dir is None or self._thread_dir.strip() == "":
            raise RuntimeError("dataset thread repository requires thread_dir")
        return self._thread_dir

    @property
    def path(self) -> str:
        return self._path

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def namespace(self) -> list[str] | None:
        return None if self._namespace is None else list(self._namespace)

    @property
    def persist_deltas(self) -> bool:
        return self._persist_deltas

    @property
    def persist_audio_input(self) -> bool:
        return self._persist_audio_input

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("turn_id", pa.string()),
                pa.field("item_id", pa.string(), nullable=False),
                pa.field("type", pa.string()),
                pa.field("sequence", pa.int64(), nullable=False),
                pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field(
                    "data",
                    pa.json_(pa.large_string()),
                    nullable=False,
                    metadata=LANCE_ZSTD_FIELD_METADATA,
                ),
                pa.field(
                    "attachment",
                    pa.large_binary(),
                    metadata=LANCE_ZSTD_FIELD_METADATA,
                ),
            ]
        )

    async def start(self) -> None:
        self._schedule_ready()
        processor_task = self._processor_task
        if processor_task is not None and not processor_task.done():
            return
        self._processor_task = asyncio.create_task(self._process_queue())

    async def wait_until_ready(self) -> None:
        await self._ensure_ready()

    async def flush(self) -> None:
        processor_task = self._processor_task
        if processor_task is None or processor_task.done():
            await self._ensure_ready()
            await self._flush_pending_insert_rows()
            return

        flush_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_FlushQueue(future=flush_future))
        await flush_future

    def unflushed_agent_messages(self) -> list[AgentThreadMessage]:
        messages: list[AgentThreadMessage] = []
        rows = [
            *self._pending_insert_rows,
            *self._pending_user_turn_rows.values(),
            *self._pending_audio_commit_rows_by_turn_id.values(),
        ]
        seen_sequences: set[int] = set()
        for row in sorted(rows, key=lambda stored: stored.sequence):
            if row.sequence in seen_sequences:
                continue
            seen_sequences.add(row.sequence)
            messages.extend(self._messages_from_row(row=row))
        return messages

    async def stop(self) -> None:
        processor_task = self._processor_task
        if processor_task is None:
            await self._flush_all_active(reason="cancelled")
            await self._flush_pending_insert_rows()
            await self._wait_for_optimize_task()
            return

        if processor_task.done():
            await processor_task
            await self._wait_for_optimize_task()
            return

        stop_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_StopQueue(future=stop_future))
        await stop_future
        await processor_task
        await self._wait_for_optimize_task()
        self._processor_task = None

    async def __aenter__(self) -> "DatasetThreadStorage":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        await self.stop()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        self._schedule_ready()
        ready_task = self._ready_task
        if ready_task is None:
            raise RuntimeError("dataset thread storage failed to schedule readiness")
        try:
            await ready_task
        except Exception:
            if self._ready_task is ready_task:
                self._ready_task = None
            raise

    def _schedule_ready(self) -> None:
        if self._ready:
            return
        ready_task = self._ready_task
        if ready_task is not None and not ready_task.done():
            return
        self._ready_task = asyncio.create_task(self._load_ready())

    async def _load_ready(self) -> None:
        if self._ready:
            return

        schema = self._schema()
        existing_schema: pa.Schema | None = None
        try:
            existing_schema = await self._client.inspect(
                table=self._table_name,
                namespace=self._namespace,
            )
        except Exception:
            await self._client.create_table_with_schema(
                name=self._table_name,
                schema=schema,
                mode="create_if_not_exists",
                namespace=self._namespace,
            )
            with contextlib.suppress(Exception):
                existing_schema = await self._client.inspect(
                    table=self._table_name,
                    namespace=self._namespace,
                )

        if existing_schema is not None:
            existing_names = set(existing_schema.names)
            missing_columns = {
                field.name: field
                for field in schema
                if field.name not in existing_names
            }
            if len(missing_columns) > 0:
                with contextlib.suppress(Exception):
                    await self._client.add_columns(
                        table=self._table_name,
                        new_columns=missing_columns,
                        namespace=self._namespace,
                    )

        rows = await self._search_ready_rows(schema=schema)
        self._rows = sorted(
            [
                row
                for row in (
                    self._stored_row_from_record(record=record)
                    for record in rows.to_pylist()
                )
                if row is not None
            ],
            key=lambda row: row.sequence,
        )
        self._next_sequence = max((row.sequence for row in self._rows), default=-1) + 1
        self._ready = True

    async def _search_ready_rows(self, *, schema: pa.Schema) -> pa.Table:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15.0
        while True:
            try:
                return await self._client.search(
                    table=self._table_name,
                    namespace=self._namespace,
                )
            except RoomException as ex:
                if "does not exist" not in str(ex) or loop.time() >= deadline:
                    raise
                with contextlib.suppress(Exception):
                    await self._client.create_table_with_schema(
                        name=self._table_name,
                        schema=schema,
                        mode="create_if_not_exists",
                        namespace=self._namespace,
                    )
                await asyncio.sleep(0.05)

    @staticmethod
    def _stored_row_from_record(
        *,
        record: dict[str, Any],
    ) -> _StoredThreadRow | None:
        item_id = record.get("item_id")
        sequence = record.get("sequence")
        raw_data = record.get("data")
        if not isinstance(item_id, str) or not isinstance(sequence, int):
            return None
        if isinstance(raw_data, DatasetJson):
            data = raw_data.to_json()
        elif isinstance(raw_data, dict):
            data = raw_data
        elif isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                return None
        else:
            return None
        if not isinstance(data, dict):
            return None

        raw_turn_id = record.get("turn_id")
        turn_id = raw_turn_id if isinstance(raw_turn_id, str) else None
        raw_type = record.get("type")
        raw_data_type = data.get("type")
        message_type = (
            raw_type
            if isinstance(raw_type, str)
            else raw_data_type
            if isinstance(raw_data_type, str)
            else None
        )
        raw_timestamp = record.get("timestamp")
        if isinstance(raw_timestamp, datetime):
            timestamp = (
                raw_timestamp.astimezone(timezone.utc)
                .isoformat()
                .replace(
                    "+00:00",
                    "Z",
                )
            )
        else:
            timestamp = str(raw_timestamp) if raw_timestamp is not None else ""
        attachment = record.get("attachment")
        if isinstance(attachment, bytearray):
            attachment = bytes(attachment)
        elif not isinstance(attachment, bytes):
            attachment = None

        return _StoredThreadRow(
            turn_id=turn_id,
            item_id=item_id,
            message_type=message_type,
            sequence=sequence,
            timestamp=timestamp,
            data=data,
            attachment=attachment,
        )

    def push_message(
        self,
        *,
        message: AgentThreadMessage,
        sender: Participant | None = None,
    ) -> None:
        try:
            self._queue.put_nowait(_QueuedThreadMessage(message=message, sender=sender))
        except asyncio.QueueShutDown:
            logger.debug("dropping dataset thread message after queue shutdown")

    async def _process_queue(self) -> None:
        await self._ensure_ready()
        while True:
            queued = await self._queue.get()
            should_stop = await self._process_queued_item(queued=queued)
            if should_stop:
                return

            while True:
                try:
                    queued = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                should_stop = await self._process_queued_item(queued=queued)
                if should_stop:
                    return

            await self._flush_pending_insert_rows()

    async def _process_queued_item(
        self, *, queued: _QueuedThreadMessage | _StopQueue | _FlushQueue
    ) -> bool:
        if isinstance(queued, _FlushQueue):
            try:
                await self._flush_pending_insert_rows()
            except Exception as exc:
                if not queued.future.done():
                    queued.future.set_exception(exc)
            else:
                if not queued.future.done():
                    queued.future.set_result(None)
            return False

        if isinstance(queued, _StopQueue):
            try:
                await self._flush_all_active(reason="cancelled")
                await self._flush_pending_insert_rows()
            except Exception as exc:
                if not queued.future.done():
                    queued.future.set_exception(exc)
            else:
                if not queued.future.done():
                    queued.future.set_result(None)
            return True

        await self._handle_message(message=queued.message, sender=queued.sender)
        return False

    async def _handle_message(
        self,
        *,
        message: AgentThreadMessage,
        sender: Participant | None,
    ) -> None:
        if isinstance(message, ThreadCleared):
            self._pending_user_turns.clear()
            self._pending_user_turn_rows.clear()
            self._pending_audio_commit_rows_by_turn_id.clear()
            self._active_content_by_item_id.clear()
            self._active_audio_transcriptions_by_item_id.clear()
            self._active_tool_calls_by_item_id.clear()
            await self._append_message_row(message=message)
            return

        if isinstance(message, TurnStart):
            await self._flush_all_active(reason="completed")
            queued = _QueuedThreadMessage(
                message=message,
                sender=sender,
            )
            self._pending_user_turns[message.message_id] = queued
            row = await self._append_message_row(
                message=self._turn_input_with_sender_name(queued=queued)
            )
            self._pending_user_turn_rows[message.message_id] = row
            return

        if isinstance(message, TurnSteer):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="completed",
            )
            queued = _QueuedThreadMessage(
                message=message,
                sender=sender,
            )
            self._pending_user_turns[message.message_id] = queued
            row = await self._append_message_row(
                message=self._turn_input_with_sender_name(queued=queued)
            )
            self._pending_user_turn_rows[message.message_id] = row
            return

        if isinstance(message, TurnStartAccepted):
            await self._flush_all_active(
                reason="completed",
                flush_pending_audio_commits=False,
            )
            await self._commit_pending_user_turn(
                source_message_id=message.source_message_id,
                turn_id=message.turn_id,
                accepted_message=message,
            )
            return

        if isinstance(message, TurnSteerAccepted):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="completed",
            )
            await self._commit_pending_user_turn(
                source_message_id=message.source_message_id,
                turn_id=message.turn_id,
                accepted_message=message,
            )
            return

        if isinstance(message, TurnSteerRejected):
            self._pending_user_turns.pop(message.source_message_id, None)
            self._pending_user_turn_rows.pop(message.source_message_id, None)
            await self._append_message_row(message=message)
            return

        if isinstance(message, TurnStartRejected):
            self._pending_user_turns.pop(message.source_message_id, None)
            self._pending_user_turn_rows.pop(message.source_message_id, None)
            await self._append_message_row(message=message)
            return

        if isinstance(message, AgentRealtimeAudioChunk):
            if self._persist_audio_input:
                await self._append_message_row(message=message)
            return

        if isinstance(message, AgentRealtimeAudioCommit):
            queued = _QueuedThreadMessage(message=message, sender=sender)
            self._pending_user_turns[message.message_id] = queued
            row = await self._reserve_message_row(message=message)
            self._pending_user_turn_rows[message.message_id] = row
            return

        if isinstance(
            message,
            (
                AgentAudioTranscriptionStarted,
                AgentAudioTranscriptionDelta,
                AgentAudioTranscriptionCompleted,
                AgentAudioTranscriptionFailed,
            ),
        ):
            self._record_audio_transcription(message=message)
            return

        if isinstance(message, AgentTextContentStarted):
            self._active_content_by_item_id[message.item_id] = _ActiveContent(
                kind="text",
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                provider=message.provider,
                model=message.model,
            )
            return

        if isinstance(message, AgentTextContentDelta):
            active = self._ensure_active_content(message=message, kind="text")
            active.parts = [
                accumulate_text_delta(
                    current="".join(active.parts),
                    delta=message.text,
                )
            ]
            await self._append_verbose_delta_row(message=message)
            return

        if isinstance(message, AgentTextContentEnded):
            await self._flush_content_item(
                item_id=message.item_id,
                reason="completed",
                ended_message=message,
            )
            return

        if isinstance(message, AgentReasoningContentStarted):
            self._active_content_by_item_id[message.item_id] = _ActiveContent(
                kind="reasoning",
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                provider=message.provider,
                model=message.model,
            )
            return

        if isinstance(message, AgentReasoningContentDelta):
            active = self._ensure_active_content(message=message, kind="reasoning")
            active.parts = [
                accumulate_text_delta(
                    current="".join(active.parts),
                    delta=message.text,
                )
            ]
            await self._append_verbose_delta_row(message=message)
            return

        if isinstance(message, AgentReasoningContentEnded):
            await self._flush_content_item(
                item_id=message.item_id,
                reason="completed",
                ended_message=message,
            )
            return

        if isinstance(message, AgentFileContentStarted):
            self._active_content_by_item_id[message.item_id] = _ActiveContent(
                kind="file",
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                provider=message.provider,
                model=message.model,
            )
            return

        if isinstance(message, AgentFileContentDelta):
            active = self._ensure_active_content(message=message, kind="file")
            active.parts.append(message.url)
            await self._append_verbose_delta_row(message=message)
            return

        if isinstance(message, AgentFileContentEnded):
            await self._flush_content_item(
                item_id=message.item_id,
                reason="completed",
                ended_message=message,
            )
            return

        if isinstance(
            message,
            (AgentToolCallPending, AgentToolCallInProgress, AgentToolCallStarted),
        ):
            self._record_tool_call_state(message=message)
            return

        if isinstance(message, AgentToolCallLogDelta):
            active = self._active_tool_calls_by_item_id.get(message.item_id)
            if active is None:
                active = _ActiveToolCall(
                    turn_id=message.turn_id,
                    item_id=message.item_id,
                    message_id=message.message_id,
                    provider=message.provider,
                    model=message.model,
                    namespace=message.namespace,
                    call_id=message.call_id,
                    stage="in_progress",
                )
                self._active_tool_calls_by_item_id[message.item_id] = active
            active.logs.extend([line.model_dump(mode="json") for line in message.lines])
            await self._append_verbose_delta_row(message=message)
            return

        if isinstance(message, AgentToolCallArgumentsDelta):
            active = self._active_tool_calls_by_item_id.get(message.item_id)
            if active is not None and message.delta != "":
                active.argument_delta_text += message.delta
                updated_arguments = _tool_arguments_from_delta_text(
                    tool=active.tool,
                    current=active.arguments,
                    text=active.argument_delta_text,
                )
                if updated_arguments is not None:
                    active.arguments = updated_arguments
            await self._append_verbose_delta_row(message=message)
            return

        if isinstance(message, AgentToolCallEnded):
            await self._flush_tool_call(
                item_id=message.item_id,
                reason="completed" if message.error is None else "failed",
                ended_message=message,
            )
            return

        if isinstance(message, AgentContextCompacted):
            await self._append_message_row(message=message)
            return

        if isinstance(message, AgentUsageUpdated):
            await self._append_message_row(message=message)
            return

        if isinstance(
            message,
            (
                AgentImageGenerationStarted,
                AgentImageGenerationPartial,
                AgentImageGenerationCompleted,
                AgentImageGenerationFailed,
            ),
        ):
            if isinstance(
                message, (AgentImageGenerationStarted, AgentImageGenerationPartial)
            ):
                self._record_image_generation_state(message=message)
                await self._append_verbose_delta_row(message=message)
                return
            if isinstance(
                message,
                (AgentImageGenerationCompleted, AgentImageGenerationFailed),
            ):
                self._active_tool_calls_by_item_id.pop(message.item_id, None)
                active = self._active_image_generations_by_item_id.pop(
                    message.item_id, None
                )
                if (
                    isinstance(message, AgentImageGenerationCompleted)
                    and len(message.images) == 0
                    and active is not None
                    and active.partial is not None
                    and active.partial.image is not None
                ):
                    message = self._completed_image_generation_from_active(
                        active=active,
                        message_id=message.message_id,
                    )
            await self._append_message_row(message=message)
            return

        if isinstance(message, AgentThreadEvent):
            await self._append_message_row(message=message)
            return

        if isinstance(message, TurnInterrupted):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="cancelled",
            )
            await self._append_message_row(message=message)
            return

        if isinstance(message, TurnEnded):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="failed" if message.error is not None else "completed",
            )
            await self._append_message_row(message=message)
            return

        await self._append_message_row(message=message)

    async def _append_verbose_delta_row(self, *, message: AgentThreadMessage) -> None:
        if self._persist_deltas:
            await self._append_message_row(message=message)

    def _ensure_active_content(
        self,
        *,
        message: (
            AgentTextContentDelta | AgentReasoningContentDelta | AgentFileContentDelta
        ),
        kind: Literal["text", "reasoning", "file"],
    ) -> _ActiveContent:
        active = self._active_content_by_item_id.get(message.item_id)
        if active is not None:
            if active.provider is None:
                active.provider = message.provider
            if active.model is None:
                active.model = message.model
            if active.sender_name is None:
                active.sender_name = message.sender_name
            if (
                active.phase is None
                and isinstance(message, AgentTextContentDelta)
                and message.phase is not None
            ):
                active.phase = message.phase
            return active

        active = _ActiveContent(
            kind=kind,
            turn_id=message.turn_id,
            item_id=message.item_id,
            message_id=message.message_id,
            provider=message.provider,
            model=message.model,
            sender_name=message.sender_name,
            phase=message.phase if isinstance(message, AgentTextContentDelta) else None,
        )
        self._active_content_by_item_id[message.item_id] = active
        return active

    def _record_tool_call_state(
        self,
        *,
        message: AgentToolCallPending | AgentToolCallInProgress | AgentToolCallStarted,
    ) -> None:
        active = self._active_tool_calls_by_item_id.get(message.item_id)
        if active is None:
            active = _ActiveToolCall(
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                provider=message.provider,
                model=message.model,
                namespace=message.namespace,
                call_id=message.call_id,
            )
            self._active_tool_calls_by_item_id[message.item_id] = active

        active.namespace = message.namespace
        active.call_id = message.call_id
        if active.provider is None:
            active.provider = message.provider
        if active.model is None:
            active.model = message.model
        active.toolkit = message.toolkit
        active.tool = message.tool
        active.arguments = message.arguments
        if isinstance(message, AgentToolCallStarted):
            active.stage = "started"
        elif isinstance(message, AgentToolCallInProgress):
            active.stage = "in_progress"
        else:
            active.stage = "pending"

    def _record_image_generation_state(
        self,
        *,
        message: AgentImageGenerationStarted | AgentImageGenerationPartial,
    ) -> None:
        active = self._active_image_generations_by_item_id.get(message.item_id)
        if active is None:
            active = _ActiveImageGeneration(
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                provider=message.provider,
                model=message.model,
            )
            self._active_image_generations_by_item_id[message.item_id] = active

        active.turn_id = message.turn_id
        if active.provider is None:
            active.provider = message.provider
        if active.model is None:
            active.model = message.model
        if isinstance(message, AgentImageGenerationPartial):
            active.partial = message
        else:
            active.started = message

    async def _commit_pending_user_turn(
        self,
        *,
        source_message_id: str,
        turn_id: str | None,
        accepted_message: TurnStartAccepted | TurnSteerAccepted,
    ) -> None:
        queued = self._pending_user_turns.pop(source_message_id, None)
        pending_row = self._pending_user_turn_rows.pop(source_message_id, None)
        if (
            queued is not None
            and isinstance(queued.message, AgentRealtimeAudioCommit)
            and pending_row is not None
        ):
            if turn_id is not None:
                self._set_reserved_row_turn_id(row=pending_row, turn_id=turn_id)
                self._pending_audio_commit_rows_by_turn_id[turn_id] = pending_row
            await self._append_message_row(message=accepted_message, turn_id=turn_id)
            return
        del queued
        if turn_id is not None and pending_row is not None:
            await self._set_row_turn_id(row=pending_row, turn_id=turn_id)
        await self._append_message_row(message=accepted_message, turn_id=turn_id)

    def _turn_input_with_sender_name(
        self,
        *,
        queued: _QueuedThreadMessage,
    ) -> TurnStart | TurnSteer:
        message = queued.message
        if not isinstance(message, (TurnStart, TurnSteer)):
            raise TypeError("queued message must be a turn input")
        sender_name = message.sender_name or self._sender_name(sender=queued.sender)
        if sender_name is None:
            return message
        return message.model_copy(update={"sender_name": sender_name})

    async def _flush_content_item(
        self,
        *,
        item_id: str,
        reason: Literal["completed", "cancelled", "failed"],
        ended_message: AgentThreadMessage | None = None,
    ) -> None:
        active = self._active_content_by_item_id.pop(item_id, None)
        if active is None:
            return

        del reason
        if active.kind in {"text", "reasoning"}:
            text = "".join(active.parts)
            if text.strip() == "":
                return
            message_cls = (
                AgentTextContentDelta
                if active.kind == "text"
                else AgentReasoningContentDelta
            )
            message_type = (
                AGENT_EVENT_TEXT_CONTENT_DELTA
                if active.kind == "text"
                else AGENT_EVENT_REASONING_CONTENT_DELTA
            )
            await self._append_message_row(
                message=message_cls(
                    type=message_type,
                    thread_id=self.path,
                    turn_id=active.turn_id,
                    item_id=active.item_id,
                    text=text,
                    provider=active.provider,
                    model=active.model,
                    sender_name=active.sender_name,
                    phase=active.phase if active.kind == "text" else None,
                )
            )
            if ended_message is not None:
                await self._append_message_row(message=ended_message)
            return

        urls = [part for part in active.parts if part.strip() != ""]
        if len(urls) == 0:
            return
        for url in urls:
            await self._append_message_row(
                message=AgentFileContentDelta(
                    type=AGENT_EVENT_FILE_CONTENT_DELTA,
                    thread_id=self.path,
                    turn_id=active.turn_id,
                    item_id=active.item_id,
                    url=url,
                    provider=active.provider,
                    model=active.model,
                    sender_name=active.sender_name,
                )
            )
        if ended_message is not None:
            await self._append_message_row(message=ended_message)

    async def _flush_tool_call(
        self,
        *,
        item_id: str,
        reason: Literal["completed", "cancelled", "failed"],
        ended_message: AgentToolCallEnded | None = None,
    ) -> None:
        active = self._active_tool_calls_by_item_id.pop(item_id, None)
        if active is None and ended_message is None:
            return

        if active is None:
            active = _ActiveToolCall(
                turn_id=ended_message.turn_id,
                item_id=ended_message.item_id,
                message_id=ended_message.message_id,
                provider=ended_message.provider,
                model=ended_message.model,
                namespace=ended_message.namespace,
                call_id=ended_message.call_id,
            )

        should_persist = (
            reason == "completed"
            or ended_message is not None
            or active.stage in {"in_progress", "started"}
            or len(active.logs) > 0
        )
        if not should_persist:
            return

        if await self._persist_generated_image_result(
            active_tool_call=active,
            message=ended_message,
        ):
            return

        del reason
        if active.toolkit is not None and active.tool is not None:
            await self._append_message_row(
                message=AgentToolCallStarted(
                    type=AGENT_EVENT_TOOL_CALL_STARTED,
                    thread_id=self.path,
                    message_id=active.message_id,
                    turn_id=active.turn_id,
                    item_id=active.item_id,
                    namespace=active.namespace,
                    call_id=active.call_id,
                    toolkit=active.toolkit,
                    tool=active.tool,
                    arguments=active.arguments,
                    provider=active.provider,
                    model=active.model,
                )
            )

        log_message = self._tool_log_delta_from_stored_lines(
            turn_id=active.turn_id,
            item_id=active.item_id,
            namespace=active.namespace,
            call_id=active.call_id,
            provider=active.provider,
            model=active.model,
            logs=active.logs,
        )
        if log_message is not None:
            await self._append_message_row(message=log_message)

        if ended_message is not None:
            await self._append_message_row(message=ended_message)

    async def _persist_generated_image_result(
        self,
        *,
        active_tool_call: _ActiveToolCall,
        message: AgentToolCallEnded | None,
    ) -> bool:
        if message is None:
            return False
        if active_tool_call.tool is None:
            return False
        if active_tool_call.tool.strip().lower() != "image_generation":
            return False
        if message.error is not None:
            self._active_image_generations_by_item_id.pop(
                active_tool_call.item_id, None
            )
            failed_message = AgentImageGenerationFailed(
                type="meshagent.agent.image_generation.failed",
                thread_id=message.thread_id,
                message_id=message.message_id,
                turn_id=active_tool_call.turn_id,
                item_id=active_tool_call.item_id,
                call_id=active_tool_call.call_id,
                toolkit=active_tool_call.toolkit or "image_generation",
                tool=active_tool_call.tool or "image_generation",
                arguments=active_tool_call.arguments,
                provider=active_tool_call.provider,
                model=active_tool_call.model,
                error=message.error,
            )
            await self._append_message_row(message=failed_message)
            return True

        image_uri: str | None = None
        mime_type: str | None = None

        if isinstance(message.result, TextContent):
            encoded_image = message.result.text.strip()
            if encoded_image != "":
                image_uri = encoded_image

        arguments = active_tool_call.arguments or {}
        if mime_type is None:
            mime_type = _mime_type_from_output_format(arguments.get("output_format"))

        width = _normalize_positive_dimension(arguments.get("width"))
        height = _normalize_positive_dimension(arguments.get("height"))
        if width is None or height is None:
            parsed_width, parsed_height = _parse_image_dimensions_from_size(
                arguments.get("size")
            )
            if width is None:
                width = parsed_width
            if height is None:
                height = parsed_height

        if image_uri is None:
            return False
        self._active_image_generations_by_item_id.pop(active_tool_call.item_id, None)
        completed_message = AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id=message.thread_id,
            message_id=message.message_id,
            turn_id=active_tool_call.turn_id,
            item_id=active_tool_call.item_id,
            call_id=active_tool_call.call_id,
            toolkit=active_tool_call.toolkit or "image_generation",
            tool=active_tool_call.tool or "image_generation",
            arguments=active_tool_call.arguments,
            provider=active_tool_call.provider,
            model=active_tool_call.model,
            images=[
                AgentGeneratedImage(
                    uri=image_uri,
                    mime_type=mime_type,
                    width=width,
                    height=height,
                    status="completed",
                )
            ],
        )
        await self._append_message_row(message=completed_message)
        return True

    async def _flush_turn_active_items(
        self,
        *,
        turn_id: str,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        await self._flush_turn_audio_transcriptions(turn_id=turn_id, reason=reason)

        content_item_ids = [
            item_id
            for item_id, active in self._active_content_by_item_id.items()
            if active.turn_id == turn_id
        ]
        for item_id in content_item_ids:
            await self._flush_content_item(item_id=item_id, reason=reason)

        tool_item_ids = [
            item_id
            for item_id, active in self._active_tool_calls_by_item_id.items()
            if active.turn_id == turn_id
        ]
        for item_id in tool_item_ids:
            await self._flush_tool_call(item_id=item_id, reason=reason)

        image_item_ids = [
            item_id
            for item_id, active in self._active_image_generations_by_item_id.items()
            if active.turn_id == turn_id
        ]
        for item_id in image_item_ids:
            await self._flush_image_generation(item_id=item_id, reason=reason)

    async def _flush_all_active(
        self,
        *,
        reason: Literal["completed", "cancelled", "failed"],
        flush_pending_audio_commits: bool = True,
    ) -> None:
        for item_id in list(self._active_content_by_item_id):
            await self._flush_content_item(item_id=item_id, reason=reason)
        for item_id in list(self._active_audio_transcriptions_by_item_id):
            active = self._active_audio_transcriptions_by_item_id.get(item_id)
            if active is not None:
                await self._flush_turn_audio_transcriptions(
                    turn_id=active.turn_id,
                    reason=reason,
                )
        for turn_id in list(self._pending_audio_commit_rows_by_turn_id):
            await self._flush_turn_audio_transcriptions(
                turn_id=turn_id,
                reason=reason,
            )
        if flush_pending_audio_commits:
            self._flush_pending_audio_commit_rows(reason=reason)
        for item_id in list(self._active_tool_calls_by_item_id):
            await self._flush_tool_call(item_id=item_id, reason=reason)
        for item_id in list(self._active_image_generations_by_item_id):
            await self._flush_image_generation(item_id=item_id, reason=reason)

    def _record_audio_transcription(
        self,
        *,
        message: AgentAudioTranscriptionStarted
        | AgentAudioTranscriptionDelta
        | AgentAudioTranscriptionCompleted
        | AgentAudioTranscriptionFailed,
    ) -> None:
        active = self._active_audio_transcriptions_by_item_id.get(message.item_id)
        if active is None:
            active = _ActiveAudioTranscription(
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
                role=message.role,
                provider=message.provider,
                model=message.model,
                response_id=message.response_id,
                content_index=message.content_index,
                sender_name=message.sender_name,
            )
            self._active_audio_transcriptions_by_item_id[message.item_id] = active

        active.turn_id = message.turn_id
        if active.role is None:
            active.role = message.role
        if active.provider is None:
            active.provider = message.provider
        if active.model is None:
            active.model = message.model
        if active.response_id is None:
            active.response_id = message.response_id
        if active.content_index is None:
            active.content_index = message.content_index
        if active.sender_name is None:
            active.sender_name = message.sender_name

        if isinstance(message, AgentAudioTranscriptionDelta):
            active.parts = [
                accumulate_text_delta(current="".join(active.parts), delta=message.text)
            ]
            active.status = "in_progress"
        elif isinstance(message, AgentAudioTranscriptionCompleted):
            if message.text is not None:
                active.parts = [message.text]
            active.status = "completed"
        elif isinstance(message, AgentAudioTranscriptionFailed):
            active.status = "failed"
            active.error = message.error

    async def _flush_turn_audio_transcriptions(
        self,
        *,
        turn_id: str,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        item_ids = [
            item_id
            for item_id, active in self._active_audio_transcriptions_by_item_id.items()
            if active.turn_id == turn_id
        ]
        if (
            len(item_ids) == 0
            and turn_id not in self._pending_audio_commit_rows_by_turn_id
        ):
            return

        commit_row = self._pending_audio_commit_rows_by_turn_id.pop(turn_id, None)
        user_transcriptions: list[_ActiveAudioTranscription] = []
        for item_id in item_ids:
            active = self._active_audio_transcriptions_by_item_id.pop(item_id)
            if active.role == "user":
                user_transcriptions.append(active)
            else:
                await self._append_audio_transcription_row(
                    active=active,
                    reason=reason,
                )

        if commit_row is None:
            return

        text = " ".join(
            part
            for part in (
                "".join(active.parts).strip() for active in user_transcriptions
            )
            if part != ""
        )
        status: Literal["completed", "cancelled", "failed"]
        transcription_item_id: str | None = None
        if any(active.status == "failed" for active in user_transcriptions):
            status = "failed"
            failed = next(
                active for active in user_transcriptions if active.status == "failed"
            )
            transcription_item_id = failed.item_id
        elif reason == "cancelled":
            status = "cancelled"
        else:
            status = "completed"
        if transcription_item_id is None and len(user_transcriptions) > 0:
            transcription_item_id = user_transcriptions[0].item_id
        self._finalize_audio_commit_row(
            row=commit_row,
            text=text,
            status=status,
            transcription_item_id=transcription_item_id,
        )

    def _flush_pending_audio_commit_rows(
        self,
        *,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        for message_id, queued in list(self._pending_user_turns.items()):
            if not isinstance(queued.message, AgentRealtimeAudioCommit):
                continue
            row = self._pending_user_turn_rows.pop(message_id, None)
            self._pending_user_turns.pop(message_id, None)
            if row is None:
                continue
            status: Literal["completed", "cancelled", "failed"] = (
                "failed" if reason == "failed" else "cancelled"
            )
            self._finalize_audio_commit_row(
                row=row,
                text="",
                status=status,
                transcription_item_id=None,
            )

    async def _append_audio_transcription_row(
        self,
        *,
        active: _ActiveAudioTranscription,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        text = "".join(active.parts)
        if text.strip() == "" and active.status != "failed":
            return
        if active.status == "failed":
            await self._append_message_row(
                message=AgentAudioTranscriptionFailed(
                    type=AGENT_EVENT_AUDIO_TRANSCRIPTION_FAILED,
                    thread_id=self.path,
                    turn_id=active.turn_id,
                    item_id=active.item_id,
                    message_id=active.message_id,
                    response_id=active.response_id,
                    content_index=active.content_index,
                    role=active.role,
                    provider=active.provider,
                    model=active.model,
                    sender_name=active.sender_name,
                    error=active.error,
                )
            )
            return
        await self._append_message_row(
            message=AgentAudioTranscriptionCompleted(
                type=AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
                thread_id=self.path,
                turn_id=active.turn_id,
                item_id=active.item_id,
                message_id=active.message_id,
                response_id=active.response_id,
                content_index=active.content_index,
                role=active.role,
                provider=active.provider,
                model=active.model,
                sender_name=active.sender_name,
                text=text,
            )
        )

    async def _flush_image_generation(
        self,
        *,
        item_id: str,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        active = self._active_image_generations_by_item_id.pop(item_id, None)
        if active is None:
            return
        if reason == "completed":
            await self._append_message_row(
                message=self._completed_image_generation_from_active(active=active)
            )
            return
        await self._append_message_row(
            message=self._failed_image_generation_from_active(
                active=active,
                reason=reason,
            )
        )

    def _completed_image_generation_from_active(
        self,
        *,
        active: _ActiveImageGeneration,
        message_id: str | None = None,
    ) -> AgentImageGenerationCompleted:
        source = active.partial or active.started
        image = active.partial.image if active.partial is not None else None
        images = (
            [image.model_copy(update={"status": "completed"})]
            if image is not None
            else []
        )
        return AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id=self.path,
            turn_id=active.turn_id,
            item_id=active.item_id,
            message_id=message_id or active.message_id,
            call_id=None if source is None else source.call_id,
            toolkit="image_generation" if source is None else source.toolkit,
            tool="image_generation" if source is None else source.tool,
            arguments=None if source is None else source.arguments,
            provider=active.provider,
            model=active.model,
            images=images,
        )

    def _failed_image_generation_from_active(
        self,
        *,
        active: _ActiveImageGeneration,
        reason: Literal["cancelled", "failed"],
    ) -> AgentImageGenerationFailed:
        source = active.partial or active.started
        return AgentImageGenerationFailed(
            type="meshagent.agent.image_generation.failed",
            thread_id=self.path,
            turn_id=active.turn_id,
            item_id=active.item_id,
            message_id=active.message_id,
            call_id=None if source is None else source.call_id,
            toolkit="image_generation" if source is None else source.toolkit,
            tool="image_generation" if source is None else source.tool,
            arguments=None if source is None else source.arguments,
            provider=active.provider,
            model=active.model,
            error=AgentError(
                message="Image generation cancelled"
                if reason == "cancelled"
                else "Image generation failed",
                code=reason,
            ),
        )

    async def _append_message_row(
        self,
        *,
        message: AgentThreadMessage,
        turn_id: str | None = None,
        item_id: str | None = None,
    ) -> _StoredThreadRow:
        data, attachment = self._message_row_data_and_attachment(message=message)
        return await self._append_row(
            turn_id=turn_id if turn_id is not None else self._message_turn_id(message),
            item_id=item_id if item_id is not None else self._message_item_id(message),
            data=data,
            attachment=attachment,
        )

    async def _reserve_message_row(
        self,
        *,
        message: AgentThreadMessage,
        turn_id: str | None = None,
        item_id: str | None = None,
    ) -> _StoredThreadRow:
        data, attachment = self._message_row_data_and_attachment(message=message)
        return await self._append_row(
            turn_id=turn_id if turn_id is not None else self._message_turn_id(message),
            item_id=item_id if item_id is not None else self._message_item_id(message),
            data=data,
            attachment=attachment,
            queue_for_insert=False,
        )

    @staticmethod
    def _message_row_data_and_attachment(
        *,
        message: AgentThreadMessage,
    ) -> tuple[dict[str, Any], bytes | None]:
        message = cast(AgentThreadMessage, scrub_agent_message_for_storage(message))
        if isinstance(message, (AgentAudioGenerationDelta, AgentRealtimeAudioChunk)):
            stored_message = message.model_copy(update={"data": b""})
            return stored_message.model_dump(mode="json"), message.data
        return message.model_dump(mode="json"), None

    @staticmethod
    def _message_turn_id(message: AgentThreadMessage) -> str | None:
        if isinstance(message, TurnStart):
            return message.turn_id
        if isinstance(
            message,
            (
                TurnSteer,
                TurnStartAccepted,
                TurnInterrupt,
                TurnInterruptAccepted,
                TurnInterrupted,
                AgentRealtimeAudioCommit,
                TurnSteerAccepted,
                TurnSteered,
                TurnSteerRejected,
                TurnStarted,
                TurnEnded,
                AgentReasoningContentStarted,
                AgentReasoningContentDelta,
                AgentReasoningContentEnded,
                AgentTextContentStarted,
                AgentTextContentDelta,
                AgentTextContentEnded,
                AgentAudioTranscriptionStarted,
                AgentAudioTranscriptionDelta,
                AgentAudioTranscriptionCompleted,
                AgentAudioTranscriptionFailed,
                AgentFileContentStarted,
                AgentFileContentDelta,
                AgentFileContentEnded,
                AgentToolCallPending,
                AgentToolCallInProgress,
                AgentToolCallStarted,
                AgentToolCallArgumentsDelta,
                AgentToolCallLogDelta,
                AgentToolCallEnded,
                AgentToolCallApprovalRequested,
                AgentImageGenerationStarted,
                AgentImageGenerationPartial,
                AgentImageGenerationCompleted,
                AgentImageGenerationFailed,
                AgentUsageUpdated,
            ),
        ):
            return message.turn_id
        if isinstance(message, AgentThreadStatus):
            return message.turn_id
        return None

    @staticmethod
    def _message_item_id(message: AgentThreadMessage) -> str:
        if isinstance(
            message,
            (
                AgentReasoningContentStarted,
                AgentReasoningContentDelta,
                AgentReasoningContentEnded,
                AgentTextContentStarted,
                AgentTextContentDelta,
                AgentTextContentEnded,
                AgentAudioTranscriptionStarted,
                AgentAudioTranscriptionDelta,
                AgentAudioTranscriptionCompleted,
                AgentAudioTranscriptionFailed,
                AgentFileContentStarted,
                AgentFileContentDelta,
                AgentFileContentEnded,
                AgentToolCallPending,
                AgentToolCallInProgress,
                AgentToolCallStarted,
                AgentToolCallLogDelta,
                AgentToolCallEnded,
                AgentToolCallApprovalRequested,
                AgentImageGenerationStarted,
                AgentImageGenerationPartial,
                AgentImageGenerationCompleted,
                AgentImageGenerationFailed,
            ),
        ):
            return message.item_id
        return message.message_id

    async def _append_row(
        self,
        *,
        turn_id: str | None,
        item_id: str,
        data: dict[str, Any],
        attachment: bytes | None = None,
        queue_for_insert: bool = True,
    ) -> _StoredThreadRow:
        timestamp = _now_iso()
        if (
            not isinstance(data.get("created_at"), str)
            or data["created_at"].strip() == ""
        ):
            data = {**data, "created_at": timestamp}
        sequence = self._next_sequence
        self._next_sequence += 1
        normalized_item_id = item_id if item_id.strip() != "" else str(uuid.uuid4())
        row = _StoredThreadRow(
            turn_id=turn_id,
            item_id=normalized_item_id,
            message_type=data.get("type")
            if isinstance(data.get("type"), str)
            else None,
            sequence=sequence,
            timestamp=timestamp,
            data=data,
            attachment=attachment,
        )
        if queue_for_insert:
            self._pending_insert_rows.append(row)
        self._rows.append(row)
        self._rows.sort(key=lambda stored: stored.sequence)
        return row

    async def _flush_pending_insert_rows(self) -> None:
        await self._ensure_ready()
        rows = self._pending_insert_rows
        if len(rows) == 0:
            return
        rows = sorted(rows, key=lambda stored: stored.sequence)
        await self._client.insert(
            table=self._table_name,
            namespace=self._namespace,
            records=[
                {
                    "turn_id": row.turn_id,
                    "item_id": row.item_id,
                    "type": row.message_type,
                    "sequence": row.sequence,
                    "timestamp": row.timestamp,
                    "data": DatasetJson(row.data),
                    "attachment": row.attachment,
                }
                for row in rows
            ],
        )
        del self._pending_insert_rows[: len(rows)]
        for _ in rows:
            self._note_appended_row()

    async def _set_row_turn_id(self, *, row: _StoredThreadRow, turn_id: str) -> None:
        if row.turn_id == turn_id and row.data.get("turn_id") == turn_id:
            return
        await self._flush_pending_insert_rows()
        updated_data = {**row.data, "turn_id": turn_id}
        await self._client.merge(
            table=self._table_name,
            namespace=self._namespace,
            on="sequence",
            records=[
                {
                    "turn_id": turn_id,
                    "item_id": row.item_id,
                    "type": row.message_type,
                    "sequence": row.sequence,
                    "timestamp": row.timestamp,
                    "data": DatasetJson(updated_data),
                    "attachment": row.attachment,
                }
            ],
        )
        row.turn_id = turn_id
        row.data = updated_data

    def _set_reserved_row_turn_id(self, *, row: _StoredThreadRow, turn_id: str) -> None:
        if row.turn_id == turn_id and row.data.get("turn_id") == turn_id:
            return
        row.turn_id = turn_id
        row.data = {**row.data, "turn_id": turn_id}

    def _finalize_audio_commit_row(
        self,
        *,
        row: _StoredThreadRow,
        text: str,
        status: Literal["completed", "cancelled", "failed"],
        transcription_item_id: str | None,
    ) -> None:
        row.data = {
            **row.data,
            "text": text,
            "status": status,
            "transcription_item_id": transcription_item_id,
        }
        if row not in self._pending_insert_rows:
            self._pending_insert_rows.append(row)

    def _note_appended_row(self) -> None:
        if self._optimize_after_append_count <= 0:
            return
        self._appends_since_optimize += 1
        if self._appends_since_optimize < self._optimize_after_append_count:
            return

        optimize_task = self._optimize_task
        if optimize_task is not None and not optimize_task.done():
            self._optimize_requested = True
            return

        self._appends_since_optimize = 0
        self._optimize_requested = False
        self._optimize_task = asyncio.create_task(self._optimize_loop())

    async def _optimize_loop(self) -> None:
        while True:
            try:
                await self._client.optimize(
                    table=self._table_name,
                    namespace=self._namespace,
                    config=DatasetOptimizeConfig(
                        compact_files=True,
                        optimize_indices=False,
                        cleanup_old_versions=False,
                    ),
                )
            except Exception:
                logger.exception(
                    "failed to optimize dataset thread table %s",
                    self._path,
                )

            if (
                not self._optimize_requested
                and self._appends_since_optimize < self._optimize_after_append_count
            ):
                return

            self._appends_since_optimize = 0
            self._optimize_requested = False

    async def _wait_for_optimize_task(self) -> None:
        optimize_task = self._optimize_task
        if optimize_task is None:
            return
        await optimize_task

    @staticmethod
    def _sender_name(*, sender: Participant | None) -> str | None:
        if sender is None:
            return None
        raw_name = sender.get_attribute("name")
        if not isinstance(raw_name, str):
            return None
        normalized = raw_name.strip()
        return normalized if normalized != "" else None

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None:
        if llm_adapter is not None:
            restored_messages: list[dict[str, Any]] = []
            reader = llm_adapter.make_agent_event_reader(
                emit_message=restored_messages.append,
                callbacks=self._agent_event_reader_callbacks(
                    context=context,
                    restored_messages=restored_messages,
                ),
            )
            for row in self._rows:
                for message in self._messages_from_row(row=row):
                    reader(message)
            reader.finalize()
            llm_adapter.restore_context_messages(
                context=context,
                messages=restored_messages,
            )
            return

        rows = self._rows
        if len(rows) > self._max_append_message_count:
            first_message = len(rows) - self._max_append_message_count
            rows = rows[first_message:]
            context.append_assistant_message(
                "there are more messages outside the current context window, "
                f"the index of the first message loaded is {first_message}"
            )

        for row in rows:
            self._restore_row(context=context, row=row)

    async def restore_session_context_async(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None:
        await self._ensure_ready()
        if llm_adapter is None:
            self.restore_session_context(context=context)
            return

        restored_messages: list[dict[str, Any]] = []
        reader = llm_adapter.make_agent_event_reader(
            emit_message=restored_messages.append,
            callbacks=self._agent_event_reader_callbacks(
                context=context,
                restored_messages=restored_messages,
            ),
        )
        images_dataset = ImagesDataset(self._client)
        for row in self._rows:
            for message in await self._messages_from_row_async(
                row=row,
                images_dataset=images_dataset,
            ):
                reader(message)
        reader.finalize()
        llm_adapter.restore_context_messages(
            context=context,
            messages=restored_messages,
        )

    @staticmethod
    def _agent_event_reader_callbacks(
        *,
        context: AgentSessionContext,
        restored_messages: list[dict[str, Any]],
    ) -> AgentEventReaderCallbacks:
        def update_usage(message: AgentUsageUpdated) -> None:
            context.last_usage = SessionUsage(
                model=_infer_usage_model(message.usage),
                usage=dict(message.usage),
                context_window_used=message.context_window.used_tokens,
                context_window_size=message.context_window.total_tokens,
            )

        def restore_compacted_context(message: AgentContextCompacted) -> None:
            context.metadata["last_compaction"] = {
                "checkpoint_id": message.checkpoint_id,
                "path": message.path,
                "through_sequence": message.through_sequence,
                "created_at": message.created_at,
            }
            if message.messages is None:
                return
            restored_messages.clear()
            context.messages.clear()
            context.previous_messages.clear()
            context.previous_response_id = None

        return AgentEventReaderCallbacks(
            record_event=lambda message: None,
            update_usage=update_usage,
            restore_compacted_context=restore_compacted_context,
        )

    def _restore_row(
        self,
        *,
        context: AgentSessionContext,
        row: _StoredThreadRow,
    ) -> None:
        data = row.data
        raw_message = self._stored_agent_message(
            value=data,
            attachment=row.attachment,
        )
        if (
            isinstance(raw_message, AgentContextCompacted)
            and raw_message.messages is not None
        ):
            context.messages.clear()
            context.messages.extend(deepcopy(raw_message.messages))
            context.previous_messages.clear()
            context.previous_response_id = None
            return

    def _messages_from_row(self, *, row: _StoredThreadRow) -> list[AgentThreadMessage]:
        data = row.data
        raw_message = self._stored_agent_message(
            value=data,
            attachment=row.attachment,
        )
        if raw_message is not None:
            return [raw_message]
        return []

    async def _messages_from_row_async(
        self,
        *,
        row: _StoredThreadRow,
        images_dataset: ImagesDataset,
    ) -> list[AgentThreadMessage]:
        messages = self._messages_from_row(row=row)
        return [
            await self._hydrate_persisted_image_generation_message(
                message=message,
                images_dataset=images_dataset,
            )
            for message in messages
        ]

    async def _hydrate_persisted_image_generation_message(
        self,
        *,
        message: AgentThreadMessage,
        images_dataset: ImagesDataset,
    ) -> AgentThreadMessage:
        if isinstance(message, AgentImageGenerationPartial):
            if message.image is None:
                return message
            image = await self._hydrate_persisted_image(
                image=message.image,
                images_dataset=images_dataset,
            )
            if image == message.image:
                return message
            return message.model_copy(update={"image": image})

        if isinstance(message, AgentImageGenerationCompleted):
            images = [
                await self._hydrate_persisted_image(
                    image=image,
                    images_dataset=images_dataset,
                )
                for image in message.images
            ]
            if images == message.images:
                return message
            return message.model_copy(update={"images": images})

        return message

    async def _hydrate_persisted_image(
        self,
        *,
        image: AgentGeneratedImage,
        images_dataset: ImagesDataset,
    ) -> AgentGeneratedImage:
        image_id = self._image_id_from_dataset_uri(uri=image.uri)
        if image_id is None:
            return image

        saved_image = await images_dataset.read(image_id=image_id)
        image_data = await images_dataset.read_data(image_id=image_id)
        if saved_image is None or image_data is None:
            return image

        mime_type = image.mime_type or saved_image.mime_type or "image/png"
        data_uri = (
            f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
        )
        return image.model_copy(
            update={
                "uri": data_uri,
                "mime_type": mime_type,
                "created_at": image.created_at or saved_image.created_at,
                "created_by": image.created_by or saved_image.created_by,
            }
        )

    @staticmethod
    def _image_id_from_dataset_uri(*, uri: str | None) -> str | None:
        if not isinstance(uri, str):
            return None
        parsed = urlparse(uri)
        if parsed.scheme != "dataset" or parsed.netloc != ImagesDataset.TABLE_NAME:
            return None
        values = parse_qs(parsed.query).get("id")
        if not values:
            return None
        image_id = values[0].strip()
        return image_id if image_id != "" else None

    @staticmethod
    def _stored_agent_message(
        *,
        value: Any,
        attachment: bytes | None = None,
    ) -> AgentThreadMessage | None:
        if not isinstance(value, dict):
            return None
        if attachment is not None:
            message_type = value.get("type")
            if message_type in (
                AGENT_EVENT_AUDIO_GENERATION_DELTA,
                AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
            ):
                value = {**value, "data": attachment}
        try:
            message = parse_agent_message(value)
        except Exception:
            return None
        if not isinstance(message, AgentThreadMessage):
            return None
        return message

    def _tool_log_delta_from_stored_lines(
        self,
        *,
        turn_id: str,
        item_id: str,
        namespace: str,
        call_id: str | None,
        logs: list[Any],
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentToolCallLogDelta | None:
        lines = []
        for line in logs:
            if not isinstance(line, dict):
                continue
            source = line.get("source")
            text = line.get("text")
            if source not in {"stdout", "stderr"} or not isinstance(text, str):
                continue
            lines.append({"source": source, "text": text})
        if len(lines) == 0:
            return None
        return AgentToolCallLogDelta.model_validate(
            {
                "type": AGENT_EVENT_TOOL_CALL_LOG_DELTA,
                "thread_id": self.path,
                "turn_id": turn_id,
                "item_id": item_id,
                "namespace": namespace,
                "call_id": call_id,
                "provider": provider,
                "model": model,
                "lines": lines,
            }
        )

    def make_toolkit(self) -> Toolkit:
        return Toolkit(
            name="search",
            description="tools for searching conversation history",
            tools=[
                self.grep_tool,
                self.get_message_range,
                self.count_tool,
            ],
        )

    def agent_messages(self) -> list[AgentThreadMessage]:
        messages: list[AgentThreadMessage] = []
        for row in self._rows:
            messages.extend(self._messages_from_row(row=row))

        return messages

    @tool(
        name="get_message_range",
        description="gets a range of messages, index 0 is the first message in the conversation",
    )
    def get_message_range(self, *, start: int, end: int) -> str:
        rows = self._rows[start:end]
        if len(rows) == 0:
            return "no messages were found within the specified range"
        return "matching messages:\n" + "\n".join(
            self._format_row_for_search(row=row) for row in rows
        )

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
        del pattern
        del ignore_case
        del messages_before
        del messages_after
        return str(len(self._rows))

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
        del messages_before
        del messages_after
        flags = re.IGNORECASE if ignore_case else 0
        matches = [
            row
            for row in self._rows
            if re.search(pattern, self._format_row_for_search(row=row), flags)
            is not None
        ]
        if len(matches) == 0:
            return "no messages were found with the specified pattern"
        return "matching messages:\n" + "\n".join(
            self._format_row_for_search(row=row) for row in matches
        )

    @staticmethod
    def _format_row_for_search(*, row: _StoredThreadRow) -> str:
        data = row.data
        message_type = data.get("type")
        if isinstance(message_type, str):
            return f"{message_type} at {row.timestamp}: {json.dumps(data)}"

        kind = data.get("kind")
        role = data.get("role")
        text = data.get("text")
        if isinstance(text, str) and text != "":
            return f"{role or kind or 'item'} at {row.timestamp}: {text}"

        if kind == "file":
            urls = data.get("urls")
            if isinstance(urls, list):
                return f"file at {row.timestamp}: {', '.join(str(url) for url in urls)}"

        if kind == "tool_call":
            toolkit = data.get("toolkit")
            tool = data.get("tool")
            status = data.get("status")
            return f"tool call at {row.timestamp}: {toolkit}.{tool} {status}"

        return f"{kind or 'item'} at {row.timestamp}: {json.dumps(data)}"
