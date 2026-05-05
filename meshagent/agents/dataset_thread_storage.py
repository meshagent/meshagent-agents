from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pyarrow as pa

from meshagent.api import DatasetOptimizeConfig, Participant, RoomClient
from meshagent.api.messaging import BinaryContent, TextContent
from meshagent.tools import Toolkit, tool

from .context import AgentSessionContext
from .images_dataset import ImagesDataset
from .messages import (
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
    AgentThreadEvent,
    AgentThreadImage,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallLogDelta,
    AgentToolCallPending,
    AgentToolCallStarted,
    ThreadCleared,
    TurnEnded,
    TurnInterrupted,
    TurnStart,
    TurnStartAccepted,
    TurnStartRejected,
    TurnSteer,
    TurnSteerAccepted,
    TurnSteerRejected,
)
from .thread_adapter import default_format_message
from .thread_storage import ThreadStorage

logger = logging.getLogger("agent.dataset_thread_storage")

_DATASET_THREAD_URL_PREFIX = "dataset://"
_IMAGE_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")

_TERMINAL_REASON_TO_STATUS = {
    "completed": "completed",
    "cancelled": "cancelled",
    "failed": "failed",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _mime_type_from_output_format(output_format: Any) -> str:
    if not isinstance(output_format, str):
        return "image/png"

    normalized = output_format.strip().lower().lstrip(".")
    if normalized == "":
        return "image/png"
    if normalized == "jpg":
        normalized = "jpeg"
    return f"image/{normalized}"


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
    sequence: int
    timestamp: str
    data: dict[str, Any]


@dataclass(slots=True)
class _QueuedThreadMessage:
    message: AgentMessage
    sender: Participant | None


@dataclass(slots=True)
class _StopQueue:
    future: asyncio.Future[None]


@dataclass(slots=True)
class _ActiveContent:
    kind: Literal["text", "reasoning", "file"]
    turn_id: str
    item_id: str
    message_id: str
    parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ActiveToolCall:
    turn_id: str
    item_id: str
    message_id: str
    toolkit: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    stage: Literal["pending", "in_progress", "started"] | None = None
    logs: list[dict[str, str]] = field(default_factory=list)


class DatasetThreadStorage(ThreadStorage):
    def __init__(
        self,
        *,
        room: RoomClient,
        path: str,
        max_append_message_count: int = 25,
        optimize_after_append_count: int = 25,
    ) -> None:
        self._room = room
        normalized_path = _normalize_dataset_thread_storage_path(path=path)
        self._path = normalized_path.url
        path_parts = _normalize_path_parts(path=normalized_path.table_path)
        self._table_name = path_parts[-1]
        namespace = path_parts[:-1]
        self._namespace = namespace if len(namespace) > 0 else None
        self._max_append_message_count = max_append_message_count
        self._optimize_after_append_count = optimize_after_append_count
        self._appends_since_optimize = 0
        self._optimize_requested = False
        self._optimize_task: asyncio.Task[None] | None = None
        self._ready = False
        self._queue: asyncio.Queue[_QueuedThreadMessage | _StopQueue] = asyncio.Queue()
        self._processor_task: asyncio.Task[None] | None = None
        self._rows: list[_StoredThreadRow] = []
        self._next_sequence = 0
        self._pending_user_turns: dict[str, _QueuedThreadMessage] = {}
        self._active_content_by_item_id: dict[str, _ActiveContent] = {}
        self._active_tool_calls_by_item_id: dict[str, _ActiveToolCall] = {}
        self._images_db = ImagesDataset(room=self._room)

    @property
    def path(self) -> str:
        return self._path

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def namespace(self) -> list[str] | None:
        return None if self._namespace is None else list(self._namespace)

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("turn_id", pa.string()),
                pa.field("item_id", pa.string(), nullable=False),
                pa.field("sequence", pa.int64(), nullable=False),
                pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("data", pa.large_string(), nullable=False),
            ]
        )

    async def start(self) -> None:
        await self._ensure_ready()
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        processor_task = self._processor_task
        if processor_task is None:
            await self._flush_all_active(reason="cancelled")
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

        schema = self._schema()
        await self._room.datasets.create_table_with_schema(
            name=self._table_name,
            schema=schema,
            mode="create_if_not_exists",
            namespace=self._namespace,
        )

        with contextlib.suppress(Exception):
            existing_schema = await self._room.datasets.inspect(
                table=self._table_name,
                namespace=self._namespace,
            )
            existing_names = set(existing_schema.names)
            missing_columns = {
                field.name: field
                for field in schema
                if field.name not in existing_names
            }
            if len(missing_columns) > 0:
                await self._room.datasets.add_columns(
                    table=self._table_name,
                    new_columns=missing_columns,
                    namespace=self._namespace,
                )

        rows = await self._room.datasets.search(
            table=self._table_name,
            namespace=self._namespace,
        )
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
        if not isinstance(raw_data, str):
            return None

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        raw_turn_id = record.get("turn_id")
        turn_id = raw_turn_id if isinstance(raw_turn_id, str) else None
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
        return _StoredThreadRow(
            turn_id=turn_id,
            item_id=item_id,
            sequence=sequence,
            timestamp=timestamp,
            data=data,
        )

    def push_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant | None = None,
    ) -> None:
        try:
            self._queue.put_nowait(_QueuedThreadMessage(message=message, sender=sender))
        except asyncio.QueueShutDown:
            logger.debug("dropping dataset thread message after queue shutdown")

    async def _process_queue(self) -> None:
        while True:
            queued = await self._queue.get()
            if isinstance(queued, _StopQueue):
                try:
                    await self._flush_all_active(reason="cancelled")
                except Exception as exc:
                    if not queued.future.done():
                        queued.future.set_exception(exc)
                else:
                    if not queued.future.done():
                        queued.future.set_result(None)
                return

            await self._handle_message(message=queued.message, sender=queued.sender)

    async def _handle_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant | None,
    ) -> None:
        if isinstance(message, ThreadCleared):
            self._pending_user_turns.clear()
            self._active_content_by_item_id.clear()
            self._active_tool_calls_by_item_id.clear()
            return

        if isinstance(message, (TurnStart, TurnSteer)):
            self._pending_user_turns[message.message_id] = _QueuedThreadMessage(
                message=message,
                sender=sender,
            )
            return

        if isinstance(message, TurnStartAccepted):
            await self._commit_pending_user_turn(
                source_message_id=message.source_message_id,
                turn_id=None,
                accepted_message=message,
            )
            return

        if isinstance(message, TurnSteerAccepted):
            await self._commit_pending_user_turn(
                source_message_id=message.source_message_id,
                turn_id=message.turn_id,
                accepted_message=message,
            )
            return

        if isinstance(message, TurnSteerRejected):
            self._pending_user_turns.pop(message.source_message_id, None)
            return

        if isinstance(message, TurnStartRejected):
            self._pending_user_turns.pop(message.source_message_id, None)
            return

        if isinstance(message, AgentTextContentStarted):
            self._active_content_by_item_id[message.item_id] = _ActiveContent(
                kind="text",
                turn_id=message.turn_id,
                item_id=message.item_id,
                message_id=message.message_id,
            )
            return

        if isinstance(message, AgentTextContentDelta):
            active = self._ensure_active_content(message=message, kind="text")
            active.parts.append(message.text)
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
            )
            return

        if isinstance(message, AgentReasoningContentDelta):
            active = self._ensure_active_content(message=message, kind="reasoning")
            active.parts.append(message.text)
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
            )
            return

        if isinstance(message, AgentFileContentDelta):
            active = self._ensure_active_content(message=message, kind="file")
            active.parts.append(message.url)
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
                    stage="in_progress",
                )
                self._active_tool_calls_by_item_id[message.item_id] = active
            active.logs.extend([line.model_dump(mode="json") for line in message.lines])
            return

        if isinstance(message, AgentToolCallEnded):
            await self._flush_tool_call(
                item_id=message.item_id,
                reason="completed" if message.error is None else "failed",
                ended_message=message,
            )
            return

        if isinstance(message, AgentThreadImage):
            await self._append_row(
                turn_id=message.turn_id,
                item_id=message.item_id or message.message_id,
                data={
                    "kind": "image",
                    "status": message.status or "completed",
                    "image_id": message.image_id,
                    "mime_type": message.mime_type,
                    "created_at": message.created_at,
                    "created_by": message.created_by,
                    "width": message.width,
                    "height": message.height,
                    "status_detail": message.status_detail,
                    "message": message.model_dump(mode="json"),
                },
            )
            return

        if isinstance(message, AgentThreadEvent):
            await self._append_row(
                turn_id=None,
                item_id=message.message_id,
                data={
                    "kind": "event",
                    "status": "completed",
                    "message": message.model_dump(mode="json"),
                },
            )
            return

        if isinstance(message, TurnInterrupted):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="cancelled",
            )
            return

        if isinstance(message, TurnEnded):
            await self._flush_turn_active_items(
                turn_id=message.turn_id,
                reason="failed" if message.error is not None else "completed",
            )

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
            return active

        active = _ActiveContent(
            kind=kind,
            turn_id=message.turn_id,
            item_id=message.item_id,
            message_id=message.message_id,
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
            )
            self._active_tool_calls_by_item_id[message.item_id] = active

        active.toolkit = message.toolkit
        active.tool = message.tool
        active.arguments = message.arguments
        if isinstance(message, AgentToolCallStarted):
            active.stage = "started"
        elif isinstance(message, AgentToolCallInProgress):
            active.stage = "in_progress"
        else:
            active.stage = "pending"

    async def _commit_pending_user_turn(
        self,
        *,
        source_message_id: str,
        turn_id: str | None,
        accepted_message: TurnStartAccepted | TurnSteerAccepted,
    ) -> None:
        queued = self._pending_user_turns.pop(source_message_id, None)
        if queued is None:
            return
        if not isinstance(queued.message, (TurnStart, TurnSteer)):
            return

        text_parts: list[str] = []
        attachments: list[str] = []
        for item in queued.message.content:
            if isinstance(item, AgentTextContent):
                normalized_text = item.text.strip()
                if normalized_text != "":
                    text_parts.append(normalized_text)
            elif isinstance(item, AgentFileContent):
                normalized_url = item.url.strip()
                if normalized_url != "":
                    attachments.append(normalized_url)

        if len(text_parts) == 0 and len(attachments) == 0:
            return

        sender_name = self._sender_name(sender=queued.sender) or "user"
        await self._append_row(
            turn_id=turn_id,
            item_id=queued.message.message_id,
            data={
                "kind": "message",
                "role": "user",
                "status": "completed",
                "text": "\n\n".join(text_parts),
                "attachments": attachments,
                "sender_name": sender_name,
                "request": queued.message.model_dump(mode="json"),
                "accepted": accepted_message.model_dump(mode="json"),
            },
        )

    async def _flush_content_item(
        self,
        *,
        item_id: str,
        reason: Literal["completed", "cancelled", "failed"],
        ended_message: AgentMessage | None = None,
    ) -> None:
        active = self._active_content_by_item_id.pop(item_id, None)
        if active is None:
            return

        status = _TERMINAL_REASON_TO_STATUS[reason]
        if active.kind in {"text", "reasoning"}:
            text = "".join(active.parts)
            if text == "":
                return
            await self._append_row(
                turn_id=active.turn_id,
                item_id=active.item_id,
                data={
                    "kind": "message" if active.kind == "text" else "reasoning",
                    "role": "assistant",
                    "status": status,
                    "text": text,
                    "message": None
                    if ended_message is None
                    else ended_message.model_dump(mode="json"),
                },
            )
            return

        urls = [part for part in active.parts if part.strip() != ""]
        if len(urls) == 0:
            return
        await self._append_row(
            turn_id=active.turn_id,
            item_id=active.item_id,
            data={
                "kind": "file",
                "role": "assistant",
                "status": status,
                "urls": urls,
                "message": None
                if ended_message is None
                else ended_message.model_dump(mode="json"),
            },
        )

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

        await self._append_row(
            turn_id=active.turn_id,
            item_id=active.item_id,
            data={
                "kind": "tool_call",
                "role": "assistant",
                "status": _TERMINAL_REASON_TO_STATUS[reason],
                "toolkit": active.toolkit,
                "tool": active.tool,
                "arguments": active.arguments,
                "logs": active.logs,
                "message": None
                if ended_message is None
                else ended_message.model_dump(mode="json"),
            },
        )

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
            return False

        image_bytes: bytes | None = None
        mime_type: str | None = None
        annotations: dict[str, str] = {}

        if isinstance(message.result, BinaryContent):
            image_bytes = message.result.get_data()
            headers = message.result.headers
            raw_mime_type = headers.get("mime_type")
            if isinstance(raw_mime_type, str) and raw_mime_type.strip() != "":
                mime_type = raw_mime_type.strip()
            for key in ("background", "output_format", "quality", "size", "status"):
                value = headers.get(key)
                if isinstance(value, str) and value.strip() != "":
                    annotations[key] = value.strip()
        elif isinstance(message.result, TextContent):
            encoded_image = message.result.text.strip()
            if encoded_image != "":
                try:
                    image_bytes = base64.b64decode(encoded_image)
                except Exception:
                    logger.warning(
                        "unable to decode image_generation tool result for %s",
                        message.item_id,
                    )
                    return False

        if image_bytes is None:
            return False

        arguments = active_tool_call.arguments or {}
        if mime_type is None:
            mime_type = _mime_type_from_output_format(arguments.get("output_format"))
        for key in (
            "background",
            "model",
            "moderation",
            "output_format",
            "quality",
            "size",
        ):
            value = arguments.get(key)
            if (
                isinstance(value, str)
                and value.strip() != ""
                and key not in annotations
            ):
                annotations[key] = value.strip()

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

        created_by = self._room.local_participant.get_attribute("name")
        if not isinstance(created_by, str):
            created_by = ""

        try:
            saved_image = await self._images_db.save(
                data=image_bytes,
                mime_type=mime_type,
                created_by=created_by,
                annotations=annotations,
            )
        except Exception as ex:
            logger.error(
                "failed to persist generated image for dataset thread %s",
                message.item_id,
                exc_info=ex,
            )
            await self._append_row(
                turn_id=active_tool_call.turn_id,
                item_id=active_tool_call.item_id,
                data={
                    "kind": "image",
                    "role": "assistant",
                    "status": "failed",
                    "status_detail": f"Image save failed: {ex}",
                    "error": str(ex),
                    "message": message.model_dump(mode="json"),
                },
            )
            return True

        await self._append_row(
            turn_id=active_tool_call.turn_id,
            item_id=active_tool_call.item_id,
            data={
                "kind": "image",
                "role": "assistant",
                "status": "completed",
                "image_id": saved_image.id,
                "mime_type": saved_image.mime_type,
                "created_at": saved_image.created_at,
                "created_by": saved_image.created_by,
                "width": width,
                "height": height,
                "status_detail": "Image saved",
                "message": message.model_dump(mode="json"),
            },
        )
        logger.info(
            "saved dataset-thread image %s for tool call %s",
            saved_image.id,
            message.item_id,
        )
        return True

    async def _flush_turn_active_items(
        self,
        *,
        turn_id: str,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
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

    async def _flush_all_active(
        self,
        *,
        reason: Literal["completed", "cancelled", "failed"],
    ) -> None:
        for item_id in list(self._active_content_by_item_id):
            await self._flush_content_item(item_id=item_id, reason=reason)
        for item_id in list(self._active_tool_calls_by_item_id):
            await self._flush_tool_call(item_id=item_id, reason=reason)

    async def _append_row(
        self,
        *,
        turn_id: str | None,
        item_id: str,
        data: dict[str, Any],
    ) -> None:
        await self._ensure_ready()
        timestamp = _now_iso()
        sequence = self._next_sequence
        self._next_sequence += 1
        normalized_item_id = item_id if item_id.strip() != "" else str(uuid.uuid4())
        row = _StoredThreadRow(
            turn_id=turn_id,
            item_id=normalized_item_id,
            sequence=sequence,
            timestamp=timestamp,
            data=data,
        )
        await self._room.datasets.insert(
            table=self._table_name,
            namespace=self._namespace,
            records=[
                {
                    "turn_id": row.turn_id,
                    "item_id": row.item_id,
                    "sequence": row.sequence,
                    "timestamp": row.timestamp,
                    "data": json.dumps(row.data),
                }
            ],
        )
        self._rows.append(row)
        self._rows.sort(key=lambda stored: stored.sequence)
        self._note_appended_row()

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
                await self._room.datasets.optimize(
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

    def restore_session_context(self, *, context: AgentSessionContext) -> None:
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

    def _restore_row(
        self,
        *,
        context: AgentSessionContext,
        row: _StoredThreadRow,
    ) -> None:
        data = row.data
        kind = data.get("kind")
        role = data.get("role")
        text = data.get("text")
        if kind == "message" and isinstance(text, str) and text != "":
            if role == "assistant":
                context.append_assistant_message(text)
            else:
                sender_name = data.get("sender_name")
                context.append_user_message(
                    default_format_message(
                        user_name=sender_name
                        if isinstance(sender_name, str)
                        else "user",
                        message=text,
                        iso_timestamp=row.timestamp,
                    )
                )

        attachments = data.get("attachments")
        if isinstance(attachments, list):
            for attachment in attachments:
                if not isinstance(attachment, str) or attachment.strip() == "":
                    continue
                if role == "assistant":
                    context.append_assistant_message(
                        f"assistant attached a file available at {attachment}"
                    )
                else:
                    sender_name = data.get("sender_name")
                    context.append_user_message(
                        f"{sender_name if isinstance(sender_name, str) else 'a user'} "
                        f"attached a file available at {attachment}"
                    )

        urls = data.get("urls")
        if kind == "file" and isinstance(urls, list):
            for url in urls:
                if not isinstance(url, str) or url.strip() == "":
                    continue
                context.append_assistant_message(
                    f"assistant attached a file available at {url}"
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
