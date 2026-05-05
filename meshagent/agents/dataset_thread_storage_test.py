from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from meshagent.agents.context import AgentSessionContext
from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_PENDING,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_MESSAGE_TURN_START,
    AgentError,
    AgentTextContentDelta,
    AgentTextContentStarted,
    AgentToolCallPending,
    AgentToolCallEnded,
    AgentToolCallStarted,
    TurnEnded,
    TurnInterrupted,
    TurnStart,
    TurnStartAccepted,
)
from meshagent.api import Participant
from meshagent.api.messaging import BinaryContent


class _FakeDatasets:
    def __init__(self) -> None:
        self.schemas: dict[tuple[tuple[str, ...], str], pa.Schema] = {}
        self.rows: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.optimize_calls: list[dict[str, Any]] = []

    @staticmethod
    def _key(
        *,
        table: str | None = None,
        name: str | None = None,
        namespace: list[str] | None,
    ) -> tuple[tuple[str, ...], str]:
        table_name = table if table is not None else name
        assert table_name is not None
        return (tuple(namespace or []), table_name)

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema: pa.Schema,
        mode: str,
        namespace: list[str] | None = None,
    ) -> None:
        self.create_calls.append(
            {
                "name": name,
                "schema": schema,
                "mode": mode,
                "namespace": namespace,
            }
        )
        key = self._key(name=name, namespace=namespace)
        self.schemas.setdefault(key, schema)
        self.rows.setdefault(key, [])

    async def inspect(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
    ) -> pa.Schema:
        return self.schemas[self._key(table=table, namespace=namespace)]

    async def add_columns(
        self,
        *,
        table: str,
        new_columns: dict[str, pa.Field],
        namespace: list[str] | None = None,
    ) -> None:
        key = self._key(table=table, namespace=namespace)
        schema = self.schemas[key]
        self.schemas[key] = pa.schema([*schema, *new_columns.values()])

    async def create_index(self, *, table: str, config: Any) -> None:
        del table
        del config

    async def search(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
    ) -> pa.Table:
        key = self._key(table=table, namespace=namespace)
        return pa.Table.from_pylist(self.rows.get(key, []), schema=self.schemas[key])

    async def insert(
        self,
        *,
        table: str,
        records: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        key = self._key(table=table, namespace=namespace)
        self.rows.setdefault(key, []).extend(records)

    async def optimize(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
        config: Any = None,
    ) -> None:
        self.optimize_calls.append(
            {
                "table": table,
                "namespace": namespace,
                "config": config,
            }
        )


class _FakeRoom:
    def __init__(self) -> None:
        self.datasets = _FakeDatasets()
        self.local_participant = _participant("assistant")


def _participant(name: str) -> Participant:
    return Participant(id=name, attributes={"name": name})


def _row_data(row: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(row["data"])
    assert isinstance(data, dict)
    return data


@pytest.mark.asyncio
async def test_dataset_thread_storage_uses_path_namespace_and_table() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/support/thread")

    await storage.start()
    await storage.stop()

    assert storage.namespace == ["threads", "support"]
    assert storage.table_name == "thread"
    assert room.datasets.create_calls[0]["namespace"] == ["threads", "support"]
    assert room.datasets.create_calls[0]["name"] == "thread"


@pytest.mark.asyncio
async def test_dataset_thread_storage_accepts_dataset_thread_urls() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://agents/demo/threads/main")

    await storage.start()
    await storage.stop()

    assert storage.path == "dataset://agents/demo/threads/main"
    assert storage.namespace == ["agents", "demo", "threads"]
    assert storage.table_name == "main"
    assert room.datasets.create_calls[0]["namespace"] == [
        "agents",
        "demo",
        "threads",
    ]
    assert room.datasets.create_calls[0]["name"] == "main"


def test_dataset_thread_storage_rejects_non_dataset_paths() -> None:
    room = _FakeRoom()

    with pytest.raises(ValueError, match="must start with dataset://"):
        DatasetThreadStorage(room=room, path="/agents/demo/threads/main")


def test_dataset_thread_storage_rejects_triple_slash_dataset_urls() -> None:
    room = _FakeRoom()

    with pytest.raises(ValueError, match="must use dataset://path"):
        DatasetThreadStorage(room=room, path="dataset:///agents/demo/threads/main")


def test_dataset_thread_storage_rejects_thread_document_paths() -> None:
    room = _FakeRoom()

    with pytest.raises(ValueError, match="must not end with .thread"):
        DatasetThreadStorage(
            room=room, path="dataset://agents/demo/threads/main.thread"
        )


@pytest.mark.asyncio
async def test_dataset_thread_storage_persists_only_accepted_user_turns() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    unaccepted = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="dataset://threads/demo",
        message_id="unaccepted",
        content=[{"type": "text", "text": "do not save"}],
    )
    accepted = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="dataset://threads/demo",
        message_id="accepted",
        content=[{"type": "text", "text": "save this"}],
    )
    storage.push_message(message=unaccepted, sender=_participant("caller"))
    storage.push_message(message=accepted, sender=_participant("caller"))
    storage.push_message(
        message=TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="dataset://threads/demo",
            source_message_id="accepted",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 1
    data = _row_data(rows[0])
    assert data["kind"] == "message"
    assert data["role"] == "user"
    assert data["text"] == "save this"
    assert data["sender_name"] == "caller"


@pytest.mark.asyncio
async def test_dataset_thread_storage_optimizes_after_append_threshold() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(
        room=room,
        path="dataset://threads/demo",
        optimize_after_append_count=2,
    )
    await storage.start()

    for index in range(2):
        message_id = f"message-{index}"
        storage.push_message(
            message=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="dataset://threads/demo",
                message_id=message_id,
                content=[{"type": "text", "text": f"message {index}"}],
            ),
            sender=_participant("caller"),
        )
        storage.push_message(
            message=TurnStartAccepted(
                type=AGENT_EVENT_TURN_START_ACCEPTED,
                thread_id="dataset://threads/demo",
                source_message_id=message_id,
            )
        )

    await storage.stop()

    assert len(room.datasets.optimize_calls) == 1
    optimize_call = room.datasets.optimize_calls[0]
    assert optimize_call["table"] == "demo"
    assert optimize_call["namespace"] == ["threads"]
    assert optimize_call["config"].compact_files is True
    assert optimize_call["config"].optimize_indices is False
    assert optimize_call["config"].cleanup_old_versions is False


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_partial_text_on_interrupt() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentTextContentStarted(
            type=AGENT_EVENT_TEXT_CONTENT_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="partial ",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="answer",
        )
    )
    storage.push_message(
        message=TurnInterrupted(
            type=AGENT_EVENT_TURN_INTERRUPTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            source_message_id="interrupt-1",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 1
    data = _row_data(rows[0])
    assert data["kind"] == "message"
    assert data["role"] == "assistant"
    assert data["status"] == "cancelled"
    assert data["text"] == "partial answer"


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_unended_text_on_successful_turn_end() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="complete enough",
        )
    )
    storage.push_message(
        message=TurnEnded(
            type=AGENT_EVENT_TURN_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            error=None,
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 1
    data = _row_data(rows[0])
    assert data["status"] == "completed"
    assert data["text"] == "complete enough"


@pytest.mark.asyncio
async def test_dataset_thread_storage_drops_pending_tool_and_flushes_started_tool() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallPending(
            type=AGENT_EVENT_TOOL_CALL_PENDING,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="pending-tool",
            toolkit="shell",
            tool="exec",
        )
    )
    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="started-tool",
            toolkit="shell",
            tool="exec",
            arguments={"cmd": "sleep 10"},
        )
    )
    storage.push_message(
        message=TurnEnded(
            type=AGENT_EVENT_TURN_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            error=AgentError(message="cancelled", code=None),
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 1
    data = _row_data(rows[0])
    assert data["kind"] == "tool_call"
    assert rows[0]["item_id"] == "started-tool"
    assert data["status"] == "failed"
    assert data["toolkit"] == "shell"
    assert data["tool"] == "exec"


@pytest.mark.asyncio
async def test_dataset_thread_storage_persists_image_generation_tool_result() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "1024x768", "output_format": "png"},
        )
    )
    storage.push_message(
        message=AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            result=BinaryContent(
                data=b"fake-image",
                headers={"mime_type": "image/png", "quality": "high"},
            ),
        )
    )
    await storage.stop()

    image_rows = room.datasets.rows[((), "images")]
    assert len(image_rows) == 1
    assert image_rows[0]["data"] == b"fake-image"
    assert image_rows[0]["mime_type"] == "image/png"
    assert image_rows[0]["created_by"] == "assistant"

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 1
    assert thread_rows[0]["item_id"] == "image-tool"
    data = _row_data(thread_rows[0])
    assert data["kind"] == "image"
    assert data["status"] == "completed"
    assert data["image_id"] == image_rows[0]["id"]
    assert data["mime_type"] == "image/png"
    assert data["created_by"] == "assistant"
    assert data["width"] == 1024
    assert data["height"] == 768


@pytest.mark.asyncio
async def test_dataset_thread_storage_loads_rows_sorted_by_sequence_for_restore() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    schema = storage._schema()
    key = (("threads",), "demo")
    room.datasets.schemas[key] = schema
    room.datasets.rows[key] = [
        {
            "turn_id": None,
            "item_id": "second",
            "sequence": 2,
            "timestamp": datetime(2026, 3, 11, tzinfo=timezone.utc),
            "data": json.dumps(
                {
                    "kind": "message",
                    "role": "assistant",
                    "status": "completed",
                    "text": "answer",
                }
            ),
        },
        {
            "turn_id": None,
            "item_id": "first",
            "sequence": 1,
            "timestamp": datetime(2026, 3, 10, tzinfo=timezone.utc),
            "data": json.dumps(
                {
                    "kind": "message",
                    "role": "user",
                    "status": "completed",
                    "sender_name": "caller",
                    "text": "question",
                }
            ),
        },
    ]

    await storage.start()
    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context)
    await storage.stop()

    assert context.messages == [
        {
            "role": "user",
            "content": "caller said at 2026-03-10T00:00:00Z: question",
        },
        {"role": "assistant", "content": "answer"},
    ]
