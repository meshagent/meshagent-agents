from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable

import pyarrow as pa
import pytest

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.agent_event_reader import (
    AccumulatingAgentEventReader,
    AgentEventReader,
    AgentEventReaderCallbacks,
    _BufferedToolCall,
)
from meshagent.agents.context import AgentSessionContext, SessionUsage
from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.images_dataset import ImagesDataset
from meshagent.agents.messages import (
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_AUDIO_GENERATION_DELTA,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_DELTA,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_PENDING,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
    AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
    AGENT_MESSAGE_TURN_STEER,
    AgentContextWindowUsage,
    AgentAudioGenerationDelta,
    AgentAudioTranscriptionCompleted,
    AgentAudioTranscriptionDelta,
    AgentContextCompacted,
    AgentError,
    AgentGeneratedImage,
    AgentImageGenerationCompleted,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentRealtimeAudioChunk,
    AgentRealtimeAudioCommit,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentThreadEvent,
    AgentToolCallArgumentsDelta,
    AgentToolCallPending,
    AgentToolCallEnded,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallStarted,
    AgentUsageUpdated,
    TurnEnded,
    TurnInterrupted,
    TurnStart,
    TurnStartAccepted,
    TurnSteer,
    TurnSteerAccepted,
)
from meshagent.api import DatasetJson, Participant, RoomException
from meshagent.api.messaging import BinaryContent, JsonContent, TextContent


class _FakeDatasets:
    def __init__(self) -> None:
        self.schemas: dict[tuple[tuple[str, ...], str], pa.Schema] = {}
        self.rows: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.insert_calls: list[dict[str, Any]] = []
        self.merge_calls: list[dict[str, Any]] = []
        self.optimize_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.raise_on_existing_create = False

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
        if self.raise_on_existing_create and key in self.schemas:
            raise ValueError(f"Table {name!r} already exists with a different schema")
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
        where: str | dict[str, Any] | None = None,
        limit: int | None = None,
        select: list[str] | None = None,
    ) -> pa.Table:
        key = self._key(table=table, namespace=namespace)
        rows = list(self.rows.get(key, []))
        if isinstance(where, dict):
            rows = [
                row
                for row in rows
                if all(row.get(field) == value for field, value in where.items())
            ]
        if limit is not None:
            rows = rows[:limit]
        if select is not None:
            rows = [{field: row.get(field) for field in select} for row in rows]
        return pa.Table.from_pylist(rows)

    async def insert(
        self,
        *,
        table: str,
        records: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        self.insert_calls.append(
            {
                "table": table,
                "records": records,
                "namespace": namespace,
            }
        )
        key = self._key(table=table, namespace=namespace)
        self.rows.setdefault(key, []).extend(
            [self._stored_record(record) for record in records]
        )

    @staticmethod
    def _stored_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to_json() if isinstance(value, DatasetJson) else value
            for key, value in record.items()
        }

    async def update(
        self,
        *,
        table: str,
        where: str,
        values: dict[str, Any],
        namespace: list[str] | None = None,
    ) -> None:
        self.update_calls.append(
            {
                "table": table,
                "where": where,
                "values": values,
                "namespace": namespace,
            }
        )
        prefix = "sequence = "
        assert where.startswith(prefix)
        sequence = int(where[len(prefix) :])
        key = self._key(table=table, namespace=namespace)
        stored_values = self._stored_record(values)
        for row in self.rows.setdefault(key, []):
            if row.get("sequence") == sequence:
                row.update(stored_values)
                return
        raise AssertionError(f"row not found for {where}")

    async def merge(
        self,
        *,
        table: str,
        on: str,
        records: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        self.merge_calls.append(
            {
                "table": table,
                "on": on,
                "records": records,
                "namespace": namespace,
            }
        )
        key = self._key(table=table, namespace=namespace)
        rows = self.rows.setdefault(key, [])
        stored_records = [self._stored_record(record) for record in records]
        for stored_record in stored_records:
            matched = False
            for row in rows:
                if row.get(on) == stored_record.get(on):
                    row.update(stored_record)
                    matched = True
                    break
            if not matched:
                rows.append(stored_record)

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


class _BlockingInspectDatasets(_FakeDatasets):
    def __init__(self) -> None:
        super().__init__()
        self.inspect_started = asyncio.Event()
        self.inspect_release = asyncio.Event()

    async def inspect(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
    ) -> pa.Schema:
        self.inspect_started.set()
        await self.inspect_release.wait()
        return await super().inspect(table=table, namespace=namespace)


class _BlockingInsertDatasets(_FakeDatasets):
    def __init__(self) -> None:
        super().__init__()
        self.insert_started = asyncio.Event()
        self.insert_release = asyncio.Event()

    async def insert(
        self,
        *,
        table: str,
        records: list[dict[str, Any]],
        namespace: list[str] | None = None,
    ) -> None:
        self.insert_started.set()
        await self.insert_release.wait()
        await super().insert(table=table, records=records, namespace=namespace)


class _EventuallyVisibleSearchDatasets(_FakeDatasets):
    def __init__(self, *, missing_searches: int) -> None:
        super().__init__()
        self._missing_searches = missing_searches
        self.search_calls = 0

    async def search(
        self,
        *,
        table: str,
        namespace: list[str] | None = None,
        where: str | dict[str, Any] | None = None,
        limit: int | None = None,
        select: list[str] | None = None,
    ) -> pa.Table:
        self.search_calls += 1
        if self.search_calls <= self._missing_searches:
            raise RoomException(f"Table {table!r} does not exist", status_code=404)
        return await super().search(
            table=table,
            namespace=namespace,
            where=where,
            limit=limit,
            select=select,
        )


class _FakeRoom:
    def __init__(self, datasets: _FakeDatasets | None = None) -> None:
        self.datasets = datasets or _FakeDatasets()
        self.local_participant = _participant("assistant")


def _participant(name: str) -> Participant:
    return Participant(id=name, attributes={"name": name})


class _DatasetStorageTestReader(AccumulatingAgentEventReader):
    def _append_user_text(self, text: str) -> None:
        self._emit_context_message({"role": "user", "content": text})

    def _append_user_content(self, content: list[dict[str, Any]]) -> None:
        self._emit_context_message({"role": "user", "content": content})

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if phase is not None:
            message["phase"] = phase
        self._emit_context_message(message)

    def _append_assistant_reasoning(self, *, text: str) -> None:
        self._emit_context_message({"role": "assistant", "content": text})

    def _append_assistant_file(self, *, url: str) -> None:
        self._emit_context_message({"role": "assistant", "content": url})

    def _append_thread_event(self, *, event: dict[str, Any]) -> None:
        self._emit_context_message(
            {"role": "assistant", "content": [{"type": "event", "event": event}]}
        )

    def _append_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "item_id": tool_call.item_id,
                        "namespace": tool_call.namespace,
                        "call_id": tool_call.call_id,
                        "toolkit": tool_call.toolkit,
                        "tool": tool_call.tool,
                        "arguments": tool_call.arguments_dict(),
                        "result": result,
                        "error": error,
                        "logs": tool_call.logs,
                    }
                ],
            }
        )

    def _append_image_generation_event(
        self,
        *,
        event_type: str,
        turn_id: str,
        item_id: str,
        call_id: str | None,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        images: list[dict[str, Any]],
        status: str,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "image_generation",
                        "event_type": event_type,
                        "turn_id": turn_id,
                        "item_id": item_id,
                        "call_id": call_id,
                        "toolkit": toolkit,
                        "tool": tool,
                        "arguments": arguments,
                        "images": images,
                        "status": status,
                    }
                ],
            }
        )

    def _append_audio_generation_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [message.model_dump(mode="json")],
            }
        )

    def _append_audio_transcription_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "content": [message.model_dump(mode="json")],
            }
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)


class _DatasetStorageTestAdapter(LLMAdapter[dict[str, Any]]):
    def make_agent_event_reader(
        self,
        *,
        emit_message: Callable[[dict[str, Any]], None],
        callbacks: AgentEventReaderCallbacks | None = None,
    ) -> AgentEventReader:
        return _DatasetStorageTestReader(
            emit_message=emit_message,
            callbacks=callbacks,
        )


def _test_llm_adapter() -> LLMAdapter[dict[str, Any]]:
    return _DatasetStorageTestAdapter()


def _row_data(row: dict[str, Any]) -> dict[str, Any]:
    raw_data = row["data"]
    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    assert isinstance(data, dict)
    return data


@pytest.mark.asyncio
async def test_dataset_thread_storage_start_does_not_wait_for_dataset_setup() -> None:
    datasets = _BlockingInspectDatasets()
    room = _FakeRoom(datasets=datasets)
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")

    await asyncio.wait_for(storage.start(), timeout=0.1)
    await asyncio.wait_for(datasets.inspect_started.wait(), timeout=0.1)

    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    assert (("threads",), "demo") not in datasets.rows

    datasets.inspect_release.set()
    await storage.stop()

    thread_rows = datasets.rows[(("threads",), "demo")]
    assert _row_data(thread_rows[0])["message_id"] == "audio-commit-1"


@pytest.mark.asyncio
async def test_dataset_thread_storage_waits_for_created_table_to_be_searchable() -> (
    None
):
    datasets = _EventuallyVisibleSearchDatasets(missing_searches=3)
    room = _FakeRoom(datasets=datasets)
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")

    await storage.wait_until_ready()

    assert storage._ready
    assert datasets.search_calls == 4
    assert len(datasets.create_calls) == 4


@pytest.mark.asyncio
async def test_dataset_thread_storage_batches_queued_writes() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    await storage.wait_until_ready()

    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    storage.push_message(
        message=AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id="dataset://threads/demo",
            message_id="usage-1",
            usage={},
            context_window=AgentContextWindowUsage(used_tokens=0),
        )
    )
    await storage.stop()

    assert len(room.datasets.insert_calls) == 1
    assert len(room.datasets.insert_calls[0]["records"]) == 2


@pytest.mark.asyncio
async def test_dataset_thread_storage_flush_drains_queued_writes_without_stopping() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    await storage.wait_until_ready()

    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    await storage.flush()

    assert room.datasets.rows[(("threads",), "demo")] == []
    unflushed_messages = storage.unflushed_agent_messages()
    assert len(unflushed_messages) == 1
    assert isinstance(unflushed_messages[0], AgentRealtimeAudioCommit)
    assert unflushed_messages[0].message_id == "audio-commit-1"

    storage.push_message(
        message=AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id="dataset://threads/demo",
            message_id="usage-1",
            usage={},
            context_window=AgentContextWindowUsage(used_tokens=0),
        )
    )
    await storage.stop()

    assert len(room.datasets.rows[(("threads",), "demo")]) == 2


@pytest.mark.asyncio
async def test_dataset_thread_storage_exposes_unflushed_agent_messages() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    await storage.wait_until_ready()

    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    await storage.flush()

    unflushed_messages = storage.unflushed_agent_messages()
    assert len(unflushed_messages) == 1
    assert isinstance(unflushed_messages[0], AgentRealtimeAudioCommit)
    assert unflushed_messages[0].message_id == "audio-commit-1"

    await storage.stop()


@pytest.mark.asyncio
async def test_dataset_thread_storage_coalesces_audio_transcription_into_commit() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    await storage.wait_until_ready()

    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    storage.push_message(
        message=TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="dataset://threads/demo",
            message_id="accepted-1",
            source_message_id="audio-commit-1",
            turn_id="turn-1",
        )
    )
    storage.push_message(
        message=AgentAudioTranscriptionDelta(
            type=AGENT_EVENT_AUDIO_TRANSCRIPTION_DELTA,
            thread_id="dataset://threads/demo",
            message_id="transcript-delta-1",
            turn_id="turn-1",
            item_id="realtime-item-1",
            role="user",
            text="hello",
        )
    )
    storage.push_message(
        message=AgentAudioTranscriptionCompleted(
            type=AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
            thread_id="dataset://threads/demo",
            message_id="transcript-completed-1",
            turn_id="turn-1",
            item_id="realtime-item-1",
            role="user",
            text="hello there",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            message_id="assistant-1",
            turn_id="turn-1",
            item_id="assistant-item-1",
            text="assistant response",
        )
    )
    storage.push_message(
        message=TurnEnded(
            type=AGENT_EVENT_TURN_ENDED,
            thread_id="dataset://threads/demo",
            message_id="ended-1",
            turn_id="turn-1",
            error=None,
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    audio_commit = _row_data(thread_rows[0])
    assert audio_commit["type"] == AGENT_MESSAGE_REALTIME_AUDIO_COMMIT
    assert audio_commit["message_id"] == "audio-commit-1"
    assert audio_commit["turn_id"] == "turn-1"
    assert audio_commit["text"] == "hello there"
    assert audio_commit["status"] == "completed"
    assert audio_commit["transcription_item_id"] == "realtime-item-1"
    assert _row_data(thread_rows[2])["text"] == "assistant response"


@pytest.mark.asyncio
async def test_dataset_thread_storage_persists_binary_agent_message_attachment() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(
        room=room,
        path="dataset://threads/demo",
        persist_audio_input=True,
    )
    await storage.start()

    storage.push_message(
        message=AgentRealtimeAudioChunk(
            type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
            thread_id="dataset://threads/demo",
            message_id="audio-input-1",
            data=b"\xf7\x00\x01",
        )
    )
    storage.push_message(
        message=AgentAudioGenerationDelta(
            type=AGENT_EVENT_AUDIO_GENERATION_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="audio-output-1",
            data=b"\xf7\x02\x03",
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert thread_rows[0]["attachment"] == b"\xf7\x00\x01"
    assert thread_rows[1]["attachment"] == b"\xf7\x02\x03"
    assert _row_data(thread_rows[0])["data"] == ""
    assert _row_data(thread_rows[1])["data"] == ""

    restored = storage.agent_messages()
    assert isinstance(restored[0], AgentRealtimeAudioChunk)
    assert restored[0].data == b"\xf7\x00\x01"
    assert isinstance(restored[1], AgentAudioGenerationDelta)
    assert restored[1].data == b"\xf7\x02\x03"


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_realtime_audio_chunk_with_llm_reader() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(
        room=room,
        path="dataset://threads/demo",
        persist_audio_input=True,
    )
    await storage.start()

    storage.push_message(
        message=AgentRealtimeAudioChunk(
            type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
            thread_id="dataset://threads/demo",
            message_id="audio-input-1",
            data=b"\xff\x00\x01",
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())
    assert "agent_events" not in context.metadata


@pytest.mark.asyncio
async def test_dataset_thread_storage_skips_realtime_audio_chunks_by_default() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentRealtimeAudioChunk(
            type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
            thread_id="dataset://threads/demo",
            message_id="audio-input-1",
            data=b"\xf7\x00\x01",
        )
    )
    storage.push_message(
        message=AgentRealtimeAudioCommit(
            type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
            thread_id="dataset://threads/demo",
            message_id="audio-commit-1",
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 1
    assert _row_data(thread_rows[0])["type"] == AGENT_MESSAGE_REALTIME_AUDIO_COMMIT
    assert thread_rows[0]["attachment"] is None


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


@pytest.mark.asyncio
async def test_dataset_thread_storage_schema_compresses_json_data() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")

    await storage.start()
    await storage.stop()

    schema = room.datasets.create_calls[0]["schema"]
    assert schema.field("data").metadata == {b"lance-encoding:compression": b"zstd"}


@pytest.mark.asyncio
async def test_dataset_thread_storage_opens_existing_table_without_recreating() -> None:
    room = _FakeRoom()
    room.datasets.raise_on_existing_create = True
    room.datasets.schemas[(("threads",), "demo")] = pa.schema(
        [
            pa.field("turn_id", pa.string()),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("type", pa.string()),
            pa.field("sequence", pa.int64(), nullable=False),
            pa.field("timestamp", pa.timestamp("us"), nullable=False),
            pa.field("data", pa.json_(pa.large_string()), nullable=False),
        ]
    )
    room.datasets.rows[(("threads",), "demo")] = []
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")

    await storage.start()
    await storage.stop()

    assert room.datasets.create_calls == []


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
            turn_id="turn-accepted",
            source_message_id="accepted",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 3
    unaccepted_data = _row_data(rows[0])
    accepted_input_data = _row_data(rows[1])
    accepted_data = _row_data(rows[2])
    assert unaccepted_data["type"] == AGENT_MESSAGE_TURN_START
    assert unaccepted_data["content"] == [{"type": "text", "text": "do not save"}]
    assert unaccepted_data["sender_name"] == "caller"
    assert rows[0]["item_id"] == "unaccepted"
    assert rows[0]["type"] == AGENT_MESSAGE_TURN_START
    assert accepted_input_data["type"] == AGENT_MESSAGE_TURN_START
    assert accepted_input_data["turn_id"] == "turn-accepted"
    assert accepted_input_data["content"] == [{"type": "text", "text": "save this"}]
    assert accepted_input_data["sender_name"] == "caller"
    assert rows[1]["item_id"] == "accepted"
    assert rows[1]["turn_id"] == "turn-accepted"
    assert rows[1]["type"] == AGENT_MESSAGE_TURN_START
    assert room.datasets.merge_calls[0]["on"] == "sequence"
    assert room.datasets.merge_calls[0]["records"][0]["sequence"] == rows[1]["sequence"]
    assert accepted_data["type"] == AGENT_EVENT_TURN_START_ACCEPTED
    assert accepted_data["turn_id"] == "turn-accepted"
    assert accepted_data["source_message_id"] == "accepted"
    assert accepted_data["content"] == []
    assert rows[2]["type"] == AGENT_EVENT_TURN_START_ACCEPTED
    assert "sender_name" not in accepted_data or accepted_data["sender_name"] is None


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

    assert len(room.datasets.optimize_calls) == 2
    optimize_call = room.datasets.optimize_calls[0]
    assert optimize_call["table"] == "demo"
    assert optimize_call["namespace"] == ["threads"]
    assert optimize_call["config"].compact_files is True
    assert optimize_call["config"].optimize_indices is False
    assert optimize_call["config"].cleanup_old_versions is False


@pytest.mark.asyncio
async def test_dataset_thread_storage_persists_usage_updates() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id="dataset://threads/demo",
            message_id="usage-1",
            turn_id="turn-1",
            usage={"input_tokens": 120.0, "output_tokens": 30.0},
            context_window=AgentContextWindowUsage(
                used_tokens=480,
                total_tokens=128000,
                compaction_mode="auto",
                compaction_threshold=64000,
            ),
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 1
    data = _row_data(rows[0])
    assert data["type"] == AGENT_EVENT_USAGE_UPDATED
    assert data["thread_id"] == "dataset://threads/demo"
    assert data["turn_id"] == "turn-1"
    assert data["usage"] == {
        "input_tokens": 120.0,
        "output_tokens": 30.0,
    }
    assert data["context_window"] == {
        "used_tokens": 480,
        "total_tokens": 128000,
        "compaction_mode": "auto",
        "compaction_threshold": 64000,
    }


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_usage_updates() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id="dataset://threads/demo",
            message_id="usage-1",
            turn_id="turn-1",
            usage={"gpt-test.input_tokens": 120.0, "gpt-test.output_tokens": 30.0},
            context_window=AgentContextWindowUsage(
                used_tokens=120,
                total_tokens=128000,
            ),
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())

    assert context.last_usage == SessionUsage(
        model="gpt-test",
        usage={"gpt-test.input_tokens": 120.0, "gpt-test.output_tokens": 30.0},
        context_window_used=120,
        context_window_size=128000,
    )


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_compacted_context_messages() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="old-answer",
            text="old answer",
        )
    )
    storage.push_message(
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="old-answer",
        )
    )
    compacted_messages = [
        {
            "id": "compaction-1",
            "type": "compaction",
            "encrypted_content": "opaque",
        }
    ]
    storage.push_message(
        message=AgentContextCompacted(
            type=AGENT_EVENT_CONTEXT_COMPACTED,
            thread_id="dataset://threads/demo",
            checkpoint_id="compaction-1",
            path="dataset://threads/demo",
            through_sequence=1,
            messages=compacted_messages,
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())

    assert context.messages == compacted_messages


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
            provider="openai",
            model="gpt-test",
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
    assert len(rows) == 2
    data = _row_data(rows[0])
    assert data["type"] == AGENT_EVENT_TEXT_CONTENT_DELTA
    assert rows[0]["item_id"] == "text-1"
    assert data["text"] == "partial answer"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-test"
    interrupted = _row_data(rows[1])
    assert interrupted["type"] == AGENT_EVENT_TURN_INTERRUPTED


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_text_sender_name() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="answer",
            sender_name="chatbot",
        )
    )
    storage.push_message(
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    await storage.stop()

    for row in room.datasets.rows[(("threads",), "demo")]:
        timestamp = row.get("timestamp")
        if isinstance(timestamp, str):
            row["timestamp"] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        data = row.get("data")
        if isinstance(data, dict):
            row["data"] = json.dumps(data)

    restored = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await restored.start()
    try:
        await restored.wait_until_ready()
        text_delta = next(
            message
            for message in restored.agent_messages()
            if isinstance(message, AgentTextContentDelta)
        )
    finally:
        await restored.stop()

    assert text_delta.sender_name == "chatbot"


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_text_phase() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="checking",
            phase="commentary",
        )
    )
    storage.push_message(
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            phase="commentary",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    data = _row_data(rows[0])
    assert data["phase"] == "commentary"

    for row in rows:
        timestamp = row.get("timestamp")
        if isinstance(timestamp, str):
            row["timestamp"] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        data = row.get("data")
        if isinstance(data, dict):
            row["data"] = json.dumps(data)

    restored = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await restored.start()
    try:
        await restored.wait_until_ready()
        context = AgentSessionContext(system_role=None)
        restored.restore_session_context(
            context=context,
            llm_adapter=_test_llm_adapter(),
        )
    finally:
        await restored.stop()

    assert context.messages == [
        {"role": "assistant", "content": "checking", "phase": "commentary"}
    ]


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_streamed_text_as_single_message() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()
    storage.push_message(
        message=AgentTextContentStarted(
            type=AGENT_EVENT_TEXT_CONTENT_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            provider="openai",
            model="gpt-test",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text=" there",
        )
    )
    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi there",
        )
    )
    storage.push_message(
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())

    assert context.messages == [{"role": "assistant", "content": "Hi there"}]
    assert "agent_events" not in context.metadata


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
    assert len(rows) == 2
    data = _row_data(rows[0])
    assert data["type"] == AGENT_EVENT_TEXT_CONTENT_DELTA
    assert data["text"] == "complete enough"
    ended = _row_data(rows[1])
    assert ended["type"] == AGENT_EVENT_TURN_ENDED


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_previous_turn_before_accepted_steer() -> (
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
            text="answer before steer",
        )
    )
    storage.push_message(
        message=TurnSteer(
            type=AGENT_MESSAGE_TURN_STEER,
            thread_id="dataset://threads/demo",
            message_id="steer-1",
            turn_id="turn-1",
            content=[{"type": "text", "text": "add this"}],
        ),
        sender=_participant("caller"),
    )
    storage.push_message(
        message=TurnSteerAccepted(
            type=AGENT_EVENT_TURN_STEER_ACCEPTED,
            thread_id="dataset://threads/demo",
            source_message_id="steer-1",
            turn_id="turn-1",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert len(rows) == 3
    first = _row_data(rows[0])
    second = _row_data(rows[1])
    third = _row_data(rows[2])
    assert first["type"] == AGENT_EVENT_TEXT_CONTENT_DELTA
    assert first["text"] == "answer before steer"
    assert second["type"] == AGENT_MESSAGE_TURN_STEER
    assert second["content"] == [{"type": "text", "text": "add this"}]
    assert second["sender_name"] == "caller"
    assert third["type"] == AGENT_EVENT_TURN_STEER_ACCEPTED
    assert third["content"] == []


@pytest.mark.asyncio
async def test_dataset_thread_storage_does_not_persist_empty_started_text() -> None:
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
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    await storage.stop()

    assert room.datasets.rows[(("threads",), "demo")] == []


@pytest.mark.asyncio
async def test_dataset_thread_storage_does_not_persist_whitespace_only_text() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
            text=" \n\t",
        )
    )
    storage.push_message(
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    await storage.stop()

    assert room.datasets.rows[(("threads",), "demo")] == []


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
            namespace="openai.responses",
            call_id="call-started",
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
    assert len(rows) == 2
    data = _row_data(rows[0])
    assert rows[0]["item_id"] == "started-tool"
    assert data["type"] == AGENT_EVENT_TOOL_CALL_STARTED
    assert data["namespace"] == "openai.responses"
    assert data["call_id"] == "call-started"
    assert data["toolkit"] == "shell"
    assert data["tool"] == "exec"
    ended = _row_data(rows[1])
    assert ended["type"] == AGENT_EVENT_TURN_ENDED


@pytest.mark.asyncio
async def test_dataset_thread_storage_does_not_write_tool_argument_deltas() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="started-tool",
            namespace="openai.responses",
            call_id="call-started",
            toolkit="storage",
            tool="write_file",
            arguments={"path": "src/app.py"},
        )
    )
    storage.push_message(
        message=AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="started-tool",
            delta='{"content":"partial',
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
    assert [row["type"] for row in rows] == [
        AGENT_EVENT_TOOL_CALL_STARTED,
        AGENT_EVENT_TURN_ENDED,
    ]


@pytest.mark.asyncio
async def test_dataset_thread_storage_can_persist_verbose_tool_argument_deltas() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(
        room=room,
        path="dataset://threads/demo",
        persist_deltas=True,
    )
    await storage.start()

    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="started-tool",
            namespace="openai.responses",
            call_id="call-started",
            toolkit="storage",
            tool="write_file",
            arguments={"path": "src/app.py"},
        )
    )
    storage.push_message(
        message=AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="started-tool",
            delta='{"content":"partial',
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
    row_types = [row["type"] for row in rows]
    assert AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA in row_types
    delta_row = next(
        row for row in rows if row["type"] == AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA
    )
    assert delta_row["turn_id"] == "turn-1"
    assert AGENT_EVENT_TOOL_CALL_STARTED in row_types
    assert row_types[-1] == AGENT_EVENT_TURN_ENDED
    delta = _row_data(
        next(
            row for row in rows if row["type"] == AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA
        )
    )
    assert delta["delta"] == '{"content":"partial'
    assert "tool" not in delta
    assert "toolkit" not in delta


@pytest.mark.asyncio
async def test_dataset_thread_storage_coalesces_apply_patch_argument_deltas() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallPending(
            type=AGENT_EVENT_TOOL_CALL_PENDING,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="patch-1",
            namespace="openai.responses",
            toolkit="openai",
            tool="apply_patch",
            arguments={},
        )
    )
    storage.push_message(
        message=AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="patch-1",
            delta="*** Begin Patch\n*** Update File: app.ts\n@@\n-old\n",
        )
    )
    storage.push_message(
        message=AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="patch-1",
            delta="+new\n*** End Patch\n",
        )
    )
    storage.push_message(
        message=AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="patch-1",
        )
    )
    await storage.stop()

    rows = room.datasets.rows[(("threads",), "demo")]
    assert [row["type"] for row in rows] == [
        AGENT_EVENT_TOOL_CALL_STARTED,
        AGENT_EVENT_TOOL_CALL_ENDED,
    ]
    data = _row_data(rows[0])
    assert data["arguments"] == {
        "patch": (
            "*** Begin Patch\n*** Update File: app.ts\n@@\n-old\n+new\n*** End Patch"
        )
    }


@pytest.mark.asyncio
async def test_dataset_thread_storage_does_not_persist_binary_image_generation_result() -> (
    None
):
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

    assert ((), "images") not in room.datasets.rows

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 2
    assert thread_rows[0]["item_id"] == "image-tool"
    data = _row_data(thread_rows[0])
    assert data["type"] == AGENT_EVENT_TOOL_CALL_STARTED
    ended = _row_data(thread_rows[1])
    assert ended["type"] == "meshagent.agent.tool_call.ended"

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())
    assert context.messages[0]["role"] == "assistant"
    content = context.messages[0]["content"][0]
    assert content["type"] == "tool_call"
    assert content["item_id"] == "image-tool"
    assert content["toolkit"] == "openai"
    assert content["tool"] == "image_generation"
    assert content["arguments"] == {"size": "1024x768", "output_format": "png"}
    assert content["result"]["type"] == "binary"


@pytest.mark.asyncio
async def test_dataset_thread_storage_persists_image_generation_url_result() -> None:
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
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            result=TextContent(text="https://example.test/generated.png"),
        )
    )
    await storage.stop()

    assert ((), "images") not in room.datasets.rows
    thread_rows = room.datasets.rows[(("threads",), "demo")]
    data = _row_data(thread_rows[0])
    assert data["type"] == AGENT_EVENT_IMAGE_GENERATION_COMPLETED
    assert data["images"][0]["uri"] == "https://example.test/generated.png"
    assert data["images"][0]["width"] == 512
    assert data["images"][0]["height"] == 512


@pytest.mark.asyncio
async def test_dataset_thread_storage_does_not_overwrite_typed_image_generation_result() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            namespace="openai.responses",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            images=[
                AgentGeneratedImage(
                    uri="dataset://images?id=image-1",
                    mime_type="image/png",
                    width=512,
                    height=512,
                    status="completed",
                )
            ],
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

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 2
    assert thread_rows[0]["item_id"] == "image-tool"
    data = _row_data(thread_rows[0])
    assert data["type"] == AGENT_EVENT_IMAGE_GENERATION_COMPLETED
    assert data["images"][0]["uri"] == "dataset://images?id=image-1"
    ended = _row_data(thread_rows[1])
    assert ended["type"] == AGENT_EVENT_TURN_ENDED


@pytest.mark.asyncio
async def test_dataset_thread_storage_async_restore_hydrates_image_dataset_uris() -> (
    None
):
    room = _FakeRoom()
    saved_image = await ImagesDataset(room.datasets).save(
        image_id="image-1",
        data=b"fake image bytes",
        mime_type="image/png",
        created_by="chatbot",
        created_at="2026-05-09T18:00:00Z",
    )
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            images=[
                AgentGeneratedImage(
                    uri="dataset://images?id=image-1",
                    mime_type="image/png",
                    width=512,
                    height=512,
                    status="completed",
                )
            ],
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    await storage.restore_session_context_async(
        context=context,
        llm_adapter=_test_llm_adapter(),
    )

    content = context.messages[0]["content"][0]
    assert content["type"] == "image_generation"
    assert content["turn_id"] == "turn-1"
    assert content["images"][0]["uri"] == (
        "data:image/png;base64,ZmFrZSBpbWFnZSBieXRlcw=="
    )
    assert content["images"][0]["created_at"] == saved_image.created_at
    assert content["images"][0]["created_by"] == saved_image.created_by

    row = _row_data(room.datasets.rows[(("threads",), "demo")][0])
    assert room.datasets.rows[(("threads",), "demo")][0]["turn_id"] == "turn-1"
    assert row["turn_id"] == "turn-1"
    assert row["images"][0]["uri"] == "dataset://images?id=image-1"


@pytest.mark.asyncio
async def test_dataset_thread_storage_skips_nonterminal_image_generation_events() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentImageGenerationStarted(
            type="meshagent.agent.image_generation.started",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            images=[AgentGeneratedImage(uri="dataset://images?id=image-1")],
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 1
    assert thread_rows[0]["item_id"] == "image-started"
    completed = _row_data(thread_rows[0])
    assert completed["type"] == AGENT_EVENT_IMAGE_GENERATION_COMPLETED


@pytest.mark.asyncio
async def test_dataset_thread_storage_fills_empty_image_completed_from_partial() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentImageGenerationPartial(
            type="meshagent.agent.image_generation.partial",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            image=AgentGeneratedImage(
                uri="dataset://images?id=image-1",
                mime_type="image/png",
                status="in_progress",
            ),
            partial_index=0,
        )
    )
    storage.push_message(
        message=AgentImageGenerationCompleted(
            type="meshagent.agent.image_generation.completed",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            images=[],
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 1
    completed = _row_data(thread_rows[0])
    assert completed["type"] == AGENT_EVENT_IMAGE_GENERATION_COMPLETED
    assert completed["images"][0]["uri"] == "dataset://images?id=image-1"
    assert completed["images"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_image_failed_on_cancellation() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentImageGenerationStarted(
            type="meshagent.agent.image_generation.started",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentImageGenerationPartial(
            type="meshagent.agent.image_generation.partial",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            image=AgentGeneratedImage(
                uri="data:image/png;base64,first",
                mime_type="image/png",
                status="in_progress",
            ),
            partial_index=0,
        )
    )
    storage.push_message(
        message=AgentImageGenerationPartial(
            type="meshagent.agent.image_generation.partial",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            image=AgentGeneratedImage(
                uri="data:image/png;base64,second",
                mime_type="image/png",
                status="in_progress",
            ),
            partial_index=1,
        )
    )
    storage.push_message(
        message=TurnInterrupted(
            type="meshagent.agent.turn.interrupted",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            source_message_id="interrupt-1",
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 2
    assert thread_rows[0]["item_id"] == "image-started"
    failed = _row_data(thread_rows[0])
    assert failed["type"] == "meshagent.agent.image_generation.failed"
    assert failed["error"]["code"] == "cancelled"
    interrupted = _row_data(thread_rows[1])
    assert interrupted["type"] == AGENT_EVENT_TURN_INTERRUPTED


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_image_failed_on_cancelled_tool_end() -> (
    None
):
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            namespace="openai.responses",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentImageGenerationPartial(
            type="meshagent.agent.image_generation.partial",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            image=AgentGeneratedImage(
                uri="data:image/png;base64,partial",
                mime_type="image/png",
                status="in_progress",
            ),
            partial_index=0,
        )
    )
    storage.push_message(
        message=AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-tool",
            namespace="openai.responses",
            call_id="call-image",
            result=None,
            error=AgentError(message="cancelled", code=None),
        )
    )
    await storage.stop()

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 1
    failed = _row_data(thread_rows[0])
    assert failed["type"] == "meshagent.agent.image_generation.failed"


@pytest.mark.asyncio
async def test_dataset_thread_storage_flushes_image_completed_from_partial() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=AgentImageGenerationStarted(
            type="meshagent.agent.image_generation.started",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
        )
    )
    storage.push_message(
        message=AgentImageGenerationPartial(
            type="meshagent.agent.image_generation.partial",
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="image-started",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            image=AgentGeneratedImage(
                uri="dataset://images?id=image-1",
                mime_type="image/png",
                width=512,
                height=512,
                status="in_progress",
            ),
            partial_index=0,
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

    thread_rows = room.datasets.rows[(("threads",), "demo")]
    assert len(thread_rows) == 2
    assert thread_rows[0]["turn_id"] == "turn-1"
    completed = _row_data(thread_rows[0])
    assert completed["type"] == AGENT_EVENT_IMAGE_GENERATION_COMPLETED
    assert completed["turn_id"] == "turn-1"
    assert completed["images"][0]["uri"] == "dataset://images?id=image-1"
    assert completed["images"][0]["status"] == "completed"
    ended = _row_data(thread_rows[1])
    assert ended["type"] == AGENT_EVENT_TURN_ENDED


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
                    "type": AGENT_EVENT_TEXT_CONTENT_DELTA,
                    "thread_id": "dataset://threads/demo",
                    "message_id": "assistant-message-1",
                    "turn_id": "turn-1",
                    "item_id": "assistant-message-1",
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
                    "type": AGENT_MESSAGE_TURN_START,
                    "thread_id": "dataset://threads/demo",
                    "message_id": "first",
                    "sender_name": "caller",
                    "content": [{"type": "text", "text": "question"}],
                }
            ),
        },
    ]

    await storage.start()
    await storage.wait_until_ready()
    restored_messages = storage.agent_messages()
    assert restored_messages[0].created_at == "2026-03-10T00:00:00Z"
    assert restored_messages[1].created_at == "2026-03-11T00:00:00Z"
    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())
    await storage.stop()

    assert context.messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.asyncio
async def test_dataset_thread_storage_restores_context_with_llm_reader() -> None:
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path="dataset://threads/demo")
    await storage.start()

    storage.push_message(
        message=TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="dataset://threads/demo",
            message_id="user-1",
            content=[AgentTextContent(type="text", text="question")],
        ),
        sender=_participant("caller"),
    )
    storage.push_message(
        message=TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="dataset://threads/demo",
            source_message_id="user-1",
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
        message=AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    storage.push_message(
        message=AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            toolkit="openai",
            tool="web_search",
            arguments={"query": "meshagent"},
        )
    )
    storage.push_message(
        message=AgentToolCallLogDelta(
            type=AGENT_EVENT_TOOL_CALL_LOG_DELTA,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            lines=[AgentToolCallLogLine(source="stdout", text="searching\n")],
        )
    )
    storage.push_message(
        message=AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="dataset://threads/demo",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            result=JsonContent(json={"results": [{"title": "MeshAgent"}]}),
        )
    )
    storage.push_message(
        message=AgentThreadEvent(
            type=AGENT_EVENT_THREAD_EVENT,
            thread_id="dataset://threads/demo",
            event={"kind": "shell", "cmd": "pwd"},
        )
    )
    storage.push_message(
        message=AgentContextCompacted(
            type=AGENT_EVENT_CONTEXT_COMPACTED,
            thread_id="dataset://threads/demo",
            checkpoint_id="checkpoint-1",
            path="dataset://threads/demo",
            through_sequence=4,
            created_at="2026-05-05T00:00:00Z",
        )
    )
    await storage.stop()

    context = AgentSessionContext(system_role=None)
    storage.restore_session_context(context=context, llm_adapter=_test_llm_adapter())

    assert context.messages[0] == {"role": "user", "content": "question"}
    assert context.messages[1] == {"role": "assistant", "content": "answer"}
    assert context.messages[2]["content"][0] == {
        "type": "tool_call",
        "item_id": "tool-1",
        "namespace": "openai.responses",
        "call_id": "call-1",
        "toolkit": "openai",
        "tool": "web_search",
        "arguments": {"query": "meshagent"},
        "result": {"type": "json", "json": {"results": [{"title": "MeshAgent"}]}},
        "error": None,
        "logs": [{"source": "stdout", "text": "searching\n"}],
    }
    assert context.messages[3] == {
        "role": "assistant",
        "content": [{"type": "event", "event": {"kind": "shell", "cmd": "pwd"}}],
    }
    assert context.metadata["last_compaction"] == {
        "checkpoint_id": "checkpoint-1",
        "path": "dataset://threads/demo",
        "through_sequence": 4,
        "created_at": "2026-05-05T00:00:00Z",
    }
    assert "agent_events" not in context.metadata


def test_dataset_thread_storage_default_name_uses_new_chat_for_uuid_paths() -> None:
    assert (
        DatasetThreadStorage.default_thread_name(
            path="dataset://threads/12345678-1234-4678-9234-123456789abc"
        )
        == "New Chat"
    )
    assert (
        DatasetThreadStorage.default_thread_name(
            path="dataset://threads/support-thread"
        )
        == "Support Thread"
    )
    assert (
        DatasetThreadStorage.default_thread_name(
            path="dataset://threads/support-thread",
            name="Customer Support",
        )
        == "Customer Support"
    )
