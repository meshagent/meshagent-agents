import pytest
import base64
from typing import Optional
import asyncio
import logging
import re

import meshagent.agents.thread_adapter as thread_adapter_module
from meshagent.api.messaging import JsonContent, TextContent
from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.tools import ToolContext


class _FakeThreadAdapter(ThreadAdapter):
    def __init__(self) -> None:
        super().__init__(room=object(), path="/threads/test")  # type: ignore[arg-type]
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def handle_custom_event(self, *, event) -> None:
        del event

    async def _process_llm_events(self) -> None:
        return None


@pytest.mark.asyncio
async def test_thread_adapter_async_manager_calls_start_and_stop() -> None:
    adapter = _FakeThreadAdapter()

    async with adapter:
        assert adapter.started == 1
        assert adapter.stopped == 0

    assert adapter.started == 1
    assert adapter.stopped == 1


class _FakeSync:
    def __init__(self) -> None:
        self.sync_calls: list[dict] = []
        self.close_calls: list[str] = []

    async def sync(self, *, path: str, data: bytes) -> None:
        self.sync_calls.append({"path": path, "data": data})

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)


class _FakeRoom:
    def __init__(self, *, is_closed: bool = False) -> None:
        self.sync = _FakeSync()
        self.is_closed = is_closed


class _FakeMeshDocument:
    def __init__(self, *, state: Optional[bytes] = None) -> None:
        self._state = state if state is not None else b""
        self.root = _FakeElement(tag_name="thread")

    def get_state(self, vector: bytes | None = None) -> bytes:
        del vector
        return self._state


class _FailingStateMeshDocument(_FakeMeshDocument):
    def get_state(self, vector: bytes | None = None) -> bytes:
        del vector
        raise RuntimeError("state failed")


class _BaseStopThreadAdapter(ThreadAdapter):
    async def handle_custom_event(self, *, event) -> None:
        del event

    async def _process_llm_events(self) -> None:
        return None


class _FakeOpenSync(_FakeSync):
    def __init__(self, document: _FakeMeshDocument) -> None:
        super().__init__()
        self.document = document
        self.open_calls: list[dict] = []

    async def open(self, *, path: str, schema) -> _FakeMeshDocument:
        self.open_calls.append({"path": path, "schema": schema})
        return self.document


class _FakeStartRoom(_FakeRoom):
    def __init__(self, document: _FakeMeshDocument) -> None:
        super().__init__()
        self.sync = _FakeOpenSync(document)


@pytest.mark.asyncio
async def test_thread_adapter_start_opens_sync_document_and_schedules_processor() -> (
    None
):
    document = _FakeMeshDocument()
    room = _FakeStartRoom(document)
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")

    await adapter.start()
    await asyncio.sleep(0)

    assert room.sync.open_calls == [
        {"path": "/threads/test", "schema": thread_adapter_module.thread_schema}
    ]
    assert adapter.thread is document
    assert [child.tag_name for child in document.root.get_children()] == [
        "members",
        "messages",
    ]
    assert adapter._processor_task is not None
    assert adapter._processor_task.done()


@pytest.mark.asyncio
async def test_thread_adapter_stop_flushes_state_before_close(monkeypatch) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _FakeRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FakeMeshDocument(state=b"state")

    await adapter.stop()

    assert room.sync.sync_calls == [
        {"path": "/threads/test", "data": base64.standard_b64encode(b"state")}
    ]
    assert room.sync.close_calls == ["/threads/test"]


@pytest.mark.asyncio
async def test_thread_adapter_stop_drains_pending_events_before_queue_shutdown(
    monkeypatch,
) -> None:
    original_sleep = asyncio.sleep
    sleep_calls: list[float] = []

    async def _yielding_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await original_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _yielding_sleep)

    room = _FakeRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FakeMeshDocument()
    adapter._llm_messages.put_nowait({"type": "event"})
    consumed: list[dict] = []

    async def _processor() -> None:
        consumed.append(await adapter._llm_messages.get())

    adapter._processor_task = asyncio.create_task(_processor())

    await adapter.stop()

    assert consumed == [{"type": "event"}]
    assert sleep_calls == [0.01, 3]
    assert adapter._processor_task is None
    assert adapter.thread is None
    assert room.sync.close_calls == ["/threads/test"]


@pytest.mark.asyncio
async def test_thread_adapter_stop_skips_flush_when_room_is_closed(monkeypatch) -> None:
    sleep_calls: list[float] = []

    async def _fast_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _FakeRoom(is_closed=True)
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FakeMeshDocument(state=b"state")

    await adapter.stop()

    assert room.sync.sync_calls == []
    assert room.sync.close_calls == ["/threads/test"]
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_thread_adapter_stop_times_out_stalled_sync_close(monkeypatch) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        thread_adapter_module,
        "_THREAD_SYNC_CLOSE_TIMEOUT_SEC",
        0.01,
    )

    class _HangingSync(_FakeSync):
        async def close(self, *, path: str) -> None:
            self.close_calls.append(path)
            await asyncio.Event().wait()

    class _HangingRoom:
        def __init__(self) -> None:
            self.sync = _HangingSync()
            self.is_closed = False

    room = _HangingRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FakeMeshDocument(state=b"state")

    await adapter.stop()

    assert room.sync.sync_calls == [
        {"path": "/threads/test", "data": base64.standard_b64encode(b"state")}
    ]
    assert room.sync.close_calls == ["/threads/test"]
    assert adapter.thread is None


@pytest.mark.asyncio
async def test_thread_adapter_stop_warns_and_closes_when_state_collection_fails(
    monkeypatch,
    caplog,
) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _FakeRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FailingStateMeshDocument()

    with caplog.at_level(logging.WARNING, logger="thread_adapter"):
        await adapter.stop()

    assert room.sync.sync_calls == []
    assert room.sync.close_calls == ["/threads/test"]
    assert adapter.thread is None
    assert "unable to collect final thread state for /threads/test" in caplog.text


@pytest.mark.asyncio
async def test_thread_adapter_stop_warns_when_final_state_sync_fails(
    monkeypatch,
    caplog,
) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    class _FailingSync(_FakeSync):
        async def sync(self, *, path: str, data: bytes) -> None:
            self.sync_calls.append({"path": path, "data": data})
            raise RuntimeError("sync failed")

    class _FailingSyncRoom:
        def __init__(self) -> None:
            self.sync = _FailingSync()
            self.is_closed = False

    room = _FailingSyncRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")
    adapter._thread = _FakeMeshDocument(state=b"state")

    with caplog.at_level(logging.WARNING, logger="thread_adapter"):
        await adapter.stop()

    assert room.sync.sync_calls == [
        {"path": "/threads/test", "data": base64.standard_b64encode(b"state")}
    ]
    assert room.sync.close_calls == ["/threads/test"]
    assert adapter.thread is None
    assert "unable to flush final thread state for /threads/test" in caplog.text


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: Optional[dict] = None) -> None:
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_FakeElement"] = []

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value

    def __getitem__(self, name: str):
        return self._attributes[name]

    def get_children(self) -> list["_FakeElement"]:
        return [*self._children]

    def get_children_by_tag_name(self, tag_name: str) -> list["_FakeElement"]:
        return [child for child in self._children if child.tag_name == tag_name]

    def append_child(
        self, *, tag_name: str, attributes: Optional[dict] = None
    ) -> "_FakeElement":
        child = _FakeElement(tag_name=tag_name, attributes=attributes)
        self._children.append(child)
        return child

    def delete(self) -> None:
        self._children.clear()

    def grep(
        self,
        pattern: str,
        *,
        ignore_case: bool,
        before: int,
        after: int,
    ) -> list["_FakeElement"]:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        matches = [
            index
            for index, child in enumerate(self._children)
            if regex.search(
                " ".join(str(value) for value in child._attributes.values())
            )
        ]
        selected_indexes: set[int] = set()
        for index in matches:
            start = max(index - before, 0)
            end = min(index + after + 1, len(self._children))
            selected_indexes.update(range(start, end))
        return [
            child
            for index, child in enumerate(self._children)
            if index in selected_indexes
        ]


class _FakeThreadDocumentForWrite:
    def __init__(self) -> None:
        self.root = _FakeElement(tag_name="thread")
        self.messages = self.root.append_child(tag_name="messages")


class _FakeLocalParticipant:
    def get_attribute(self, name: str):
        if name == "name":
            return "assistant"
        return None


class _FakeWriteRoom:
    def __init__(self) -> None:
        self.local_participant = _FakeLocalParticipant()


class _FakeAgentSessionContext:
    def __init__(self) -> None:
        self.assistant_messages: list[str] = []
        self.user_messages: list[str] = []

    def append_assistant_message(self, message: str) -> None:
        self.assistant_messages.append(message)

    def append_user_message(self, message: str) -> None:
        self.user_messages.append(message)


@pytest.mark.asyncio
async def test_thread_adapter_make_toolkit_invokes_registered_tools_against_thread(
    monkeypatch,
) -> None:
    monkeypatch.setattr(thread_adapter_module, "Element", _FakeElement)
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]
    thread = _FakeThreadDocumentForWrite()
    thread.messages.append_child(
        tag_name="message",
        attributes={
            "text": "hello",
            "created_at": "2026-05-01T00:00:00Z",
            "author_name": "Alice",
        },
    )
    thread.messages.append_child(
        tag_name="ignored",
        attributes={
            "text": "this is not a message",
            "created_at": "2026-05-01T00:01:00Z",
            "author_name": "System",
        },
    )
    thread.messages.append_child(
        tag_name="message",
        attributes={
            "text": "needle",
            "created_at": "2026-05-01T00:02:00Z",
            "author_name": "Bob",
        },
    )
    adapter._thread = thread  # type: ignore[assignment]
    toolkit = adapter.make_toolkit()
    context = ToolContext(caller=object())  # type: ignore[arg-type]

    range_result = await toolkit.execute(
        context=context,
        name="get_message_range",
        input=JsonContent(json={"start": 0, "end": 10}),
    )
    assert isinstance(range_result, TextContent)
    assert range_result.text == (
        "matching messages:\n"
        "Alice said at 2026-05-01T00:00:00Z: hello\n"
        "Bob said at 2026-05-01T00:02:00Z: needle"
    )

    count_result = await toolkit.execute(
        context=context,
        name="count_current_thread_messages",
        input=JsonContent(
            json={
                "pattern": "unused",
                "ignore_case": False,
                "messages_before": 0,
                "messages_after": 0,
            }
        ),
    )
    assert isinstance(count_result, TextContent)
    assert count_result.text == "2"

    grep_result = await toolkit.execute(
        context=context,
        name="grep_current_thread",
        input=JsonContent(
            json={
                "pattern": "NEEDLE",
                "ignore_case": True,
                "messages_before": 2,
                "messages_after": 0,
            }
        ),
    )
    assert isinstance(grep_result, TextContent)
    assert grep_result.text == (
        "matching messages:\n"
        "Alice said at 2026-05-01T00:00:00Z: hello\n"
        "Bob said at 2026-05-01T00:02:00Z: needle"
    )


def test_thread_adapter_member_and_text_message_mutate_thread_elements(
    monkeypatch,
) -> None:
    monkeypatch.setattr(thread_adapter_module, "Element", _FakeElement)
    monkeypatch.setattr(thread_adapter_module, "Participant", _FakeLocalParticipant)
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]
    thread = _FakeThreadDocumentForWrite()
    adapter._thread = thread  # type: ignore[assignment]

    adapter.ensure_member(participant=" Alice ")
    adapter.ensure_member(participant="Alice")
    adapter.ensure_member(participant=" ")
    adapter.write_text_message(
        text="hello",
        participant=room.local_participant,
        message_id=" message-1 ",
        turn_id=" turn-1 ",
        attachments=[{"path": "/files/a.txt"}, {"path": ""}, {"other": "ignored"}],
    )

    members = thread.root.get_children_by_tag_name("members")[0]
    assert [member.get_attribute("name") for member in members.get_children()] == [
        "Alice",
        "assistant",
    ]
    message = thread.messages.get_children()[0]
    assert message.get_attribute("text") == "hello"
    assert message.get_attribute("author_name") == "assistant"
    assert message.get_attribute("role") == "agent"
    assert message.get_attribute("id") == "message-1"
    assert message.get_attribute("turn_id") == "turn-1"
    assert [child.get_attribute("path") for child in message.get_children()] == [
        "/files/a.txt"
    ]


def test_thread_adapter_append_messages_uses_concrete_thread_elements(
    monkeypatch,
) -> None:
    monkeypatch.setattr(thread_adapter_module, "Element", _FakeElement)
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(
        room=room,
        path="/threads/test",
        max_append_message_count=1,
    )  # type: ignore[arg-type]
    thread = _FakeThreadDocumentForWrite()
    thread.messages.append_child(
        tag_name="message",
        attributes={
            "text": "old",
            "created_at": "2026-05-01T00:00:00Z",
            "author_name": "Alice",
        },
    )
    recent = thread.messages.append_child(
        tag_name="message",
        attributes={
            "text": "recent",
            "created_at": "2026-05-01T00:02:00Z",
            "author_name": "Bob",
        },
    )
    recent.append_child(tag_name="file", attributes={"path": None})
    adapter._thread = thread  # type: ignore[assignment]
    context = _FakeAgentSessionContext()

    adapter.append_messages(context=context)  # type: ignore[arg-type]

    assert context.assistant_messages == [
        "there are more messages outside the current context window, the index of the first message loaded is 1",
        "the user attached a file at the path 'None'",
    ]
    assert context.user_messages == ["Bob said at 2026-05-01T00:02:00Z: recent"]


def test_write_image_marks_created_message_as_agent() -> None:
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]
    thread = _FakeThreadDocumentForWrite()
    adapter._thread = thread  # type: ignore[assignment]

    message_id = adapter.write_image(
        message_id="image-message",
        image_id="image-row",
        mime_type="image/png",
    )

    assert message_id == "image-message"
    message = thread.messages.get_children()[0]
    assert message.get_attribute("role") == "agent"


def test_write_image_backfills_role_on_existing_message() -> None:
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]
    thread = _FakeThreadDocumentForWrite()
    message = thread.messages.append_child(
        tag_name="message",
        attributes={
            "id": "image-message",
            "text": "",
            "created_at": "2026-05-01T00:00:00Z",
            "author_name": "assistant",
        },
    )
    adapter._thread = thread  # type: ignore[assignment]

    adapter.write_image(
        message_id="image-message",
        image_id="image-row",
        mime_type="image/png",
    )

    assert message.get_attribute("role") == "agent"


def test_write_image_without_messages_element_uses_python_index_error() -> None:
    room = _FakeWriteRoom()
    adapter = _BaseStopThreadAdapter(room=room, path="/threads/test")  # type: ignore[arg-type]
    adapter._thread = _FakeThreadDocumentForWrite()  # type: ignore[assignment]
    adapter._thread.root._children.clear()  # type: ignore[attr-defined]

    with pytest.raises(IndexError, match="list index out of range"):
        adapter.write_image(message_id="image-message")
