import pytest
import base64
from typing import Optional
import asyncio

import meshagent.agents.thread_adapter as thread_adapter_module
from meshagent.agents.thread_adapter import ThreadAdapter


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

    def get_state(self, vector: bytes | None = None) -> bytes:
        del vector
        return self._state


class _BaseStopThreadAdapter(ThreadAdapter):
    async def handle_custom_event(self, *, event) -> None:
        del event

    async def _process_llm_events(self) -> None:
        return None


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


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: Optional[dict] = None) -> None:
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_FakeElement"] = []

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value

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
