import pytest

from meshagent.agents.responses_thread_adapter import (
    ResponsesThreadAdapter,
    _extract_image_dimensions,
)


class _FakeParticipant:
    def __init__(self, *, name: str):
        self._name = name

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant(name="assistant")


def test_extract_image_dimensions_prefers_explicit_fields_then_size():
    width, height = _extract_image_dimensions(
        item={"width": 1536, "height": 1024},
        event={"size": "1024x1024"},
    )
    assert width == 1536
    assert height == 1024

    width, height = _extract_image_dimensions(
        item={"size": "512x768"},
        event=None,
    )
    assert width == 512
    assert height == 768


@pytest.mark.asyncio
async def test_emit_image_status_event_updates_image_element_state():
    adapter = object.__new__(ResponsesThreadAdapter)
    adapter._room = _FakeRoom()

    events: list[dict] = []
    writes: list[dict] = []

    async def _fake_handle_custom_event(*, messages, event):
        del messages
        events.append(event)

    def _fake_write_image(**kwargs):
        writes.append(kwargs)
        return kwargs.get("message_id", "")

    adapter.handle_custom_event = _fake_handle_custom_event  # type: ignore[assignment]
    adapter.write_image = _fake_write_image  # type: ignore[assignment]

    await adapter._emit_image_status_event(
        messages=object(),
        item_id="img-item-1",
        state="in_progress",
        headline="Generating image",
        width=1024,
        height=768,
    )

    assert len(events) == 1
    assert events[0]["state"] == "in_progress"
    assert events[0]["item_id"] == "img-item-1"

    assert len(writes) == 1
    assert writes[0]["message_id"] == "img-item-1"
    assert writes[0]["status"] == "generating"
    assert writes[0]["status_detail"] == "Generating image"
    assert writes[0]["width"] == 1024
    assert writes[0]["height"] == 768
