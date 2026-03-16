import pytest

from meshagent.agents.responses_thread_adapter import (
    _headline_for_response_event,
    ResponsesThreadAdapter,
    _extract_image_dimensions,
    response_event_to_agent_event,
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


def test_computer_call_events_are_not_classified_as_exec():
    event = {
        "type": "response.output_item.added",
        "item_id": "item_123",
        "item": {
            "id": "item_123",
            "type": "computer_call",
            "status": "in_progress",
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["headline"] == "Using computer"


def test_computer_call_click_event_headline_is_friendly_without_coordinates():
    event = {
        "type": "response.output_item.added",
        "item_id": "item_456",
        "item": {
            "id": "item_456",
            "type": "computer_call",
            "status": "in_progress",
            "action": {
                "type": "click",
                "x": 140,
                "y": 320,
            },
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["headline"] == "Clicking on page"
    assert normalized["details"] == []


def test_computer_call_scroll_event_uses_human_friendly_direction():
    event = {
        "type": "response.output_item.done",
        "item_id": "item_789",
        "item": {
            "id": "item_789",
            "type": "computer_call",
            "status": "completed",
            "action": {
                "type": "scroll",
                "scroll_x": 0,
                "scroll_y": -640,
            },
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["headline"] == "Scrolled page"
    assert normalized["details"] == ["Direction: up"]


def test_computer_startup_event_uses_startup_specific_headline():
    event = {
        "type": "response.output_item.done",
        "item_id": "startup_1",
        "name": "computer.startup",
        "state": "completed",
    }

    headline = _headline_for_response_event(event=event, kind="tool", state="completed")
    assert headline == "Computer ready"
