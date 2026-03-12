import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from meshagent.api import Element, RoomException

from .thread_adapter import ThreadAdapter, tracer

_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_SUPPORTED_EVENT_KINDS = {
    "exec",
    "tool",
    "web",
    "search",
    "diff",
    "image",
    "approval",
    "collab",
    "plan",
}
_COMPLETION_CHUNK_OBJECTS = {
    "chat.completion.chunk",
    "chat.completions.chunk",
}
_COMPLETION_OBJECTS = {
    "chat.completion",
    "chat.completions",
}


def _to_text(*, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_to_text(value=item) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text

        for key in ("content", "value", "message"):
            if key in value:
                nested = _to_text(value=value.get(key))
                if nested != "":
                    return nested

    return ""


def _event_object(*, event: dict) -> Optional[str]:
    obj = event.get("object")
    if isinstance(obj, str) and obj.strip() != "":
        return obj.strip().lower()

    evt_type = event.get("type")
    if isinstance(evt_type, str) and evt_type.strip() != "":
        return evt_type.strip().lower()

    return None


def _choices(*, event: dict) -> list[dict]:
    raw_choices = event.get("choices")
    if not isinstance(raw_choices, list):
        return []

    parsed: list[dict] = []
    for raw in raw_choices:
        if isinstance(raw, dict):
            parsed.append(raw)

    return parsed


def _is_completion_chunk_event(*, event: dict) -> bool:
    obj = _event_object(event=event)
    if obj in _COMPLETION_CHUNK_OBJECTS:
        return True
    if obj in _COMPLETION_OBJECTS:
        return False

    for choice in _choices(event=event):
        if isinstance(choice.get("delta"), dict):
            return True
    return False


def _is_completion_event(*, event: dict) -> bool:
    obj = _event_object(event=event)
    if obj in _COMPLETION_OBJECTS:
        return True
    if obj in _COMPLETION_CHUNK_OBJECTS:
        return False

    for choice in _choices(event=event):
        if isinstance(choice.get("message"), dict):
            return True
    return False


def _choice_index(*, choice: dict) -> int:
    index = choice.get("index")
    if isinstance(index, int) and index >= 0:
        return index
    return 0


def _choice_delta_text(*, choice: dict) -> str:
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return ""

    content = delta.get("content")
    return _to_text(value=content)


def _choice_message_text(*, choice: dict) -> str:
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = _to_text(value=content)
        if text != "":
            return text

    text = choice.get("text")
    if isinstance(text, str):
        return text

    return ""


def _choice_finish_reason(*, choice: dict) -> Optional[str]:
    reason = choice.get("finish_reason")
    if not isinstance(reason, str):
        return None

    normalized = reason.strip().lower()
    if normalized in ("", "none", "null"):
        return None

    return normalized


class CompletionsThreadAdapter(ThreadAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_events_by_key: dict[str, Element] = {}

    async def stop(self) -> None:
        await super().stop()
        self._active_events_by_key.clear()

    async def handle_custom_event(
        self,
        *,
        messages: Element,
        event: dict,
    ) -> None:
        event_type = event.get("type")
        if event_type not in ("agent.event", "codex.event"):
            return

        source = event.get("source")
        if not isinstance(source, str) or source.strip() == "":
            source = "agent" if event_type == "agent.event" else "codex"
        source = source.strip()

        name = event.get("name")
        if not isinstance(name, str) or name.strip() == "":
            name = event.get("event_type")
        if not isinstance(name, str) or name.strip() == "":
            name = event_type
        name = name.strip()

        kind = event.get("kind")
        if not isinstance(kind, str) or kind.strip() == "":
            return
        kind = kind.strip().lower()
        if kind not in _SUPPORTED_EVENT_KINDS:
            return

        state = event.get("state")
        if not isinstance(state, str) or state.strip() == "":
            state = "info"
        state = state.strip().lower()

        method = event.get("method")
        if not isinstance(method, str) or method.strip() == "":
            method = name
        method = method.strip()

        summary = event.get("summary")
        if not isinstance(summary, str) or summary.strip() == "":
            summary = method
        summary = summary.strip()

        headline = event.get("headline")
        if not isinstance(headline, str):
            headline = ""
        headline = headline.strip()

        item_id = event.get("item_id")
        if not isinstance(item_id, str):
            item_id = ""

        item_type = event.get("item_type")
        if not isinstance(item_type, str):
            item_type = ""
        event_path = event.get("path")
        if not isinstance(event_path, str):
            event_path = ""
        preview = event.get("preview")
        if not isinstance(preview, str):
            preview = ""

        raw_details = event.get("details")
        details: str
        if isinstance(raw_details, list):
            detail_lines = [
                line.strip() for line in raw_details if isinstance(line, str)
            ]
            details = "\n".join(line for line in detail_lines if line != "")
        elif isinstance(raw_details, str):
            details = raw_details.strip()
        else:
            details = ""

        data = event.get("data")
        if not isinstance(data, str):
            data = json.dumps(event, ensure_ascii=False, default=str)
        persisted_data = data if kind == "diff" and data != "" else ""

        correlation_key = event.get("correlation_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = event.get("event_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = None
        else:
            correlation_key = correlation_key.strip()

        in_progress = state in _ACTIVE_STATES
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        event_element: Element | None = None
        if correlation_key is not None:
            event_element = self._active_events_by_key.get(correlation_key)

        if event_element is None:
            attributes = {
                "id": str(uuid.uuid4()),
                "source": source,
                "name": name,
                "kind": kind,
                "state": state,
                "method": method,
                "item_id": item_id,
                "item_type": item_type,
                "path": event_path,
                "preview": preview,
                "summary": summary,
                "headline": headline,
                "details": details,
                "created_at": now,
                "updated_at": now,
            }
            if persisted_data != "":
                attributes["data"] = persisted_data
            event_element = messages.append_child(
                tag_name="event",
                attributes=attributes,
            )
        else:
            event_element.set_attribute("source", source)
            event_element.set_attribute("name", name)
            event_element.set_attribute("kind", kind)
            event_element.set_attribute("state", state)
            event_element.set_attribute("method", method)
            event_element.set_attribute("item_id", item_id)
            event_element.set_attribute("item_type", item_type)
            event_element.set_attribute("path", event_path)
            if preview != "" or event_element.get_attribute("preview") in (None, ""):
                event_element.set_attribute("preview", preview)
            event_element.set_attribute("summary", summary)
            event_element.set_attribute("headline", headline)
            if details != "" or event_element.get_attribute("details") in (None, ""):
                event_element.set_attribute("details", details)
            if persisted_data != "":
                event_element.set_attribute("data", persisted_data)
            event_element.set_attribute("updated_at", now)

        if correlation_key is not None:
            if in_progress:
                self._active_events_by_key[correlation_key] = event_element
            elif state in _TERMINAL_STATES:
                self._active_events_by_key.pop(correlation_key, None)

    async def _process_llm_events(self) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            raise RoomException("messages element is missing from thread document")
        doc_messages = messages[0]

        updates: asyncio.Queue = asyncio.Queue()
        update_thread_stop = object()
        message_elements_by_choice: dict[int, Element] = {}
        partial_text_by_choice: dict[int, str] = {}

        local_participant_name = self._room.local_participant.get_attribute("name")
        if not isinstance(local_participant_name, str):
            local_participant_name = ""

        async def update_thread() -> None:
            while True:
                entry = await updates.get()
                if entry is update_thread_stop:
                    break

                element, text = entry
                changes: dict[Element, str] = {element: text}
                should_stop = False
                while True:
                    try:
                        pending = updates.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if pending is update_thread_stop:
                        should_stop = True
                        break

                    pending_element, pending_text = pending
                    changes[pending_element] = pending_text

                for changed_element, changed_text in changes.items():
                    changed_element["text"] = changed_text

                if should_stop:
                    break

        def ensure_message_element(*, choice_index: int) -> Element:
            existing = message_elements_by_choice.get(choice_index)
            if existing is not None:
                return existing

            element = doc_messages.append_child(
                tag_name="message",
                attributes={
                    "text": "",
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "author_name": local_participant_name,
                },
            )
            message_elements_by_choice[choice_index] = element
            return element

        def queue_text_update(*, choice_index: int, text: str) -> None:
            element = ensure_message_element(choice_index=choice_index)
            updates.put_nowait((element, text))

        def set_final_text(*, choice_index: int, text: str) -> None:
            queue_text_update(choice_index=choice_index, text=text)
            with tracer.start_as_current_span("chatbot.thread.message") as span:
                span.set_attribute("from_participant_name", local_participant_name)
                span.set_attribute("role", "assistant")
                span.set_attribute("text", text)

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await self._llm_messages.get()
                if not isinstance(evt, dict):
                    continue

                if _is_completion_chunk_event(event=evt):
                    for choice in _choices(event=evt):
                        choice_index = _choice_index(choice=choice)

                        delta = _choice_delta_text(choice=choice)
                        if delta != "":
                            current = partial_text_by_choice.get(choice_index, "")
                            next_text = current + delta
                            partial_text_by_choice[choice_index] = next_text
                            queue_text_update(choice_index=choice_index, text=next_text)

                        if _choice_finish_reason(choice=choice) is not None:
                            final_text = partial_text_by_choice.get(choice_index, "")
                            if (
                                final_text != ""
                                or choice_index in message_elements_by_choice
                            ):
                                set_final_text(
                                    choice_index=choice_index,
                                    text=final_text,
                                )

                            partial_text_by_choice.pop(choice_index, None)
                    continue

                if _is_completion_event(event=evt):
                    choices = _choices(event=evt)
                    if len(choices) == 0:
                        text = _to_text(value=evt.get("text"))
                        if text != "":
                            set_final_text(choice_index=0, text=text)
                        continue

                    for choice in choices:
                        choice_index = _choice_index(choice=choice)
                        text = _choice_message_text(choice=choice)
                        if text == "":
                            text = partial_text_by_choice.get(choice_index, "")

                        if text == "":
                            continue

                        partial_text_by_choice[choice_index] = text
                        set_final_text(choice_index=choice_index, text=text)
                        partial_text_by_choice.pop(choice_index, None)
                    continue

                await self.handle_custom_event(messages=doc_messages, event=evt)
        except asyncio.QueueShutDown:
            pass
        finally:
            updates.put_nowait(update_thread_stop)

        await update_thread_task
