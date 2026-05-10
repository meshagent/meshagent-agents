from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StreamStatus = Literal["pending", "in_progress", "completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class AccumulatedTextContent:
    item_id: str
    turn_id: str | None = None
    status: StreamStatus = "in_progress"
    text: str = ""
    sender_name: str | None = None
    phase: Literal["commentary", "final_answer"] | None = None


class TextContentAccumulator:
    def __init__(self) -> None:
        self._items: dict[str, AccumulatedTextContent] = {}

    def upsert(
        self,
        *,
        item_id: str,
        turn_id: str | None = None,
        sender_name: str | None = None,
        phase: Literal["commentary", "final_answer"] | None = None,
    ) -> AccumulatedTextContent:
        current = self._items.get(item_id)
        next_item = AccumulatedTextContent(
            item_id=item_id,
            turn_id=_coalesce_text(turn_id, current.turn_id if current else None),
            status="in_progress",
            text=current.text if current else "",
            sender_name=_coalesce_text(
                sender_name, current.sender_name if current else None
            ),
            phase=phase if phase is not None else (current.phase if current else None),
        )
        self._items[item_id] = next_item
        return next_item

    def append_delta(
        self,
        *,
        item_id: str,
        delta: str,
        turn_id: str | None = None,
        sender_name: str | None = None,
        phase: Literal["commentary", "final_answer"] | None = None,
    ) -> AccumulatedTextContent:
        current = self.upsert(
            item_id=item_id,
            turn_id=turn_id,
            sender_name=sender_name,
            phase=phase,
        )
        next_item = AccumulatedTextContent(
            item_id=item_id,
            turn_id=current.turn_id,
            status="in_progress",
            text=accumulate_text_delta(current=current.text, delta=delta),
            sender_name=current.sender_name,
            phase=current.phase,
        )
        self._items[item_id] = next_item
        return next_item

    def get(self, item_id: str) -> AccumulatedTextContent | None:
        return self._items.get(item_id)

    def remove(self, item_id: str) -> AccumulatedTextContent | None:
        return self._items.pop(item_id, None)

    def complete(
        self,
        *,
        item_id: str,
        status: StreamStatus = "completed",
    ) -> AccumulatedTextContent | None:
        current = self._items.get(item_id)
        if current is None:
            return None
        next_item = AccumulatedTextContent(
            item_id=current.item_id,
            turn_id=current.turn_id,
            status=status,
            text=current.text,
            sender_name=current.sender_name,
            phase=current.phase,
        )
        self._items[item_id] = next_item
        return next_item

    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


def accumulate_text_delta(*, current: str, delta: str) -> str:
    if current == "":
        return delta
    if delta == "":
        return current
    if delta == current:
        return current
    if delta.startswith(current):
        return delta
    if current.endswith(delta):
        return current
    return current + delta


@dataclass(frozen=True, slots=True)
class AccumulatedFileContent:
    item_id: str
    turn_id: str | None = None
    status: StreamStatus = "in_progress"
    urls: tuple[str, ...] = field(default_factory=tuple)
    sender_name: str | None = None

    @property
    def latest_url(self) -> str | None:
        return self.urls[-1] if self.urls else None


class FileContentAccumulator:
    def __init__(self) -> None:
        self._items: dict[str, AccumulatedFileContent] = {}

    def upsert(
        self,
        *,
        item_id: str,
        turn_id: str | None = None,
        sender_name: str | None = None,
    ) -> AccumulatedFileContent:
        current = self._items.get(item_id)
        next_item = AccumulatedFileContent(
            item_id=item_id,
            turn_id=_coalesce_text(turn_id, current.turn_id if current else None),
            status="in_progress",
            urls=current.urls if current else (),
            sender_name=_coalesce_text(
                sender_name, current.sender_name if current else None
            ),
        )
        self._items[item_id] = next_item
        return next_item

    def append_url(
        self,
        *,
        item_id: str,
        url: str,
        turn_id: str | None = None,
        sender_name: str | None = None,
    ) -> AccumulatedFileContent:
        current = self.upsert(
            item_id=item_id,
            turn_id=turn_id,
            sender_name=sender_name,
        )
        normalized_url = url.strip()
        urls = current.urls
        if normalized_url != "" and normalized_url not in urls:
            urls = (*urls, normalized_url)
        next_item = AccumulatedFileContent(
            item_id=item_id,
            turn_id=current.turn_id,
            status="in_progress",
            urls=urls,
            sender_name=current.sender_name,
        )
        self._items[item_id] = next_item
        return next_item

    def get(self, item_id: str) -> AccumulatedFileContent | None:
        return self._items.get(item_id)

    def remove(self, item_id: str) -> AccumulatedFileContent | None:
        return self._items.pop(item_id, None)

    def complete(
        self,
        *,
        item_id: str,
        status: StreamStatus = "completed",
    ) -> AccumulatedFileContent | None:
        current = self._items.get(item_id)
        if current is None:
            return None
        next_item = AccumulatedFileContent(
            item_id=current.item_id,
            turn_id=current.turn_id,
            status=status,
            urls=current.urls,
            sender_name=current.sender_name,
        )
        self._items[item_id] = next_item
        return next_item

    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


def _coalesce_text(value: str | None, fallback: str | None) -> str | None:
    if isinstance(value, str) and value.strip() != "":
        return value.strip()
    return fallback
