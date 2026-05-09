from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol

from meshagent.api import Participant
from meshagent.agents.messages import (
    AGENT_EVENT_THREAD_STATUS,
    AgentThreadMessage,
    AgentThreadStatus,
)

logger = logging.getLogger("agent.thread_status_publisher")

ThreadStatusMode = Literal["busy", "steerable"]


def thread_status_attribute_path_suffix(*, path: str) -> str:
    return path.strip()


class ThreadStatusPublisher(Protocol):
    async def set_thread_turn_id(self, *, turn_id: str | None) -> None: ...

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, Any]],
    ) -> None: ...

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode: ThreadStatusMode | None = None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
    ) -> None: ...

    async def clear_thread_status(self) -> None: ...


class ParticipantAttributeThreadStatusPublisher:
    """Compatibility no-op for the removed participant-attribute status path."""

    def __init__(
        self,
        *,
        participant: Participant,
        path: str,
        mode: ThreadStatusMode = "steerable",
    ) -> None:
        del participant
        self._path = path
        self._mode = mode
        self._lock = asyncio.Lock()
        self._generation = 0
        self._status_value: str | None = None
        self._mode_value: ThreadStatusMode | None = None
        self._started_at_value: str | None = None
        self._turn_id_value: str | None = None
        self._pending_messages_value: list[dict[str, Any]] = []
        self._pending_item_id_value: str | None = None
        self._total_bytes_value: int | None = None

    async def set_thread_turn_id(self, *, turn_id: str | None) -> None:
        async with self._lock:
            if self._turn_id_value == turn_id:
                return

            self._turn_id_value = turn_id

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, Any]],
    ) -> None:
        normalized = json.loads(json.dumps(pending_messages, ensure_ascii=False))

        async with self._lock:
            if self._pending_messages_value == normalized:
                return

            self._pending_messages_value = normalized

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode: ThreadStatusMode | None = None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
    ) -> None:
        async with self._lock:
            if status is None or status.strip() == "":
                self._status_value = None
                self._mode_value = None
                self._started_at_value = None
                self._pending_item_id_value = None
                self._total_bytes_value = None
                return

            normalized_status = status.strip()
            normalized_mode = mode if mode is not None else self._mode
            normalized_pending_item_id = (
                pending_item_id.strip()
                if isinstance(pending_item_id, str) and pending_item_id.strip() != ""
                else None
            )
            normalized_total_bytes = (
                total_bytes
                if isinstance(total_bytes, int) and total_bytes > 0
                else None
            )
            started_at = self._started_at_value
            if (
                started_at is None
                or self._status_value != normalized_status
                or self._mode_value != normalized_mode
            ):
                started_at = (
                    datetime.now(timezone.utc)
                    .isoformat()
                    .replace(
                        "+00:00",
                        "Z",
                    )
                )

            if (
                self._status_value == normalized_status
                and self._mode_value == normalized_mode
                and self._started_at_value == started_at
                and self._pending_item_id_value == normalized_pending_item_id
                and self._total_bytes_value == normalized_total_bytes
            ):
                return

            self._status_value = normalized_status
            self._mode_value = normalized_mode
            self._started_at_value = started_at
            self._pending_item_id_value = normalized_pending_item_id
            self._total_bytes_value = normalized_total_bytes

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    async def _apply_thread_status(
        self,
        *,
        status: str | None,
        generation: int | None = None,
    ) -> None:
        async with self._lock:
            if generation is not None and generation != self._generation:
                return

        await self.set_thread_status(status=status)

    def set_thread_status_nowait(self, *, status: str | None) -> None:
        generation = self._next_generation()

        async def run() -> None:
            try:
                await self._apply_thread_status(
                    status=status,
                    generation=generation,
                )
            except Exception:
                logger.exception("unable to set thread status for %s", self._path)

        asyncio.create_task(run())

    async def clear_thread_status(self) -> None:
        self._next_generation()
        await self.set_thread_status(status=None)


class AgentMessageThreadStatusPublisher:
    def __init__(
        self,
        *,
        thread_id: str,
        publish: Callable[[AgentThreadMessage], None],
        mode: ThreadStatusMode = "steerable",
    ) -> None:
        self._thread_id = thread_id
        self._publish = publish
        self._mode = mode
        self._lock = asyncio.Lock()
        self._generation = 0
        self._status_value: str | None = None
        self._mode_value: ThreadStatusMode | None = None
        self._started_at_value: str | None = None
        self._turn_id_value: str | None = None
        self._pending_item_id_value: str | None = None
        self._total_bytes_value: int | None = None

    def _publish_current_status(self) -> None:
        try:
            self._publish(
                AgentThreadStatus(
                    type=AGENT_EVENT_THREAD_STATUS,
                    thread_id=self._thread_id,
                    status=self._status_value,
                    mode=self._mode_value,
                    started_at=self._started_at_value,
                    turn_id=self._turn_id_value,
                    pending_item_id=self._pending_item_id_value,
                    total_bytes=self._total_bytes_value,
                )
            )
        except Exception:
            logger.exception("unable to publish thread status for %s", self._thread_id)

    async def set_thread_turn_id(self, *, turn_id: str | None) -> None:
        async with self._lock:
            if self._turn_id_value == turn_id:
                return

            self._turn_id_value = turn_id
            self._publish_current_status()

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, Any]],
    ) -> None:
        del pending_messages

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode: ThreadStatusMode | None = None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
    ) -> None:
        async with self._lock:
            if status is None or status.strip() == "":
                if (
                    self._status_value is None
                    and self._mode_value is None
                    and self._started_at_value is None
                    and self._pending_item_id_value is None
                    and self._total_bytes_value is None
                ):
                    return

                self._status_value = None
                self._mode_value = None
                self._started_at_value = None
                self._pending_item_id_value = None
                self._total_bytes_value = None
                self._publish_current_status()
                return

            normalized_status = status.strip()
            normalized_mode = mode if mode is not None else self._mode
            normalized_pending_item_id = (
                pending_item_id.strip()
                if isinstance(pending_item_id, str) and pending_item_id.strip() != ""
                else None
            )
            normalized_total_bytes = (
                total_bytes
                if isinstance(total_bytes, int) and total_bytes > 0
                else None
            )
            started_at = self._started_at_value
            if (
                started_at is None
                or self._status_value != normalized_status
                or self._mode_value != normalized_mode
            ):
                started_at = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )

            if (
                self._status_value == normalized_status
                and self._mode_value == normalized_mode
                and self._started_at_value == started_at
                and self._pending_item_id_value == normalized_pending_item_id
                and self._total_bytes_value == normalized_total_bytes
            ):
                return

            self._status_value = normalized_status
            self._mode_value = normalized_mode
            self._started_at_value = started_at
            self._pending_item_id_value = normalized_pending_item_id
            self._total_bytes_value = normalized_total_bytes
            self._publish_current_status()

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    async def _apply_thread_status(
        self,
        *,
        status: str | None,
        generation: int | None = None,
    ) -> None:
        async with self._lock:
            if generation is not None and generation != self._generation:
                return

        await self.set_thread_status(status=status)

    def set_thread_status_nowait(self, *, status: str | None) -> None:
        generation = self._next_generation()

        async def run() -> None:
            try:
                await self._apply_thread_status(
                    status=status,
                    generation=generation,
                )
            except Exception:
                logger.exception("unable to set thread status for %s", self._thread_id)

        asyncio.create_task(run())

    async def clear_thread_status(self) -> None:
        self._next_generation()
        await self.set_thread_status(status=None)
