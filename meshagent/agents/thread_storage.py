from __future__ import annotations

import asyncio
import posixpath
import uuid
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Literal,
    Protocol,
)

from meshagent.api import Participant, RoomClient
from meshagent.tools import Toolkit

from .context import AgentSessionContext
from .messages import AgentThreadMessage

if TYPE_CHECKING:
    from .adapter import LLMAdapter

THREAD_PATH_EXISTS_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ThreadListEntry:
    name: str
    path: str
    created_at: str
    modified_at: str


@dataclass(frozen=True, slots=True)
class ThreadListPage:
    threads: list[ThreadListEntry]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True, slots=True)
class ThreadListEvent:
    type: Literal["upserted", "renamed", "deleted"]
    path: str
    entry: ThreadListEntry | None = None


async def allocate_thread_path(
    *,
    room: RoomClient,
    base_path: str,
    extension: str = ".thread",
) -> str:
    try:
        exists = await asyncio.wait_for(
            room.storage.exists(path=base_path),
            timeout=THREAD_PATH_EXISTS_TIMEOUT_SECONDS,
        )
    except Exception:
        return base_path

    if not exists:
        return base_path

    thread_dir, filename = posixpath.split(base_path)
    if extension != "" and filename.endswith(extension):
        base_name = filename[: -len(extension)]
    else:
        base_name = filename

    for index in range(2, 1000):
        candidate = posixpath.join(thread_dir, f"{base_name} {index}{extension}")
        try:
            candidate_exists = await asyncio.wait_for(
                room.storage.exists(path=candidate),
                timeout=THREAD_PATH_EXISTS_TIMEOUT_SECONDS,
            )
            if not candidate_exists:
                return candidate
        except Exception:
            return candidate

    return posixpath.join(thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}{extension}")


class ThreadStorageRepository(Protocol):
    @property
    def is_ephemeral(self) -> bool: ...

    def thread_list_path(self) -> str: ...

    async def list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage: ...

    async def upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None: ...

    async def delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None: ...

    async def rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None: ...

    def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]: ...


class NoopThreadStorageRepository:
    @property
    def is_ephemeral(self) -> bool:
        return True

    def thread_list_path(self) -> str:
        return ""

    async def list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        return ThreadListPage(
            threads=[],
            total=0,
            offset=max(0, int(offset)),
            limit=max(1, min(200, int(limit))),
        )

    async def upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        del path
        del name
        del created_at
        del modified_at
        return None

    async def delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None:
        del path
        del delete_storage

    async def rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None:
        del path
        del name
        return None

    async def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]:
        del poll_interval
        await asyncio.Event().wait()
        if False:
            yield


class ThreadStorage(Protocol):
    @property
    def path(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def wait_until_ready(self) -> None: ...

    def unflushed_agent_messages(self) -> list[AgentThreadMessage]: ...

    def agent_messages(self) -> list[AgentThreadMessage]: ...

    def push_message(
        self,
        *,
        message: AgentThreadMessage,
        sender: Participant | None = None,
    ) -> None: ...

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None: ...

    async def restore_session_context_async(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None: ...

    def make_toolkit(self) -> Toolkit: ...
