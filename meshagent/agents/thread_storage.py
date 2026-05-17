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


def thread_dir_for_namespace(*, thread_dir: str, namespace: str | None) -> str:
    if namespace is None or namespace == "":
        return thread_dir
    return posixpath.join(thread_dir, namespace)


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
    @classmethod
    def thread_list_path_for_dir(cls, *, thread_dir: str) -> str: ...

    @classmethod
    async def list_threads(
        cls,
        *,
        room: RoomClient,
        thread_dir: str,
        namespace: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage: ...

    @classmethod
    async def upsert_thread(
        cls,
        *,
        room: RoomClient,
        thread_dir: str,
        namespace: str | None = None,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> None: ...

    @classmethod
    async def delete_thread(
        cls,
        *,
        room: RoomClient,
        thread_dir: str,
        namespace: str | None = None,
        path: str,
        delete_storage: bool = True,
    ) -> None: ...

    @classmethod
    async def rename_thread(
        cls,
        *,
        room: RoomClient,
        thread_dir: str,
        namespace: str | None = None,
        path: str,
        name: str,
    ) -> None: ...

    @classmethod
    def watch_threads(
        cls,
        *,
        room: RoomClient,
        thread_dir: str,
        namespace: str | None = None,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]: ...


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
