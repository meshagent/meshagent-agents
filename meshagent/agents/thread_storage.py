from __future__ import annotations

import asyncio
import posixpath
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
    normalized_thread_dir = thread_dir.strip().strip("/")
    if normalized_thread_dir == "":
        raise ValueError("thread_dir must not be empty")

    if namespace is None:
        return normalized_thread_dir

    normalized_namespace = namespace.strip().strip("/")
    if normalized_namespace == "":
        return normalized_thread_dir

    namespace_parts = normalized_namespace.split("/")
    if any(part in {"", ".", ".."} for part in namespace_parts):
        raise ValueError("namespace must not contain empty or relative path parts")
    return posixpath.join(normalized_thread_dir, *namespace_parts)


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
    def scheme(self) -> str: ...

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
    def scheme(self) -> str:
        return "none"

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


def _parse_thread_list_datetime(*, value: str) -> datetime:
    raw = value.strip()
    if raw == "":
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def sort_thread_list_entries(entries: list[ThreadListEntry]) -> list[ThreadListEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            _parse_thread_list_datetime(value=entry.modified_at),
            _parse_thread_list_datetime(value=entry.created_at),
            entry.path,
        ),
        reverse=True,
    )


def thread_storage_scheme_for_path(path: str) -> str | None:
    normalized = path.strip()
    if normalized == "":
        return None
    scheme_end = normalized.find("://")
    if scheme_end <= 0:
        return None
    return normalized[:scheme_end].casefold()


class MultiThreadStorageRepository:
    def __init__(
        self,
        *,
        repositories: list[ThreadStorageRepository],
        default_scheme: str | None = None,
    ) -> None:
        if len(repositories) == 0:
            raise ValueError("at least one thread storage repository is required")
        self._repositories = repositories
        self._repositories_by_scheme: dict[str, ThreadStorageRepository] = {}
        for repository in repositories:
            scheme = repository.scheme.strip().casefold()
            if scheme == "":
                raise ValueError("thread storage repository scheme must not be empty")
            if scheme in self._repositories_by_scheme:
                raise ValueError(f"duplicate thread storage repository scheme {scheme}")
            self._repositories_by_scheme[scheme] = repository
        resolved_default_scheme = (
            default_scheme.strip().casefold()
            if isinstance(default_scheme, str) and default_scheme.strip() != ""
            else repositories[0].scheme.strip().casefold()
        )
        if resolved_default_scheme not in self._repositories_by_scheme:
            raise ValueError(
                f"default thread storage scheme {resolved_default_scheme} is not configured"
            )
        self._default_scheme = resolved_default_scheme

    @property
    def scheme(self) -> str:
        return self._default_scheme

    @property
    def is_ephemeral(self) -> bool:
        return all(repository.is_ephemeral for repository in self._repositories)

    def thread_list_path(self) -> str:
        return self._repositories_by_scheme[self._default_scheme].thread_list_path()

    def repository_for_scheme(self, scheme: str) -> ThreadStorageRepository:
        normalized = scheme.strip().casefold()
        repository = self._repositories_by_scheme.get(normalized)
        if repository is None:
            raise ValueError(f"thread storage scheme {scheme!r} is not configured")
        return repository

    async def _repository_for_path(self, path: str) -> ThreadStorageRepository:
        scheme = thread_storage_scheme_for_path(path)
        if scheme is not None:
            repository = self._repositories_by_scheme.get(scheme)
            if repository is None:
                raise ValueError(f"thread storage scheme {scheme!r} is not configured")
            return repository

        for repository in self._repositories:
            read_offset = 0
            while True:
                page = await repository.list_threads(limit=200, offset=read_offset)
                if any(entry.path == path for entry in page.threads):
                    return repository
                read_offset = page.offset + len(page.threads)
                if read_offset >= page.total or len(page.threads) == 0:
                    break
        return self._repositories_by_scheme[self._default_scheme]

    async def list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        normalized_limit = max(1, min(200, int(limit)))
        normalized_offset = max(0, int(offset))

        async def read_all(
            repository: ThreadStorageRepository,
        ) -> list[ThreadListEntry]:
            entries: list[ThreadListEntry] = []
            read_offset = 0
            while True:
                page = await repository.list_threads(limit=200, offset=read_offset)
                entries.extend(page.threads)
                read_offset = page.offset + len(page.threads)
                if read_offset >= page.total or len(page.threads) == 0:
                    return entries

        provider_entries = await asyncio.gather(
            *[read_all(repository) for repository in self._repositories]
        )
        entries = sort_thread_list_entries(
            [entry for page_entries in provider_entries for entry in page_entries]
        )
        selected = entries[normalized_offset : normalized_offset + normalized_limit]
        return ThreadListPage(
            threads=selected,
            total=len(entries),
            offset=normalized_offset,
            limit=normalized_limit,
        )

    async def upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        repository = await self._repository_for_path(path)
        return await repository.upsert_thread(
            path=path,
            name=name,
            created_at=created_at,
            modified_at=modified_at,
        )

    async def delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None:
        repository = await self._repository_for_path(path)
        await repository.delete_thread(path=path, delete_storage=delete_storage)

    async def rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None:
        repository = await self._repository_for_path(path)
        return await repository.rename_thread(path=path, name=name)

    async def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]:
        queue: asyncio.Queue[ThreadListEvent] = asyncio.Queue()
        tasks: list[asyncio.Task[None]] = []

        async def watch_repository(repository: ThreadStorageRepository) -> None:
            async for event in repository.watch_threads(poll_interval=poll_interval):
                await queue.put(event)

        for repository in self._repositories:
            tasks.append(asyncio.create_task(watch_repository(repository)))

        try:
            while True:
                yield await queue.get()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class ThreadStorage(Protocol):
    @property
    def scheme(self) -> str: ...

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
