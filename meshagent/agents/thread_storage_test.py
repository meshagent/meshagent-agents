import asyncio

import pytest

from meshagent.agents.thread_storage import (
    MultiThreadStorageRepository,
    ThreadListEntry,
    ThreadListEvent,
    ThreadListPage,
    thread_dir_for_namespace,
)


class _FakeThreadStorageRepository:
    def __init__(self, *, scheme: str, entries: list[ThreadListEntry]) -> None:
        self._scheme = scheme
        self.entries = {entry.path: entry for entry in entries}
        self.upserts: list[str] = []
        self.deletes: list[str] = []
        self.renames: list[tuple[str, str]] = []
        self.watch_queue: asyncio.Queue[ThreadListEvent] = asyncio.Queue()

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def is_ephemeral(self) -> bool:
        return False

    def thread_list_path(self) -> str:
        return f"{self._scheme}://index"

    async def list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        threads = list(self.entries.values())
        return ThreadListPage(
            threads=threads[offset : offset + limit],
            total=len(threads),
            offset=offset,
            limit=limit,
        )

    async def upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        self.upserts.append(path)
        entry = ThreadListEntry(
            name=name or path,
            path=path,
            created_at=created_at or "2026-01-01T00:00:00Z",
            modified_at=modified_at or "2026-01-01T00:00:00Z",
        )
        self.entries[path] = entry
        return entry

    async def delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None:
        del delete_storage
        self.deletes.append(path)
        self.entries.pop(path, None)

    async def rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None:
        self.renames.append((path, name))
        entry = self.entries[path]
        updated = ThreadListEntry(
            name=name,
            path=entry.path,
            created_at=entry.created_at,
            modified_at="2026-01-04T00:00:00Z",
        )
        self.entries[path] = updated
        return updated

    async def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ):
        del poll_interval
        while True:
            yield await self.watch_queue.get()


def test_thread_dir_for_namespace_returns_base_thread_dir_without_namespace() -> None:
    assert (
        thread_dir_for_namespace(thread_dir=" /threads/ ", namespace=None) == "threads"
    )


def test_thread_dir_for_namespace_appends_namespace() -> None:
    assert (
        thread_dir_for_namespace(
            thread_dir="threads",
            namespace="jesse.ezell@timu.com",
        )
        == "threads/jesse.ezell@timu.com"
    )


def test_thread_dir_for_namespace_rejects_empty_thread_dir() -> None:
    with pytest.raises(ValueError, match="thread_dir must not be empty"):
        thread_dir_for_namespace(thread_dir=" / ", namespace=None)


def test_thread_dir_for_namespace_rejects_relative_namespace_parts() -> None:
    with pytest.raises(
        ValueError,
        match="namespace must not contain empty or relative path parts",
    ):
        thread_dir_for_namespace(thread_dir="threads", namespace="../other")


def test_multi_thread_storage_upserts_to_path_scheme_provider() -> None:
    async def run() -> None:
        meshdocument = _FakeThreadStorageRepository(scheme="meshdocument", entries=[])
        dataset = _FakeThreadStorageRepository(scheme="dataset", entries=[])
        repository = MultiThreadStorageRepository(
            repositories=[meshdocument, dataset],
            default_scheme="meshdocument",
        )

        await repository.upsert_thread(path="dataset://threads/one", name="Dataset")
        await repository.upsert_thread(
            path="meshdocument://threads/two.thread",
            name="Mesh",
        )

        assert dataset.upserts == ["dataset://threads/one"]
        assert meshdocument.upserts == ["meshdocument://threads/two.thread"]

    asyncio.run(run())


def test_multi_thread_storage_routes_unschemed_paths_by_existing_entry() -> None:
    async def run() -> None:
        meshdocument = _FakeThreadStorageRepository(
            scheme="meshdocument",
            entries=[
                ThreadListEntry(
                    name="Legacy",
                    path="/threads/legacy.thread",
                    created_at="2026-01-01T00:00:00Z",
                    modified_at="2026-01-01T00:00:00Z",
                )
            ],
        )
        dataset = _FakeThreadStorageRepository(scheme="dataset", entries=[])
        repository = MultiThreadStorageRepository(
            repositories=[dataset, meshdocument],
            default_scheme="dataset",
        )

        await repository.rename_thread(path="/threads/legacy.thread", name="Renamed")
        await repository.delete_thread(path="/threads/legacy.thread")

        assert meshdocument.renames == [("/threads/legacy.thread", "Renamed")]
        assert meshdocument.deletes == ["/threads/legacy.thread"]
        assert dataset.renames == []
        assert dataset.deletes == []

    asyncio.run(run())


def test_multi_thread_storage_lists_all_providers_sorted_descending() -> None:
    async def run() -> None:
        meshdocument = _FakeThreadStorageRepository(
            scheme="meshdocument",
            entries=[
                ThreadListEntry(
                    name="Older",
                    path="meshdocument://threads/older.thread",
                    created_at="2026-01-01T00:00:00Z",
                    modified_at="2026-01-01T00:00:00Z",
                )
            ],
        )
        dataset = _FakeThreadStorageRepository(
            scheme="dataset",
            entries=[
                ThreadListEntry(
                    name="Newer",
                    path="dataset://threads/newer",
                    created_at="2026-01-02T00:00:00Z",
                    modified_at="2026-01-03T00:00:00Z",
                )
            ],
        )
        repository = MultiThreadStorageRepository(
            repositories=[meshdocument, dataset],
            default_scheme="meshdocument",
        )

        page = await repository.list_threads(limit=20, offset=0)

        assert [entry.path for entry in page.threads] == [
            "dataset://threads/newer",
            "meshdocument://threads/older.thread",
        ]
        assert page.total == 2

    asyncio.run(run())


def test_multi_thread_storage_watch_yields_events_from_all_providers() -> None:
    async def run() -> None:
        meshdocument = _FakeThreadStorageRepository(scheme="meshdocument", entries=[])
        dataset = _FakeThreadStorageRepository(scheme="dataset", entries=[])
        repository = MultiThreadStorageRepository(
            repositories=[meshdocument, dataset],
            default_scheme="meshdocument",
        )

        watch = repository.watch_threads()
        await dataset.watch_queue.put(
            ThreadListEvent(type="deleted", path="dataset://threads/one")
        )

        event = await asyncio.wait_for(watch.__anext__(), timeout=1)
        await watch.aclose()

        assert event.type == "deleted"
        assert event.path == "dataset://threads/one"

    asyncio.run(run())
