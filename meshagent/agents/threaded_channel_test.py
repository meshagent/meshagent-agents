import asyncio
import logging

import pytest

from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.thread_schema import thread_list_schema
from meshagent.agents.thread_storage import ThreadListEntry, ThreadListPage
from meshagent.agents.threaded_channel import ThreadedChannel


class _FakeParticipant:
    def __init__(self, *, name: str = "assistant") -> None:
        self._name = name
        self.attributes: list[dict[str, str]] = []

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None

    async def set_attribute(self, key: str, value: str) -> None:
        self.attributes.append({"key": key, "value": value})


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: dict | None = None) -> None:
        self.tag_name = tag_name
        self.attributes = dict(attributes or {})
        self.children: list["_FakeElement"] = []

    def get_attribute(self, key: str):
        return self.attributes.get(key)

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value

    def append_child(
        self,
        *,
        tag_name: str,
        attributes: dict | None = None,
    ) -> "_FakeElement":
        child = _FakeElement(tag_name=tag_name, attributes=attributes)
        self.children.append(child)
        return child

    def get_children(self) -> list["_FakeElement"]:
        return [*self.children]


class _FakeDocument:
    def __init__(self) -> None:
        self.root = _FakeElement(tag_name="threads")


class _FakeSync:
    def __init__(self, document: _FakeDocument) -> None:
        self.document = document
        self.open_calls: list[dict] = []
        self.close_calls: list[str] = []

    async def open(self, *, path: str, schema):
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)


class _FakeSearchTable:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict]:
        return [dict(row) for row in self.rows]


class _FakeDatasets:
    def __init__(self, *, tables: list[str], rows: list[dict]) -> None:
        self.tables = tables
        self.rows = rows
        self.list_tables_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.search_calls: list[dict] = []

    async def list_tables(self, *, namespace=None) -> list[str]:
        self.list_tables_calls.append({"namespace": namespace})
        return [*self.tables]

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema=None,
        mode=None,
        namespace=None,
    ) -> None:
        self.create_calls.append(
            {"name": name, "schema": schema, "mode": mode, "namespace": namespace}
        )

    async def search(self, *, table: str, namespace=None) -> _FakeSearchTable:
        self.search_calls.append({"table": table, "namespace": namespace})
        return _FakeSearchTable(self.rows)


class _FakeRoom:
    def __init__(
        self,
        *,
        participant_name: str = "assistant",
        datasets: _FakeDatasets | None = None,
    ) -> None:
        self.local_participant = _FakeParticipant(name=participant_name)
        self.document = _FakeDocument()
        self.sync = _FakeSync(self.document)
        self.datasets = datasets


class _FakeThreadNameContext:
    def __init__(self) -> None:
        self.rules: list[str] | None = None
        self.user_messages: list[str] = []
        self.entered = False
        self.exited = False

    def copy(self) -> "_FakeThreadNameContext":
        return self

    async def __aenter__(self) -> "_FakeThreadNameContext":
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        self.exited = True

    def replace_rules(self, *, rules: list[str]) -> None:
        self.rules = rules

    def append_user_message(self, message: str) -> None:
        self.user_messages.append(message)


class _FakeThreadNameAdapter:
    def __init__(self) -> None:
        self.context = _FakeThreadNameContext()
        self.response_calls: list[dict] = []

    def create_session(self) -> _FakeThreadNameContext:
        return self.context

    def default_model(self) -> str:
        return "thread-name-model"

    async def create_response(self, **kwargs):
        self.response_calls.append(kwargs)
        return {"thread_name": " generated title.thread "}


class _FakeThreadRepository:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, int]] = []
        self.upsert_calls: list[dict] = []

    @property
    def scheme(self) -> str:
        return "repo"

    @property
    def is_ephemeral(self) -> bool:
        return False

    def thread_list_path(self) -> str:
        return "repo://threads/index.threadl"

    async def list_threads(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ThreadListPage:
        self.list_calls.append({"limit": limit, "offset": offset})
        return ThreadListPage(
            threads=[
                ThreadListEntry(
                    name="Repo Thread",
                    path="repo://threads/repo.thread",
                    created_at="2026-01-01T00:00:00Z",
                    modified_at="2026-01-02T00:00:00Z",
                )
            ],
            total=1,
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
    ) -> ThreadListEntry:
        self.upsert_calls.append(
            {
                "path": path,
                "name": name,
                "created_at": created_at,
                "modified_at": modified_at,
            }
        )
        return ThreadListEntry(
            name=name or "",
            path=path,
            created_at=created_at or "",
            modified_at=modified_at or "",
        )


class _RepositoryBackedThreadedChannel(ThreadedChannel):
    def __init__(self, *, repository: _FakeThreadRepository, room: _FakeRoom) -> None:
        self.repository = repository
        super().__init__(
            room=room,  # type: ignore[arg-type]
            threading_mode="default-new",
            thread_dir="threads",
            thread_url_scheme="repo",
        )

    def _thread_storage_repository(self):
        return self.repository


class _DatasetBackedThreadedChannel(ThreadedChannel):
    def _thread_storage_repository(self):
        return DatasetThreadStorage(
            room=self.room,  # type: ignore[arg-type]
            thread_dir=self.get_thread_dir(),
        )


@pytest.mark.asyncio
async def test_threaded_channel_publish_and_document_lifecycle_execute_provider_calls() -> (
    None
):
    room = _FakeRoom(participant_name="agent one")
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )

    await channel.publish_thread_attributes()
    await channel.open_thread_list_document()
    await channel.open_thread_list_document()
    await channel.close_thread_list_document()

    assert room.local_participant.attributes == [
        {"key": "meshagent.chatbot.threading", "value": "default-new"},
        {"key": "meshagent.chatbot.thread-dir", "value": ".threads/agent one"},
        {
            "key": "meshagent.chatbot.thread-list",
            "value": ".threads/agent one/index.threadl",
        },
    ]
    assert room.sync.open_calls == [
        {
            "path": ".threads/agent one/index.threadl",
            "schema": thread_list_schema,
        }
    ]
    assert room.sync.close_calls == [".threads/agent one/index.threadl"]
    assert channel._thread_list_document is None
    assert channel._thread_list_path is None


@pytest.mark.asyncio
async def test_threaded_channel_list_and_bump_threads_mutate_open_document() -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    await channel.open_thread_list_document()

    channel.bump_thread(path=".threads/assistant/old.thread", name="Old")
    channel.bump_thread(path=".threads/assistant/new.thread", name="New")
    first_entry = room.document.root.children[0]
    first_entry.set_attribute("modified_at", "2026-01-01T00:00:00Z")
    second_entry = room.document.root.children[1]
    second_entry.set_attribute("modified_at", "2026-01-02T00:00:00Z")
    channel.bump_thread(path=".threads/assistant/new.thread", name="Ignored")

    page = await channel.list_threads(limit=1, offset=0)

    assert len(room.document.root.children) == 2
    assert second_entry.get_attribute("name") == "New"
    assert page.total == 2
    assert page.limit == 1
    assert page.offset == 0
    assert [(entry.name, entry.path) for entry in page.threads] == [
        ("New", ".threads/assistant/new.thread")
    ]


@pytest.mark.asyncio
async def test_threaded_channel_repository_backed_list_and_upsert_delegate() -> None:
    repository = _FakeThreadRepository()
    room = _FakeRoom()
    channel = _RepositoryBackedThreadedChannel(repository=repository, room=room)

    page = await channel.list_threads(limit=500, offset=-10)
    channel.bump_thread(
        path="repo://threads/repo.thread",
        name="Repo Name",
    )
    await channel._wait_for_thread_list_background_tasks()

    assert repository.list_calls == [{"limit": 200, "offset": 0}]
    assert page.threads[0].name == "Repo Thread"
    assert repository.upsert_calls == [
        {
            "path": "repo://threads/repo.thread",
            "name": "Repo Name",
            "created_at": None,
            "modified_at": repository.upsert_calls[0]["modified_at"],
        }
    ]
    assert repository.upsert_calls[0]["modified_at"] is not None
    assert channel._thread_list_background_tasks == set()


@pytest.mark.asyncio
async def test_threaded_channel_dataset_repository_list_threads_executes_client_path() -> (
    None
):
    datasets = _FakeDatasets(
        tables=["index"],
        rows=[
            {
                "path": "dataset://threads/older.thread",
                "name": " Older ",
                "created_at": "2026-01-01T00:00:00Z",
                "modified_at": "2026-01-01T00:00:00Z",
            },
            {
                "path": "dataset://threads/missing-modified.thread",
                "name": 7,
                "created_at": "2026-01-03T00:00:00Z",
                "modified_at": None,
            },
            {
                "path": "  dataset://threads/newer.thread  ",
                "name": " Newer ",
                "created_at": "2026-01-02T00:00:00Z",
                "modified_at": "2026-01-04T00:00:00Z",
            },
            {
                "path": "",
                "name": "ignored",
                "created_at": "2026-01-05T00:00:00Z",
                "modified_at": "2026-01-05T00:00:00Z",
            },
        ],
    )
    room = _FakeRoom(datasets=datasets)
    channel = _DatasetBackedThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
        thread_dir="threads",
        thread_url_scheme="dataset",
    )

    page = await channel.list_threads(limit=2, offset=0)

    assert datasets.list_tables_calls == [{"namespace": ["threads"]}]
    assert datasets.create_calls == []
    assert datasets.search_calls == [{"table": "index", "namespace": ["threads"]}]
    assert page.total == 3
    assert page.offset == 0
    assert page.limit == 2
    assert [entry.path for entry in page.threads] == [
        "dataset://threads/newer.thread",
        "dataset://threads/older.thread",
    ]
    assert [entry.name for entry in page.threads] == ["Newer", "Older"]
    assert [entry.created_at for entry in page.threads] == [
        "2026-01-02T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ]
    assert page.threads[0].modified_at == "2026-01-04T00:00:00Z"

    second_page = await channel.list_threads(limit=2, offset=2)
    assert [entry.path for entry in second_page.threads] == [
        "dataset://threads/missing-modified.thread"
    ]
    assert second_page.threads[0].name == ""
    assert second_page.threads[0].modified_at == ""


@pytest.mark.asyncio
async def test_threaded_channel_new_thread_determines_name_before_bump(
    monkeypatch,
) -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    await channel.open_thread_list_document()
    calls: list[dict] = []

    async def _fake_determine_thread_name(**kwargs):
        calls.append({"method": "determine", **kwargs})
        return "Friendly"

    async def _fake_new_thread_path():
        calls.append({"method": "path"})
        return ".threads/assistant/generated.thread"

    monkeypatch.setattr(channel, "_determine_thread_name", _fake_determine_thread_name)
    monkeypatch.setattr(channel, "_new_thread_path", _fake_new_thread_path)
    on_behalf_of = object()

    path, name = await channel.new_thread(
        message_text="hello",
        attachments=["room://file.txt"],
        on_behalf_of=on_behalf_of,  # type: ignore[arg-type]
    )

    assert path == ".threads/assistant/generated.thread"
    assert name == "Friendly"
    assert calls == [
        {
            "method": "determine",
            "message_text": "hello",
            "attachments": ["room://file.txt"],
            "on_behalf_of": on_behalf_of,
        },
        {"method": "path"},
    ]
    assert room.document.root.children[0].get_attribute("name") == "Friendly"
    assert (
        room.document.root.children[0].get_attribute("path")
        == ".threads/assistant/generated.thread"
    )


@pytest.mark.asyncio
async def test_threaded_channel_new_thread_allocates_path_and_indexes_thread() -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    await channel.open_thread_list_document()

    path, name = await channel.new_thread(
        message_text="new project thread",
        attachments=None,
    )

    assert name == "New Project Thread"
    assert path.startswith(".threads/assistant/")
    assert path.endswith(".thread")
    assert len(room.document.root.children) == 1
    entry = room.document.root.children[0]
    assert entry.get_attribute("name") == "New Project Thread"
    assert entry.get_attribute("path") == path
    assert entry.get_attribute("created_at") is not None
    assert entry.get_attribute("modified_at") is not None


@pytest.mark.asyncio
async def test_threaded_channel_determine_thread_name_calls_configured_adapter() -> (
    None
):
    adapter = _FakeThreadNameAdapter()
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
        llm_adapter=adapter,  # type: ignore[arg-type]
        thread_name_rules=["rule one", "rule two"],
    )
    on_behalf_of = object()

    name = await channel._determine_thread_name(
        message_text="  hello world  ",
        attachments=["room://folder/example.txt"],
        on_behalf_of=on_behalf_of,  # type: ignore[arg-type]
    )

    assert name == "Generated Title"
    assert adapter.context.entered is True
    assert adapter.context.exited is True
    assert adapter.context.rules == ["rule one", "rule two"]
    assert adapter.context.user_messages == [
        "Message:\nhello world\n\nAttachments:\n- example.txt"
    ]
    assert len(adapter.response_calls) == 1
    call = adapter.response_calls[0]
    assert call["context"] is adapter.context
    assert call["caller"] is room.local_participant
    assert call["model"] == "thread-name-model"
    assert call["on_behalf_of"] is on_behalf_of
    assert call["toolkits"] == []
    assert call["output_schema"]["required"] == ["thread_name"]


@pytest.mark.asyncio
async def test_threaded_channel_determine_thread_name_uses_fallback_without_adapter() -> (
    None
):
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )

    name = await channel._determine_thread_name(
        message_text="",
        attachments=["room://folder/report.pdf", "/tmp/notes.md"],
    )

    assert name == "Report.Pdf, Notes.Md"


@pytest.mark.asyncio
async def test_threaded_channel_schedule_pending_thread_list_entry_success() -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    await channel.open_thread_list_document()
    channel._begin_pending_thread_list_entry(path=".threads/assistant/generated.thread")

    async def _determine_thread_name(**kwargs):
        assert kwargs["message_text"] == "hello"
        assert kwargs["attachments"] == ["room://file.txt"]
        return "Generated"

    channel._determine_thread_name = _determine_thread_name  # type: ignore[method-assign]
    channel._schedule_pending_thread_list_entry(
        path=".threads/assistant/generated.thread",
        message_text="hello",
        attachments=["room://file.txt"],
    )

    await channel._wait_for_thread_list_background_tasks()

    assert channel._pending_thread_list_paths == set()
    assert channel._thread_list_background_tasks == set()
    assert room.document.root.children[0].get_attribute("name") == "Generated"
    assert (
        room.document.root.children[0].get_attribute("path")
        == ".threads/assistant/generated.thread"
    )


@pytest.mark.asyncio
async def test_threaded_channel_schedule_pending_thread_list_entry_fallback(
    caplog,
) -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    await channel.open_thread_list_document()
    channel._begin_pending_thread_list_entry(path=".threads/assistant/generated.thread")

    async def _determine_thread_name(**kwargs):
        del kwargs
        raise RuntimeError("name failed")

    channel._determine_thread_name = _determine_thread_name  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="threaded-channel"):
        channel._schedule_pending_thread_list_entry(
            path=".threads/assistant/generated.thread",
            message_text="fallback title",
            attachments=None,
        )
        await channel._wait_for_thread_list_background_tasks()

    assert "unable to determine deferred thread name" in caplog.text
    assert channel._pending_thread_list_paths == set()
    assert room.document.root.children[0].get_attribute("name") == "Fallback Title"


@pytest.mark.asyncio
async def test_threaded_channel_cancel_background_tasks_clears_state() -> None:
    room = _FakeRoom()
    channel = ThreadedChannel(
        room=room,  # type: ignore[arg-type]
        threading_mode="default-new",
    )
    started = asyncio.Event()

    async def _pending_task() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_pending_task())
    channel._pending_thread_list_paths.add(".threads/assistant/pending.thread")
    channel._track_thread_list_background_task(task=task)
    await started.wait()

    await channel._cancel_thread_list_background_tasks()

    assert task.cancelled()
    assert channel._thread_list_background_tasks == set()
    assert channel._pending_thread_list_paths == set()
