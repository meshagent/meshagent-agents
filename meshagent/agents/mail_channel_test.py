import asyncio
import base64
from email.message import EmailMessage
import logging
import uuid

import pytest

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.mail_channel import MailChannel
from meshagent.agents.mail_common import SmtpConfiguration
from meshagent.agents.messages import (
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    TurnEnded,
    TurnStart,
    TurnStarted,
)
from meshagent.agents.process import Message
from meshagent.agents.thread_schema import thread_list_schema
from meshagent.api import Participant
from meshagent.api.messaging import FileContent
from meshagent.api.room_server_client import RoomException
from meshagent.tools import ToolContext


class _FakeLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})
        self.set_attribute_calls: list[tuple[str, object]] = []

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value
        self.set_attribute_calls.append((name, value))


class _FakeProtocol:
    def __init__(self) -> None:
        self.token = "token"


class _FakeQueues:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self.receive_calls: list[dict[str, object]] = []

    async def receive(self, *, name: str, create: bool, wait: bool):
        self.receive_calls.append({"name": name, "create": create, "wait": wait})
        return await self._queue.get()

    async def push(self, payload: dict) -> None:
        await self._queue.put(payload)


class _FakeStorage:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.exists_calls: list[str] = []

    async def upload(self, *, path: str, data: bytes) -> None:
        self.uploaded[path] = data

    async def exists(self, *, path: str) -> bool:
        self.exists_calls.append(path)
        return path in self.uploaded

    async def download(self, *, path: str) -> FileContent:
        return FileContent(
            data=self.uploaded[path],
            name=path.rsplit("/", 1)[-1],
            mime_type="application/octet-stream",
        )


class _FakeDatabase:
    def __init__(self) -> None:
        self.tables: dict[tuple[tuple[str, ...], str], list[dict[str, object]]] = {}
        self.create_calls: list[dict[str, object]] = []

    def _key(
        self, *, table: str, namespace: list[str] | None
    ) -> tuple[tuple[str, ...], str]:
        return (tuple(namespace or []), table)

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema,
        mode: str,
        namespace: list[str] | None = None,
    ) -> None:
        del schema
        self.create_calls.append(
            {"name": name, "mode": mode, "namespace": list(namespace or [])}
        )
        self.tables.setdefault(self._key(table=name, namespace=namespace), [])

    async def insert(
        self,
        *,
        table: str,
        records: list[dict[str, object]],
        namespace: list[str] | None = None,
    ) -> None:
        self.tables.setdefault(self._key(table=table, namespace=namespace), []).extend(
            [dict(record) for record in records]
        )

    async def search(
        self,
        *,
        table: str,
        where,
        namespace: list[str] | None = None,
    ) -> list[dict[str, object]]:
        rows = self.tables.setdefault(self._key(table=table, namespace=namespace), [])
        if isinstance(where, dict):
            return [
                row
                for row in rows
                if all(row.get(key) == value for key, value in where.items())
            ]
        raise AssertionError("test database only supports dict where clauses")


class _NamespaceRejectingDatabase(_FakeDatabase):
    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema,
        mode: str,
        namespace: list[str] | None = None,
    ) -> None:
        self.create_calls.append(
            {"name": name, "mode": mode, "namespace": list(namespace or [])}
        )
        if namespace:
            raise RoomException(
                "Error creating table 'emails': Invalid input, Location must be provided when namespace is not empty"
            )
        del schema
        self.tables.setdefault(self._key(table=name, namespace=namespace), [])


class _FakeThreadListElement:
    def __init__(self, *, tag_name: str, attributes: dict[str, str]) -> None:
        self.tag_name = tag_name
        self._attributes = dict(attributes)

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


class _FakeThreadListRoot:
    def __init__(self) -> None:
        self._children: list[_FakeThreadListElement] = []

    def get_children(self) -> list[_FakeThreadListElement]:
        return [*self._children]

    def append_child(
        self,
        *,
        tag_name: str,
        attributes: dict[str, str],
    ) -> _FakeThreadListElement:
        element = _FakeThreadListElement(tag_name=tag_name, attributes=attributes)
        self._children.append(element)
        return element


class _FakeThreadListDocument:
    def __init__(self) -> None:
        self.root = _FakeThreadListRoot()


class _FakeSync:
    def __init__(self) -> None:
        self.document = _FakeThreadListDocument()
        self.open_calls: list[dict[str, object]] = []
        self.close_calls: list[str] = []

    async def open(self, *, path: str, schema=None) -> _FakeThreadListDocument:
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)


class _FakeRoom:
    def __init__(self, *, database: _FakeDatabase | None = None) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.protocol = _FakeProtocol()
        self.queues = _FakeQueues()
        self.storage = _FakeStorage()
        self.database = database if database is not None else _FakeDatabase()
        self.sync = _FakeSync()
        self.is_closed = False


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


class _FakeThreadNameAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str) -> None:
        self.generated_thread_name = generated_thread_name
        self.prompts: list[str] = []

    def default_model(self) -> str:
        return "thread-name-model"

    async def next(
        self,
        *,
        context,
        caller,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        options=None,
    ):
        del caller
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        self.prompts = [
            message["content"]
            for message in context.messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        return {"thread_name": self.generated_thread_name}


def _assert_uuid_thread_path(*, path: str, prefix: str) -> None:
    assert path.startswith(prefix)
    assert path.endswith(".thread")
    basename = path[len(prefix) : -len(".thread")]
    parsed = uuid.UUID(basename)
    assert str(parsed) == basename


def _email_bytes(
    *,
    from_address: str,
    to_address: str,
    subject: str,
    body: str,
    in_reply_to: str | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
) -> bytes:
    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject
    if in_reply_to is not None:
        message["In-Reply-To"] = in_reply_to
    message.set_content(body)
    for file_name, data in attachments or []:
        message.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=file_name,
        )
    return message.as_bytes()


async def _drain() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_mail_channel_creates_new_thread_and_persists_thread_mapping() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
    )
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        raw_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Quarterly update",
            body="hello from email",
            attachments=[("report.txt", b"hi")],
        )
        await room.queues.push(
            {"base64": base64.b64encode(raw_message).decode("ascii")}
        )
        await _drain()

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        _assert_uuid_thread_path(
            path=outbound.data.thread_id,
            prefix=".threads/assistant/",
        )
        assert outbound.sender is not None
        assert outbound.sender.get_attribute("name") == "Alice"
        assert room.database.create_calls == [
            {
                "name": "emails",
                "mode": "create_if_not_exists",
                "namespace": [".threads", "assistant"],
            }
        ]
        stored_rows = room.database.tables[(((".threads", "assistant")), "emails")]
        assert len(stored_rows) == 1
        assert stored_rows[0]["thread_id"] == outbound.data.thread_id
        assert any(
            path.endswith("/attachments/report.txt") for path in room.storage.uploaded
        )
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_maps_reply_to_existing_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        thread_dir=".threads/helpdesk",
    )

    async def fake_send(message, *, hostname, port, username, password, local_hostname):
        del message
        del hostname
        del port
        del username
        del password
        del local_hostname

    monkeypatch.setattr("meshagent.agents.mail_channel.aiosmtplib.send", fake_send)
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        original_saved = await channel.start_thread(
            thread_id=".threads/helpdesk/ticket.thread",
            to_address="bob@example.com",
            subject="Initial",
            body="hi",
        )
        reply = _email_bytes(
            from_address="Bob <bob@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Re: Initial",
            body="following up",
            in_reply_to=original_saved["id"],
        )
        await room.queues.push({"base64": base64.b64encode(reply).decode("ascii")})
        await _drain()

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert outbound.data.thread_id == ".threads/helpdesk/ticket.thread"
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_default_new_indexes_new_threads_from_subject() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        threading_mode="default-new",
    )
    expected_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

    original_uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: expected_uuid
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        raw_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Quarterly update",
            body="hello from email",
        )
        await room.queues.push(
            {"base64": base64.b64encode(raw_message).decode("ascii")}
        )
        await _drain()

        assert room.local_participant.set_attribute_calls == [
            ("meshagent.chatbot.threading", "default-new"),
            ("meshagent.chatbot.thread-dir", ".threads/assistant"),
            ("meshagent.chatbot.thread-list", ".threads/assistant/index.threadl"),
        ]
        assert room.sync.open_calls == [
            {
                "path": ".threads/assistant/index.threadl",
                "schema": thread_list_schema,
            }
        ]
        assert room.storage.exists_calls == [
            ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        ]

        assert len(supervisor.sent) == 1
        outbound = supervisor.sent[0]
        assert isinstance(outbound.data, TurnStart)
        assert (
            outbound.data.thread_id
            == ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        )

        entries = room.sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Quarterly update"
        assert (
            entries[0].get_attribute("path")
            == ".threads/assistant/12345678-1234-5678-1234-567812345678.thread"
        )
    finally:
        uuid.uuid4 = original_uuid4
        await channel.stop(supervisor)  # type: ignore[arg-type]

    assert room.sync.close_calls == [".threads/assistant/index.threadl"]


@pytest.mark.asyncio
async def test_mail_channel_default_new_uses_attachment_names_for_llm_thread_naming() -> (
    None
):
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    adapter = _FakeThreadNameAdapter(generated_thread_name="Quarterly Files")
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        threading_mode="default-new",
        llm_adapter=adapter,
    )

    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        raw_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="",
            body="See the files",
            attachments=[("quarterly-report.pdf", b"pdf-bytes")],
        )
        await room.queues.push(
            {"base64": base64.b64encode(raw_message).decode("ascii")}
        )
        await _drain()

        assert adapter.prompts == [
            "Message:\nSee the files\n\nAttachments:\n- quarterly-report.pdf"
        ]
        entries = room.sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Quarterly Files"
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_default_new_reply_keeps_existing_thread_list_name() -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        threading_mode="default-new",
    )

    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        original_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Quarterly update",
            body="hello from email",
        )
        await room.queues.push(
            {"base64": base64.b64encode(original_message).decode("ascii")}
        )
        await _drain()

        stored_rows = room.database.tables[(((".threads", "assistant")), "emails")]
        reply_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Re: Quarterly update",
            body="following up",
            in_reply_to=stored_rows[0]["id"],
        )
        await room.queues.push(
            {"base64": base64.b64encode(reply_message).decode("ascii")}
        )
        await _drain()

        entries = room.sync.document.root.get_children()
        assert len(entries) == 1
        assert entries[0].get_attribute("name") == "Quarterly update"
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_falls_back_to_flat_table_when_namespace_create_is_rejected() -> (
    None
):
    database = _NamespaceRejectingDatabase()
    room = _FakeRoom(database=database)
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        thread_dir=".threads/helpdesk",
    )
    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        raw_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Fallback path",
            body="hello from email",
        )
        await room.queues.push(
            {"base64": base64.b64encode(raw_message).decode("ascii")}
        )
        await _drain()

        assert database.create_calls == [
            {
                "name": "emails",
                "mode": "create_if_not_exists",
                "namespace": [".threads", "helpdesk"],
            },
            {
                "name": "emails__threads__helpdesk",
                "mode": "create_if_not_exists",
                "namespace": [],
            },
        ]
        stored_rows = database.tables[((), "emails__threads__helpdesk")]
        assert len(stored_rows) == 1
        thread_id = stored_rows[0]["thread_id"]
        assert isinstance(thread_id, str)
        _assert_uuid_thread_path(path=thread_id, prefix=".threads/helpdesk/")
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_sends_reply_when_turn_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
    )
    sent_messages: list[EmailMessage] = []

    async def fake_send(message, *, hostname, port, username, password, local_hostname):
        del hostname
        del port
        del username
        del password
        del local_hostname
        sent_messages.append(message)

    monkeypatch.setattr("meshagent.agents.mail_channel.aiosmtplib.send", fake_send)

    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        raw_message = _email_bytes(
            from_address="Alice <alice@example.com>",
            to_address="mailbox@mail.meshagent.com",
            subject="Need help",
            body="please respond",
        )
        await room.queues.push(
            {"base64": base64.b64encode(raw_message).decode("ascii")}
        )
        await _drain()

        turn_start = supervisor.sent[0].data
        assert isinstance(turn_start, TurnStart)

        channel.send(
            Message(
                data=TurnStarted(
                    type="meshagent.agent.turn.started",
                    thread_id=turn_start.thread_id,
                    turn_id="turn-1",
                    source_message_id=turn_start.message_id,
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentStarted(
                    type="meshagent.agent.text_content.started",
                    thread_id=turn_start.thread_id,
                    turn_id="turn-1",
                    item_id="item-1",
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentDelta(
                    type="meshagent.agent.text_content.delta",
                    thread_id=turn_start.thread_id,
                    turn_id="turn-1",
                    item_id="item-1",
                    text="Thanks for the note.",
                )
            )
        )
        channel.send(
            Message(
                data=AgentTextContentEnded(
                    type="meshagent.agent.text_content.ended",
                    thread_id=turn_start.thread_id,
                    turn_id="turn-1",
                    item_id="item-1",
                )
            )
        )
        channel.send(
            Message(
                data=TurnEnded(
                    type="meshagent.agent.turn.ended",
                    thread_id=turn_start.thread_id,
                    turn_id="turn-1",
                    error=None,
                )
            )
        )
        await _drain()

        assert len(sent_messages) == 1
        assert sent_messages[0]["Subject"] == "RE: Need help"
        assert (
            "Thanks for the note."
            in sent_messages[0].get_body(("plain",)).get_content()
        )
        stored_rows = room.database.tables[(((".threads", "assistant")), "emails")]
        assert len(stored_rows) == 2
        assert stored_rows[-1]["thread_id"] == turn_start.thread_id
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_new_email_thread_tool_uses_current_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = _FakeRoom()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
    )
    sent_messages: list[EmailMessage] = []

    async def fake_send(message, *, hostname, port, username, password, local_hostname):
        del hostname
        del port
        del username
        del password
        del local_hostname
        sent_messages.append(message)

    monkeypatch.setattr("meshagent.agents.mail_channel.aiosmtplib.send", fake_send)

    await channel.start(_RecordingSupervisor())  # type: ignore[arg-type]
    try:
        room.storage.uploaded["notes.txt"] = b"hello"
        toolkit = channel.get_agent_toolkits()[0]
        new_thread_tool = toolkit.tools[0]
        await new_thread_tool.execute(
            ToolContext(
                caller=room.local_participant,
                caller_context={"thread_id": ".threads/assistant/customer.thread"},
            ),
            to="customer@example.com",
            subject="Follow up",
            body="Here is the update.",
            attachments=["notes.txt"],
        )

        assert len(sent_messages) == 1
        assert sent_messages[0]["To"] == "customer@example.com"
        rows = room.database.tables[(((".threads", "assistant")), "emails")]
        assert len(rows) == 1
        assert rows[0]["thread_id"] == ".threads/assistant/customer.thread"
    finally:
        await channel.stop(channel.supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_uses_roompool_hostname_fallback_for_smtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_HOSTNAME", raising=False)
    monkeypatch.delenv("SMTP_LOCAL_HOSTNAME", raising=False)
    monkeypatch.setenv("HOSTNAME", "roompool-rpt6h-r2677")
    monkeypatch.setattr("meshagent.agents.mail_common.socket.getfqdn", lambda: "")
    monkeypatch.setattr("meshagent.agents.mail_common.socket.gethostname", lambda: "")

    room = _FakeRoom()
    supervisor = _RecordingSupervisor()
    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
        domain="mail.meshagent.com",
        smtp=SmtpConfiguration(),
    )
    sent: dict[str, object] = {}

    async def fake_send(
        message,
        *,
        hostname,
        port,
        username,
        password,
        local_hostname,
    ):
        del message
        sent["hostname"] = hostname
        sent["port"] = port
        sent["username"] = username
        sent["password"] = password
        sent["local_hostname"] = local_hostname

    monkeypatch.setattr("meshagent.agents.mail_channel.aiosmtplib.send", fake_send)

    await channel.start(supervisor)  # type: ignore[arg-type]
    try:
        await channel.start_thread(
            thread_id=".threads/assistant/customer.thread",
            to_address="customer@example.com",
            subject="Follow up",
            body="Here is the update.",
        )
        assert sent == {
            "hostname": "mail.meshagent.com",
            "port": 587,
            "username": "assistant",
            "password": "token",
            "local_hostname": "roompool-rpt6h-r2677",
        }
    finally:
        await channel.stop(supervisor)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mail_channel_receive_loop_exits_quietly_after_room_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    room = _FakeRoom()
    room.is_closed = True

    async def _closed_receive(*, name: str, create: bool, wait: bool):
        del name
        del create
        del wait
        raise RoomException("room connection closed before request completed")

    room.queues.receive = _closed_receive  # type: ignore[method-assign]

    channel = MailChannel(
        room=room,
        queue_name="mailbox@mail.meshagent.com",
        email_address="mailbox@mail.meshagent.com",
    )

    with caplog.at_level(logging.ERROR, logger="mail-channel"):
        receive_task = asyncio.create_task(channel._receive_loop())
        await asyncio.wait_for(receive_task, timeout=1.0)

    assert receive_task.done()
    assert [
        record
        for record in caplog.records
        if record.name == "mail-channel" and record.levelno >= logging.ERROR
    ] == []
