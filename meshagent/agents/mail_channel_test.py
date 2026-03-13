import asyncio
import base64
from email.message import EmailMessage

import pytest

from meshagent.agents.mail_channel import MailChannel
from meshagent.agents.messages import (
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    TurnEnded,
    TurnStart,
    TurnStarted,
)
from meshagent.agents.process import Message
from meshagent.api import Participant
from meshagent.api.messaging import FileContent
from meshagent.api.room_server_client import RoomException
from meshagent.tools import ToolContext


class _FakeLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})

    async def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


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

    async def upload(self, *, path: str, data: bytes) -> None:
        self.uploaded[path] = data

    async def exists(self, *, path: str) -> bool:
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


class _FakeRoom:
    def __init__(self, *, database: _FakeDatabase | None = None) -> None:
        self.local_participant = _FakeLocalParticipant()
        self.protocol = _FakeProtocol()
        self.queues = _FakeQueues()
        self.storage = _FakeStorage()
        self.database = database if database is not None else _FakeDatabase()


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


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
        assert outbound.data.thread_id == ".threads/assistant/Quarterly update.thread"
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
        assert (
            stored_rows[0]["thread_id"] == ".threads/assistant/Quarterly update.thread"
        )
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

    async def fake_send(message, *, hostname, port, username, password):
        del message
        del hostname
        del port
        del username
        del password

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
        assert stored_rows[0]["thread_id"] == ".threads/helpdesk/Fallback path.thread"
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

    async def fake_send(message, *, hostname, port, username, password):
        del hostname
        del port
        del username
        del password
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

    async def fake_send(message, *, hostname, port, username, password):
        del hostname
        del port
        del username
        del password
        sent_messages.append(message)

    monkeypatch.setattr("meshagent.agents.mail_channel.aiosmtplib.send", fake_send)

    await channel.start(_RecordingSupervisor())  # type: ignore[arg-type]
    try:
        room.storage.uploaded["notes.txt"] = b"hello"
        toolkit = channel.get_agent_toolkits()[0]
        new_thread_tool = toolkit.tools[0]
        await new_thread_tool.execute(
            ToolContext(
                room=room,  # type: ignore[arg-type]
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
