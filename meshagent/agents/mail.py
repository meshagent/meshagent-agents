from meshagent.agents.worker import Worker
from meshagent.api.room_server_client import TextDataType
from email import message_from_bytes
from email.message import EmailMessage
from email.policy import default
import email.utils
from meshagent.api import ParticipantToken

from meshagent.api import RoomClient

from datetime import datetime, timezone

import secrets

from typing import Literal, Optional
import json

import uuid
import logging

import os
import aiosmtplib

logger = logging.getLogger("mail")

type MessageRole = Literal["user", "agent"]


def create_reply_email_message(
    *, message: dict, from_address: str, body: str
) -> EmailMessage:
    subject: str = message.get("subject")

    if not subject.startswith("RE:"):
        subject = "RE: " + subject

    _, addr = email.utils.parseaddr(from_address)
    domain = addr.split("@")[-1].lower()
    id = f"<{uuid.uuid4()}@{domain}>"

    msg = EmailMessage()
    msg["Message-ID"] = id
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = message.get("reply_to")
    msg["In-Reply-To"] = message.get("id")
    msg.set_content(body)

    return msg


def message_to_json(*, message: EmailMessage, role: MessageRole):
    body_part = message.get_body(
        ("plain", "html")
    )  # returns the “best” part :contentReference[oaicite:0]{index=0}
    if body_part:
        body = body_part.get_content()
    else:  # simple, non-MIME message
        body = message.get_content()

    id = message.get("Message-ID")
    if id is None:
        mfrom = message.get("From")
        _, addr = email.utils.parseaddr(mfrom)
        domain = addr.split("@")[-1].lower()
        id = f"{uuid.uuid4()}@{domain}"

    return {
        "id": id,
        "in_reply_to": message.get("In-Reply-To"),
        "reply_to": message.get("Reply-To", message.get("From")),
        "references": message.get("References"),
        "from": message.get("From"),
        "to": message.get_all("To"),
        "subject": message.get("Subject"),
        "body": body,
        "attachments": [],
        "role": role,
    }


async def load_message(*, room: RoomClient, message_id: str) -> dict | None:
    messages = await room.database.search(table="emails", where={"id": message_id})

    if len(messages) == 0:
        return None

    return json.loads(messages[0]["json"])


async def save_email_message(
    *, room: RoomClient, content: bytes, role: MessageRole
) -> dict:
    message = message_from_bytes(content, policy=default)

    now = datetime.now(timezone.utc)

    folder_path = (
        now.strftime("%Y/%m/%d")
        + "/"
        + now.strftime("%H/%M/%S")
        + "/"
        + secrets.token_hex(3)
    )

    queued_message = message_to_json(message=message, role=role)
    message_id = queued_message["id"]

    queued_message["role"] = role

    queued_message["path"] = f".emails/{message_id}/message.json"

    for part in (
        message.iter_attachments()
    ):  # ↔ only the “real” attachments :contentReference[oaicite:0]{index=0}
        fname = (
            part.get_filename() or "attachment.bin"
        )  # RFC 2183 filename, if any :contentReference[oaicite:1]{index=1}

        # get_content() auto-decodes transfer-encodings; returns
        # *str* for text/*, *bytes* for everything else :contentReference[oaicite:2]{index=2}
        data = part.get_content()

        # make sure we write binary data
        bin_data = (
            data.encode(part.get_content_charset("utf-8"))
            if isinstance(data, str)
            else data
        )

        path = f".emails/{folder_path}/attachments/{fname}"
        handle = await room.storage.open(path=path)
        try:
            logger.info(f"writing content to {path}")
            await room.storage.write(handle=handle, data=bin_data)
        finally:
            await room.storage.close(handle=handle)

        queued_message["attachments"].append(path)

    logger.info(f"received mail, {queued_message}")

    # write email
    path = f".emails/{folder_path}/message.eml"
    handle = await room.storage.open(path=path)
    try:
        logger.info(f"writing source message.eml to {path}")
        await room.storage.write(handle=handle, data=content)
    finally:
        await room.storage.close(handle=handle)

    path = f".emails/{folder_path}/message.json"
    handle = await room.storage.open(path=path)
    try:
        logger.info(f"writing source message.json to {path}")
        await room.storage.write(
            handle=handle, data=json.dumps(queued_message, indent=4).encode("utf-8")
        )
    finally:
        await room.storage.close(handle=handle)

    # create email table if it doesn't exist
    tables = await room.database.list_tables()

    if "emails" not in tables:
        await room.database.create_table_with_schema(
            name="emails",
            schema={"id": TextDataType(), "json": TextDataType()},
            mode="create_if_not_exists",
        )

        await room.database.create_scalar_index(table="emails", column="id")

    await room.database.insert(
        table="emails", records=[{"id": message_id, "json": json.dumps(queued_message)}]
    )

    return queued_message


async def load_thread(*, room: RoomClient, message: dict, thread: list[dict]):
    in_reply_to = message.get("in_reply_to", None)
    if in_reply_to is not None:
        source = await load_message(room=room, message_id=in_reply_to)

        if source is not None:
            thread.insert(0, source)

            await load_thread(room=room, message=source, thread=thread)

        else:
            logger.warning(f"message not found {in_reply_to}")


class SmtpConfiguration:
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: Optional[int] = None,
        hostname: Optional[str] = None,
    ):
        if username is None:
            username = os.getenv("SMTP_USERNAME")

        if password is None:
            password = os.getenv("SMTP_PASSWORD")

        if port is None:
            port = int(os.getenv("SMTP_PORT", "587"))

        if hostname is None:
            hostname = os.getenv("SMTP_HOSTNAME")

        self.username = username
        self.password = password
        self.port = port
        self.hostname = hostname


class MailWorker(Worker):
    def __init__(
        self,
        *,
        queue: str = "email",
        name,
        title=None,
        description=None,
        requires=None,
        llm_adapter,
        tool_adapter=None,
        toolkits=None,
        rules=None,
        domain: str = os.getenv("MESHAGENT_MAIL_DOMAIN", "mail.meshagent.com"),
        smtp: Optional[SmtpConfiguration] = None,
    ):
        if smtp is None:
            smtp = SmtpConfiguration()

        self._domain = domain
        self._smtp = smtp
        super().__init__(
            queue=queue,
            name=name,
            title=title,
            description=description,
            requires=requires,
            llm_adapter=llm_adapter,
            tool_adapter=tool_adapter,
            toolkits=toolkits,
            rules=rules,
        )

    async def start(self, *, room):
        await super().start(room=room)

        token = ParticipantToken.from_jwt(room.protocol.token, validate=False)
        self._email_address = room_address(
            project_id=token.project_id, room_name=room.room_name, domain=self._domain
        )

    async def append_message_context(self, *, room, message, chat_context):
        thread = [message]

        await load_thread(room=room, message=message, thread=thread)

        for msg in thread:
            if msg["role"] == "agent":
                chat_context.append_assistant_message(json.dumps(msg))

            else:
                chat_context.append_user_message(json.dumps(msg))

        # TODO: load previous messages
        return await super().append_message_context(
            room=room, message=message, chat_context=chat_context
        )

    async def process_message(self, *, chat_context, room, message, toolkits):
        logger.info(f"processing message {message}")
        body = await super().process_message(
            chat_context=chat_context, room=room, message=message, toolkits=toolkits
        )

        msg = create_reply_email_message(
            message=message, from_address=self._email_address, body=body
        )

        reply_msg_dict = await save_email_message(
            room=room, content=msg.as_bytes(), role="agent"
        )

        logger.info(f"replying with message {reply_msg_dict}")

        username = self._smtp.username
        if username is None:
            username = self.room.local_participant.get_attribute("name")

        password = self._smtp.password
        if password is None:
            password = self.room.protocol.token

        await aiosmtplib.send(
            msg,
            hostname=self._smtp.hostname,
            port=self._smtp.port,
            username=username,
            password=password,
        )


def base36encode(number: int):
    if not isinstance(number, int):
        raise TypeError("number must be an integer")
    if number < 0:
        raise ValueError("number must be non-negative")

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if number == 0:
        return "0"
    base36 = ""
    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36
    return base36


def compress_uuid(guid_string: str):
    guid_int = int(guid_string.replace("-", ""), 16)
    return base36encode(guid_int)


def base36decode(number_str: str) -> int:
    """Decode a base36-encoded string into an integer."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    base = 36
    number = 0
    for char in number_str:
        try:
            value = alphabet.index(char)
        except ValueError:
            raise ValueError(f"Invalid character '{char}' for base36 encoding")
        number = number * base + value
    return number


def decompress_uuid(compressed_uuid: str) -> str:
    """
    Reverse the compressed UUID to its standard 36-character UUID format.

    Args:
        compressed_uuid: A base36 string that represents a UUID compressed from its standard form.

    Returns:
        A string in the UUID format (8-4-4-4-12 hexadecimal characters).
    """
    # Decode the base36 string back to the original integer.
    guid_int = base36decode(compressed_uuid)

    # Convert the integer into a 32-digit hexadecimal string with leading zeros.
    hex_str = f"{guid_int:032x}"

    # Reinsert dashes to match the standard UUID format: 8-4-4-4-12.
    standard_uuid = f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
    return standard_uuid


def room_address(
    *,
    project_id: str,
    room_name: str,
    domain: str = os.getenv("MESHAGENT_MAIL_DOMAIN", "mail.meshagent.com"),
):
    return f"{compress_uuid(project_id)}+{room_name}@{domain}"
