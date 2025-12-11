from meshagent.agents.worker import Worker
from meshagent.tools import RemoteToolkit, ToolContext, Tool, Toolkit
from meshagent.api.room_server_client import TextDataType
from email import message_from_bytes
from email.message import EmailMessage
from meshagent.api import RoomClient
from email.policy import default
import email.utils
from meshagent.agents import AgentChatContext
from datetime import datetime, timezone
import base64
import secrets

from typing import Literal, Optional
import json

import uuid
import logging

import os
import aiosmtplib

import mistune

logger = logging.getLogger("mail")

type MessageRole = Literal["user", "agent"]


class MailThreadContext:
    def __init__(self, *, chat: AgentChatContext, message: dict, thread: list[dict]):
        self.chat = chat
        self.message = message
        self.thread = thread


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


class NewEmailThread(Tool):
    def __init__(self, *, agent: "MailWorker"):
        self.agent = agent
        super().__init__(
            name="new_email_thread",
            title="New Email Thread",
            description="Starts a new email thread that is managed by the mailbot",
            input_schema={
                "type": "object",
                "required": ["to", "body", "subject"],
                "additionalProperties": False,
                "properties": {
                    "to": {
                        "type": "string",
                    },
                    "subject": {
                        "type": "string",
                    },
                    "body": {
                        "type": "string",
                    },
                },
            },
        )

    async def execute(self, context: ToolContext, *, to: str, subject: str, body: str):
        await self.agent.start_thread(to_address=to, subject=subject, body=body)
        return {}


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
        email_address: str,
        domain: str = os.getenv("MESHAGENT_MAIL_DOMAIN", "mail.meshagent.com"),
        smtp: Optional[SmtpConfiguration] = None,
        toolkit_name: Optional[str] = None,
        whitelist: Optional[list[str]] = None,
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
            rules=rules
            or [
                "You MUST reply with plain text, do not reply in JSON format or HTML format"
            ],
        )
        self._email_address = email_address
        self._whitelist = whitelist

        if toolkit_name is not None:
            logger.info(f"mailbox will start toolkit {toolkit_name}")
            self._toolkit = RemoteToolkit(
                name=toolkit_name,
                tools=[
                    NewEmailThread(agent=self),
                ],
            )
        else:
            self._toolkit = None

    async def load_message(self, *, message_id: str) -> dict | None:
        room = self.room
        messages = await room.database.search(table="emails", where={"id": message_id})

        if len(messages) == 0:
            return None

        return json.loads(messages[0]["json"])

    def message_to_json(self, *, message: EmailMessage, role: MessageRole):
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
            "correlation_id": message.get("Meshagent-Correlation-ID"),
        }

    async def save_email_message(self, *, content: bytes, role: MessageRole) -> dict:
        room = self.room
        message = message_from_bytes(content, policy=default)

        now = datetime.now(timezone.utc)

        folder_path = (
            now.strftime("%Y/%m/%d")
            + "/"
            + now.strftime("%H/%M/%S")
            + "/"
            + secrets.token_hex(3)
        )

        queued_message = self.message_to_json(message=message, role=role)
        message_id = queued_message["id"]

        queued_message["role"] = role

        queued_message["path"] = f".emails/{folder_path}/message.json"

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
            table="emails",
            records=[{"id": message_id, "json": json.dumps(queued_message)}],
        )

        return queued_message

    async def load_thread(self, *, message: dict, thread: list[dict]):
        in_reply_to = message.get("in_reply_to", None)
        if in_reply_to is not None:
            source = await self.load_message(message_id=in_reply_to)

            if source is not None:
                thread.insert(0, source)

                await self.load_thread(message=source, thread=thread)

            else:
                logger.warning(f"message not found {in_reply_to}")

    async def append_message_context(
        self,
        *,
        message: dict,
        chat_context: AgentChatContext,
        thread: list[dict],
    ):
        for msg in thread:
            if msg["role"] == "agent":
                chat_context.append_assistant_message(json.dumps(msg))

            else:
                chat_context.append_user_message(json.dumps(msg))

        # TODO: load previous messages
        return await super().append_message_context(
            message=message, chat_context=chat_context
        )

    async def get_rules(self):
        return [*self._rules]

    async def process_message(
        self,
        *,
        chat_context: AgentChatContext,
        message: dict,
        toolkits: list[Toolkit],
    ):
        logger.info("received a mail message")

        rules = await self.get_rules()

        logger.info(f"using rules {rules}")

        chat_context.replace_rules(rules)

        message_bytes = base64.b64decode(message["base64"])

        if self._whitelist is not None:
            message = message_from_bytes(message_bytes, policy=default)
            from_address = message["From"]
            _, addr = email.utils.parseaddr(from_address)

            if addr.casefold() not in self._whitelist:
                logger.info(
                    f"{from_address} not found in whitelist, discarding message"
                )
                return

        message = await self.save_email_message(content=message_bytes, role="user")

        thread = [message]

        await self.load_thread(message=message, thread=thread)

        await self.append_message_context(
            message=message, chat_context=chat_context, thread=thread
        )

        thread_context = MailThreadContext(
            chat=chat_context, message=message, thread=thread
        )
        toolkits = await self.get_thread_toolkits(thread_context=thread_context)

        reply = await self._llm_adapter.next(
            context=chat_context,
            room=self.room,
            toolkits=toolkits,
            tool_adapter=self._tool_adapter,
        )

        logger.info(f"replying: {reply}")

        return await self.send_reply_message(message=message, reply=reply)

    def render_markdown(self, body: str):
        markdown = mistune.create_markdown()
        return markdown(body)

    def create_email_message(
        self,
        *,
        to_address: str,
        from_address: str,
        subject: str,
        body: str,
        correlation_id: Optional[str] = None,
    ) -> EmailMessage:
        _, addr = email.utils.parseaddr(from_address)
        domain = addr.split("@")[-1].lower()
        id = f"<{uuid.uuid4()}@{domain}>"

        msg = EmailMessage()
        msg["Message-ID"] = id
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = to_address
        if correlation_id is not None:
            msg["Meshagent-Correlation-ID"] = correlation_id

        msg.set_content(body)

        msg.add_alternative(self.render_markdown(body), subtype="html")

        return msg

    async def start(self, *, room: RoomClient):
        await super().start(room=room)
        await self._toolkit.start(room=room)

    async def stop(self):
        await self._toolkit.stop()
        await super().stop()

    async def start_thread(
        self,
        *,
        to_address: str,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ):
        msg = self.create_email_message(
            to_address=to_address,
            from_address=from_address or self._email_address,
            subject=subject,
            body=body,
        )

        reply_msg_dict = await self.save_email_message(
            content=msg.as_bytes(), role="agent"
        )

        logger.info(f"starting thread with message {reply_msg_dict}")

        username = self._smtp.username
        if username is None:
            username = self.room.local_participant.get_attribute("name")

        password = self._smtp.password
        if password is None:
            password = self.room.protocol.token

        hostname = self._smtp.hostname
        if hostname is None:
            hostname = self._domain

        port = self._smtp.port

        logger.info(f"using smtp {username}@{hostname}:{port}")

        await aiosmtplib.send(
            msg,
            hostname=hostname,
            port=port,
            username=username,
            password=password,
        )

    def create_reply_email_message(
        self, *, message: dict, from_address: str, body: str
    ) -> EmailMessage:
        subject: str = message.get("subject") or ""

        if not subject.lower().startswith("re:"):
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
        correlation_id = message.get("correlation_id")
        if correlation_id is not None:
            msg["Meshagent-Correlation-ID"] = correlation_id

        msg.set_content(body)
        msg.add_alternative(self.render_markdown(body), subtype="html")

        return msg

    async def send_reply_message(self, *, message: dict, reply: str):
        msg = self.create_reply_email_message(
            message=message, from_address=self._email_address, body=reply
        )

        reply_msg_dict = await self.save_email_message(
            content=msg.as_bytes(), role="agent"
        )

        logger.info(f"replying with message {reply_msg_dict}")

        username = self._smtp.username
        if username is None:
            username = self.room.local_participant.get_attribute("name")

        password = self._smtp.password
        if password is None:
            password = self.room.protocol.token

        hostname = self._smtp.hostname
        if hostname is None:
            hostname = self._domain

        port = self._smtp.port

        logger.info(f"using smtp {username}@{hostname}:{port}")

        await aiosmtplib.send(
            msg,
            hostname=hostname,
            port=port,
            username=username,
            password=password,
        )

    async def get_thread_toolkits(
        self,
        *,
        thread_context: MailThreadContext,
    ) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                room=self.room,
                caller=self.room.local_participant,
                caller_context={"chat": thread_context.chat.to_json()},
            )
        )

        return [*self._toolkits, *toolkits]
