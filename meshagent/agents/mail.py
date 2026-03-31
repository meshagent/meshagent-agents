from meshagent.agents.worker import Worker
from meshagent.tools import (
    RemoteToolkit,
    ToolContext,
    FunctionTool,
    Toolkit,
    FileContent,
)
from meshagent.tools.storage import StorageToolkit
from meshagent.api.room_server_client import TextDataType, RoomException
from email import message_from_bytes
from email.message import EmailMessage
from meshagent.api import RoomClient
from meshagent.api import RequiredTable
from email.policy import default
from meshagent.agents import AgentSessionContext
from datetime import datetime, timezone
import base64
import secrets

from typing import Literal, Optional
import json

import logging

import os

from pathlib import Path
from . import mail_common
from meshagent.agents.skills import to_prompt

logger = logging.getLogger("mail")


class _MissingAioSmtplib:
    async def send(self, *args: object, **kwargs: object) -> None:
        del args
        del kwargs
        raise ModuleNotFoundError(
            "aiosmtplib is required to use mail agent SMTP features"
        )


try:
    import aiosmtplib
except ModuleNotFoundError:
    aiosmtplib = _MissingAioSmtplib()

type MessageRole = Literal["user", "agent"]


class MailThreadContext:
    def __init__(self, *, chat: AgentSessionContext, message: dict, thread: list[dict]):
        self.chat = chat
        self.message = message
        self.thread = thread


SmtpConfiguration = mail_common.SmtpConfiguration


class NewEmailThreadWithAttachments(FunctionTool):
    def __init__(self, *, agent: "MailBot"):
        self.agent = agent
        super().__init__(
            name="new_email_thread",
            title="New Email Thread",
            description="Starts a new email thread that is managed by the mailbot",
            input_schema={
                "type": "object",
                "required": ["to", "body", "subject", "attachments"],
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
                    "attachments": {
                        "type": "array",
                        "description": "a list of paths from the room's storage of files to attach",
                        "items": {"type": "string"},
                    },
                },
            },
        )

    async def execute(
        self,
        context: ToolContext,
        *,
        to: str,
        subject: str,
        body: str,
        attachments: list[str],
    ):
        attachment_data = list[FileContent]()

        for attachment in attachments:
            try:
                attachment_data.append(
                    await context.room.storage.download(path=attachment)
                )
            except Exception as ex:
                logger.error(f"Unable to download file {ex}", exc_info=ex)
                raise RoomException(
                    f"Could not download a file from the room with the path {attachment}. Are you sure the path is correct file?"
                )

        await self.agent.start_thread(
            to_address=to, subject=subject, body=body, attachments=attachment_data
        )
        return {}


class NewEmailThread(FunctionTool):
    def __init__(self, *, agent: "MailBot"):
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

    async def execute(
        self,
        context: ToolContext,
        *,
        to: str,
        subject: str,
        body: str,
    ):
        await self.agent.start_thread(
            to_address=to, subject=subject, body=body, attachments=[]
        )
        return {}


class MailBot(Worker):
    def __init__(
        self,
        *,
        queue: Optional[str] = None,
        name=None,
        title=None,
        description=None,
        requires=None,
        llm_adapter,
        toolkits=None,
        rules=None,
        email_address: str,
        domain: str = os.getenv("MESHAGENT_MAIL_DOMAIN", "mail.meshagent.com"),
        smtp: Optional[SmtpConfiguration] = None,
        toolkit_name: Optional[str] = None,
        whitelist: Optional[list[str]] = None,
        reply_all: bool = False,
        enable_attachments: bool = True,
        skill_dirs: Optional[list[str]] = None,
        annotations: Optional[list[str]] = None,
    ):
        if smtp is None:
            smtp = SmtpConfiguration()

        if queue is None:
            queue = email_address

        self._domain = domain
        self._smtp = smtp
        self._reply_all = reply_all
        self._enable_attachments = enable_attachments

        super().__init__(
            queue=queue,
            name=name,
            title=title,
            description=description,
            requires=requires,
            llm_adapter=llm_adapter,
            toolkits=toolkits,
            annotations=annotations,
            rules=rules
            or [
                "You MUST reply with plain text or markdown, do not reply in JSON format or HTML format"
            ],
        )
        self._email_address = email_address
        self._whitelist = whitelist

        if toolkit_name is not None:
            logger.info(f"mailbox will start toolkit {toolkit_name}")
            self._toolkit = RemoteToolkit(
                name=toolkit_name,
                tools=[
                    NewEmailThreadWithAttachments(agent=self)
                    if enable_attachments
                    else NewEmailThread(agent=self),
                ],
            )
        else:
            self._toolkit = None

        self._skill_dirs = skill_dirs

    def get_requirements(self):
        return [
            *super().get_requirements(),
            RequiredTable(
                name="emails",
                schema={"id": TextDataType(), "json": TextDataType()},
                scalar_indexes=["id"],
            ),
        ]

    async def load_message(self, *, message_id: str) -> dict | None:
        room = self.room
        messages = await room.database.search(table="emails", where={"id": message_id})

        if len(messages) == 0:
            return None

        return json.loads(messages[0]["json"])

    def message_to_json(self, *, message: EmailMessage, role: "MessageRole") -> dict:
        return mail_common.message_to_json(message=message, role=role)

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

        for fname, bin_data in mail_common.iter_message_attachments(message):
            path = f".emails/{folder_path}/attachments/{fname}"
            logger.info(f"writing content to {path}")
            await room.storage.upload(path=path, data=bin_data)

            queued_message["attachments"].append(path)

        logger.info(f"received mail, {queued_message}")

        # write email
        path = f".emails/{folder_path}/message.eml"
        logger.info(f"writing source message.eml to {path}")
        await room.storage.upload(path=path, data=content)

        path = f".emails/{folder_path}/message.json"
        logger.info(f"writing source message.json to {path}")
        await room.storage.upload(
            path=path,
            data=json.dumps(queued_message, indent=4).encode("utf-8"),
        )

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
        chat_context: AgentSessionContext,
        thread: list[dict],
    ):
        for msg in thread:
            if msg["role"] == "agent":
                chat_context.append_assistant_message(json.dumps(msg))

            else:
                chat_context.append_user_message(json.dumps(msg))

    async def get_rules(self):
        rules = [*self._rules]

        if self._skill_dirs is not None and len(self._skill_dirs) > 0:
            rules.append(
                "You have access to to following skills which follow the agentskills spec:"
            )
            rules.append(await to_prompt([*(Path(p) for p in self._skill_dirs)]))
            rules.append(
                "Use the shell or storage tool to find out more about skills and execute them when they are required"
            )

        return rules

    async def should_reply(self, *, message: dict) -> bool:
        should_reply, reason = mail_common.should_reply_to_message(
            message=message,
            email_address=self._email_address,
            whitelist=self._whitelist,
        )
        if not should_reply and reason == "message is not addressed to this mailbox":
            logger.warn(
                f"message not addressed to {self._email_address}, message will be ignored by the mailbot; "
                f"to={message.get('to')!r} cc={message.get('cc')!r} delivered_to={((message.get('meta') or {}).get('delivered_to') or '')!r}"
            )
            return False
        if not should_reply and reason is not None:
            logger.info(f"discarding message (error/auto-reply): {reason}")
            return False
        return should_reply

    async def process_message(
        self,
        *,
        chat_context: AgentSessionContext,
        message: dict,
        toolkits: list[Toolkit],
    ):
        logger.info("received a mail message")

        rules = await self.get_rules()

        logger.info(f"using rules {rules}")

        chat_context.replace_rules(rules)

        message_bytes = base64.b64decode(message["base64"])

        message = await self.save_email_message(content=message_bytes, role="user")

        if not await self.should_reply(message=message):
            return

        thread = [message]

        await self.load_thread(message=message, thread=thread)

        await self.append_message_context(
            message=message, chat_context=chat_context, thread=thread
        )

        thread_context = MailThreadContext(
            chat=chat_context, message=message, thread=thread
        )
        toolkits = await self.get_thread_toolkits(thread_context=thread_context)

        attachment_data = []

        try:
            if self._enable_attachments:

                class AttachTool(FunctionTool):
                    def __init__(self):
                        super().__init__(
                            name="attach file",
                            description="attach a file from the room to the conversation",
                            input_schema={
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["path"],
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "a path to a file in the room's storage",
                                    }
                                },
                            },
                        )

                    async def execute(self, context: ToolContext, *, path: str):
                        try:
                            storage_toolkits = [
                                t for t in toolkits if isinstance(t, StorageToolkit)
                            ]

                            if len(storage_toolkits) > 0:
                                attachment_data.append(
                                    await storage_toolkits[0].read_file(
                                        context=context, path=path
                                    )
                                )
                            else:
                                attachment_data.append(
                                    await context.room.storage.download(path=path)
                                )
                        except Exception as ex:
                            logger.error(f"Unable to download file {ex}", exc_info=ex)
                            raise RoomException(
                                f"Could not download a file from the room with the path {path}. Are you sure the path is correct file?"
                            )

                toolkits = [
                    *toolkits,
                    Toolkit(name="attachments", tools=[AttachTool()]),
                ]

            reply = await self._llm_adapter.next(
                context=chat_context,
                room=self.room,
                toolkits=toolkits,
            )
        except Exception as ex:
            logger.error(f"error while processing message {ex}", exc_info=ex)
            reply = "An error occurred while processing your message, please try again."

        logger.info(f"replying: {reply}")

        return await self.send_reply_message(
            message=message, reply=reply, attachments=attachment_data
        )

    def render_markdown(self, body: str):
        return mail_common.render_mail_markdown(body=body)

    def create_email_message(
        self,
        *,
        to_address: str,
        from_address: str,
        subject: str,
        body: str,
        correlation_id: Optional[str] = None,
    ) -> EmailMessage:
        return mail_common.create_email_message(
            to_address=to_address,
            from_address=from_address,
            subject=subject,
            body=body,
            correlation_id=correlation_id,
        )

    async def start(self, *, room: RoomClient):
        await super().start(room=room)
        if self._toolkit is not None:
            await self._toolkit.start(room=room)

    async def stop(self):
        if self._toolkit is not None:
            await self._toolkit.stop()
        await super().stop()

    async def start_thread(
        self,
        *,
        to_address: str,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
        attachments: Optional[list[FileContent]] = None,
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

        if attachments is not None:
            reply_msg_dict["attachments"] = [*(x.name for x in attachments)]

            for attachment in attachments:
                maintype, subtype = attachment.mime_type.split("/")
                msg.add_attachment(
                    attachment.data,
                    maintype=maintype,
                    subtype=subtype,
                    filename=attachment.name,
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
            local_hostname=self._smtp.effective_local_hostname(),
        )

    def create_reply_email_message(
        self,
        *,
        message: dict,
        from_address: str,
        body: str,
        reply_all: bool = False,  # <-- choose behavior
    ) -> EmailMessage:
        return mail_common.create_reply_email_message(
            message=message,
            from_address=from_address,
            body=body,
            email_address=self._email_address,
            reply_all=reply_all,
        )

    async def send_reply_message(
        self,
        *,
        message: dict,
        reply: str,
        attachments: Optional[list[FileContent]] = None,
    ):
        msg = self.create_reply_email_message(
            message=message,
            from_address=self._email_address,
            body=reply,
            reply_all=self._reply_all,
        )

        reply_msg_dict = await self.save_email_message(
            content=msg.as_bytes(), role="agent"
        )

        if attachments is not None:
            reply_msg_dict["attachments"] = [*(x.name for x in attachments)]

            for attachment in attachments:
                maintype, subtype = attachment.mime_type.split("/")
                msg.add_attachment(
                    attachment.data,
                    maintype=maintype,
                    subtype=subtype,
                    filename=attachment.name,
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
            local_hostname=self._smtp.effective_local_hostname(),
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
