from __future__ import annotations

import asyncio
import base64
import contextlib
from dataclasses import dataclass, field
from email import message_from_bytes
import email.utils
import json
import logging
import os
import posixpath
import re
from typing import Any
import uuid

from meshagent.api import Participant, RoomClient, RoomException
from meshagent.api.room_server_client import TextDataType
from meshagent.tools import FileContent, FunctionTool, ToolContext, Toolkit
from email.policy import default

from .legacy_chat_channel import LegacyChatChannel
from .mail_common import (
    SmtpConfiguration,
    create_email_message,
    create_reply_email_message,
    iter_message_attachments,
    message_to_json,
    should_reply_to_message,
)
from .messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_TURN_START,
    AgentFileContent,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    ThreadCleared,
    TurnEnded,
    TurnStart,
    TurnStarted,
)
from .process import Channel, Message

logger = logging.getLogger("mail-channel")


class _MissingAioSmtplib:
    async def send(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs
        raise ModuleNotFoundError(
            "aiosmtplib is required to use MailChannel SMTP features"
        )


try:
    import aiosmtplib
except ModuleNotFoundError:
    aiosmtplib = _MissingAioSmtplib()

_MAIL_TABLE_NAME = "emails"
_MAIL_STORAGE_ROOT = ".emails"


@dataclass(slots=True)
class _PendingInboundMail:
    thread_id: str
    message: dict[str, Any]


@dataclass(slots=True)
class _ActiveMailTurn:
    thread_id: str
    source_message: dict[str, Any]
    text_by_item_id: dict[str, str] = field(default_factory=dict)
    completed_text_parts: list[str] = field(default_factory=list)


class MailChannel(Channel):
    def __init__(
        self,
        *,
        room: RoomClient,
        queue_name: str,
        email_address: str,
        thread_dir: str | None = None,
        domain: str = os.getenv("MESHAGENT_MAIL_DOMAIN", "mail.meshagent.com"),
        smtp: SmtpConfiguration | None = None,
        whitelist: list[str] | None = None,
        reply_all: bool = False,
        enable_attachments: bool = True,
    ) -> None:
        super().__init__()
        normalized_queue = queue_name.strip()
        if normalized_queue == "":
            raise ValueError("queue_name must not be empty")

        normalized_email = email_address.strip()
        if normalized_email == "":
            raise ValueError("email_address must not be empty")

        self._room = room
        self._queue_name = normalized_queue
        self._email_address = normalized_email
        self._thread_dir = LegacyChatChannel._normalize_thread_dir(
            thread_dir=thread_dir
        )
        self._domain = domain
        self._smtp = smtp if smtp is not None else SmtpConfiguration()
        self._whitelist = list(whitelist) if whitelist is not None else None
        self._reply_all = reply_all
        self._enable_attachments = enable_attachments
        self._receive_task: asyncio.Task[None] | None = None
        self._pending_messages_by_source_id: dict[str, _PendingInboundMail] = {}
        self._active_turns_by_turn_id: dict[str, _ActiveMailTurn] = {}
        self._database_namespace_supported: bool = True

    @property
    def room(self) -> RoomClient:
        return self._room

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_EVENT_TEXT_CONTENT_STARTED,
            AGENT_EVENT_TEXT_CONTENT_DELTA,
            AGENT_EVENT_TEXT_CONTENT_ENDED,
            AGENT_EVENT_TURN_STARTED,
            AGENT_EVENT_TURN_ENDED,
            AGENT_EVENT_THREAD_CLEARED,
        }

    def get_agent_toolkits(self) -> list[Toolkit]:
        return [Toolkit(name="mail", tools=[self._make_new_email_thread_tool()])]

    async def on_start(self) -> None:
        await self._ensure_database_table()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def on_stop(self) -> None:
        receive_task = self._receive_task
        self._receive_task = None
        if receive_task is not None:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task
        self._pending_messages_by_source_id.clear()
        self._active_turns_by_turn_id.clear()

    async def on_message(self, message: Message) -> None:
        data = message.data
        if isinstance(data, TurnStarted):
            pending = self._pending_messages_by_source_id.pop(
                data.source_message_id,
                None,
            )
            if pending is None:
                return
            self._active_turns_by_turn_id[data.turn_id] = _ActiveMailTurn(
                thread_id=pending.thread_id,
                source_message=pending.message,
            )
            return

        if isinstance(data, AgentTextContentStarted):
            active = self._active_turns_by_turn_id.get(data.turn_id)
            if active is None:
                return
            active.text_by_item_id[data.item_id] = ""
            return

        if isinstance(data, AgentTextContentDelta):
            active = self._active_turns_by_turn_id.get(data.turn_id)
            if active is None:
                return
            current_text = active.text_by_item_id.get(data.item_id, "")
            active.text_by_item_id[data.item_id] = current_text + data.text
            return

        if isinstance(data, AgentTextContentEnded):
            active = self._active_turns_by_turn_id.get(data.turn_id)
            if active is None:
                return
            completed_text = active.text_by_item_id.pop(data.item_id, "")
            if completed_text.strip() != "":
                active.completed_text_parts.append(completed_text)
            return

        if isinstance(data, ThreadCleared):
            self._clear_thread_state(thread_id=data.thread_id)
            return

        if not isinstance(data, TurnEnded):
            return

        active = self._active_turns_by_turn_id.pop(data.turn_id, None)
        if active is None:
            return

        if data.error is not None:
            return

        reply = self._reply_text_for_turn(active=active)
        if reply == "":
            return

        try:
            await self.send_reply_message(
                thread_id=active.thread_id,
                source_message=active.source_message,
                reply=reply,
            )
        except Exception:
            logger.exception(
                "failed sending email reply for thread %s", active.thread_id
            )

    async def start_thread(
        self,
        *,
        thread_id: str,
        to_address: str,
        subject: str,
        body: str,
        attachments: list[FileContent] | None = None,
    ) -> dict[str, Any]:
        message = create_email_message(
            to_address=to_address,
            from_address=self._email_address,
            subject=subject,
            body=body,
        )
        for attachment in attachments or []:
            maintype, subtype = attachment.mime_type.split("/", 1)
            message.add_attachment(
                attachment.data,
                maintype=maintype,
                subtype=subtype,
                filename=attachment.name,
            )

        saved_message = await self._save_email_message(
            content=message.as_bytes(),
            role="agent",
            thread_id=thread_id,
        )
        await self._smtp_send(message=message)
        return saved_message

    async def send_reply_message(
        self,
        *,
        thread_id: str,
        source_message: dict[str, Any],
        reply: str,
    ) -> dict[str, Any]:
        message = create_reply_email_message(
            message=source_message,
            from_address=self._email_address,
            body=reply,
            email_address=self._email_address,
            reply_all=self._reply_all,
        )
        saved_message = await self._save_email_message(
            content=message.as_bytes(),
            role="agent",
            thread_id=thread_id,
        )
        await self._smtp_send(message=message)
        return saved_message

    async def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                queued_message = await self._room.queues.receive(
                    name=self._queue_name,
                    create=True,
                    wait=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "mail queue receive failed for queue %s",
                    self._queue_name,
                )
                await asyncio.sleep(1)
                continue

            if queued_message is None:
                continue

            try:
                await self._handle_inbound_queue_message(queued_message=queued_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "mail channel failed to process inbound queue message for %s",
                    self._queue_name,
                )

    async def _handle_inbound_queue_message(self, *, queued_message: Any) -> None:
        if not isinstance(queued_message, dict):
            logger.warning(
                "ignoring mail queue payload with unsupported type %s",
                type(queued_message),
            )
            return

        encoded_message = queued_message.get("base64")
        if not isinstance(encoded_message, str) or encoded_message.strip() == "":
            logger.warning("ignoring mail queue payload without base64 email content")
            return

        raw_bytes = base64.b64decode(encoded_message)
        email_message = message_from_bytes(raw_bytes, policy=default)
        parsed_message = message_to_json(message=email_message, role="user")

        should_reply, reason = should_reply_to_message(
            message=parsed_message,
            email_address=self._email_address,
            whitelist=self._whitelist,
        )
        if not should_reply:
            logger.info(
                "discarding inbound mail for %s: %s",
                self._queue_name,
                reason or "message is not replyable",
            )
            return

        thread_id = await self._resolve_thread_id(message=parsed_message)
        saved_message = await self._save_email_message(
            content=raw_bytes,
            role="user",
            thread_id=thread_id,
        )

        turn_start = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id=thread_id,
            content=self._content_from_mail_message(message=saved_message),
        )
        self._pending_messages_by_source_id[turn_start.message_id] = (
            _PendingInboundMail(
                thread_id=thread_id,
                message=saved_message,
            )
        )
        self.emit(
            sender=self._participant_for_message(message=saved_message),
            payload=turn_start,
        )

    async def _ensure_database_table(self) -> None:
        table_name, namespace = self._database_table_ref()
        try:
            await self._room.database.create_table_with_schema(
                name=table_name,
                schema={
                    "id": TextDataType(nullable=False),
                    "thread_id": TextDataType(nullable=False),
                    "in_reply_to": TextDataType(),
                    "role": TextDataType(nullable=False),
                    "json": TextDataType(nullable=False),
                },
                mode="create_if_not_exists",
                namespace=namespace,
            )
        except RoomException as exc:
            if (
                namespace is None
                or not self._is_namespace_location_error(exc=exc)
                or not self._database_namespace_supported
            ):
                raise

            self._database_namespace_supported = False
            fallback_table_name, fallback_namespace = self._database_table_ref()
            logger.warning(
                "mail channel falling back to flat database table %s because the room server rejected namespaced table creation",
                fallback_table_name,
            )
            await self._room.database.create_table_with_schema(
                name=fallback_table_name,
                schema={
                    "id": TextDataType(nullable=False),
                    "thread_id": TextDataType(nullable=False),
                    "in_reply_to": TextDataType(),
                    "role": TextDataType(nullable=False),
                    "json": TextDataType(nullable=False),
                },
                mode="create_if_not_exists",
                namespace=fallback_namespace,
            )

    async def _load_message(self, *, message_id: str) -> dict[str, Any] | None:
        table_name, namespace = self._database_table_ref()
        results = await self._room.database.search(
            table=table_name,
            where={"id": message_id},
            namespace=namespace,
        )
        if len(results) == 0:
            return None
        raw_json = results[0].get("json")
        if not isinstance(raw_json, str):
            return None
        return json.loads(raw_json)

    async def _resolve_thread_id(self, *, message: dict[str, Any]) -> str:
        in_reply_to = message.get("in_reply_to")
        if isinstance(in_reply_to, str) and in_reply_to.strip() != "":
            parent_message = await self._load_message(message_id=in_reply_to.strip())
            if parent_message is not None:
                thread_id = parent_message.get("thread_id")
                if isinstance(thread_id, str) and thread_id.strip() != "":
                    return thread_id

        thread_name_source = (
            str(message.get("subject") or "").strip()
            or str(message.get("from") or "").strip()
            or str(message.get("body") or "").strip()
            or "New Mail"
        )
        thread_name = LegacyChatChannel._sanitize_thread_name(value=thread_name_source)
        thread_path = LegacyChatChannel._thread_path_for_name(
            thread_name=thread_name,
            thread_dir=self._get_thread_dir(),
        )
        return await self._next_available_thread_path(base_path=thread_path)

    async def _save_email_message(
        self,
        *,
        content: bytes,
        role: str,
        thread_id: str,
    ) -> dict[str, Any]:
        email_message = message_from_bytes(content, policy=default)
        queued_message = message_to_json(message=email_message, role=role)
        queued_message["thread_id"] = thread_id

        folder_path = posixpath.join(
            uuid.uuid4().hex[:8],
            uuid.uuid4().hex[:8],
        )
        base_path = posixpath.join(_MAIL_STORAGE_ROOT, folder_path)
        queued_message["path"] = posixpath.join(base_path, "message.json")

        for file_name, attachment_bytes in iter_message_attachments(email_message):
            attachment_path = posixpath.join(base_path, "attachments", file_name)
            await self._room.storage.upload(path=attachment_path, data=attachment_bytes)
            queued_message["attachments"].append(attachment_path)

        await self._room.storage.upload(
            path=posixpath.join(base_path, "message.eml"),
            data=content,
        )
        await self._room.storage.upload(
            path=posixpath.join(base_path, "message.json"),
            data=json.dumps(queued_message, indent=4).encode("utf-8"),
        )
        table_name, namespace = self._database_table_ref()
        await self._room.database.insert(
            table=table_name,
            namespace=namespace,
            records=[
                {
                    "id": queued_message["id"],
                    "thread_id": thread_id,
                    "in_reply_to": queued_message.get("in_reply_to"),
                    "role": role,
                    "json": json.dumps(queued_message),
                }
            ],
        )
        return queued_message

    def _content_from_mail_message(
        self,
        *,
        message: dict[str, Any],
    ) -> list[AgentTextContent | AgentFileContent]:
        body = str(message.get("body") or "")
        from_header = str(message.get("from") or "")
        subject = str(message.get("subject") or "")
        to_header = ", ".join(str(item) for item in message.get("to") or [])
        cc_header = ", ".join(str(item) for item in message.get("cc") or [])

        text_lines = ["Incoming email message:"]
        if from_header != "":
            text_lines.append(f"From: {from_header}")
        if to_header != "":
            text_lines.append(f"To: {to_header}")
        if cc_header != "":
            text_lines.append(f"Cc: {cc_header}")
        if subject != "":
            text_lines.append(f"Subject: {subject}")
        if body != "":
            text_lines.extend(["", body])

        content: list[AgentTextContent | AgentFileContent] = [
            AgentTextContent(
                type="text",
                text="\n".join(text_lines),
            )
        ]
        if self._enable_attachments:
            for attachment_path in message.get("attachments") or []:
                if not isinstance(attachment_path, str):
                    continue
                normalized = attachment_path.strip().lstrip("/")
                if normalized == "":
                    continue
                content.append(
                    AgentFileContent(
                        type="file",
                        url=f"room:///{normalized}",
                    )
                )

        return content

    def _participant_for_message(self, *, message: dict[str, Any]) -> Participant:
        from_header = str(message.get("from") or "")
        display_name, address = email.utils.parseaddr(from_header)
        participant_name = display_name.strip() or address.strip() or "email user"
        participant_id = f"mail:{(address or message.get('id') or uuid.uuid4().hex)}"
        return Participant(id=participant_id, attributes={"name": participant_name})

    def _reply_text_for_turn(self, *, active: _ActiveMailTurn) -> str:
        parts = [part for part in active.completed_text_parts if part.strip() != ""]
        for text in active.text_by_item_id.values():
            if text.strip() != "":
                parts.append(text)
        return "\n\n".join(parts).strip()

    def _clear_thread_state(self, *, thread_id: str) -> None:
        self._pending_messages_by_source_id = {
            source_id: pending
            for source_id, pending in self._pending_messages_by_source_id.items()
            if pending.thread_id != thread_id
        }
        self._active_turns_by_turn_id = {
            turn_id: active
            for turn_id, active in self._active_turns_by_turn_id.items()
            if active.thread_id != thread_id
        }

    async def _smtp_send(self, *, message) -> None:
        username = self._smtp.username
        if username is None:
            username_value = self._room.local_participant.get_attribute("name")
            username = username_value if isinstance(username_value, str) else None

        password = self._smtp.password
        if password is None:
            password = self._room.protocol.token

        hostname = (
            self._smtp.hostname if self._smtp.hostname is not None else self._domain
        )
        await aiosmtplib.send(
            message,
            hostname=hostname,
            port=self._smtp.port,
            username=username,
            password=password,
        )

    def _default_thread_dir(self) -> str:
        local_name_value = self._room.local_participant.get_attribute("name")
        local_name = (
            local_name_value.strip()
            if isinstance(local_name_value, str) and local_name_value.strip() != ""
            else "mail"
        )
        return LegacyChatChannel._normalize_thread_dir(
            thread_dir=posixpath.join(".threads", local_name)
        )

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir
        return self._default_thread_dir()

    def _database_namespace(self) -> list[str]:
        return [part for part in self._get_thread_dir().split("/") if part != ""]

    @staticmethod
    def _is_namespace_location_error(*, exc: RoomException) -> bool:
        return "Location must be provided when namespace is not empty" in str(exc)

    @staticmethod
    def _sanitize_table_name_component(*, value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
        normalized = normalized.strip("_")
        if normalized == "":
            return "root"
        return normalized

    def _flat_mail_table_name(self) -> str:
        namespace = self._database_namespace()
        if len(namespace) == 0:
            return _MAIL_TABLE_NAME

        suffix = "__".join(
            self._sanitize_table_name_component(value=part) for part in namespace
        )
        return f"{_MAIL_TABLE_NAME}__{suffix}"

    def _database_table_ref(self) -> tuple[str, list[str] | None]:
        namespace = self._database_namespace()
        if self._database_namespace_supported:
            return _MAIL_TABLE_NAME, namespace
        return self._flat_mail_table_name(), None

    async def _next_available_thread_path(self, *, base_path: str) -> str:
        try:
            exists = await self._room.storage.exists(path=base_path)
        except Exception:
            return base_path

        if not exists:
            return base_path

        thread_dir, filename = posixpath.split(base_path)
        if filename.endswith(".thread"):
            base_name = filename[: -len(".thread")]
        else:
            base_name = filename

        for index in range(2, 1000):
            candidate = posixpath.join(thread_dir, f"{base_name} {index}.thread")
            try:
                if not await self._room.storage.exists(path=candidate):
                    return candidate
            except Exception:
                return candidate

        return posixpath.join(thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}.thread")

    def _thread_id_from_tool_context(self, *, context: ToolContext) -> str:
        caller_context = context.caller_context
        if not isinstance(caller_context, dict):
            raise RoomException("mail tool requires a thread_id in caller_context")

        thread_id = caller_context.get("thread_id")
        if not isinstance(thread_id, str) or thread_id.strip() == "":
            raise RoomException("mail tool requires a non-empty thread_id")
        return thread_id

    async def _attachment_files_from_paths(
        self,
        *,
        context: ToolContext,
        attachments: list[str],
    ) -> list[FileContent]:
        files: list[FileContent] = []
        for attachment_path in attachments:
            try:
                files.append(await context.room.storage.download(path=attachment_path))
            except Exception as exc:
                logger.error(
                    "unable to download file %s",
                    attachment_path,
                    exc_info=exc,
                )
                raise RoomException(
                    f"Could not download a file from the room with the path {attachment_path}. Are you sure the path is correct file?"
                ) from exc
        return files

    def _make_new_email_thread_tool(self) -> FunctionTool:
        outer = self

        class NewEmailThreadTool(FunctionTool):
            def __init__(self) -> None:
                super().__init__(
                    name="new_email_thread",
                    description="starts a new outbound email thread from the current agent thread",
                    supports_context=True,
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["to", "subject", "body", "attachments"],
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                            "attachments": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "a list of room storage paths to attach",
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
            ) -> dict[str, Any]:
                thread_id = outer._thread_id_from_tool_context(context=context)
                files = await outer._attachment_files_from_paths(
                    context=context,
                    attachments=attachments,
                )
                await outer.start_thread(
                    thread_id=thread_id,
                    to_address=to,
                    subject=subject,
                    body=body,
                    attachments=files,
                )
                return {}

        return NewEmailThreadTool()
