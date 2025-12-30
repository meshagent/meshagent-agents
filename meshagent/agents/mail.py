from meshagent.agents.worker import Worker
from meshagent.tools import RemoteToolkit, ToolContext, Tool, Toolkit
from meshagent.api.room_server_client import TextDataType
from email import message_from_bytes
from email.message import EmailMessage
from meshagent.api import RoomClient
from meshagent.api import RequiredTable
from email.policy import default
import email.utils
from meshagent.agents import AgentChatContext
from datetime import datetime, timezone
import base64
import secrets

from typing import Literal, Optional, Iterable
import json

import uuid
import logging

import os
import aiosmtplib

import mistune

import re

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


_DSN_STATUS_RE = re.compile(r"\b([245]\.\d{1,3}\.\d{1,3})\b")
_SMTP_DIAG_RE = re.compile(r"\b([245]\d\d)\b")  # e.g. 550, 421 (fallback)

_BOUNCE_SUBJECT_SNIPPETS = (
    "delivery status notification",
    "delivery failure",
    "undelivered mail",
    "returned to sender",
    "mail delivery subsystem",
    "failure notice",
)


def _parse_addrs(values) -> set[str]:
    """
    values may be:
      - None
      - a single header string
      - a list[str] of header strings (your JSON uses list)
    Returns a set of casefolded email addresses.
    """
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        addr.casefold() for _, addr in email.utils.getaddresses(list(values)) if addr
    }


def _first_addr(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    _, addr = email.utils.parseaddr(value)
    return addr or None


def _fmt_addr_list(addrs: Iterable[str]) -> str:
    # EmailMessage headers want a single string.
    return ", ".join(addrs)


def _looks_like_error_or_autoreply(msg: dict) -> tuple[bool, str]:
    meta = msg.get("meta") or {}
    dsn = meta.get("dsn") or {}

    # 1) DSN extracted (best)
    if dsn.get("is_dsn"):
        status = (dsn.get("status") or "").strip()
        diag = (dsn.get("diagnostic_code") or "").strip()
        return True, f"DSN detected (status={status!r}, diagnostic={diag!r})"

    # 2) Content-Type heuristic (very strong)
    ctype = (meta.get("content_type") or "").casefold()
    if ctype == "multipart/report":
        rt = (meta.get("report_type") or "").casefold()  # if you add later
        return True, f"multipart/report detected (report_type={rt or 'unknown'})"

    # 3) Auto replies / system-generated
    auto_submitted = (meta.get("auto_submitted") or "").casefold()
    if auto_submitted and auto_submitted != "no":
        return True, f"Auto-Submitted={meta.get('auto_submitted')!r}"

    precedence = (meta.get("precedence") or "").casefold()
    if precedence in {"bulk", "junk", "list", "auto_reply", "auto-reply"}:
        return True, f"Precedence={meta.get('precedence')!r}"

    # 4) Return-Path <> (common for bounces)
    return_path = (meta.get("return_path") or "").strip()
    if return_path == "<>":
        return True, "Return-Path is <>"

    # 5) From patterns (weaker, but useful)
    from_header = msg.get("from") or ""
    _, from_addr = email.utils.parseaddr(from_header)
    fa = from_addr.casefold()
    if "mailer-daemon" in fa or fa.startswith("postmaster@"):
        return True, f"From looks like system sender ({from_addr or from_header})"

    # 6) Subject fallback (weak)
    subject = (msg.get("subject") or "").casefold()
    if any(
        s in subject
        for s in (
            "delivery status notification",
            "undelivered mail",
            "returned to sender",
            "mail delivery subsystem",
            "delivery failure",
            "failure notice",
        )
    ):
        return True, f"Subject indicates bounce/DSN ({msg.get('subject')!r})"

    return False, "not identified as error/auto-reply"


def _get_first_header(msg: EmailMessage, name: str) -> Optional[str]:
    v = msg.get(name)
    return v if v is not None else None


def _get_all_headers(msg: EmailMessage, name: str) -> list[str]:
    vals = msg.get_all(name) or []
    # email.message returns list[str] (or Header objects); normalize to str
    return [str(v) for v in vals if v is not None]


def _parse_dsn_fields(msg: EmailMessage) -> dict[str, any]:
    """
    If this is a DSN (multipart/report with a message/delivery-status part),
    extract standardized fields.
    """
    out: dict[str, any] = {
        "is_dsn": False,
        "action": None,
        "status": None,
        "diagnostic_code": None,
        "final_recipient": None,
        "failed_recipients": _get_all_headers(msg, "X-Failed-Recipients") or None,
    }

    ctype = (msg.get_content_type() or "").lower()
    if ctype == "multipart/report":
        # report-type param is often "delivery-status"
        params = dict((k.lower(), v) for k, v in (msg.get_params() or []))
        report_type = (params.get("report-type") or "").lower()
        if report_type == "delivery-status":
            out["is_dsn"] = True

    # Walk parts to find message/delivery-status
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type().lower() == "message/delivery-status":
                out["is_dsn"] = True

                # delivery-status is structured as one or more header blocks
                payload = part.get_payload()
                # In the stdlib, this is often a list of Message objects
                blocks = payload if isinstance(payload, list) else [payload]

                # Look for per-recipient block first, then fallback to any
                for block in blocks:
                    if block is None:
                        continue
                    # block behaves like Message with get()
                    action = block.get("Action")
                    status = block.get("Status")
                    diag = block.get("Diagnostic-Code")
                    final_rcpt = block.get("Final-Recipient")

                    if action and not out["action"]:
                        out["action"] = str(action)
                    if status and not out["status"]:
                        out["status"] = str(status)
                    if diag and not out["diagnostic_code"]:
                        out["diagnostic_code"] = str(diag)
                    if final_rcpt and not out["final_recipient"]:
                        out["final_recipient"] = str(final_rcpt)

                break

    # If not multipart/report, still try the easy headers (some providers flatten these)
    if not out["status"]:
        out["status"] = _get_first_header(msg, "Status")
    if not out["action"]:
        out["action"] = _get_first_header(msg, "Action")
    if not out["diagnostic_code"]:
        out["diagnostic_code"] = _get_first_header(msg, "Diagnostic-Code")
    if not out["final_recipient"]:
        out["final_recipient"] = _get_first_header(msg, "Final-Recipient")

    # Last-chance: try to spot X.Y.Z code in subject/body-ish snippets if you pass them along later
    if out["status"]:
        m = _DSN_STATUS_RE.search(out["status"])
        out["status"] = m.group(1) if m else str(out["status"]).strip()

    return out


def _clean_header_list(values) -> list[str]:
    """
    Normalize get_all() results:
    - None → []
    - drop empty / whitespace-only strings
    - ensure list[str]
    """
    if not values:
        return []
    return [v for v in (str(x).strip() for x in values) if v]


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
        reply_all: bool = False,
    ):
        if smtp is None:
            smtp = SmtpConfiguration()

        self._domain = domain
        self._smtp = smtp
        self._reply_all = reply_all

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
        # Body extraction (yours, kept)
        body_part = message.get_body(("plain", "html"))
        if body_part:
            body = body_part.get_content()
            body_content_type = body_part.get_content_type()
        else:
            body = message.get_content()
            body_content_type = message.get_content_type()

        # Ensure Message-ID exists (yours, kept)
        msg_id = message.get("Message-ID")
        if msg_id is None:
            mfrom = message.get("From", "")
            _, addr = email.utils.parseaddr(mfrom)
            domain = (addr.split("@")[-1] if "@" in addr else "local").lower()
            msg_id = f"{uuid.uuid4()}@{domain}"

        # Address fields (normalize)
        to_list = _clean_header_list(_get_all_headers(message, "To"))
        cc_list = _clean_header_list(_get_all_headers(message, "Cc"))

        # Meta for bounce/auto-reply detection
        dsn = _parse_dsn_fields(message)

        meta = {
            # routing / addressing signals
            "delivered_to": _get_first_header(message, "Delivered-To"),
            "return_path": _get_first_header(message, "Return-Path"),
            "to": to_list,
            "cc": cc_list,
            # automation / system-generated signals
            "auto_submitted": _get_first_header(message, "Auto-Submitted"),
            "precedence": _get_first_header(message, "Precedence"),
            "list_id": _get_first_header(message, "List-Id"),
            # content signals
            "content_type": message.get_content_type(),
            "is_multipart": message.is_multipart(),
            "body_content_type": body_content_type,
            # DSN extracted fields (best)
            "dsn": dsn,
        }

        return {
            "id": msg_id,
            "in_reply_to": message.get("In-Reply-To"),
            "reply_to": message.get("Reply-To", message.get("From")),
            "references": message.get("References"),
            "from": message.get("From"),
            "to": to_list,
            "cc": cc_list,
            "subject": message.get("Subject"),
            "body": body,
            "attachments": [],
            "role": role,
            "correlation_id": message.get("Meshagent-Correlation-ID"),
            "meta": meta,
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

    async def should_reply(self, *, message: dict) -> bool:
        my_addr = self._email_address.casefold()

        # Addressed-to check (supports your JSON: "to" is list[str], plus "cc" and meta.delivered_to)
        to_addrs = _parse_addrs(message.get("to"))
        cc_addrs = _parse_addrs(message.get("cc"))
        delivered_to = (
            (message.get("meta") or {}).get("delivered_to") or ""
        ).casefold()

        addressed = (
            (my_addr in to_addrs) or (my_addr in cc_addrs) or (delivered_to == my_addr)
        )
        if not addressed:
            logger.info(
                f"message not addressed to {self._email_address}; "
                f"to={message.get('to')!r} cc={message.get('cc')!r} delivered_to={delivered_to!r}"
            )
            return False

        # Drop bounces / DSNs / auto-replies
        is_bad, reason = _looks_like_error_or_autoreply(message)
        if is_bad:
            logger.info(f"discarding message (error/auto-reply): {reason}")
            return False

        # Whitelist gate (apply after we know it's not a system email)
        if self._whitelist is not None:
            from_header = message.get("from") or ""
            _, addr = email.utils.parseaddr(from_header)
            if addr.casefold() not in self._whitelist:
                logger.info(f"{from_header} not found in whitelist, discarding message")
                return False

        return True

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

        try:
            reply = await self._llm_adapter.next(
                context=chat_context,
                room=self.room,
                toolkits=toolkits,
                tool_adapter=self._tool_adapter,
            )
        except Exception as ex:
            logger.error(f"error while processing message {ex}", exc_info=ex)
            reply = "An error occurred while processing your message, please try again."

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
        self,
        *,
        message: dict,
        from_address: str,
        body: str,
        reply_all: bool = False,  # <-- choose behavior
    ) -> EmailMessage:
        subject: str = message.get("subject") or ""
        if not subject.lower().startswith("re:"):
            subject = "RE: " + subject

        _, addr = email.utils.parseaddr(from_address)
        domain = (addr.split("@")[-1] if "@" in addr else "local").lower()
        msg_id = f"<{uuid.uuid4()}@{domain}>"

        # Sender we should reply to
        reply_to_header = message.get("reply_to") or message.get("from") or ""
        reply_to_addr = _first_addr(reply_to_header)

        my_addr = self._email_address.casefold()

        # Original recipients
        orig_to = _parse_addrs(message.get("to"))  # list[str] header lines
        orig_cc = _parse_addrs(message.get("cc"))  # list[str] header lines

        # Build recipients
        to_addrs = []
        cc_addrs = []

        if reply_to_addr:
            to_addrs = [reply_to_addr]

        if reply_all:
            # Reply-all semantics:
            # Cc everyone else (original To + Cc) excluding me + excluding the sender
            sender_cf = (reply_to_addr or "").casefold()

            everyone_else = []
            for a in orig_to | orig_cc:
                acf = a.casefold()
                if acf == my_addr:
                    continue
                if sender_cf and acf == sender_cf:
                    continue
                if acf in {x.casefold() for x in to_addrs}:
                    continue
                everyone_else.append(a)

            cc_addrs = everyone_else
        else:
            # Plain "Reply": do NOT copy thread recipients
            cc_addrs = []

        msg = EmailMessage()
        msg["Message-ID"] = msg_id
        msg["Subject"] = subject
        msg["From"] = from_address

        if to_addrs:
            msg["To"] = _fmt_addr_list(to_addrs)
        else:
            # Fallback: if we couldn't parse Reply-To/From, at least don't create a malformed message
            msg["To"] = reply_to_header

        # Set Cc only if non-empty
        if cc_addrs:
            msg["Cc"] = _fmt_addr_list(cc_addrs)

        if message.get("id"):
            msg["In-Reply-To"] = message["id"]
        if message.get("references"):
            msg["References"] = message["references"]

        correlation_id = message.get("correlation_id")
        if correlation_id is not None:
            msg["Meshagent-Correlation-ID"] = correlation_id

        msg.set_content(body)
        msg.add_alternative(self.render_markdown(body), subtype="html")
        return msg

    async def send_reply_message(self, *, message: dict, reply: str):
        msg = self.create_reply_email_message(
            message=message,
            from_address=self._email_address,
            body=reply,
            reply_all=self._reply_all,
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
