from __future__ import annotations

from email.message import EmailMessage
from typing import Any, Iterable, Iterator, Literal
import email.utils
import os
import re
import socket
import uuid

import mistune

type MessageRole = Literal["user", "agent"]

_DSN_STATUS_RE = re.compile(r"\b([245]\.\d{1,3}\.\d{1,3})\b")
_SMTP_DIAG_RE = re.compile(r"\b([245]\d\d)\b")


class SmtpConfiguration:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        port: int | None = None,
        hostname: str | None = None,
        local_hostname: str | None = None,
    ) -> None:
        if username is None:
            username = os.getenv("SMTP_USERNAME")

        if password is None:
            password = os.getenv("SMTP_PASSWORD")

        if port is None:
            port = int(os.getenv("SMTP_PORT", "587"))

        if hostname is None:
            hostname = os.getenv("SMTP_HOSTNAME")

        if local_hostname is None:
            local_hostname = os.getenv("SMTP_LOCAL_HOSTNAME")

        self.username = username
        self.password = password
        self.port = port
        self.hostname = hostname
        self.local_hostname = local_hostname

    def effective_local_hostname(self) -> str:
        candidates = [
            self.local_hostname,
            socket.getfqdn(),
            socket.gethostname(),
            os.getenv("HOSTNAME"),
            "localhost",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = candidate.strip()
            if normalized != "":
                return normalized

        return "localhost"


def _parse_addrs(values: Any) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        addr.casefold() for _, addr in email.utils.getaddresses(list(values)) if addr
    }


def _first_addr(value: str | None) -> str | None:
    if not value:
        return None
    _, addr = email.utils.parseaddr(value)
    return addr or None


def _fmt_addr_list(addrs: Iterable[str]) -> str:
    return ", ".join(addrs)


def _get_first_header(msg: EmailMessage, name: str) -> str | None:
    value = msg.get(name)
    return str(value).strip() if value is not None else None


def _get_all_headers(msg: EmailMessage, name: str) -> list[str]:
    values = msg.get_all(name) or []
    return [str(v) for v in values]


def _clean_header_list(values: Any) -> list[str]:
    if not values:
        return []
    return [v for v in (str(x).strip() for x in values) if v]


def _parse_dsn_fields(msg: EmailMessage) -> dict[str, Any]:
    out: dict[str, Any] = {
        "is_dsn": False,
        "status": None,
        "action": None,
        "diagnostic_code": None,
        "final_recipient": None,
    }

    report_type = msg.get_param("report-type", header="Content-Type")
    if (
        msg.get_content_type().lower() == "multipart/report"
        and isinstance(report_type, str)
        and report_type.lower() == "delivery-status"
    ):
        out["is_dsn"] = True

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type().lower() != "message/delivery-status":
                continue

            out["is_dsn"] = True
            payload = part.get_payload()
            blocks = payload if isinstance(payload, list) else [payload]
            for block in blocks:
                if block is None:
                    continue
                action = block.get("Action")
                status = block.get("Status")
                diagnostic = block.get("Diagnostic-Code")
                final_recipient = block.get("Final-Recipient")

                if action and not out["action"]:
                    out["action"] = str(action)
                if status and not out["status"]:
                    out["status"] = str(status)
                if diagnostic and not out["diagnostic_code"]:
                    out["diagnostic_code"] = str(diagnostic)
                if final_recipient and not out["final_recipient"]:
                    out["final_recipient"] = str(final_recipient)
            break

    if not out["status"]:
        out["status"] = _get_first_header(msg, "Status")
    if not out["action"]:
        out["action"] = _get_first_header(msg, "Action")
    if not out["diagnostic_code"]:
        out["diagnostic_code"] = _get_first_header(msg, "Diagnostic-Code")
    if not out["final_recipient"]:
        out["final_recipient"] = _get_first_header(msg, "Final-Recipient")

    if out["status"]:
        match = _DSN_STATUS_RE.search(str(out["status"]))
        out["status"] = match.group(1) if match else str(out["status"]).strip()

    if out["diagnostic_code"]:
        diagnostic_match = _SMTP_DIAG_RE.search(str(out["diagnostic_code"]))
        if diagnostic_match is not None:
            out["diagnostic_code"] = diagnostic_match.group(1)

    return out


def _looks_like_error_or_autoreply(message: dict[str, Any]) -> tuple[bool, str]:
    meta = message.get("meta") or {}
    dsn = meta.get("dsn") or {}

    if dsn.get("is_dsn"):
        status = (dsn.get("status") or "").strip()
        diagnostic = (dsn.get("diagnostic_code") or "").strip()
        return True, f"DSN detected (status={status!r}, diagnostic={diagnostic!r})"

    content_type = (meta.get("content_type") or "").casefold()
    if content_type == "multipart/report":
        return True, "multipart/report detected"

    auto_submitted = (meta.get("auto_submitted") or "").casefold()
    if auto_submitted and auto_submitted != "no":
        return True, f"Auto-Submitted={meta.get('auto_submitted')!r}"

    precedence = (meta.get("precedence") or "").casefold()
    if precedence in {"bulk", "junk", "list", "auto_reply", "auto-reply"}:
        return True, f"Precedence={meta.get('precedence')!r}"

    return_path = (meta.get("return_path") or "").strip()
    if return_path == "<>":
        return True, "Return-Path is <>"

    from_header = message.get("from") or ""
    _, from_addr = email.utils.parseaddr(from_header)
    normalized_from = from_addr.casefold()
    if "mailer-daemon" in normalized_from or normalized_from.startswith("postmaster@"):
        return True, f"From looks like system sender ({from_addr or from_header})"

    subject = (message.get("subject") or "").casefold()
    for snippet in (
        "delivery status notification",
        "undelivered mail",
        "returned to sender",
        "mail delivery subsystem",
        "failure notice",
    ):
        if snippet in subject:
            return True, f"Subject looks like a bounce ({subject!r})"

    return False, ""


def message_to_json(*, message: EmailMessage, role: MessageRole) -> dict[str, Any]:
    body_part = message.get_body(("plain", "html"))
    if body_part is not None:
        body = body_part.get_content()
        body_content_type = body_part.get_content_type()
    else:
        body = message.get_content()
        body_content_type = message.get_content_type()

    message_id = message.get("Message-ID")
    if message_id is None:
        from_header = message.get("From", "")
        _, address = email.utils.parseaddr(from_header)
        domain = (address.split("@")[-1] if "@" in address else "local").lower()
        message_id = f"{uuid.uuid4()}@{domain}"

    to_list = _clean_header_list(_get_all_headers(message, "To"))
    cc_list = _clean_header_list(_get_all_headers(message, "Cc"))
    dsn = _parse_dsn_fields(message)

    return {
        "id": message_id,
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
        "meta": {
            "delivered_to": _get_first_header(message, "Delivered-To"),
            "return_path": _get_first_header(message, "Return-Path"),
            "to": to_list,
            "cc": cc_list,
            "auto_submitted": _get_first_header(message, "Auto-Submitted"),
            "precedence": _get_first_header(message, "Precedence"),
            "list_id": _get_first_header(message, "List-Id"),
            "content_type": message.get_content_type(),
            "is_multipart": message.is_multipart(),
            "body_content_type": body_content_type,
            "dsn": dsn,
        },
    }


def iter_message_attachments(message: EmailMessage) -> Iterator[tuple[str, bytes]]:
    for part in message.iter_attachments():
        file_name = part.get_filename() or "attachment.bin"
        data = part.get_content()
        if isinstance(data, str):
            yield file_name, data.encode(part.get_content_charset("utf-8"))
        else:
            yield file_name, data


def should_reply_to_message(
    *,
    message: dict[str, Any],
    email_address: str,
    whitelist: list[str] | None = None,
) -> tuple[bool, str | None]:
    my_addr = email_address.casefold()
    to_addrs = _parse_addrs(message.get("to"))
    cc_addrs = _parse_addrs(message.get("cc"))
    delivered_to = ((message.get("meta") or {}).get("delivered_to") or "").casefold()

    addressed = (
        (my_addr in to_addrs) or (my_addr in cc_addrs) or (delivered_to == my_addr)
    )
    if not addressed:
        return (
            False,
            "message is not addressed to this mailbox",
        )

    is_error, reason = _looks_like_error_or_autoreply(message)
    if is_error:
        return False, reason

    if whitelist is not None:
        from_header = message.get("from") or ""
        _, addr = email.utils.parseaddr(from_header)
        if addr.casefold() not in {entry.casefold() for entry in whitelist}:
            return False, f"{from_header} not found in whitelist"

    return True, None


def render_mail_markdown(*, body: str) -> str:
    markdown = mistune.create_markdown(plugins=["table"])
    return markdown(body)


def create_email_message(
    *,
    to_address: str,
    from_address: str,
    subject: str,
    body: str,
    correlation_id: str | None = None,
) -> EmailMessage:
    _, address = email.utils.parseaddr(from_address)
    domain = (address.split("@")[-1] if "@" in address else "local").lower()
    message_id = f"<{uuid.uuid4()}@{domain}>"

    message = EmailMessage()
    message["Message-ID"] = message_id
    message["Subject"] = subject
    message["From"] = from_address
    message["To"] = to_address
    if correlation_id is not None:
        message["Meshagent-Correlation-ID"] = correlation_id

    message.set_content(body)
    message.add_alternative(render_mail_markdown(body=body), subtype="html")
    return message


def create_reply_email_message(
    *,
    message: dict[str, Any],
    from_address: str,
    body: str,
    email_address: str,
    reply_all: bool = False,
) -> EmailMessage:
    subject = str(message.get("subject") or "")
    if not subject.lower().startswith("re:"):
        subject = "RE: " + subject

    _, address = email.utils.parseaddr(from_address)
    domain = (address.split("@")[-1] if "@" in address else "local").lower()
    message_id = f"<{uuid.uuid4()}@{domain}>"

    reply_to_header = str(message.get("reply_to") or message.get("from") or "")
    reply_to_addr = _first_addr(reply_to_header)

    my_addr = email_address.casefold()
    original_to = _parse_addrs(message.get("to"))
    original_cc = _parse_addrs(message.get("cc"))

    to_addrs: list[str] = []
    cc_addrs: list[str] = []

    if reply_to_addr:
        to_addrs = [reply_to_addr]

    if reply_all:
        sender_casefold = (reply_to_addr or "").casefold()
        current_to = {item.casefold() for item in to_addrs}
        for address_value in original_to | original_cc:
            normalized = address_value.casefold()
            if normalized == my_addr:
                continue
            if sender_casefold != "" and normalized == sender_casefold:
                continue
            if normalized in current_to:
                continue
            cc_addrs.append(address_value)

    reply_message = EmailMessage()
    reply_message["Message-ID"] = message_id
    reply_message["Subject"] = subject
    reply_message["From"] = from_address
    if to_addrs:
        reply_message["To"] = _fmt_addr_list(to_addrs)
    else:
        reply_message["To"] = reply_to_header
    if len(cc_addrs) > 0:
        reply_message["Cc"] = _fmt_addr_list(cc_addrs)

    parent_message_id = message.get("id")
    if isinstance(parent_message_id, str) and parent_message_id.strip() != "":
        reply_message["In-Reply-To"] = parent_message_id

    references = message.get("references")
    if isinstance(references, str) and references.strip() != "":
        reply_message["References"] = references

    correlation_id = message.get("correlation_id")
    if isinstance(correlation_id, str) and correlation_id.strip() != "":
        reply_message["Meshagent-Correlation-ID"] = correlation_id

    reply_message.set_content(body)
    reply_message.add_alternative(render_mail_markdown(body=body), subtype="html")
    return reply_message
