from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Sequence
from urllib.parse import urlparse

from meshagent.api import Participant

from .adapter import LLMAdapter

logger = logging.getLogger("thread-naming")

DEFAULT_THREAD_NAME = "New Chat"
DEFAULT_CHANNEL_THREAD_NAME_RULES = [
    "generate a concise, friendly title for this chat thread",
    "return only a thread_name value suitable for display in a thread list",
    "thread_name should be 2-6 words and topic-focused",
    "use normal capitalization and spaces, and do not include a .thread extension",
]


def sanitize_thread_name(*, value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".thread"):
        normalized = normalized[: -len(".thread")]

    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
    normalized = re.sub(r"[^A-Za-z0-9 .,!?':()&]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
    if normalized == "":
        return DEFAULT_THREAD_NAME
    if normalized == normalized.lower() or normalized == normalized.upper():
        normalized = normalized.title()
    return normalized[:64].strip() or DEFAULT_THREAD_NAME


def attachment_name_for_thread_title(*, attachment: str) -> str:
    normalized = attachment.strip()
    if normalized == "":
        return ""

    parsed = urlparse(normalized)
    if parsed.scheme != "":
        attachment_name = PurePosixPath(parsed.path).name
        if attachment_name != "":
            return attachment_name

    attachment_name = PurePosixPath(normalized).name
    if attachment_name != "":
        return attachment_name

    return normalized


def normalized_thread_title_attachments(
    *,
    attachments: Sequence[str] | None,
) -> list[str]:
    attachment_names: list[str] = []
    for attachment in attachments or []:
        normalized = attachment_name_for_thread_title(attachment=attachment)
        if normalized != "":
            attachment_names.append(normalized)
    return attachment_names


def thread_name_input_text(
    *,
    message_text: str,
    attachments: Sequence[str] | None = None,
) -> str:
    parts: list[str] = []
    normalized_message_text = message_text.strip()
    if normalized_message_text != "":
        parts.append(f"Message:\n{normalized_message_text}")

    attachment_names = normalized_thread_title_attachments(attachments=attachments)
    if len(attachment_names) > 0:
        attachment_lines = "\n".join(
            f"- {attachment_name}" for attachment_name in attachment_names
        )
        parts.append(f"Attachments:\n{attachment_lines}")

    if len(parts) == 0:
        return DEFAULT_THREAD_NAME
    return "\n\n".join(parts)


def fallback_thread_name(
    *,
    message_text: str,
    attachments: Sequence[str] | None = None,
) -> str:
    normalized_message_text = message_text.strip()
    if normalized_message_text != "":
        return sanitize_thread_name(value=normalized_message_text)

    attachment_names = normalized_thread_title_attachments(attachments=attachments)
    if len(attachment_names) > 0:
        return sanitize_thread_name(
            value=", ".join(attachment_names[:3]),
        )

    return DEFAULT_THREAD_NAME


async def determine_thread_name(
    *,
    adapter: LLMAdapter | None,
    caller: Participant,
    message_text: str,
    attachments: Sequence[str] | None = None,
    on_behalf_of: Participant | None = None,
    thread_name_rules: Sequence[str] | None = None,
    model: str | None = None,
) -> str:
    generated_name = fallback_thread_name(
        message_text=message_text,
        attachments=attachments,
    )
    if adapter is None:
        return generated_name

    session = adapter.create_session()
    cloned_context = session.copy()
    async with cloned_context:
        cloned_context.replace_rules(
            rules=list(thread_name_rules or DEFAULT_CHANNEL_THREAD_NAME_RULES)
        )
        cloned_context.append_user_message(
            thread_name_input_text(
                message_text=message_text,
                attachments=attachments,
            )
        )
        try:
            response = await adapter.create_response(
                context=cloned_context,
                caller=caller,
                model=model or adapter.default_model(),
                on_behalf_of=on_behalf_of,
                toolkits=[],
                output_schema={
                    "type": "object",
                    "required": ["thread_name"],
                    "additionalProperties": False,
                    "properties": {
                        "thread_name": {
                            "type": "string",
                            "description": "2-6 word topic name for the task thread",
                        }
                    },
                },
            )
            if isinstance(response, dict):
                thread_name = response.get("thread_name")
                if isinstance(thread_name, str):
                    generated_name = sanitize_thread_name(value=thread_name)
        except Exception as ex:
            logger.warning(
                "unable to auto-generate thread name, using fallback",
                exc_info=ex,
            )

    return generated_name
