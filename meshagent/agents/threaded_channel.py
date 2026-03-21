from __future__ import annotations

import logging
import posixpath
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Sequence
from urllib.parse import urlparse

from meshagent.api import Element, MeshDocument, Participant, RoomClient

from .adapter import LLMAdapter
from .context import AgentSessionContext
from .process import Channel
from .thread_schema import thread_list_schema

logger = logging.getLogger("threaded-channel")

DEFAULT_CHANNEL_THREAD_NAME_RULES = [
    "generate a concise, friendly title for this chat thread",
    "return only a thread_name value suitable for display in a thread list",
    "thread_name should be 2-6 words and topic-focused",
    "use normal capitalization and spaces, and do not include a .thread extension",
]


class ThreadedChannel(Channel):
    def __init__(
        self,
        *,
        room: RoomClient,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        llm_adapter: LLMAdapter | None = None,
        thread_name_rules: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self._room = room
        self._threading_mode = self._normalize_threading_mode(
            threading_mode=threading_mode
        )
        self._thread_dir = self._normalize_thread_dir(thread_dir=thread_dir)
        self._llm_adapter = llm_adapter
        if thread_name_rules is not None and len(thread_name_rules) > 0:
            self._thread_name_rules = [*thread_name_rules]
        else:
            self._thread_name_rules = [*DEFAULT_CHANNEL_THREAD_NAME_RULES]
        self._thread_list_document: MeshDocument | None = None
        self._thread_list_path: str | None = None

    @property
    def room(self) -> RoomClient:
        return self._room

    def thread_name_adapter(self) -> LLMAdapter | None:
        return self._llm_adapter

    @staticmethod
    def _normalize_threading_mode(*, threading_mode: str | None) -> str | None:
        if threading_mode is None:
            return None

        normalized = threading_mode.strip()
        if normalized == "" or normalized == "none":
            return None
        return normalized

    @staticmethod
    def _normalize_thread_dir(*, thread_dir: str | None) -> str | None:
        if thread_dir is None:
            return None

        normalized = thread_dir.strip().rstrip("/")
        if normalized == "":
            raise ValueError("thread_dir must not be empty")
        return normalized

    def _default_thread_dir_fallback_name(self) -> str:
        return "chat"

    def _default_thread_name(self) -> str:
        return "New Chat"

    def _uses_explicit_thread_dir_for_thread_list(self) -> bool:
        return False

    def _default_thread_dir(self) -> str:
        local_name = self._default_thread_dir_fallback_name()
        participant_name = self._room.local_participant.get_attribute("name")
        if isinstance(participant_name, str) and participant_name.strip() != "":
            local_name = participant_name.strip()

        return self._normalize_thread_dir(
            thread_dir=posixpath.join(".threads", local_name)
        )

    def _thread_list_dir(self) -> str | None:
        if (
            self._thread_dir is not None
            and self._uses_explicit_thread_dir_for_thread_list()
        ):
            return self._thread_dir
        if self._threading_mode == "default-new":
            return self._get_thread_dir()
        return None

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir
        return self._default_thread_dir()

    def get_thread_dir(self) -> str:
        return self._get_thread_dir()

    @staticmethod
    def _sanitize_thread_name(*, value: str) -> str:
        normalized = value.strip()
        if normalized.endswith(".thread"):
            normalized = normalized[: -len(".thread")]

        normalized = re.sub(r"[-_/]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        normalized = re.sub(r"[^A-Za-z0-9 .,!?':()&]+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        if normalized == "":
            return "New Chat"
        if normalized == normalized.lower() or normalized == normalized.upper():
            normalized = normalized.title()
        return normalized[:64].strip() or "New Chat"

    @staticmethod
    def _thread_path_for_name(*, thread_name: str, thread_dir: str) -> str:
        return posixpath.join(thread_dir, f"{thread_name}.thread")

    async def _publish_thread_attributes(self) -> None:
        if self._threading_mode is not None:
            await self._room.local_participant.set_attribute(
                "meshagent.chatbot.threading",
                self._threading_mode,
            )
        thread_dir = self._thread_list_dir()
        if thread_dir is not None:
            await self._room.local_participant.set_attribute(
                "meshagent.chatbot.thread-dir",
                thread_dir,
            )
        thread_list_path = self._thread_list_index_path()
        if thread_list_path is not None:
            await self._room.local_participant.set_attribute(
                "meshagent.chatbot.thread-list",
                thread_list_path,
            )

    async def publish_thread_attributes(self) -> None:
        await self._publish_thread_attributes()

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _thread_list_index_path(self) -> str | None:
        thread_dir = self._thread_list_dir()
        if thread_dir is None:
            return None
        return posixpath.join(thread_dir, "index.threadl")

    def _thread_list_entry_name_for_path(self, *, path: str) -> str:
        filename = posixpath.basename(path.strip())
        if filename.endswith(".thread"):
            filename = filename[: -len(".thread")]
        raw_name = filename.strip()
        if raw_name != "":
            try:
                parsed_uuid = uuid.UUID(raw_name)
            except ValueError:
                parsed_uuid = None
            if parsed_uuid is not None and str(parsed_uuid) == raw_name.lower():
                return self._default_thread_name()

        normalized = self._sanitize_thread_name(value=filename)
        if normalized == "New Chat":
            return self._default_thread_name()
        return normalized

    def _find_thread_list_entry(self, *, path: str) -> Element | None:
        if self._thread_list_document is None:
            return None

        for child in self._thread_list_document.root.get_children():
            if child.tag_name != "thread":
                continue
            if child.get_attribute("path") == path:
                return child

        return None

    def _is_index_managed_path(self, *, path: str) -> bool:
        thread_dir = self._thread_list_dir()
        if thread_dir is None:
            return False

        normalized_path = path.strip().strip("/")
        normalized_dir = thread_dir.strip().strip("/")
        if normalized_path == "" or normalized_dir == "":
            return False
        return normalized_path == normalized_dir or normalized_path.startswith(
            f"{normalized_dir}/"
        )

    def _upsert_thread_list_entry(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> None:
        if self._thread_list_document is None or not self._is_index_managed_path(
            path=path
        ):
            return

        now = self._utc_now_iso()
        provided_name = name.strip() if isinstance(name, str) else ""
        entry = self._find_thread_list_entry(path=path)
        if entry is None:
            resolved_created_at = (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
            resolved_modified_at = (
                modified_at.strip()
                if isinstance(modified_at, str) and modified_at.strip() != ""
                else resolved_created_at
            )
            self._thread_list_document.root.append_child(
                tag_name="thread",
                attributes={
                    "name": (
                        provided_name
                        if provided_name != ""
                        else self._thread_list_entry_name_for_path(path=path)
                    ),
                    "path": path,
                    "created_at": resolved_created_at,
                    "modified_at": resolved_modified_at,
                },
            )
            return

        entry.set_attribute("path", path)
        if provided_name != "":
            entry.set_attribute("name", provided_name)
        else:
            existing_name = entry.get_attribute("name")
            if not isinstance(existing_name, str) or existing_name.strip() == "":
                entry.set_attribute(
                    "name",
                    self._thread_list_entry_name_for_path(path=path),
                )

        existing_created_at = entry.get_attribute("created_at")
        resolved_created_at = (
            existing_created_at.strip()
            if isinstance(existing_created_at, str)
            and existing_created_at.strip() != ""
            else (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
        )
        entry.set_attribute("created_at", resolved_created_at)
        resolved_modified_at = (
            modified_at.strip()
            if isinstance(modified_at, str) and modified_at.strip() != ""
            else now
        )
        entry.set_attribute("modified_at", resolved_modified_at)

    def bump_thread(self, *, path: str, name: str | None = None) -> None:
        entry = self._find_thread_list_entry(path=path)
        resolved_name = name
        if entry is not None:
            existing_name = entry.get_attribute("name")
            if isinstance(existing_name, str) and existing_name.strip() != "":
                resolved_name = None
        self._upsert_thread_list_entry(
            path=path,
            name=resolved_name,
            modified_at=self._utc_now_iso(),
        )

    def _thread_list_entries(self) -> list[Element]:
        if self._thread_list_dir() is None or self._thread_list_document is None:
            return []

        return [
            child
            for child in self._thread_list_document.root.get_children()
            if child.tag_name == "thread"
        ]

    @staticmethod
    def _parse_iso_datetime(*, value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None

        raw = value.strip()
        if raw == "":
            return None

        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _thread_sort_datetime(self, *, entry: Element) -> datetime:
        modified_at = self._parse_iso_datetime(value=entry.get_attribute("modified_at"))
        if modified_at is not None:
            return modified_at
        created_at = self._parse_iso_datetime(value=entry.get_attribute("created_at"))
        if created_at is not None:
            return created_at
        return datetime.min.replace(tzinfo=timezone.utc)

    def _sorted_thread_list_entries(self) -> list[Element]:
        return sorted(
            self._thread_list_entries(),
            key=lambda entry: self._thread_sort_datetime(entry=entry),
            reverse=True,
        )

    @staticmethod
    def _thread_list_slice(
        *,
        entries: list[Element],
        limit: int,
        offset: int,
    ) -> list[Element]:
        normalized_offset = max(0, int(offset))
        normalized_limit = max(1, min(200, int(limit)))
        return entries[normalized_offset : normalized_offset + normalized_limit]

    async def _open_thread_list_document(self) -> None:
        index_path = self._thread_list_index_path()
        if index_path is None:
            return

        if (
            self._thread_list_document is not None
            and self._thread_list_path == index_path
        ):
            return

        self._thread_list_document = await self._room.sync.open(
            path=index_path,
            schema=thread_list_schema,
        )
        self._thread_list_path = index_path

    async def open_thread_list_document(self) -> None:
        await self._open_thread_list_document()

    async def _close_thread_list_document(self) -> None:
        thread_list_path = self._thread_list_path
        if self._thread_list_document is None or thread_list_path is None:
            return

        self._thread_list_document = None
        self._thread_list_path = None
        await self._room.sync.close(path=thread_list_path)

    async def close_thread_list_document(self) -> None:
        await self._close_thread_list_document()

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

    @staticmethod
    def _attachment_name_for_thread_title(*, attachment: str) -> str:
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

    def _normalized_thread_title_attachments(
        self,
        *,
        attachments: Sequence[str] | None,
    ) -> list[str]:
        attachment_names: list[str] = []
        for attachment in attachments or []:
            normalized = self._attachment_name_for_thread_title(attachment=attachment)
            if normalized != "":
                attachment_names.append(normalized)
        return attachment_names

    def _thread_name_input_text(
        self,
        *,
        message_text: str,
        attachments: Sequence[str] | None = None,
    ) -> str:
        parts: list[str] = []
        normalized_message_text = message_text.strip()
        if normalized_message_text != "":
            parts.append(f"Message:\n{normalized_message_text}")

        attachment_names = self._normalized_thread_title_attachments(
            attachments=attachments
        )
        if len(attachment_names) > 0:
            attachment_lines = "\n".join(
                f"- {attachment_name}" for attachment_name in attachment_names
            )
            parts.append(f"Attachments:\n{attachment_lines}")

        if len(parts) == 0:
            return self._default_thread_name()
        return "\n\n".join(parts)

    def _fallback_thread_name(
        self,
        *,
        message_text: str,
        attachments: Sequence[str] | None = None,
    ) -> str:
        normalized_message_text = message_text.strip()
        if normalized_message_text != "":
            return self._sanitize_thread_name(value=normalized_message_text)

        attachment_names = self._normalized_thread_title_attachments(
            attachments=attachments
        )
        if len(attachment_names) > 0:
            return self._sanitize_thread_name(
                value=", ".join(attachment_names[:3]),
            )

        return self._default_thread_name()

    def fallback_thread_name(
        self,
        *,
        message_text: str,
        attachments: Sequence[str] | None = None,
    ) -> str:
        return self._fallback_thread_name(
            message_text=message_text,
            attachments=attachments,
        )

    async def _determine_thread_name(
        self,
        *,
        message_text: str,
        attachments: Sequence[str] | None = None,
        caller_context: dict[str, Any] | None = None,
        on_behalf_of: Participant | None = None,
    ) -> str:
        generated_name = self._fallback_thread_name(
            message_text=message_text,
            attachments=attachments,
        )
        adapter = self.thread_name_adapter()
        if adapter is None:
            return generated_name

        chat_context_json = None
        if isinstance(caller_context, dict):
            candidate = caller_context.get("chat")
            if isinstance(candidate, dict):
                chat_context_json = candidate

        session = adapter.create_session()
        if chat_context_json is not None:
            prior_context = AgentSessionContext.from_json(chat_context_json)
            session.messages.extend(deepcopy(prior_context.messages))
            session.previous_messages.extend(deepcopy(prior_context.previous_messages))
            session.previous_response_id = prior_context.previous_response_id

        cloned_context = session.copy()
        async with cloned_context:
            cloned_context.replace_rules(rules=self._thread_name_rules)
            cloned_context.append_user_message(
                self._thread_name_input_text(
                    message_text=message_text,
                    attachments=attachments,
                )
            )
            try:
                response = await adapter.next(
                    context=cloned_context,
                    room=self._room,
                    model=adapter.default_model(),
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
                        generated_name = self._sanitize_thread_name(
                            value=thread_name,
                        )
            except Exception as ex:
                logger.warning(
                    "unable to auto-generate thread name, using fallback",
                    exc_info=ex,
                )

        return generated_name

    async def _new_thread_path(self) -> str:
        base_path = self._thread_path_for_name(
            thread_name=str(uuid.uuid4()),
            thread_dir=self._get_thread_dir(),
        )
        return await self._next_available_thread_path(base_path=base_path)

    async def new_thread(
        self,
        *,
        message_text: str,
        attachments: Sequence[str] | None = None,
        caller_context: dict[str, Any] | None = None,
        on_behalf_of: Participant | None = None,
    ) -> tuple[str, str]:
        friendly_name = await self._determine_thread_name(
            message_text=message_text,
            attachments=attachments,
            caller_context=caller_context,
            on_behalf_of=on_behalf_of,
        )
        path = await self._new_thread_path()
        self.bump_thread(path=path, name=friendly_name)
        return path, friendly_name
