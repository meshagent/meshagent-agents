from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import json
import logging
from pathlib import PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse
import uuid

from meshagent.api import Participant, RoomClient
from pydantic import TypeAdapter, ValidationError

from .adapter import LLMAdapter
from .messages import (
    AgentFileContent,
    AgentInputContent,
    AgentTextContent,
    TurnStart,
)
from .threaded_channel import ThreadedChannel

logger = logging.getLogger("queue-channel")

_agent_input_content_list_adapter = TypeAdapter(list[AgentInputContent])
_thread_id_template_pattern = re.compile(r"\{([^{}]+)\}")


class QueueChannel(ThreadedChannel):
    def __init__(
        self,
        *,
        room: RoomClient,
        queue_name: str,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        thread_url_scheme: str | None = None,
        thread_path_extension: str = ".thread",
        llm_adapter: LLMAdapter | None = None,
    ) -> None:
        super().__init__(
            room=room,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_url_scheme=thread_url_scheme,
            thread_path_extension=thread_path_extension,
            llm_adapter=llm_adapter,
        )
        normalized_queue_name = queue_name.strip()
        if normalized_queue_name == "":
            raise ValueError("queue_name must not be empty")

        self._queue_name = normalized_queue_name
        self._receive_task: asyncio.Task[None] | None = None

    def _default_thread_dir_fallback_name(self) -> str:
        return "queue"

    def handles(self, message) -> bool:
        del message
        return False

    async def on_start(self) -> None:
        await self.publish_thread_attributes()
        await self.open_thread_list_document()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def on_stop(self) -> None:
        receive_task = self._receive_task
        self._receive_task = None
        if receive_task is not None:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task
        await self._cancel_thread_list_background_tasks()
        await self.close_thread_list_document()

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
                if self._room.is_closed:
                    logger.debug(
                        "stopping queue receive loop after room close for queue %s",
                        self._queue_name,
                    )
                    return
                logger.exception(
                    "queue receive failed for queue %s",
                    self._queue_name,
                )
                await asyncio.sleep(1)
                continue

            if queued_message is None:
                continue

            try:
                logger.info(f"processing message from queue {self._queue_name}")
                payload = self._payload_from_queue_message(message=queued_message)
                content = await self._content_from_payload(payload=payload)
                thread_text = self._thread_text_from_content(
                    payload=payload,
                    content=content,
                )
                self.emit(
                    sender=self._sender_from_payload(payload=payload),
                    payload=TurnStart(
                        type="meshagent.agent.turn.start",
                        thread_id=await self._thread_id_from_payload(
                            payload=payload,
                            thread_text=thread_text,
                        ),
                        content=content,
                        model=self._model_from_payload(payload=payload),
                        instructions=self._instructions_from_payload(payload=payload),
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "queue channel failed to process a message from %s",
                    self._queue_name,
                )

    def _payload_from_queue_message(self, *, message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)

        if isinstance(message, str):
            return {"prompt": message}

        return {
            "prompt": json.dumps(
                message,
                ensure_ascii=False,
                default=str,
            )
        }

    async def _content_from_payload(
        self,
        *,
        payload: dict[str, Any],
    ) -> list[AgentInputContent]:
        source, content = self._content_items_from_payload(payload=payload)
        legacy_prompt_file_text = await self._legacy_prompt_file_text_from_payload(
            payload=payload
        )
        if legacy_prompt_file_text is not None:
            content = [
                AgentTextContent(type="text", text=legacy_prompt_file_text),
                *content,
            ]

        if source == "prompt":
            content = await self._resolve_prompt_file_content_items(content=content)

        if len(content) > 0:
            return content

        return [
            AgentTextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, default=str),
            )
        ]

    def _content_items_from_payload(
        self,
        *,
        payload: dict[str, Any],
    ) -> tuple[str, list[AgentInputContent]]:
        raw_content = payload.get("content")
        if raw_content is not None:
            return (
                "content",
                self._validate_content_items(raw_content=raw_content, key="content"),
            )

        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            if prompt.strip() != "":
                return ("prompt", [AgentTextContent(type="text", text=prompt)])
            if self._has_legacy_prompt_file(payload=payload):
                return ("prompt", [])
        elif isinstance(prompt, (dict, list)):
            return (
                "prompt",
                self._validate_content_items(raw_content=prompt, key="prompt"),
            )

        if self._has_legacy_prompt_file(payload=payload):
            return ("prompt", [])

        logger.warning(
            "prompt property not found on queue message from %s, inserting whole message into context",
            self._queue_name,
        )
        return (
            "fallback",
            [
                AgentTextContent(
                    type="text",
                    text=json.dumps(payload, ensure_ascii=False, default=str),
                )
            ],
        )

    @staticmethod
    def _validate_content_items(
        *,
        raw_content: dict[str, Any] | list[Any],
        key: str,
    ) -> list[AgentInputContent]:
        content_input: list[Any]
        if isinstance(raw_content, dict):
            content_input = [raw_content]
        elif isinstance(raw_content, list):
            content_input = raw_content
        else:
            raise ValueError(f"queue message {key} must be AgentInputContent JSON")

        try:
            return _agent_input_content_list_adapter.validate_python(content_input)
        except ValidationError as exc:
            raise ValueError(
                f"queue message {key} must be AgentInputContent JSON"
            ) from exc

    async def _legacy_prompt_file_text_from_payload(
        self,
        *,
        payload: dict[str, Any],
    ) -> str | None:
        prompt_file = payload.get("prompt_file")
        if not isinstance(prompt_file, str):
            return None

        normalized_prompt_file = prompt_file.strip()
        if normalized_prompt_file == "":
            return None

        try:
            storage_path = self._normalize_room_storage_path(
                path=normalized_prompt_file
            )
        except ValueError:
            logger.warning(
                "invalid prompt_file %s on queue %s",
                normalized_prompt_file,
                self._queue_name,
            )
            return None

        return await self._resolve_prompt_file_content(path=storage_path)

    async def _resolve_prompt_file_content_items(
        self,
        *,
        content: list[AgentInputContent],
    ) -> list[AgentInputContent]:
        resolved_content: list[AgentInputContent] = []
        for item in content:
            resolved_item = await self._resolve_prompt_content_item(item=item)
            if resolved_item is None:
                continue

            resolved_content.append(resolved_item)

        return resolved_content

    async def _resolve_prompt_content_item(
        self,
        *,
        item: AgentInputContent,
    ) -> AgentInputContent | None:
        if not isinstance(item, AgentFileContent):
            return item

        try:
            storage_path = self._room_storage_path_from_url(url=item.url)
        except ValueError:
            logger.warning(
                "invalid room prompt file url %s on queue %s",
                item.url,
                self._queue_name,
            )
            return None

        if storage_path is None:
            return item

        resolved_prompt_text = await self._resolve_prompt_file_content(
            path=storage_path
        )
        if resolved_prompt_text is None:
            return None

        return AgentTextContent(
            type="text",
            text=resolved_prompt_text,
        )

    async def _resolve_prompt_file_content(
        self,
        *,
        path: str,
    ) -> str | None:
        if not await self._room.storage.exists(path=path):
            logger.warning(
                "prompt_file %s not found in room storage for queue %s",
                path,
                self._queue_name,
            )
            return None

        prompt_file_content = await self._room.storage.download(path=path)
        try:
            return prompt_file_content.data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "prompt_file %s is not valid utf-8 for queue %s",
                path,
                self._queue_name,
            )
            return None

    @staticmethod
    def _normalize_room_storage_path(*, path: str) -> str:
        normalized = PurePosixPath("/" + path.strip().lstrip("/")).as_posix().strip("/")
        if normalized == "":
            raise ValueError("room storage path must reference a non-root path")

        if any(part in {".", ".."} for part in PurePosixPath(normalized).parts):
            raise ValueError("room storage path cannot contain '.' or '..' segments")

        return normalized

    @classmethod
    def _room_storage_path_from_url(cls, *, url: str) -> str | None:
        parsed_url = urlparse(url)
        if parsed_url.scheme == "":
            return None
        if parsed_url.scheme != "room":
            return None

        raw_path = f"{parsed_url.netloc}{parsed_url.path}"
        return cls._normalize_room_storage_path(path=raw_path)

    def _has_legacy_prompt_file(self, *, payload: dict[str, Any]) -> bool:
        prompt_file = payload.get("prompt_file")
        return isinstance(prompt_file, str) and prompt_file.strip() != ""

    def _thread_text_from_content(
        self,
        *,
        payload: dict[str, Any],
        content: list[AgentInputContent],
    ) -> str:
        text_parts = [
            item.text
            for item in content
            if isinstance(item, AgentTextContent) and item.text.strip() != ""
        ]
        if len(text_parts) > 0:
            return "\n\n".join(text_parts)

        return json.dumps(payload, ensure_ascii=False, default=str)

    async def _thread_id_from_payload(
        self,
        *,
        payload: dict[str, Any],
        thread_text: str,
    ) -> str:
        for key in ("thread_id", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip() != "":
                return self._expand_thread_id_template(value=value.strip())

        path, _ = await self.new_thread(
            message_text=thread_text,
        )
        return path

    def _expand_thread_id_template(self, *, value: str) -> str:
        now = self._now()
        year = now.strftime("%Y")
        short_year = now.strftime("%y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        second = now.strftime("%S")

        # Keep the legacy short month/minute forms exact-case to avoid the
        # MM/mm ambiguity, and offer unambiguous aliases that are matched
        # case-insensitively.
        legacy_exact_replacements = {
            "YYYY": year,
            "YY": short_year,
            "MM": month,
            "DD": day,
            "HH": hour,
            "mm": minute,
            "SS": second,
            "ss": second,
        }
        case_insensitive_replacements = {
            "YYYY": year,
            "YEAR": year,
            "YY": short_year,
            "YR": short_year,
            "DD": day,
            "DAY": day,
            "HH": hour,
            "HOUR": hour,
            "SS": second,
            "SECOND": second,
            "SEC": second,
            "MONTH": month,
            "MON": month,
            "MINUTE": minute,
            "MIN": minute,
        }

        def replace_match(match: re.Match[str]) -> str:
            token = match.group(1)
            exact_replacement = legacy_exact_replacements.get(token)
            if exact_replacement is not None:
                return exact_replacement

            return case_insensitive_replacements.get(
                token.upper(),
                match.group(0),
            )

        return _thread_id_template_pattern.sub(replace_match, value)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _sender_from_payload(self, *, payload: dict[str, Any]) -> Participant | None:
        sender_name = payload.get("sender_name")
        if not isinstance(sender_name, str) or sender_name.strip() == "":
            sender_name = payload.get("from")
            if not isinstance(sender_name, str) or sender_name.strip() == "":
                return None

        normalized_sender = sender_name.strip()
        return Participant(
            id=f"queue:{self._queue_name}:{uuid.uuid5(uuid.NAMESPACE_URL, normalized_sender)}",
            attributes={"name": normalized_sender},
        )

    @staticmethod
    def _model_from_payload(*, payload: dict[str, Any]) -> str | None:
        model = payload.get("model")
        if not isinstance(model, str) or model.strip() == "":
            return None
        return model

    @staticmethod
    def _instructions_from_payload(*, payload: dict[str, Any]) -> str | None:
        instructions = payload.get("instructions")
        if not isinstance(instructions, str) or instructions.strip() == "":
            return None
        return instructions
