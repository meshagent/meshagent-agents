from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import posixpath
import uuid
from datetime import datetime, timezone
from typing import Any

from meshagent.api import MeshDocument, Participant, RoomClient

from .legacy_chat_channel import LegacyChatChannel
from .messages import AgentTextContent, TurnStart
from .process import Channel
from .thread_schema import thread_list_schema

logger = logging.getLogger("queue-channel")


class QueueChannel(Channel):
    def __init__(
        self,
        *,
        room: RoomClient,
        queue_name: str,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
    ) -> None:
        super().__init__()
        normalized_queue_name = queue_name.strip()
        if normalized_queue_name == "":
            raise ValueError("queue_name must not be empty")

        self._room = room
        self._queue_name = normalized_queue_name
        self._threading_mode = LegacyChatChannel._normalize_threading_mode(
            threading_mode=threading_mode
        )
        self._thread_dir = LegacyChatChannel._normalize_thread_dir(
            thread_dir=thread_dir
        )
        self._receive_task: asyncio.Task[None] | None = None
        self._thread_list_document: MeshDocument | None = None
        self._thread_list_path: str | None = None

    @property
    def room(self) -> RoomClient:
        return self._room

    def handles(self, message) -> bool:
        del message
        return False

    async def on_start(self) -> None:
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
        await self._open_thread_list_document()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def on_stop(self) -> None:
        receive_task = self._receive_task
        self._receive_task = None
        if receive_task is not None:
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task
        await self._close_thread_list_document()

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
                prompt = self._prompt_from_payload(payload=payload)
                self.emit(
                    sender=self._sender_from_payload(payload=payload),
                    payload=TurnStart(
                        type="meshagent.agent.turn.start",
                        thread_id=await self._thread_id_from_payload(
                            payload=payload,
                            prompt=prompt,
                        ),
                        content=[
                            AgentTextContent(
                                type="text",
                                text=prompt,
                            )
                        ],
                        toolkits=self._toolkits_from_payload(payload=payload),
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

    def _prompt_from_payload(self, *, payload: dict[str, Any]) -> str:
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip() != "":
            return prompt

        logger.warning(
            "prompt property not found on queue message from %s, inserting whole message into context",
            self._queue_name,
        )
        return json.dumps(payload, ensure_ascii=False, default=str)

    async def _thread_id_from_payload(
        self,
        *,
        payload: dict[str, Any],
        prompt: str,
    ) -> str:
        for key in ("thread_id", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()

        if self._threading_mode == "default-new":
            return await self._new_thread_path(prompt=prompt)

        return posixpath.join(self._get_thread_dir(), f"{uuid.uuid4()}.thread")

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir

        local_name = self._room.local_participant.get_attribute("name")
        if isinstance(local_name, str) and local_name.strip() != "":
            return LegacyChatChannel._normalize_thread_dir(
                thread_dir=posixpath.join(".threads", local_name.strip())
            )

        return LegacyChatChannel._normalize_thread_dir(thread_dir=".threads/queue")

    def _thread_list_dir(self) -> str | None:
        if self._threading_mode != "default-new":
            return None
        return self._get_thread_dir()

    def _thread_list_index_path(self) -> str | None:
        thread_dir = self._thread_list_dir()
        if thread_dir is None:
            return None
        return posixpath.join(thread_dir, "index.threadl")

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _find_thread_list_entry(self, *, path: str):
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
        resolved_name = (
            name.strip() if isinstance(name, str) and name.strip() != "" else None
        )
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
        entry = self._find_thread_list_entry(path=path)
        if entry is None:
            self._thread_list_document.root.append_child(
                tag_name="thread",
                attributes={
                    "name": resolved_name or "New Chat",
                    "path": path,
                    "created_at": resolved_created_at,
                    "modified_at": resolved_modified_at,
                },
            )
            return

        if resolved_name is not None:
            entry.set_attribute("name", resolved_name)
        entry.set_attribute("path", path)
        entry.set_attribute("created_at", resolved_created_at)
        entry.set_attribute("modified_at", resolved_modified_at)

    def _record_new_thread_in_index(
        self,
        *,
        path: str,
        name: str,
    ) -> None:
        now = self._utc_now_iso()
        self._upsert_thread_list_entry(
            path=path,
            name=name,
            created_at=now,
            modified_at=now,
        )

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

    async def _close_thread_list_document(self) -> None:
        thread_list_path = self._thread_list_path
        if self._thread_list_document is None or thread_list_path is None:
            return

        self._thread_list_document = None
        self._thread_list_path = None
        await self._room.sync.close(path=thread_list_path)

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

    async def _new_thread_path(self, *, prompt: str) -> str:
        path = LegacyChatChannel._thread_path_for_name(
            thread_name=str(uuid.uuid4()),
            thread_dir=self._get_thread_dir(),
        )
        path = await self._next_available_thread_path(base_path=path)
        self._record_new_thread_in_index(
            path=path,
            name=self._fallback_thread_name(prompt=prompt),
        )
        return path

    @staticmethod
    def _fallback_thread_name(*, prompt: str) -> str:
        normalized_prompt = prompt.strip()
        if normalized_prompt != "":
            return LegacyChatChannel._sanitize_thread_name(value=normalized_prompt)
        return "New Chat"

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
    def _toolkits_from_payload(
        *, payload: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        raw_toolkits = payload.get("tools")
        if not isinstance(raw_toolkits, list):
            return None

        toolkits = [toolkit for toolkit in raw_toolkits if isinstance(toolkit, dict)]
        if len(toolkits) == 0:
            return None
        return toolkits

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
