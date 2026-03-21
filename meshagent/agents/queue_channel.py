from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any
import uuid

from meshagent.api import Participant, RoomClient

from .adapter import LLMAdapter
from .messages import AgentTextContent, TurnStart
from .threaded_channel import ThreadedChannel

logger = logging.getLogger("queue-channel")


class QueueChannel(ThreadedChannel):
    def __init__(
        self,
        *,
        room: RoomClient,
        queue_name: str,
        threading_mode: str | None = None,
        thread_dir: str | None = None,
        llm_adapter: LLMAdapter | None = None,
    ) -> None:
        super().__init__(
            room=room,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
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

        path, _ = await self.new_thread(
            message_text=prompt,
        )
        return path

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
