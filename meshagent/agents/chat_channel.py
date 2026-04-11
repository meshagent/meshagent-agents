from __future__ import annotations

import logging
from typing import Any

from meshagent.api import RoomMessage
from pydantic import BaseModel
from meshagent.tools import Toolkit

from .legacy_chat_channel import LegacyChatChannel
from .messages import (
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentMessage,
    ApproveAgentToolCall,
    ClearThread,
    RejectAgentToolCall,
    TurnInterrupt,
    TurnStart,
    TurnSteer,
)
from .process import Message

logger = logging.getLogger("chat-channel")


class _AgentMessageEnvelope(BaseModel):
    payload: dict[str, Any]


_INBOUND_AGENT_MESSAGE_MODELS: dict[str, type[AgentMessage]] = {
    AGENT_MESSAGE_TURN_START: TurnStart,
    AGENT_MESSAGE_TURN_STEER: TurnSteer,
    AGENT_MESSAGE_TURN_INTERRUPT: TurnInterrupt,
    AGENT_MESSAGE_THREAD_CLEAR: ClearThread,
    AGENT_MESSAGE_TOOL_CALL_APPROVE: ApproveAgentToolCall,
    AGENT_MESSAGE_TOOL_CALL_REJECT: RejectAgentToolCall,
}

_OUTBOUND_AGENT_MESSAGE_TYPES = {
    AGENT_EVENT_THREAD_CLEARED,
}


class ChatChannel(LegacyChatChannel):
    def get_exposed_toolkits(self) -> list[Toolkit]:
        return [self.make_toolkit()]

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        if message_type in _INBOUND_AGENT_MESSAGE_MODELS:
            return False
        if message_type in _OUTBOUND_AGENT_MESSAGE_TYPES:
            return True
        return message_type.startswith("meshagent.agent.")

    async def on_start(self) -> None:
        await super().on_start()
        await self.room.local_participant.set_attribute("supports_agent_messages", True)

    async def on_stop(self) -> None:
        await self.room.local_participant.set_attribute("supports_agent_messages", None)
        await super().on_stop()

    async def on_message(self, message: Message) -> None:
        payload = message.data.model_dump(mode="json")
        for participant in self._open_participants(thread_id=message.data.thread_id):
            if participant.id == self.room.local_participant.id:
                continue

            self.room.messaging.send_message_nowait(
                to=participant,
                type="agent-message",
                message={"payload": payload},
            )

    @staticmethod
    def _normalize_agent_message_payload(
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_payload = dict(payload)
        message_type = normalized_payload.get("type")
        if message_type not in {
            AGENT_MESSAGE_TURN_START,
            AGENT_MESSAGE_TURN_STEER,
        }:
            return normalized_payload

        raw_content = normalized_payload.get("content")
        if not isinstance(raw_content, list):
            return normalized_payload

        normalized_content: list[Any] = []
        content_changed = False
        for item in raw_content:
            if not isinstance(item, dict):
                normalized_content.append(item)
                continue

            if item.get("type") != "file":
                normalized_content.append(item)
                continue

            url = item.get("url")
            if not isinstance(url, str):
                normalized_content.append(item)
                continue

            normalized_url = LegacyChatChannel._normalize_attachment_url(path=url)
            if normalized_url is None:
                normalized_content.append(item)
                continue

            if normalized_url == url:
                normalized_content.append(item)
                continue

            normalized_content.append({**item, "url": normalized_url})
            content_changed = True

        if content_changed:
            normalized_payload["content"] = normalized_content

        return normalized_payload

    @staticmethod
    def _thread_id_from_room_message(*, message: RoomMessage) -> str | None:
        if message.type != "agent-message":
            return LegacyChatChannel._thread_id_from_room_message(message=message)

        try:
            envelope = _AgentMessageEnvelope.model_validate(message.message)
        except Exception:
            return None

        raw_thread_id = envelope.payload.get("thread_id")
        if not isinstance(raw_thread_id, str):
            return None

        thread_id = raw_thread_id.strip()
        if thread_id == "":
            return None

        return thread_id

    def _agent_message_from_room_message(
        self,
        *,
        message: RoomMessage,
    ) -> AgentMessage | None:
        if message.type != "agent-message":
            logger.debug(
                "ignoring chat room message of type %s because it has no agent equivalent",
                message.type,
            )
            return None

        envelope = _AgentMessageEnvelope.model_validate(message.message)
        payload = self._normalize_agent_message_payload(payload=envelope.payload)
        message_type = payload.get("type")
        if not isinstance(message_type, str):
            raise ValueError("agent-message payload must include a string type")

        model = _INBOUND_AGENT_MESSAGE_MODELS.get(message_type)
        if model is None:
            raise ValueError(f"unsupported agent-message payload type: {message_type}")

        return model.model_validate(payload)
