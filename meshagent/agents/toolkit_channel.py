from __future__ import annotations

import asyncio
import posixpath
import uuid
from dataclasses import dataclass, field
from typing import Any

from meshagent.api import Participant, RoomClient, RoomException
from meshagent.tools import (
    FunctionTool,
    ToolContext,
    Toolkit,
    ToolkitBuilder,
)

from .legacy_chat_channel import LegacyChatChannel
from .messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_TURN_START,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallApprovalRequested,
    RejectAgentToolCall,
    TurnEnded,
    TurnStart,
    TurnStarted,
)
from .process import Channel, Message
from .toolkit_schema import build_tools_property_schema


@dataclass(slots=True)
class _PendingToolkitTurn:
    source_message_id: str
    sender: Participant | None
    thread_id: str
    future: asyncio.Future[str]
    turn_id: str | None = None
    text_by_item_id: dict[str, str] = field(default_factory=dict)
    completed_text_parts: list[str] = field(default_factory=list)


class ToolkitChannel(Channel):
    def __init__(
        self,
        *,
        room: RoomClient,
        toolkit_name: str,
        tool_name: str | None = None,
        thread_dir: str | None = None,
        public: bool = True,
        toolkit_builders: list[ToolkitBuilder] | None = None,
    ) -> None:
        super().__init__()
        normalized_toolkit_name = toolkit_name.strip()
        if normalized_toolkit_name == "":
            raise ValueError("toolkit_name must not be empty")

        normalized_tool_name = (
            tool_name.strip()
            if isinstance(tool_name, str) and tool_name.strip() != ""
            else f"run_{normalized_toolkit_name}_task"
        )

        self._room = room
        self._toolkit_name = normalized_toolkit_name
        self._tool_name = normalized_tool_name
        self._thread_dir = LegacyChatChannel._normalize_thread_dir(
            thread_dir=thread_dir
        )
        self._public = public
        self._toolkit_builders = list(toolkit_builders or [])
        self._pending_by_source_message_id: dict[str, _PendingToolkitTurn] = {}
        self._pending_by_turn_id: dict[str, _PendingToolkitTurn] = {}
        self._input_schema = self._build_input_schema()

    @property
    def room(self) -> RoomClient:
        return self._room

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_EVENT_TURN_STARTED,
            AGENT_EVENT_TURN_ENDED,
            AGENT_EVENT_TEXT_CONTENT_STARTED,
            AGENT_EVENT_TEXT_CONTENT_DELTA,
            AGENT_EVENT_TEXT_CONTENT_ENDED,
            AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
        }

    def get_agent_toolkits(self) -> list[Toolkit]:
        return []

    def get_exposed_toolkits(self) -> list[Toolkit]:
        return [self.make_toolkit()]

    async def on_stop(self) -> None:
        self._fail_pending_turns(
            error=RoomException("toolkit channel stopped before the turn completed")
        )

    async def on_message(self, message: Message) -> None:
        data = message.data

        if isinstance(data, TurnStarted):
            pending = self._pending_by_source_message_id.pop(
                data.source_message_id,
                None,
            )
            if pending is None:
                return
            pending.turn_id = data.turn_id
            self._pending_by_turn_id[data.turn_id] = pending
            return

        if isinstance(data, AgentTextContentStarted):
            pending = self._pending_by_turn_id.get(data.turn_id)
            if pending is None:
                return
            pending.text_by_item_id[data.item_id] = ""
            return

        if isinstance(data, AgentTextContentDelta):
            pending = self._pending_by_turn_id.get(data.turn_id)
            if pending is None:
                return
            current_text = pending.text_by_item_id.get(data.item_id, "")
            pending.text_by_item_id[data.item_id] = current_text + data.text
            return

        if isinstance(data, AgentTextContentEnded):
            pending = self._pending_by_turn_id.get(data.turn_id)
            if pending is None:
                return
            completed_text = pending.text_by_item_id.pop(data.item_id, "")
            if completed_text.strip() != "":
                pending.completed_text_parts.append(completed_text)
            return

        if isinstance(data, AgentToolCallApprovalRequested):
            pending = self._pending_by_turn_id.get(data.turn_id)
            if pending is None:
                return

            self.emit(
                sender=pending.sender,
                payload=RejectAgentToolCall(
                    type=AGENT_MESSAGE_TOOL_CALL_REJECT,
                    thread_id=data.thread_id,
                    turn_id=data.turn_id,
                    item_id=data.item_id,
                ),
            )
            return

        if not isinstance(data, TurnEnded):
            return

        pending = self._pending_by_turn_id.pop(data.turn_id, None)
        if pending is None:
            return

        if data.error is not None:
            if not pending.future.done():
                pending.future.set_exception(RoomException(data.error.message))
            return

        if not pending.future.done():
            pending.future.set_result(self._response_text(pending=pending))

    def make_toolkit(self) -> Toolkit:
        local_name_value = self._room.local_participant.get_attribute("name")
        local_name = (
            local_name_value.strip()
            if isinstance(local_name_value, str) and local_name_value.strip() != ""
            else self._toolkit_name
        )
        return Toolkit(
            name=self._toolkit_name,
            description=f"send a message to {local_name} and return the reply text",
            public=self._public,
            tools=[self._make_run_task_tool()],
        )

    def _build_input_schema(self) -> dict[str, Any]:
        tools_schema, defs = build_tools_property_schema(
            toolkit_builders=self._toolkit_builders
        )
        if tools_schema is None:
            tools_schema = {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                    {"type": "null"},
                ]
            }

        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "a prompt that will be sent to the agent",
                },
                "path": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "optional thread path to continue an existing thread",
                },
                "thread_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "alias for path",
                },
                "model": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "optional model override for this turn",
                },
                "instructions": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "optional instructions override for this turn",
                },
                "tools": tools_schema,
            },
        }
        if len(defs) > 0:
            schema["$defs"] = defs
        return schema

    def _get_thread_dir(self) -> str:
        if self._thread_dir is not None:
            return self._thread_dir

        local_name_value = self._room.local_participant.get_attribute("name")
        local_name = (
            local_name_value.strip()
            if isinstance(local_name_value, str) and local_name_value.strip() != ""
            else self._toolkit_name
        )
        return LegacyChatChannel._normalize_thread_dir(
            thread_dir=posixpath.join(".threads", local_name)
        )

    def _fallback_thread_name(self, *, prompt: str) -> str:
        normalized_prompt = prompt.strip()
        if normalized_prompt == "":
            return "Task"
        return LegacyChatChannel._sanitize_thread_name(value=normalized_prompt)

    async def _next_available_thread_path(self, *, base_path: str) -> str:
        try:
            exists = await self._room.storage.exists(path=base_path)
        except Exception:
            return base_path

        if not exists:
            return base_path

        thread_dir, filename = posixpath.split(base_path)
        base_name = (
            filename[: -len(".thread")] if filename.endswith(".thread") else filename
        )
        for index in range(2, 1000):
            candidate = posixpath.join(thread_dir, f"{base_name} {index}.thread")
            try:
                if not await self._room.storage.exists(path=candidate):
                    return candidate
            except Exception:
                return candidate

        return posixpath.join(thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}.thread")

    async def _resolve_thread_id(
        self,
        *,
        prompt: str,
        path: str | None,
        thread_id: str | None,
    ) -> str:
        normalized_path = path.strip() if isinstance(path, str) else ""
        normalized_thread_id = thread_id.strip() if isinstance(thread_id, str) else ""

        if (
            normalized_path != ""
            and normalized_thread_id != ""
            and normalized_path != normalized_thread_id
        ):
            raise RoomException("path and thread_id must match when both are provided")

        if normalized_path != "":
            return normalized_path
        if normalized_thread_id != "":
            return normalized_thread_id

        base_path = LegacyChatChannel._thread_path_for_name(
            thread_name=self._fallback_thread_name(prompt=prompt),
            thread_dir=self._get_thread_dir(),
        )
        return await self._next_available_thread_path(base_path=base_path)

    def _response_text(self, *, pending: _PendingToolkitTurn) -> str:
        parts = [part for part in pending.completed_text_parts if part.strip() != ""]
        for text in pending.text_by_item_id.values():
            if text.strip() != "":
                parts.append(text)
        return "\n\n".join(parts).strip()

    def _register_pending_turn(
        self,
        *,
        sender: Participant | None,
        thread_id: str,
        source_message_id: str,
    ) -> _PendingToolkitTurn:
        pending = _PendingToolkitTurn(
            source_message_id=source_message_id,
            sender=sender,
            thread_id=thread_id,
            future=asyncio.get_running_loop().create_future(),
        )
        self._pending_by_source_message_id[source_message_id] = pending
        return pending

    def _remove_pending_turn(self, *, pending: _PendingToolkitTurn) -> None:
        self._pending_by_source_message_id.pop(pending.source_message_id, None)
        if pending.turn_id is not None:
            self._pending_by_turn_id.pop(pending.turn_id, None)

    def _fail_pending_turns(self, *, error: Exception) -> None:
        pending_turns = [
            *self._pending_by_source_message_id.values(),
            *[
                pending
                for pending in self._pending_by_turn_id.values()
                if pending.source_message_id not in self._pending_by_source_message_id
            ],
        ]
        self._pending_by_source_message_id.clear()
        self._pending_by_turn_id.clear()
        for pending in pending_turns:
            if not pending.future.done():
                pending.future.set_exception(error)

    def _make_run_task_tool(self) -> FunctionTool:
        outer = self

        class RunTaskTool(FunctionTool):
            def __init__(self) -> None:
                super().__init__(
                    name=outer._tool_name,
                    title=f"Run {outer._toolkit_name} Task",
                    description=(
                        "send a prompt to the agent and return the final assistant text"
                    ),
                    input_schema=outer._input_schema,
                )

            async def execute(
                self,
                context: ToolContext,
                *,
                prompt: str,
                path: str | None = None,
                thread_id: str | None = None,
                model: str | None = None,
                instructions: str | None = None,
                tools: list[dict[str, Any]] | None = None,
            ) -> str:
                if outer.supervisor is None:
                    raise RoomException(
                        "toolkit channel must be attached to a supervisor before it can be used"
                    )

                sender = context.on_behalf_of or context.caller
                resolved_thread_id = await outer._resolve_thread_id(
                    prompt=prompt,
                    path=path,
                    thread_id=thread_id,
                )
                turn_start = TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id=resolved_thread_id,
                    content=[AgentTextContent(type="text", text=prompt)],
                    toolkits=tools,
                    model=model,
                    instructions=instructions,
                )
                pending = outer._register_pending_turn(
                    sender=sender,
                    thread_id=resolved_thread_id,
                    source_message_id=turn_start.message_id,
                )

                try:
                    outer.emit(sender=sender, payload=turn_start)
                    return await pending.future
                except asyncio.CancelledError:
                    outer._remove_pending_turn(pending=pending)
                    raise

        return RunTaskTool()
