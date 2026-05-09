from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import mimetypes
import re
import shlex
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from urllib.parse import urlparse

from meshagent.api import Participant
from meshagent.api.messaging import FileContent
from meshagent.agents.adapter import LLMAdapter, ToolCallApprovalRequest
from meshagent.agents.context import AgentSessionContext
from meshagent.tools import ToolContext, Toolkit
from opentelemetry import trace
from pydantic_core import from_json as pydantic_core_from_json
from .thread_adapter import default_format_message
from .thread_storage import ThreadStorage
from .thread_status_publisher import ThreadStatusPublisher
from .version import __version__ as agents_version
from .messages import (
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentError,
    AgentFileContent,
    AgentContextCompacted,
    AgentLLMMessage,
    AgentMessage,
    AgentTextContent,
    AgentToolCallArgumentsDelta,
    AgentToolCallApprovalRequested,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallPending,
    AgentToolCallStarted,
    AgentImageGenerationCompleted,
    AgentImageGenerationFailed,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentThreadEvent,
    AgentContextWindowUsage,
    AgentUsageUpdated,
    ApproveAgentToolCall,
    CapabilitiesRequest,
    CapabilitiesResponse,
    ClearThread,
    CloseThread,
    OpenThread,
    RejectAgentToolCall,
    ToolkitCapabilities,
    ToolkitToolCapabilities,
    ToolChoice,
    TurnEnded,
    TurnInterrupt,
    TurnInterrupted,
    TurnInterruptAccepted,
    TurnStartAccepted,
    TurnStartRejected,
    TurnStart,
    TurnSteerAccepted,
    TurnSteered,
    TurnSteer,
    TurnSteerRejected,
    TurnStarted,
)
from .shell_semantics import analyze_shell_command

logger = logging.getLogger("agent-process")
tracer = trace.get_tracer("meshagent.agents")
_THREAD_STATUS_ACTIVE_STATES = {
    "queued",
    "in_progress",
    "running",
    "pending",
    "searching",
}
_THREAD_STATUS_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_THREAD_ADAPTER_REQUEST_MESSAGE_TYPES = frozenset(
    {
        AGENT_MESSAGE_THREAD_CLEAR,
        AGENT_MESSAGE_CAPABILITIES_REQUEST,
        AGENT_MESSAGE_CAPABILITIES_RESPONSE,
        AGENT_MESSAGE_THREAD_CLOSE,
        AGENT_MESSAGE_THREAD_OPEN,
        AGENT_MESSAGE_TOOL_CALL_APPROVE,
        AGENT_MESSAGE_TOOL_CALL_REJECT,
        AGENT_MESSAGE_TURN_INTERRUPT,
        AGENT_MESSAGE_TURN_START,
        AGENT_MESSAGE_TURN_STEER,
    }
)

LifecycleState = Literal["stopped", "starting", "started", "stopping", "failed"]
ChannelState = LifecycleState
SessionInitializer = Callable[[], Awaitable[AgentSessionContext]]
TurnInstructionsProvider = Callable[[Participant | None], Awaitable[str | None]]
TurnToolkitsBuilder = Callable[
    [Participant | None, str, list["TurnStart | TurnSteer"]],
    Awaitable[list[Toolkit]],
]
ContentDownload = Callable[[str], Awaitable[FileContent]]


@runtime_checkable
class ThreadStorageLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ContentScheme:
    prefix: str
    download: ContentDownload

    def __post_init__(self) -> None:
        if self.prefix == "":
            raise ValueError("content scheme prefix cannot be empty")


@dataclass(frozen=True, slots=True)
class _StatusToolCall:
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None
    state: str
    argument_delta_bytes: int = 0


_APPLY_PATCH_PATH_RES = (
    re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (?P<path>.+)$", re.MULTILINE),
    re.compile(r"^(?:\+\+\+ b/|--- a/)(?P<path>.+)$", re.MULTILINE),
)


def _humanize_tool_name(name: str) -> str:
    normalized = name.strip().replace("_", " ").replace("-", " ")
    if normalized == "":
        return ""
    return " ".join(part.lower() for part in normalized.split())


def _command_text(*, value: Any, multiline: bool = False) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        string_items = [item.strip() for item in value if isinstance(item, str)]
        string_items = [item for item in string_items if item != ""]
        if len(string_items) == len(value) and len(string_items) > 0:
            if multiline:
                return "\n".join(string_items)
            with contextlib.suppress(ValueError, TypeError):
                return shlex.join(string_items)
            return " ".join(string_items)

        parts = [_command_text(value=item, multiline=multiline) for item in value]
        parts = [part for part in parts if part != ""]
        if len(parts) == 0:
            return ""
        return "\n".join(parts) if multiline else " ".join(parts)

    if isinstance(value, dict):
        for key in ("command", "commands", "cmd", "code", "text", "value"):
            if key not in value:
                continue
            text = _command_text(
                value=value[key],
                multiline=multiline or key == "commands",
            )
            if text != "":
                return text
        nested = value.get("content")
        if nested is not None:
            return _command_text(value=nested, multiline=multiline)

    return ""


def _first_nested_text(*, value: Any, keys: tuple[str, ...]) -> str:
    key_set = {key.lower() for key in keys}

    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() not in key_set:
                continue
            text = _command_text(value=nested, multiline=key.endswith("s"))
            if text != "":
                return text

        for nested in value.values():
            text = _first_nested_text(value=nested, keys=keys)
            if text != "":
                return text

    if isinstance(value, list):
        for nested in value:
            text = _first_nested_text(value=nested, keys=keys)
            if text != "":
                return text

    return ""


def _extract_tool_command(*, tool: str, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""

    action = arguments.get("action")
    if isinstance(action, dict):
        for key in ("commands", "command", "cmd"):
            if key not in action:
                continue
            text = _command_text(value=action[key], multiline=key == "commands")
            if text != "":
                return text

    for key in ("commands", "command", "cmd"):
        if key not in arguments:
            continue
        text = _command_text(value=arguments[key], multiline=key == "commands")
        if text != "":
            return text

    if tool == "code_interpreter":
        text = _command_text(value=arguments.get("code"))
        if text != "":
            return text

    return _first_nested_text(
        value=arguments,
        keys=("command", "commands", "cmd", "shell_command", "raw_command"),
    )


def _extract_web_query(*, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""

    queries = arguments.get("queries")
    if isinstance(queries, list):
        values = [item.strip() for item in queries if isinstance(item, str)]
        values = [item for item in values if item != ""]
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            return ", ".join(values)

    return _first_nested_text(value=arguments, keys=("query", "queries", "q"))


def _extract_apply_patch_text(*, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    return _first_nested_text(value=arguments, keys=("patch", "input", "diff"))


def _apply_patch_path(*, patch: str) -> str:
    for pattern in _APPLY_PATCH_PATH_RES:
        match = pattern.search(patch)
        if match is None:
            continue
        path = match.group("path").strip()
        if path != "":
            return path
    return ""


def _status_total_bytes(total_bytes: int) -> int | None:
    return total_bytes if total_bytes > 100 else None


def _tool_argument_snapshot_bytes(arguments: dict[str, Any] | None) -> int:
    if arguments is None or len(arguments) == 0:
        return 0
    return len(
        json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _merge_tool_arguments(
    *,
    current: dict[str, Any] | None,
    update: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(current) if current is not None else {}
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_tool_arguments(current=existing, update=value)
        else:
            merged[key] = value
    return merged


def _partial_json_tool_arguments(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped == "" or stripped[0] not in "{[":
        return None

    try:
        parsed = pydantic_core_from_json(
            stripped.encode("utf-8"),
            allow_partial=True,
        )
    except ValueError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _shell_delta_arguments(
    *,
    current: dict[str, Any] | None,
    command: str,
) -> dict[str, Any]:
    merged = deepcopy(current) if current is not None else {}
    action = merged.get("action")
    if isinstance(action, dict):
        existing_commands = action.get("commands")
        if isinstance(existing_commands, list):
            action["commands"] = [command]
        else:
            action["command"] = command
        return merged

    merged["command"] = command
    return merged


def _tool_arguments_from_delta_text(
    *,
    tool: str,
    current: dict[str, Any] | None,
    text: str,
) -> dict[str, Any] | None:
    partial_arguments = _partial_json_tool_arguments(text)
    if partial_arguments is not None:
        return _merge_tool_arguments(current=current, update=partial_arguments)

    normalized_tool = tool.strip().lower()
    if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
        command = text.strip()
        if command != "":
            return _shell_delta_arguments(current=current, command=command)

    return None


_STATUS_TEXT_REPLACEMENTS = {
    "Preparing Command": "Preparing",
    "Running Command": "Running command",
    "Ran Command": "Ran command",
    "Command Cancelled": "Command cancelled",
    "Web Search Cancelled": "Web search cancelled",
    "Patch Cancelled": "Patch cancelled",
    "Image Generation Cancelled": "Image generation cancelled",
}


def _normalize_status_text(text: str) -> str:
    return _STATUS_TEXT_REPLACEMENTS.get(text, text)


def _storage_status_text(
    *,
    state: str,
    toolkit: str,
    tool: str,
    arguments: dict[str, Any] | None,
) -> str | None:
    if arguments is None or toolkit.strip().lower() != "storage":
        return None

    normalized_tool = tool.strip().lower()
    if normalized_tool == "grep_file":
        path = arguments.get("path")
        if not isinstance(path, str) or path.strip() == "":
            if state == "pending":
                return "Preparing"
            return None
        normalized_path = path.strip()
        if state == "pending":
            return f"Preparing to search {normalized_path}"
        if state in _THREAD_STATUS_ACTIVE_STATES:
            return f"Searching {normalized_path}"
        if state == "failed":
            return f"Attempted to search file {normalized_path}"
        if state == "cancelled":
            return f"Cancelled searching file {normalized_path}"
        return f"Searched {normalized_path}"

    if normalized_tool not in {"read_file", "write_file"}:
        return None

    path = arguments.get("path")
    if not isinstance(path, str) or path.strip() == "":
        if state == "pending":
            return "Preparing"
        return None
    normalized_path = path.strip()
    operation = "read" if normalized_tool == "read_file" else "write"
    present = "Reading" if operation == "read" else "Writing"
    past = "Read" if operation == "read" else "Wrote"

    if state == "pending":
        return f"Preparing to {operation} {normalized_path}"
    if state in _THREAD_STATUS_ACTIVE_STATES:
        return f"{present} {normalized_path}"
    if state == "failed":
        return f"Attempted to {operation} file {normalized_path}"
    if state == "cancelled":
        return f"Cancelled {operation}ing file {normalized_path}"
    return f"{past} {normalized_path}"


def _tool_status_text(
    *,
    state: str,
    toolkit: str,
    tool: str,
    arguments: dict[str, Any] | None,
) -> str:
    storage_text = _storage_status_text(
        state=state,
        toolkit=toolkit,
        tool=tool,
        arguments=arguments,
    )
    if storage_text is not None:
        return storage_text

    normalized_tool = tool.strip().lower()
    if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
        command = _extract_tool_command(tool=normalized_tool, arguments=arguments)
        if state == "pending" and command == "":
            return "Preparing"
        return (
            analyze_shell_command(command=command)
            .display.phase_for_state(state=state)
            .headline
        )

    if normalized_tool == "web_search":
        query = _extract_web_query(arguments=arguments)
        if state == "pending":
            if query == "":
                return "Preparing"
            return "Preparing web search"
        if state in _THREAD_STATUS_ACTIVE_STATES:
            return "Searching the web"
        if state == "failed":
            return "Attempted to search the web"
        if state == "cancelled":
            return "Web search cancelled"
        if query != "":
            return f"Searched for {query}"
        return "Searched the web"

    if normalized_tool == "apply_patch":
        patch = _extract_apply_patch_text(arguments=arguments)
        path = _apply_patch_path(patch=patch)
        if path != "":
            if state == "pending":
                return f"Preparing to edit {path}"
            if state == "failed":
                return f"Attempted to patch {path}"
            if state == "cancelled":
                return f"Patch cancelled: {path}"
            if state in _THREAD_STATUS_ACTIVE_STATES:
                return f"Editing {path}"
            return f"Edited {path}"
        if state == "pending":
            if patch == "":
                return "Preparing"
            return "Preparing patch"
        if state == "failed":
            return "Attempted to patch"
        if state == "cancelled":
            return "Patch cancelled"
        if state in _THREAD_STATUS_ACTIVE_STATES:
            return "Applying patch"
        return "Applied patch"

    if normalized_tool == "image_generation":
        if state == "pending":
            return "Preparing image generation"
        if state in _THREAD_STATUS_ACTIVE_STATES:
            return "Generating image"
        if state == "failed":
            return "Attempted to generate image"
        if state == "cancelled":
            return "Image generation cancelled"
        return "Generated image"

    humanized = _humanize_tool_name(tool)
    if state == "pending":
        return f"Preparing {humanized}" if humanized != "" else "Preparing tool call"
    if state in _THREAD_STATUS_ACTIVE_STATES:
        return f"Calling {humanized}" if humanized != "" else "Calling tool"
    if state == "failed":
        return (
            f"Attempted to call {humanized}"
            if humanized != ""
            else "Attempted to call tool"
        )
    if state == "cancelled":
        return f"{humanized} cancelled" if humanized != "" else "Tool call cancelled"
    return f"Called {humanized}" if humanized != "" else "Called tool"


@dataclass(slots=True)
class Message:
    data: AgentMessage
    sender: Participant | None = None
    source: Channel | AgentProcess | None = None


_MessageT = TypeVar("_MessageT", bound=AgentMessage)


def _coerce_message_data(data: AgentMessage, model: type[_MessageT]) -> _MessageT:
    if isinstance(data, model):
        return data

    return model.model_validate(data.model_dump(mode="python"))


class Channel:
    def __init__(self) -> None:
        self._supervisor: AgentSupervisor | None = None
        self._state: ChannelState = "stopped"
        self._run_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._lifecycle_lock = asyncio.Lock()

    @property
    def state(self) -> ChannelState:
        return self._state

    @property
    def supervisor(self) -> AgentSupervisor | None:
        return self._supervisor

    def handles(self, message: Message) -> bool:
        del message
        return True

    def send(self, message: Message) -> None:
        if self._state != "started" or self._supervisor is None:
            logger.debug("dropping channel message while channel is not started")
            return

        if self._stop.is_set():
            logger.debug("dropping channel message during shutdown")
            return

        if self.handles(message):
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueShutDown:
                logger.debug("dropping channel message after queue shutdown")

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        supervisor = self.supervisor
        if supervisor is None:
            return

        supervisor.send(Message(data=payload, sender=sender, source=self))

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        del participant
        del payload
        return False

    def get_agent_toolkits(self) -> list[Toolkit]:
        return []

    def get_exposed_toolkits(self) -> list[Toolkit]:
        return []

    async def on_start(self) -> None:
        return None

    async def on_message(self, message: Message) -> None:
        del message
        return None

    async def on_stop(self) -> None:
        return None

    async def run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(asyncio.QueueShutDown):
                message = await self._queue.get()
                await self.on_message(message)

    async def start(self, supervisor: AgentSupervisor) -> None:
        async with self._lifecycle_lock:
            if self._state not in {"stopped", "failed"}:
                raise ValueError("already started")

            if self._supervisor is not None:
                raise ValueError("already started")

            self._state = "starting"
            self._stop.clear()

            try:
                self._supervisor = supervisor

                await self.on_start()

                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
            except Exception:
                self._state = "failed"
                self._supervisor = None
                self._run_task = None
                raise

    async def stop(self, supervisor: AgentSupervisor) -> None:
        async with self._lifecycle_lock:
            if self._supervisor is None or self._supervisor is not supervisor:
                raise ValueError("not started")

            self._state = "stopping"

            try:
                self._stop.set()
                self._queue.shutdown()
                if self._run_task is not None:
                    await self._run_task
                await self.on_stop()
            except Exception:
                logger.exception("channel failed during stop")
                self._state = "failed"
                self._supervisor = None
                self._run_task = None
                raise
            else:
                self._supervisor = None
                self._run_task = None
                self._state = "stopped"


SupervisorState = LifecycleState


class AgentSupervisor:
    def __init__(self) -> None:
        self.channels: list[Channel] = []
        self.processes: list[AgentProcess] = []
        self._stop = asyncio.Event()
        self._state: SupervisorState = "stopped"
        self._run_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._lifecycle_lock = asyncio.Lock()
        self._route_lock = asyncio.Lock()

    @property
    def state(self) -> SupervisorState:
        return self._state

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def stop_channel(self, channel: Channel) -> None:
        if channel in self.channels:
            self.channels.remove(channel)

    def add_process(self, process: AgentProcess) -> None:
        self.processes.append(process)

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_thread_process"
        )

    def send(self, message: Message) -> None:
        if self._stop.is_set() or self._state == "stopping":
            logger.debug("dropping supervisor message during shutdown")
            return

        try:
            self._queue.put_nowait(message)
        except asyncio.QueueShutDown:
            logger.debug("dropping supervisor message after queue shutdown")

    def emit(
        self,
        *,
        sender: Participant | None,
        payload: AgentMessage,
    ) -> None:
        self.send(Message(data=payload, sender=sender))

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    async def route(self, message: Message) -> None:
        if self._stop.is_set() or self._state == "stopping":
            logger.debug("dropping supervisor message during shutdown")
            return

        if self._state != "started":
            self.send(message)
            return

        async with self._route_lock:
            await self._route(message)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._state not in {"stopped", "failed"}:
                raise ValueError("already started")

            self._state = "starting"
            self._stop.clear()

            try:
                await self.on_start()
                await self._ensure_children_started()
                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
            except Exception:
                logger.exception("agent supervisor failed during start")
                self._state = "failed"
                self._run_task = None
                raise

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._run_task is None:
                raise ValueError("not started")

            self._state = "stopping"

            try:
                self._stop.set()
                self._queue.shutdown()
                await self._run_task
                await self.on_stop()
            except Exception:
                logger.exception("agent supervisor failed during stop")
                self._state = "failed"
                self._run_task = None
                raise
            else:
                self._run_task = None
                self._state = "stopped"

    def _process_for_thread(self, *, thread_id: str) -> AgentProcess | None:
        for process in self.processes:
            if process.thread_id == thread_id:
                return process
        return None

    def _process_for_turn(self, *, turn_id: str) -> AgentProcess | None:
        for process in self.processes:
            if process.turn_id == turn_id:
                return process
        return None

    @staticmethod
    def _copy_message(message: Message, *, data: AgentMessage) -> Message:
        return Message(
            data=data,
            sender=message.sender,
            source=message.source,
        )

    def _send_to_channels(self, message: Message) -> None:
        for channel in self.channels:
            if message.source is channel:
                continue
            channel.send(message)

    def _send_to_processes(
        self,
        message: Message,
        *,
        processes: list[AgentProcess] | None = None,
    ) -> None:
        target_processes = self.processes if processes is None else processes
        for process in target_processes:
            if message.source is process:
                continue
            process.send(message)

    async def _ensure_routing_processes_started(
        self,
        *,
        processes: list[AgentProcess] | None,
    ) -> list[AgentProcess] | None:
        if processes is None:
            return None

        started_processes: list[AgentProcess] = []
        for process in processes:
            if process.state == "stopped":
                try:
                    await process.start(self)
                except Exception:
                    logger.exception(
                        "process %s failed during routed start; dropping message",
                        process.__class__.__name__,
                    )
                    continue

            if process.state == "started" and process.supervisor is self:
                started_processes.append(process)

        return started_processes

    def _thread_process_creation_rejection(self, *, error: Exception) -> AgentError:
        return AgentError(
            message=str(error) or error.__class__.__name__,
            code="thread_process_creation_failed",
        )

    def _create_thread_process_for_route(
        self, *, thread_id: str
    ) -> tuple[AgentProcess | None, AgentError | None]:
        try:
            process = self.create_thread_process(thread_id)
        except Exception as exc:
            logger.exception(
                "failed to create process for thread %s; rejecting message",
                thread_id,
            )
            return None, self._thread_process_creation_rejection(error=exc)
        self.add_process(process)
        return process, None

    def _emit_turn_start_rejected(
        self,
        *,
        turn_start: TurnStart,
        sender: Participant | None,
        error: AgentError,
    ) -> None:
        self._send_to_channels(
            Message(
                data=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=turn_start.thread_id,
                    source_message_id=turn_start.message_id,
                    error=error,
                ),
                sender=sender,
            )
        )

    def _emit_turn_steer_rejected(
        self,
        *,
        turn_steer: TurnSteer,
        sender: Participant | None,
        error: AgentError,
    ) -> None:
        self._send_to_channels(
            Message(
                data=TurnSteerRejected(
                    type=AGENT_EVENT_TURN_STEER_REJECTED,
                    thread_id=turn_steer.thread_id,
                    turn_id=turn_steer.turn_id,
                    source_message_id=turn_steer.message_id,
                    error=error,
                ),
                sender=sender,
            )
        )

    async def _route(self, message: Message) -> None:
        routed_message = message
        target_processes: list[AgentProcess] | None = None
        message_type = message.data.type
        if message_type == AGENT_MESSAGE_TURN_START:
            turn_start = _coerce_message_data(message.data, TurnStart)
            process = self._process_for_thread(thread_id=turn_start.thread_id)
            if process is None:
                process, rejection = self._create_thread_process_for_route(
                    thread_id=turn_start.thread_id
                )
                if process is None:
                    if rejection is not None:
                        self._emit_turn_start_rejected(
                            turn_start=turn_start,
                            sender=message.sender,
                            error=rejection,
                        )
                    return

            routed_message = self._copy_message(
                message,
                data=turn_start,
            )
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TURN_STEER:
            turn_steer = _coerce_message_data(message.data, TurnSteer)
            process = self._process_for_thread(thread_id=turn_steer.thread_id)
            if process is None:
                process, rejection = self._create_thread_process_for_route(
                    thread_id=turn_steer.thread_id
                )
                if process is None:
                    if rejection is not None:
                        self._emit_turn_steer_rejected(
                            turn_steer=turn_steer,
                            sender=message.sender,
                            error=rejection,
                        )
                    return
            routed_message = self._copy_message(
                message,
                data=turn_steer,
            )
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TURN_INTERRUPT:
            turn_interrupt = _coerce_message_data(message.data, TurnInterrupt)
            process = self._process_for_thread(thread_id=turn_interrupt.thread_id)
            routed_message = self._copy_message(
                message,
                data=turn_interrupt,
            )
            if process is not None and process.turn_id == turn_interrupt.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_TOOL_CALL_APPROVE:
            approval = _coerce_message_data(message.data, ApproveAgentToolCall)
            process = self._process_for_thread(thread_id=approval.thread_id)
            routed_message = self._copy_message(
                message,
                data=approval,
            )
            if process is not None and process.turn_id == approval.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_TOOL_CALL_REJECT:
            rejection = _coerce_message_data(message.data, RejectAgentToolCall)
            process = self._process_for_thread(thread_id=rejection.thread_id)
            routed_message = self._copy_message(
                message,
                data=rejection,
            )
            if process is not None and process.turn_id == rejection.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_THREAD_CLEAR:
            clear_thread = _coerce_message_data(message.data, ClearThread)
            process = self._process_for_thread(thread_id=clear_thread.thread_id)
            if process is None:
                process, _ = self._create_thread_process_for_route(
                    thread_id=clear_thread.thread_id
                )
                if process is None:
                    return

            routed_message = self._copy_message(
                message,
                data=clear_thread,
            )
            target_processes = [process]

        elif message_type in {AGENT_MESSAGE_THREAD_OPEN, AGENT_MESSAGE_THREAD_CLOSE}:
            if message_type == AGENT_MESSAGE_THREAD_OPEN:
                thread_message = _coerce_message_data(message.data, OpenThread)
            else:
                thread_message = _coerce_message_data(message.data, CloseThread)

            process = self._process_for_thread(thread_id=thread_message.thread_id)
            if process is None and message_type == AGENT_MESSAGE_THREAD_OPEN:
                process, _ = self._create_thread_process_for_route(
                    thread_id=thread_message.thread_id
                )
            if process is None:
                return

            routed_message = self._copy_message(
                message,
                data=thread_message,
            )
            target_processes = [process]

        target_processes = await self._ensure_routing_processes_started(
            processes=target_processes
        )
        self._send_to_channels(routed_message)
        self._send_to_processes(routed_message, processes=target_processes)

    async def _ensure_children_started(self) -> None:
        for channel in self.channels:
            if channel.state == "stopped":
                try:
                    await channel.start(self)
                except Exception:
                    logger.exception(
                        "channel %s failed during start; continuing",
                        channel.__class__.__name__,
                    )

        for process in self.processes:
            if process.state == "stopped":
                try:
                    await process.start(self)
                except Exception:
                    logger.exception(
                        "process %s failed during start; continuing",
                        process.__class__.__name__,
                    )

    async def _stop_children(self) -> None:
        errors: list[Exception] = []

        for process in self.processes:
            if process.supervisor is not self:
                continue
            if process.state != "stopped":
                try:
                    await process.stop(self)
                except (
                    Exception
                ) as exc:  # pragma: no cover - error path tested via caller state
                    errors.append(exc)

        for channel in self.channels:
            if channel.supervisor is not self:
                continue
            if channel.state != "stopped":
                try:
                    await channel.stop(self)
                except (
                    Exception
                ) as exc:  # pragma: no cover - error path tested via caller state
                    errors.append(exc)

        if errors:
            raise errors[0]

    async def run(self) -> None:
        try:
            while not self._stop.is_set():
                await self._ensure_children_started()

                with contextlib.suppress(asyncio.QueueShutDown):
                    message = await self._queue.get()
                    async with self._route_lock:
                        await self._route(message)
        finally:
            await self._stop_children()


ProcessState = LifecycleState


class AgentProcess:
    def __init__(
        self,
        supervisor: AgentSupervisor | None = None,
        *,
        thread_id: str | None = None,
        thread_storage: ThreadStorage | None = None,
        thread_adapter: ThreadStorage | None = None,
    ) -> None:
        del supervisor
        if thread_storage is None:
            thread_storage = thread_adapter
        elif thread_adapter is not None:
            raise ValueError("thread_storage and thread_adapter cannot both be set")

        if thread_storage is not None:
            if thread_id is None:
                thread_id = thread_storage.path

        self._supervisor: AgentSupervisor | None = None
        self._thread_id = thread_id
        self._thread_storage = thread_storage
        self._state: ProcessState = "stopped"
        self._run_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._lifecycle_lock = asyncio.Lock()

    def send(self, message: Message) -> None:
        if self._state != "started" or self._supervisor is None:
            logger.debug("dropping process message while process is not started")
            return

        if self._stop.is_set():
            logger.debug("dropping process message during shutdown")
            return

        message = self._message_with_sender_name(message)

        if self._should_mirror_to_thread_storage(message=message):
            thread_storage = self._thread_storage
            if thread_storage is not None:
                thread_storage.push_message(
                    message=message.data,
                    sender=message.sender,
                )

        if self.handles(message):
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueShutDown:
                logger.debug("dropping process message after queue shutdown")

    def handles(self, message: Message) -> bool:
        return False

    def _should_mirror_to_thread_storage(self, *, message: Message) -> bool:
        thread_storage = self._thread_storage
        if thread_storage is None:
            return False

        thread_id = self._thread_id
        if thread_id is None or message.data.thread_id != thread_id:
            return False

        if message.source is self:
            return False

        message_type = message.data.type
        if not message_type.startswith("meshagent.agent."):
            return False

        return message_type not in _THREAD_ADAPTER_REQUEST_MESSAGE_TYPES

    @property
    def state(self) -> ProcessState:
        return self._state

    @property
    def supervisor(self) -> AgentSupervisor | None:
        return self._supervisor

    @property
    def thread_id(self) -> str | None:
        return self._thread_id

    @property
    def thread_storage(self) -> ThreadStorage | None:
        return self._thread_storage

    @property
    def thread_adapter(self) -> ThreadStorage | None:
        return self._thread_storage

    @property
    def turn_id(self) -> str | None:
        return None

    @staticmethod
    def _sender_name(sender: Participant | None) -> str | None:
        if sender is None:
            return None

        raw_name = sender.get_attribute("name")
        if not isinstance(raw_name, str):
            return None

        name = raw_name.strip()
        if name == "":
            return None

        return name

    @classmethod
    def _message_with_sender_name(cls, message: Message) -> Message:
        sender_name = cls._sender_name(message.sender)
        if sender_name is None:
            return message

        existing_sender_name = message.data.sender_name
        if existing_sender_name is not None and existing_sender_name.strip() != "":
            return message

        return Message(
            data=message.data.model_copy(update={"sender_name": sender_name}),
            sender=message.sender,
            source=message.source,
        )

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        supervisor = self.supervisor
        if supervisor is None:
            return

        supervisor.send(Message(data=payload, sender=sender, source=self))

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    async def start(self, supervisor: AgentSupervisor) -> None:
        async with self._lifecycle_lock:
            if self._state not in {"stopped", "failed"}:
                raise ValueError("already started")

            if self._supervisor is not None:
                raise ValueError("already started")

            self._state = "starting"
            self._stop.clear()

            try:
                self._supervisor = supervisor
                thread_storage = self._thread_storage
                if isinstance(thread_storage, ThreadStorageLifecycle):
                    await thread_storage.start()
                await self.on_start()
                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
            except Exception:
                self._state = "failed"
                thread_storage = self._thread_storage
                if isinstance(thread_storage, ThreadStorageLifecycle):
                    with contextlib.suppress(Exception):
                        await thread_storage.stop()
                self._supervisor = None
                self._run_task = None
                raise

    async def on_message(self, message: Message) -> None:
        del message
        return None

    async def run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(asyncio.QueueShutDown):
                message = await self._queue.get()
                await self.on_message(message)

    async def stop(self, supervisor: AgentSupervisor) -> None:
        async with self._lifecycle_lock:
            if self._supervisor is None or self._supervisor is not supervisor:
                raise ValueError("not started")

            self._state = "stopping"

            try:
                self._stop.set()
                self._queue.shutdown()
                if self._run_task is not None:
                    await self._run_task
                await self.on_stop()
                thread_storage = self._thread_storage
                if isinstance(thread_storage, ThreadStorageLifecycle):
                    await thread_storage.stop()
            except Exception:
                logger.exception("agent process failed during stop")
                self._state = "failed"
                self._supervisor = None
                self._run_task = None
                raise
            else:
                self._supervisor = None
                self._run_task = None
                self._state = "stopped"


@dataclass(slots=True)
class _QueuedTurn:
    sender: Participant | None
    request: TurnStart
    queued_messages: list[_QueuedTurnMessage] = field(default_factory=list)


@dataclass(slots=True)
class _QueuedTurnMessage:
    sender: Participant | None
    request: TurnStart | TurnSteer
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LLMAgentProcess(AgentProcess):
    def __init__(
        self,
        *,
        thread_id: str,
        participant: Participant,
        llm_adapter: LLMAdapter,
        toolkits: Optional[list[Toolkit]] = None,
        thread_storage: ThreadStorage | None = None,
        thread_adapter: ThreadStorage | None = None,
        thread_status_publisher: ThreadStatusPublisher | None = None,
        format_message: Callable[..., str] | None = None,
        session_initializer: SessionInitializer | None = None,
        turn_instructions_provider: TurnInstructionsProvider | None = None,
        turn_toolkits_builder: TurnToolkitsBuilder | None = None,
    ) -> None:
        if thread_storage is None:
            thread_storage = thread_adapter
        elif thread_adapter is not None:
            raise ValueError("thread_storage and thread_adapter cannot both be set")

        super().__init__(thread_id=thread_id, thread_storage=thread_storage)
        self._thread_status_publisher = thread_status_publisher
        self._format_message = format_message or default_format_message
        self.llm_adapter = llm_adapter
        self._turn_id: str | None = None
        self._last_usage_update: AgentUsageUpdated | None = None
        self._handlers: dict[str, Callable[[Message], Awaitable[None]]] = {
            AGENT_MESSAGE_THREAD_OPEN: self.on_thread_open,
            AGENT_MESSAGE_THREAD_CLOSE: self.on_thread_close,
            AGENT_MESSAGE_TURN_START: self.on_turn_start,
            AGENT_MESSAGE_TURN_STEER: self.on_turn_steer,
            AGENT_MESSAGE_TURN_INTERRUPT: self.on_turn_interrupt,
            AGENT_MESSAGE_CAPABILITIES_REQUEST: self.on_capabilities_request,
            AGENT_MESSAGE_TOOL_CALL_APPROVE: self.on_tool_call_approve,
            AGENT_MESSAGE_TOOL_CALL_REJECT: self.on_tool_call_reject,
        }
        self._session_context: AgentSessionContext | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._active_next_task: asyncio.Task[Any] | None = None
        self._pending_turns: asyncio.Queue[_QueuedTurn] = asyncio.Queue()
        self._priority_turn: _QueuedTurn | None = None
        self._active_turn_queue: asyncio.Queue[_QueuedTurnMessage] | None = None
        self._active_turn_queue_updated: asyncio.Event | None = None
        self._toolkits = list(toolkits or [])
        self._participant = participant
        self._content_schemes: list[ContentScheme] = []
        self._pending_tool_call_approvals: dict[str, asyncio.Future[bool]] = {}
        self._active_turn_sender: Participant | None = None
        self._pending_status_messages: list[_QueuedTurnMessage] = []
        self._interrupt_requested_turn_id: str | None = None
        self._interrupt_source_message_id: str | None = None
        self._active_turn_toolkit_client_options: dict[str, dict[str, Any]] = {}
        self._active_turn_tool_choice: ToolChoice | None = None
        self._status_tool_calls_by_item_id: dict[str, _StatusToolCall] = {}
        self._status_tool_argument_delta_bytes_by_item_id: dict[str, int] = {}
        self._status_tool_argument_delta_text_by_item_id: dict[str, str] = {}
        self._status_text_by_item_id: dict[str, str] = {}
        self._latest_status_text: str | None = None
        self._session_initializer = session_initializer
        self._turn_instructions_provider = turn_instructions_provider
        self._turn_toolkits_builder = turn_toolkits_builder
        self.llm_adapter.set_tool_call_approval_handler(
            self._request_tool_call_approval
        )

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def session_context(self) -> AgentSessionContext | None:
        return self._session_context

    @property
    def thread_storage(self) -> ThreadStorage | None:
        return super().thread_storage

    @property
    def toolkits(self) -> list[Toolkit]:
        return self._toolkits

    @property
    def thread_status_publisher(self) -> ThreadStatusPublisher | None:
        return self._thread_status_publisher

    def _agent_message_with_participant_name(
        self,
        payload: AgentMessage,
    ) -> AgentMessage:
        participant_name = self._sender_name(self._participant)
        if participant_name is None:
            return payload

        sender_name = payload.sender_name
        if sender_name is not None and sender_name.strip() != "":
            return payload

        return payload.model_copy(update={"sender_name": participant_name})

    def register_content_scheme(self, scheme: ContentScheme) -> None:
        self._content_schemes.append(scheme)

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        payload = self._agent_message_with_participant_name(payload)
        thread_storage = self.thread_storage
        if thread_storage is not None:
            thread_storage.push_message(message=payload, sender=sender)

        super().emit(sender=sender, payload=payload)

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        if message_type not in self._handlers:
            return False

        return message.data.thread_id == self._thread_id

    async def on_session_context_created(self) -> None:
        if self._session_initializer is None:
            return None

        session_context = self.session_context
        if session_context is None:
            return None

        initialized_context = await self._session_initializer()
        session_context.messages.extend(initialized_context.messages)
        session_context.previous_messages.extend(initialized_context.previous_messages)
        session_context.previous_response_id = initialized_context.previous_response_id
        session_context.instructions = initialized_context.instructions
        return None

    # used to restore any persisted agent state
    async def on_restore_session_context(
        self,
        turn_id: str,
        session_context: AgentSessionContext,
    ) -> None:
        del turn_id
        del session_context
        return None

    async def ensure_session_context(
        self, *, turn_id: str | None
    ) -> AgentSessionContext:
        restore_turn_id = turn_id or ""
        with tracer.start_as_current_span("agent.turn.context.load") as span:
            span.set_attribute("thread_id", self.thread_id)
            if turn_id is not None:
                span.set_attribute("turn_id", turn_id)
            span.set_attribute("context.cached", self._session_context is not None)
            if self._session_context is None:
                self._session_context = self.llm_adapter.create_session()

                with tracer.start_as_current_span("agent.turn.context.initialize"):
                    await self.on_session_context_created()

                thread_storage = self.thread_storage
                if thread_storage is not None:
                    thread_storage.restore_session_context(
                        context=self._session_context,
                        llm_adapter=self.llm_adapter,
                    )

                with tracer.start_as_current_span("agent.turn.context.restore_hooks"):
                    await self.on_restore_session_context(
                        restore_turn_id, self._session_context
                    )

                with tracer.start_as_current_span("agent.turn.context.start"):
                    await self._session_context.start()

        return self._session_context

    @staticmethod
    def _usage_context_window_total(value: float) -> int | None:
        if not math.isfinite(value):
            return None
        return max(0, int(value))

    @staticmethod
    def _last_response_usage(usage: object) -> dict[str, float]:
        if not isinstance(usage, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in usage.items():
            if (
                isinstance(key, str)
                and isinstance(value, int | float)
                and math.isfinite(value)
            ):
                out[key] = float(value)
        return out

    @staticmethod
    def _last_response_context_used_tokens(value: object) -> int:
        if isinstance(value, int | float) and math.isfinite(value):
            return max(0, int(value))
        return 0

    def _cached_usage_update(
        self,
        *,
        turn_id: str | None,
    ) -> AgentUsageUpdated | None:
        last_usage_update = self._last_usage_update
        if last_usage_update is None:
            return None
        return last_usage_update.model_copy(update={"turn_id": turn_id})

    async def _build_usage_update(
        self,
        *,
        session: AgentSessionContext,
        model: str,
        toolkits: list[Toolkit],
        turn_id: str | None,
        include_usage: bool = True,
        restore_context_from_storage: bool = False,
    ) -> AgentUsageUpdated:
        thread_id = self.thread_id
        if thread_id is None:
            raise RuntimeError("usage update requested without an active thread")

        usage = (
            self._last_response_usage(
                session.metadata.get("last_response_flattened_usage")
            )
            if include_usage
            else {}
        )
        context = session
        restored_context_from_storage = False
        if restore_context_from_storage:
            thread_storage = self.thread_storage
            if thread_storage is not None:
                context = self.llm_adapter.create_session()
                context.instructions = session.instructions
                context.metadata.update(session.metadata)
                context.previous_messages.extend(session.previous_messages)
                context.previous_response_id = session.previous_response_id
                thread_storage.restore_session_context(
                    context=context,
                    llm_adapter=self.llm_adapter,
                )
                restored_context_from_storage = True

        used_tokens = self._last_response_context_used_tokens(
            session.metadata.get("last_response_context_used_tokens")
        )
        count_missing_restored_usage = (
            restored_context_from_storage
            and include_usage
            and len(usage) == 0
            and (
                len(context.messages) > 0
                or len(context.previous_messages) > 0
                or context.previous_response_id is not None
            )
        )
        if count_missing_restored_usage:
            try:
                used_tokens = await self.llm_adapter.get_input_tokens(
                    context=context,
                    model=model,
                    toolkits=toolkits,
                )
            except Exception:
                logger.debug("failed to compute context window usage", exc_info=True)
        compaction_threshold = session.metadata.get(
            "last_response_compaction_threshold"
        )
        if isinstance(compaction_threshold, int):
            used_tokens = min(used_tokens, compaction_threshold)
        try:
            context_window_size = self.llm_adapter.context_window_size(model)
        except Exception:
            logger.debug("failed to compute context window size", exc_info=True)
            context_window_size = float("inf")
        try:
            context_management_mode = self.llm_adapter.context_management_mode()
        except Exception:
            logger.debug("failed to read context management mode", exc_info=True)
            context_management_mode = None
        try:
            configured_compaction_threshold = self.llm_adapter.compaction_threshold(
                model
            )
        except Exception:
            logger.debug("failed to read compaction threshold", exc_info=True)
            configured_compaction_threshold = None
        return AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id=thread_id,
            turn_id=turn_id,
            usage=usage,
            context_window=AgentContextWindowUsage(
                used_tokens=max(0, int(used_tokens)),
                total_tokens=self._usage_context_window_total(context_window_size),
                compaction_mode=context_management_mode,
                compaction_threshold=configured_compaction_threshold,
            ),
        )

    async def _emit_usage_update(
        self,
        *,
        session: AgentSessionContext,
        model: str,
        toolkits: list[Toolkit],
        turn_id: str | None,
        sender: Participant | None,
        include_usage: bool = True,
        restore_context_from_storage: bool = False,
    ) -> None:
        try:
            usage_update = await self._build_usage_update(
                session=session,
                model=model,
                toolkits=toolkits,
                turn_id=turn_id,
                include_usage=include_usage,
                restore_context_from_storage=restore_context_from_storage,
            )
        except Exception:
            logger.debug("failed to publish agent usage update", exc_info=True)
            return

        self._last_usage_update = usage_update
        self.emit(sender=sender, payload=usage_update)

    def _emit_cached_usage_update(
        self,
        *,
        turn_id: str | None,
        sender: Participant | None,
        source: Channel | AgentProcess | None = None,
    ) -> bool:
        usage_update = self._cached_usage_update(turn_id=turn_id)
        if usage_update is None:
            return False
        self._send_thread_open_usage_update(
            usage_update=usage_update,
            sender=sender,
            source=source,
        )
        return True

    def _send_thread_open_usage_update(
        self,
        *,
        usage_update: AgentUsageUpdated,
        sender: Participant | None,
        source: Channel | AgentProcess | None = None,
    ) -> None:
        if (
            sender is not None
            and isinstance(source, Channel)
            and source.send_agent_message_to_participant(
                participant=sender,
                payload=self._agent_message_with_participant_name(usage_update),
            )
        ):
            return

        super().emit(
            sender=sender,
            payload=self._agent_message_with_participant_name(usage_update),
        )

    async def _compact_context_if_needed(
        self,
        *,
        session: AgentSessionContext,
        model: str,
        publish_event: Callable[[AgentMessage], None],
    ) -> None:
        if not self.llm_adapter.needs_compaction(context=session):
            return

        thread_id = self.thread_id
        if thread_id is None:
            return

        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is not None:
            await thread_status_publisher.set_thread_status(status="Compacting context")

        try:
            await self.llm_adapter.compact(context=session, model=model)
        finally:
            if thread_status_publisher is not None:
                await thread_status_publisher.set_thread_status(status="Thinking")

        publish_event(
            AgentContextCompacted(
                type=AGENT_EVENT_CONTEXT_COMPACTED,
                thread_id=thread_id,
                checkpoint_id=str(uuid.uuid4()),
                path=thread_id,
                through_sequence=0,
                created_at=datetime.now(timezone.utc).isoformat(),
                messages=deepcopy(session.messages),
            )
        )

    def _record_accepted_turns(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        thread_storage = self.thread_storage
        if thread_storage is None:
            return

        for queued_message in queued_messages:
            thread_storage.push_message(
                message=queued_message.request,
                sender=queued_message.sender,
            )

    @staticmethod
    def _sender_name(sender: Participant | None) -> str | None:
        if sender is None:
            return None

        raw_name = sender.get_attribute("name")
        if not isinstance(raw_name, str):
            return None

        name = raw_name.strip()
        if name == "":
            return None

        return name

    def _format_live_turn_message(
        self,
        *,
        sender: Participant | None,
        message: str,
    ) -> str:
        sender_name = self._sender_name(sender)
        if sender_name is None or message == "":
            return message

        iso_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return self._format_message(
            user_name=sender_name,
            message=message,
            iso_timestamp=iso_timestamp,
        )

    def _file_attachment_message(
        self,
        *,
        sender: Participant | None,
        url: str,
    ) -> str:
        sender_name = self._sender_name(sender)
        if sender_name is None:
            return f"the user attached a file available at {url}"

        return f"{sender_name} attached a file available at {url}"

    @classmethod
    def _pending_status_message_payload(
        cls,
        *,
        queued_message: _QueuedTurnMessage,
    ) -> dict[str, Any]:
        request = queued_message.request
        content = request.model_dump(mode="json").get("content", [])
        return {
            "message_id": request.message_id,
            "message_type": request.type,
            "sender_name": cls._sender_name(queued_message.sender),
            "created_at": queued_message.created_at.isoformat(),
            "content": content if isinstance(content, list) else [],
        }

    async def _sync_pending_status_messages(self) -> None:
        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is None:
            return

        await thread_status_publisher.set_pending_messages(
            pending_messages=[
                self._pending_status_message_payload(queued_message=queued_message)
                for queued_message in self._pending_status_messages
            ]
        )

    async def _add_pending_status_messages(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        if len(queued_messages) == 0:
            return

        self._pending_status_messages.extend(queued_messages)
        await self._sync_pending_status_messages()

    async def _remove_pending_status_messages(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        if len(queued_messages) == 0 or len(self._pending_status_messages) == 0:
            return

        removed_message_ids = {
            queued_message.request.message_id for queued_message in queued_messages
        }
        remaining_messages = [
            queued_message
            for queued_message in self._pending_status_messages
            if queued_message.request.message_id not in removed_message_ids
        ]
        if len(remaining_messages) == len(self._pending_status_messages):
            return

        self._pending_status_messages = remaining_messages
        await self._sync_pending_status_messages()

    async def _clear_pending_status_messages(self) -> None:
        if len(self._pending_status_messages) == 0:
            return

        self._pending_status_messages.clear()
        await self._sync_pending_status_messages()

    def _cancel_pending_tool_call_approvals(self) -> None:
        pending_approvals = list(self._pending_tool_call_approvals.values())
        self._pending_tool_call_approvals.clear()
        for future in pending_approvals:
            if not future.done():
                future.cancel()

    async def _request_tool_call_approval(
        self,
        context: ToolContext,
        request: ToolCallApprovalRequest,
    ) -> bool:
        del context

        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            raise RuntimeError("tool call approval requested without an active turn")

        approval_future: asyncio.Future[bool] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_tool_call_approvals[request.item_id] = approval_future
        self.emit(
            sender=self._active_turn_sender,
            payload=AgentToolCallApprovalRequested(
                type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
                thread_id=thread_id,
                turn_id=turn_id,
                item_id=request.item_id,
                toolkit=request.toolkit,
                tool=request.tool,
                arguments=request.arguments,
            ),
        )

        try:
            return await approval_future
        finally:
            existing_future = self._pending_tool_call_approvals.get(request.item_id)
            if existing_future is approval_future:
                del self._pending_tool_call_approvals[request.item_id]

    async def _resolve_tool_call_approval(
        self, *, item_id: str, approved: bool
    ) -> None:
        approval_future = self._pending_tool_call_approvals.get(item_id)
        if approval_future is None or approval_future.done():
            return

        approval_future.set_result(approved)

    def _resolve_turn_toolkit_client_options(
        self,
        *,
        turns: list[TurnStart | TurnSteer],
    ) -> dict[str, dict[str, Any]]:
        configured_options: dict[str, dict[str, Any]] | None = None
        for turn in turns:
            if not isinstance(turn, TurnStart):
                continue
            configured_options = {}
            if turn.toolkits is None:
                continue
            for toolkit_name, toolkit_config in turn.toolkits.items():
                client_options = toolkit_config.client_options
                if client_options is None:
                    continue
                configured_options[toolkit_name] = client_options

        if configured_options is not None:
            self._active_turn_toolkit_client_options = configured_options
            return dict(configured_options)

        return dict(self._active_turn_toolkit_client_options)

    def _resolve_turn_tool_choice(
        self,
        *,
        turns: list[TurnStart | TurnSteer],
    ) -> ToolChoice | None:
        configured_tool_choice: ToolChoice | None = None
        saw_turn_start = False
        for turn in turns:
            if not isinstance(turn, TurnStart):
                continue
            saw_turn_start = True
            configured_tool_choice = turn.tool_choice

        if saw_turn_start:
            self._active_turn_tool_choice = configured_tool_choice
            return configured_tool_choice

        return self._active_turn_tool_choice

    @staticmethod
    def _merge_toolkit_capabilities(
        *,
        capabilities: list[ToolkitCapabilities],
    ) -> list[ToolkitCapabilities]:
        merged: dict[str, ToolkitCapabilities] = {}
        for capability in capabilities:
            existing = merged.get(capability.name)
            if existing is None:
                merged[capability.name] = capability
                continue

            existing.rules = [
                *existing.rules,
                *[rule for rule in capability.rules if rule not in existing.rules],
            ]
            existing.hidden = existing.hidden and capability.hidden
            existing.tools.extend(
                [
                    tool
                    for tool in capability.tools
                    if tool.name
                    not in {existing_tool.name for existing_tool in existing.tools}
                ]
            )
        return list(merged.values())

    async def _build_capabilities(
        self,
        *,
        sender: Participant | None,
    ) -> list[ToolkitCapabilities]:
        toolkits = await self._build_turn_toolkits(
            model=self.llm_adapter.default_model(),
            turns=[],
            sender=sender,
            toolkit_client_options={},
        )
        capabilities: list[ToolkitCapabilities] = []
        for toolkit in toolkits:
            if toolkit.hidden:
                continue
            capabilities.append(
                ToolkitCapabilities(
                    name=toolkit.name,
                    title=toolkit.title,
                    description=toolkit.description,
                    thumbnail_url=toolkit.thumbnail_url,
                    rules=[*toolkit.rules],
                    client_options=toolkit.client_options,
                    hidden=toolkit.hidden,
                    tools=[
                        ToolkitToolCapabilities(
                            name=tool.name,
                            title=tool.title,
                            description=tool.description,
                        )
                        for tool in toolkit.get_tools(client_options=None)
                    ],
                )
            )
        return self._merge_toolkit_capabilities(capabilities=capabilities)

    async def _build_turn_toolkits(
        self,
        *,
        model: str,
        turns: list[TurnStart | TurnSteer],
        sender: Participant | None = None,
        toolkit_client_options: dict[str, dict[str, Any]] | None = None,
    ) -> list[Toolkit]:
        with tracer.start_as_current_span("agent.turn.toolkits.build") as span:
            span.set_attribute("thread_id", self.thread_id)
            span.set_attribute("turn_count", len(turns))
            span.set_attribute("model", model)
            span.set_attribute(
                "custom_builder", self._turn_toolkits_builder is not None
            )
            if self._turn_toolkits_builder is not None:
                combined_toolkits = await self._turn_toolkits_builder(
                    sender, model, turns
                )
            else:
                combined_toolkits = [*self._toolkits]
                supervisor = self.supervisor
                if supervisor is not None:
                    for channel in supervisor.channels:
                        combined_toolkits.extend(channel.get_agent_toolkits())

            resolved_toolkits: list[Toolkit] = []
            for toolkit in combined_toolkits:
                resolved_toolkits.append(
                    toolkit.with_client_options(
                        client_options=(
                            None
                            if toolkit_client_options is None
                            else toolkit_client_options.get(toolkit.name)
                        )
                    )
                )

            span.set_attribute("toolkit_count", len(resolved_toolkits))
            return resolved_toolkits

    @staticmethod
    def _guess_url_mime_type(*, url: str) -> str | None:
        guessed_mime_type, _ = mimetypes.guess_type(urlparse(url).path)
        return guessed_mime_type

    def _resolve_content_scheme(self, *, url: str) -> ContentScheme | None:
        matched_scheme: ContentScheme | None = None
        matched_prefix_length = -1
        for scheme in self._content_schemes:
            if not url.startswith(scheme.prefix):
                continue
            prefix_length = len(scheme.prefix)
            if prefix_length <= matched_prefix_length:
                continue
            matched_scheme = scheme
            matched_prefix_length = prefix_length
        return matched_scheme

    def _append_downloaded_file_content(
        self,
        *,
        session: AgentSessionContext,
        file_content: FileContent,
        url: str,
        sender: Participant | None,
    ) -> None:
        mime_type = file_content.mime_type or "application/octet-stream"
        if mime_type.startswith("image/") and session.supports_images:
            session.append_image_message(
                mime_type=mime_type,
                data=file_content.data,
            )
            return

        if session.supports_files:
            session.append_file_message(
                filename=file_content.name,
                mime_type=mime_type,
                data=file_content.data,
            )
            return

        session.append_user_message(
            self._file_attachment_message(sender=sender, url=url)
        )

    def _append_remote_file_content(
        self,
        *,
        session: AgentSessionContext,
        url: str,
        sender: Participant | None,
    ) -> None:
        guessed_mime_type = self._guess_url_mime_type(url=url)
        if (
            guessed_mime_type is not None
            and guessed_mime_type.startswith("image/")
            and session.supports_images
        ):
            session.append_image_url(url=url)
            return

        if session.supports_files:
            session.append_file_url(url=url)
            return

        session.append_user_message(
            self._file_attachment_message(sender=sender, url=url)
        )

    async def _append_file_content(
        self,
        *,
        session: AgentSessionContext,
        url: str,
        sender: Participant | None,
    ) -> None:
        content_scheme = self._resolve_content_scheme(url=url)
        if content_scheme is not None:
            file_content = await content_scheme.download(url)
            self._append_downloaded_file_content(
                session=session,
                file_content=file_content,
                url=url,
                sender=sender,
            )
            return

        self._append_remote_file_content(
            session=session,
            url=url,
            sender=sender,
        )

    async def _append_turn_content(
        self,
        *,
        session: AgentSessionContext,
        sender: Participant | None,
        turns: list[TurnStart | TurnSteer],
    ) -> None:
        for turn in turns:
            for item in turn.content:
                if isinstance(item, AgentTextContent):
                    session.append_user_message(
                        self._format_live_turn_message(
                            sender=sender,
                            message=item.text,
                        )
                    )
                elif isinstance(item, AgentFileContent):
                    await self._append_file_content(
                        session=session,
                        url=item.url,
                        sender=sender,
                    )

    async def _append_queued_turn_messages(
        self,
        *,
        session: AgentSessionContext,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        for queued_message in queued_messages:
            await self._append_turn_content(
                session=session,
                sender=queued_message.sender,
                turns=[queued_message.request],
            )

    def _turn_error(self, *, message: str, code: str | None = None) -> AgentError:
        return AgentError(message=message, code=code)

    def _sender_for_turn_batch(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> Participant | None:
        if len(queued_messages) == 0:
            return None

        first_sender = queued_messages[0].sender
        for queued_message in queued_messages[1:]:
            if queued_message.sender != first_sender:
                return None

        return first_sender

    def _turn_ended_rejection(self) -> AgentError:
        return self._turn_error(
            message="turn ended before queued steer was processed",
            code="turn_ended",
        )

    async def _resolve_turn_instructions(
        self,
        *,
        queued_turn: _QueuedTurn,
    ) -> str | None:
        with tracer.start_as_current_span("agent.turn.rules.load") as span:
            span.set_attribute("thread_id", queued_turn.request.thread_id)
            span.set_attribute(
                "inline_instructions",
                queued_turn.request.instructions is not None,
            )
            span.set_attribute(
                "provider_configured",
                self._turn_instructions_provider is not None,
            )
            if queued_turn.request.instructions is not None:
                return queued_turn.request.instructions

            if self._turn_instructions_provider is None:
                return None

            return await self._turn_instructions_provider(queued_turn.sender)

    def _drain_queued_turn_messages(
        self,
        *,
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage] | None,
    ) -> list[_QueuedTurnMessage]:
        if active_turn_queue is None:
            return []

        drained_messages: list[_QueuedTurnMessage] = []
        while True:
            try:
                drained_messages.append(active_turn_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return drained_messages

    async def _wait_for_active_turn_queue_idle(
        self,
        *,
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage],
    ) -> bool:
        if not active_turn_queue.empty():
            return False

        active_turn_queue_updated = self._active_turn_queue_updated
        if active_turn_queue_updated is None:
            return True

        active_turn_queue_updated.clear()
        if not active_turn_queue.empty():
            return False

        try:
            await asyncio.wait_for(active_turn_queue_updated.wait(), timeout=0.25)
        except TimeoutError:
            pass

        return active_turn_queue.empty()

    def _drain_pending_turns(self) -> list[_QueuedTurn]:
        drained_turns: list[_QueuedTurn] = []
        if self._priority_turn is not None:
            drained_turns.append(self._priority_turn)
            self._priority_turn = None
        while True:
            try:
                drained_turns.append(self._pending_turns.get_nowait())
            except asyncio.QueueEmpty:
                break

        return drained_turns

    def _has_pending_turns(self) -> bool:
        return self._priority_turn is not None or not self._pending_turns.empty()

    async def _next_pending_turn(self) -> _QueuedTurn:
        if self._priority_turn is not None:
            queued_turn = self._priority_turn
            self._priority_turn = None
            return queued_turn

        return await self._pending_turns.get()

    def _emit_rejected_queued_turn_steers(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
        error: AgentError,
    ) -> None:
        for queued_message in queued_messages:
            turn = queued_message.request
            if not isinstance(turn, TurnSteer):
                continue

            self.emit(
                sender=queued_message.sender,
                payload=TurnSteerRejected(
                    type=AGENT_EVENT_TURN_STEER_REJECTED,
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    source_message_id=turn.message_id,
                    error=error,
                ),
            )

    def _emit_turn_steered_events(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            return

        for queued_message in queued_messages:
            turn = queued_message.request
            if not isinstance(turn, TurnSteer):
                continue

            self.emit(
                sender=queued_message.sender,
                payload=TurnSteered(
                    type=AGENT_EVENT_TURN_STEERED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    source_message_id=turn.message_id,
                ),
            )

    async def _apply_pending_turn_steers(
        self,
        *,
        session: AgentSessionContext,
    ) -> bool:
        turn_id = self._turn_id
        if turn_id is None or self._interrupt_requested_turn_id == turn_id:
            return False

        queued_messages = self._drain_queued_turn_messages(
            active_turn_queue=self._active_turn_queue,
        )
        steer_messages = [
            queued_message
            for queued_message in queued_messages
            if isinstance(queued_message.request, TurnSteer)
        ]
        if len(steer_messages) == 0:
            return False

        await self._remove_pending_status_messages(queued_messages=steer_messages)
        await self._append_queued_turn_messages(
            session=session,
            queued_messages=steer_messages,
        )
        self._emit_turn_steered_events(queued_messages=steer_messages)
        return True

    async def _prepare_turn_batch(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
        session: AgentSessionContext,
        model: str,
    ) -> tuple[Participant | None, list[Toolkit], ToolChoice | None]:
        turns = [queued_message.request for queued_message in queued_messages]
        sender = self._sender_for_turn_batch(queued_messages=queued_messages)
        toolkit_client_options = self._resolve_turn_toolkit_client_options(turns=turns)
        tool_choice = self._resolve_turn_tool_choice(turns=turns)

        await self._append_queued_turn_messages(
            session=session,
            queued_messages=queued_messages,
        )
        self._emit_turn_steered_events(queued_messages=queued_messages)
        combined_toolkits = await self._build_turn_toolkits(
            model=model,
            turns=turns,
            sender=sender,
            toolkit_client_options=toolkit_client_options,
        )
        return sender, combined_toolkits, tool_choice

    async def _run_adapter_next(
        self,
        *,
        session: AgentSessionContext,
        sender: Participant | None,
        combined_toolkits: list[Toolkit],
        tool_choice: ToolChoice | None,
        model: str,
    ) -> None:
        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            raise RuntimeError("turn publisher requested without an active turn")

        llm_provider = self.llm_adapter.provider_name()

        def enrich_llm_message(message: AgentMessage) -> AgentMessage:
            if not isinstance(message, AgentLLMMessage):
                return self._agent_message_with_participant_name(message)

            updates: dict[str, str] = {}
            if message.provider is None and llm_provider is not None:
                updates["provider"] = llm_provider
            if message.model is None:
                updates["model"] = model
            if len(updates) == 0:
                return self._agent_message_with_participant_name(message)
            return self._agent_message_with_participant_name(
                message.model_copy(update=updates)
            )

        def thread_status_from_agent_message(
            message: AgentMessage,
        ) -> tuple[str, str | None, int | None] | None:
            if isinstance(message, AgentThreadEvent):
                event = message.event
                event_type = event.get("type")
                if event_type not in ("agent.event", "codex.event"):
                    return None
                raw_state = event.get("state")
                if not isinstance(raw_state, str):
                    return None
                state = raw_state.strip().lower()
                if state not in (
                    _THREAD_STATUS_ACTIVE_STATES | _THREAD_STATUS_TERMINAL_STATES
                ):
                    return None

                headline = event.get("headline")
                summary = event.get("summary")
                details = event.get("details")
                status_text = None
                if event.get("name") == "computer.startup" and isinstance(
                    details, list
                ):
                    detail_texts = [
                        item.strip() for item in details if isinstance(item, str)
                    ]
                    detail_texts = [item for item in detail_texts if item != ""]
                    if len(detail_texts) > 0:
                        status_text = detail_texts[0]
                if status_text is None:
                    if isinstance(headline, str) and headline.strip() != "":
                        status_text = _normalize_status_text(headline.strip())
                    elif isinstance(summary, str) and summary.strip() != "":
                        status_text = _normalize_status_text(summary.strip())
                item_id = event.get("item_id")
                status_item_id = (
                    item_id.strip()
                    if isinstance(item_id, str) and item_id.strip() != ""
                    else ""
                )
                if state in _THREAD_STATUS_TERMINAL_STATES:
                    if status_item_id.strip() != "":
                        self._status_text_by_item_id.pop(status_item_id, None)
                        self._status_tool_argument_delta_bytes_by_item_id.pop(
                            status_item_id, None
                        )
                        self._status_tool_argument_delta_text_by_item_id.pop(
                            status_item_id, None
                        )
                    self._latest_status_text = None
                    return "Thinking", None, None
                if status_text is None:
                    return None
                self._latest_status_text = status_text
                if status_item_id.strip() != "":
                    self._status_text_by_item_id[status_item_id] = status_text
                return (
                    status_text,
                    status_item_id if status_item_id.strip() != "" else None,
                    _status_total_bytes(
                        self._status_tool_argument_delta_bytes_by_item_id.get(
                            status_item_id, 0
                        )
                    ),
                )

            if isinstance(
                message,
                (AgentToolCallPending, AgentToolCallInProgress, AgentToolCallStarted),
            ):
                state = (
                    "pending"
                    if isinstance(message, AgentToolCallPending)
                    else "in_progress"
                )
                argument_total_bytes = max(
                    self._status_tool_argument_delta_bytes_by_item_id.get(
                        message.item_id, 0
                    ),
                    _tool_argument_snapshot_bytes(message.arguments),
                    message.argument_bytes or 0,
                )
                self._status_tool_calls_by_item_id[message.item_id] = _StatusToolCall(
                    toolkit=message.toolkit,
                    tool=message.tool,
                    arguments=message.arguments,
                    state=state,
                    argument_delta_bytes=argument_total_bytes,
                )
                if message.argument_bytes is not None:
                    self._status_tool_argument_delta_bytes_by_item_id[
                        message.item_id
                    ] = argument_total_bytes
                return (
                    _tool_status_text(
                        state=state,
                        toolkit=message.toolkit,
                        tool=message.tool,
                        arguments=message.arguments,
                    ),
                    message.item_id,
                    _status_total_bytes(argument_total_bytes),
                )

            if isinstance(message, AgentToolCallArgumentsDelta):
                tool_call = self._status_tool_calls_by_item_id.get(message.item_id)
                previous_argument_bytes = (
                    self._status_tool_argument_delta_bytes_by_item_id.get(
                        message.item_id, 0
                    )
                )
                argument_delta_bytes = previous_argument_bytes + len(
                    message.delta.encode("utf-8")
                )
                self._status_tool_argument_delta_bytes_by_item_id[message.item_id] = (
                    argument_delta_bytes
                )
                delta_text = (
                    self._status_tool_argument_delta_text_by_item_id.get(
                        message.item_id, ""
                    )
                    + message.delta
                )
                self._status_tool_argument_delta_text_by_item_id[message.item_id] = (
                    delta_text
                )
                if tool_call is None:
                    status_text = self._status_text_by_item_id.get(message.item_id)
                    if status_text is None:
                        status_text = self._latest_status_text
                    if status_text is None:
                        return None
                    self._status_text_by_item_id[message.item_id] = status_text
                    return (
                        status_text,
                        message.item_id,
                        _status_total_bytes(argument_delta_bytes),
                    )
                updated_arguments = _tool_arguments_from_delta_text(
                    tool=tool_call.tool,
                    current=tool_call.arguments,
                    text=delta_text,
                )
                updated_tool_call = _StatusToolCall(
                    toolkit=tool_call.toolkit,
                    tool=tool_call.tool,
                    arguments=updated_arguments
                    if updated_arguments is not None
                    else tool_call.arguments,
                    state=tool_call.state,
                    argument_delta_bytes=argument_delta_bytes,
                )
                self._status_tool_calls_by_item_id[message.item_id] = updated_tool_call
                return (
                    _tool_status_text(
                        state=updated_tool_call.state,
                        toolkit=updated_tool_call.toolkit,
                        tool=updated_tool_call.tool,
                        arguments=updated_tool_call.arguments,
                    ),
                    message.item_id,
                    _status_total_bytes(updated_tool_call.argument_delta_bytes),
                )

            if isinstance(message, AgentToolCallApprovalRequested):
                return "Waiting for approval", message.item_id, None

            if isinstance(message, AgentToolCallEnded):
                tool_call = self._status_tool_calls_by_item_id.get(message.item_id)
                event_status_text = self._status_text_by_item_id.pop(
                    message.item_id, None
                )
                if (
                    event_status_text is not None
                    and self._latest_status_text == event_status_text
                ):
                    self._latest_status_text = None
                if tool_call is None:
                    self._status_tool_argument_delta_bytes_by_item_id.pop(
                        message.item_id, None
                    )
                    self._status_tool_argument_delta_text_by_item_id.pop(
                        message.item_id, None
                    )
                    return "Thinking", None, None
                self._status_tool_argument_delta_bytes_by_item_id.pop(
                    message.item_id, None
                )
                self._status_tool_argument_delta_text_by_item_id.pop(
                    message.item_id, None
                )
                state = (
                    "completed"
                    if message.error is None
                    else (message.error.code or "failed").strip().lower()
                )
                if state not in _THREAD_STATUS_TERMINAL_STATES:
                    state = "failed"
                status = _tool_status_text(
                    state=state,
                    toolkit=tool_call.toolkit,
                    tool=tool_call.tool,
                    arguments=tool_call.arguments,
                )
                if state == "completed":
                    return "Thinking", None, None
                return status, message.item_id, None

            if isinstance(
                message,
                (AgentImageGenerationStarted, AgentImageGenerationPartial),
            ):
                return "Generating image", message.item_id, None

            if isinstance(message, AgentImageGenerationCompleted):
                return "Thinking", None, None

            if isinstance(message, AgentImageGenerationFailed):
                return "Attempted to generate image", message.item_id, None

            if isinstance(message, TurnInterrupted):
                return "Turn interrupted", None, None

            return None

        def publish_event(message: AgentMessage) -> None:
            if self._interrupt_requested_turn_id == turn_id:
                return
            message = enrich_llm_message(message)
            publish_agent_message_status(message)
            self.emit(sender=sender, payload=message)

        thread_status_publisher = self.thread_status_publisher
        thread_status_publish_tail: asyncio.Task[None] | None = None

        def publish_agent_message_status(message: AgentMessage) -> None:
            nonlocal thread_status_publish_tail
            if thread_status_publisher is None:
                return

            status = thread_status_from_agent_message(message)
            if status is None:
                return
            status_text, pending_item_id, total_bytes = status
            previous_publish = thread_status_publish_tail

            async def publish_status_in_order() -> None:
                if previous_publish is not None:
                    with contextlib.suppress(Exception):
                        await previous_publish
                await thread_status_publisher.set_thread_status(
                    status=status_text,
                    pending_item_id=pending_item_id,
                    total_bytes=total_bytes,
                )

            thread_status_publish_tail = asyncio.create_task(publish_status_in_order())

        def publish_custom_event(event: dict[str, Any]) -> None:
            if self._interrupt_requested_turn_id == turn_id:
                return

            event_type = event.get("type")
            if event_type not in ("agent.event", "codex.event"):
                return
            publish_event(
                AgentThreadEvent(
                    type=AGENT_EVENT_THREAD_EVENT,
                    thread_id=thread_id,
                    event=event,
                )
            )

        handle_event = self.llm_adapter.make_agent_event_publisher(
            turn_id=turn_id,
            thread_id=thread_id,
            callback=publish_event,
            custom_event_callback=publish_custom_event,
        )

        self._active_turn_sender = sender
        had_thread_id = "thread_id" in session.metadata
        previous_thread_id = session.metadata.get("thread_id")
        had_turn_id = "turn_id" in session.metadata
        previous_turn_id = session.metadata.get("turn_id")
        session.metadata["thread_id"] = thread_id
        session.metadata["turn_id"] = turn_id
        try:
            completed = False
            with tracer.start_as_current_span("agent.turn.llm") as span:
                span.set_attribute("thread_id", thread_id)
                span.set_attribute("turn_id", turn_id)
                span.set_attribute("model", model)
                span.set_attribute("toolkit_count", len(combined_toolkits))
                span.set_attribute("tool_choice.configured", tool_choice is not None)
                await self._compact_context_if_needed(
                    session=session,
                    model=model,
                    publish_event=publish_event,
                )
                next_task = asyncio.create_task(
                    self.llm_adapter.next(
                        context=session,
                        toolkits=combined_toolkits,
                        caller=self._participant,
                        event_handler=handle_event,
                        steering_callback=lambda: self._apply_pending_turn_steers(
                            session=session
                        ),
                        model=model,
                        on_behalf_of=sender,
                        tool_choice=tool_choice,
                    )
                )
                self._active_next_task = next_task
                await next_task
                completed = self._interrupt_requested_turn_id != turn_id
        finally:
            await self._emit_usage_update(
                session=session,
                model=model,
                toolkits=combined_toolkits,
                turn_id=turn_id,
                sender=sender,
                include_usage=completed,
                restore_context_from_storage=not completed,
            )
            if had_thread_id:
                session.metadata["thread_id"] = previous_thread_id
            else:
                session.metadata.pop("thread_id", None)

            if had_turn_id:
                session.metadata["turn_id"] = previous_turn_id
            else:
                session.metadata.pop("turn_id", None)
            self._active_next_task = None
            self._active_turn_sender = None

    async def _continue_interrupted_turn(
        self,
        *,
        session: AgentSessionContext,
        model: str,
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage],
    ) -> bool:
        queued_messages = self._drain_queued_turn_messages(
            active_turn_queue=active_turn_queue,
        )
        steer_messages = [
            queued_message
            for queued_message in queued_messages
            if isinstance(queued_message.request, TurnSteer)
        ]
        if len(steer_messages) == 0:
            return False

        await self._remove_pending_status_messages(queued_messages=steer_messages)
        self.llm_adapter.on_turn_steer(context=session, interrupted=True)
        sender, combined_toolkits, tool_choice = await self._prepare_turn_batch(
            queued_messages=steer_messages,
            session=session,
            model=model,
        )
        await self._run_adapter_next(
            session=session,
            sender=sender,
            combined_toolkits=combined_toolkits,
            tool_choice=tool_choice,
            model=model,
        )
        return True

    async def _execute_turn_batch(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
        session: AgentSessionContext,
        model: str,
    ) -> None:
        sender, combined_toolkits, tool_choice = await self._prepare_turn_batch(
            queued_messages=queued_messages,
            session=session,
            model=model,
        )
        await self._run_adapter_next(
            session=session,
            sender=sender,
            combined_toolkits=combined_toolkits,
            tool_choice=tool_choice,
            model=model,
        )

    async def _handle_interrupt(
        self,
        *,
        queued_turn: _QueuedTurn,
        turn_id: str,
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage],
    ) -> Literal["continue", "cancel"] | None:
        if self._interrupt_requested_turn_id != turn_id:
            return None

        interrupt_source_message_id = self._interrupt_source_message_id
        self._interrupt_requested_turn_id = None
        self._interrupt_source_message_id = None
        self._cancel_pending_tool_call_approvals()

        if interrupt_source_message_id is not None:
            self.emit(
                sender=queued_turn.sender,
                payload=TurnInterrupted(
                    type=AGENT_EVENT_TURN_INTERRUPTED,
                    thread_id=queued_turn.request.thread_id,
                    turn_id=turn_id,
                    source_message_id=interrupt_source_message_id,
                ),
            )

        if active_turn_queue.empty():
            return "cancel"
        return "continue"

    async def _run_next_turn(self) -> None:
        queued_turn = await self._next_pending_turn()
        queued_turn_messages = [
            _QueuedTurnMessage(
                sender=queued_turn.sender,
                request=queued_turn.request,
            ),
            *queued_turn.queued_messages,
        ]
        turn_id = queued_turn.request.turn_id or str(uuid.uuid4())
        turn_span_context = tracer.start_as_current_span("agent.turn")
        turn_span = turn_span_context.__enter__()
        turn_span.set_attribute("thread_id", queued_turn.request.thread_id)
        turn_span.set_attribute("turn_id", turn_id)
        turn_span.set_attribute("source_message_id", queued_turn.request.message_id)
        turn_span.set_attribute("queued_message_count", len(queued_turn_messages))
        try:
            await self._remove_pending_status_messages(
                queued_messages=queued_turn_messages
            )
        except BaseException as exc:
            turn_span_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        self._turn_id = turn_id
        self._status_tool_calls_by_item_id.clear()
        self._status_tool_argument_delta_bytes_by_item_id.clear()
        self._status_tool_argument_delta_text_by_item_id.clear()
        self._status_text_by_item_id.clear()
        self._latest_status_text = None
        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is not None:
            await thread_status_publisher.set_thread_turn_id(turn_id=turn_id)
            await thread_status_publisher.set_thread_status(status="Thinking")
        self._interrupt_requested_turn_id = None
        self._interrupt_source_message_id = None
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage] = asyncio.Queue()
        self._active_turn_queue = active_turn_queue
        self._active_turn_queue_updated = asyncio.Event()
        for queued_message in queued_turn_messages:
            active_turn_queue.put_nowait(queued_message)
        self.emit(
            sender=queued_turn.sender,
            payload=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id=queued_turn.request.thread_id,
                turn_id=turn_id,
                source_message_id=queued_turn.request.message_id,
            ),
        )

        error: AgentError | None = None
        session: AgentSessionContext | None = None
        original_instructions: str | None = None
        try:
            session = await self.ensure_session_context(turn_id=turn_id)
            model = (
                queued_turn.request.model
                if queued_turn.request.model is not None
                else self.llm_adapter.default_model()
            )
            turn_span.set_attribute("model", model)
            original_instructions = session.instructions
            turn_instructions = await self._resolve_turn_instructions(
                queued_turn=queued_turn
            )
            if turn_instructions is not None:
                session.instructions = turn_instructions

            continue_interrupted_turn = False
            while True:
                if continue_interrupted_turn:
                    continue_interrupted_turn = False
                    continued = await self._continue_interrupted_turn(
                        session=session,
                        model=model,
                        active_turn_queue=active_turn_queue,
                    )
                    if not continued:
                        error = self._turn_error(
                            message="turn cancelled",
                            code="cancelled",
                        )
                        break
                else:
                    queued_messages = [await active_turn_queue.get()]
                    while True:
                        try:
                            queued_messages.append(active_turn_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break

                    await self._remove_pending_status_messages(
                        queued_messages=queued_messages
                    )

                    await self._execute_turn_batch(
                        queued_messages=queued_messages,
                        session=session,
                        model=model,
                    )
                interrupt_action = await self._handle_interrupt(
                    queued_turn=queued_turn,
                    turn_id=turn_id,
                    active_turn_queue=active_turn_queue,
                )
                if interrupt_action == "continue":
                    continue_interrupted_turn = True
                    continue
                if interrupt_action == "cancel":
                    error = self._turn_error(message="turn cancelled", code="cancelled")
                    break
                if await self._wait_for_active_turn_queue_idle(
                    active_turn_queue=active_turn_queue
                ):
                    break
        except asyncio.CancelledError:
            interrupt_action = await self._handle_interrupt(
                queued_turn=queued_turn,
                turn_id=turn_id,
                active_turn_queue=active_turn_queue,
            )
            if interrupt_action == "continue":
                continue_interrupted_turn = True
                while True:
                    try:
                        if continue_interrupted_turn:
                            continue_interrupted_turn = False
                            continued = await self._continue_interrupted_turn(
                                session=session,
                                model=model,
                                active_turn_queue=active_turn_queue,
                            )
                            if not continued:
                                error = self._turn_error(
                                    message="turn cancelled",
                                    code="cancelled",
                                )
                                break
                        else:
                            queued_messages = [await active_turn_queue.get()]
                            while True:
                                try:
                                    queued_messages.append(
                                        active_turn_queue.get_nowait()
                                    )
                                except asyncio.QueueEmpty:
                                    break

                            await self._remove_pending_status_messages(
                                queued_messages=queued_messages
                            )
                            await self._execute_turn_batch(
                                queued_messages=queued_messages,
                                session=session,
                                model=model,
                            )
                    except asyncio.CancelledError:
                        interrupt_action = await self._handle_interrupt(
                            queued_turn=queued_turn,
                            turn_id=turn_id,
                            active_turn_queue=active_turn_queue,
                        )
                        if interrupt_action == "continue":
                            continue_interrupted_turn = True
                            continue
                        error = self._turn_error(
                            message="turn cancelled",
                            code="cancelled",
                        )
                        break

                    interrupt_action = await self._handle_interrupt(
                        queued_turn=queued_turn,
                        turn_id=turn_id,
                        active_turn_queue=active_turn_queue,
                    )
                    if interrupt_action == "continue":
                        continue_interrupted_turn = True
                        continue
                    if interrupt_action == "cancel":
                        error = self._turn_error(
                            message="turn cancelled",
                            code="cancelled",
                        )
                        break
                    if await self._wait_for_active_turn_queue_idle(
                        active_turn_queue=active_turn_queue
                    ):
                        break
            else:
                error = self._turn_error(message="turn cancelled", code="cancelled")
        except Exception as exc:
            logger.exception("turn failed")
            error_message = str(exc) if str(exc) != "" else exc.__class__.__name__
            error = self._turn_error(
                message=error_message,
                code=exc.__class__.__name__,
            )
        finally:
            try:
                self._interrupt_requested_turn_id = None
                self._interrupt_source_message_id = None
                self._active_turn_queue = None
                self._active_turn_queue_updated = None
                self._active_turn_toolkit_client_options = {}
                self._active_turn_tool_choice = None
                self._status_tool_calls_by_item_id.clear()
                self._status_tool_argument_delta_bytes_by_item_id.clear()
                self._status_tool_argument_delta_text_by_item_id.clear()
                self._status_text_by_item_id.clear()
                self._latest_status_text = None
                remaining_queued_messages = self._drain_queued_turn_messages(
                    active_turn_queue=active_turn_queue,
                )
                if len(remaining_queued_messages) > 0:
                    await self._remove_pending_status_messages(
                        queued_messages=remaining_queued_messages
                    )
                    self._emit_rejected_queued_turn_steers(
                        queued_messages=remaining_queued_messages,
                        error=self._turn_ended_rejection(),
                    )
                self._cancel_pending_tool_call_approvals()
                if session is not None:
                    session.instructions = original_instructions
                self.emit(
                    sender=queued_turn.sender,
                    payload=TurnEnded(
                        type=AGENT_EVENT_TURN_ENDED,
                        thread_id=queued_turn.request.thread_id,
                        turn_id=turn_id,
                        error=error,
                    ),
                )
                thread_status_publisher = self.thread_status_publisher
                if thread_status_publisher is not None:
                    await thread_status_publisher.set_thread_turn_id(turn_id=None)
                    await thread_status_publisher.clear_thread_status()
            finally:
                turn_span.set_attribute("error", error is not None)
                if error is not None:
                    turn_span.set_attribute("error.code", error.code or "unknown")
                turn_span_context.__exit__(None, None, None)

    def _schedule_next_turn(self) -> None:
        if (
            self._turn_task is not None
            or self._stop.is_set()
            or not self._has_pending_turns()
        ):
            return

        self._turn_task = asyncio.create_task(self._run_next_turn())
        self._turn_task.add_done_callback(self._on_turn_done)

    def _on_turn_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            logger.debug("turn task cancelled")
        else:
            exc = task.exception()
            if exc is not None:
                logger.exception("turn failed", exc_info=exc)

        self._turn_id = None
        self._turn_task = None
        self._active_turn_queue = None
        self._active_turn_queue_updated = None

        if not self._stop.is_set():
            self._schedule_next_turn()

    async def on_thread_open(self, message: Message) -> None:
        _coerce_message_data(message.data, OpenThread)
        if self._emit_cached_usage_update(
            turn_id=self._turn_id,
            sender=message.sender,
            source=message.source,
        ):
            return
        session = await self.ensure_session_context(turn_id=None)
        try:
            usage_update = await self._build_usage_update(
                session=session,
                model=self.llm_adapter.default_model(),
                toolkits=self._toolkits,
                turn_id=self._turn_id,
                restore_context_from_storage=True,
            )
        except Exception:
            logger.debug("failed to publish agent usage update", exc_info=True)
            return
        self._last_usage_update = usage_update
        self._send_thread_open_usage_update(
            usage_update=usage_update,
            sender=message.sender,
            source=message.source,
        )

    async def on_thread_close(self, message: Message) -> None:
        _coerce_message_data(message.data, CloseThread)

    async def on_turn_start(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnStart)
        turn_id = turn.turn_id or str(uuid.uuid4())
        turn = turn.model_copy(update={"turn_id": turn_id})
        queued_message = _QueuedTurnMessage(
            sender=message.sender,
            request=turn,
        )
        self._record_accepted_turns(queued_messages=[queued_message])
        should_track_pending_status = (
            self._turn_task is not None or self._has_pending_turns()
        )
        await self._pending_turns.put(
            _QueuedTurn(
                sender=message.sender,
                request=turn,
            )
        )
        self.emit(
            sender=message.sender,
            payload=TurnStartAccepted(
                type=AGENT_EVENT_TURN_START_ACCEPTED,
                thread_id=turn.thread_id,
                turn_id=turn_id,
                source_message_id=turn.message_id,
            ),
        )
        if should_track_pending_status:
            await self._add_pending_status_messages(queued_messages=[queued_message])
        self._schedule_next_turn()

    async def on_capabilities_request(self, message: Message) -> None:
        request = _coerce_message_data(message.data, CapabilitiesRequest)
        capabilities = await self._build_capabilities(sender=message.sender)
        self.emit(
            sender=message.sender,
            payload=CapabilitiesResponse(
                type=AGENT_MESSAGE_CAPABILITIES_RESPONSE,
                thread_id=request.thread_id,
                source_message_id=request.message_id,
                version=agents_version,
                toolkits=capabilities,
            ),
        )

    async def on_turn_steer(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnSteer)

        active_turn_queue = self._active_turn_queue
        if (
            self._turn_id is None
            or self._turn_id != turn.turn_id
            or active_turn_queue is None
        ):
            rejection = self._turn_error(
                message="turn is not in progress",
                code="turn_not_in_progress",
            )
            self.emit(
                sender=message.sender,
                payload=TurnSteerRejected(
                    type=AGENT_EVENT_TURN_STEER_REJECTED,
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    source_message_id=turn.message_id,
                    error=rejection,
                ),
            )
            return

        queued_message = _QueuedTurnMessage(
            sender=message.sender,
            request=turn,
        )
        self._record_accepted_turns(queued_messages=[queued_message])
        active_turn_queue.put_nowait(queued_message)
        active_turn_queue_updated = self._active_turn_queue_updated
        if active_turn_queue_updated is not None:
            active_turn_queue_updated.set()
        self.emit(
            sender=message.sender,
            payload=TurnSteerAccepted(
                type=AGENT_EVENT_TURN_STEER_ACCEPTED,
                thread_id=turn.thread_id,
                turn_id=turn.turn_id,
                source_message_id=turn.message_id,
            ),
        )
        await self._add_pending_status_messages(queued_messages=[queued_message])

    async def on_turn_interrupt(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnInterrupt)
        if self._turn_id != turn.turn_id or self._turn_task is None:
            return

        self._interrupt_requested_turn_id = turn.turn_id
        self._interrupt_source_message_id = turn.message_id
        self.emit(
            sender=message.sender,
            payload=TurnInterruptAccepted(
                type=AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
                thread_id=turn.thread_id,
                turn_id=turn.turn_id,
                source_message_id=turn.message_id,
            ),
        )
        active_next_task = self._active_next_task
        if active_next_task is not None:
            active_next_task.cancel()

    async def on_tool_call_approve(self, message: Message) -> None:
        approval = _coerce_message_data(message.data, ApproveAgentToolCall)
        if self._turn_id != approval.turn_id:
            return

        await self._resolve_tool_call_approval(
            item_id=approval.item_id,
            approved=True,
        )

    async def on_tool_call_reject(self, message: Message) -> None:
        rejection = _coerce_message_data(message.data, RejectAgentToolCall)
        if self._turn_id != rejection.turn_id:
            return

        await self._resolve_tool_call_approval(
            item_id=rejection.item_id,
            approved=False,
        )

    async def on_stop(self) -> None:
        if self._turn_task is not None:
            active_next_task = self._active_next_task
            if active_next_task is not None:
                active_next_task.cancel()
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._active_turn_queue = None
        self._active_turn_queue_updated = None
        self._active_turn_sender = None
        self._interrupt_requested_turn_id = None
        self._interrupt_source_message_id = None
        self._cancel_pending_tool_call_approvals()
        await self._clear_pending_status_messages()
        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is not None:
            await thread_status_publisher.set_thread_turn_id(turn_id=None)
            await thread_status_publisher.clear_thread_status()

        if self._session_context is not None:
            await self._session_context.close()
            self._session_context = None

    async def on_message(self, message: Message) -> None:
        handler = self._handlers.get(message.data.type)
        if handler is None:
            return

        try:
            await handler(message)
        except Exception:
            logger.exception("llm agent process failed to handle message")
