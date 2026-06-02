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
)
from urllib.parse import urlparse

from meshagent.api import Participant
from meshagent.api.messaging import Content, ErrorContent, FileContent, JsonContent
from meshagent.agents.adapter import (
    LLMAdapter,
    LLMAudioFormat,
    LLMModelInfo,
    LLMProvider,
    ToolCallApprovalRequest,
)
from meshagent.agents.context import AgentSessionContext, SessionUsage
from meshagent.tools import FunctionTool, ToolContext, Toolkit, tool
from opentelemetry import trace
from pydantic_core import from_json as pydantic_core_from_json
from .thread_adapter import default_format_message
from .thread_storage import (
    NoopThreadStorageRepository,
    ThreadListEntry,
    ThreadListPage,
    ThreadStorage,
    ThreadStorageRepository,
)
from .thread_naming import (
    DEFAULT_THREAD_NAME,
    determine_thread_name,
    fallback_thread_name,
)
from .thread_status_publisher import ThreadStatusPublisher
from .tool_call_accumulator import ToolCallAccumulator
from .version import __version__ as agents_version
from .messages import (
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AGENT_MESSAGE_MODEL_CHANGE,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_MODELS_RESPONSE,
    AGENT_MESSAGE_PARTICIPANT_CONNECT,
    AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
    AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_DELETE,
    AGENT_MESSAGE_THREAD_LIST,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_RENAME,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_THREAD_UNWATCH,
    AGENT_MESSAGE_THREAD_WATCH,
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_CLIENT_TOOL_CALL_CANCELLED,
    AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_THREAD_CREATED,
    AGENT_EVENT_THREAD_DELETED,
    AGENT_EVENT_THREAD_LISTED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_UPDATED,
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
    AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
    AGENT_MESSAGE_SECRET_RESPONSE,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AGENT_EVENT_SECRET_REQUESTED,
    AgentError,
    AgentFileContent,
    AgentContextCompacted,
    AgentLLMMessage,
    AgentMessage,
    AgentAudioFormat,
    AgentTextContent,
    AgentToolCallArgumentsDelta,
    AgentToolCallApprovalRequested,
    AgentClientToolCallCancelled,
    AgentClientToolCallRequested,
    AgentClientToolCallResponse,
    ClientToolkitDescription,
    AgentSecretRequested,
    AgentSecretResponse,
    AgentThreadListEntry,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallPending,
    AgentToolCallStarted,
    AgentImageGenerationCompleted,
    AgentImageGenerationFailed,
    AgentImageGenerationPartial,
    AgentImageGenerationStarted,
    AgentModelChanged,
    AgentModelInfo,
    AgentAudioTranscriptionCompleted,
    AgentAudioTranscriptionDelta,
    AgentAudioTranscriptionFailed,
    AgentAudioTranscriptionStarted,
    AgentThreadEvent,
    AgentThreadMessage,
    AgentContextWindowUsage,
    AgentUsageUpdated,
    AgentProviderInfo,
    AgentRealtimeAudioChunk,
    AgentRealtimeAudioCommit,
    ApproveAgentToolCall,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    CapabilitiesRequest,
    CapabilitiesResponse,
    ChangeModel,
    ClearThread,
    CloseThread,
    DeleteThread,
    ListThreads,
    ModelsRequest,
    ModelsResponse,
    OpenThread,
    ParticipantConnect,
    ParticipantDisconnect,
    RejectAgentToolCall,
    RenameThread,
    StartThread,
    ThreadLoaded,
    ThreadCreated,
    ThreadDeleted,
    ThreadsListed,
    ThreadStarted,
    ThreadUpdated,
    UnwatchThreads,
    WatchThreads,
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
        AGENT_MESSAGE_MODEL_CHANGE,
        AGENT_EVENT_MODEL_CHANGED,
        AGENT_MESSAGE_THREAD_CLOSE,
        AGENT_MESSAGE_THREAD_DELETE,
        AGENT_MESSAGE_THREAD_LIST,
        AGENT_MESSAGE_THREAD_OPEN,
        AGENT_MESSAGE_THREAD_RENAME,
        AGENT_MESSAGE_TOOL_CALL_APPROVE,
        AGENT_MESSAGE_TOOL_CALL_REJECT,
        AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
        AGENT_MESSAGE_SECRET_RESPONSE,
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


def _extract_apply_patch_path(*, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    return _first_nested_text(value=arguments, keys=("path",))


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


def _patch_line_counts(*, patch: str) -> tuple[int | None, int | None]:
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if added == 0 and removed == 0:
        return None, None
    return added, removed


def _tool_patch_line_counts(
    *,
    tool: str,
    arguments: dict[str, Any] | None,
) -> tuple[int | None, int | None]:
    if tool.strip().lower() != "apply_patch":
        return None, None
    return _patch_line_counts(patch=_extract_apply_patch_text(arguments=arguments))


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
    if normalized_tool == "apply_patch":
        patch = text.strip()
        if patch != "":
            return _merge_tool_arguments(current=current, update={"patch": patch})

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
        path = _extract_apply_patch_path(arguments=arguments) or _apply_patch_path(
            patch=patch
        )
        if path != "":
            if state == "pending":
                return f"Editing {path}"
            if state == "failed":
                return f"Attempted to patch {path}"
            if state == "cancelled":
                return f"Patch cancelled: {path}"
            if state in _THREAD_STATUS_ACTIVE_STATES:
                return f"Editing {path}"
            return f"Edited {path}"
        if state == "pending":
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
    to_participant_id: str | None = None


@dataclass(frozen=True, slots=True)
class CreatedAgentThread:
    thread_id: str
    name: str | None = None
    metadata: Callable[[], Awaitable[ThreadListEntry | None]] | None = None


class AgentBackend(Protocol):
    @property
    def name(self) -> str: ...

    async def on_start(self) -> None: ...

    async def on_stop(self) -> None: ...

    def model_providers(
        self,
        *,
        current_backend: str | None,
        current_provider: str | None,
        current_model: str | None,
    ) -> list[AgentProviderInfo]: ...

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None: ...

    async def create_realtime_connection(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> Any: ...

    async def create_thread(
        self,
        *,
        supervisor: AgentSupervisor,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> CreatedAgentThread: ...

    def create_thread_process(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
    ) -> AgentProcess: ...


_MessageT = TypeVar("_MessageT", bound=AgentMessage)


def _coerce_message_data(data: AgentMessage, model: type[_MessageT]) -> _MessageT:
    if isinstance(data, model):
        return data

    return model.model_validate(data.model_dump(mode="python"))


def _message_turn_id(message: AgentThreadMessage) -> str | None:
    data = message.model_dump(mode="python")
    value = data.get("turn_id")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized != "" else None


def _agent_audio_format_from_llm_audio_format(
    audio_format: LLMAudioFormat | None,
) -> AgentAudioFormat | None:
    if audio_format is None:
        return None
    return AgentAudioFormat(
        type=audio_format.type,
        sample_rate=audio_format.sample_rate,
        bitrate=audio_format.bitrate,
    )


def _audio_format_mismatch(
    *,
    declared: AgentAudioFormat,
    expected: AgentAudioFormat | None,
) -> str | None:
    if expected is None:
        return None
    if expected.type.strip().lower() != declared.type.strip().lower():
        return f"expected audio type {expected.type!r}, got {declared.type!r}"
    if (
        expected.sample_rate is not None
        and declared.sample_rate != expected.sample_rate
    ):
        return (
            f"expected audio sample rate {expected.sample_rate}, "
            f"got {declared.sample_rate}"
        )
    if expected.bitrate is not None and declared.bitrate != expected.bitrate:
        return f"expected audio bitrate {expected.bitrate}, got {declared.bitrate}"
    return None


def agent_model_info(
    *,
    provider: LLMProvider,
    model_info: LLMModelInfo,
    current_provider: str,
    current_model: str,
) -> AgentModelInfo:
    return AgentModelInfo(
        name=model_info.name,
        friendly_name=model_info.friendly_name,
        description=model_info.description,
        context_window=model_info.context_window,
        pricing=model_info.pricing,
        modalities=list(model_info.modalities),
        available_voices=list(model_info.available_voices),
        default_output_voice=model_info.default_output_voice,
        input_format=_agent_audio_format_from_llm_audio_format(model_info.input_format),
        output_format=_agent_audio_format_from_llm_audio_format(
            model_info.output_format
        ),
        turn_detection=model_info.turn_detection,
        realtime_protocols=list(model_info.realtime_protocols),
        supports_attachments=model_info.supports_attachments,
        accepts=list(model_info.accepts),
        active=provider.name == current_provider and model_info.name == current_model,
    )


def agent_provider_info(
    *,
    provider: LLMProvider,
    current_provider: str,
    current_model: str,
    backend: str | None = None,
) -> AgentProviderInfo:
    return AgentProviderInfo(
        name=provider.name,
        friendly_name=provider.adapter.provider_friendly_name(),
        description=provider.adapter.provider_description(),
        backend=backend,
        default_model=provider.adapter.default_model(),
        models=[
            agent_model_info(
                provider=provider,
                model_info=model_info,
                current_provider=current_provider,
                current_model=current_model,
            )
            for model_info in provider.adapter.list_models()
        ],
    )


def _start_thread_name_input(start_thread: StartThread) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    attachments: list[str] = []
    for item in start_thread.content or []:
        if isinstance(item, AgentTextContent) and item.text.strip() != "":
            text_parts.append(item.text.strip())
        elif isinstance(item, AgentFileContent) and item.url.strip() != "":
            attachments.append(item.url.strip())
    return " ".join(text_parts).strip(), attachments


def _fallback_start_thread_name(start_thread: StartThread) -> str:
    if isinstance(start_thread.name, str) and start_thread.name.strip() != "":
        return start_thread.name.strip()

    message_text, attachments = _start_thread_name_input(start_thread)
    return fallback_thread_name(message_text=message_text, attachments=attachments)


class LLMBackend:
    def __init__(
        self,
        *,
        name: str = "llm",
        providers: list[LLMProvider],
        default_provider: LLMProvider | None = None,
        process_factory: Callable[[str, str], AgentProcess],
        thread_id_factory: Callable[[StartThread, Participant | None], Awaitable[str]],
        realtime_connection_factory: Callable[
            [str, StartThread, Participant | None],
            Awaitable[Any],
        ]
        | None = None,
        thread_name_adapter: LLMAdapter | None = None,
        thread_name_caller: Callable[[], Participant] | None = None,
        thread_name_model: str | None = None,
    ) -> None:
        self._name = name
        self._providers = LLMAgentProcess._normalize_llm_providers(
            llm_adapter=None,
            llm_providers=providers,
            default_provider=default_provider,
        )
        self._providers_by_name = {
            provider.name: provider for provider in self._providers
        }
        self._default_provider = (
            self._providers_by_name[default_provider.name]
            if default_provider is not None
            else self._providers[0]
        )
        self._process_factory = process_factory
        self._thread_id_factory = thread_id_factory
        self._realtime_connection_factory = realtime_connection_factory
        self._thread_name_adapter = thread_name_adapter
        self._thread_name_caller = thread_name_caller
        self._thread_name_model = thread_name_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_provider(self) -> LLMProvider:
        return self._default_provider

    def model_providers(
        self,
        *,
        current_backend: str | None,
        current_provider: str | None,
        current_model: str | None,
    ) -> list[AgentProviderInfo]:
        del current_backend
        provider = self._default_provider
        provider_name = current_provider or provider.name
        model_name = current_model or provider.adapter.default_model()
        return [
            agent_provider_info(
                provider=candidate,
                current_provider=provider_name,
                current_model=model_name,
                backend=self._name,
            )
            for candidate in self._providers
        ]

    def _resolve_provider(self, provider_name: str | None) -> LLMProvider:
        if provider_name is None or provider_name.strip() == "":
            return self._default_provider
        provider = self._providers_by_name.get(provider_name)
        if provider is not None:
            return provider
        names = ", ".join(sorted(self._providers_by_name))
        raise ValueError(
            f"unknown provider {provider_name!r}; available providers: {names}"
        )

    @staticmethod
    def _adapter_uses_default_model_list(adapter: LLMAdapter) -> bool:
        if not isinstance(adapter, LLMAdapter):
            return True
        return type(adapter).list_models is LLMAdapter.list_models

    def _resolve_model(self, *, provider: LLMProvider, model: str | None) -> str:
        if model is None or model.strip() == "":
            return provider.adapter.default_model()
        if self._adapter_uses_default_model_list(provider.adapter):
            return model
        models = provider.adapter.list_models()
        for model_info in models:
            if model_info.name == model:
                return model
        names = ", ".join(model_info.name for model_info in models)
        raise ValueError(
            f"unknown model {model!r} for provider {provider.name!r}; "
            f"available models: {names}"
        )

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        if (
            turn_start.backend is not None
            and turn_start.backend.strip() != ""
            and turn_start.backend != self._name
        ):
            return AgentError(
                message=f"unknown backend {turn_start.backend!r}",
                code="unknown_backend",
            )
        try:
            provider = self._resolve_provider(turn_start.provider)
            resolved_model = self._resolve_model(
                provider=provider, model=turn_start.model
            )
        except ValueError as exc:
            code = (
                "unknown_provider"
                if "unknown provider" in str(exc)
                else "unknown_model"
            )
            return AgentError(message=str(exc), code=code)
        model_info = next(
            (
                candidate
                for candidate in provider.adapter.list_models()
                if candidate.name == resolved_model
            ),
            None,
        )
        if model_info is not None:
            unsupported_output_modalities = [
                output
                for output in (turn_start.output_modalities or [])
                if output not in model_info.modalities
            ]
            if len(unsupported_output_modalities) > 0:
                unsupported = ", ".join(
                    repr(item) for item in unsupported_output_modalities
                )
                return AgentError(
                    message=(
                        f"model {model_info.name!r} does not support "
                        f"{unsupported} output modalities"
                    ),
                    code="unsupported_modality",
                )
        return None

    async def create_realtime_connection(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> Any:
        del supervisor
        if self._realtime_connection_factory is None:
            return None
        return await self._realtime_connection_factory(thread_id, start_thread, sender)

    async def create_thread(
        self,
        *,
        supervisor: AgentSupervisor,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> CreatedAgentThread:
        message_text, attachments = _start_thread_name_input(start_thread)
        fallback_name = _fallback_start_thread_name(start_thread)
        thread_id_task = asyncio.create_task(
            self._thread_id_factory(start_thread, sender)
        )

        async def create_metadata() -> ThreadListEntry | None:
            name = fallback_name
            if not (
                isinstance(start_thread.name, str) and start_thread.name.strip() != ""
            ):
                caller = (
                    self._thread_name_caller()
                    if self._thread_name_caller is not None
                    else sender
                )
                if caller is not None:
                    with tracer.start_as_current_span("agent.thread.name.generate"):
                        name = await determine_thread_name(
                            adapter=self._thread_name_adapter,
                            caller=caller,
                            message_text=message_text,
                            attachments=attachments,
                            on_behalf_of=sender,
                            model=self._thread_name_model,
                        )

            thread_id = await thread_id_task
            with tracer.start_as_current_span("agent.thread.storage.upsert"):
                return await supervisor.on_thread_started(
                    created_thread=CreatedAgentThread(thread_id=thread_id, name=name),
                    start_thread=start_thread,
                    sender=sender,
                )

        thread_id = await thread_id_task
        return CreatedAgentThread(
            thread_id=thread_id,
            name=start_thread.name.strip()
            if isinstance(start_thread.name, str) and start_thread.name.strip() != ""
            else None,
            metadata=create_metadata,
        )

    def create_thread_process(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
    ) -> AgentProcess:
        del supervisor
        return self._process_factory(thread_id, self._name)

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None


class ChatBackend:
    def __init__(
        self,
        *,
        name: str = "chat",
        thread_id_factory: Callable[[StartThread, Participant | None], Awaitable[str]],
        process_factory: Callable[[str, str], AgentProcess] | None = None,
    ) -> None:
        self._name = name
        self._process_factory = process_factory
        self._thread_id_factory = thread_id_factory

    @property
    def name(self) -> str:
        return self._name

    def model_providers(
        self,
        *,
        current_backend: str | None,
        current_provider: str | None,
        current_model: str | None,
    ) -> list[AgentProviderInfo]:
        provider_name = current_provider or "chat"
        model_name = current_model or "none"
        return [
            AgentProviderInfo(
                name="chat",
                friendly_name="Chat",
                description="Chat without an agent response.",
                backend=self._name,
                default_model="none",
                models=[
                    AgentModelInfo(
                        name="none",
                        friendly_name="None",
                        description="Accept turns and end them without generating a response.",
                        active=(
                            current_backend == self._name
                            and provider_name == "chat"
                            and model_name == "none"
                        ),
                    )
                ],
            )
        ]

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        if (
            turn_start.backend is not None
            and turn_start.backend.strip() != ""
            and turn_start.backend != self._name
        ):
            return AgentError(
                message=f"unknown backend {turn_start.backend!r}",
                code="unknown_backend",
            )
        if turn_start.provider not in (None, "", "chat"):
            return AgentError(
                message=f"unknown provider {turn_start.provider!r}",
                code="unknown_provider",
            )
        if turn_start.model not in (None, "", "none"):
            return AgentError(
                message=f"unknown model {turn_start.model!r} for provider 'chat'; available models: none",
                code="unknown_model",
            )
        unsupported_output_modalities = [
            output
            for output in (turn_start.output_modalities or [])
            if output != "text"
        ]
        if len(unsupported_output_modalities) > 0:
            unsupported = ", ".join(
                repr(item) for item in unsupported_output_modalities
            )
            return AgentError(
                message=f"model 'none' does not support {unsupported} output modalities",
                code="unsupported_modality",
            )
        return None

    async def create_realtime_connection(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> Any:
        del supervisor
        del thread_id
        del start_thread
        del sender
        return None

    async def create_thread(
        self,
        *,
        supervisor: AgentSupervisor,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> CreatedAgentThread:
        thread_id = await self._thread_id_factory(start_thread, sender)
        name = _fallback_start_thread_name(start_thread)

        async def create_metadata() -> ThreadListEntry | None:
            with tracer.start_as_current_span("agent.thread.storage.upsert"):
                return await supervisor.on_thread_started(
                    created_thread=CreatedAgentThread(thread_id=thread_id, name=name),
                    start_thread=start_thread,
                    sender=sender,
                )

        return CreatedAgentThread(
            thread_id=thread_id,
            name=name,
            metadata=create_metadata,
        )

    def create_thread_process(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
    ) -> AgentProcess:
        del supervisor
        if self._process_factory is not None:
            return self._process_factory(thread_id, self._name)
        return ChatAgentProcess(thread_id=thread_id, backend=self._name)

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None


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

    def emit(
        self,
        *,
        sender: Participant | None,
        payload: AgentMessage,
        to_participant_id: str | None = None,
    ) -> None:
        supervisor = self.supervisor
        if supervisor is None:
            return

        supervisor.send(
            Message(
                data=payload,
                sender=sender,
                source=self,
                to_participant_id=to_participant_id,
            )
        )

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        del participant
        del payload
        return False

    async def send_agent_message_to_participant_and_wait(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        return self.send_agent_message_to_participant(
            participant=participant,
            payload=payload,
        )

    def get_turn_toolkits(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
    ) -> list[Toolkit]:
        del thread_id
        del turn_id
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

ThreadIsolationMode = Literal["global", "participant"]


class AgentSupervisor:
    def __init__(
        self,
        *,
        thread_isolation: ThreadIsolationMode = "global",
        thread_storage_repository: ThreadStorageRepository | None = None,
        agent_backends: list[AgentBackend] | None = None,
    ) -> None:
        self.channels: list[Channel] = []
        self.processes: list[AgentProcess] = []
        self._agent_backends = list(agent_backends or [])
        self._agent_backends_by_name = {
            backend.name: backend for backend in self._agent_backends
        }
        if len(self._agent_backends_by_name) != len(self._agent_backends):
            raise ValueError("agent backend names must be unique")
        self._default_agent_backend = (
            self._agent_backends[0] if len(self._agent_backends) > 0 else None
        )
        self._thread_storage_repository = (
            thread_storage_repository or NoopThreadStorageRepository()
        )
        self._stop = asyncio.Event()
        self._state: SupervisorState = "stopped"
        self._run_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._lifecycle_lock = asyncio.Lock()
        self._route_lock = asyncio.Lock()
        self._participants_by_client_id: dict[str, Participant] = {}
        self._participant_connection_counts_by_client_id: dict[str, int] = {}
        self._thread_watchers_by_client_id: dict[str, Participant] = {}
        self._open_thread_ids_by_client_id: dict[str, set[str]] = {}
        self._open_client_ids_by_thread_id: dict[str, set[str]] = {}
        self._thread_namespace_by_thread_id: dict[str, str | None] = {}
        self._backend_by_thread_id: dict[str, str] = {}
        self._thread_isolation: ThreadIsolationMode = thread_isolation
        self._pending_stop_thread_ids_after_turn: set[str] = set()
        self._pending_stop_thread_tasks_by_thread_id: dict[str, asyncio.Task[None]] = {}
        self._pending_thread_metadata_tasks: set[asyncio.Task[None]] = set()

    @property
    def state(self) -> SupervisorState:
        return self._state

    @property
    def thread_storage_repository(self) -> ThreadStorageRepository:
        return self._thread_storage_repository

    @property
    def agent_backends(self) -> list[AgentBackend]:
        return list(self._agent_backends)

    @property
    def default_agent_backend(self) -> AgentBackend | None:
        return self._default_agent_backend

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def stop_channel(self, channel: Channel) -> None:
        if channel in self.channels:
            self.channels.remove(channel)

    def add_process(self, process: AgentProcess) -> None:
        self.processes.append(process)

    def get_turn_toolkits(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
        thread_storage: ThreadStorage | None = None,
    ) -> list[Toolkit]:
        toolkits: list[Toolkit] = []
        seen_tools: set[tuple[str, str]] = set()

        def append_toolkit(toolkit: Toolkit) -> None:
            for toolkit_tool in toolkit.tools:
                key = (toolkit.name, toolkit_tool.name)
                if key in seen_tools:
                    raise ValueError(
                        f"duplicate turn tool registered: {toolkit.name}.{toolkit_tool.name}"
                    )
                seen_tools.add(key)
            toolkits.append(toolkit)

        if thread_storage is not None:
            append_toolkit(self._make_thread_list_toolkit())

        for channel in self.channels:
            for toolkit in channel.get_turn_toolkits(
                thread_id=thread_id,
                turn_id=turn_id,
            ):
                append_toolkit(toolkit)

        return toolkits

    def _make_thread_list_toolkit(self) -> Toolkit:
        read_file_hint = (
            "Use read_file with a thread path to read that thread's contents."
        )
        outer = self

        def to_json_entry(entry: ThreadListEntry) -> dict[str, str]:
            return {
                "name": str(entry.name),
                "path": str(entry.path),
                "modified_at": str(entry.modified_at),
                "created_at": str(entry.created_at),
            }

        @tool(
            name="list_threads",
            description="lists recent threads sorted by last modified date (newest first). Use read_file with a thread path to read that thread's contents.",
        )
        async def list_threads(
            context: ToolContext,
            *,
            limit: int = 20,
            offset: int = 0,
        ) -> JsonContent:
            normalized_offset = max(0, int(offset))
            normalized_limit = max(1, min(200, int(limit)))
            page = await outer.list_threads(
                list_threads=ListThreads(
                    type=AGENT_MESSAGE_THREAD_LIST,
                    limit=normalized_limit,
                    offset=normalized_offset,
                ),
                sender=context.on_behalf_of or context.caller,
            )

            if page.total == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": 0,
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "no threads were found in the thread list",
                        "read_file_hint": read_file_hint,
                    }
                )

            if len(page.threads) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total": page.total,
                        "offset": normalized_offset,
                        "limit": normalized_limit,
                        "message": "no threads were found for the requested limit/offset",
                        "read_file_hint": read_file_hint,
                    }
                )

            return JsonContent(
                json={
                    "threads": [to_json_entry(entry) for entry in page.threads],
                    "total": page.total,
                    "offset": normalized_offset,
                    "limit": normalized_limit,
                    "sort": "modified_at_desc",
                    "read_file_hint": read_file_hint,
                }
            )

        @tool(
            name="grep_thread_list",
            description="searches the thread list for matching thread names and paths. Use read_file with a thread path to read that thread's contents.",
        )
        async def grep_thread_list(
            context: ToolContext,
            *,
            pattern: str,
            ignore_case: bool = True,
        ) -> JsonContent:
            needle = pattern.strip()
            if needle == "":
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "message": "pattern is required",
                        "read_file_hint": read_file_hint,
                    }
                )

            flags = re.IGNORECASE if ignore_case else 0
            try:
                matcher = re.compile(needle, flags)
            except re.error as ex:
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "error": "invalid_regex_pattern",
                        "message": f"invalid regex pattern: {ex}",
                        "read_file_hint": read_file_hint,
                    }
                )

            matches: list[dict[str, str]] = []
            page = await outer.list_threads(
                list_threads=ListThreads(
                    type=AGENT_MESSAGE_THREAD_LIST,
                    limit=200,
                    offset=0,
                ),
                sender=context.on_behalf_of or context.caller,
            )
            for entry in page.threads:
                haystack = (
                    f"{entry.name}\n{entry.path}\n"
                    f"{entry.created_at}\n{entry.modified_at}"
                )
                if matcher.search(haystack) is not None:
                    matches.append(to_json_entry(entry))

            if len(matches) == 0:
                return JsonContent(
                    json={
                        "threads": [],
                        "total_matches": 0,
                        "pattern": needle,
                        "ignore_case": ignore_case,
                        "message": "no matching threads were found",
                        "read_file_hint": read_file_hint,
                    }
                )

            return JsonContent(
                json={
                    "threads": matches,
                    "total_matches": len(matches),
                    "pattern": needle,
                    "ignore_case": ignore_case,
                    "read_file_hint": read_file_hint,
                }
            )

        return Toolkit(
            name="chat",
            tools=[list_threads, grep_thread_list],
            validation_mode="content_types",
        )

    @property
    def thread_isolation(self) -> ThreadIsolationMode:
        return self._thread_isolation

    def set_thread_isolation(self, thread_isolation: ThreadIsolationMode) -> None:
        self._thread_isolation = thread_isolation

    def thread_namespace(self, *, thread_id: str) -> str | None:
        if self._thread_isolation != "participant":
            return None
        return self._thread_namespace_by_thread_id.get(thread_id)

    def _participant_namespace(self, *, participant: Participant | None) -> str | None:
        if participant is None:
            return None
        participant_name = participant.get_attribute("name")
        if isinstance(participant_name, str) and participant_name.strip() != "":
            return participant_name.strip()
        return None

    def _participants_for_namespace(
        self,
        *,
        namespace: str,
        fallback: Participant | None = None,
    ) -> list[Participant]:
        participants_by_id: dict[str, Participant] = {}
        for participant in self._participants_by_client_id.values():
            if self._participant_namespace(participant=participant) == namespace:
                participants_by_id[participant.id] = participant
        if (
            fallback is not None
            and self._participant_namespace(participant=fallback) == namespace
        ):
            participants_by_id[fallback.id] = fallback
        return list(participants_by_id.values())

    def _connected_participants(
        self,
        *,
        fallback: Participant | None = None,
    ) -> list[Participant]:
        participants_by_id = dict(self._participants_by_client_id)
        if fallback is not None:
            participants_by_id[fallback.id] = fallback
        return list(participants_by_id.values())

    def _track_thread_watcher(self, *, sender: Participant | None) -> None:
        if sender is None or sender.id.strip() == "":
            return
        self._thread_watchers_by_client_id[sender.id.strip()] = sender

    def _untrack_thread_watcher(self, *, sender: Participant | None) -> None:
        if sender is None or sender.id.strip() == "":
            return
        self._thread_watchers_by_client_id.pop(sender.id.strip(), None)

    def _thread_watchers(
        self,
        *,
        fallback: Participant | None = None,
    ) -> list[Participant]:
        participants_by_id = dict(self._thread_watchers_by_client_id)
        if fallback is not None:
            participants_by_id[fallback.id] = fallback
        return list(participants_by_id.values())

    def _thread_watchers_for_namespace(
        self,
        *,
        namespace: str,
        fallback: Participant | None = None,
    ) -> list[Participant]:
        participants_by_id: dict[str, Participant] = {}
        for participant in self._thread_watchers_by_client_id.values():
            if self._participant_namespace(participant=participant) == namespace:
                participants_by_id[participant.id] = participant
        if (
            fallback is not None
            and self._participant_namespace(participant=fallback) == namespace
        ):
            participants_by_id[fallback.id] = fallback
        return list(participants_by_id.values())

    def _participant_for_targeted_message(
        self,
        *,
        participant_id: str,
        fallback: Participant | None = None,
    ) -> Participant | None:
        client_id = participant_id.strip()
        if client_id == "":
            return None
        participant = self._participants_by_client_id.get(client_id)
        if participant is not None:
            return participant
        if fallback is not None and fallback.id == client_id:
            return fallback
        return None

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        backend = self.agent_backend_for_thread(thread_id=thread_id)
        if backend is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement create_thread_process"
            )
        return backend.create_thread_process(supervisor=self, thread_id=thread_id)

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
        to_participant_id: str | None = None,
    ) -> None:
        self.send(
            Message(
                data=payload,
                sender=sender,
                to_participant_id=to_participant_id,
            )
        )

    async def on_start(self) -> None:
        for backend in self._agent_backends:
            await backend.on_start()
        return None

    async def on_stop(self) -> None:
        for backend in reversed(self._agent_backends):
            await backend.on_stop()
        return None

    async def on_models_request(self, message: Message) -> None:
        if len(self._agent_backends) == 0:
            self._send_to_processes(message)
            return
        if not isinstance(message.data, ModelsRequest):
            return
        providers: list[AgentProviderInfo] = []
        for backend in self._agent_backends:
            providers.extend(
                backend.model_providers(
                    current_backend=None,
                    current_provider=None,
                    current_model=None,
                )
            )
        self._send_models_response(
            Message(
                data=ModelsResponse(
                    type=AGENT_MESSAGE_MODELS_RESPONSE,
                    source_message_id=message.data.message_id,
                    providers=providers,
                ),
                sender=message.sender,
            )
        )

    def _send_models_response(self, message: Message) -> None:
        self._send_to_channels(message)

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        if self._requires_explicit_backend(turn_start.backend):
            return self._missing_backend_error()
        backend = self.agent_backend_for_name(backend_name=turn_start.backend)
        if backend is None:
            if turn_start.backend is not None and turn_start.backend.strip() != "":
                return AgentError(
                    message=f"unknown backend {turn_start.backend!r}",
                    code="unknown_backend",
                )
            return None
        return await backend.validate_turn_start(turn_start)

    async def on_participant_connect_message(
        self, participant_connect: ParticipantConnect, sender: Participant | None
    ) -> ParticipantConnect:
        del sender
        return participant_connect

    async def on_participant_disconnect_message(
        self, participant_disconnect: ParticipantDisconnect, sender: Participant | None
    ) -> ParticipantDisconnect:
        del sender
        return participant_disconnect

    async def on_thread_start_message(
        self, start_thread: StartThread, sender: Participant | None
    ) -> StartThread:
        del sender
        return start_thread

    async def on_turn_start_message(
        self, turn_start: TurnStart, sender: Participant | None
    ) -> TurnStart:
        del sender
        return turn_start

    async def on_model_change_message(
        self, change_model: ChangeModel, sender: Participant | None
    ) -> ChangeModel:
        del sender
        return change_model

    async def on_turn_steer_message(
        self, turn_steer: TurnSteer, sender: Participant | None
    ) -> TurnSteer:
        del sender
        return turn_steer

    async def on_turn_interrupt_message(
        self, turn_interrupt: TurnInterrupt, sender: Participant | None
    ) -> TurnInterrupt:
        del sender
        return turn_interrupt

    async def on_realtime_audio_chunk_message(
        self, audio_chunk: AgentRealtimeAudioChunk, sender: Participant | None
    ) -> AgentRealtimeAudioChunk:
        del sender
        return audio_chunk

    async def on_realtime_audio_commit_message(
        self, audio_commit: AgentRealtimeAudioCommit, sender: Participant | None
    ) -> AgentRealtimeAudioCommit:
        del sender
        return audio_commit

    async def on_tool_call_approve_message(
        self, approval: ApproveAgentToolCall, sender: Participant | None
    ) -> ApproveAgentToolCall:
        del sender
        return approval

    async def on_tool_call_reject_message(
        self, rejection: RejectAgentToolCall, sender: Participant | None
    ) -> RejectAgentToolCall:
        del sender
        return rejection

    async def on_secret_response_message(
        self, response: AgentSecretResponse, sender: Participant | None
    ) -> AgentSecretResponse:
        del sender
        return response

    async def on_thread_clear_message(
        self, clear_thread: ClearThread, sender: Participant | None
    ) -> ClearThread:
        del sender
        return clear_thread

    async def on_thread_list_message(
        self, list_threads: ListThreads, sender: Participant | None
    ) -> ListThreads:
        del sender
        return list_threads

    async def on_thread_watch_message(
        self, watch_threads: WatchThreads, sender: Participant | None
    ) -> WatchThreads:
        del sender
        return watch_threads

    async def on_thread_unwatch_message(
        self, unwatch_threads: UnwatchThreads, sender: Participant | None
    ) -> UnwatchThreads:
        del sender
        return unwatch_threads

    async def on_thread_delete_message(
        self, delete_thread: DeleteThread, sender: Participant | None
    ) -> DeleteThread:
        del sender
        return delete_thread

    async def on_thread_rename_message(
        self, rename_thread: RenameThread, sender: Participant | None
    ) -> RenameThread:
        del sender
        return rename_thread

    async def on_thread_open_message(
        self, open_thread: OpenThread, sender: Participant | None
    ) -> OpenThread:
        del sender
        return open_thread

    async def on_thread_close_message(
        self, close_thread: CloseThread, sender: Participant | None
    ) -> CloseThread:
        del sender
        return close_thread

    async def on_models_request_message(
        self, models_request: ModelsRequest, sender: Participant | None
    ) -> ModelsRequest:
        del sender
        return models_request

    async def create_realtime_connection(
        self,
        *,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ):
        if self._requires_explicit_backend(start_thread.backend):
            raise ValueError(self._missing_backend_error().message)
        backend = self.agent_backend_for_name(backend_name=start_thread.backend)
        if backend is not None:
            return await backend.create_realtime_connection(
                supervisor=self,
                thread_id=thread_id,
                start_thread=start_thread,
                sender=sender,
            )
        return None

    async def create_thread_id(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del start_thread
        del sender
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_thread_id"
        )

    async def create_thread(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> CreatedAgentThread:
        if self._requires_explicit_backend(start_thread.backend):
            raise ValueError(self._missing_backend_error().message)
        backend = self.agent_backend_for_name(backend_name=start_thread.backend)
        if backend is not None:
            return await backend.create_thread(
                supervisor=self,
                start_thread=start_thread,
                sender=sender,
            )
        thread_id = await self.create_thread_id(
            start_thread=start_thread,
            sender=sender,
        )
        metadata_name = _fallback_start_thread_name(start_thread)
        immediate_name = (
            start_thread.name.strip()
            if isinstance(start_thread.name, str) and start_thread.name.strip() != ""
            else None
        )
        return CreatedAgentThread(
            thread_id=thread_id,
            name=immediate_name,
            metadata=lambda: self.on_thread_started(
                created_thread=CreatedAgentThread(
                    thread_id=thread_id,
                    name=metadata_name,
                ),
                start_thread=start_thread,
                sender=sender,
            ),
        )

    def agent_backend_for_name(
        self, *, backend_name: str | None
    ) -> AgentBackend | None:
        if backend_name is None or backend_name.strip() == "":
            return self._default_agent_backend
        return self._agent_backends_by_name.get(backend_name)

    def _requires_explicit_backend(self, backend_name: str | None) -> bool:
        return len(self._agent_backends) > 1 and (
            backend_name is None or backend_name.strip() == ""
        )

    def _missing_backend_error(self) -> AgentError:
        names = ", ".join(backend.name for backend in self._agent_backends)
        return AgentError(
            message=(
                "backend is required when multiple agent backends are configured; "
                f"available backends: {names}"
            ),
            code="backend_required",
        )

    def agent_backend_for_thread(self, *, thread_id: str) -> AgentBackend | None:
        backend_name = self._backend_by_thread_id.get(thread_id)
        return self.agent_backend_for_name(backend_name=backend_name)

    def set_thread_backend(
        self,
        *,
        thread_id: str,
        backend: AgentBackend | None,
    ) -> None:
        if backend is None:
            self._backend_by_thread_id.pop(thread_id, None)
        else:
            self._backend_by_thread_id[thread_id] = backend.name

    async def on_thread_started(
        self,
        *,
        created_thread: CreatedAgentThread,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del start_thread
        del sender
        return await self.thread_storage_repository.upsert_thread(
            path=created_thread.thread_id,
            name=created_thread.name,
        )

    def _created_thread_list_entry(
        self,
        *,
        created_thread: CreatedAgentThread,
        start_thread: StartThread,
    ) -> ThreadListEntry:
        return ThreadListEntry(
            path=created_thread.thread_id,
            name=created_thread.name or DEFAULT_THREAD_NAME,
            created_at=start_thread.created_at,
            modified_at=start_thread.created_at,
        )

    def _emit_created_thread_and_publish_metadata(
        self,
        *,
        created_thread: CreatedAgentThread,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> None:
        immediate_entry = self._created_thread_list_entry(
            created_thread=created_thread,
            start_thread=start_thread,
        )
        self._emit_thread_created(entry=immediate_entry, sender=sender)

        if created_thread.metadata is None:
            return

        async def publish_metadata() -> None:
            try:
                entry = await created_thread.metadata()
                if entry is None:
                    return
                if (
                    entry.path != immediate_entry.path
                    or entry.name != immediate_entry.name
                ):
                    self._emit_thread_updated(entry=entry, sender=sender)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to publish thread metadata")

        task = asyncio.create_task(publish_metadata())
        self._pending_thread_metadata_tasks.add(task)
        task.add_done_callback(self._pending_thread_metadata_tasks.discard)

    async def on_thread_deleted(
        self,
        *,
        delete_thread: DeleteThread,
        sender: Participant | None,
    ) -> None:
        del sender
        await self.thread_storage_repository.delete_thread(
            path=delete_thread.thread_id,
        )

    async def on_thread_renamed(
        self,
        *,
        rename_thread: RenameThread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del sender
        name = " ".join(rename_thread.name.split())
        if name == "":
            return None
        return await self.thread_storage_repository.rename_thread(
            path=rename_thread.thread_id,
            name=name,
        )

    async def list_threads(
        self,
        *,
        list_threads: ListThreads,
        sender: Participant | None,
    ) -> ThreadListPage:
        del sender
        return await self.thread_storage_repository.list_threads(
            limit=list_threads.limit,
            offset=list_threads.offset,
        )

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
                logger.info(
                    "agent supervisor starting with %d channels and %d processes",
                    len(self.channels),
                    len(self.processes),
                )
                await self.on_start()
                await self._ensure_children_started(fatal=True)
                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
                logger.info("agent supervisor started")
            except Exception:
                logger.exception("agent supervisor failed during start")
                with contextlib.suppress(Exception):
                    await self._stop_children()
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
                for task in self._pending_stop_thread_tasks_by_thread_id.values():
                    task.cancel()
                self._pending_stop_thread_tasks_by_thread_id.clear()
                for task in self._pending_thread_metadata_tasks:
                    task.cancel()
                self._pending_thread_metadata_tasks.clear()
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

    async def _stop_thread_process(self, *, thread_id: str) -> None:
        self._pending_stop_thread_ids_after_turn.discard(thread_id)
        task = self._pending_stop_thread_tasks_by_thread_id.pop(thread_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._forget_thread_tracking(thread_id=thread_id)
        self._backend_by_thread_id.pop(thread_id, None)
        await self._stop_thread_process_without_forgetting_tracking(thread_id=thread_id)

    async def _stop_thread_process_without_forgetting_tracking(
        self,
        *,
        thread_id: str,
    ) -> None:
        process = self._process_for_thread(thread_id=thread_id)
        if process is None:
            return
        if process.supervisor is self:
            with contextlib.suppress(ValueError):
                await process.stop(self)
        if process in self.processes:
            self.processes.remove(process)

    async def _replace_thread_process_backend(
        self,
        *,
        thread_id: str,
        backend: AgentBackend,
    ) -> tuple[AgentProcess | None, AgentError | None]:
        current_process = self._process_for_thread(thread_id=thread_id)
        if current_process is not None and current_process.backend == backend.name:
            return current_process, None

        self._pending_stop_thread_ids_after_turn.discard(thread_id)
        task = self._pending_stop_thread_tasks_by_thread_id.pop(thread_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

        if current_process is not None:
            await self._stop_thread_process_without_forgetting_tracking(
                thread_id=thread_id
            )
        self.set_thread_backend(thread_id=thread_id, backend=backend)
        return self._create_thread_process_for_route(thread_id=thread_id)

    async def _stop_thread_process_when_idle(self, *, thread_id: str) -> None:
        process = self._process_for_thread(thread_id=thread_id)
        if process is not None and process.turn_id is not None:
            self._pending_stop_thread_ids_after_turn.add(thread_id)
            self._schedule_pending_thread_stop(thread_id=thread_id)
            return
        await self._stop_thread_process(thread_id=thread_id)

    def _thread_has_open_clients(self, *, thread_id: str) -> bool:
        client_ids = self._open_client_ids_by_thread_id.get(thread_id)
        return client_ids is not None and len(client_ids) > 0

    async def _stop_pending_thread_process_when_idle(self, *, thread_id: str) -> None:
        try:
            while thread_id in self._pending_stop_thread_ids_after_turn:
                if self._stop.is_set() or self._state == "stopping":
                    return
                if self._thread_has_open_clients(thread_id=thread_id):
                    await asyncio.sleep(0.05)
                    continue
                process = self._process_for_thread(thread_id=thread_id)
                if process is None:
                    self._pending_stop_thread_ids_after_turn.discard(thread_id)
                    return
                if process.turn_id is None:
                    await self._stop_thread_process(thread_id=thread_id)
                    return
                await asyncio.sleep(0.05)
        finally:
            task = self._pending_stop_thread_tasks_by_thread_id.get(thread_id)
            if task is asyncio.current_task():
                self._pending_stop_thread_tasks_by_thread_id.pop(thread_id, None)

    def _schedule_pending_thread_stop(self, *, thread_id: str) -> None:
        task = self._pending_stop_thread_tasks_by_thread_id.get(thread_id)
        if task is not None and not task.done():
            return
        self._pending_stop_thread_tasks_by_thread_id[thread_id] = asyncio.create_task(
            self._stop_pending_thread_process_when_idle(thread_id=thread_id)
        )

    def _defer_stop_thread_process_after_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        async def stop_after_turn_cleanup() -> None:
            for _ in range(20):
                await asyncio.sleep(0.01)
                process = self._process_for_thread(thread_id=thread_id)
                if process is None or process.turn_id != turn_id:
                    break
            if self._thread_has_open_clients(thread_id=thread_id):
                self._pending_stop_thread_ids_after_turn.discard(thread_id)
                return
            process = self._process_for_thread(thread_id=thread_id)
            if process is not None and process.turn_id is not None:
                return
            await self._stop_thread_process(thread_id=thread_id)

        asyncio.create_task(stop_after_turn_cleanup())

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
            to_participant_id=message.to_participant_id,
        )

    @staticmethod
    def _client_id_for_thread_tracking(message: Message) -> str | None:
        sender = message.sender
        if sender is None or sender.id.strip() == "":
            return None
        return sender.id

    def _participant_namespace_for_message(self, message: Message) -> str | None:
        return self._participant_namespace(participant=message.sender)

    def _thread_access_error(self, *, thread_id: str) -> AgentError:
        del thread_id
        return AgentError(
            message="thread is not available to this participant",
            code="thread_not_available",
        )

    def _ensure_thread_access_for_message(
        self,
        *,
        thread_id: str,
        message: Message,
    ) -> AgentError | None:
        if self._thread_isolation == "global":
            return None
        if isinstance(message.source, AgentProcess):
            return None

        namespace = self._participant_namespace_for_message(message)
        if namespace is None:
            return AgentError(
                message="participant name is required for isolated threads",
                code="participant_name_required",
            )

        existing = self._thread_namespace_by_thread_id.get(thread_id)
        if existing is None:
            self._thread_namespace_by_thread_id[thread_id] = namespace
            return None
        if existing != namespace:
            return self._thread_access_error(thread_id=thread_id)
        return None

    def _track_thread_open_for_client(
        self,
        *,
        thread_id: str,
        client_id: str | None,
    ) -> bool:
        if client_id is None:
            return False

        self._pending_stop_thread_ids_after_turn.discard(thread_id)
        task = self._pending_stop_thread_tasks_by_thread_id.pop(thread_id, None)
        if task is not None:
            task.cancel()
        thread_ids = self._open_thread_ids_by_client_id.setdefault(client_id, set())
        client_ids = self._open_client_ids_by_thread_id.setdefault(
            thread_id,
            set(),
        )
        was_open = len(client_ids) > 0
        thread_ids.add(thread_id)
        client_ids.add(client_id)
        return was_open

    def _track_thread_open_for_message(
        self,
        *,
        thread_id: str,
        message: Message,
    ) -> bool:
        return self._track_thread_open_for_client(
            thread_id=thread_id,
            client_id=self._client_id_for_thread_tracking(message),
        )

    def _track_thread_close_for_client(
        self,
        *,
        thread_id: str,
        client_id: str | None,
    ) -> bool:
        if client_id is None:
            self._forget_thread_tracking(thread_id=thread_id)
            return False

        thread_ids = self._open_thread_ids_by_client_id.get(client_id)
        if thread_ids is not None:
            thread_ids.discard(thread_id)
            if len(thread_ids) == 0:
                self._open_thread_ids_by_client_id.pop(client_id, None)

        client_ids = self._open_client_ids_by_thread_id.get(thread_id)
        if client_ids is None:
            return False

        client_ids.discard(client_id)
        if len(client_ids) > 0:
            return True

        self._open_client_ids_by_thread_id.pop(thread_id, None)
        return False

    def _track_participant_connected(
        self,
        *,
        participant_id: str,
        sender: Participant | None,
    ) -> None:
        client_id = participant_id.strip()
        if client_id == "":
            return
        if sender is not None:
            self._participants_by_client_id[client_id] = sender
        self._participant_connection_counts_by_client_id[client_id] = (
            self._participant_connection_counts_by_client_id.get(client_id, 0) + 1
        )

    def is_participant_connected(self, *, participant_id: str) -> bool:
        client_id = participant_id.strip()
        if client_id == "":
            return False
        return self._participant_connection_counts_by_client_id.get(client_id, 0) > 0

    async def _track_participant_disconnected(self, *, participant_id: str) -> None:
        client_id = participant_id.strip()
        if client_id == "":
            return

        connection_count = (
            self._participant_connection_counts_by_client_id.get(client_id, 0) - 1
        )
        if connection_count > 0:
            self._participant_connection_counts_by_client_id[client_id] = (
                connection_count
            )
            return

        self._participant_connection_counts_by_client_id.pop(client_id, None)
        self._participants_by_client_id.pop(client_id, None)
        self._thread_watchers_by_client_id.pop(client_id, None)
        for process in list(self.processes):
            await process.on_participant_disconnected(participant_id=client_id)
        thread_ids = list(self._open_thread_ids_by_client_id.pop(client_id, set()))
        for thread_id in thread_ids:
            client_ids = self._open_client_ids_by_thread_id.get(thread_id)
            if client_ids is None:
                continue
            client_ids.discard(client_id)
            if len(client_ids) > 0:
                continue
            self._open_client_ids_by_thread_id.pop(thread_id, None)
            await self._stop_thread_process_when_idle(thread_id=thread_id)

    def _forget_thread_tracking(self, *, thread_id: str) -> None:
        client_ids = self._open_client_ids_by_thread_id.pop(thread_id, set())
        for client_id in client_ids:
            thread_ids = self._open_thread_ids_by_client_id.get(client_id)
            if thread_ids is None:
                continue
            thread_ids.discard(thread_id)
            if len(thread_ids) == 0:
                self._open_thread_ids_by_client_id.pop(client_id, None)

    def _send_to_channels(self, message: Message) -> None:
        to_participant_id = message.to_participant_id
        if to_participant_id is not None:
            participant = self._participant_for_targeted_message(
                participant_id=to_participant_id,
                fallback=message.sender,
            )
            if participant is None:
                logger.debug(
                    "dropping targeted agent message %s for disconnected participant %s",
                    message.data.type,
                    to_participant_id,
                )
                return
            for channel in self.channels:
                if message.source is channel:
                    continue
                channel.send_agent_message_to_participant(
                    participant=participant,
                    payload=message.data,
                )
            return

        for channel in self.channels:
            if message.source is channel:
                continue
            channel.send(message)

    def _send_thread_list_response(
        self,
        *,
        request: Message,
        response: ThreadsListed,
    ) -> None:
        if request.sender is not None:
            for channel in self.channels:
                if channel.send_agent_message_to_participant(
                    participant=request.sender,
                    payload=response,
                ):
                    return
        self._send_to_channels(Message(data=response, sender=request.sender))

    def _agent_thread_list_entry(self, entry: ThreadListEntry) -> AgentThreadListEntry:
        return AgentThreadListEntry(
            path=entry.path,
            name=entry.name,
            created_at=entry.created_at,
            modified_at=entry.modified_at,
        )

    def _emit_thread_created(
        self, *, entry: ThreadListEntry, sender: Participant | None
    ) -> None:
        self._send_thread_list_event(
            payload=ThreadCreated(
                type=AGENT_EVENT_THREAD_CREATED,
                thread=self._agent_thread_list_entry(entry),
            ),
            sender=sender,
        )

    def _emit_thread_updated(
        self, *, entry: ThreadListEntry, sender: Participant | None
    ) -> None:
        self._send_thread_list_event(
            payload=ThreadUpdated(
                type=AGENT_EVENT_THREAD_UPDATED,
                thread=self._agent_thread_list_entry(entry),
            ),
            sender=sender,
        )

    def _emit_thread_deleted(self, *, path: str, sender: Participant | None) -> None:
        self._send_thread_list_event(
            payload=ThreadDeleted(type=AGENT_EVENT_THREAD_DELETED, path=path),
            sender=sender,
        )

    def _send_thread_list_event(
        self,
        *,
        payload: ThreadCreated | ThreadUpdated | ThreadDeleted,
        sender: Participant | None,
    ) -> None:
        if self._thread_isolation == "participant":
            namespace = self._participant_namespace(participant=sender)
            if namespace is None:
                return
            participants = self._thread_watchers_for_namespace(
                namespace=namespace,
                fallback=sender,
            )
        else:
            participants = self._thread_watchers(fallback=sender)

        for participant in participants:
            for channel in self.channels:
                channel.send_agent_message_to_participant(
                    participant=participant,
                    payload=payload,
                )

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

    def _buffer_unflushed_thread_storage_for_open(
        self,
        *,
        processes: list[AgentProcess] | None,
    ) -> None:
        if processes is None:
            return

        for process in processes:
            thread_storage = process.thread_storage
            if thread_storage is None:
                continue
            for agent_message in thread_storage.unflushed_agent_messages():
                message = Message(data=agent_message, source=process)
                self._send_to_channels(message)

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

    def _emit_start_thread_rejected(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
        thread_id: str,
        error: AgentError,
    ) -> None:
        self._send_to_channels(
            Message(
                data=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=thread_id,
                    source_message_id=start_thread.message_id,
                    error=error,
                ),
                sender=sender,
            )
        )

    def _emit_thread_started(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
        thread_id: str,
        realtime_connection: Any = None,
    ) -> None:
        self._send_to_channels(
            Message(
                data=ThreadStarted(
                    type=AGENT_EVENT_THREAD_STARTED,
                    source_message_id=start_thread.message_id,
                    thread_id=thread_id,
                    realtime_connection=realtime_connection,
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
        if message.to_participant_id is not None:
            self._send_to_channels(message)
            return

        if message_type == AGENT_MESSAGE_PARTICIPANT_CONNECT:
            participant_connect = await self.on_participant_connect_message(
                _coerce_message_data(message.data, ParticipantConnect),
                message.sender,
            )
            self._track_participant_connected(
                participant_id=participant_connect.participant_id,
                sender=message.sender,
            )
            return

        elif message_type == AGENT_MESSAGE_PARTICIPANT_DISCONNECT:
            participant_disconnect = await self.on_participant_disconnect_message(
                _coerce_message_data(
                    message.data,
                    ParticipantDisconnect,
                ),
                message.sender,
            )
            await self._track_participant_disconnected(
                participant_id=participant_disconnect.participant_id
            )
            return

        elif message_type == AGENT_EVENT_TURN_ENDED:
            turn_ended = _coerce_message_data(message.data, TurnEnded)
            if not self._thread_has_open_clients(thread_id=turn_ended.thread_id):
                self._pending_stop_thread_ids_after_turn.add(turn_ended.thread_id)
                self._schedule_pending_thread_stop(thread_id=turn_ended.thread_id)

        elif message_type == AGENT_EVENT_THREAD_CREATED:
            self._send_thread_list_event(
                payload=_coerce_message_data(message.data, ThreadCreated),
                sender=message.sender,
            )
            return

        elif message_type == AGENT_EVENT_THREAD_UPDATED:
            self._send_thread_list_event(
                payload=_coerce_message_data(message.data, ThreadUpdated),
                sender=message.sender,
            )
            return

        elif message_type == AGENT_EVENT_THREAD_DELETED:
            self._send_thread_list_event(
                payload=_coerce_message_data(message.data, ThreadDeleted),
                sender=message.sender,
            )
            return

        elif message_type == AGENT_MESSAGE_THREAD_START:
            start_thread = await self.on_thread_start_message(
                _coerce_message_data(message.data, StartThread),
                message.sender,
            )
            if self._requires_explicit_backend(start_thread.backend):
                self._emit_start_thread_rejected(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id="",
                    error=self._missing_backend_error(),
                )
                return
            backend = self.agent_backend_for_name(backend_name=start_thread.backend)
            if (
                backend is None
                and start_thread.backend is not None
                and start_thread.backend.strip() != ""
            ):
                self._emit_start_thread_rejected(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id="",
                    error=AgentError(
                        message=f"unknown backend {start_thread.backend!r}",
                        code="unknown_backend",
                    ),
                )
                return
            try:
                with tracer.start_as_current_span("agent.thread.create") as span:
                    span.set_attribute(
                        "agent.thread.backend", start_thread.backend or ""
                    )
                    created_thread = await self.create_thread(
                        start_thread=start_thread,
                        sender=message.sender,
                    )
                    span.set_attribute("agent.thread.id", created_thread.thread_id)
            except Exception as exc:
                logger.exception("failed to create thread; rejecting start thread")
                self._emit_start_thread_rejected(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id="",
                    error=self._thread_process_creation_rejection(error=exc),
                )
                return
            thread_id = created_thread.thread_id

            access_error = self._ensure_thread_access_for_message(
                thread_id=thread_id,
                message=message,
            )
            if access_error is not None:
                self._emit_start_thread_rejected(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id=thread_id,
                    error=access_error,
                )
                return

            self.set_thread_backend(thread_id=thread_id, backend=backend)
            if start_thread.content is None:
                try:
                    realtime_connection = await self.create_realtime_connection(
                        thread_id=thread_id,
                        start_thread=start_thread,
                        sender=message.sender,
                    )
                except Exception as exc:
                    self._emit_start_thread_rejected(
                        start_thread=start_thread,
                        sender=message.sender,
                        thread_id=thread_id,
                        error=AgentError(
                            message=str(exc),
                            code="realtime_connection_failed",
                        ),
                    )
                    return
                self._emit_created_thread_and_publish_metadata(
                    created_thread=created_thread,
                    start_thread=start_thread,
                    sender=message.sender,
                )
                self._emit_thread_started(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id=thread_id,
                    realtime_connection=realtime_connection,
                )
                self._track_thread_open_for_message(
                    thread_id=thread_id,
                    message=message,
                )
                return

            turn_start = TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=start_thread.message_id,
                thread_id=thread_id,
                turn_id=str(uuid.uuid4()),
                content=start_thread.content,
                sender_name=start_thread.sender_name,
                provider=start_thread.provider,
                backend=start_thread.backend,
                model=start_thread.model,
                voice=start_thread.voice,
                output_modalities=start_thread.output_modalities,
                instructions=start_thread.instructions,
                mcp=start_thread.mcp,
                toolkits=start_thread.toolkits,
                client_toolkits=start_thread.client_toolkits,
                tool_choice=start_thread.tool_choice,
                storage=start_thread.storage,
            )
            turn_start = await self.on_turn_start_message(
                turn_start,
                message.sender,
            )
            error = await self.validate_turn_start(turn_start)
            if error is not None:
                self._emit_start_thread_rejected(
                    start_thread=start_thread,
                    sender=message.sender,
                    thread_id=thread_id,
                    error=error,
                )
                return

            process = self._process_for_thread(thread_id=thread_id)
            if process is None:
                process, rejection = self._create_thread_process_for_route(
                    thread_id=thread_id
                )
                if process is None:
                    if rejection is not None:
                        self._emit_start_thread_rejected(
                            start_thread=start_thread,
                            sender=message.sender,
                            thread_id=thread_id,
                            error=rejection,
                        )
                    return
            self._emit_created_thread_and_publish_metadata(
                created_thread=created_thread,
                start_thread=start_thread,
                sender=message.sender,
            )
            self._emit_thread_started(
                start_thread=start_thread,
                sender=message.sender,
                thread_id=thread_id,
            )
            self._track_thread_open_for_message(thread_id=thread_id, message=message)
            routed_message = self._copy_message(message, data=turn_start)
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TURN_START:
            turn_start = await self.on_turn_start_message(
                _coerce_message_data(message.data, TurnStart),
                message.sender,
            )
            error = await self.validate_turn_start(turn_start)
            if error is not None:
                self._emit_turn_start_rejected(
                    turn_start=turn_start,
                    sender=message.sender,
                    error=error,
                )
                return
            access_error = self._ensure_thread_access_for_message(
                thread_id=turn_start.thread_id,
                message=message,
            )
            if access_error is not None:
                self._emit_turn_start_rejected(
                    turn_start=turn_start,
                    sender=message.sender,
                    error=access_error,
                )
                return
            self._track_thread_open_for_message(
                thread_id=turn_start.thread_id,
                message=message,
            )
            backend = self.agent_backend_for_name(backend_name=turn_start.backend)
            process = self._process_for_thread(thread_id=turn_start.thread_id)
            if backend is not None and (
                process is None or process.backend != backend.name
            ):
                process, rejection = await self._replace_thread_process_backend(
                    thread_id=turn_start.thread_id,
                    backend=backend,
                )
                if process is None:
                    if rejection is not None:
                        self._emit_turn_start_rejected(
                            turn_start=turn_start,
                            sender=message.sender,
                            error=rejection,
                        )
                    return
            elif process is None:
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

        elif message_type == AGENT_MESSAGE_MODEL_CHANGE:
            change_model = await self.on_model_change_message(
                _coerce_message_data(message.data, ChangeModel),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=change_model.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            backend = self.agent_backend_for_name(backend_name=change_model.backend)
            if (
                backend is None
                and change_model.backend is not None
                and change_model.backend.strip() != ""
            ):
                return
            process = self._process_for_thread(thread_id=change_model.thread_id)
            if backend is not None and (
                process is None or process.backend != backend.name
            ):
                process, _ = await self._replace_thread_process_backend(
                    thread_id=change_model.thread_id,
                    backend=backend,
                )
            elif process is None:
                process, _ = self._create_thread_process_for_route(
                    thread_id=change_model.thread_id
                )
            if process is None:
                return
            routed_message = self._copy_message(message, data=change_model)
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TURN_STEER:
            turn_steer = await self.on_turn_steer_message(
                _coerce_message_data(message.data, TurnSteer),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=turn_steer.thread_id,
                message=message,
            )
            if access_error is not None:
                self._emit_turn_steer_rejected(
                    turn_steer=turn_steer,
                    sender=message.sender,
                    error=access_error,
                )
                return
            self._track_thread_open_for_message(
                thread_id=turn_steer.thread_id,
                message=message,
            )
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
            turn_interrupt = await self.on_turn_interrupt_message(
                _coerce_message_data(message.data, TurnInterrupt),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=turn_interrupt.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            process = self._process_for_thread(thread_id=turn_interrupt.thread_id)
            routed_message = self._copy_message(
                message,
                data=turn_interrupt,
            )
            if process is not None and process.turn_id == turn_interrupt.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_REALTIME_AUDIO_CHUNK:
            audio_chunk = await self.on_realtime_audio_chunk_message(
                _coerce_message_data(message.data, AgentRealtimeAudioChunk),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=audio_chunk.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            process = self._process_for_thread(thread_id=audio_chunk.thread_id)
            if process is None:
                process, _ = self._create_thread_process_for_route(
                    thread_id=audio_chunk.thread_id
                )
            if process is None:
                return
            routed_message = self._copy_message(message, data=audio_chunk)
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_REALTIME_AUDIO_COMMIT:
            audio_commit = await self.on_realtime_audio_commit_message(
                _coerce_message_data(message.data, AgentRealtimeAudioCommit),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=audio_commit.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            process = self._process_for_thread(thread_id=audio_commit.thread_id)
            if process is None:
                process, _ = self._create_thread_process_for_route(
                    thread_id=audio_commit.thread_id
                )
            if process is None:
                return
            routed_message = self._copy_message(message, data=audio_commit)
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TOOL_CALL_APPROVE:
            approval = await self.on_tool_call_approve_message(
                _coerce_message_data(message.data, ApproveAgentToolCall),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=approval.thread_id,
                message=message,
            )
            if access_error is not None:
                return
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
            rejection = await self.on_tool_call_reject_message(
                _coerce_message_data(message.data, RejectAgentToolCall),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=rejection.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            process = self._process_for_thread(thread_id=rejection.thread_id)
            routed_message = self._copy_message(
                message,
                data=rejection,
            )
            if process is not None and process.turn_id == rejection.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_SECRET_RESPONSE:
            response = await self.on_secret_response_message(
                _coerce_message_data(message.data, AgentSecretResponse),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=response.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            process = self._process_for_thread(thread_id=response.thread_id)
            routed_message = self._copy_message(
                message,
                data=response,
            )
            if process is not None and process.turn_id == response.turn_id:
                target_processes = [process]
            else:
                target_processes = []

        elif message_type == AGENT_MESSAGE_THREAD_CLEAR:
            clear_thread = await self.on_thread_clear_message(
                _coerce_message_data(message.data, ClearThread),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=clear_thread.thread_id,
                message=message,
            )
            if access_error is not None:
                return
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

        elif message_type == AGENT_MESSAGE_THREAD_LIST:
            list_threads = await self.on_thread_list_message(
                _coerce_message_data(message.data, ListThreads),
                message.sender,
            )
            page = await self.list_threads(
                list_threads=list_threads,
                sender=message.sender,
            )
            response = ThreadsListed(
                type=AGENT_EVENT_THREAD_LISTED,
                source_message_id=list_threads.message_id,
                threads=[
                    AgentThreadListEntry(
                        path=entry.path,
                        name=entry.name,
                        created_at=entry.created_at,
                        modified_at=entry.modified_at,
                    )
                    for entry in page.threads
                ],
                total=page.total,
                offset=page.offset,
                limit=page.limit,
            )
            self._send_thread_list_response(request=message, response=response)
            return

        elif message_type == AGENT_MESSAGE_THREAD_WATCH:
            await self.on_thread_watch_message(
                _coerce_message_data(message.data, WatchThreads),
                message.sender,
            )
            self._track_thread_watcher(sender=message.sender)
            return

        elif message_type == AGENT_MESSAGE_THREAD_UNWATCH:
            await self.on_thread_unwatch_message(
                _coerce_message_data(message.data, UnwatchThreads),
                message.sender,
            )
            self._untrack_thread_watcher(sender=message.sender)
            return

        elif message_type == AGENT_MESSAGE_THREAD_DELETE:
            delete_thread = await self.on_thread_delete_message(
                _coerce_message_data(message.data, DeleteThread),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=delete_thread.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            await self.on_thread_deleted(
                delete_thread=delete_thread,
                sender=message.sender,
            )
            self._thread_namespace_by_thread_id.pop(delete_thread.thread_id, None)
            await self._stop_thread_process(thread_id=delete_thread.thread_id)
            self._emit_thread_deleted(
                path=delete_thread.thread_id,
                sender=message.sender,
            )
            return

        elif message_type == AGENT_MESSAGE_THREAD_RENAME:
            rename_thread = await self.on_thread_rename_message(
                _coerce_message_data(message.data, RenameThread),
                message.sender,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=rename_thread.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            updated_entry = await self.on_thread_renamed(
                rename_thread=rename_thread,
                sender=message.sender,
            )
            if updated_entry is not None:
                self._emit_thread_updated(
                    entry=updated_entry,
                    sender=message.sender,
                )
            return

        elif message_type in {AGENT_MESSAGE_THREAD_OPEN, AGENT_MESSAGE_THREAD_CLOSE}:
            if message_type == AGENT_MESSAGE_THREAD_OPEN:
                thread_message = await self.on_thread_open_message(
                    _coerce_message_data(message.data, OpenThread),
                    message.sender,
                )
            else:
                thread_message = await self.on_thread_close_message(
                    _coerce_message_data(message.data, CloseThread),
                    message.sender,
                )

            routed_message = self._copy_message(
                message,
                data=thread_message,
            )
            access_error = self._ensure_thread_access_for_message(
                thread_id=thread_message.thread_id,
                message=message,
            )
            if access_error is not None:
                return
            client_id = self._client_id_for_thread_tracking(message)
            if message_type == AGENT_MESSAGE_THREAD_OPEN:
                assert isinstance(thread_message, OpenThread)
                backend = self.agent_backend_for_name(
                    backend_name=thread_message.backend
                )
                if (
                    backend is None
                    and thread_message.backend is not None
                    and thread_message.backend.strip() != ""
                ):
                    return
                process = self._process_for_thread(thread_id=thread_message.thread_id)
                thread_was_open = self._track_thread_open_for_client(
                    thread_id=thread_message.thread_id,
                    client_id=client_id,
                )
                if (
                    thread_was_open
                    and thread_message.load is not True
                    and (
                        backend is None
                        or (process is not None and process.backend == backend.name)
                    )
                ):
                    self._send_to_channels(routed_message)
                    return
            else:
                thread_still_has_clients = self._track_thread_close_for_client(
                    thread_id=thread_message.thread_id,
                    client_id=client_id,
                )
                self._send_to_channels(routed_message)
                if thread_still_has_clients:
                    return
                await self._stop_thread_process_when_idle(
                    thread_id=thread_message.thread_id
                )
                return

            if backend is not None and (
                process is None or process.backend != backend.name
            ):
                process, _ = await self._replace_thread_process_backend(
                    thread_id=thread_message.thread_id,
                    backend=backend,
                )
            elif process is None:
                process, _ = self._create_thread_process_for_route(
                    thread_id=thread_message.thread_id
                )
            if process is None:
                return

            target_processes = [process]

        elif message_type == AGENT_MESSAGE_MODELS_REQUEST:
            models_request = await self.on_models_request_message(
                _coerce_message_data(message.data, ModelsRequest),
                message.sender,
            )
            await self.on_models_request(
                self._copy_message(message, data=models_request)
            )
            return

        with tracer.start_as_current_span("agent.route.dispatch"):
            target_processes = await self._ensure_routing_processes_started(
                processes=target_processes
            )
        if (
            message_type == AGENT_MESSAGE_THREAD_OPEN
            and isinstance(routed_message.data, OpenThread)
            and routed_message.data.load is not True
        ):
            self._buffer_unflushed_thread_storage_for_open(processes=target_processes)
        self._send_to_channels(routed_message)
        self._send_to_processes(routed_message, processes=target_processes)

    async def _ensure_children_started(self, *, fatal: bool = False) -> None:
        for channel in self.channels:
            if channel.state == "stopped":
                try:
                    await channel.start(self)
                except Exception:
                    logger.exception(
                        "channel %s failed during start",
                        channel.__class__.__name__,
                    )
                    if fatal:
                        raise

        for process in self.processes:
            if process.state == "stopped":
                try:
                    await process.start(self)
                except Exception:
                    logger.exception(
                        "process %s failed during start",
                        process.__class__.__name__,
                    )
                    if fatal:
                        raise

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
                        try:
                            await self._route(message)
                        except Exception:
                            logger.exception(
                                "agent supervisor failed while routing %s message",
                                message.data.type,
                            )
        except Exception:
            logger.exception("agent supervisor run loop failed")
            self._state = "failed"
            raise
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
        backend: str | None = None,
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
        self._backend = backend
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

        if isinstance(message.data, AgentRealtimeAudioChunk):
            from .dataset_thread_storage import DatasetThreadStorage

            return (
                isinstance(thread_storage, DatasetThreadStorage)
                and thread_storage.persist_audio_input
            )

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
    def backend(self) -> str | None:
        return self._backend

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
                if thread_storage is not None:
                    await thread_storage.start()
                await self.on_start()
                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
            except Exception:
                self._state = "failed"
                thread_storage = self._thread_storage
                if thread_storage is not None:
                    with contextlib.suppress(Exception):
                        await thread_storage.stop()
                self._supervisor = None
                self._run_task = None
                raise

    async def on_message(self, message: Message) -> None:
        del message
        return None

    async def on_participant_disconnected(self, *, participant_id: str) -> None:
        del participant_id
        return None

    async def _send_thread_open_response(
        self,
        *,
        request_message: Message,
        payload: AgentMessage,
    ) -> None:
        source = request_message.source
        sender = request_message.sender
        if source is not None and sender is not None and isinstance(source, Channel):
            sent = await source.send_agent_message_to_participant_and_wait(
                participant=sender,
                payload=payload,
            )
            if sent:
                return

        AgentProcess.emit(self, sender=sender, payload=payload)

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
                if thread_storage is not None:
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


class ChatAgentProcess(AgentProcess):
    def __init__(
        self,
        *,
        thread_id: str,
        backend: str = "chat",
        thread_storage: ThreadStorage | None = None,
    ) -> None:
        super().__init__(
            thread_id=thread_id,
            backend=backend,
            thread_storage=thread_storage,
        )
        self._agent_messages: list[AgentThreadMessage] = []

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_MESSAGE_THREAD_OPEN,
            AGENT_MESSAGE_TURN_START,
        }

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        thread_storage = self.thread_storage
        if (
            thread_storage is not None
            and isinstance(payload, AgentThreadMessage)
            and payload.thread_id == self.thread_id
        ):
            thread_storage.push_message(message=payload, sender=sender)

        if (
            isinstance(payload, AgentThreadMessage)
            and payload.thread_id == self.thread_id
        ):
            self._agent_messages.append(payload)

        super().emit(sender=sender, payload=payload)

    async def on_message(self, message: Message) -> None:
        if isinstance(message.data, OpenThread):
            await self._on_thread_open(message=message, request=message.data)
            return

        if not isinstance(message.data, TurnStart):
            return

        turn_start = message.data
        turn_id = turn_start.turn_id or str(uuid.uuid4())
        turn_start = turn_start.model_copy(update={"turn_id": turn_id})
        thread_storage = self.thread_storage
        if thread_storage is not None and turn_start.thread_id == self.thread_id:
            thread_storage.push_message(message=turn_start, sender=message.sender)
        self._agent_messages.append(turn_start)

        self.emit(
            sender=message.sender,
            payload=TurnStartAccepted(
                type=AGENT_EVENT_TURN_START_ACCEPTED,
                thread_id=turn_start.thread_id,
                turn_id=turn_id,
                source_message_id=turn_start.message_id,
                content=turn_start.content,
                sender_name=turn_start.sender_name,
            ),
        )
        self.emit(
            sender=message.sender,
            payload=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id=turn_start.thread_id,
                turn_id=turn_id,
                source_message_id=turn_start.message_id,
            ),
        )
        self.emit(
            sender=message.sender,
            payload=TurnEnded(
                type=AGENT_EVENT_TURN_ENDED,
                thread_id=turn_start.thread_id,
                turn_id=turn_id,
            ),
        )

    async def _on_thread_open(self, *, message: Message, request: OpenThread) -> None:
        thread_storage = self.thread_storage
        if thread_storage is not None:
            await thread_storage.wait_until_ready()

        if request.load is not True:
            return

        if thread_storage is not None:
            stored_messages = thread_storage.agent_messages()
        else:
            stored_messages = []
        messages = list(stored_messages)
        for live_message in self._agent_messages:
            if live_message not in messages:
                messages.append(live_message)

        for stored_message in self._agent_messages_since_turn(
            messages=messages,
            thread_id=request.thread_id,
            since_turn=request.since_turn,
        ):
            await self._send_thread_open_response(
                request_message=message,
                payload=stored_message,
            )

        await self._send_thread_open_response(
            request_message=message,
            payload=ThreadLoaded(
                type=AGENT_EVENT_THREAD_LOADED,
                thread_id=request.thread_id,
                source_message_id=request.message_id,
                since_turn=request.since_turn,
            ),
        )

    @staticmethod
    def _agent_messages_since_turn(
        *,
        messages: list[AgentThreadMessage],
        thread_id: str,
        since_turn: str | None,
    ) -> list[AgentThreadMessage]:
        if since_turn is None or since_turn.strip() == "":
            return messages

        normalized_since_turn = since_turn.strip()
        for index, stored_message in enumerate(messages):
            if stored_message.thread_id != thread_id:
                continue
            if stored_message.message_id == normalized_since_turn:
                return messages[index:]
            turn_id = _message_turn_id(stored_message)
            if turn_id == normalized_since_turn:
                return messages[index:]
            if isinstance(stored_message, (TurnStart, TurnSteer)):
                if stored_message.message_id == normalized_since_turn:
                    return messages[index:]
        return []


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


@dataclass(slots=True)
class _PendingRealtimeAudioChunk:
    data: bytes
    format: AgentAudioFormat


@dataclass(slots=True)
class _PendingClientToolCall:
    future: asyncio.Future[Content]
    participant: Participant
    thread_id: str
    turn_id: str
    request_id: str
    toolkit: str
    tool: str


class _AgentMessageProxyClientTool(FunctionTool):
    def __init__(
        self,
        *,
        description: ClientToolkitDescription,
        request_tool_call: Callable[[str, dict[str, Any]], Awaitable[Content]],
    ) -> None:
        self._request_tool_call = request_tool_call
        super().__init__(
            name=description.name,
            title=description.title,
            description=description.description,
            input_schema=description.input_schema,
        )

    async def execute(self, context: ToolContext, **kwargs: Any) -> Content:
        del context
        return await self._request_tool_call(self.name, kwargs)


class LLMAgentProcess(AgentProcess):
    def __init__(
        self,
        *,
        thread_id: str,
        participant: Participant,
        llm_adapter: LLMAdapter | None = None,
        llm_providers: list[LLMProvider] | None = None,
        default_provider: LLMProvider | None = None,
        backend_name: str | None = "llm",
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

        super().__init__(
            thread_id=thread_id,
            thread_storage=thread_storage,
            backend=backend_name,
        )
        self._thread_status_publisher = thread_status_publisher
        self._format_message = format_message or default_format_message
        providers = self._normalize_llm_providers(
            llm_adapter=llm_adapter,
            llm_providers=llm_providers,
            default_provider=default_provider,
        )
        self._llm_providers_by_name = {
            provider.name: provider for provider in providers
        }
        self._default_provider = (
            self._llm_providers_by_name[default_provider.name]
            if default_provider is not None
            else providers[0]
        )
        self._backend_name = backend_name
        self._current_provider = self._default_provider
        self.llm_adapter = self._current_provider.adapter
        self._current_model = self.llm_adapter.default_model()
        current_model_info = self._current_model_info()
        self._current_voice = (
            current_model_info.default_output_voice
            if current_model_info is not None
            else None
        )
        self._current_output_modalities: tuple[Literal["text", "audio"], ...] = (
            "text",
        )
        self._turn_id: str | None = None
        self._last_usage_update: AgentUsageUpdated | None = None
        self._handlers: dict[str, Callable[[Message], Awaitable[None]]] = {
            AGENT_MESSAGE_THREAD_OPEN: self.on_thread_open,
            AGENT_MESSAGE_THREAD_CLOSE: self.on_thread_close,
            AGENT_MESSAGE_TURN_START: self.on_turn_start,
            AGENT_MESSAGE_TURN_STEER: self.on_turn_steer,
            AGENT_MESSAGE_TURN_INTERRUPT: self.on_turn_interrupt,
            AGENT_MESSAGE_REALTIME_AUDIO_CHUNK: self.on_realtime_audio_chunk,
            AGENT_MESSAGE_REALTIME_AUDIO_COMMIT: self.on_realtime_audio_commit,
            AGENT_MESSAGE_CAPABILITIES_REQUEST: self.on_capabilities_request,
            AGENT_MESSAGE_MODELS_REQUEST: self.on_models_request,
            AGENT_MESSAGE_MODEL_CHANGE: self.on_model_change,
            AGENT_MESSAGE_TOOL_CALL_APPROVE: self.on_tool_call_approve,
            AGENT_MESSAGE_TOOL_CALL_REJECT: self.on_tool_call_reject,
            AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE: self.on_client_tool_call_response,
            AGENT_MESSAGE_SECRET_RESPONSE: self.on_secret_response,
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
        self._pending_secret_requests: dict[
            str, asyncio.Future[AgentSecretResponse]
        ] = {}
        self._pending_client_tool_calls: dict[str, _PendingClientToolCall] = {}
        self._active_turn_sender: Participant | None = None
        self._active_turn_client_tool_owner_id: str | None = None
        self._active_turn_toolkits: list[Toolkit] | None = None
        self._usage_emitted_turn_ids: set[str] = set()
        self._pending_status_messages: list[_QueuedTurnMessage] = []
        self._pending_realtime_audio_chunks: list[_PendingRealtimeAudioChunk] = []
        self._pending_realtime_audio_chunks_by_turn_id: dict[
            str, list[_PendingRealtimeAudioChunk]
        ] = {}
        self._pending_runtime_configuration: (
            tuple[
                list[LLMProvider],
                list[Toolkit],
            ]
            | None
        ) = None
        self._pending_realtime_audio_events_by_turn_id: dict[
            str, list[dict[str, Any]]
        ] = {}
        self._pending_realtime_audio_status_active = False
        self._interrupt_requested_turn_id: str | None = None
        self._interrupt_source_message_id: str | None = None
        self._active_turn_toolkit_client_options: dict[str, dict[str, Any]] = {}
        self._active_turn_tool_choice: ToolChoice | None = None
        self._client_tool_call_timeout_seconds = 300.0
        self._status_tool_call_accumulator = ToolCallAccumulator()
        self._status_text_by_item_id: dict[str, str] = {}
        self._status_text_phase_by_item_id: dict[
            str, Literal["commentary", "final_answer"]
        ] = {}
        self._latest_status_text: str | None = None
        self._session_initializer = session_initializer
        self._turn_instructions_provider = turn_instructions_provider
        self._turn_toolkits_builder = turn_toolkits_builder
        for provider in providers:
            provider.adapter.set_tool_call_approval_handler(
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

    def configure_runtime(
        self,
        *,
        llm_providers: list[LLMProvider],
        toolkits: list[Toolkit],
    ) -> None:
        if self._turn_task is not None or self._turn_id is not None:
            self._pending_runtime_configuration = (llm_providers, toolkits)
            return

        self._apply_runtime_configuration(
            llm_providers=llm_providers,
            toolkits=toolkits,
        )

    def _apply_runtime_configuration(
        self,
        *,
        llm_providers: list[LLMProvider],
        toolkits: list[Toolkit],
    ) -> None:
        providers = self._normalize_llm_providers(
            llm_adapter=None,
            llm_providers=llm_providers,
            default_provider=None,
        )
        current_provider_name = self._current_provider.name
        self._llm_providers_by_name = {
            provider.name: provider for provider in providers
        }
        self._default_provider = providers[0]
        self._current_provider = self._llm_providers_by_name.get(
            current_provider_name, self._default_provider
        )
        self.llm_adapter = self._current_provider.adapter
        self._current_model = self.llm_adapter.default_model()
        current_model_info = self._current_model_info()
        self._current_voice = (
            current_model_info.default_output_voice
            if current_model_info is not None
            else None
        )
        self._toolkits = list(toolkits)

    def _apply_pending_runtime_configuration(self) -> None:
        pending = self._pending_runtime_configuration
        if pending is None:
            return
        self._pending_runtime_configuration = None
        llm_providers, toolkits = pending
        self._apply_runtime_configuration(
            llm_providers=llm_providers,
            toolkits=toolkits,
        )

    @staticmethod
    def _provider_name_for_adapter(adapter: LLMAdapter) -> str:
        provider_name = adapter.provider_name()
        if provider_name is not None and provider_name.strip() != "":
            return provider_name
        return "default"

    @classmethod
    def _normalize_llm_providers(
        cls,
        *,
        llm_adapter: LLMAdapter | None,
        llm_providers: list[LLMProvider] | None,
        default_provider: LLMProvider | None,
    ) -> list[LLMProvider]:
        providers: list[LLMProvider] = []
        if llm_providers is not None:
            providers.extend(llm_providers)
        if llm_adapter is not None:
            if len(providers) > 0:
                raise ValueError("llm_adapter and llm_providers cannot both be set")
            providers.append(
                LLMProvider(
                    name=cls._provider_name_for_adapter(llm_adapter),
                    adapter=llm_adapter,
                )
            )
        if default_provider is not None:
            matching_provider = next(
                (
                    provider
                    for provider in providers
                    if provider.name == default_provider.name
                ),
                None,
            )
            if matching_provider is None:
                providers.append(default_provider)
            elif matching_provider.adapter is not default_provider.adapter:
                raise ValueError(
                    "default_provider must match the provider with the same name"
                )
        if len(providers) == 0:
            raise ValueError("at least one LLM provider is required")

        seen: set[str] = set()
        for provider in providers:
            provider_name = provider.name.strip()
            if provider_name == "":
                raise ValueError("LLM provider name cannot be empty")
            if provider_name != provider.name:
                raise ValueError(
                    "LLM provider name cannot have leading or trailing whitespace"
                )
            if provider_name in seen:
                raise ValueError(f"duplicate LLM provider name: {provider_name}")
            seen.add(provider_name)
        return providers

    def _resolve_llm_provider(self, provider_name: str | None) -> LLMProvider:
        if provider_name is None or provider_name.strip() == "":
            return self._current_provider
        provider = self._llm_providers_by_name.get(provider_name)
        if provider is None:
            names = ", ".join(sorted(self._llm_providers_by_name))
            raise ValueError(
                f"unknown LLM provider {provider_name!r}; available providers: {names}"
            )
        return provider

    @staticmethod
    def _provider_models(provider: LLMProvider) -> list[LLMModelInfo]:
        return provider.adapter.list_models()

    @staticmethod
    def _adapter_uses_default_model_list(adapter: LLMAdapter) -> bool:
        if not isinstance(adapter, LLMAdapter):
            return True
        return type(adapter).list_models is LLMAdapter.list_models

    def _resolve_llm_model(self, *, provider: LLMProvider, model: str | None) -> str:
        if model is None or model.strip() == "":
            if provider.name == self._current_provider.name:
                return self._current_model
            return provider.adapter.default_model()

        if self._adapter_uses_default_model_list(provider.adapter):
            return model

        models = self._provider_models(provider)
        for model_info in models:
            if model_info.name == model:
                return model

        names = ", ".join(model_info.name for model_info in models)
        raise ValueError(
            f"unknown model {model!r} for provider {provider.name!r}; "
            f"available models: {names}"
        )

    def _resolve_llm_model_info(
        self,
        *,
        provider: LLMProvider,
        model: str | None,
    ) -> LLMModelInfo:
        resolved_model = self._resolve_llm_model(provider=provider, model=model)
        for model_info in self._provider_models(provider):
            if model_info.name == resolved_model:
                return model_info
        raise ValueError(
            f"unknown model {resolved_model!r} for provider {provider.name!r}"
        )

    def _agent_model_info(
        self,
        *,
        provider: LLMProvider,
        model_info: LLMModelInfo,
    ) -> AgentModelInfo:
        return agent_model_info(
            provider=provider,
            model_info=model_info,
            current_provider=self._current_provider.name,
            current_model=self._current_model,
        )

    def _agent_provider_info(self, provider: LLMProvider) -> AgentProviderInfo:
        return agent_provider_info(
            provider=provider,
            current_provider=self._current_provider.name,
            current_model=self._current_model,
            backend=self._backend_name,
        )

    def _current_model_info(self) -> LLMModelInfo | None:
        if self._adapter_uses_default_model_list(self._current_provider.adapter):
            return LLMModelInfo(name=self._current_model)
        for model_info in self._provider_models(self._current_provider):
            if model_info.name == self._current_model:
                return model_info
        return None

    def _current_model_modalities(self) -> tuple[Literal["text", "audio"], ...]:
        model_info = self._current_model_info()
        if model_info is None:
            return ("text",)
        return model_info.modalities

    @staticmethod
    def _supported_output_modalities(
        *,
        output_modalities: tuple[Literal["text", "audio"], ...],
        supported_modalities: tuple[Literal["text", "audio"], ...],
    ) -> tuple[Literal["text", "audio"], ...]:
        supported = set(supported_modalities)
        selected: list[Literal["text", "audio"]] = []
        for modality in output_modalities:
            if modality not in supported or modality in selected:
                continue
            selected.append(modality)
            break
        if len(selected) == 0:
            return ("text",)
        return tuple(selected)

    def _current_model_changed_output_modalities(
        self,
    ) -> tuple[Literal["text", "audio"], ...]:
        supported = self._current_model_modalities()
        output_modalities = tuple(
            modality
            for modality in self._current_output_modalities
            if modality in supported
        )
        if len(output_modalities) > 0:
            return (output_modalities[0],)
        return ("text",)

    def _build_model_changed(
        self,
        *,
        thread_id: str,
        source_message_id: str | None,
    ) -> AgentModelChanged:
        model_info = self._current_model_info()
        return AgentModelChanged(
            type=AGENT_EVENT_MODEL_CHANGED,
            thread_id=thread_id,
            source_message_id=source_message_id,
            provider=self._current_provider.name,
            backend=self._backend_name,
            model=self._current_model,
            voice=self._current_voice,
            input_format=(
                _agent_audio_format_from_llm_audio_format(model_info.input_format)
                if model_info is not None
                else None
            ),
            output_format=(
                _agent_audio_format_from_llm_audio_format(model_info.output_format)
                if model_info is not None
                else None
            ),
            turn_detection=model_info.turn_detection
            if model_info is not None
            else None,
            output_modalities=list(self._current_model_changed_output_modalities()),
            realtime_protocols=(
                list(model_info.realtime_protocols) if model_info is not None else []
            ),
            supports_attachments=(
                model_info.supports_attachments if model_info is not None else False
            ),
            accepts=(list(model_info.accepts) if model_info is not None else []),
        )

    def _restore_current_model_from_thread_storage(self) -> bool:
        thread_storage = self.thread_storage
        if thread_storage is None:
            return False

        restored_provider: LLMProvider | None = None
        restored_model: str | None = None
        restored_voice: str | None = None
        restored_output_modalities: tuple[Literal["text", "audio"], ...] = ("text",)
        for stored_message in thread_storage.agent_messages():
            if not isinstance(stored_message, AgentModelChanged):
                continue
            try:
                provider = self._resolve_llm_provider(stored_message.provider)
                model = self._resolve_llm_model(
                    provider=provider,
                    model=stored_message.model,
                )
            except ValueError:
                logger.debug(
                    "ignoring unavailable persisted model selection",
                    extra={
                        "provider": stored_message.provider,
                        "model": stored_message.model,
                    },
                )
                continue
            restored_provider = provider
            restored_model = model
            restored_voice = stored_message.voice
            model_info = next(
                (
                    candidate
                    for candidate in self._provider_models(provider)
                    if candidate.name == model
                ),
                None,
            )
            restored_output_modalities = (
                self._supported_output_modalities(
                    output_modalities=tuple(stored_message.output_modalities),
                    supported_modalities=model_info.modalities,
                )
                if model_info is not None
                else ("text",)
            )

        if restored_provider is None or restored_model is None:
            return False
        if (
            restored_provider.name == self._current_provider.name
            and restored_model == self._current_model
            and restored_voice == self._current_voice
            and restored_output_modalities == self._current_output_modalities
        ):
            return False

        self._current_provider = restored_provider
        self.llm_adapter = restored_provider.adapter
        self._current_model = restored_model
        self._current_voice = restored_voice
        self._current_output_modalities = restored_output_modalities
        self._session_context = None
        return True

    async def _initialize_session_context(
        self, *, turn_id: str | None
    ) -> AgentSessionContext:
        restore_turn_id = turn_id or ""
        self._session_context = self.llm_adapter.create_session(
            usage_callback=self._on_session_usage_updated
        )

        with tracer.start_as_current_span("agent.turn.context.initialize"):
            await self.on_session_context_created()

        thread_storage = self.thread_storage
        if thread_storage is not None:
            await thread_storage.restore_session_context_async(
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

    async def _switch_llm_provider_if_needed(
        self,
        *,
        provider: LLMProvider,
        model: str,
        turn_id: str | None,
    ) -> bool:
        provider_changed = provider.name != self._current_provider.name
        model_changed = model != self._current_model
        if not provider_changed and not model_changed:
            return False

        previous_context = self._session_context
        previous_provider = self._current_provider
        self._current_provider = provider
        self.llm_adapter = provider.adapter
        self._current_model = model
        model_info = self._current_model_info()
        self._current_voice = (
            model_info.default_output_voice if model_info is not None else None
        )
        self._current_output_modalities = self._supported_output_modalities(
            output_modalities=self._current_output_modalities,
            supported_modalities=self._current_model_modalities(),
        )
        if previous_context is None:
            return True

        if not provider_changed:
            return True

        await previous_provider.adapter.stop_session(context=previous_context)
        await previous_context.close()
        self._session_context = None
        await self._initialize_session_context(turn_id=turn_id)
        return True

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
        if (
            thread_storage is not None
            and isinstance(payload, AgentThreadMessage)
            and payload.thread_id == self._thread_id
        ):
            thread_storage.push_message(message=payload, sender=sender)

        super().emit(sender=sender, payload=payload)

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        if message_type not in self._handlers:
            return False

        if message_type == AGENT_MESSAGE_MODELS_REQUEST:
            return True

        if not isinstance(message.data, AgentThreadMessage):
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
                await self._initialize_session_context(turn_id=restore_turn_id)

        return self._session_context

    @staticmethod
    def _usage_context_window_total(value: float) -> int | None:
        if not math.isfinite(value):
            return None
        return max(0, int(value))

    def _cached_usage_update(
        self,
        *,
        turn_id: str | None,
    ) -> AgentUsageUpdated | None:
        last_usage_update = self._last_usage_update
        if last_usage_update is None:
            return None
        return last_usage_update.model_copy(update={"turn_id": turn_id})

    def _on_session_usage_updated(self, usage: SessionUsage) -> None:
        turn_id = self._turn_id
        sender = self._active_turn_sender
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self._emit_session_usage_update(
                usage=usage,
                turn_id=turn_id,
                sender=sender,
            )
        )

        def done_callback(done_task: asyncio.Task[None]) -> None:
            try:
                done_task.result()
            except Exception:
                logger.debug("failed to publish session usage update", exc_info=True)

        task.add_done_callback(done_callback)
        if turn_id is not None:
            self._usage_emitted_turn_ids.add(turn_id)

    async def _emit_session_usage_update(
        self,
        *,
        usage: SessionUsage,
        turn_id: str | None,
        sender: Participant | None,
    ) -> None:
        thread_id = self.thread_id
        if thread_id is None:
            return
        try:
            context_management_mode = self.llm_adapter.context_management_mode()
        except Exception:
            logger.debug("failed to read context management mode", exc_info=True)
            context_management_mode = None
        try:
            configured_compaction_threshold = self.llm_adapter.compaction_threshold(
                usage.model
            )
        except Exception:
            logger.debug("failed to read compaction threshold", exc_info=True)
            configured_compaction_threshold = None

        usage_update = AgentUsageUpdated(
            type=AGENT_EVENT_USAGE_UPDATED,
            thread_id=thread_id,
            turn_id=turn_id,
            usage=usage.usage,
            context_window=AgentContextWindowUsage(
                used_tokens=usage.context_window_used or 0,
                total_tokens=usage.context_window_size,
                compaction_mode=context_management_mode,
                compaction_threshold=configured_compaction_threshold,
            ),
        )
        self._last_usage_update = usage_update
        self.emit(sender=sender, payload=usage_update)

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
                await thread_storage.restore_session_context_async(
                    context=context,
                    llm_adapter=self.llm_adapter,
                )
                restored_context_from_storage = True

        session_usage = context.last_usage
        usage = {}
        if include_usage and session_usage is not None:
            usage = session_usage.usage

        if session_usage is not None and session_usage.context_window_used is not None:
            used_tokens = session_usage.context_window_used
        else:
            used_tokens = 0
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
        if session_usage is not None and session_usage.context_window_size is not None:
            context_window_size = float(session_usage.context_window_size)
        else:
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

    def _cancel_pending_secret_requests(self) -> None:
        pending_requests = list(self._pending_secret_requests.values())
        self._pending_secret_requests.clear()
        for future in pending_requests:
            if not future.done():
                future.cancel()

    async def _send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        supervisor = self.supervisor
        if supervisor is None:
            return False

        if not supervisor.is_participant_connected(participant_id=participant.id):
            return False

        if supervisor.state == "started":
            supervisor.send(
                Message(
                    data=payload,
                    sender=participant,
                    source=self,
                    to_participant_id=participant.id,
                )
            )
            return True

        sent = False
        for channel in supervisor.channels:
            if await channel.send_agent_message_to_participant_and_wait(
                participant=participant,
                payload=payload,
            ):
                sent = True
        return sent

    async def _cancel_pending_client_tool_calls(
        self,
        *,
        participant_id: str | None = None,
        reason: str,
    ) -> None:
        pending_items = list(self._pending_client_tool_calls.items())
        for request_id, pending in pending_items:
            if participant_id is not None and pending.participant.id != participant_id:
                continue
            self._pending_client_tool_calls.pop(request_id, None)
            if not pending.future.done():
                pending.future.set_result(
                    ErrorContent(
                        text=f"client toolkit call cancelled: {reason}",
                    )
                )
            with contextlib.suppress(Exception):
                await self._send_agent_message_to_participant(
                    participant=pending.participant,
                    payload=AgentClientToolCallCancelled(
                        type=AGENT_EVENT_CLIENT_TOOL_CALL_CANCELLED,
                        thread_id=pending.thread_id,
                        turn_id=pending.turn_id,
                        request_id=pending.request_id,
                        toolkit=pending.toolkit,
                        tool=pending.tool,
                        reason=reason,
                    ),
                )

    async def on_participant_disconnected(self, *, participant_id: str) -> None:
        await self._cancel_pending_client_tool_calls(
            participant_id=participant_id,
            reason="participant_disconnected",
        )

    async def _request_client_tool_call(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
    ) -> Content:
        turn_id = self._turn_id
        thread_id = self.thread_id
        participant = self._active_turn_sender
        supervisor = self.supervisor
        if turn_id is None or thread_id is None:
            return ErrorContent(
                text="client toolkit call requested without an active turn",
            )
        if participant is None:
            return ErrorContent(
                text="client toolkit call requested without a participant",
            )
        if supervisor is None or not supervisor.is_participant_connected(
            participant_id=participant.id
        ):
            return ErrorContent(
                text="client toolkit participant is not connected",
            )

        request_id = str(uuid.uuid4())
        request_future: asyncio.Future[Content] = (
            asyncio.get_running_loop().create_future()
        )
        pending = _PendingClientToolCall(
            future=request_future,
            participant=participant,
            thread_id=thread_id,
            turn_id=turn_id,
            request_id=request_id,
            toolkit="client",
            tool=tool,
        )
        self._pending_client_tool_calls[request_id] = pending
        sent = await self._send_agent_message_to_participant(
            participant=participant,
            payload=AgentClientToolCallRequested(
                type=AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED,
                thread_id=thread_id,
                turn_id=turn_id,
                request_id=request_id,
                toolkit=pending.toolkit,
                tool=tool,
                arguments=arguments,
            ),
        )
        if not sent:
            self._pending_client_tool_calls.pop(request_id, None)
            return ErrorContent(
                text="client toolkit participant is not connected",
            )

        try:
            return await asyncio.wait_for(
                request_future,
                timeout=self._client_tool_call_timeout_seconds,
            )
        except TimeoutError:
            self._pending_client_tool_calls.pop(request_id, None)
            await self._send_agent_message_to_participant(
                participant=participant,
                payload=AgentClientToolCallCancelled(
                    type=AGENT_EVENT_CLIENT_TOOL_CALL_CANCELLED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    toolkit=pending.toolkit,
                    tool=tool,
                    reason="timeout",
                ),
            )
            return ErrorContent(
                text="client toolkit call timed out",
            )
        finally:
            existing_pending = self._pending_client_tool_calls.get(request_id)
            if existing_pending is pending:
                del self._pending_client_tool_calls[request_id]

    async def request_user_secret(
        self,
        *,
        secret_name: str,
        prompt: str | None = None,
        oauth: dict[str, Any] | None = None,
        challenge: str | None = None,
    ) -> AgentSecretResponse:
        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            raise RuntimeError("user secret requested without an active turn")

        request_id = str(uuid.uuid4())
        request_future: asyncio.Future[AgentSecretResponse] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_secret_requests[request_id] = request_future
        self.emit(
            sender=self._active_turn_sender,
            payload=AgentSecretRequested(
                type=AGENT_EVENT_SECRET_REQUESTED,
                thread_id=thread_id,
                turn_id=turn_id,
                request_id=request_id,
                secret_name=secret_name,
                prompt=prompt,
                oauth=oauth,
                challenge=challenge,
            ),
        )

        try:
            return await request_future
        finally:
            existing_future = self._pending_secret_requests.get(request_id)
            if existing_future is request_future:
                del self._pending_secret_requests[request_id]

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

    async def _resolve_secret_request(self, response: AgentSecretResponse) -> None:
        request_future = self._pending_secret_requests.get(response.request_id)
        if request_future is None or request_future.done():
            return

        request_future.set_result(response)

    async def _resolve_client_tool_call_response(
        self,
        response: AgentClientToolCallResponse,
        *,
        sender: Participant | None,
    ) -> None:
        pending = self._pending_client_tool_calls.get(response.request_id)
        if pending is None or pending.future.done():
            return
        if response.turn_id != pending.turn_id:
            return
        if sender is None or sender.id != pending.participant.id:
            return

        pending.future.set_result(response.response)

    @staticmethod
    def _resolve_turn_client_toolkits(
        *,
        turns: list[TurnStart | TurnSteer],
    ) -> list[ClientToolkitDescription]:
        configured_toolkits: list[ClientToolkitDescription] | None = None
        for turn in turns:
            if not isinstance(turn, TurnStart):
                continue
            configured_toolkits = turn.client_toolkits
        return list(configured_toolkits or [])

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
            if turn.mcp is not None:
                configured_options["mcp"] = {
                    "servers": [*turn.mcp.servers],
                }
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
    def _resolve_turn_response_options(
        *,
        turns: list[TurnStart | TurnSteer],
    ) -> dict[str, Any] | None:
        output_modalities: list[Literal["text", "audio"]] | None = None
        for turn in turns:
            if isinstance(turn, TurnStart) and turn.output_modalities is not None:
                output_modalities = turn.output_modalities

        if output_modalities is None:
            return None
        return {"output_modalities": output_modalities}

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
                turn_id = self._turn_id
                if turn_id is None and len(turns) > 0:
                    turn_id = turns[-1].turn_id
                combined_toolkits.extend(
                    supervisor.get_turn_toolkits(
                        thread_id=self.thread_id,
                        turn_id=turn_id,
                        thread_storage=self.thread_storage,
                    )
                )

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

            client_toolkits = self._resolve_turn_client_toolkits(turns=turns)
            if len(client_toolkits) > 0:
                resolved_toolkits.append(
                    Toolkit(
                        name="client",
                        tools=[
                            _AgentMessageProxyClientTool(
                                description=description,
                                request_tool_call=lambda tool, arguments: (
                                    self._request_client_tool_call(
                                        tool=tool,
                                        arguments=arguments,
                                    )
                                ),
                            )
                            for description in client_toolkits
                        ],
                        title="Client",
                        description="Client-side tools provided by the participant.",
                        hidden=True,
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
        filename: str | None = None,
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
            session.append_file_url(url=url, filename=filename)
            return

        session.append_user_message(
            self._file_attachment_message(sender=sender, url=url)
        )

    async def _append_file_content(
        self,
        *,
        session: AgentSessionContext,
        url: str,
        filename: str | None = None,
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
            filename=filename,
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
                        filename=item.name,
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
    ) -> tuple[Participant | None, list[Toolkit], ToolChoice | None, str | None]:
        turns = [queued_message.request for queued_message in queued_messages]
        sender = self._sender_for_turn_batch(queued_messages=queued_messages)
        toolkit_client_options = self._resolve_turn_toolkit_client_options(turns=turns)
        tool_choice = self._resolve_turn_tool_choice(turns=turns)
        client_tool_owner_id: str | None = None
        if len(self._resolve_turn_client_toolkits(turns=turns)) > 0:
            client_tool_owner_id = None if sender is None else sender.id

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
        return sender, combined_toolkits, tool_choice, client_tool_owner_id

    async def _run_adapter_next(
        self,
        *,
        session: AgentSessionContext,
        sender: Participant | None,
        combined_toolkits: list[Toolkit],
        tool_choice: ToolChoice | None,
        client_tool_owner_id: str | None,
        model: str,
        response_options: dict[str, Any] | None = None,
    ) -> None:
        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            raise RuntimeError("turn publisher requested without an active turn")

        llm_provider = self._current_provider.name
        first_event_span: Any | None = None
        first_text_delta_span: Any | None = None

        def enrich_llm_message(message: AgentMessage) -> AgentMessage:
            participant_name = self._sender_name(sender)
            if not isinstance(message, AgentLLMMessage):
                return self._agent_message_with_participant_name(message)

            updates: dict[str, str] = {}
            if message.provider is None and llm_provider is not None:
                updates["provider"] = llm_provider
            if message.model is None:
                updates["model"] = model
            if (
                isinstance(
                    message,
                    (
                        AgentAudioTranscriptionCompleted,
                        AgentAudioTranscriptionDelta,
                        AgentAudioTranscriptionFailed,
                        AgentAudioTranscriptionStarted,
                    ),
                )
                and message.role == "user"
                and message.sender_name is None
                and participant_name is not None
            ):
                updates["sender_name"] = participant_name
            if len(updates) == 0:
                return self._agent_message_with_participant_name(message)
            return self._agent_message_with_participant_name(
                message.model_copy(update=updates)
            )

        def thread_status_from_agent_message(
            message: AgentMessage,
        ) -> tuple[str, str | None, int | None, int | None, int | None] | None:
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
                        self._status_text_phase_by_item_id.pop(status_item_id, None)
                        self._status_tool_call_accumulator.remove(status_item_id)
                    self._latest_status_text = None
                    return "Thinking", None, None, None, None
                if status_text is None:
                    return None
                self._latest_status_text = status_text
                if status_item_id.strip() != "":
                    self._status_text_by_item_id[status_item_id] = status_text
                return (
                    status_text,
                    status_item_id if status_item_id.strip() != "" else None,
                    _status_total_bytes(
                        self._status_tool_call_accumulator.total_bytes(status_item_id)
                    ),
                    None,
                    None,
                )

            if isinstance(message, AgentTextContentStarted):
                if message.phase is not None:
                    self._status_text_phase_by_item_id[message.item_id] = message.phase
                if message.phase == "final_answer":
                    return "Writing", message.item_id, None, None, None
                if message.phase == "commentary":
                    return "Planning", message.item_id, None, None, None
                return None

            if isinstance(message, AgentTextContentDelta):
                phase = message.phase
                if phase is None:
                    phase = self._status_text_phase_by_item_id.get(message.item_id)
                elif phase is not None:
                    self._status_text_phase_by_item_id[message.item_id] = phase
                if phase == "final_answer":
                    return "Writing", message.item_id, None, None, None
                if phase == "commentary":
                    return "Planning", message.item_id, None, None, None
                return None

            if isinstance(message, AgentTextContentEnded):
                phase = message.phase
                if phase is None:
                    phase = self._status_text_phase_by_item_id.pop(
                        message.item_id, None
                    )
                else:
                    self._status_text_phase_by_item_id.pop(message.item_id, None)
                if phase == "final_answer":
                    return "Writing", None, None, None, None
                return None

            if isinstance(
                message,
                (AgentToolCallPending, AgentToolCallInProgress, AgentToolCallStarted),
            ):
                state = (
                    "pending"
                    if isinstance(message, AgentToolCallPending)
                    else "in_progress"
                )
                snapshot = self._status_tool_call_accumulator.upsert_lifecycle(
                    item_id=message.item_id,
                    toolkit=message.toolkit,
                    tool=message.tool,
                    arguments=message.arguments,
                    state=state,
                    argument_bytes=message.argument_bytes,
                )
                return (
                    snapshot.text,
                    snapshot.item_id,
                    snapshot.total_bytes,
                    snapshot.lines_added,
                    snapshot.lines_removed,
                )

            if isinstance(message, AgentToolCallArgumentsDelta):
                snapshot = self._status_tool_call_accumulator.append_delta(
                    item_id=message.item_id,
                    delta=message.delta,
                )
                if snapshot is None:
                    status_text = self._status_text_by_item_id.get(message.item_id)
                    if status_text is None:
                        status_text = self._latest_status_text
                    if status_text is None:
                        return None
                    self._status_text_by_item_id[message.item_id] = status_text
                    return (
                        status_text,
                        message.item_id,
                        _status_total_bytes(
                            self._status_tool_call_accumulator.total_bytes(
                                message.item_id
                            )
                        ),
                        None,
                        None,
                    )
                return (
                    snapshot.text,
                    snapshot.item_id,
                    snapshot.total_bytes,
                    snapshot.lines_added,
                    snapshot.lines_removed,
                )

            if isinstance(message, AgentToolCallApprovalRequested):
                return "Waiting for approval", message.item_id, None, None, None

            if isinstance(message, AgentToolCallEnded):
                tool_call = self._status_tool_call_accumulator.get(message.item_id)
                event_status_text = self._status_text_by_item_id.pop(
                    message.item_id, None
                )
                self._status_text_phase_by_item_id.pop(message.item_id, None)
                if (
                    event_status_text is not None
                    and self._latest_status_text == event_status_text
                ):
                    self._latest_status_text = None
                if tool_call is None:
                    self._status_tool_call_accumulator.remove(message.item_id)
                    return "Thinking", None, None, None, None
                self._status_tool_call_accumulator.remove(message.item_id)
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
                    return "Thinking", None, None, None, None
                return status, message.item_id, None, None, None

            if isinstance(
                message,
                (AgentImageGenerationStarted, AgentImageGenerationPartial),
            ):
                return "Generating image", message.item_id, None, None, None

            if isinstance(message, AgentImageGenerationCompleted):
                return "Thinking", None, None, None, None

            if isinstance(message, AgentImageGenerationFailed):
                return "Attempted to generate image", message.item_id, None, None, None

            if isinstance(message, TurnInterrupted):
                return "Turn interrupted", None, None, None, None

            return None

        def publish_event(message: AgentMessage) -> None:
            nonlocal first_event_span
            nonlocal first_text_delta_span
            if self._interrupt_requested_turn_id == turn_id:
                return
            message = enrich_llm_message(message)
            if first_event_span is not None:
                first_event_span.set_attribute("message_type", message.type)
                first_event_span.end()
                first_event_span = None
            if (
                first_text_delta_span is not None
                and isinstance(message, AgentTextContentDelta)
                and message.text != ""
            ):
                first_text_delta_span.set_attribute("message_type", message.type)
                first_text_delta_span.end()
                first_text_delta_span = None
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
            status_text, pending_item_id, total_bytes, lines_added, lines_removed = (
                status
            )
            previous_publish = thread_status_publish_tail

            async def publish_status_in_order() -> None:
                if previous_publish is not None:
                    with contextlib.suppress(Exception):
                        await previous_publish
                await thread_status_publisher.set_thread_status(
                    status=status_text,
                    pending_item_id=pending_item_id,
                    total_bytes=total_bytes,
                    lines_added=lines_added,
                    lines_removed=lines_removed,
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
        self._active_turn_toolkits = combined_toolkits
        self._active_turn_client_tool_owner_id = client_tool_owner_id
        had_thread_id = "thread_id" in session.metadata
        previous_thread_id = session.metadata.get("thread_id")
        had_turn_id = "turn_id" in session.metadata
        previous_turn_id = session.metadata.get("turn_id")
        had_voice = "voice" in session.metadata
        previous_voice = session.metadata.get("voice")
        session.metadata["thread_id"] = thread_id
        session.metadata["turn_id"] = turn_id
        if self._current_voice is not None:
            session.metadata["voice"] = self._current_voice
        else:
            session.metadata.pop("voice", None)
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
                with tracer.start_as_current_span("agent.turn.llm.start_session"):
                    await self.llm_adapter.start_session(
                        context=session,
                        event_handler=handle_event,
                    )
                pending_audio_chunks = (
                    self._pending_realtime_audio_chunks_by_turn_id.pop(turn_id, None)
                )
                if pending_audio_chunks is not None:
                    for audio_chunk in pending_audio_chunks:
                        await session.append_realtime_audio_chunk(
                            mime_type=audio_chunk.format.type,
                            data=audio_chunk.data,
                            sample_rate=audio_chunk.format.sample_rate,
                            bitrate=audio_chunk.format.bitrate,
                        )
                    await session.commit_realtime_audio()
                    self._pending_realtime_audio_status_active = False
                for event in self._pending_realtime_audio_events_by_turn_id.pop(
                    turn_id, []
                ):
                    handle_event(event)
                first_event_span = tracer.start_span("agent.turn.llm.first_event")
                first_event_span.set_attribute("thread_id", thread_id)
                first_event_span.set_attribute("turn_id", turn_id)
                first_event_span.set_attribute("model", model)
                first_text_delta_span = tracer.start_span(
                    "agent.turn.llm.first_text_delta"
                )
                first_text_delta_span.set_attribute("thread_id", thread_id)
                first_text_delta_span.set_attribute("turn_id", turn_id)
                first_text_delta_span.set_attribute("model", model)
                with tracer.start_as_current_span("agent.turn.llm.create_response"):
                    next_task = asyncio.create_task(
                        self.llm_adapter.create_response(
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
                            options=response_options,
                        )
                    )
                    self._active_next_task = next_task
                    await next_task
                completed = self._interrupt_requested_turn_id != turn_id
        finally:
            if first_event_span is not None:
                first_event_span.end()
                first_event_span = None
            if first_text_delta_span is not None:
                first_text_delta_span.end()
                first_text_delta_span = None
            usage_already_emitted = turn_id in self._usage_emitted_turn_ids
            if not (completed and usage_already_emitted):
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
            if had_voice:
                session.metadata["voice"] = previous_voice
            else:
                session.metadata.pop("voice", None)
            self._active_next_task = None
            self._active_turn_sender = None
            self._active_turn_toolkits = None
            self._active_turn_client_tool_owner_id = None
            self._usage_emitted_turn_ids.discard(turn_id)

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
        (
            sender,
            combined_toolkits,
            tool_choice,
            client_tool_owner_id,
        ) = await self._prepare_turn_batch(
            queued_messages=steer_messages,
            session=session,
            model=model,
        )
        await self._run_adapter_next(
            session=session,
            sender=sender,
            combined_toolkits=combined_toolkits,
            tool_choice=tool_choice,
            client_tool_owner_id=client_tool_owner_id,
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
        (
            sender,
            combined_toolkits,
            tool_choice,
            client_tool_owner_id,
        ) = await self._prepare_turn_batch(
            queued_messages=queued_messages,
            session=session,
            model=model,
        )
        response_options = self._resolve_turn_response_options(
            turns=[queued_message.request for queued_message in queued_messages],
        )
        await self._run_adapter_next(
            session=session,
            sender=sender,
            combined_toolkits=combined_toolkits,
            tool_choice=tool_choice,
            client_tool_owner_id=client_tool_owner_id,
            model=model,
            response_options=response_options,
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
        await self._cancel_pending_client_tool_calls(reason="turn_interrupted")

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
            with tracer.start_as_current_span("agent.turn.pending_status.remove"):
                await self._remove_pending_status_messages(
                    queued_messages=queued_turn_messages
                )
        except BaseException as exc:
            turn_span_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        self._turn_id = turn_id
        self._status_tool_call_accumulator.clear()
        self._status_text_by_item_id.clear()
        self._status_text_phase_by_item_id.clear()
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
        with tracer.start_as_current_span("agent.turn.started.emit"):
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
        model = self._current_model
        try:
            provider = self._resolve_llm_provider(queued_turn.request.provider)
            model = self._resolve_llm_model(
                provider=provider,
                model=queued_turn.request.model,
            )
            requested_output_modalities = queued_turn.request.output_modalities
            output_modalities_changed = (
                requested_output_modalities is not None
                and tuple(requested_output_modalities)
                != self._current_output_modalities
            )
            if requested_output_modalities is not None:
                self._current_output_modalities = tuple(requested_output_modalities)
            requested_voice = queued_turn.request.voice
            voice_changed = False
            if requested_voice is not None:
                normalized_voice = requested_voice.strip() or None
                voice_changed = normalized_voice != self._current_voice
                self._current_voice = normalized_voice
            changed_model = await self._switch_llm_provider_if_needed(
                provider=provider,
                model=model,
                turn_id=turn_id,
            )
            if changed_model or output_modalities_changed or voice_changed:
                self.emit(
                    sender=queued_turn.sender,
                    payload=self._build_model_changed(
                        thread_id=queued_turn.request.thread_id,
                        source_message_id=queued_turn.request.message_id,
                    ),
                )
            session = await self.ensure_session_context(turn_id=turn_id)
            turn_span.set_attribute("provider", provider.name)
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
                self._status_tool_call_accumulator.clear()
                self._status_text_by_item_id.clear()
                self._status_text_phase_by_item_id.clear()
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
        self._apply_pending_runtime_configuration()

        if not self._stop.is_set():
            self._schedule_next_turn()

    async def on_thread_open(self, message: Message) -> None:
        request = _coerce_message_data(message.data, OpenThread)
        thread_storage = self.thread_storage
        if thread_storage is not None:
            await thread_storage.wait_until_ready()
        self._restore_current_model_from_thread_storage()
        if request.load is True:
            if thread_storage is not None:
                for stored_message in self._stored_agent_messages_since_turn(
                    thread_storage=thread_storage,
                    since_turn=request.since_turn,
                ):
                    await self._send_thread_open_response(
                        request_message=message,
                        payload=self._agent_message_with_participant_name(
                            stored_message
                        ),
                    )
            await self._send_thread_open_response(
                request_message=message,
                payload=self._agent_message_with_participant_name(
                    ThreadLoaded(
                        type=AGENT_EVENT_THREAD_LOADED,
                        thread_id=request.thread_id,
                        source_message_id=request.message_id,
                        since_turn=request.since_turn,
                    )
                ),
            )
        super().emit(
            sender=message.sender,
            payload=self._agent_message_with_participant_name(
                self._build_model_changed(
                    thread_id=request.thread_id,
                    source_message_id=request.message_id,
                )
            ),
        )
        if self._turn_task is not None or self._turn_id is not None:
            return
        session = await self.ensure_session_context(turn_id=None)
        await self.llm_adapter.start_session(context=session)
        if self._emit_cached_usage_update(
            turn_id=self._turn_id,
            sender=message.sender,
            source=message.source,
        ):
            return
        try:
            usage_update = await self._build_usage_update(
                session=session,
                model=self._current_model,
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

    async def _send_thread_open_response(
        self,
        *,
        request_message: Message,
        payload: AgentMessage,
    ) -> None:
        source = request_message.source
        sender = request_message.sender
        if source is not None and sender is not None and isinstance(source, Channel):
            sent = await source.send_agent_message_to_participant_and_wait(
                participant=sender,
                payload=payload,
            )
            if sent:
                return

        super().emit(sender=sender, payload=payload)

    @staticmethod
    def _stored_agent_messages_since_turn(
        *,
        thread_storage: ThreadStorage,
        since_turn: str | None,
    ) -> list[AgentThreadMessage]:
        messages = thread_storage.agent_messages()
        if since_turn is None or since_turn.strip() == "":
            return messages

        normalized_since_turn = since_turn.strip()
        for index, stored_message in enumerate(messages):
            if stored_message.thread_id != thread_storage.path:
                continue
            if stored_message.message_id == normalized_since_turn:
                return messages[index:]
            turn_id = _message_turn_id(stored_message)
            if turn_id == normalized_since_turn:
                return messages[index:]
            if isinstance(stored_message, (TurnStart, TurnSteer)):
                if stored_message.message_id == normalized_since_turn:
                    return messages[index:]
        return []

    async def on_thread_close(self, message: Message) -> None:
        _coerce_message_data(message.data, CloseThread)
        if self._session_context is not None:
            await self.llm_adapter.stop_session(context=self._session_context)
            await self._session_context.close()
            self._session_context = None

    async def on_turn_start(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnStart)
        try:
            provider = self._resolve_llm_provider(turn.provider)
        except ValueError as exc:
            self.emit(
                sender=message.sender,
                payload=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=turn.thread_id,
                    source_message_id=turn.message_id,
                    error=self._turn_error(
                        message=str(exc),
                        code="unknown_provider",
                    ),
                ),
            )
            return
        try:
            resolved_model = self._resolve_llm_model(
                provider=provider,
                model=turn.model,
            )
        except ValueError as exc:
            self.emit(
                sender=message.sender,
                payload=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=turn.thread_id,
                    source_message_id=turn.message_id,
                    error=self._turn_error(
                        message=str(exc),
                        code="unknown_model",
                    ),
                ),
            )
            return
        if turn.output_modalities is not None:
            model_info = self._resolve_llm_model_info(
                provider=provider,
                model=resolved_model,
            )
            unsupported_output_modalities = [
                modality
                for modality in turn.output_modalities
                if modality not in model_info.modalities
            ]
            if len(unsupported_output_modalities) > 0:
                unsupported = ", ".join(
                    repr(item) for item in unsupported_output_modalities
                )
                self.emit(
                    sender=message.sender,
                    payload=TurnStartRejected(
                        type=AGENT_EVENT_TURN_START_REJECTED,
                        thread_id=turn.thread_id,
                        source_message_id=turn.message_id,
                        error=self._turn_error(
                            message=(
                                f"model {model_info.name!r} does not support "
                                f"{unsupported} output modalities"
                            ),
                            code="unsupported_modality",
                        ),
                    ),
                )
                return

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
        with tracer.start_as_current_span("agent.turn.queue"):
            await self._pending_turns.put(
                _QueuedTurn(
                    sender=message.sender,
                    request=turn,
                )
            )
        with tracer.start_as_current_span("agent.turn.accepted.emit"):
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

    async def on_models_request(self, message: Message) -> None:
        request = _coerce_message_data(message.data, ModelsRequest)
        self.emit(
            sender=message.sender,
            payload=ModelsResponse(
                type=AGENT_MESSAGE_MODELS_RESPONSE,
                source_message_id=request.message_id,
                providers=[
                    self._agent_provider_info(provider)
                    for provider in self._llm_providers_by_name.values()
                ],
            ),
        )

    async def on_model_change(self, message: Message) -> None:
        request = _coerce_message_data(message.data, ChangeModel)
        try:
            provider = self._resolve_llm_provider(request.provider)
            model = self._resolve_llm_model(provider=provider, model=request.model)
        except ValueError as exc:
            self.emit(
                sender=message.sender,
                payload=AgentThreadEvent(
                    type=AGENT_EVENT_THREAD_EVENT,
                    thread_id=request.thread_id,
                    provider=self._current_provider.name,
                    model=self._current_model,
                    event={
                        "type": "model_change_rejected",
                        "error": {
                            "message": str(exc),
                            "code": "unknown_model",
                        },
                        "source_message_id": request.message_id,
                    },
                ),
            )
            return

        changed = await self._switch_llm_provider_if_needed(
            provider=provider,
            model=model,
            turn_id=self._turn_id,
        )
        del changed
        if request.voice is not None:
            requested_voice = request.voice.strip()
            if requested_voice == "":
                requested_voice = None
            model_info = self._current_model_info()
            if (
                requested_voice is not None
                and model_info is not None
                and len(model_info.available_voices) > 0
                and requested_voice not in model_info.available_voices
            ):
                self.emit(
                    sender=message.sender,
                    payload=AgentThreadEvent(
                        type=AGENT_EVENT_THREAD_EVENT,
                        thread_id=request.thread_id,
                        provider=self._current_provider.name,
                        model=self._current_model,
                        event={
                            "type": "model_change_rejected",
                            "error": {
                                "message": f"unknown voice {requested_voice!r}",
                                "code": "unknown_voice",
                            },
                            "source_message_id": request.message_id,
                        },
                    ),
                )
                return
            self._current_voice = requested_voice
        self.emit(
            sender=message.sender,
            payload=self._build_model_changed(
                thread_id=request.thread_id,
                source_message_id=request.message_id,
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

        client_tool_owner_id = self._active_turn_client_tool_owner_id
        if client_tool_owner_id is not None and (
            message.sender is None or message.sender.id != client_tool_owner_id
        ):
            rejection = self._turn_error(
                message=(
                    "turn is using client toolkits and can only be steered by "
                    "the participant that started it"
                ),
                code="turn_owned_by_participant",
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

    async def on_client_tool_call_response(self, message: Message) -> None:
        response = _coerce_message_data(message.data, AgentClientToolCallResponse)
        if self._turn_id != response.turn_id:
            return

        await self._resolve_client_tool_call_response(
            response,
            sender=message.sender,
        )

    async def on_realtime_audio_chunk(self, message: Message) -> None:
        chunk = _coerce_message_data(message.data, AgentRealtimeAudioChunk)
        if "audio" not in self._current_model_modalities():
            return

        model_info = self._current_model_info()
        automatic_turn_detection = (
            model_info is not None and model_info.turn_detection == "automatic"
        )
        expected_input_format = (
            _agent_audio_format_from_llm_audio_format(model_info.input_format)
            if model_info is not None
            else None
        )
        declared_input_format = chunk.format
        mismatch = _audio_format_mismatch(
            declared=declared_input_format,
            expected=expected_input_format,
        )
        if mismatch is not None:
            logger.warning("rejecting realtime audio chunk: %s", mismatch)
            return

        session = await self.ensure_session_context(turn_id=self._turn_id)
        if not session.supports_realtime_audio:
            return

        if automatic_turn_detection:
            automatic_turn_id = self._turn_id
            publisher: Callable[[dict[str, Any]], None] | None = None

            def ensure_automatic_publisher() -> None:
                nonlocal publisher
                if automatic_turn_id is None or publisher is not None:
                    return

                def publish_automatic_payload(payload: AgentMessage) -> None:
                    participant_name = self._sender_name(message.sender)
                    updates: dict[str, str] = {}
                    if isinstance(payload, AgentLLMMessage):
                        if payload.provider is None:
                            updates["provider"] = self._current_provider.name
                        if payload.model is None:
                            updates["model"] = self._current_model
                    if (
                        isinstance(
                            payload,
                            (
                                AgentAudioTranscriptionCompleted,
                                AgentAudioTranscriptionDelta,
                                AgentAudioTranscriptionFailed,
                                AgentAudioTranscriptionStarted,
                            ),
                        )
                        and payload.role == "user"
                        and payload.sender_name is None
                        and participant_name is not None
                    ):
                        updates["sender_name"] = participant_name
                    if len(updates) > 0:
                        payload = payload.model_copy(update=updates)
                    self.emit(sender=message.sender, payload=payload)

                publisher = self.llm_adapter.make_agent_event_publisher(
                    turn_id=automatic_turn_id,
                    thread_id=chunk.thread_id,
                    callback=publish_automatic_payload,
                )

            def start_automatic_turn() -> None:
                nonlocal automatic_turn_id, publisher
                automatic_turn_id = str(uuid.uuid4())
                self._turn_id = automatic_turn_id
                session.metadata["thread_id"] = chunk.thread_id
                session.metadata["turn_id"] = automatic_turn_id
                self.emit(
                    sender=message.sender,
                    payload=TurnStarted(
                        type=AGENT_EVENT_TURN_STARTED,
                        thread_id=chunk.thread_id,
                        turn_id=automatic_turn_id,
                        source_message_id=chunk.message_id,
                    ),
                )
                publisher = None
                ensure_automatic_publisher()

            def end_automatic_turn(*, error: AgentError | None = None) -> None:
                nonlocal automatic_turn_id, publisher
                if automatic_turn_id is None:
                    return
                self.emit(
                    sender=message.sender,
                    payload=TurnEnded(
                        type=AGENT_EVENT_TURN_ENDED,
                        thread_id=chunk.thread_id,
                        turn_id=automatic_turn_id,
                        error=error,
                    ),
                )
                if self._turn_id == automatic_turn_id:
                    self._turn_id = None
                automatic_turn_id = None
                publisher = None

            def handle_automatic_realtime_event(event: dict[str, Any]) -> None:
                nonlocal publisher
                event_type = event.get("type")
                if event_type == "input_audio_buffer.speech_started":
                    if automatic_turn_id is not None:
                        self.emit(
                            sender=message.sender,
                            payload=TurnInterrupted(
                                type=AGENT_EVENT_TURN_INTERRUPTED,
                                thread_id=chunk.thread_id,
                                turn_id=automatic_turn_id,
                                source_message_id=chunk.message_id,
                            ),
                        )
                        end_automatic_turn(
                            error=AgentError(
                                message="turn interrupted by input speech",
                                code="interrupted",
                            )
                        )
                    start_automatic_turn()
                elif automatic_turn_id is None:
                    start_automatic_turn()

                ensure_automatic_publisher()
                if publisher is not None:
                    publisher(event)
                if event_type in {
                    "response.done",
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    end_automatic_turn()

            combined_toolkits = await self._build_turn_toolkits(
                model=self._current_model,
                turns=[],
                sender=message.sender,
                toolkit_client_options={},
            )
            await self.llm_adapter.start_realtime_session(
                context=session,
                event_handler=handle_automatic_realtime_event,
                caller=message.sender or self._participant,
                toolkits=combined_toolkits,
                model=self._current_model,
            )

            thread_status_publisher = self.thread_status_publisher
            if chunk.data != b"":
                await session.append_realtime_audio_chunk(
                    mime_type=declared_input_format.type,
                    data=chunk.data,
                    sample_rate=declared_input_format.sample_rate,
                    bitrate=declared_input_format.bitrate,
                )
                if (
                    thread_status_publisher is not None
                    and not self._pending_realtime_audio_status_active
                ):
                    self._pending_realtime_audio_status_active = True
                    await thread_status_publisher.set_thread_status(
                        status="Listening",
                        mode="busy",
                    )
            return

        if chunk.data == b"":
            return

        self._pending_realtime_audio_chunks.append(
            _PendingRealtimeAudioChunk(data=chunk.data, format=declared_input_format)
        )
        thread_status_publisher = self.thread_status_publisher
        if (
            thread_status_publisher is not None
            and not self._pending_realtime_audio_status_active
        ):
            self._pending_realtime_audio_status_active = True
            await thread_status_publisher.set_thread_status(
                status="Listening",
                mode="busy",
            )

    async def on_realtime_audio_commit(self, message: Message) -> None:
        commit = _coerce_message_data(message.data, AgentRealtimeAudioCommit)
        if "audio" not in self._current_model_modalities():
            return

        session = await self.ensure_session_context(turn_id=self._turn_id)
        if not session.supports_realtime_audio:
            return
        turn_id = commit.turn_id or self._turn_id
        if turn_id is None:
            return

        model_info = self._current_model_info()
        automatic_turn_detection = (
            model_info is not None and model_info.turn_detection == "automatic"
        )
        if not automatic_turn_detection:
            self._pending_realtime_audio_chunks_by_turn_id[turn_id] = [
                *self._pending_realtime_audio_chunks
            ]
            self._pending_realtime_audio_chunks.clear()
            thread_status_publisher = self.thread_status_publisher
            if thread_status_publisher is not None:
                await thread_status_publisher.set_thread_status(
                    status="Processing audio",
                    mode="busy",
                )
            return

        event_handler = None
        pending_events = self._pending_realtime_audio_events_by_turn_id.setdefault(
            turn_id, []
        )

        def buffer_realtime_audio_event(event: dict[str, Any]) -> None:
            pending_events.append(event)

        event_handler = buffer_realtime_audio_event

        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is not None:
            await thread_status_publisher.set_thread_status(
                status="Processing audio",
                mode="busy",
            )

        had_thread_id = "thread_id" in session.metadata
        previous_thread_id = session.metadata.get("thread_id")
        had_turn_id = "turn_id" in session.metadata
        previous_turn_id = session.metadata.get("turn_id")
        session.metadata["thread_id"] = commit.thread_id
        if turn_id is not None:
            session.metadata["turn_id"] = turn_id
        try:
            await self.llm_adapter.start_session(
                context=session,
                event_handler=event_handler,
            )
            await session.commit_realtime_audio()
            self._pending_realtime_audio_status_active = False
        finally:
            if had_thread_id:
                session.metadata["thread_id"] = previous_thread_id
            else:
                session.metadata.pop("thread_id", None)

            if had_turn_id:
                session.metadata["turn_id"] = previous_turn_id
            else:
                session.metadata.pop("turn_id", None)

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

    async def on_secret_response(self, message: Message) -> None:
        response = _coerce_message_data(message.data, AgentSecretResponse)
        if self._turn_id != response.turn_id:
            return

        await self._resolve_secret_request(response)

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
        self._pending_realtime_audio_chunks.clear()
        self._pending_realtime_audio_chunks_by_turn_id.clear()
        self._pending_realtime_audio_events_by_turn_id.clear()
        self._pending_realtime_audio_status_active = False
        self._cancel_pending_tool_call_approvals()
        self._cancel_pending_secret_requests()
        await self._cancel_pending_client_tool_calls(reason="process_stopped")
        await self._clear_pending_status_messages()
        thread_status_publisher = self.thread_status_publisher
        if thread_status_publisher is not None:
            await thread_status_publisher.set_thread_turn_id(turn_id=None)
            await thread_status_publisher.clear_thread_status()

        if self._session_context is not None:
            await self.llm_adapter.stop_session(context=self._session_context)
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
