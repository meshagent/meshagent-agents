from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from meshagent.api.agent_content import (
    AgentContent,
    AgentFileContent,
    AgentInputContent,
    AgentTextContent,
)
from meshagent.api.messaging import Content
from meshagent.api.messaging import unpack_content_parts
from pydantic import BaseModel, ConfigDict, Field, field_serializer

__all__ = [
    "AgentContent",
    "AgentFileContent",
    "AgentInputContent",
    "AgentTextContent",
]

AGENT_MESSAGE_TURN_START = "meshagent.agent.turn.start"
AGENT_MESSAGE_TURN_STEER = "meshagent.agent.turn.steer"
AGENT_MESSAGE_TURN_INTERRUPT = "meshagent.agent.turn.interrupt"
AGENT_MESSAGE_THREAD_START = "meshagent.agent.thread.start"
AGENT_MESSAGE_THREAD_CLEAR = "meshagent.agent.thread.clear"
AGENT_MESSAGE_THREAD_OPEN = "meshagent.agent.thread.open"
AGENT_MESSAGE_THREAD_CLOSE = "meshagent.agent.thread.close"
AGENT_MESSAGE_CAPABILITIES_REQUEST = "meshagent.agent.capabilities_request"
AGENT_MESSAGE_CAPABILITIES_RESPONSE = "meshagent.agent.capabilities_response"
AGENT_EVENT_THREAD_CLEARED = "meshagent.agent.thread.cleared"
AGENT_EVENT_TURN_START_ACCEPTED = "meshagent.agent.turn.start.accepted"
AGENT_EVENT_TURN_START_REJECTED = "meshagent.agent.turn.start.rejected"
AGENT_EVENT_TURN_INTERRUPT_ACCEPTED = "meshagent.agent.turn.interrupt.accepted"
AGENT_EVENT_TURN_INTERRUPTED = "meshagent.agent.turn.interrupted"
AGENT_EVENT_TURN_STEER_ACCEPTED = "meshagent.agent.turn.steer.accepted"
AGENT_EVENT_TURN_STEERED = "meshagent.agent.turn.steered"
AGENT_EVENT_TURN_STEER_REJECTED = "meshagent.agent.turn.steer.rejected"
AGENT_EVENT_TURN_STARTED = "meshagent.agent.turn.started"
AGENT_EVENT_TURN_ENDED = "meshagent.agent.turn.ended"
AGENT_EVENT_THREAD_STARTED = "meshagent.agent.thread.started"
AGENT_EVENT_REASONING_CONTENT_STARTED = "meshagent.agent.reasoning_content.started"
AGENT_EVENT_REASONING_CONTENT_DELTA = "meshagent.agent.reasoning_content.delta"
AGENT_EVENT_REASONING_CONTENT_ENDED = "meshagent.agent.reasoning_content.ended"
AGENT_EVENT_TEXT_CONTENT_STARTED = "meshagent.agent.text_content.started"
AGENT_EVENT_TEXT_CONTENT_DELTA = "meshagent.agent.text_content.delta"
AGENT_EVENT_TEXT_CONTENT_ENDED = "meshagent.agent.text_content.ended"
AGENT_EVENT_FILE_CONTENT_STARTED = "meshagent.agent.file_content.started"
AGENT_EVENT_FILE_CONTENT_DELTA = "meshagent.agent.file_content.delta"
AGENT_EVENT_FILE_CONTENT_ENDED = "meshagent.agent.file_content.ended"
AGENT_EVENT_TOOL_CALL_PENDING = "meshagent.agent.tool_call.pending"
AGENT_EVENT_TOOL_CALL_IN_PROGRESS = "meshagent.agent.tool_call.in_progress"
AGENT_EVENT_TOOL_CALL_STARTED = "meshagent.agent.tool_call.started"
AGENT_EVENT_TOOL_CALL_LOG_DELTA = "meshagent.agent.tool_call.log_delta"
AGENT_EVENT_TOOL_CALL_ENDED = "meshagent.agent.tool_call.ended"
AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED = (
    "meshagent.agent.tool_call.approval_requested"
)
AGENT_EVENT_THREAD_STATUS = "meshagent.agent.thread.status"
AGENT_EVENT_THREAD_EVENT = "meshagent.agent.thread.event"
AGENT_EVENT_IMAGE_GENERATION_STARTED = "meshagent.agent.image_generation.started"
AGENT_EVENT_IMAGE_GENERATION_PARTIAL = "meshagent.agent.image_generation.partial"
AGENT_EVENT_IMAGE_GENERATION_COMPLETED = "meshagent.agent.image_generation.completed"
AGENT_EVENT_IMAGE_GENERATION_FAILED = "meshagent.agent.image_generation.failed"
AGENT_EVENT_CONTEXT_COMPACTED = "meshagent.agent.context.compacted"
AGENT_EVENT_USAGE_UPDATED = "meshagent.agent.usage.updated"
AGENT_MESSAGE_TOOL_CALL_APPROVE = "meshagent.agent.tool_call.approve"
AGENT_MESSAGE_TOOL_CALL_REJECT = "meshagent.agent.tool_call.reject"


class AgentMessage(BaseModel):
    type: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class AgentThreadMessage(AgentMessage):
    thread_id: str


class ToolChoice(BaseModel):
    toolkit_name: str
    tool_name: str


class TurnToolkitConfig(BaseModel):
    client_options: dict[str, Any] | None = None


class StartThread(AgentMessage):
    type: Literal[AGENT_MESSAGE_THREAD_START]
    content: list[AgentInputContent]
    sender_name: str | None = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    toolkits: dict[str, TurnToolkitConfig] | None = None
    tool_choice: ToolChoice | None = None


class TurnStart(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TURN_START]
    content: list[AgentInputContent]
    sender_name: str | None = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    toolkits: dict[str, TurnToolkitConfig] | None = None
    tool_choice: ToolChoice | None = None


class TurnSteer(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TURN_STEER]
    content: list[AgentInputContent]
    turn_id: str
    sender_name: str | None = None


class TurnInterrupt(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TURN_INTERRUPT]
    turn_id: str


class ClearThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_CLEAR]


class OpenThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_OPEN]


class CloseThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_CLOSE]


class CapabilitiesRequest(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_CAPABILITIES_REQUEST]


class AgentError(BaseModel):
    message: str
    code: Optional[str]


class ToolkitToolCapabilities(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None


class ToolkitCapabilities(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    thumbnail_url: str | None = None
    rules: list[str] = Field(default_factory=list)
    client_options: dict[str, Any] | None = None
    hidden: bool = False
    tools: list[ToolkitToolCapabilities] = Field(default_factory=list)


class CapabilitiesResponse(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_CAPABILITIES_RESPONSE]
    source_message_id: str
    version: str
    toolkits: list[ToolkitCapabilities]


class ThreadCleared(AgentThreadMessage):
    type: Literal[AGENT_EVENT_THREAD_CLEARED]
    source_message_id: str


class TurnStartAccepted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_START_ACCEPTED]
    source_message_id: str
    content: list[AgentInputContent] = Field(default_factory=list)
    sender_name: str | None = None


class TurnStartRejected(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_START_REJECTED]
    source_message_id: str
    error: AgentError


class TurnInterruptAccepted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_INTERRUPT_ACCEPTED]
    turn_id: str
    source_message_id: str


class TurnInterrupted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_INTERRUPTED]
    turn_id: str
    source_message_id: str


class TurnSteerAccepted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_STEER_ACCEPTED]
    turn_id: str
    source_message_id: str
    content: list[AgentInputContent] = Field(default_factory=list)
    sender_name: str | None = None


class TurnSteered(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_STEERED]
    turn_id: str
    source_message_id: str


class TurnSteerRejected(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_STEER_REJECTED]
    turn_id: str
    source_message_id: str
    error: AgentError


class TurnStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_STARTED]
    turn_id: str
    source_message_id: str


class TurnEnded(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_ENDED]
    turn_id: str
    error: Optional[AgentError]


class ThreadStarted(AgentMessage):
    type: Literal[AGENT_EVENT_THREAD_STARTED]
    source_message_id: str
    thread_id: str


class AgentReasoningContentStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentReasoningContentDelta(AgentThreadMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str


class AgentReasoningContentEnded(AgentThreadMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentTextContentStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentTextContentDelta(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str
    sender_name: str | None = None


class AgentTextContentEnded(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentFileContentStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentFileContentDelta(AgentThreadMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_DELTA]
    turn_id: str
    item_id: str
    url: str
    sender_name: str | None = None


class AgentFileContentEnded(AgentThreadMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentToolCallPending(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_PENDING]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallInProgress(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_IN_PROGRESS]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_STARTED]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallLogLine(BaseModel):
    source: Literal["stdout", "stderr"]
    text: str


class AgentToolCallLogDelta(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_LOG_DELTA]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    lines: list[AgentToolCallLogLine]


class AgentToolCallEnded(AgentThreadMessage):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal[AGENT_EVENT_TOOL_CALL_ENDED]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    result: Content | None = None
    error: AgentError | None = None

    @field_serializer("result", when_used="json")
    def _serialize_result(self, result: Content | None) -> dict[str, Any] | None:
        if result is None:
            return None

        return result.to_json()


class AgentToolCallApprovalRequested(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict[str, Any]] = None


class AgentThreadStatus(AgentThreadMessage):
    type: Literal[AGENT_EVENT_THREAD_STATUS]
    status: str | None
    mode: Literal["busy", "steerable"] | None = None
    started_at: str | None = None
    turn_id: str | None = None


class AgentThreadEvent(AgentThreadMessage):
    type: Literal[AGENT_EVENT_THREAD_EVENT]
    event: dict[str, Any]


class AgentGeneratedImage(BaseModel):
    uri: str | None = None
    mime_type: str | None = None
    created_at: str | None = None
    created_by: str | None = None
    width: int | float | None = None
    height: int | float | None = None
    status: str | None = None
    status_detail: str | None = None


class AgentImageGenerationStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_STARTED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    status_detail: str | None = None


class AgentImageGenerationPartial(AgentThreadMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_PARTIAL]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    image: AgentGeneratedImage | None = None
    partial_index: int | None = None
    status_detail: str | None = None


class AgentImageGenerationCompleted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_COMPLETED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    images: list[AgentGeneratedImage] = Field(default_factory=list)
    status_detail: str | None = None


class AgentImageGenerationFailed(AgentThreadMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_FAILED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    error: AgentError | None = None
    status_detail: str | None = None


class AgentContextCompacted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_CONTEXT_COMPACTED]
    checkpoint_id: str
    path: str
    through_sequence: int
    created_at: str | None = None
    messages: list[dict[str, Any]] | None = None


class AgentContextWindowUsage(BaseModel):
    used_tokens: int
    total_tokens: int | None = None
    compaction_mode: str | None = None
    compaction_threshold: int | None = None


class AgentUsageUpdated(AgentThreadMessage):
    type: Literal[AGENT_EVENT_USAGE_UPDATED]
    turn_id: str | None = None
    usage: dict[str, float] = Field(default_factory=dict)
    context_window: AgentContextWindowUsage


class ApproveAgentToolCall(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TOOL_CALL_APPROVE]
    turn_id: str
    item_id: str


class RejectAgentToolCall(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TOOL_CALL_REJECT]
    turn_id: str
    item_id: str


_AGENT_MESSAGE_MODELS: dict[str, type[AgentMessage]] = {
    AGENT_MESSAGE_THREAD_START: StartThread,
    AGENT_MESSAGE_TURN_START: TurnStart,
    AGENT_MESSAGE_TURN_STEER: TurnSteer,
    AGENT_MESSAGE_TURN_INTERRUPT: TurnInterrupt,
    AGENT_MESSAGE_THREAD_CLEAR: ClearThread,
    AGENT_MESSAGE_THREAD_OPEN: OpenThread,
    AGENT_MESSAGE_THREAD_CLOSE: CloseThread,
    AGENT_MESSAGE_CAPABILITIES_REQUEST: CapabilitiesRequest,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE: CapabilitiesResponse,
    AGENT_EVENT_THREAD_CLEARED: ThreadCleared,
    AGENT_EVENT_TURN_START_ACCEPTED: TurnStartAccepted,
    AGENT_EVENT_TURN_START_REJECTED: TurnStartRejected,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED: TurnInterruptAccepted,
    AGENT_EVENT_TURN_INTERRUPTED: TurnInterrupted,
    AGENT_EVENT_TURN_STEER_ACCEPTED: TurnSteerAccepted,
    AGENT_EVENT_TURN_STEERED: TurnSteered,
    AGENT_EVENT_TURN_STEER_REJECTED: TurnSteerRejected,
    AGENT_EVENT_TURN_STARTED: TurnStarted,
    AGENT_EVENT_TURN_ENDED: TurnEnded,
    AGENT_EVENT_THREAD_STARTED: ThreadStarted,
    AGENT_EVENT_REASONING_CONTENT_STARTED: AgentReasoningContentStarted,
    AGENT_EVENT_REASONING_CONTENT_DELTA: AgentReasoningContentDelta,
    AGENT_EVENT_REASONING_CONTENT_ENDED: AgentReasoningContentEnded,
    AGENT_EVENT_TEXT_CONTENT_STARTED: AgentTextContentStarted,
    AGENT_EVENT_TEXT_CONTENT_DELTA: AgentTextContentDelta,
    AGENT_EVENT_TEXT_CONTENT_ENDED: AgentTextContentEnded,
    AGENT_EVENT_FILE_CONTENT_STARTED: AgentFileContentStarted,
    AGENT_EVENT_FILE_CONTENT_DELTA: AgentFileContentDelta,
    AGENT_EVENT_FILE_CONTENT_ENDED: AgentFileContentEnded,
    AGENT_EVENT_TOOL_CALL_PENDING: AgentToolCallPending,
    AGENT_EVENT_TOOL_CALL_IN_PROGRESS: AgentToolCallInProgress,
    AGENT_EVENT_TOOL_CALL_STARTED: AgentToolCallStarted,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA: AgentToolCallLogDelta,
    AGENT_EVENT_TOOL_CALL_ENDED: AgentToolCallEnded,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED: AgentToolCallApprovalRequested,
    AGENT_EVENT_THREAD_STATUS: AgentThreadStatus,
    AGENT_EVENT_THREAD_EVENT: AgentThreadEvent,
    AGENT_EVENT_IMAGE_GENERATION_STARTED: AgentImageGenerationStarted,
    AGENT_EVENT_IMAGE_GENERATION_PARTIAL: AgentImageGenerationPartial,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED: AgentImageGenerationCompleted,
    AGENT_EVENT_IMAGE_GENERATION_FAILED: AgentImageGenerationFailed,
    AGENT_EVENT_CONTEXT_COMPACTED: AgentContextCompacted,
    AGENT_EVENT_USAGE_UPDATED: AgentUsageUpdated,
    AGENT_MESSAGE_TOOL_CALL_APPROVE: ApproveAgentToolCall,
    AGENT_MESSAGE_TOOL_CALL_REJECT: RejectAgentToolCall,
}


def parse_agent_message(data: dict[str, Any]) -> AgentMessage:
    message_type = data.get("type")
    if not isinstance(message_type, str):
        raise ValueError("agent message is missing required string field 'type'")

    model = _AGENT_MESSAGE_MODELS.get(message_type)
    if model is None:
        raise ValueError(f"unsupported agent message type: {message_type}")

    payload = dict(data)
    if model is AgentToolCallEnded:
        result = payload.get("result")
        if isinstance(result, dict):
            payload["result"] = unpack_content_parts(header=result, payload=b"")

    return model.model_validate(payload)
