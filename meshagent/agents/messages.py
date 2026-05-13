from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from meshagent.api.agent_content import (
    AgentContent,
    AgentAudioContent,
    AgentFileContent,
    AgentInputContent,
    AgentTextContent,
)
from meshagent.api.messaging import Content
from meshagent.api.messaging import unpack_content_parts
from pydantic import BaseModel, ConfigDict, Field, field_serializer

__all__ = [
    "AgentContent",
    "AgentAudioContent",
    "AgentFileContent",
    "AgentInputContent",
    "AgentTextContent",
]

AGENT_MESSAGE_TURN_START = "meshagent.agent.turn.start"
AGENT_MESSAGE_TURN_STEER = "meshagent.agent.turn.steer"
AGENT_MESSAGE_TURN_INTERRUPT = "meshagent.agent.turn.interrupt"
AGENT_MESSAGE_REALTIME_AUDIO_CHUNK = "meshagent.agent.realtime_audio.chunk"
AGENT_MESSAGE_REALTIME_AUDIO_COMMIT = "meshagent.agent.realtime_audio.commit"
AGENT_MESSAGE_THREAD_START = "meshagent.agent.thread.start"
AGENT_MESSAGE_THREAD_CLEAR = "meshagent.agent.thread.clear"
AGENT_MESSAGE_THREAD_OPEN = "meshagent.agent.thread.open"
AGENT_MESSAGE_THREAD_CLOSE = "meshagent.agent.thread.close"
AGENT_MESSAGE_THREAD_DELETE = "meshagent.agent.thread.delete"
AGENT_MESSAGE_THREAD_RENAME = "meshagent.agent.thread.rename"
AGENT_MESSAGE_CAPABILITIES_REQUEST = "meshagent.agent.capabilities_request"
AGENT_MESSAGE_CAPABILITIES_RESPONSE = "meshagent.agent.capabilities_response"
AGENT_MESSAGE_MODELS_REQUEST = "meshagent.agent.models.request"
AGENT_MESSAGE_MODELS_RESPONSE = "meshagent.agent.models.response"
AGENT_MESSAGE_MODEL_CHANGE = "meshagent.agent.model.change"
AGENT_EVENT_MODEL_CHANGED = "meshagent.agent.model.changed"
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
AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA = "meshagent.agent.tool_call.arguments_delta"
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
AGENT_EVENT_AUDIO_GENERATION_STARTED = "meshagent.agent.audio_generation.started"
AGENT_EVENT_AUDIO_GENERATION_DELTA = "meshagent.agent.audio_generation.delta"
AGENT_EVENT_AUDIO_GENERATION_COMPLETED = "meshagent.agent.audio_generation.completed"
AGENT_EVENT_AUDIO_GENERATION_FAILED = "meshagent.agent.audio_generation.failed"
AGENT_EVENT_AUDIO_TRANSCRIPTION_STARTED = "meshagent.agent.audio_transcription.started"
AGENT_EVENT_AUDIO_TRANSCRIPTION_DELTA = "meshagent.agent.audio_transcription.delta"
AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED = (
    "meshagent.agent.audio_transcription.completed"
)
AGENT_EVENT_AUDIO_TRANSCRIPTION_FAILED = "meshagent.agent.audio_transcription.failed"
AGENT_EVENT_AUDIO_INPUT_SPEECH_STARTED = "meshagent.agent.audio_input.speech_started"
AGENT_EVENT_AUDIO_INPUT_SPEECH_ENDED = "meshagent.agent.audio_input.speech_ended"
AGENT_EVENT_CONTEXT_COMPACTED = "meshagent.agent.context.compacted"
AGENT_EVENT_USAGE_UPDATED = "meshagent.agent.usage.updated"
AGENT_MESSAGE_TOOL_CALL_APPROVE = "meshagent.agent.tool_call.approve"
AGENT_MESSAGE_TOOL_CALL_REJECT = "meshagent.agent.tool_call.reject"


class AgentMessage(BaseModel):
    type: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_name: str | None = None


class AgentThreadMessage(AgentMessage):
    thread_id: str


class AgentLLMMessage(AgentThreadMessage):
    provider: str | None = None
    model: str | None = None


class ToolChoice(BaseModel):
    toolkit_name: str
    tool_name: str


class TurnToolkitConfig(BaseModel):
    client_options: dict[str, Any] | None = None


class StartThread(AgentMessage):
    type: Literal[AGENT_MESSAGE_THREAD_START]
    content: list[AgentInputContent] | None = None
    name: str | None = None
    realtime_protocol: Literal["websocket", "webrtc"] | None = None
    sender_name: str | None = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: str | None = None
    output_modalities: list[Literal["text", "audio"]] | None = Field(
        default=None,
        max_length=1,
    )
    instructions: Optional[str] = None
    toolkits: dict[str, TurnToolkitConfig] | None = None
    tool_choice: ToolChoice | None = None


class TurnStart(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_TURN_START]
    turn_id: str | None = None
    content: list[AgentInputContent] = Field(default_factory=list)
    sender_name: str | None = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: str | None = None
    output_modalities: list[Literal["text", "audio"]] | None = Field(
        default=None,
        max_length=1,
    )
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


class AgentAudioFormat(BaseModel):
    type: str = "audio/pcm"
    sample_rate: int | None = 24000
    bitrate: int | None = None


class AgentRealtimeAudioChunk(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_REALTIME_AUDIO_CHUNK]
    data: bytes = b""
    format: AgentAudioFormat = Field(default_factory=AgentAudioFormat)


class AgentRealtimeAudioCommit(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_REALTIME_AUDIO_COMMIT]
    turn_id: str | None = None
    text: str | None = None
    status: Literal["in_progress", "completed", "cancelled", "failed"] | None = None
    transcription_item_id: str | None = None


class ClearThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_CLEAR]


class OpenThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_OPEN]


class CloseThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_CLOSE]


class DeleteThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_DELETE]


class RenameThread(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_THREAD_RENAME]
    name: str


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


class AgentModelInfo(BaseModel):
    name: str
    friendly_name: str | None = None
    description: str | None = None
    context_window: int | None = None
    pricing: dict[str, float] | None = None
    modalities: list[Literal["text", "audio"]] = Field(default_factory=lambda: ["text"])
    available_voices: list[str] = Field(default_factory=list)
    default_output_voice: str | None = None
    input_format: "AgentAudioFormat | None" = None
    output_format: "AgentAudioFormat | None" = None
    turn_detection: Literal["none", "automatic"] | None = None
    realtime_protocols: list[Literal["websocket", "webrtc"]] = Field(
        default_factory=list
    )
    active: bool = False


class AgentProviderInfo(BaseModel):
    name: str
    friendly_name: str
    description: str | None = None
    default_model: str
    models: list[AgentModelInfo] = Field(default_factory=list)


class ModelsRequest(AgentMessage):
    model_config = ConfigDict(extra="forbid")

    type: Literal[AGENT_MESSAGE_MODELS_REQUEST]


class ModelsResponse(AgentMessage):
    model_config = ConfigDict(extra="forbid")

    type: Literal[AGENT_MESSAGE_MODELS_RESPONSE]
    source_message_id: str
    providers: list[AgentProviderInfo]


class ChangeModel(AgentThreadMessage):
    type: Literal[AGENT_MESSAGE_MODEL_CHANGE]
    provider: str | None = None
    model: str | None = None
    voice: str | None = None


class AgentModelChanged(AgentThreadMessage):
    type: Literal[AGENT_EVENT_MODEL_CHANGED]
    source_message_id: str | None = None
    provider: str
    model: str
    voice: str | None = None
    input_format: "AgentAudioFormat | None" = None
    output_format: "AgentAudioFormat | None" = None
    turn_detection: Literal["none", "automatic"] | None = None
    output_modalities: list[Literal["text", "audio"]] = Field(
        default_factory=lambda: ["text"],
        max_length=1,
    )
    realtime_protocols: list[Literal["websocket", "webrtc"]] = Field(
        default_factory=list
    )


class ThreadCleared(AgentThreadMessage):
    type: Literal[AGENT_EVENT_THREAD_CLEARED]
    source_message_id: str


class TurnStartAccepted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_TURN_START_ACCEPTED]
    turn_id: str | None = None
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
    realtime_connection: "AgentRealtimeConnectionInfo | None" = None


class AgentRealtimeConnectionInfo(BaseModel):
    protocol: Literal["websocket", "webrtc"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    web_only_protocol: str | None = None


class AgentReasoningContentStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentReasoningContentDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str


class AgentReasoningContentEnded(AgentLLMMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentTextContentStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_STARTED]
    turn_id: str
    item_id: str
    phase: Literal["commentary", "final_answer"] | None = None


class AgentTextContentDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str
    sender_name: str | None = None
    phase: Literal["commentary", "final_answer"] | None = None


class AgentTextContentEnded(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_ENDED]
    turn_id: str
    item_id: str
    phase: Literal["commentary", "final_answer"] | None = None


class AgentFileContentStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentFileContentDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_DELTA]
    turn_id: str
    item_id: str
    url: str
    sender_name: str | None = None


class AgentFileContentEnded(AgentLLMMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentToolCallPending(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_PENDING]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None
    argument_bytes: int | None = None


class AgentToolCallInProgress(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_IN_PROGRESS]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None
    argument_bytes: int | None = None


class AgentToolCallStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_STARTED]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str
    tool: str
    arguments: Optional[dict] = None
    argument_bytes: int | None = None


class AgentToolCallArgumentsDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    delta: str


class AgentToolCallLogLine(BaseModel):
    source: Literal["stdout", "stderr"]
    text: str


class AgentToolCallLogDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_LOG_DELTA]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    lines: list[AgentToolCallLogLine]


class AgentToolCallEnded(AgentLLMMessage):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal[AGENT_EVENT_TOOL_CALL_ENDED]
    turn_id: str
    item_id: str
    namespace: str = "meshagent"
    call_id: str | None = None
    toolkit: str | None = None
    tool: str | None = None
    result: Content | None = None
    error: AgentError | None = None

    @field_serializer("result", when_used="json")
    def _serialize_result(self, result: Content | None) -> dict[str, Any] | None:
        if result is None:
            return None

        return result.to_json()


class AgentToolCallApprovalRequested(AgentLLMMessage):
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
    status: str | None = None
    mode: Literal["busy", "steerable"] | None = None
    started_at: str | None = None
    turn_id: str | None = None
    pending_item_id: str | None = None
    total_bytes: int | None = None
    lines_added: int | None = None
    lines_removed: int | None = None


class AgentThreadEvent(AgentLLMMessage):
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


class AgentImageGenerationStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_STARTED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None


class AgentImageGenerationPartial(AgentLLMMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_PARTIAL]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    image: AgentGeneratedImage | None = None
    partial_index: int | None = None


class AgentImageGenerationCompleted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_COMPLETED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    images: list[AgentGeneratedImage] = Field(default_factory=list)


class AgentImageGenerationFailed(AgentLLMMessage):
    type: Literal[AGENT_EVENT_IMAGE_GENERATION_FAILED]
    turn_id: str
    item_id: str
    call_id: str | None = None
    toolkit: str = "image_generation"
    tool: str = "image_generation"
    arguments: Optional[dict[str, Any]] = None
    error: AgentError | None = None


class AgentGeneratedAudio(BaseModel):
    uri: str | None = None
    mime_type: str | None = None
    created_at: str | None = None
    created_by: str | None = None
    status: str | None = None
    transcript: str | None = None


class AgentAudioGenerationStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_GENERATION_STARTED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None


class AgentAudioGenerationDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_GENERATION_DELTA]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    data: bytes = b""
    mime_type: str | None = None
    output_format: AgentAudioFormat | None = None


class AgentAudioGenerationCompleted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_GENERATION_COMPLETED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    audio: AgentGeneratedAudio | None = None
    output_format: AgentAudioFormat | None = None


class AgentAudioGenerationFailed(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_GENERATION_FAILED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    error: AgentError | None = None


class AgentAudioTranscriptionStarted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_TRANSCRIPTION_STARTED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    role: str | None = None


class AgentAudioTranscriptionDelta(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_TRANSCRIPTION_DELTA]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    role: str | None = None
    text: str


class AgentAudioTranscriptionCompleted(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    role: str | None = None
    text: str | None = None


class AgentAudioTranscriptionFailed(AgentLLMMessage):
    type: Literal[AGENT_EVENT_AUDIO_TRANSCRIPTION_FAILED]
    turn_id: str
    item_id: str
    response_id: str | None = None
    content_index: int | None = None
    role: str | None = None
    error: AgentError | None = None


class AgentAudioInputSpeechStarted(AgentThreadMessage):
    type: Literal[AGENT_EVENT_AUDIO_INPUT_SPEECH_STARTED]
    turn_id: str
    item_id: str | None = None
    audio_start_ms: int | None = None


class AgentAudioInputSpeechEnded(AgentThreadMessage):
    type: Literal[AGENT_EVENT_AUDIO_INPUT_SPEECH_ENDED]
    turn_id: str
    item_id: str | None = None
    audio_end_ms: int | None = None


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
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK: AgentRealtimeAudioChunk,
    AGENT_MESSAGE_REALTIME_AUDIO_COMMIT: AgentRealtimeAudioCommit,
    AGENT_MESSAGE_THREAD_CLEAR: ClearThread,
    AGENT_MESSAGE_THREAD_OPEN: OpenThread,
    AGENT_MESSAGE_THREAD_CLOSE: CloseThread,
    AGENT_MESSAGE_THREAD_DELETE: DeleteThread,
    AGENT_MESSAGE_THREAD_RENAME: RenameThread,
    AGENT_MESSAGE_CAPABILITIES_REQUEST: CapabilitiesRequest,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE: CapabilitiesResponse,
    AGENT_MESSAGE_MODELS_REQUEST: ModelsRequest,
    AGENT_MESSAGE_MODELS_RESPONSE: ModelsResponse,
    AGENT_MESSAGE_MODEL_CHANGE: ChangeModel,
    AGENT_EVENT_MODEL_CHANGED: AgentModelChanged,
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
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA: AgentToolCallArgumentsDelta,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA: AgentToolCallLogDelta,
    AGENT_EVENT_TOOL_CALL_ENDED: AgentToolCallEnded,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED: AgentToolCallApprovalRequested,
    AGENT_EVENT_THREAD_STATUS: AgentThreadStatus,
    AGENT_EVENT_THREAD_EVENT: AgentThreadEvent,
    AGENT_EVENT_IMAGE_GENERATION_STARTED: AgentImageGenerationStarted,
    AGENT_EVENT_IMAGE_GENERATION_PARTIAL: AgentImageGenerationPartial,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED: AgentImageGenerationCompleted,
    AGENT_EVENT_IMAGE_GENERATION_FAILED: AgentImageGenerationFailed,
    AGENT_EVENT_AUDIO_GENERATION_STARTED: AgentAudioGenerationStarted,
    AGENT_EVENT_AUDIO_GENERATION_DELTA: AgentAudioGenerationDelta,
    AGENT_EVENT_AUDIO_GENERATION_COMPLETED: AgentAudioGenerationCompleted,
    AGENT_EVENT_AUDIO_GENERATION_FAILED: AgentAudioGenerationFailed,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_STARTED: AgentAudioTranscriptionStarted,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_DELTA: AgentAudioTranscriptionDelta,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED: AgentAudioTranscriptionCompleted,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_FAILED: AgentAudioTranscriptionFailed,
    AGENT_EVENT_AUDIO_INPUT_SPEECH_STARTED: AgentAudioInputSpeechStarted,
    AGENT_EVENT_AUDIO_INPUT_SPEECH_ENDED: AgentAudioInputSpeechEnded,
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
