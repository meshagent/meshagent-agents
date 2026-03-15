from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, Optional, TypeAlias

from meshagent.api.messaging import Content
from pydantic import BaseModel, ConfigDict, Field, field_serializer

AGENT_MESSAGE_TURN_START = "meshagent.agent.turn.start"
AGENT_MESSAGE_TURN_STEER = "meshagent.agent.turn.steer"
AGENT_MESSAGE_TURN_INTERRUPT = "meshagent.agent.turn.interrupt"
AGENT_MESSAGE_THREAD_CLEAR = "meshagent.agent.thread.clear"
AGENT_EVENT_THREAD_CLEARED = "meshagent.agent.thread.cleared"
AGENT_EVENT_TURN_START_ACCEPTED = "meshagent.agent.turn.start.accepted"
AGENT_EVENT_TURN_INTERRUPT_ACCEPTED = "meshagent.agent.turn.interrupt.accepted"
AGENT_EVENT_TURN_INTERRUPTED = "meshagent.agent.turn.interrupted"
AGENT_EVENT_TURN_STEER_ACCEPTED = "meshagent.agent.turn.steer.accepted"
AGENT_EVENT_TURN_STEERED = "meshagent.agent.turn.steered"
AGENT_EVENT_TURN_STEER_REJECTED = "meshagent.agent.turn.steer.rejected"
AGENT_EVENT_TURN_STARTED = "meshagent.agent.turn.started"
AGENT_EVENT_TURN_ENDED = "meshagent.agent.turn.ended"
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
AGENT_MESSAGE_TOOL_CALL_APPROVE = "meshagent.agent.tool_call.approve"
AGENT_MESSAGE_TOOL_CALL_REJECT = "meshagent.agent.tool_call.reject"


class AgentMessage(BaseModel):
    type: str
    thread_id: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


AGENT_CONTENT_TYPE_TEXT = "text"
AGENT_CONTENT_TYPE_FILE = "file"


class AgentContent(BaseModel):
    pass


class AgentTextContent(AgentContent):
    type: Literal[AGENT_CONTENT_TYPE_TEXT]
    text: str


class AgentFileContent(AgentContent):
    type: Literal[AGENT_CONTENT_TYPE_FILE]
    url: str


AgentInputContent: TypeAlias = Annotated[
    AgentTextContent | AgentFileContent,
    Field(discriminator="type"),
]


class TurnStart(AgentMessage):
    type: Literal[AGENT_MESSAGE_TURN_START]
    content: list[AgentInputContent]
    toolkits: Optional[list[dict[str, Any]]] = None
    model: Optional[str] = None
    instructions: Optional[str] = None


class TurnSteer(AgentMessage):
    type: Literal[AGENT_MESSAGE_TURN_STEER]
    content: list[AgentInputContent]
    toolkits: Optional[list[dict[str, Any]]] = None
    turn_id: str


class TurnInterrupt(AgentMessage):
    type: Literal[AGENT_MESSAGE_TURN_INTERRUPT]
    turn_id: str


class ClearThread(AgentMessage):
    type: Literal[AGENT_MESSAGE_THREAD_CLEAR]


class AgentError(BaseModel):
    message: str
    code: Optional[str]


class ThreadCleared(AgentMessage):
    type: Literal[AGENT_EVENT_THREAD_CLEARED]
    source_message_id: str


class TurnStartAccepted(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_START_ACCEPTED]
    source_message_id: str


class TurnInterruptAccepted(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_INTERRUPT_ACCEPTED]
    turn_id: str
    source_message_id: str


class TurnInterrupted(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_INTERRUPTED]
    turn_id: str
    source_message_id: str


class TurnSteerAccepted(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_STEER_ACCEPTED]
    turn_id: str
    source_message_id: str


class TurnSteered(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_STEERED]
    turn_id: str
    source_message_id: str


class TurnSteerRejected(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_STEER_REJECTED]
    turn_id: str
    source_message_id: str
    error: AgentError


class TurnStarted(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_STARTED]
    turn_id: str
    source_message_id: str


class TurnEnded(AgentMessage):
    type: Literal[AGENT_EVENT_TURN_ENDED]
    turn_id: str
    error: Optional[AgentError]


class AgentReasoningContentStarted(AgentMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentReasoningContentDelta(AgentMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str


class AgentReasoningContentEnded(AgentMessage):
    type: Literal[AGENT_EVENT_REASONING_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentTextContentStarted(AgentMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentTextContentDelta(AgentMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_DELTA]
    turn_id: str
    item_id: str
    text: str


class AgentTextContentEnded(AgentMessage):
    type: Literal[AGENT_EVENT_TEXT_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentFileContentStarted(AgentMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_STARTED]
    turn_id: str
    item_id: str


class AgentFileContentDelta(AgentMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_DELTA]
    turn_id: str
    item_id: str
    url: str


class AgentFileContentEnded(AgentMessage):
    type: Literal[AGENT_EVENT_FILE_CONTENT_ENDED]
    turn_id: str
    item_id: str


class AgentToolCallPending(AgentMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_PENDING]
    turn_id: str
    item_id: str
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallInProgress(AgentMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_IN_PROGRESS]
    turn_id: str
    item_id: str
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallStarted(AgentMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_STARTED]
    turn_id: str
    item_id: str
    toolkit: str
    tool: str
    arguments: Optional[dict] = None


class AgentToolCallLogLine(BaseModel):
    source: Literal["stdout", "stderr"]
    text: str


class AgentToolCallLogDelta(AgentMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_LOG_DELTA]
    turn_id: str
    item_id: str
    lines: list[AgentToolCallLogLine]


class AgentToolCallEnded(AgentMessage):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal[AGENT_EVENT_TOOL_CALL_ENDED]
    turn_id: str
    item_id: str
    result: Content | None = None
    error: AgentError | None = None

    @field_serializer("result", when_used="json")
    def _serialize_result(self, result: Content | None) -> dict[str, Any] | None:
        if result is None:
            return None

        return result.to_json()


class AgentToolCallApprovalRequested(AgentMessage):
    type: Literal[AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED]
    turn_id: str
    item_id: str
    toolkit: str
    tool: str
    arguments: Optional[dict[str, Any]] = None


class ApproveAgentToolCall(AgentMessage):
    type: Literal[AGENT_MESSAGE_TOOL_CALL_APPROVE]
    turn_id: str
    item_id: str


class RejectAgentToolCall(AgentMessage):
    type: Literal[AGENT_MESSAGE_TOOL_CALL_REJECT]
    turn_id: str
    item_id: str
