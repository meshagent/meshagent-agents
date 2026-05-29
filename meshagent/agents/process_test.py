import asyncio
import re
import uuid
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse


import pytest
import pyarrow as pa
from pydantic import ValidationError

import meshagent.agents.process as process_module
import meshagent.agents.process_thread_adapter as process_thread_adapter_module
import meshagent.agents.thread_adapter as thread_adapter_module
from meshagent.agents import MeshDocumentThreadStorage
from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.thread_status_publisher import (
    AgentMessageThreadStatusPublisher,
    ParticipantAttributeThreadStatusPublisher,
)
from meshagent.agents.thread_schema import thread_schema
from meshagent.agents.adapter import (
    LLMAdapter,
    LLMModelInfo,
    LLMProvider,
    ToolCallApprovalRequest,
)
from meshagent.agents.context import AgentSessionContext, SessionUsage
from meshagent.agents.messages import (
    AGENT_EVENT_FILE_CONTENT_DELTA,
    AGENT_EVENT_FILE_CONTENT_ENDED,
    AGENT_EVENT_FILE_CONTENT_STARTED,
    AGENT_EVENT_AUDIO_GENERATION_DELTA,
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TOOL_CALL_PENDING,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_REASONING_CONTENT_DELTA,
    AGENT_EVENT_REASONING_CONTENT_ENDED,
    AGENT_EVENT_REASONING_CONTENT_STARTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_DELETE,
    AGENT_MESSAGE_THREAD_LIST,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_RENAME,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_PARTICIPANT_CONNECT,
    AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
    AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    ApproveAgentToolCall,
    AgentError,
    AgentAudioFormat,
    AgentAudioGenerationDelta,
    AgentAudioTranscriptionCompleted,
    AgentContextCompacted,
    AgentContextWindowUsage,
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentMessage,
    AgentModelChanged,
    AgentProviderInfo,
    AgentRealtimeAudioChunk,
    AgentRealtimeAudioCommit,
    AgentTextContent,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallArgumentsDelta,
    AgentClientToolCallCancelled,
    AgentClientToolCallRequested,
    AgentClientToolCallResponse,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallPending,
    AgentToolCallStarted,
    AgentToolCallEnded,
    AgentThreadEvent,
    AgentThreadStatus,
    AgentUsageUpdated,
    ClearThread,
    CloseThread,
    DeleteThread,
    ListThreads,
    ModelsRequest,
    OpenThread,
    ParticipantConnect,
    ParticipantDisconnect,
    RejectAgentToolCall,
    RenameThread,
    ThreadCreated,
    ThreadDeleted,
    ThreadsListed,
    ThreadLoaded,
    ThreadUpdated,
    TurnStart,
    TurnStartAccepted,
    StartThread,
    ClientToolkitDescription,
    TurnStartRejected,
    TurnEnded,
    TurnInterrupted,
    TurnInterrupt,
    TurnStarted,
    ToolChoice,
    TurnSteerAccepted,
    TurnSteer,
    TurnSteerRejected,
    TurnMCPConfig,
    TurnToolkitConfig,
)
from meshagent.agents.process import (
    AgentProcess,
    AgentSupervisor,
    Channel,
    ChatAgentProcess,
    ContentScheme,
    LLMAgentProcess,
    Message,
    ThreadIsolationMode,
)
from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.agents.thread_storage import ThreadListEntry, ThreadListPage
from meshagent.api import MeshDocument, Participant, RemoteParticipant
from meshagent.api.messaging import (
    BinaryContent,
    Content,
    ErrorContent,
    FileContent,
    JsonContent,
    TextContent,
)
from meshagent.tools import LocalRoomTool, ToolContext, Toolkit, tool


class _LifecycleChannel(Channel):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.stopped = 0
        self.start_event = asyncio.Event()
        self.stop_event = asyncio.Event()

    async def on_start(self) -> None:
        self.started += 1
        self.start_event.set()

    async def on_stop(self) -> None:
        self.stopped += 1
        self.stop_event.set()


def test_supervisor_rejects_duplicate_channel_turn_tools() -> None:
    @tool(name="attach_file")
    async def attach_file(path: str) -> None:
        del path

    @tool(name="list_threads")
    async def list_threads() -> JsonContent:
        return JsonContent(json={"threads": []})

    @tool(name="grep_thread_list")
    async def grep_thread_list(pattern: str) -> JsonContent:
        del pattern
        return JsonContent(json={"threads": []})

    @tool(name="channel_specific")
    async def channel_specific() -> None:
        return None

    class _ThreadToolChannel(Channel):
        def __init__(self, tools) -> None:
            super().__init__()
            self._tools = tools

        def get_turn_toolkits(
            self,
            *,
            thread_id: str,
            turn_id: str | None,
        ) -> list[Toolkit]:
            assert thread_id == "/threads/current.thread"
            assert turn_id == "turn-1"
            return [Toolkit(name="chat", tools=[*self._tools])]

    supervisor = AgentSupervisor()
    supervisor.add_channel(
        _ThreadToolChannel([attach_file, list_threads, grep_thread_list])
    )
    supervisor.add_channel(
        _ThreadToolChannel(
            [attach_file, list_threads, grep_thread_list, channel_specific]
        )
    )

    with pytest.raises(ValueError, match="duplicate turn tool registered: chat"):
        supervisor.get_turn_toolkits(
            thread_id="/threads/current.thread",
            turn_id="turn-1",
        )


@pytest.mark.asyncio
async def test_supervisor_registers_thread_list_tools_only_with_thread_storage() -> (
    None
):
    supervisor = _ListThreadSupervisor()

    without_storage = supervisor.get_turn_toolkits(
        thread_id="/threads/current.thread",
        turn_id="turn-1",
        thread_storage=None,
    )
    assert without_storage == []

    with_storage = supervisor.get_turn_toolkits(
        thread_id="/threads/current.thread",
        turn_id="turn-1",
        thread_storage=_LifecycleThreadStorage(path="/threads/current.thread"),
    )
    assert [toolkit.name for toolkit in with_storage] == ["chat"]
    assert {tool.name for tool in with_storage[0].tools} == {
        "list_threads",
        "grep_thread_list",
    }

    caller = RemoteParticipant(id="caller-id")
    list_tool = next(
        tool for tool in with_storage[0].tools if tool.name == "list_threads"
    )
    result = await list_tool.execute(context=ToolContext(caller=caller))

    assert isinstance(result, JsonContent)
    assert result.json["threads"] == [
        {
            "path": "/threads/one.thread",
            "name": "One",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-02T00:00:00Z",
        }
    ]


def test_supervisor_rejects_channel_thread_tool_duplicate_when_thread_storage_exists() -> (
    None
):
    @tool(name="list_threads")
    async def list_threads() -> JsonContent:
        return JsonContent(json={"threads": []})

    class _ThreadToolChannel(Channel):
        def get_turn_toolkits(
            self,
            *,
            thread_id: str,
            turn_id: str | None,
        ) -> list[Toolkit]:
            del thread_id
            del turn_id
            return [Toolkit(name="chat", tools=[list_threads])]

    supervisor = AgentSupervisor()
    supervisor.add_channel(_ThreadToolChannel())

    with pytest.raises(
        ValueError, match="duplicate turn tool registered: chat.list_threads"
    ):
        supervisor.get_turn_toolkits(
            thread_id="/threads/current.thread",
            turn_id="turn-1",
            thread_storage=_LifecycleThreadStorage(path="/threads/current.thread"),
        )


def test_agent_model_info_includes_modalities() -> None:
    class _Adapter(LLMAdapter):
        def default_model(self) -> str:
            return "gpt-realtime"

        def list_models(self) -> list[LLMModelInfo]:
            return [
                LLMModelInfo(
                    name="gpt-realtime",
                    modalities=("text", "audio"),
                )
            ]

    adapter = _Adapter()
    model_info = process_module.agent_model_info(
        provider=LLMProvider(name="openai-realtime", adapter=adapter),
        model_info=adapter.list_models()[0],
        current_provider="openai-realtime",
        current_model="gpt-realtime",
    )

    assert model_info.modalities == ["text", "audio"]


def test_agent_model_info_includes_attachment_capabilities() -> None:
    class _Adapter(LLMAdapter):
        def default_model(self) -> str:
            return "gpt-vision"

        def list_models(self) -> list[LLMModelInfo]:
            return [
                LLMModelInfo(
                    name="gpt-vision",
                    supports_attachments=True,
                    accepts=("image/*", "application/pdf"),
                )
            ]

    adapter = _Adapter()
    model_info = process_module.agent_model_info(
        provider=LLMProvider(name="openai", adapter=adapter),
        model_info=adapter.list_models()[0],
        current_provider="openai",
        current_model="gpt-vision",
    )

    assert model_info.supports_attachments is True
    assert model_info.accepts == ["image/*", "application/pdf"]


class _RecordingChannel(_LifecycleChannel):
    def __init__(self, *, handled_type: str | None = None) -> None:
        super().__init__()
        self.handled_type = handled_type
        self.received: list[Message] = []
        self.message_event = asyncio.Event()

    def handles(self, message: Message) -> bool:
        if self.handled_type is None:
            return True

        return message.data.type == self.handled_type

    async def on_message(self, message: Message) -> None:
        self.received.append(message)
        self.message_event.set()


class _ThreadOpenResponseChannel(_RecordingChannel):
    def __init__(self) -> None:
        super().__init__()
        self.direct_payloads: list[AgentMessage] = []

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        del participant
        self.direct_payloads.append(payload)
        return True


class _ParticipantRoutingChannel(_RecordingChannel):
    def __init__(self) -> None:
        super().__init__()
        self.direct_payloads_by_participant_id: dict[str, list[AgentMessage]] = {}

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        self.direct_payloads_by_participant_id.setdefault(participant.id, []).append(
            payload
        )
        return True


class _RecordingProcess(AgentProcess):
    def __init__(self, *, handled_type: str) -> None:
        super().__init__()
        self.handled_type = handled_type
        self.started = 0
        self.stopped = 0
        self.received: list[Message] = []
        self.start_event = asyncio.Event()
        self.message_event = asyncio.Event()
        self.stop_event = asyncio.Event()

    def handles(self, message: Message) -> bool:
        return message.data.type == self.handled_type

    async def on_start(self) -> None:
        self.started += 1
        self.start_event.set()

    async def on_message(self, message: Message) -> None:
        self.received.append(message)
        self.message_event.set()

    async def on_stop(self) -> None:
        self.stopped += 1
        self.stop_event.set()


class _BackendRecordingProcess(AgentProcess):
    def __init__(self, *, thread_id: str, backend: str) -> None:
        super().__init__(thread_id=thread_id, backend=backend)
        self.received: list[Message] = []
        self.stopped = 0

    def handles(self, message: Message) -> bool:
        return message.data.type in {
            AGENT_MESSAGE_THREAD_OPEN,
            AGENT_MESSAGE_TURN_START,
        }

    async def on_message(self, message: Message) -> None:
        self.received.append(message)

    async def on_stop(self) -> None:
        self.stopped += 1


class _RecordingBackend:
    def __init__(self, *, name: str, thread_id: str = "/threads/backend") -> None:
        self._name = name
        self.thread_id = thread_id
        self.created_processes: list[_BackendRecordingProcess] = []

    @property
    def name(self) -> str:
        return self._name

    async def on_start(self) -> None:
        return None

    async def on_stop(self) -> None:
        return None

    def model_providers(
        self,
        *,
        current_backend: str | None,
        current_provider: str | None,
        current_model: str | None,
    ) -> list[AgentProviderInfo]:
        del current_backend
        del current_provider
        del current_model
        return []

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        del turn_start
        return None

    async def create_realtime_connection(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> None:
        del supervisor
        del thread_id
        del start_thread
        del sender
        return None

    async def create_thread_id(
        self,
        *,
        supervisor: AgentSupervisor,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del supervisor
        del start_thread
        del sender
        return self.thread_id

    def create_thread_process(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
    ) -> AgentProcess:
        del supervisor
        process = _BackendRecordingProcess(thread_id=thread_id, backend=self.name)
        self.created_processes.append(process)
        return process


class _FailingStartProcess(AgentProcess):
    async def on_start(self) -> None:
        raise RuntimeError("boom")


class _FailingStartChannel(Channel):
    async def on_start(self) -> None:
        raise RuntimeError("boom")


class _BlockingStopProcess(AgentProcess):
    def __init__(self, *, handled_type: str = "work") -> None:
        super().__init__()
        self.handled_type = handled_type
        self.received: list[Message] = []
        self.started_event = asyncio.Event()
        self.stop_started = asyncio.Event()
        self.release_stop = asyncio.Event()
        self.on_stop_calls = 0

    def handles(self, message: Message) -> bool:
        return message.data.type == self.handled_type

    async def on_start(self) -> None:
        self.started_event.set()

    async def on_message(self, message: Message) -> None:
        self.received.append(message)

    async def on_stop(self) -> None:
        self.on_stop_calls += 1
        self.stop_started.set()
        await self.release_stop.wait()


class _EmittingProcess(_RecordingProcess):
    def __init__(self, *, handled_type: str, emitted_type: str) -> None:
        super().__init__(handled_type=handled_type)
        self.emitted_type = emitted_type
        self.emitted_event = asyncio.Event()

    async def on_message(self, message: Message) -> None:
        await super().on_message(message)
        payload_message = _PayloadMessage.model_validate(
            message.data.model_dump(mode="python")
        )
        self.emit(
            sender=message.sender,
            payload=_PayloadMessage(
                type=self.emitted_type,
                thread_id=payload_message.thread_id,
                payload=payload_message.payload,
            ),
        )
        self.emitted_event.set()


class _ThreadRecordingProcess(AgentProcess):
    def __init__(self, *, thread_id: str) -> None:
        super().__init__(thread_id=thread_id)
        self.received: list[Message] = []

    def handles(self, message: Message) -> bool:
        return (
            message.data.type
            in {
                AGENT_MESSAGE_TURN_START,
                AGENT_MESSAGE_TURN_STEER,
                AGENT_MESSAGE_THREAD_CLEAR,
            }
            and message.data.thread_id == self.thread_id
        )

    async def on_message(self, message: Message) -> None:
        self.received.append(message)


class _OpenCloseThreadRecordingProcess(AgentProcess):
    def __init__(self, *, thread_id: str) -> None:
        super().__init__(thread_id=thread_id)
        self.received: list[Message] = []
        self.stopped = 0
        self.stop_event = asyncio.Event()

    def handles(self, message: Message) -> bool:
        return (
            message.data.type
            in {
                AGENT_MESSAGE_THREAD_OPEN,
                AGENT_MESSAGE_THREAD_CLOSE,
            }
            and message.data.thread_id == self.thread_id
        )

    async def on_message(self, message: Message) -> None:
        self.received.append(message)

    async def on_stop(self) -> None:
        self.stopped += 1
        self.stop_event.set()


class _LifecycleThreadStorage:
    def __init__(self, *, path: str) -> None:
        self._path = path
        self.started = 0
        self.stopped = 0
        self.flushed = 0
        self.messages: list[AgentMessage] = []

    @property
    def path(self) -> str:
        return self._path

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def wait_until_ready(self) -> None:
        return None

    async def flush(self) -> None:
        self.flushed += 1

    def unflushed_agent_messages(self) -> list[AgentMessage]:
        return [*self.messages]

    def push_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant | None = None,
    ) -> None:
        del sender
        self.messages.append(message)

    def agent_messages(self) -> list[AgentMessage]:
        return [*self.messages]

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter=None,
    ) -> None:
        del context
        del llm_adapter

    async def restore_session_context_async(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter=None,
    ) -> None:
        self.restore_session_context(context=context, llm_adapter=llm_adapter)

    def make_toolkit(self) -> Toolkit:
        return Toolkit(name="thread-storage", tools=[])


class _RestoringLifecycleThreadStorage(_LifecycleThreadStorage):
    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter=None,
    ) -> None:
        del llm_adapter
        assistant_text_by_item_id: dict[str, list[str]] = {}
        for message in self.messages:
            if isinstance(message, TurnStart):
                for item in message.content:
                    if isinstance(item, AgentTextContent) and item.text != "":
                        context.append_user_message(item.text)
                continue

            if isinstance(message, AgentTextContentDelta):
                assistant_text_by_item_id.setdefault(message.item_id, []).append(
                    message.text
                )

        for parts in assistant_text_by_item_id.values():
            text = "".join(parts)
            if text != "":
                context.append_assistant_message(text)


class _ProviderRestoreRecordingThreadStorage(_LifecycleThreadStorage):
    def __init__(self, *, path: str) -> None:
        super().__init__(path=path)
        self.restore_calls: list[dict[str, Any]] = []

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter=None,
    ) -> None:
        self.restore_calls.append(
            {
                "context": context,
                "llm_adapter": llm_adapter,
            }
        )
        context.append_assistant_message(f"restored {len(self.restore_calls)}")


class _StorageThreadRecordingProcess(AgentProcess):
    def __init__(
        self,
        *,
        thread_id: str,
        thread_storage: _LifecycleThreadStorage,
    ) -> None:
        super().__init__(thread_id=thread_id, thread_storage=thread_storage)
        self.received: list[Message] = []

    def handles(self, message: Message) -> bool:
        return (
            message.data.type == AGENT_MESSAGE_TURN_START
            and message.data.thread_id == self.thread_id
        )

    async def on_message(self, message: Message) -> None:
        self.received.append(message)


class _RecordingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def payloads(self, *, message_type: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for message in self.sent:
            if message.data.type == message_type:
                payloads.append(message.data.model_dump(mode="json"))
        return payloads


@pytest.mark.asyncio
async def test_chat_agent_process_records_accepted_turn_to_thread_storage() -> None:
    thread_storage = _LifecycleThreadStorage(path="/threads/test.thread")
    process = ChatAgentProcess(
        thread_id="/threads/test.thread",
        thread_storage=thread_storage,
    )
    supervisor = _RecordingSupervisor()

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=_ThreadParticipant(
                    name="caller",
                    participant_id="caller-id",
                ),
            )
        )

        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )
    finally:
        await process.stop(supervisor)

    assert [message.type for message in thread_storage.messages] == [
        AGENT_MESSAGE_TURN_START,
        AGENT_EVENT_TURN_START_ACCEPTED,
        AGENT_EVENT_TURN_STARTED,
        AGENT_EVENT_TURN_ENDED,
    ]
    persisted_turn = thread_storage.messages[0]
    assert isinstance(persisted_turn, TurnStart)
    assert persisted_turn.turn_id is not None
    accepted = thread_storage.messages[1]
    assert isinstance(accepted, TurnStartAccepted)
    assert accepted.content == persisted_turn.content
    assert accepted.sender_name == "caller"


class _RecordedSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}

    def set_attribute(self, name: str, value: object) -> None:
        self.attributes[name] = value


class _RecordedSpanContext:
    def __init__(self, spans: list[_RecordedSpan], name: str) -> None:
        self._spans = spans
        self._span = _RecordedSpan(name)

    def __enter__(self) -> _RecordedSpan:
        self._spans.append(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class _RecordedTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []

    def start_as_current_span(self, name: str) -> _RecordedSpanContext:
        return _RecordedSpanContext(self.spans, name)


class _PayloadMessage(AgentMessage):
    thread_id: str | None = None
    payload: str | None = None


class _ThreadCreatingSupervisor(AgentSupervisor):
    def __init__(self, *, thread_isolation: ThreadIsolationMode = "global") -> None:
        super().__init__(thread_isolation=thread_isolation)
        self.created_processes: list[_ThreadRecordingProcess] = []

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        process = _ThreadRecordingProcess(thread_id=thread_id)
        self.created_processes.append(process)
        return process


class _OpenCloseThreadCreatingSupervisor(AgentSupervisor):
    def __init__(self, *, thread_isolation: ThreadIsolationMode = "global") -> None:
        super().__init__(thread_isolation=thread_isolation)
        self.created_processes: list[_OpenCloseThreadRecordingProcess] = []

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        process = _OpenCloseThreadRecordingProcess(thread_id=thread_id)
        self.created_processes.append(process)
        return process


class _FailingThreadCreatingSupervisor(AgentSupervisor):
    def create_thread_process(self, thread_id: str) -> AgentProcess:
        del thread_id
        raise ValueError("invalid thread id")


class _FailingTurnValidationSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.validation_started = asyncio.Event()

    async def create_thread_id(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del start_thread
        del sender
        return "/threads/failure"

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        return _ThreadRecordingProcess(thread_id=thread_id)

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        del turn_start
        self.validation_started.set()
        raise RuntimeError("validation failed")


class _StorageThreadCreatingSupervisor(AgentSupervisor):
    def __init__(self, *, thread_storage: _LifecycleThreadStorage) -> None:
        super().__init__()
        self.thread_storage = thread_storage
        self.created_processes: list[_StorageThreadRecordingProcess] = []

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        process = _StorageThreadRecordingProcess(
            thread_id=thread_id,
            thread_storage=self.thread_storage,
        )
        self.created_processes.append(process)
        return process


class _ListThreadSupervisor(AgentSupervisor):
    async def list_threads(
        self,
        *,
        list_threads: ListThreads,
        sender: Participant | None,
    ) -> ThreadListPage:
        del sender
        return ThreadListPage(
            threads=[
                ThreadListEntry(
                    path="/threads/one.thread",
                    name="One",
                    created_at="2026-01-01T00:00:00Z",
                    modified_at="2026-01-02T00:00:00Z",
                )
            ],
            total=1,
            offset=list_threads.offset,
            limit=list_threads.limit,
        )


class _ThreadLifecycleEventSupervisor(AgentSupervisor):
    async def create_thread_id(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del start_thread
        del sender
        return "/threads/created.thread"

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        del thread_id
        return _RecordingProcess(handled_type=AGENT_MESSAGE_TURN_START)

    async def on_thread_started(
        self,
        *,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del sender
        return ThreadListEntry(
            path=thread_id,
            name=start_thread.name or "New Chat",
            created_at="2026-01-01T00:00:00Z",
            modified_at="2026-01-01T00:00:00Z",
        )

    async def on_thread_renamed(
        self,
        *,
        rename_thread: RenameThread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del sender
        return ThreadListEntry(
            path=rename_thread.thread_id,
            name=rename_thread.name,
            created_at="2026-01-01T00:00:00Z",
            modified_at="2026-01-02T00:00:00Z",
        )


class _GenericThreadAdapter(ThreadAdapter):
    async def handle_custom_event(
        self,
        *,
        event: dict,
    ) -> None:
        del event

    async def _process_llm_events(self) -> None:
        return None


class _RecordingThreadStatusPublisher:
    def __init__(self) -> None:
        self.turn_ids: list[str | None] = []
        self.pending_messages: list[list[dict[str, Any]]] = []
        self.statuses: list[dict[str, Any]] = []
        self.clear_count = 0

    async def set_thread_turn_id(self, *, turn_id: str | None) -> None:
        self.turn_ids.append(turn_id)

    async def set_pending_messages(
        self,
        *,
        pending_messages: list[dict[str, Any]],
    ) -> None:
        self.pending_messages.append(pending_messages)

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode=None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
        lines_added: int | None = None,
        lines_removed: int | None = None,
    ) -> None:
        self.statuses.append(
            {
                "status": status,
                "mode": mode,
                "pending_item_id": pending_item_id,
                "total_bytes": total_bytes,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
            }
        )

    async def clear_thread_status(self) -> None:
        self.clear_count += 1


class _DelayedPreparingThreadStatusPublisher(_RecordingThreadStatusPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.first_preparing_started = asyncio.Event()
        self.release_first_preparing = asyncio.Event()
        self._delayed_first_preparing = False

    async def set_thread_status(
        self,
        *,
        status: str | None,
        mode=None,
        pending_item_id: str | None = None,
        total_bytes: int | None = None,
        lines_added: int | None = None,
        lines_removed: int | None = None,
    ) -> None:
        if (
            not self._delayed_first_preparing
            and status == "Preparing to write src/app.py"
            and pending_item_id == "write-tool"
            and total_bytes is None
        ):
            self._delayed_first_preparing = True
            self.first_preparing_started.set()
            await self.release_first_preparing.wait()

        await super().set_thread_status(
            status=status,
            mode=mode,
            pending_item_id=pending_item_id,
            total_bytes=total_bytes,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )


class _LifecycleSession(AgentSessionContext):
    def __init__(self) -> None:
        super().__init__(system_role=None)
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1


class _ToolCallerParticipant:
    def __init__(self) -> None:
        self.id = "participant-1"

    def get_attribute(self, name: str) -> str | None:
        del name
        return None


class _AttachmentRecordingSession(_LifecycleSession):
    def __init__(self) -> None:
        super().__init__()
        self.image_message_calls: list[dict[str, Any]] = []
        self.file_message_calls: list[dict[str, Any]] = []
        self.image_url_calls: list[str] = []
        self.file_url_calls: list[dict[str, str | None]] = []

    @property
    def supports_images(self) -> bool:
        return True

    @property
    def supports_files(self) -> bool:
        return True

    def append_image_message(self, *, mime_type: str, data: bytes) -> dict:
        self.image_message_calls.append({"mime_type": mime_type, "data": data})
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image-bytes",
                    "mime_type": mime_type,
                    "size": len(data),
                }
            ],
        }
        self.messages.append(message)
        return message

    def append_file_message(
        self, *, filename: str, mime_type: str, data: bytes
    ) -> dict:
        self.file_message_calls.append(
            {
                "filename": filename,
                "mime_type": mime_type,
                "data": data,
            }
        )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "file-bytes",
                    "filename": filename,
                    "mime_type": mime_type,
                    "size": len(data),
                }
            ],
        }
        self.messages.append(message)
        return message

    def append_image_url(self, *, url: str) -> dict:
        self.image_url_calls.append(url)
        message = {
            "role": "user",
            "content": [{"type": "image-url", "url": url}],
        }
        self.messages.append(message)
        return message

    def append_file_url(self, *, url: str, filename: str | None = None) -> dict:
        self.file_url_calls.append({"url": url, "filename": filename})
        content = {"type": "file-url", "url": url}
        if filename is not None:
            content["filename"] = filename
        message = {
            "role": "user",
            "content": [content],
        }
        self.messages.append(message)
        return message


class _RealtimeAudioRecordingSession(_LifecycleSession):
    def __init__(self) -> None:
        super().__init__()
        self.audio_chunk_calls: list[dict[str, Any]] = []
        self.commit_calls = 0
        self.operation_order: list[str] = []
        self.event_handler = None

    @property
    def supports_realtime_audio(self) -> bool:
        return True

    async def append_realtime_audio_chunk(
        self,
        *,
        mime_type: str,
        data: bytes,
        sample_rate: int | None = None,
        bitrate: int | None = None,
    ) -> None:
        self.operation_order.append("append")
        self.audio_chunk_calls.append(
            {
                "mime_type": mime_type,
                "data": data,
                "sample_rate": sample_rate,
                "bitrate": bitrate,
            }
        )

    async def commit_realtime_audio(self) -> None:
        self.operation_order.append("commit")
        self.commit_calls += 1
        if self.event_handler is not None:
            self.event_handler(
                {
                    "type": "input_audio_transcription.completed",
                    "item_id": "user-audio-1",
                    "text": "hello from audio",
                }
            )


class _RecordingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self, *, session: _LifecycleSession | None = None) -> None:
        self.session = session if session is not None else _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.start_session_calls: list[dict[str, Any]] = []
        self.realtime_session_calls: list[dict[str, Any]] = []
        self.stop_session_calls: list[dict[str, Any]] = []
        self.call_order: list[str] = []
        self.call_event = asyncio.Event()
        self.start_session_event = asyncio.Event()
        self.stop_session_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def provider_name(self) -> str | None:
        return "test-provider"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        self.session.set_usage_callback(usage_callback)
        return self.session

    async def start_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler=None,
    ) -> None:
        self.call_order.append("start_session")
        self.start_session_calls.append(
            {
                "context": context,
                "messages": [*context.messages],
                "metadata": dict(context.metadata),
                "event_handler": event_handler,
            }
        )
        self.start_session_event.set()

    async def start_realtime_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler=None,
        caller=None,
        toolkits: list[Toolkit] | None = None,
        tool_choice: ToolChoice | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.realtime_session_calls.append(
            {
                "context": context,
                "caller": caller,
                "toolkits": [toolkit.name for toolkit in toolkits or []],
                "tool_choice": tool_choice,
                "model": model,
                "options": options,
            }
        )
        await self.start_session(context=context, event_handler=event_handler)

    async def stop_session(
        self,
        *,
        context: AgentSessionContext,
    ) -> None:
        self.call_order.append("stop_session")
        self.stop_session_calls.append({"context": context})
        self.stop_session_event.set()

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del output_schema
        del steering_callback
        del on_behalf_of
        self.call_order.append("create_response")
        self.calls.append(
            {
                "context": context,
                "caller": caller,
                "messages": [*context.messages],
                "metadata": dict(context.metadata),
                "toolkits": [toolkit.name for toolkit in toolkits],
                "model": model,
                "options": options,
            }
        )
        if event_handler is not None:
            event_handler({"type": "adapter.event", "call_index": len(self.calls) - 1})
        self.call_event.set()
        return {"ok": True}


class _ClientToolkitInvokingLLMAdapter(_RecordingLLMAdapter):
    def __init__(self) -> None:
        super().__init__(session=_LifecycleSession())
        self.tool_request_started = asyncio.Event()
        self.result: Content | None = None

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        client_toolkit = next(
            toolkit for toolkit in toolkits if toolkit.name == "client"
        )
        self.tool_request_started.set()
        self.result = await client_toolkit.invoke(
            context=ToolContext(caller=caller),
            name="pick_color",
            input=JsonContent(json={"color": "blue"}),
        )
        self.call_event.set()
        return {"ok": True}


class _BlockingSessionLLMAdapter(_RecordingLLMAdapter):
    def __init__(self, *, session: _LifecycleSession | None = None) -> None:
        super().__init__(session=session)
        self.block_next_start_session = False
        self.block_next_stop_session = False
        self.start_session_entered = asyncio.Event()
        self.stop_session_entered = asyncio.Event()
        self.release_start_session = asyncio.Event()
        self.release_stop_session = asyncio.Event()

    async def start_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler=None,
    ) -> None:
        if self.block_next_start_session:
            self.block_next_start_session = False
            self.start_session_entered.set()
            await self.release_start_session.wait()
        await super().start_session(context=context, event_handler=event_handler)

    async def stop_session(
        self,
        *,
        context: AgentSessionContext,
    ) -> None:
        if self.block_next_stop_session:
            self.block_next_stop_session = False
            self.stop_session_entered.set()
            await self.release_stop_session.wait()
        await super().stop_session(context=context)


class _AudioRecordingLLMAdapter(_RecordingLLMAdapter):
    def list_models(self) -> list[LLMModelInfo]:
        return [
            LLMModelInfo(
                name=self.default_model(),
                modalities=("text", "audio"),
            )
        ]

    async def start_session(
        self,
        *,
        context: AgentSessionContext,
        event_handler=None,
    ) -> None:
        await super().start_session(context=context, event_handler=event_handler)
        if isinstance(context, _RealtimeAudioRecordingSession):
            context.event_handler = event_handler

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            if event["type"] == "input_audio_transcription.completed":
                callback(
                    AgentAudioTranscriptionCompleted(
                        type=AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id=event["item_id"],
                        role="user",
                        text=event["text"],
                    )
                )

        return publish


class _AutomaticAudioRecordingLLMAdapter(_AudioRecordingLLMAdapter):
    def list_models(self) -> list[LLMModelInfo]:
        return [
            LLMModelInfo(
                name=self.default_model(),
                modalities=("text", "audio"),
                turn_detection="automatic",
            )
        ]


class _UsageRecordingLLMAdapter(_RecordingLLMAdapter):
    def __init__(self) -> None:
        super().__init__(session=_LifecycleSession())
        self.input_token_calls = 0

    def context_window_size(self, model: str) -> float:
        assert model in {"default-model", "gpt-test"}
        return 128000

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: list | None = None,
        output_schema: dict | None = None,
    ) -> int:
        del output_schema
        assert model in {"default-model", "gpt-test"}
        assert toolkits is not None
        self.input_token_calls += 1
        return 120 + (self.input_token_calls * 10)

    async def create_response(self, **kwargs) -> Any:
        context = kwargs["context"]
        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={
                "gpt-test.input_tokens": 1000.0,
                "gpt-test.output_tokens": 250.0,
            },
            context_window_used=1250,
        )
        return await super().create_response(**kwargs)


class _SessionUsageCallbackLLMAdapter(_UsageRecordingLLMAdapter):
    async def create_response(self, **kwargs) -> Any:
        context = kwargs["context"]
        context.emit_usage_updated(
            SessionUsage(
                model="gpt-test",
                usage={
                    "gpt-test.input_tokens": 1000.0,
                    "gpt-test.output_tokens": 250.0,
                },
                context_window_used=1250,
                context_window_size=4096,
            )
        )
        return await _RecordingLLMAdapter.create_response(self, **kwargs)


class _UsageCountingFailingLLMAdapter(_UsageRecordingLLMAdapter):
    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: list | None = None,
        output_schema: dict | None = None,
    ) -> int:
        del context
        del model
        del toolkits
        del output_schema
        self.input_token_calls += 1
        raise RuntimeError("count failed")


class _CompactedUsageLLMAdapter(_UsageRecordingLLMAdapter):
    async def create_response(self, **kwargs) -> Any:
        await _RecordingLLMAdapter.create_response(self, **kwargs)
        context = kwargs["context"]
        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={
                "gpt-test.input_tokens": 1000.0,
                "gpt-test.output_tokens": 250.0,
            },
            context_window_used=250,
        )
        return {}


class _OpenAIStyleUsageLLMAdapter(_UsageRecordingLLMAdapter):
    async def create_response(self, **kwargs) -> Any:
        context = kwargs["context"]
        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={
                "gpt-test.input_tokens": 64000.0,
                "gpt-test.output_tokens": 1200.0,
            },
            context_window_used=65200,
        )
        return await _RecordingLLMAdapter.create_response(self, **kwargs)


class _CompactingLLMAdapter(_RecordingLLMAdapter):
    def __init__(self) -> None:
        super().__init__(session=_LifecycleSession())
        self.compact_calls = 0

    def needs_compaction(self, *, context: AgentSessionContext) -> bool:
        del context
        return self.compact_calls == 0

    async def compact(
        self,
        *,
        context: AgentSessionContext,
        model: str | None = None,
    ) -> None:
        del model
        self.compact_calls += 1
        context.messages.clear()
        context.messages.append(
            {"id": "compaction-1", "type": "compaction", "encrypted_content": "opaque"}
        )


class _ThresholdManualCompactionLLMAdapter(_RecordingLLMAdapter):
    def __init__(self, *, initial_tokens: int, compact_threshold: int) -> None:
        session = _LifecycleSession()
        session.metadata["token_count"] = initial_tokens
        super().__init__(session=session)
        self.compact_threshold = compact_threshold
        self.compact_calls: list[int] = []

    def context_window_size(self, model: str) -> float:
        assert model == "gpt-test"
        return 128000

    def context_management_mode(self) -> str | None:
        return "standalone"

    def compaction_threshold(self, model: str) -> int | None:
        assert model == "gpt-test"
        return self.compact_threshold

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: list | None = None,
        output_schema: dict | None = None,
    ) -> int:
        del toolkits
        del output_schema
        assert model == "gpt-test"
        return int(context.metadata.get("token_count", 0))

    def needs_compaction(self, *, context: AgentSessionContext) -> bool:
        return int(context.metadata.get("token_count", 0)) >= self.compact_threshold

    async def compact(
        self,
        *,
        context: AgentSessionContext,
        model: str | None = None,
    ) -> None:
        assert model == "gpt-test"
        token_count = int(context.metadata.get("token_count", 0))
        self.compact_calls.append(token_count)
        context.messages.clear()
        context.previous_messages.clear()
        context.previous_response_id = None
        context.messages.append(
            {
                "id": "manual-compaction-1",
                "type": "compaction",
                "encrypted_content": "manual-opaque",
            }
        )
        context.metadata["token_count"] = 128

    async def create_response(self, **kwargs) -> Any:
        context = kwargs["context"]
        token_count = int(context.metadata.get("token_count", 0))
        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={"gpt-test.input_tokens": float(token_count)},
            context_window_used=token_count,
        )
        return await super().create_response(**kwargs)


class _ThresholdAutoCompactionLLMAdapter(_RecordingLLMAdapter):
    def __init__(self, *, initial_tokens: int, compact_threshold: int) -> None:
        session = _LifecycleSession()
        session.metadata["token_count"] = initial_tokens
        super().__init__(session=session)
        self.compact_threshold = compact_threshold
        self.compaction_calls: list[int] = []

    def context_window_size(self, model: str) -> float:
        assert model == "gpt-test"
        return 128000

    def context_management_mode(self) -> str | None:
        return "auto"

    def compaction_threshold(self, model: str) -> int | None:
        assert model == "gpt-test"
        return self.compact_threshold

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: list | None = None,
        output_schema: dict | None = None,
    ) -> int:
        del toolkits
        del output_schema
        assert model == "gpt-test"
        return int(context.metadata.get("token_count", 0))

    async def create_response(self, **kwargs) -> Any:
        context = kwargs["context"]
        event_handler = kwargs.get("event_handler")
        token_count = int(context.metadata.get("token_count", 0))
        if token_count >= self.compact_threshold:
            self.compaction_calls.append(token_count)
            context.messages.clear()
            context.previous_messages.clear()
            context.previous_response_id = None
            context.messages.append(
                {
                    "id": "auto-compaction-1",
                    "type": "compaction",
                    "encrypted_content": "auto-opaque",
                }
            )
            context.metadata["token_count"] = 96
            if event_handler is not None:
                event_handler(
                    AgentContextCompacted(
                        type=AGENT_EVENT_CONTEXT_COMPACTED,
                        thread_id=str(context.metadata["thread_id"]),
                        checkpoint_id="auto-compaction-1",
                        path=str(context.metadata["thread_id"]),
                        through_sequence=0,
                        messages=[context.messages[0]],
                    )
                )
        token_count = int(context.metadata.get("token_count", 0))
        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={"gpt-test.input_tokens": float(token_count)},
            context_window_used=token_count,
        )
        return await super().create_response(**kwargs)


class _CancellationUsageLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.call_started = asyncio.Event()
        self.call_cancelled = asyncio.Event()
        self.input_context_messages: list[list[dict[str, Any]]] = []

    def default_model(self) -> str:
        return "gpt-test"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        session = _LifecycleSession()
        session.set_usage_callback(usage_callback)
        return session

    def context_window_size(self, model: str) -> float:
        assert model == "gpt-test"
        return 128000

    async def get_input_tokens(
        self,
        *,
        context: AgentSessionContext,
        model: str,
        toolkits: list | None = None,
        output_schema: dict | None = None,
    ) -> int:
        del toolkits
        del output_schema
        assert model == "gpt-test"
        self.input_context_messages.append([*context.messages])
        total = 0
        for message in context.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
        return total

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del caller
        del toolkits
        del output_schema
        del steering_callback
        del on_behalf_of
        del options
        del tool_choice
        assert model == "gpt-test"

        context.last_usage = SessionUsage(
            model="gpt-test",
            usage={"gpt-test.input_tokens": 999.0},
            context_window_used=999,
        )
        if event_handler is not None:
            event_handler(
                AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id="thread-1",
                    turn_id=str(context.metadata["turn_id"]),
                    item_id="assistant-1",
                    text="partial reply",
                )
            )
        self.call_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.call_cancelled.set()
            raise


class _CustomEventLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.retry_event_sent = asyncio.Event()
        self.release_completion = asyncio.Event()
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del steering_callback
        del model
        del on_behalf_of
        del options

        if event_handler is not None:
            event_handler(
                {
                    "type": "agent.event",
                    "source": "openai",
                    "name": "openai.retry",
                    "kind": "message",
                    "state": "in_progress",
                    "method": "openai.retry",
                    "correlation_key": "llm.retry:test",
                    "headline": "Reconnecting to the LLM (retry 1/10)",
                    "details": ["Retry 1 of 10 in 1.00s."],
                }
            )
        self.retry_event_sent.set()
        await self.release_completion.wait()
        if event_handler is not None:
            event_handler(
                {
                    "type": "agent.event",
                    "source": "openai",
                    "name": "openai.retry",
                    "kind": "message",
                    "state": "completed",
                    "method": "openai.retry",
                    "correlation_key": "llm.retry:test",
                    "headline": "Reconnected to the LLM",
                    "details": ["Recovered after 1 retry."],
                }
            )
        self.call_event.set()
        return {"ok": True}


class _ImageGenerationStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            if event["type"] == "image_started":
                callback(
                    AgentToolCallStarted(
                        type=AGENT_EVENT_TOOL_CALL_STARTED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="image-tool",
                        toolkit="openai",
                        tool="image_generation",
                    )
                )
                return

            callback(
                AgentToolCallEnded(
                    type=AGENT_EVENT_TOOL_CALL_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="image-tool",
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        if event_handler is not None:
            event_handler({"type": "image_started"})
        self.started.set()
        await self.release.wait()
        if event_handler is not None:
            event_handler({"type": "image_ended"})
        return {"ok": True}


class _FinalAnswerTextStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            event_type = event["type"]
            if event_type == "tool_started":
                callback(
                    AgentToolCallStarted(
                        type=AGENT_EVENT_TOOL_CALL_STARTED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="tool-1",
                        toolkit="openai",
                        tool="shell",
                    )
                )
                return
            if event_type == "tool_ended":
                callback(
                    AgentToolCallEnded(
                        type=AGENT_EVENT_TOOL_CALL_ENDED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="tool-1",
                    )
                )
                return
            if event_type == "text_started":
                callback(
                    AgentTextContentStarted(
                        type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="final-1",
                        phase="final_answer",
                    )
                )
                return
            callback(
                AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="final-1",
                    text="done",
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        if event_handler is not None:
            event_handler({"type": "tool_started"})
            event_handler({"type": "tool_ended"})
            event_handler({"type": "text_started"})
            event_handler({"type": "text_delta"})
        return {"ok": True}


class _ShellStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(
        self,
        *,
        command: str | list[str] = "sed -n '1,20p' src/app.py",
        argument_bytes: int | None = None,
        pending: bool = False,
        command_deltas: list[str] | None = None,
    ) -> None:
        self.session = _LifecycleSession()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.command = command
        self.argument_bytes = argument_bytes
        self.pending = pending
        self.command_deltas = command_deltas or []

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            del event
            message_cls = AgentToolCallPending if self.pending else AgentToolCallStarted
            callback(
                message_cls(
                    type=(
                        AGENT_EVENT_TOOL_CALL_PENDING
                        if self.pending
                        else AGENT_EVENT_TOOL_CALL_STARTED
                    ),
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="shell-tool",
                    toolkit="openai",
                    tool="shell",
                    arguments={"action": {"command": self.command}},
                    argument_bytes=self.argument_bytes,
                )
            )
            for delta in self.command_deltas:
                callback(
                    AgentToolCallArgumentsDelta(
                        type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="shell-tool",
                        delta=delta,
                    )
                )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        if event_handler is not None:
            event_handler({"type": "shell_started"})
        self.started.set()
        await self.release.wait()
        return {"ok": True}


class _ToolArgumentDeltaStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            del event
            callback(
                AgentToolCallPending(
                    type=AGENT_EVENT_TOOL_CALL_PENDING,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="write-tool",
                    toolkit="storage",
                    tool="write_file",
                    arguments={"path": "src/app.py"},
                )
            )
            callback(
                AgentToolCallArgumentsDelta(
                    type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="write-tool",
                    delta="x" * 120,
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        if event_handler is not None:
            event_handler({"type": "argument_delta_started"})
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        self.started.set()
        await self.release.wait()
        return {"ok": True}


class _PartialToolArgumentDeltaStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(
        self,
        *,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any],
        deltas: list[str],
    ) -> None:
        self.session = _LifecycleSession()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.toolkit = toolkit
        self.tool = tool
        self.arguments = arguments
        self.deltas = deltas

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            del event
            callback(
                AgentToolCallPending(
                    type=AGENT_EVENT_TOOL_CALL_PENDING,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="partial-tool",
                    toolkit=self.toolkit,
                    tool=self.tool,
                    arguments=self.arguments,
                )
            )
            for delta in self.deltas:
                callback(
                    AgentToolCallArgumentsDelta(
                        type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="partial-tool",
                        delta=delta,
                    )
                )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        if event_handler is not None:
            event_handler({"type": "partial_delta_started"})
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        self.started.set()
        await self.release.wait()
        return {"ok": True}


class _ToolEventArgumentDeltaStatusLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        del custom_event_callback

        def publish(event: dict[str, Any]) -> None:
            del event
            callback(
                AgentThreadEvent(
                    type=AGENT_EVENT_THREAD_EVENT,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="shell-tool",
                    event={
                        "type": "agent.event",
                        "state": "pending",
                        "headline": "Preparing",
                    },
                )
            )
            callback(
                AgentToolCallArgumentsDelta(
                    type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="shell-tool",
                    delta="x" * 140,
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        if event_handler is not None:
            event_handler({"type": "argument_delta_started"})
        del steering_callback
        del model
        del on_behalf_of
        del options
        del tool_choice
        self.started.set()
        await self.release.wait()
        return {"ok": True}


class _PublishingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def provider_name(self) -> str | None:
        return "test-provider"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        def publish(event: dict[str, Any]) -> None:
            item_id = event["item_id"]
            text = event["text"]
            callback(
                AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                )
            )
            callback(
                AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    text=text,
                )
            )
            callback(
                AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del model
        del steering_callback
        del on_behalf_of
        del options
        if event_handler is not None:
            event_handler({"item_id": "assistant-1", "text": "hello"})
        self.call_event.set()
        return {"ok": True}


class _QueuedSteerLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.started_events = [asyncio.Event(), asyncio.Event()]
        self.release_events = [asyncio.Event(), asyncio.Event()]

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del output_schema
        del event_handler
        del steering_callback
        del on_behalf_of
        del options
        call_index = len(self.calls)
        self.calls.append(
            {
                "context": context,
                "caller": caller,
                "messages": [*context.messages],
                "metadata": dict(context.metadata),
                "toolkits": [toolkit.name for toolkit in toolkits],
                "model": model,
            }
        )
        self.started_events[call_index].set()
        await self.release_events[call_index].wait()
        return {"call_index": call_index}


class _InterruptAwareQueuedSteerLLMAdapter(_QueuedSteerLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.on_turn_steer_calls: list[dict[str, Any]] = []

    def on_turn_steer(self, *, context: AgentSessionContext, interrupted: bool) -> None:
        self.on_turn_steer_calls.append(
            {
                "interrupted": interrupted,
                "messages_before": [*context.messages],
            }
        )
        context.append_assistant_message("TURN INTERRUPTED")


class _ToolBoundarySteeringLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.call_started = asyncio.Event()
        self.release_tool_boundary = asyncio.Event()
        self.tool_boundary_applied = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del caller
        del toolkits
        del output_schema
        del event_handler
        del model
        del on_behalf_of
        del options

        self.call_started.set()
        await self.release_tool_boundary.wait()
        messages_before_boundary = [*context.messages]
        steered = False
        if steering_callback is not None:
            steered = await steering_callback()
        self.calls.append(
            {
                "messages_before_boundary": messages_before_boundary,
                "messages_after_boundary": [*context.messages],
                "steered": steered,
            }
        )
        self.tool_boundary_applied.set()
        return {"ok": True}


class _ToolBoundaryThreadOrderingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.call_started = asyncio.Event()
        self.release_tool_boundary = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        def publish(event: dict[str, Any]) -> None:
            event_type = event["type"]
            if event_type == "tool_started":
                callback(
                    AgentToolCallStarted(
                        type=AGENT_EVENT_TOOL_CALL_STARTED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="tool-1",
                        toolkit="storage",
                        tool="write_file",
                        arguments={
                            "path": "/website/openai3/index.html",
                            "text": "<html>hi</html>",
                        },
                    )
                )
                return

            if event_type == "tool_ended":
                callback(
                    AgentToolCallEnded(
                        type=AGENT_EVENT_TOOL_CALL_ENDED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="tool-1",
                        result=JsonContent(
                            json={"ok": True, "path": "/website/openai3/index.html"}
                        ),
                    )
                )
                return

            if event_type == "text":
                callback(
                    AgentTextContentStarted(
                        type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="assistant-1",
                    )
                )
                callback(
                    AgentTextContentDelta(
                        type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="assistant-1",
                        text=event["text"],
                    )
                )
                callback(
                    AgentTextContentEnded(
                        type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        item_id="assistant-1",
                    )
                )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del caller
        del toolkits
        del output_schema
        del model
        del on_behalf_of
        del options

        if event_handler is not None:
            event_handler({"type": "tool_started"})
        self.call_started.set()
        await self.release_tool_boundary.wait()
        if event_handler is not None:
            event_handler({"type": "tool_ended"})
        if steering_callback is not None:
            await steering_callback()
        if event_handler is not None:
            event_handler({"type": "text", "text": "steered reply"})
        return {"ok": True}


class _RestoringLLMAgentProcess(LLMAgentProcess):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.restore_calls: list[dict[str, Any]] = []

    async def on_restore_session_context(
        self,
        turn_id: str,
        session_context: AgentSessionContext,
    ) -> None:
        self.restore_calls.append(
            {
                "turn_id": turn_id,
                "session_context": session_context,
            }
        )


class _DownloadRecordingStorage:
    def __init__(self, *, files: dict[str, FileContent]) -> None:
        self.files = files
        self.download_calls: list[str] = []

    async def download(self, *, path: str) -> FileContent:
        self.download_calls.append(path)
        return self.files[path]


class _RecordingDatasets:
    def __init__(self) -> None:
        self.schemas: dict[str, pa.Schema] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema: dict[str, Any] | pa.Schema,
        mode: str,
    ) -> None:
        del mode
        if isinstance(schema, pa.Schema):
            arrow_schema = schema
        else:
            arrow_schema = pa.schema(
                [
                    pa.field(field_name, field_type)
                    for field_name, field_type in schema.items()
                ]
            )
        self.schemas.setdefault(name, arrow_schema)
        self.rows.setdefault(name, [])

    async def inspect(self, *, table: str) -> pa.Schema:
        return self.schemas.get(table, pa.schema([]))

    async def add_columns(
        self,
        *,
        table: str,
        new_columns: dict[str, Any],
    ) -> None:
        schema = self.schemas.setdefault(table, pa.schema([]))
        existing_names = set(schema.names)
        for field_name, field_value in new_columns.items():
            if field_name not in existing_names:
                if isinstance(field_value, pa.Field):
                    schema = schema.append(field_value)
                else:
                    schema = schema.append(pa.field(field_name, field_value))
        self.schemas[table] = schema
        for row in self.rows.setdefault(table, []):
            for key in new_columns:
                row.setdefault(key, None)

    async def create_index(self, *, table: str, config: Any) -> None:
        del table
        del config

    async def insert(self, *, table: str, records: list[dict[str, Any]]) -> None:
        stored_rows = self.rows.setdefault(table, [])
        for record in records:
            stored_rows.append(dict(record))

    async def search(
        self,
        *,
        table: str,
        where: str | dict[str, Any] | None = None,
        limit: int | None = None,
        select: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(where, str):
            raise AssertionError("string where clauses are not supported in tests")

        results: list[dict[str, Any]] = []
        for row in self.rows.get(table, []):
            if isinstance(where, dict):
                matches = True
                for key, expected_value in where.items():
                    if row.get(key) != expected_value:
                        matches = False
                        break
                if not matches:
                    continue

            if select is None:
                results.append(dict(row))
            else:
                results.append({column: row.get(column) for column in select})

            if limit is not None and len(results) >= limit:
                break

        return results


class _DownloadRecordingRoom:
    def __init__(self, *, files: dict[str, FileContent] | None = None) -> None:
        self.storage = _DownloadRecordingStorage(files=files or {})
        self.datasets = _RecordingDatasets()
        self.local_participant = _ThreadLocalParticipant()
        self.is_closed = False


def _normalize_room_content_path(*, url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.scheme != "room":
        raise ValueError(f"unsupported room file url: {url}")

    raw_path = f"{parsed_url.netloc}{parsed_url.path}".lstrip("/")
    normalized = PurePosixPath("/" + raw_path).as_posix().strip("/")
    if normalized == "":
        raise ValueError("room file url must reference a non-root storage path")

    if any(part in {".", ".."} for part in PurePosixPath(normalized).parts):
        raise ValueError("room file url cannot contain '.' or '..' segments")

    return normalized


def _room_content_scheme(*, room: _DownloadRecordingRoom) -> ContentScheme:
    async def _download(url: str) -> FileContent:
        path = _normalize_room_content_path(url=url)
        return await room.storage.download(path=path)

    return ContentScheme(prefix="room://", download=_download)


def _make_llm_agent_process(
    *,
    room: _DownloadRecordingRoom,
    process_cls: type[LLMAgentProcess] = LLMAgentProcess,
    **kwargs,
) -> LLMAgentProcess:
    process = process_cls(
        participant=room.local_participant,
        **kwargs,
    )
    process.register_content_scheme(_room_content_scheme(room=room))
    return process


class _ReplayThreadCreatingSupervisor(AgentSupervisor):
    def __init__(
        self,
        *,
        room: _DownloadRecordingRoom,
        thread_storage: _LifecycleThreadStorage,
    ) -> None:
        super().__init__()
        self.room = room
        self.thread_storage = thread_storage
        self.created_processes: list[LLMAgentProcess] = []

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        process = _make_llm_agent_process(
            room=self.room,
            thread_id=thread_id,
            llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
            thread_storage=self.thread_storage,
        )
        self.created_processes.append(process)
        return process


class _ThreadParticipant(RemoteParticipant):
    def __init__(
        self,
        *,
        name: str,
        participant_id: str,
        role: str = "user",
    ) -> None:
        super().__init__(
            id=participant_id,
            role=role,
            attributes={"name": name},
            online=True,
        )


class _ThreadLocalParticipant(Participant):
    def __init__(self) -> None:
        super().__init__(id="assistant-id", attributes={"name": "assistant"})
        self._attributes: dict[str, str] = {"name": "assistant"}
        self.set_attribute_calls: list[tuple[str, str | None]] = []

    def get_attribute(self, key: str) -> str | None:
        return self._attributes.get(key)

    async def set_attribute(self, key: str, value: str | None) -> None:
        if value is None:
            self._attributes.pop(key, None)
        else:
            self._attributes[key] = value
        self.set_attribute_calls.append((key, value))


class _ThreadElement:
    def __init__(self, *, tag_name: str, attributes: dict[str, Any] | None = None):
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_ThreadElement"] = []
        self._parent: "_ThreadElement | None" = None

    def get_attribute(self, key: str) -> Any:
        return self._attributes.get(key)

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def get_children(self) -> list["_ThreadElement"]:
        return [*self._children]

    def get_children_by_tag_name(self, tag_name: str) -> list["_ThreadElement"]:
        return [child for child in self._children if child.tag_name == tag_name]

    def append_child(
        self,
        tag_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> "_ThreadElement":
        child = _ThreadElement(tag_name=tag_name, attributes=attributes)
        child._parent = self
        self._children.append(child)
        return child

    def delete(self) -> None:
        if self._parent is None:
            return
        self._parent._children.remove(self)
        self._parent = None


class _ThreadRoot(_ThreadElement):
    def __init__(self) -> None:
        super().__init__(tag_name="thread", attributes={"name": "thread"})
        self.members = self.append_child("members")
        self.messages = self.append_child("messages")


class _ThreadDocument:
    def __init__(self) -> None:
        self.root = _ThreadRoot()

    def get_state(self, vector: bytes | None = None) -> bytes:
        del vector
        return b""

    @property
    def message_elements(self) -> list[_ThreadElement]:
        return self.root.messages.get_children_by_tag_name("message")

    @property
    def event_elements(self) -> list[_ThreadElement]:
        return self.root.messages.get_children_by_tag_name("event")

    @property
    def reasoning_elements(self) -> list[_ThreadElement]:
        return self.root.messages.get_children_by_tag_name("reasoning")

    @property
    def member_names(self) -> list[str]:
        names: list[str] = []
        for child in self.root.members.get_children_by_tag_name("member"):
            name = child.get_attribute("name")
            if isinstance(name, str):
                names.append(name)
        return names


class _ThreadSync:
    def __init__(self, *, document: _ThreadDocument | None = None) -> None:
        self.document = document if document is not None else _ThreadDocument()
        self.open_calls: list[dict[str, Any]] = []
        self.close_calls: list[str] = []
        self.sync_calls: list[dict[str, Any]] = []

    async def open(self, *, path: str, schema=None) -> _ThreadDocument:
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str) -> None:
        self.close_calls.append(path)

    async def sync(self, *, path: str, data: bytes) -> None:
        self.sync_calls.append({"path": path, "data": data})


class _ThreadRoom(_DownloadRecordingRoom):
    def __init__(
        self,
        *,
        document: _ThreadDocument | None = None,
        files: dict[str, FileContent] | None = None,
    ) -> None:
        super().__init__(files=files)
        self.local_participant = _ThreadLocalParticipant()
        self.sync = _ThreadSync(document=document)


class _ApprovalCapableLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self._approval_handler = None
        self.approval_requested = asyncio.Event()
        self.approval_resolved = asyncio.Event()
        self.approval_decisions: list[bool] = []

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def set_tool_call_approval_handler(self, handler) -> None:
        self._approval_handler = handler

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del context
        del toolkits
        del output_schema
        del event_handler
        del model
        del steering_callback
        del on_behalf_of
        del options
        if self._approval_handler is None:
            raise AssertionError("approval handler was not installed")

        request = ToolCallApprovalRequest(
            item_id="approval-1",
            toolkit="filesystem",
            tool="delete",
            arguments={"path": "tmp/file.txt"},
        )
        self.approval_requested.set()
        decision = await self._approval_handler(
            ToolContext(
                caller=_ToolCallerParticipant(),  # type: ignore[arg-type]
            ),
            request,
        )
        self.approval_decisions.append(decision)
        self.approval_resolved.set()
        return {"approved": decision}


class _RoomBindingTool(LocalRoomTool):
    def __init__(self, *, room) -> None:
        super().__init__(
            room=room,
            name="room_binding_tool",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        )

    async def execute(self, context: ToolContext):
        del context
        return TextContent(text=self.room.local_participant.id)


class _ThreadPublishingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
        custom_event_callback=None,
    ):
        def publish(event: dict[str, Any]) -> None:
            del event
            callback(
                AgentToolCallStarted(
                    type=AGENT_EVENT_TOOL_CALL_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="tool-1",
                    toolkit="openai",
                    tool="web_search",
                    arguments={"q": "meshagent"},
                )
            )
            callback(
                AgentToolCallEnded(
                    type=AGENT_EVENT_TOOL_CALL_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="tool-1",
                    result=TextContent(text="result"),
                )
            )
            callback(
                AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="assistant-1",
                )
            )
            callback(
                AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="assistant-1",
                    text="hello",
                )
            )
            callback(
                AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id="assistant-1",
                )
            )

        return publish

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del caller
        del toolkits
        del output_schema
        del model
        del steering_callback
        del on_behalf_of
        del options
        self.calls.append({"messages": [*context.messages]})
        if event_handler is not None:
            event_handler({"type": "agent.process"})
        await asyncio.sleep(0)
        self.call_event.set()
        return {"ok": True}


class _CancellationIgnoringLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.first_call_started = asyncio.Event()
        self.first_call_cancelled = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self, *, usage_callback=None) -> AgentSessionContext:
        return self.session

    async def create_response(
        self,
        *,
        context: AgentSessionContext,
        caller,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Any:
        del caller
        del toolkits
        del output_schema
        del model
        del steering_callback
        del on_behalf_of
        del options

        call_index = len(self.calls)
        self.calls.append({"messages": [*context.messages], "context": context})
        if call_index == 0:
            self.first_call_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.first_call_cancelled.set()
                if event_handler is not None:
                    event_handler(
                        AgentTextContentDelta(
                            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                            thread_id="thread-1",
                            turn_id="late-turn",
                            item_id="assistant-1",
                            text="late text",
                        )
                    )
                return {"ignored_cancellation": True}

        return {"ok": True}


async def _wait_for(
    predicate,
    *,
    timeout: float = 1,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise asyncio.TimeoutError()
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_agent_supervisor_backend_switch_replaces_process_and_keeps_subscribers() -> (
    None
):
    first_backend = _RecordingBackend(name="first")
    second_backend = _RecordingBackend(name="second")
    supervisor = AgentSupervisor(agent_backends=[first_backend, second_backend])
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    participant = _ThreadParticipant(name="User", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=StartThread(
                    type=AGENT_MESSAGE_THREAD_START,
                    backend="first",
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(first_backend.created_processes) == 1)
        first_process = first_backend.created_processes[0]
        assert first_process.backend == "first"

        await supervisor.route(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id=first_backend.thread_id,
                    turn_id="turn-2",
                    backend="second",
                    content=[AgentTextContent(type="text", text="switch")],
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(second_backend.created_processes) == 1)
        second_process = second_backend.created_processes[0]
        assert first_process.stopped == 1
        assert second_process.backend == "second"
        await _wait_for(lambda: len(second_process.received) == 1)
        assert len(second_process.received) == 1
        assert isinstance(second_process.received[0].data, TurnStart)

        await supervisor.route(
            Message(
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id=first_backend.thread_id,
                    load=False,
                ),
                sender=participant,
                source=channel,
            )
        )

        await asyncio.sleep(0)
        assert [
            message.data
            for message in second_process.received
            if isinstance(message.data, OpenThread)
        ] == []
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_single_backend_infers_missing_turn_backend() -> None:
    backend = _RecordingBackend(name="only")
    supervisor = AgentSupervisor(agent_backends=[backend])
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    participant = _ThreadParticipant(name="User", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=StartThread(
                    type=AGENT_MESSAGE_THREAD_START,
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(backend.created_processes) == 1)
        process = backend.created_processes[0]
        assert process.backend == "only"
        assert (
            supervisor.agent_backend_for_thread(thread_id=backend.thread_id) is backend
        )
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_multiple_backends_require_turn_backend() -> None:
    first_backend = _RecordingBackend(name="first")
    second_backend = _RecordingBackend(name="second")
    supervisor = AgentSupervisor(agent_backends=[first_backend, second_backend])
    channel = _RecordingChannel(handled_type=AGENT_EVENT_TURN_START_REJECTED)
    supervisor.add_channel(channel)
    participant = _ThreadParticipant(name="User", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/ambiguous",
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(channel.received) == 1)
        rejected = channel.received[0].data
        assert isinstance(rejected, TurnStartRejected)
        assert rejected.error.code == "backend_required"
        assert len(first_backend.created_processes) == 0
        assert len(second_backend.created_processes) == 0
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_multiple_backends_require_start_thread_backend() -> (
    None
):
    first_backend = _RecordingBackend(name="first")
    second_backend = _RecordingBackend(name="second")
    supervisor = AgentSupervisor(agent_backends=[first_backend, second_backend])
    channel = _RecordingChannel(handled_type=AGENT_EVENT_TURN_START_REJECTED)
    supervisor.add_channel(channel)
    participant = _ThreadParticipant(name="User", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=StartThread(
                    type=AGENT_MESSAGE_THREAD_START,
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(channel.received) == 1)
        rejected = channel.received[0].data
        assert isinstance(rejected, TurnStartRejected)
        assert rejected.error.code == "backend_required"
        assert len(first_backend.created_processes) == 0
        assert len(second_backend.created_processes) == 0
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_channel_start_and_stop_update_state_and_supervisor() -> None:
    supervisor = AgentSupervisor()
    channel = _LifecycleChannel()

    await channel.start(supervisor)

    assert channel.state == "started"
    assert channel.supervisor is supervisor
    assert channel.started == 1

    await channel.stop(supervisor)

    assert channel.state == "stopped"
    assert channel.supervisor is None
    assert channel.stopped == 1


@pytest.mark.asyncio
async def test_agent_process_handles_only_matching_messages() -> None:
    supervisor = AgentSupervisor()
    process = _RecordingProcess(handled_type="work")

    await process.start(supervisor)
    await asyncio.wait_for(process.start_event.wait(), timeout=1)

    process.send(Message(data=AgentMessage(type="ignore", thread_id="thread-1")))

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(process.message_event.wait(), timeout=0.05)

    handled_message = Message(
        data=_PayloadMessage(type="work", thread_id="thread-1", payload="ok")
    )
    process.send(handled_message)

    await asyncio.wait_for(process.message_event.wait(), timeout=1)

    assert process.received == [handled_message]

    await process.stop(supervisor)

    assert process.state == "stopped"
    assert process.supervisor is None
    assert process.stopped == 1


@pytest.mark.asyncio
async def test_agent_supervisor_starts_children_routes_messages_and_stops_them() -> (
    None
):
    supervisor = AgentSupervisor()
    channel = _RecordingChannel(handled_type="work")
    process = _RecordingProcess(handled_type="work")
    supervisor.add_channel(channel)
    supervisor.add_process(process)

    await supervisor.start()
    await asyncio.wait_for(channel.start_event.wait(), timeout=1)
    await asyncio.wait_for(process.start_event.wait(), timeout=1)

    message = Message(data=AgentMessage(type="work", thread_id="thread-1"))
    supervisor.send(message)

    await asyncio.wait_for(channel.message_event.wait(), timeout=1)
    await asyncio.wait_for(process.message_event.wait(), timeout=1)

    assert channel.state == "started"
    assert process.state == "started"
    assert channel.received == [message]
    assert process.received == [message]

    await supervisor.stop()

    assert supervisor.state == "stopped"
    assert channel.state == "stopped"
    assert process.state == "stopped"
    assert channel.stopped == 1
    assert process.stopped == 1


@pytest.mark.asyncio
async def test_agent_supervisor_route_initializes_thread_storage_before_returning() -> (
    None
):
    thread_storage = _LifecycleThreadStorage(path="/threads/created")
    supervisor = _StorageThreadCreatingSupervisor(thread_storage=thread_storage)

    await supervisor.start()

    try:
        turn_start = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="/threads/created",
            content=[],
        )

        await supervisor.route(Message(data=turn_start))

        assert thread_storage.started == 1
        assert len(supervisor.created_processes) == 1
        assert supervisor.created_processes[0].state == "started"
    finally:
        await supervisor.stop()

    assert thread_storage.stopped == 1


@pytest.mark.asyncio
async def test_agent_supervisor_lists_threads_over_agent_messages() -> None:
    supervisor = _ListThreadSupervisor()
    channel = _ThreadOpenResponseChannel()
    supervisor.add_channel(channel)

    await supervisor.start()

    try:
        request = ListThreads(
            type=AGENT_MESSAGE_THREAD_LIST,
            limit=50,
            offset=10,
        )
        sender = RemoteParticipant(
            id="participant-1",
            attributes={"name": "participant"},
        )

        await supervisor.route(Message(data=request, sender=sender))

        assert len(channel.direct_payloads) == 1
        response = channel.direct_payloads[0]
        assert isinstance(response, ThreadsListed)
        assert response.source_message_id == request.message_id
        assert response.total == 1
        assert response.offset == 10
        assert response.limit == 50
        assert response.threads[0].path == "/threads/one.thread"
        assert response.threads[0].name == "One"
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_broadcasts_thread_list_lifecycle_events() -> None:
    supervisor = _ThreadLifecycleEventSupervisor()
    channel = _ParticipantRoutingChannel()
    supervisor.add_channel(channel)
    sender = RemoteParticipant(
        id="participant-1",
        attributes={"name": "participant"},
    )
    same_participant_other_tab = RemoteParticipant(
        id="participant-2",
        attributes={"name": "participant"},
    )
    different_participant = RemoteParticipant(
        id="participant-3",
        attributes={"name": "other-participant"},
    )

    await supervisor.start()

    try:
        for participant in [sender, same_participant_other_tab, different_participant]:
            await supervisor.route(
                Message(
                    data=ParticipantConnect(
                        type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                        participant_id=participant.id,
                    ),
                    sender=participant,
                )
            )

        await supervisor.route(
            Message(
                data=StartThread(
                    type=AGENT_MESSAGE_THREAD_START,
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadCreated)
                for payload in channel.direct_payloads_by_participant_id.get(
                    same_participant_other_tab.id, []
                )
            )
        )

        created = [
            payload
            for payload in channel.direct_payloads_by_participant_id[sender.id]
            if isinstance(payload, ThreadCreated)
        ]
        assert len(created) == 1
        assert created[0].thread.path == "/threads/created.thread"
        assert created[0].thread.name == "New Chat"

        await supervisor.route(
            Message(
                data=RenameThread(
                    type=AGENT_MESSAGE_THREAD_RENAME,
                    thread_id="/threads/created.thread",
                    name="Renamed",
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadUpdated)
                for payload in channel.direct_payloads_by_participant_id.get(
                    different_participant.id, []
                )
            )
        )
        updated = [
            payload
            for payload in channel.direct_payloads_by_participant_id[sender.id]
            if isinstance(payload, ThreadUpdated)
        ]
        assert len(updated) == 1
        assert updated[0].thread.name == "Renamed"

        await supervisor.route(
            Message(
                data=DeleteThread(
                    type=AGENT_MESSAGE_THREAD_DELETE,
                    thread_id="/threads/created.thread",
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadDeleted)
                for payload in channel.direct_payloads_by_participant_id.get(
                    same_participant_other_tab.id, []
                )
            )
        )
        deleted = [
            payload
            for payload in channel.direct_payloads_by_participant_id[sender.id]
            if isinstance(payload, ThreadDeleted)
        ]
        assert len(deleted) == 1
        assert deleted[0].path == "/threads/created.thread"
        for participant in [sender, same_participant_other_tab, different_participant]:
            direct_payloads = channel.direct_payloads_by_participant_id[participant.id]
            assert [type(payload) for payload in direct_payloads] == [
                ThreadCreated,
                ThreadUpdated,
                ThreadDeleted,
            ]
        assert [
            message.data
            for message in channel.received
            if isinstance(message.data, (ThreadCreated, ThreadUpdated, ThreadDeleted))
        ] == []
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_participant_isolation_sends_thread_lifecycle_events_to_owner() -> (
    None
):
    supervisor = _ThreadLifecycleEventSupervisor(thread_isolation="participant")
    channel = _ParticipantRoutingChannel()
    supervisor.add_channel(channel)
    sender = RemoteParticipant(
        id="participant-1",
        attributes={"name": "participant"},
    )
    same_participant_other_tab = RemoteParticipant(
        id="participant-2",
        attributes={"name": "participant"},
    )
    different_participant = RemoteParticipant(
        id="participant-3",
        attributes={"name": "other-participant"},
    )

    await supervisor.start()

    try:
        for participant in [sender, same_participant_other_tab, different_participant]:
            await supervisor.route(
                Message(
                    data=ParticipantConnect(
                        type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                        participant_id=participant.id,
                    ),
                    sender=participant,
                )
            )

        await supervisor.route(
            Message(
                data=StartThread(
                    type=AGENT_MESSAGE_THREAD_START,
                    content=[AgentTextContent(type="text", text="hello")],
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadCreated)
                for payload in channel.direct_payloads_by_participant_id.get(
                    same_participant_other_tab.id, []
                )
            )
        )

        await supervisor.route(
            Message(
                data=RenameThread(
                    type=AGENT_MESSAGE_THREAD_RENAME,
                    thread_id="/threads/created.thread",
                    name="Renamed",
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadUpdated)
                for payload in channel.direct_payloads_by_participant_id.get(
                    sender.id, []
                )
            )
        )

        await supervisor.route(
            Message(
                data=DeleteThread(
                    type=AGENT_MESSAGE_THREAD_DELETE,
                    thread_id="/threads/created.thread",
                ),
                sender=sender,
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(payload, ThreadDeleted)
                for payload in channel.direct_payloads_by_participant_id.get(
                    sender.id, []
                )
            )
        )

        direct_payloads = channel.direct_payloads_by_participant_id[sender.id]
        assert [type(payload) for payload in direct_payloads] == [
            ThreadCreated,
            ThreadUpdated,
            ThreadDeleted,
        ]
        mirrored_direct_payloads = channel.direct_payloads_by_participant_id[
            same_participant_other_tab.id
        ]
        assert [type(payload) for payload in mirrored_direct_payloads] == [
            ThreadCreated,
            ThreadUpdated,
            ThreadDeleted,
        ]
        assert (
            channel.direct_payloads_by_participant_id.get(different_participant.id)
            is None
        )
        assert [
            message.data
            for message in channel.received
            if isinstance(message.data, (ThreadCreated, ThreadUpdated, ThreadDeleted))
        ] == []
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_route_sends_unflushed_storage_before_thread_open_reaches_channel() -> (
    None
):
    thread_storage = _LifecycleThreadStorage(path="/threads/created")
    unflushed = AgentUsageUpdated(
        type=AGENT_EVENT_USAGE_UPDATED,
        thread_id="/threads/created",
        usage={},
        context_window=AgentContextWindowUsage(used_tokens=0),
    )
    thread_storage.messages.append(unflushed)
    supervisor = _StorageThreadCreatingSupervisor(thread_storage=thread_storage)
    channel = _RecordingChannel()
    supervisor.add_channel(channel)

    await supervisor.start()
    try:
        open_thread = OpenThread(
            type=AGENT_MESSAGE_THREAD_OPEN,
            thread_id="/threads/created",
        )

        await supervisor.route(Message(data=open_thread))

        assert thread_storage.flushed == 0
        await _wait_for(lambda: len(channel.received) == 2)
        assert channel.received[0].data is unflushed
        assert channel.received[1].data is open_thread
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_thread_open_with_load_replays_stored_messages_since_turn() -> (
    None
):
    thread_storage = _LifecycleThreadStorage(path="/threads/created")
    old_message = AgentTextContentDelta(
        type=AGENT_EVENT_TEXT_CONTENT_DELTA,
        thread_id="/threads/created",
        turn_id="turn-old",
        item_id="old",
        text="old",
    )
    replay_message = AgentTextContentDelta(
        type=AGENT_EVENT_TEXT_CONTENT_DELTA,
        thread_id="/threads/created",
        turn_id="turn-new",
        item_id="new",
        text="new",
        created_at="2026-05-28T16:11:48.538Z",
    )
    thread_storage.messages.extend([old_message, replay_message])
    room = _DownloadRecordingRoom()
    supervisor = _ReplayThreadCreatingSupervisor(
        room=room,
        thread_storage=thread_storage,
    )
    channel = _RecordingChannel()
    supervisor.add_channel(channel)

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/created",
                    load=True,
                    since_turn="turn-new",
                )
            )
        )

        await _wait_for(lambda: len(channel.received) >= 3)
        received_messages = [
            message.data for message in channel.received if message.source is not None
        ]
        replayed_text_messages = [
            message
            for message in received_messages
            if isinstance(message, AgentTextContentDelta)
        ]
        assert {message.item_id for message in replayed_text_messages} == {"new"}
        loaded_messages = [
            message
            for message in received_messages
            if isinstance(message, ThreadLoaded)
        ]
        assert len(loaded_messages) == 1
        assert loaded_messages[0].type == AGENT_EVENT_THREAD_LOADED
        assert loaded_messages[0].since_turn == "turn-new"
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_thread_open_with_load_sends_replay_directly_before_loaded() -> (
    None
):
    thread_storage = _LifecycleThreadStorage(path="/threads/created")
    replay_message = AgentTextContentDelta(
        type=AGENT_EVENT_TEXT_CONTENT_DELTA,
        thread_id="/threads/created",
        turn_id="turn-new",
        item_id="new",
        text="new",
    )
    thread_storage.messages.append(replay_message)
    room = _DownloadRecordingRoom()
    supervisor = _ReplayThreadCreatingSupervisor(
        room=room,
        thread_storage=thread_storage,
    )
    channel = _ThreadOpenResponseChannel()
    supervisor.add_channel(channel)
    participant = _ThreadParticipant(name="User", participant_id="user-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/created",
                    load=True,
                ),
                sender=participant,
                source=channel,
            )
        )

        await _wait_for(lambda: len(channel.direct_payloads) >= 2)
        assert isinstance(channel.direct_payloads[0], AgentTextContentDelta)
        assert channel.direct_payloads[0].item_id == replay_message.item_id
        assert channel.direct_payloads[0].created_at == replay_message.created_at
        assert isinstance(channel.direct_payloads[1], ThreadLoaded)
        assert [
            message.data
            for message in channel.received
            if isinstance(message.data, (AgentTextContentDelta, ThreadLoaded))
        ] == []
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_tracks_open_threads_by_client_until_last_close() -> (
    None
):
    supervisor = _OpenCloseThreadCreatingSupervisor()
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    first_client = _ThreadParticipant(name="First", participant_id="client-1")
    second_client = _ThreadParticipant(name="Second", participant_id="client-2")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=first_client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/shared",
                ),
            )
        )
        await supervisor.route(
            Message(
                sender=second_client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)
        assert process.received[0].sender is first_client
        assert process.received[0].data.type == AGENT_MESSAGE_THREAD_OPEN

        await supervisor.route(
            Message(
                sender=first_client,
                data=CloseThread(
                    type=AGENT_MESSAGE_THREAD_CLOSE,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert process.state == "started"
        assert process in supervisor.processes
        assert process.stopped == 0

        await supervisor.route(
            Message(
                sender=second_client,
                data=CloseThread(
                    type=AGENT_MESSAGE_THREAD_CLOSE,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert process.state == "stopped"
        assert process not in supervisor.processes
        assert process.stopped == 1
        assert supervisor._open_client_ids_by_thread_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
        assert [message.data.type for message in channel.received] == [
            AGENT_MESSAGE_THREAD_OPEN,
            AGENT_MESSAGE_THREAD_OPEN,
            AGENT_MESSAGE_THREAD_CLOSE,
            AGENT_MESSAGE_THREAD_CLOSE,
        ]
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_forwards_load_open_for_already_open_thread() -> None:
    supervisor = _OpenCloseThreadCreatingSupervisor()
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)

        await supervisor.route(
            Message(
                sender=client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/shared",
                    load=True,
                ),
            )
        )

        await _wait_for(lambda: len(process.received) == 2)
        assert process.received[1].data.type == AGENT_MESSAGE_THREAD_OPEN
        assert isinstance(process.received[1].data, OpenThread)
        assert process.received[1].data.load is True
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_implicitly_opens_thread_on_turn_start_until_disconnect() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantConnect(
                    type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                    participant_id=client.id,
                ),
            )
        )
        await supervisor.route(
            Message(
                sender=client,
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/implicit",
                    content=[AgentTextContent(type="text", text="hello")],
                ),
            )
        )

        assert supervisor._open_client_ids_by_thread_id == {
            "/threads/implicit": {"client-1"}
        }
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)

        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantDisconnect(
                    type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                    participant_id=client.id,
                ),
            )
        )

        assert process.state == "stopped"
        assert supervisor._open_client_ids_by_thread_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
        assert supervisor._participant_connection_counts_by_client_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_implicitly_opens_thread_on_turn_steer_until_close() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=client,
                data=TurnSteer(
                    type=AGENT_MESSAGE_TURN_STEER,
                    thread_id="/threads/implicit",
                    turn_id="turn-1",
                    content=[AgentTextContent(type="text", text="keep going")],
                ),
            )
        )

        assert supervisor._open_client_ids_by_thread_id == {
            "/threads/implicit": {"client-1"}
        }
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)

        await supervisor.route(
            Message(
                sender=client,
                data=CloseThread(
                    type=AGENT_MESSAGE_THREAD_CLOSE,
                    thread_id="/threads/implicit",
                ),
            )
        )

        assert process.state == "stopped"
        assert supervisor._open_client_ids_by_thread_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_participant_isolation_rejects_other_participant_turn_start() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor(thread_isolation="participant")
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    first_client = _ThreadParticipant(name="First", participant_id="client-1")
    second_client = _ThreadParticipant(name="Second", participant_id="client-2")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=first_client,
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/private",
                    content=[AgentTextContent(type="text", text="hello")],
                ),
            )
        )
        assert supervisor.thread_namespace(thread_id="/threads/private") == "First"
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)

        await supervisor.route(
            Message(
                sender=second_client,
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/private",
                    content=[AgentTextContent(type="text", text="hello from second")],
                ),
            )
        )

        await _wait_for(
            lambda: any(
                message.data.type == AGENT_EVENT_TURN_START_REJECTED
                for message in channel.received
            )
        )
        rejected = [
            message.data
            for message in channel.received
            if isinstance(message.data, TurnStartRejected)
        ][0]
        assert rejected.error.code == "thread_not_available"
        assert len(process.received) == 1
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_global_isolation_allows_shared_thread_access() -> None:
    supervisor = _ThreadCreatingSupervisor(thread_isolation="global")
    first_client = _ThreadParticipant(name="First", participant_id="client-1")
    second_client = _ThreadParticipant(name="Second", participant_id="client-2")

    await supervisor.start()
    try:
        for participant, text in [
            (first_client, "hello"),
            (second_client, "hello from second"),
        ]:
            await supervisor.route(
                Message(
                    sender=participant,
                    data=TurnStart(
                        type=AGENT_MESSAGE_TURN_START,
                        thread_id="/threads/shared",
                        content=[AgentTextContent(type="text", text=text)],
                    ),
                )
            )

        assert supervisor.thread_namespace(thread_id="/threads/shared") is None
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 2)
        assert [message.sender for message in process.received] == [
            first_client,
            second_client,
        ]
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_participant_isolation_blocks_open_from_other_participant() -> (
    None
):
    supervisor = _OpenCloseThreadCreatingSupervisor(thread_isolation="participant")
    channel = _RecordingChannel()
    supervisor.add_channel(channel)
    first_client = _ThreadParticipant(name="First", participant_id="client-1")
    second_client = _ThreadParticipant(name="Second", participant_id="client-2")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=first_client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/private",
                ),
            )
        )
        assert supervisor.thread_namespace(thread_id="/threads/private") == "First"
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]
        await _wait_for(lambda: len(process.received) == 1)

        await supervisor.route(
            Message(
                sender=second_client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/private",
                ),
            )
        )
        await asyncio.sleep(0)

        assert len(process.received) == 1
        assert [message.data.type for message in channel.received] == [
            AGENT_MESSAGE_THREAD_OPEN
        ]
        assert supervisor._open_client_ids_by_thread_id == {
            "/threads/private": {"client-1"}
        }
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_connection_lifecycle_disconnect_closes_all_implicitly_open_threads() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        for thread_id in ["/threads/one", "/threads/two"]:
            await supervisor.route(
                Message(
                    sender=client,
                    data=TurnStart(
                        type=AGENT_MESSAGE_TURN_START,
                        thread_id=thread_id,
                        content=[AgentTextContent(type="text", text=thread_id)],
                    ),
                )
            )

        assert supervisor._open_thread_ids_by_client_id == {
            "client-1": {"/threads/one", "/threads/two"}
        }
        assert supervisor._open_client_ids_by_thread_id == {
            "/threads/one": {"client-1"},
            "/threads/two": {"client-1"},
        }
        assert len(supervisor.created_processes) == 2

        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantDisconnect(
                    type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                    participant_id=client.id,
                ),
            )
        )

        assert all(
            process.state == "stopped" for process in supervisor.created_processes
        )
        assert supervisor.processes == []
        assert supervisor._open_thread_ids_by_client_id == {}
        assert supervisor._open_client_ids_by_thread_id == {}
        assert supervisor._participants_by_client_id == {}
        assert supervisor._participant_connection_counts_by_client_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_connection_lifecycle_expected_close_keeps_other_client_thread_open() -> (
    None
):
    supervisor = _OpenCloseThreadCreatingSupervisor()
    first_client = _ThreadParticipant(name="First", participant_id="client-1")
    second_client = _ThreadParticipant(name="Second", participant_id="client-2")

    await supervisor.start()
    try:
        for client in [first_client, second_client]:
            await supervisor.route(
                Message(
                    sender=client,
                    data=OpenThread(
                        type=AGENT_MESSAGE_THREAD_OPEN,
                        thread_id="/threads/shared",
                    ),
                )
            )

        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]

        await supervisor.route(
            Message(
                sender=first_client,
                data=CloseThread(
                    type=AGENT_MESSAGE_THREAD_CLOSE,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert process.state == "started"
        assert supervisor._open_client_ids_by_thread_id == {
            "/threads/shared": {"client-2"}
        }
        assert supervisor._open_thread_ids_by_client_id == {
            "client-2": {"/threads/shared"}
        }

        await supervisor.route(
            Message(
                sender=second_client,
                data=CloseThread(
                    type=AGENT_MESSAGE_THREAD_CLOSE,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert process.state == "stopped"
        assert supervisor._open_client_ids_by_thread_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_connection_lifecycle_disconnect_uses_connection_ref_counts() -> None:
    supervisor = _OpenCloseThreadCreatingSupervisor()
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        for _ in range(2):
            await supervisor.route(
                Message(
                    sender=client,
                    data=ParticipantConnect(
                        type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                        participant_id=client.id,
                    ),
                )
            )
        await supervisor.route(
            Message(
                sender=client,
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="/threads/shared",
                ),
            )
        )

        assert supervisor._participant_connection_counts_by_client_id == {"client-1": 2}
        assert len(supervisor.created_processes) == 1
        process = supervisor.created_processes[0]

        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantDisconnect(
                    type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                    participant_id=client.id,
                ),
            )
        )

        assert process.state == "started"
        assert supervisor._participant_connection_counts_by_client_id == {"client-1": 1}
        assert supervisor._open_thread_ids_by_client_id == {
            "client-1": {"/threads/shared"}
        }

        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantDisconnect(
                    type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                    participant_id=client.id,
                ),
            )
        )

        assert process.state == "stopped"
        assert supervisor._participant_connection_counts_by_client_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
        assert supervisor._open_client_ids_by_thread_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_connection_lifecycle_disconnect_without_open_threads_does_not_leak_tracking() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()
    client = _ThreadParticipant(name="Client", participant_id="client-1")

    await supervisor.start()
    try:
        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantConnect(
                    type=AGENT_MESSAGE_PARTICIPANT_CONNECT,
                    participant_id=client.id,
                ),
            )
        )
        await supervisor.route(
            Message(
                sender=client,
                data=ParticipantDisconnect(
                    type=AGENT_MESSAGE_PARTICIPANT_DISCONNECT,
                    participant_id=client.id,
                ),
            )
        )

        assert supervisor.created_processes == []
        assert supervisor._participants_by_client_id == {}
        assert supervisor._participant_connection_counts_by_client_id == {}
        assert supervisor._open_thread_ids_by_client_id == {}
        assert supervisor._open_client_ids_by_thread_id == {}
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_process_enriches_received_messages_with_sender_name() -> None:
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    supervisor = AgentSupervisor()
    process = _StorageThreadRecordingProcess(
        thread_id="thread-1",
        thread_storage=thread_storage,
    )
    supervisor.add_process(process)

    await supervisor.start()
    try:
        process.send(
            Message(
                sender=_ThreadParticipant(
                    name="Jesse",
                    participant_id="caller-id",
                ),
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[],
                ),
            )
        )

        await _wait_for(lambda: len(process.received) == 1)
    finally:
        await supervisor.stop()

    assert process.received[0].data.sender_name == "Jesse"


@pytest.mark.asyncio
async def test_agent_supervisor_rejects_turn_start_when_thread_process_creation_fails() -> (
    None
):
    supervisor = _FailingThreadCreatingSupervisor()
    channel = _RecordingChannel(handled_type=AGENT_EVENT_TURN_START_REJECTED)
    supervisor.add_channel(channel)

    await supervisor.start()
    try:
        await asyncio.wait_for(channel.start_event.wait(), timeout=1)
        turn_start = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="/threads/not-dataset",
            content=[],
        )

        await supervisor.route(Message(data=turn_start, source=channel))
        await asyncio.wait_for(channel.message_event.wait(), timeout=1)

        assert supervisor.state == "started"
        assert supervisor.processes == []
        assert len(channel.received) == 1
        rejection = channel.received[0].data
        assert isinstance(rejection, TurnStartRejected)
        assert rejection.thread_id == turn_start.thread_id
        assert rejection.source_message_id == turn_start.message_id
        assert rejection.error.code == "thread_process_creation_failed"
    finally:
        await supervisor.stop()

    assert supervisor.state == "stopped"


@pytest.mark.asyncio
async def test_agent_supervisor_rejects_steer_when_thread_process_creation_fails() -> (
    None
):
    supervisor = _FailingThreadCreatingSupervisor()
    channel = _RecordingChannel(handled_type=AGENT_EVENT_TURN_STEER_REJECTED)
    supervisor.add_channel(channel)

    await supervisor.start()
    try:
        await asyncio.wait_for(channel.start_event.wait(), timeout=1)
        turn_steer = TurnSteer(
            type=AGENT_MESSAGE_TURN_STEER,
            thread_id="/threads/not-dataset",
            turn_id="turn-1",
            content=[],
        )

        await supervisor.route(Message(data=turn_steer, source=channel))
        await asyncio.wait_for(channel.message_event.wait(), timeout=1)

        assert supervisor.state == "started"
        assert supervisor.processes == []
        assert len(channel.received) == 1
        rejection = channel.received[0].data
        assert isinstance(rejection, TurnSteerRejected)
        assert rejection.thread_id == turn_steer.thread_id
        assert rejection.turn_id == turn_steer.turn_id
        assert rejection.source_message_id == turn_steer.message_id
        assert rejection.error.code == "thread_process_creation_failed"
    finally:
        await supervisor.stop()

    assert supervisor.state == "stopped"


@pytest.mark.asyncio
async def test_agent_supervisor_stop_ignores_queued_message_with_thread_process_creation_failure() -> (
    None
):
    supervisor = _FailingThreadCreatingSupervisor()
    channel = _RecordingChannel(handled_type=AGENT_EVENT_TURN_START_REJECTED)
    supervisor.add_channel(channel)

    await supervisor.start()
    await asyncio.wait_for(channel.start_event.wait(), timeout=1)
    turn_start = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="/threads/not-dataset",
        content=[],
    )
    supervisor.send(Message(data=turn_start, source=channel))
    await asyncio.sleep(0)
    await asyncio.wait_for(channel.message_event.wait(), timeout=1)

    await supervisor.stop()

    assert supervisor.state == "stopped"
    assert len(channel.received) == 1


@pytest.mark.asyncio
async def test_agent_supervisor_does_not_echo_message_to_origin_channel() -> None:
    supervisor = AgentSupervisor()
    source_channel = _RecordingChannel(handled_type="work")
    other_channel = _RecordingChannel(handled_type="work")
    process = _RecordingProcess(handled_type="work")
    supervisor.add_channel(source_channel)
    supervisor.add_channel(other_channel)
    supervisor.add_process(process)

    await supervisor.start()
    await asyncio.wait_for(source_channel.start_event.wait(), timeout=1)
    await asyncio.wait_for(other_channel.start_event.wait(), timeout=1)
    await asyncio.wait_for(process.start_event.wait(), timeout=1)

    source_channel.emit(
        sender=None,
        payload=AgentMessage(type="work", thread_id="thread-1"),
    )

    await asyncio.wait_for(other_channel.message_event.wait(), timeout=1)
    await asyncio.wait_for(process.message_event.wait(), timeout=1)
    await asyncio.sleep(0.05)

    assert source_channel.received == []
    assert len(other_channel.received) == 1
    assert len(process.received) == 1
    assert other_channel.received[0].source is source_channel
    assert process.received[0].source is source_channel
    assert other_channel.received[0].data.type == "work"
    assert process.received[0].data.type == "work"

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_routes_process_emitted_events_to_channels() -> None:
    supervisor = AgentSupervisor()
    channel = _RecordingChannel(handled_type="event")
    process = _EmittingProcess(handled_type="work", emitted_type="event")
    supervisor.add_channel(channel)
    supervisor.add_process(process)

    await supervisor.start()
    await asyncio.wait_for(channel.start_event.wait(), timeout=1)
    await asyncio.wait_for(process.start_event.wait(), timeout=1)

    supervisor.send(
        Message(data=_PayloadMessage(type="work", thread_id="thread-1", payload="ok"))
    )

    await asyncio.wait_for(process.emitted_event.wait(), timeout=1)
    await asyncio.wait_for(channel.message_event.wait(), timeout=1)

    assert [message.data.model_dump(mode="json") for message in channel.received] == [
        {
            "type": "event",
            "thread_id": "thread-1",
            "payload": "ok",
            "message_id": channel.received[0].data.message_id,
            "sender_name": None,
        }
    ]

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_logs_route_failures_without_stopping_children(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = _FailingTurnValidationSupervisor()
    channel = _LifecycleChannel()
    supervisor.add_channel(channel)

    await supervisor.start()
    try:
        await asyncio.wait_for(channel.start_event.wait(), timeout=1)
        with caplog.at_level("ERROR", logger="agent-process"):
            supervisor.send(
                Message(
                    sender=_ThreadParticipant(name="Client", participant_id="client-1"),
                    data=StartThread(
                        type=AGENT_MESSAGE_THREAD_START,
                        content=[AgentTextContent(type="text", text="hello")],
                    ),
                )
            )
            await asyncio.wait_for(supervisor.validation_started.wait(), timeout=1)
            await _wait_for(
                lambda: (
                    "agent supervisor failed while routing meshagent.agent.thread.start message"
                    in caplog.text
                )
            )

        assert supervisor.state == "started"
        assert channel.state == "started"
        assert channel.supervisor is supervisor

    finally:
        if supervisor.state == "started":
            await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_start_fails_when_channel_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = AgentSupervisor()
    failing_channel = _FailingStartChannel()
    healthy_channel = _LifecycleChannel()
    supervisor.add_channel(failing_channel)
    supervisor.add_channel(healthy_channel)

    with caplog.at_level("ERROR", logger="agent-process"):
        with pytest.raises(RuntimeError, match="boom"):
            await supervisor.start()

    assert failing_channel.state == "failed"
    assert failing_channel.supervisor is None
    assert healthy_channel.state == "stopped"
    assert healthy_channel.supervisor is None
    assert supervisor.state == "failed"
    assert "agent supervisor failed during start" in caplog.text


@pytest.mark.asyncio
async def test_agent_supervisor_start_fails_when_process_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = AgentSupervisor()
    failing_process = _FailingStartProcess()
    healthy_process = _RecordingProcess(handled_type="work")
    supervisor.add_process(failing_process)
    supervisor.add_process(healthy_process)

    with caplog.at_level("ERROR", logger="agent-process"):
        with pytest.raises(RuntimeError, match="boom"):
            await supervisor.start()

    assert failing_process.state == "failed"
    assert failing_process.supervisor is None
    assert healthy_process.state == "stopped"
    assert healthy_process.supervisor is None
    assert supervisor.state == "failed"
    assert "agent supervisor failed during start" in caplog.text


@pytest.mark.asyncio
async def test_agent_process_start_failure_sets_failed_state_and_reraises() -> None:
    supervisor = AgentSupervisor()
    process = _FailingStartProcess()

    with pytest.raises(RuntimeError, match="boom"):
        await process.start(supervisor)

    assert process.state == "failed"
    assert process.supervisor is None


@pytest.mark.asyncio
async def test_agent_process_stop_is_serialized_and_on_stop_runs_once() -> None:
    supervisor = AgentSupervisor()
    process = _BlockingStopProcess()

    await process.start(supervisor)
    await asyncio.wait_for(process.started_event.wait(), timeout=1)

    first_stop = asyncio.create_task(process.stop(supervisor))
    await asyncio.wait_for(process.stop_started.wait(), timeout=1)

    second_stop = asyncio.create_task(process.stop(supervisor))
    await asyncio.sleep(0)

    process.release_stop.set()
    await first_stop

    with pytest.raises(ValueError, match="not started"):
        await second_stop

    assert process.on_stop_calls == 1


@pytest.mark.asyncio
async def test_agent_process_send_drops_messages_during_shutdown() -> None:
    supervisor = AgentSupervisor()
    process = _BlockingStopProcess()

    await process.start(supervisor)
    await asyncio.wait_for(process.started_event.wait(), timeout=1)

    stop_task = asyncio.create_task(process.stop(supervisor))
    await asyncio.wait_for(process.stop_started.wait(), timeout=1)

    process.send(
        Message(data=_PayloadMessage(type="work", thread_id="thread-1", payload="late"))
    )
    await asyncio.sleep(0.05)

    assert process.received == []

    process.release_stop.set()
    await stop_task


@pytest.mark.asyncio
async def test_agent_supervisor_send_drops_messages_during_shutdown() -> None:
    supervisor = AgentSupervisor()
    process = _BlockingStopProcess()
    supervisor.add_process(process)

    await supervisor.start()
    await asyncio.wait_for(process.started_event.wait(), timeout=1)

    stop_task = asyncio.create_task(supervisor.stop())
    await asyncio.wait_for(process.stop_started.wait(), timeout=1)

    supervisor.send(
        Message(data=_PayloadMessage(type="work", thread_id="thread-1", payload="late"))
    )
    await asyncio.sleep(0.05)

    assert process.received == []

    process.release_stop.set()
    await stop_task


@pytest.mark.asyncio
async def test_agent_supervisor_creates_and_reuses_thread_processes_for_turn_start() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()

    await supervisor.start()

    supervisor.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "first"}],
            )
        )
    )

    await _wait_for(lambda: len(supervisor.created_processes) == 1)
    process_one = supervisor.created_processes[0]
    await _wait_for(lambda: len(process_one.received) == 1)
    uuid.UUID(process_one.received[0].data.message_id)

    supervisor.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "second"}],
            )
        )
    )

    await _wait_for(lambda: len(process_one.received) == 2)
    assert len(supervisor.created_processes) == 1

    supervisor.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-2",
                content=[{"type": "text", "text": "other"}],
            )
        )
    )

    await _wait_for(lambda: len(supervisor.created_processes) == 2)
    process_two = supervisor.created_processes[1]
    await _wait_for(lambda: len(process_two.received) == 1)

    assert process_one.thread_id == "thread-1"
    assert process_two.thread_id == "thread-2"
    assert len(supervisor.processes) == 2

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_creates_thread_processes_for_clear_thread() -> None:
    supervisor = _ThreadCreatingSupervisor()

    await supervisor.start()

    supervisor.send(
        Message(
            data=ClearThread(
                type=AGENT_MESSAGE_THREAD_CLEAR,
                thread_id="thread-1",
            )
        )
    )

    await _wait_for(lambda: len(supervisor.created_processes) == 1)
    process = supervisor.created_processes[0]
    await _wait_for(lambda: len(process.received) == 1)

    clear_message = process.received[0].data
    assert isinstance(clear_message, ClearThread)
    assert clear_message.thread_id == "thread-1"
    uuid.UUID(clear_message.message_id)

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_routes_turn_steer_to_thread_process_for_rejection() -> (
    None
):
    supervisor = _ThreadCreatingSupervisor()

    await supervisor.start()

    supervisor.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                thread_id="thread-1",
                turn_id="missing-turn",
                content=[{"type": "text", "text": "translate"}],
            )
        )
    )

    await _wait_for(lambda: len(supervisor.created_processes) == 1)
    process = supervisor.created_processes[0]
    await _wait_for(lambda: len(process.received) == 1)

    steer_message = process.received[0].data
    assert isinstance(steer_message, TurnSteer)
    assert steer_message.thread_id == "thread-1"
    assert steer_message.turn_id == "missing-turn"

    await supervisor.stop()


def test_turn_start_rejects_file_content_without_url() -> None:
    with pytest.raises(ValidationError, match="url"):
        TurnStart.model_validate(
            {
                "type": AGENT_MESSAGE_TURN_START,
                "thread_id": "thread-1",
                "content": [{"type": "file"}],
            }
        )


def test_turn_start_defaults_message_id_to_uuid() -> None:
    turn_start = TurnStart.model_validate(
        {
            "type": AGENT_MESSAGE_TURN_START,
            "thread_id": "thread-1",
            "content": [{"type": "text", "text": "hello"}],
        }
    )

    uuid.UUID(turn_start.message_id)


def test_turn_steer_requires_thread_id() -> None:
    with pytest.raises(ValidationError, match="thread_id"):
        TurnSteer.model_validate(
            {
                "type": AGENT_MESSAGE_TURN_STEER,
                "turn_id": "turn-1",
                "content": [{"type": "text", "text": "continue"}],
            }
        )


@pytest.mark.parametrize(
    ("message_type", "model"),
    [
        (AGENT_MESSAGE_TOOL_CALL_APPROVE, ApproveAgentToolCall),
        (AGENT_MESSAGE_TOOL_CALL_REJECT, RejectAgentToolCall),
    ],
)
def test_tool_call_approval_messages_require_thread_id(
    message_type: str,
    model: type[ApproveAgentToolCall] | type[RejectAgentToolCall],
) -> None:
    with pytest.raises(ValidationError, match="thread_id"):
        model.model_validate(
            {
                "type": message_type,
                "turn_id": "turn-1",
                "item_id": "approval-1",
            }
        )


def test_realtime_audio_chunk_protocol_only_carries_audio_data_and_format() -> None:
    chunk = AgentRealtimeAudioChunk(
        type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
        thread_id="thread-1",
        message_id="audio-chunk-1",
        data=b"pcm",
        format=AgentAudioFormat(type="audio/pcm", sample_rate=24000),
    )

    payload = chunk.model_dump(mode="json")
    assert payload["format"] == {
        "type": "audio/pcm",
        "sample_rate": 24000,
        "bitrate": None,
    }
    assert "provider" not in payload
    assert "model" not in payload
    assert "voice" not in payload
    assert "output_modalities" not in payload
    assert "final" not in payload
    assert "input_format" not in payload
    assert "status_detail" not in payload


def test_agent_tool_call_ended_serializes_result_and_error() -> None:
    event = AgentToolCallEnded(
        type=AGENT_EVENT_TOOL_CALL_ENDED,
        thread_id="thread-1",
        turn_id="turn-1",
        item_id="tool-1",
        result=TextContent(text="ok"),
        error=AgentError(message="failed", code="tool_failed"),
    )

    payload = event.model_dump(mode="json")

    uuid.UUID(payload["message_id"])
    assert payload["result"] == {"type": "text", "text": "ok"}
    assert payload["error"] == {
        "message": "failed",
        "code": "tool_failed",
    }


def _client_toolkit_description() -> ClientToolkitDescription:
    return ClientToolkitDescription(
        name="pick_color",
        description="Pick a color.",
        input_schema={
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
            "additionalProperties": False,
        },
    )


async def _start_client_toolkit_process(
    *,
    adapter: _ClientToolkitInvokingLLMAdapter,
    sender: Participant | None,
    connect_sender: bool = True,
    client_tool_call_timeout_seconds: float | None = None,
) -> tuple[_RecordingSupervisor, _ParticipantRoutingChannel, LLMAgentProcess]:
    supervisor = _RecordingSupervisor()
    channel = _ParticipantRoutingChannel()
    supervisor.channels.append(channel)
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )
    supervisor.add_process(process)
    await process.start(supervisor)
    if client_tool_call_timeout_seconds is not None:
        process._client_tool_call_timeout_seconds = client_tool_call_timeout_seconds
    if sender is not None and connect_sender:
        supervisor._track_participant_connected(
            participant_id=sender.id,
            sender=sender,
        )
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "use client tool"}],
                client_toolkits=[_client_toolkit_description()],
            ),
            sender=sender,
        )
    )
    return supervisor, channel, process


@pytest.mark.asyncio
async def test_client_toolkit_request_targets_turn_start_participant() -> None:
    sender = _ThreadParticipant(name="User", participant_id="client-1")
    other = _ThreadParticipant(name="Other", participant_id="client-2")
    adapter = _ClientToolkitInvokingLLMAdapter()
    supervisor, channel, process = await _start_client_toolkit_process(
        adapter=adapter,
        sender=sender,
    )
    supervisor._track_participant_connected(participant_id=other.id, sender=other)

    try:
        await asyncio.wait_for(adapter.tool_request_started.wait(), timeout=1)
        await _wait_for(
            lambda: (
                len(channel.direct_payloads_by_participant_id.get(sender.id, [])) == 1
            )
        )

        assert channel.direct_payloads_by_participant_id.get(other.id, []) == []
        request = channel.direct_payloads_by_participant_id[sender.id][0]
        assert isinstance(request, AgentClientToolCallRequested)
        assert request.type == AGENT_EVENT_CLIENT_TOOL_CALL_REQUESTED
        assert request.toolkit == "client"
        assert request.tool == "pick_color"
        assert request.arguments == {"color": "blue"}

        process.send(
            Message(
                data=AgentClientToolCallResponse(
                    type=AGENT_MESSAGE_CLIENT_TOOL_CALL_RESPONSE,
                    thread_id="thread-1",
                    turn_id=request.turn_id,
                    request_id=request.request_id,
                    response=JsonContent(json={"selected": "blue"}),
                ),
                sender=sender,
            )
        )

        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        assert isinstance(adapter.result, JsonContent)
        assert adapter.result.json == {"selected": "blue"}
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_client_toolkit_fails_fast_when_participant_is_offline() -> None:
    sender = _ThreadParticipant(name="User", participant_id="client-1")
    adapter = _ClientToolkitInvokingLLMAdapter()
    supervisor, channel, process = await _start_client_toolkit_process(
        adapter=adapter,
        sender=sender,
        connect_sender=False,
    )

    try:
        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        assert channel.direct_payloads_by_participant_id == {}
        assert isinstance(adapter.result, ErrorContent)
        assert adapter.result.text == "client toolkit participant is not connected"
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_client_toolkit_disconnect_cancels_pending_call() -> None:
    sender = _ThreadParticipant(name="User", participant_id="client-1")
    adapter = _ClientToolkitInvokingLLMAdapter()
    supervisor, channel, process = await _start_client_toolkit_process(
        adapter=adapter,
        sender=sender,
    )

    try:
        await asyncio.wait_for(adapter.tool_request_started.wait(), timeout=1)
        await _wait_for(
            lambda: (
                len(channel.direct_payloads_by_participant_id.get(sender.id, [])) == 1
            )
        )
        await supervisor._track_participant_disconnected(participant_id=sender.id)

        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        assert isinstance(adapter.result, ErrorContent)
        assert "participant_disconnected" in adapter.result.text
        payloads = channel.direct_payloads_by_participant_id[sender.id]
        assert len(payloads) == 1
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_client_toolkit_timeout_cancels_pending_call() -> None:
    sender = _ThreadParticipant(name="User", participant_id="client-1")
    adapter = _ClientToolkitInvokingLLMAdapter()
    supervisor, channel, process = await _start_client_toolkit_process(
        adapter=adapter,
        sender=sender,
        client_tool_call_timeout_seconds=0.01,
    )

    try:
        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        assert isinstance(adapter.result, ErrorContent)
        assert adapter.result.text == "client toolkit call timed out"
        payloads = channel.direct_payloads_by_participant_id[sender.id]
        assert isinstance(payloads[1], AgentClientToolCallCancelled)
        assert payloads[1].reason == "timeout"
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_client_toolkit_turn_rejects_steering_from_other_participants() -> None:
    sender = _ThreadParticipant(name="User", participant_id="client-1")
    other = _ThreadParticipant(name="Other", participant_id="client-2")
    adapter = _ClientToolkitInvokingLLMAdapter()
    supervisor, _channel, process = await _start_client_toolkit_process(
        adapter=adapter,
        sender=sender,
    )

    try:
        await asyncio.wait_for(adapter.tool_request_started.wait(), timeout=1)
        turn_started = next(
            message.data
            for message in supervisor.sent
            if message.data.type == AGENT_EVENT_TURN_STARTED
        )
        assert isinstance(turn_started, TurnStarted)
        process.send(
            Message(
                data=TurnSteer(
                    type=AGENT_MESSAGE_TURN_STEER,
                    thread_id="thread-1",
                    turn_id=turn_started.turn_id,
                    content=[{"type": "text", "text": "steer"}],
                ),
                sender=other,
            )
        )

        await _wait_for(
            lambda: any(
                message.data.type == AGENT_EVENT_TURN_STEER_REJECTED
                for message in supervisor.sent
            )
        )
        rejection = next(
            message.data
            for message in supervisor.sent
            if message.data.type == AGENT_EVENT_TURN_STEER_REJECTED
        )
        assert isinstance(rejection, TurnSteerRejected)
        assert rejection.error.code == "turn_owned_by_participant"
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_traces_turn_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_tracer = _RecordedTracer()
    monkeypatch.setattr(process_module, "tracer", recorded_tracer)

    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()

    async def _initialize_session() -> AgentSessionContext:
        session = AgentSessionContext(system_role=None)
        session.instructions = "initial secret instructions"
        return session

    async def _load_turn_instructions(
        sender: Participant | None,
    ) -> str:
        del sender
        return "runtime secret instructions"

    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        session_initializer=_initialize_session,
        turn_instructions_provider=_load_turn_instructions,
        toolkits=[Toolkit(name="storage", tools=[])],
    )

    await process.start(supervisor)

    turn_start_message_id = "00000000-0000-0000-0000-000000000001"
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=turn_start_message_id,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )
    await process.stop(supervisor)

    span_names = [span.name for span in recorded_tracer.spans]
    assert span_names == [
        "agent.turn",
        "agent.turn.context.load",
        "agent.turn.context.initialize",
        "agent.turn.context.restore_hooks",
        "agent.turn.context.start",
        "agent.turn.rules.load",
        "agent.turn.toolkits.build",
        "agent.turn.llm",
    ]

    turn_span = next(
        span for span in recorded_tracer.spans if span.name == "agent.turn"
    )
    assert turn_span.attributes["thread_id"] == "thread-1"
    assert turn_span.attributes["source_message_id"] == turn_start_message_id
    assert turn_span.attributes["queued_message_count"] == 1
    assert turn_span.attributes["model"] == "gpt-test"
    assert turn_span.attributes["error"] is False

    for span in recorded_tracer.spans:
        for value in span.attributes.values():
            assert "secret instructions" not in str(value)


@pytest.mark.asyncio
async def test_llm_agent_process_passes_turn_id_to_restore_session_context() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        process_cls=_RestoringLLMAgentProcess,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    turn_start_message_id = "00000000-0000-0000-0000-000000000002"
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=turn_start_message_id,
                thread_id="thread-1",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)) == 1
    )

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    turn_id = started_payload["turn_id"]
    assert started_payload["thread_id"] == "thread-1"
    assert started_payload["source_message_id"] == turn_start_message_id
    assert isinstance(turn_id, str)
    assert len(process.restore_calls) == 1
    assert process.restore_calls[0]["turn_id"] == turn_id
    assert process.restore_calls[0]["session_context"] is adapter.session
    assert adapter.calls[0]["caller"] is room.local_participant

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_starts_adapter_session_before_create_response() -> (
    None
):
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)

    assert adapter.call_order == ["start_session", "create_response"]
    assert len(adapter.start_session_calls) == 1
    assert adapter.start_session_calls[0]["context"] is adapter.session
    assert adapter.start_session_calls[0]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert callable(adapter.start_session_calls[0]["event_handler"])

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_starts_adapter_session_on_thread_open() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )

    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    assert adapter.call_order == ["start_session"]
    assert len(adapter.start_session_calls) == 1
    assert adapter.start_session_calls[0]["context"] is adapter.session
    assert adapter.start_session_calls[0]["event_handler"] is None

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_thread_open_during_active_turn_does_not_block_next_turn() -> (
    None
):
    class _BlockingOpenAdapter(_RecordingLLMAdapter):
        def __init__(self) -> None:
            super().__init__(session=_LifecycleSession())
            self.release_response = asyncio.Event()
            self.open_session_started = asyncio.Event()

        async def start_session(
            self,
            *,
            context: AgentSessionContext,
            event_handler=None,
        ) -> None:
            await super().start_session(context=context, event_handler=event_handler)
            if event_handler is None:
                self.open_session_started.set()
                await asyncio.Event().wait()

        async def create_response(self, **kwargs) -> Any:
            result = await super().create_response(**kwargs)
            await self.release_response.wait()
            return result

    adapter = _BlockingOpenAdapter()
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id="first-message",
                thread_id="thread-1",
                content=[{"type": "text", "text": "hi"}],
            )
        )
    )
    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id="second-message",
                thread_id="thread-1",
                content=[{"type": "text", "text": "hi"}],
            )
        )
    )

    await _wait_for(
        lambda: any(
            payload["source_message_id"] == "second-message"
            for payload in supervisor.payloads(
                message_type=AGENT_EVENT_TURN_START_ACCEPTED
            )
        )
    )

    assert not adapter.open_session_started.is_set()
    assert all(
        call["event_handler"] is not None for call in adapter.start_session_calls
    )

    adapter.release_response.set()
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) >= 2
    )

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_stops_adapter_session_on_thread_close() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    process.send(
        Message(
            data=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            )
        )
    )

    await asyncio.wait_for(adapter.stop_session_event.wait(), timeout=1)

    assert adapter.call_order == ["start_session", "stop_session"]
    assert adapter.stop_session_calls == [{"context": adapter.session}]
    assert thread_storage.stopped == 0

    await process.stop(supervisor)

    assert thread_storage.stopped == 1


@pytest.mark.asyncio
async def test_llm_agent_process_keeps_thread_storage_open_on_thread_close() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    process.send(
        Message(
            data=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.stop_session_event.wait(), timeout=1)

    adapter.start_session_event.clear()
    adapter.stop_session_event.clear()

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    assert thread_storage.started == 1
    assert thread_storage.stopped == 0
    assert len(adapter.start_session_calls) == 2
    assert len(adapter.stop_session_calls) == 1

    await process.stop(supervisor)

    assert thread_storage.stopped == 1


@pytest.mark.asyncio
async def test_llm_agent_process_open_queued_while_thread_close_is_finishing() -> None:
    adapter = _BlockingSessionLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    adapter.block_next_stop_session = True
    process.send(
        Message(
            data=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.stop_session_entered.wait(), timeout=1)

    adapter.start_session_event.clear()
    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )

    assert len(adapter.start_session_calls) == 1

    adapter.release_stop_session.set()
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)

    assert thread_storage.started == 1
    assert thread_storage.stopped == 0
    assert len(adapter.start_session_calls) == 2
    assert len(adapter.stop_session_calls) == 1

    await process.stop(supervisor)

    assert thread_storage.stopped == 1


@pytest.mark.asyncio
async def test_llm_agent_process_close_queued_while_thread_open_is_finishing() -> None:
    adapter = _BlockingSessionLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            )
        )
    )
    await _wait_for(lambda: len(adapter.call_order) == 0)

    adapter.block_next_start_session = True
    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await asyncio.wait_for(adapter.start_session_entered.wait(), timeout=1)

    process.send(
        Message(
            data=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            )
        )
    )

    assert len(adapter.start_session_calls) == 0
    assert len(adapter.stop_session_calls) == 0

    adapter.release_start_session.set()
    await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)
    await asyncio.wait_for(adapter.stop_session_event.wait(), timeout=1)

    assert thread_storage.started == 1
    assert thread_storage.stopped == 0
    assert len(adapter.start_session_calls) == 1
    assert len(adapter.stop_session_calls) == 1

    await process.stop(supervisor)

    assert thread_storage.stopped == 1


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_usage_updates_on_open_and_after_adapter_create_response() -> (
    None
):
    adapter = _UsageRecordingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        toolkits=[Toolkit(name="storage", tools=[])],
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    turn_id = started_payload["turn_id"]
    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)

    assert [payload["turn_id"] for payload in usage_payloads] == [
        None,
        turn_id,
    ]
    assert [payload["context_window"]["used_tokens"] for payload in usage_payloads] == [
        0,
        1250,
    ]
    assert all(
        payload["context_window"]["total_tokens"] == 128000
        for payload in usage_payloads
    )
    assert usage_payloads[0]["usage"] == {}
    assert usage_payloads[1]["usage"] == {
        "gpt-test.input_tokens": 1000.0,
        "gpt-test.output_tokens": 250.0,
    }
    assert adapter.input_token_calls == 0

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 3
    )
    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)

    assert usage_payloads[2]["turn_id"] is None
    assert usage_payloads[2]["context_window"]["used_tokens"] == 1250
    assert usage_payloads[2]["usage"] == {
        "gpt-test.input_tokens": 1000.0,
        "gpt-test.output_tokens": 250.0,
    }
    assert adapter.input_token_calls == 0
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_adds_participant_name_to_agent_messages() -> None:
    class _AgentMessageAdapter(_RecordingLLMAdapter):
        async def create_response(self, **kwargs) -> Any:
            event_handler = kwargs.get("event_handler")
            if event_handler is not None:
                event_handler(
                    AgentTextContentDelta(
                        type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                        thread_id=str(kwargs["context"].metadata["thread_id"]),
                        turn_id=str(kwargs["context"].metadata["turn_id"]),
                        item_id="assistant-1",
                        text="hello",
                    )
                )
                event_handler(
                    AgentToolCallPending(
                        type=AGENT_EVENT_TOOL_CALL_PENDING,
                        thread_id=str(kwargs["context"].metadata["thread_id"]),
                        turn_id=str(kwargs["context"].metadata["turn_id"]),
                        item_id="tool-1",
                        toolkit="shell",
                        tool="exec",
                        arguments={"cmd": "pwd"},
                    )
                )
            return await super().create_response(**kwargs)

    room = _DownloadRecordingRoom()
    await room.local_participant.set_attribute("name", "chatbot")
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=_AgentMessageAdapter(),
    )

    await process.start(supervisor)
    process.send(
        Message(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            ),
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_DELTA)) == 1
            and len(supervisor.payloads(message_type=AGENT_EVENT_TOOL_CALL_PENDING))
            == 1
        )
    )

    accepted_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_START_ACCEPTED
    )[0]
    text_payload = supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_DELTA)[0]
    tool_payload = supervisor.payloads(message_type=AGENT_EVENT_TOOL_CALL_PENDING)[0]
    assert accepted_payload["sender_name"] == "chatbot"
    assert text_payload["sender_name"] == "chatbot"
    assert tool_payload["sender_name"] == "chatbot"
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_session_usage_callback_without_duplicate_final_update() -> (
    None
):
    adapter = _SessionUsageCallbackLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )
    await asyncio.sleep(0)

    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)
    assert len(usage_payloads) == 2
    assert usage_payloads[1]["context_window"]["used_tokens"] == 1250
    assert usage_payloads[1]["usage"] == {
        "gpt-test.input_tokens": 1000.0,
        "gpt-test.output_tokens": 250.0,
    }

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_thread_open_usage_replies_to_source_without_storage_or_broadcast() -> (
    None
):
    adapter = _UsageRecordingLLMAdapter()
    supervisor = _RecordingSupervisor()
    source_channel = _ThreadOpenResponseChannel()
    broadcast_channel = _RecordingChannel()
    supervisor.add_channel(broadcast_channel)
    thread_storage = _LifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            ),
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            source=source_channel,
        )
    )
    await _wait_for(lambda: len(source_channel.direct_payloads) == 1)

    assert source_channel.direct_payloads[0].type == AGENT_EVENT_USAGE_UPDATED
    assert thread_storage.messages == []
    assert broadcast_channel.received == []

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_counts_open_context_only_when_saved_usage_missing() -> (
    None
):
    adapter = _UsageRecordingLLMAdapter()
    supervisor = _RecordingSupervisor()
    thread_storage = _RestoringLifecycleThreadStorage(path="thread-1")
    thread_storage.messages.append(
        TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            content=[AgentTextContent(type="text", text="saved prompt")],
        )
    )
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            )
        )
    )
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    usage_payload = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)[0]
    assert usage_payload["usage"] == {}
    assert usage_payload["context_window"]["used_tokens"] == 130
    assert adapter.input_token_calls == 1

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_usage_without_context_counting() -> None:
    adapter = _UsageCountingFailingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        toolkits=[Toolkit(name="storage", tools=[])],
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    usage_payload = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)[0]
    assert usage_payload["context_window"]["used_tokens"] == 1250
    assert usage_payload["usage"] == {
        "gpt-test.input_tokens": 1000.0,
        "gpt-test.output_tokens": 250.0,
    }
    assert adapter.input_token_calls == 0
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_last_flattened_usage_not_cumulative_usage() -> (
    None
):
    adapter = _OpenAIStyleUsageLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)
    for _ in range(2):
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    model="gpt-test",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 2
    )

    usage_payload = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)[-1]
    assert usage_payload["usage"] == {
        "gpt-test.input_tokens": 64000.0,
        "gpt-test.output_tokens": 1200.0,
    }
    assert usage_payload["context_window"]["used_tokens"] == 65200
    assert usage_payload["context_window"]["total_tokens"] == 128000

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_caps_context_usage_after_compaction_event() -> None:
    adapter = _CompactedUsageLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    usage_payload = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)[0]
    assert usage_payload["usage"] == {
        "gpt-test.input_tokens": 1000.0,
        "gpt-test.output_tokens": 250.0,
    }
    assert usage_payload["context_window"] == {
        "used_tokens": 250,
        "total_tokens": 128000,
        "compaction_mode": None,
        "compaction_threshold": None,
    }
    assert adapter.input_token_calls == 0
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_compaction_status_and_event() -> None:
    adapter = _CompactingLLMAdapter()
    supervisor = _RecordingSupervisor()
    publisher = _RecordingThreadStatusPublisher()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    compaction_payloads = supervisor.payloads(
        message_type=AGENT_EVENT_CONTEXT_COMPACTED
    )
    assert adapter.compact_calls == 1
    assert len(compaction_payloads) == 1
    assert compaction_payloads[0]["messages"] == [
        {"id": "compaction-1", "type": "compaction", "encrypted_content": "opaque"}
    ]
    assert [entry["status"] for entry in publisher.statuses[:3]] == [
        "Thinking",
        "Compacting context",
        "Thinking",
    ]
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_manual_compaction_emits_event_and_compacted_usage_at_threshold() -> (
    None
):
    adapter = _ThresholdManualCompactionLLMAdapter(
        initial_tokens=1000,
        compact_threshold=1000,
    )
    supervisor = _RecordingSupervisor()
    publisher = _RecordingThreadStatusPublisher()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert adapter.compact_calls == [1000]
    compaction_payloads = supervisor.payloads(
        message_type=AGENT_EVENT_CONTEXT_COMPACTED
    )
    assert len(compaction_payloads) == 1
    assert compaction_payloads[0]["messages"] == [
        {
            "id": "manual-compaction-1",
            "type": "compaction",
            "encrypted_content": "manual-opaque",
        }
    ]
    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)
    assert usage_payloads[-1]["context_window"] == {
        "used_tokens": 128,
        "total_tokens": 128000,
        "compaction_mode": "standalone",
        "compaction_threshold": 1000,
    }
    assert [entry["status"] for entry in publisher.statuses[:3]] == [
        "Thinking",
        "Compacting context",
        "Thinking",
    ]
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_auto_compaction_emits_event_and_compacted_usage_at_threshold() -> (
    None
):
    adapter = _ThresholdAutoCompactionLLMAdapter(
        initial_tokens=1000,
        compact_threshold=1000,
    )
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert adapter.compaction_calls == [1000]
    compaction_payloads = supervisor.payloads(
        message_type=AGENT_EVENT_CONTEXT_COMPACTED
    )
    assert len(compaction_payloads) == 1
    assert compaction_payloads[0]["messages"] == [
        {
            "id": "auto-compaction-1",
            "type": "compaction",
            "encrypted_content": "auto-opaque",
        }
    ]
    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)
    assert usage_payloads[-1]["context_window"] == {
        "used_tokens": 96,
        "total_tokens": 128000,
        "compaction_mode": "auto",
        "compaction_threshold": 1000,
    }
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_auto_compaction_does_not_emit_below_threshold() -> (
    None
):
    adapter = _ThresholdAutoCompactionLLMAdapter(
        initial_tokens=999,
        compact_threshold=1000,
    )
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert adapter.compaction_calls == []
    assert supervisor.payloads(message_type=AGENT_EVENT_CONTEXT_COMPACTED) == []
    usage_payloads = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)
    assert usage_payloads[-1]["context_window"] == {
        "used_tokens": 999,
        "total_tokens": 128000,
        "compaction_mode": "auto",
        "compaction_threshold": 1000,
    }
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_empty_usage_after_cancelled_adapter_call() -> (
    None
):
    adapter = _CancellationUsageLLMAdapter()
    supervisor = _RecordingSupervisor()
    thread_storage = _RestoringLifecycleThreadStorage(path="thread-1")
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_storage=thread_storage,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                model="gpt-test",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await asyncio.wait_for(adapter.call_started.wait(), timeout=1)
    turn_id = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]["turn_id"]
    process.send(
        Message(
            data=TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                thread_id="thread-1",
                turn_id=turn_id,
            )
        )
    )

    await asyncio.wait_for(adapter.call_cancelled.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)) == 1
    )

    usage_payload = supervisor.payloads(message_type=AGENT_EVENT_USAGE_UPDATED)[0]
    assert usage_payload["turn_id"] == turn_id
    assert usage_payload["usage"] == {}
    assert usage_payload["context_window"] == {
        "used_tokens": 0,
        "total_tokens": 128000,
        "compaction_mode": None,
        "compaction_threshold": None,
    }
    assert adapter.input_context_messages == []

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )
    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]["error"] == {
        "message": "turn cancelled",
        "code": "cancelled",
    }
    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_adapter_agent_event_publisher() -> None:
    adapter = _PublishingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_ENDED)) == 1
        )
    )

    started_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TEXT_CONTENT_STARTED
    )[0]
    delta_payload = supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_DELTA)[0]
    ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_ENDED)[0]

    assert started_payload["thread_id"] == "thread-1"
    assert started_payload["item_id"] == "assistant-1"
    assert started_payload["provider"] == "test-provider"
    assert started_payload["model"] == "default-model"
    assert delta_payload["message_id"] != started_payload["message_id"]
    assert delta_payload["message_id"] != delta_payload["item_id"]
    assert delta_payload["turn_id"] == started_payload["turn_id"]
    assert delta_payload["text"] == "hello"
    assert delta_payload["provider"] == "test-provider"
    assert delta_payload["model"] == "default-model"
    assert ended_payload["turn_id"] == started_payload["turn_id"]
    assert ended_payload["provider"] == "test-provider"
    assert ended_payload["model"] == "default-model"

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_builder_returned_room_bound_toolkits() -> None:
    room = _DownloadRecordingRoom()
    adapter = _RecordingLLMAdapter()

    async def _build_toolkits(sender, model, turns) -> list[Toolkit]:
        del sender
        del model
        del turns
        return [Toolkit(name="dynamic", room=room, tools=[_RoomBindingTool(room=room)])]

    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
        turn_toolkits_builder=_build_toolkits,
    )

    toolkits = await process._build_turn_toolkits(
        model="default-model",
        turns=[],
    )

    result = await toolkits[0].execute(
        context=ToolContext(caller=room.local_participant),
        name="room_binding_tool",
        input=JsonContent(json={}),
    )

    assert isinstance(result, TextContent)
    assert result.text == room.local_participant.id


@pytest.mark.asyncio
async def test_llm_agent_process_adds_supervisor_toolkits_with_custom_builder() -> None:
    room = _DownloadRecordingRoom()
    adapter = _RecordingLLMAdapter()
    supervisor = AgentSupervisor()
    thread_storage = _LifecycleThreadStorage(path="/threads/test.thread")

    async def _build_toolkits(sender, model, turns) -> list[Toolkit]:
        del sender
        del model
        del turns
        return [Toolkit(name="dynamic", tools=[])]

    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=adapter,
        thread_storage=thread_storage,
        turn_toolkits_builder=_build_toolkits,
    )

    await process.start(supervisor)
    try:
        toolkits = await process._build_turn_toolkits(
            model="default-model",
            turns=[],
        )
    finally:
        await process.stop(supervisor)

    assert [toolkit.name for toolkit in toolkits] == ["dynamic", "chat"]


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_custom_retry_status_events(
    monkeypatch,
) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    thread_adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _CustomEventLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=thread_adapter,
        thread_status_publisher=ParticipantAttributeThreadStatusPublisher(
            participant=room.local_participant,
            path="/threads/test.thread",
        ),
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await asyncio.wait_for(llm_adapter.retry_event_sent.wait(), timeout=1)
        await _wait_for(
            lambda: any(
                event.get_attribute("name") == "openai.retry"
                and event.get_attribute("headline")
                == "Reconnecting to the LLM (retry 1/10)"
                for event in room.sync.document.event_elements
            )
        )

        llm_adapter.release_completion.set()
        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)
        assert not any(
            name.startswith("thread.status")
            for name, _value in room.local_participant.set_attribute_calls
        )
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_can_publish_thread_status_as_agent_messages() -> None:
    adapter = _RecordingLLMAdapter()
    supervisor = _RecordingSupervisor()
    room = _ThreadRoom(document=_ThreadDocument())

    def publish_thread_status(message: AgentMessage) -> None:
        process.emit(sender=None, payload=message)

    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=adapter,
        thread_status_publisher=AgentMessageThreadStatusPublisher(
            thread_id="/threads/test.thread",
            publish=publish_thread_status,
        ),
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await _wait_for(
            lambda: any(
                payload["status"] == "Thinking"
                and payload["turn_id"] is not None
                and payload["mode"] == "steerable"
                for payload in supervisor.payloads(
                    message_type=AGENT_EVENT_THREAD_STATUS
                )
            )
        )
        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        await _wait_for(
            lambda: any(
                payload["status"] is None and payload["turn_id"] is None
                for payload in supervisor.payloads(
                    message_type=AGENT_EVENT_THREAD_STATUS
                )
            )
        )
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_processes_queued_steer_messages_before_turn_end() -> (
    None
):
    adapter = _QueuedSteerLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    turn_start_message_id = "00000000-0000-0000-0000-000000000003"
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=turn_start_message_id,
                thread_id="thread-1",
                content=[{"type": "text", "text": "first"}],
            )
        )
    )

    await asyncio.wait_for(adapter.started_events[0].wait(), timeout=1)

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    turn_id = started_payload["turn_id"]
    assert started_payload["thread_id"] == "thread-1"
    assert started_payload["source_message_id"] == turn_start_message_id
    assert isinstance(turn_id, str)

    steer_message_id_one = "00000000-0000-0000-0000-000000000004"
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                message_id=steer_message_id_one,
                thread_id="thread-1",
                turn_id=turn_id,
                content=[{"type": "text", "text": "second"}],
            )
        )
    )

    steer_message_id_two = "00000000-0000-0000-0000-000000000005"
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                message_id=steer_message_id_two,
                thread_id="thread-1",
                turn_id=turn_id,
                content=[{"type": "text", "text": "third"}],
            )
        )
    )

    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED) == []
    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED) == []

    adapter.release_events[0].set()
    await asyncio.wait_for(adapter.started_events[1].wait(), timeout=1)

    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED) == []
    assert len(adapter.calls) == 2
    accepted_payloads = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_STEER_ACCEPTED
    )
    steered_payloads = supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED)
    assert [payload["source_message_id"] for payload in accepted_payloads] == [
        steer_message_id_one,
        steer_message_id_two,
    ]
    for payload in accepted_payloads:
        uuid.UUID(payload["message_id"])
        assert payload["turn_id"] == turn_id
        assert payload["thread_id"] == "thread-1"
    assert [payload["source_message_id"] for payload in steered_payloads] == [
        steer_message_id_one,
        steer_message_id_two,
    ]
    for payload in steered_payloads:
        uuid.UUID(payload["message_id"])
        assert payload["turn_id"] == turn_id
        assert payload["thread_id"] == "thread-1"
    assert adapter.calls[0]["messages"] == [{"role": "user", "content": "first"}]
    assert adapter.calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]

    adapter.release_events[1].set()
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]
    assert ended_payload["thread_id"] == "thread-1"
    assert ended_payload["error"] is None

    await process.stop(supervisor)

    assert adapter.session.closed == 1


@pytest.mark.asyncio
async def test_llm_agent_process_applies_queued_steer_at_tool_boundary() -> None:
    adapter = _ToolBoundarySteeringLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    try:
        turn_start_message_id = "00000000-0000-0000-0000-000000000030"
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    message_id=turn_start_message_id,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "first"}],
                )
            )
        )

        await asyncio.wait_for(adapter.call_started.wait(), timeout=1)

        started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
        turn_id = started_payload["turn_id"]

        steer_message_id = "00000000-0000-0000-0000-000000000031"
        process.send(
            Message(
                data=TurnSteer(
                    type=AGENT_MESSAGE_TURN_STEER,
                    message_id=steer_message_id,
                    thread_id="thread-1",
                    turn_id=turn_id,
                    content=[{"type": "text", "text": "second"}],
                )
            )
        )

        await _wait_for(
            lambda: (
                len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_ACCEPTED))
                == 1
            )
        )
        accepted_payloads = supervisor.payloads(
            message_type=AGENT_EVENT_TURN_STEER_ACCEPTED
        )
        assert [payload["source_message_id"] for payload in accepted_payloads] == [
            steer_message_id
        ]
        assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED) == []
        assert supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED) == []

        adapter.release_tool_boundary.set()

        await asyncio.wait_for(adapter.tool_boundary_applied.wait(), timeout=1)
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )

        assert len(adapter.calls) == 1
        assert adapter.calls[0]["messages_before_boundary"] == [
            {"role": "user", "content": "first"}
        ]
        assert adapter.calls[0]["messages_after_boundary"] == [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert adapter.calls[0]["steered"] is True

        steered_payloads = supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED)
        assert [payload["source_message_id"] for payload in steered_payloads] == [
            steer_message_id
        ]
        assert steered_payloads[0]["turn_id"] == turn_id

        ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]
        assert ended_payload["error"] is None
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_rejects_steer_with_thread_id_on_event() -> None:
    adapter = _RecordingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    steer_message_id = "00000000-0000-0000-0000-000000000006"
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                message_id=steer_message_id,
                thread_id="thread-1",
                turn_id="missing-turn",
                content=[{"type": "text", "text": "continue"}],
            )
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_REJECTED)) == 1
        )
    )

    rejected_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_STEER_REJECTED
    )[0]
    uuid.UUID(rejected_payload["message_id"])
    assert rejected_payload["thread_id"] == "thread-1"
    assert rejected_payload["turn_id"] == "missing-turn"
    assert rejected_payload["source_message_id"] == steer_message_id
    assert rejected_payload["error"]["code"] == "turn_not_in_progress"

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_continues_queued_steer_when_turn_is_interrupted() -> (
    None
):
    adapter = _QueuedSteerLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "first"}],
            )
        )
    )

    await asyncio.wait_for(adapter.started_events[0].wait(), timeout=1)

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    turn_id = started_payload["turn_id"]

    steer_message_id = "00000000-0000-0000-0000-000000000007"
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                message_id=steer_message_id,
                thread_id="thread-1",
                turn_id=turn_id,
                content=[{"type": "text", "text": "second"}],
            )
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_ACCEPTED)) == 1
        )
    )
    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED) == []

    process.send(
        Message(
            data=TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                message_id="00000000-0000-0000-0000-000000000009",
                thread_id="thread-1",
                turn_id=turn_id,
            )
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TURN_INTERRUPT_ACCEPTED))
            == 1
        )
    )
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_INTERRUPTED)) == 1
    )
    await asyncio.wait_for(adapter.started_events[1].wait(), timeout=1)

    interrupt_accepted_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_INTERRUPT_ACCEPTED
    )[0]
    assert interrupt_accepted_payload["thread_id"] == "thread-1"
    assert interrupt_accepted_payload["turn_id"] == turn_id
    assert (
        interrupt_accepted_payload["source_message_id"]
        == "00000000-0000-0000-0000-000000000009"
    )

    interrupted_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_INTERRUPTED
    )[0]
    assert interrupted_payload["thread_id"] == "thread-1"
    assert interrupted_payload["turn_id"] == turn_id
    assert (
        interrupted_payload["source_message_id"]
        == "00000000-0000-0000-0000-000000000009"
    )
    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_REJECTED) == []
    assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)) == 1

    assert adapter.calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    assert adapter.calls[1]["metadata"] == {
        "thread_id": "thread-1",
        "turn_id": turn_id,
    }
    steered_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED)[0]
    assert steered_payload["source_message_id"] == steer_message_id
    assert steered_payload["turn_id"] == turn_id

    adapter.release_events[1].set()
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )
    ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]
    assert ended_payload["turn_id"] == turn_id
    assert ended_payload["error"] is None

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_sets_thread_and_turn_metadata_during_adapter_call() -> (
    None
):
    session = _LifecycleSession()
    adapter = _RecordingLLMAdapter(session=session)
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    assert adapter.calls[0]["metadata"] == {
        "thread_id": "thread-1",
        "turn_id": started_payload["turn_id"],
    }
    assert session.metadata == {}

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_calls_on_turn_steer_before_interrupt_continuation() -> (
    None
):
    adapter = _InterruptAwareQueuedSteerLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "first"}],
            )
        )
    )

    await asyncio.wait_for(adapter.started_events[0].wait(), timeout=1)

    turn_id = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]["turn_id"]
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                thread_id="thread-1",
                turn_id=turn_id,
                content=[{"type": "text", "text": "second"}],
            )
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_ACCEPTED)) == 1
        )
    )

    process.send(
        Message(
            data=TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                thread_id="thread-1",
                turn_id=turn_id,
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_INTERRUPTED)) == 1
    )
    await asyncio.wait_for(adapter.started_events[1].wait(), timeout=1)

    assert adapter.on_turn_steer_calls == [
        {
            "interrupted": True,
            "messages_before": [{"role": "user", "content": "first"}],
        }
    ]
    assert adapter.calls[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "TURN INTERRUPTED"},
        {"role": "user", "content": "second"},
    ]

    adapter.release_events[1].set()
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_stops_after_interrupt_even_if_adapter_swallows_cancellation() -> (
    None
):
    adapter = _CancellationIgnoringLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "tell a story"}],
            )
        )
    )

    await asyncio.wait_for(adapter.first_call_started.wait(), timeout=1)

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    turn_id = started_payload["turn_id"]

    steer_message_id = "00000000-0000-0000-0000-000000000008"
    process.send(
        Message(
            data=TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                message_id=steer_message_id,
                thread_id="thread-1",
                turn_id=turn_id,
                content=[{"type": "text", "text": "change direction"}],
            )
        )
    )

    await _wait_for(
        lambda: (
            len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_ACCEPTED)) == 1
        )
    )

    process.send(
        Message(
            data=TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                message_id="00000000-0000-0000-0000-000000000010",
                thread_id="thread-1",
                turn_id=turn_id,
            )
        )
    )

    await asyncio.wait_for(adapter.first_call_cancelled.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_INTERRUPTED)) == 1
    )
    interrupted_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_INTERRUPTED
    )[0]
    assert interrupted_payload["source_message_id"] == (
        "00000000-0000-0000-0000-000000000010"
    )
    assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)) == 1

    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_REJECTED) == []
    steered_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STEERED)[0]
    assert steered_payload["source_message_id"] == steer_message_id
    assert steered_payload["turn_id"] == turn_id
    assert supervisor.payloads(message_type=AGENT_EVENT_TEXT_CONTENT_DELTA) == []
    assert len(adapter.calls) == 2
    assert adapter.calls[1]["messages"] == [
        {"role": "user", "content": "tell a story"},
        {"role": "user", "content": "change direction"},
    ]
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )
    ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]
    assert ended_payload["turn_id"] == turn_id
    assert ended_payload["error"] is None

    await process.stop(supervisor)


@pytest.mark.parametrize(
    ("response_type", "expected_decision"),
    [
        (AGENT_MESSAGE_TOOL_CALL_APPROVE, True),
        (AGENT_MESSAGE_TOOL_CALL_REJECT, False),
    ],
)
@pytest.mark.asyncio
async def test_llm_agent_process_waits_for_pending_tool_call_approvals(
    response_type: str,
    expected_decision: bool,
) -> None:
    adapter = _ApprovalCapableLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "run tool"}],
            )
        )
    )

    await asyncio.wait_for(adapter.approval_requested.wait(), timeout=1)
    await _wait_for(
        lambda: (
            len(
                supervisor.payloads(
                    message_type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED
                )
            )
            == 1
        )
    )

    approval_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED
    )[0]
    assert approval_payload["thread_id"] == "thread-1"
    assert approval_payload["toolkit"] == "filesystem"
    assert approval_payload["tool"] == "delete"
    assert approval_payload["arguments"] == {"path": "tmp/file.txt"}
    assert supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED) == []
    assert set(process._pending_tool_call_approvals) == {"approval-1"}

    process.send(
        Message(
            data=(
                ApproveAgentToolCall(
                    type=AGENT_MESSAGE_TOOL_CALL_APPROVE,
                    thread_id="thread-1",
                    turn_id=approval_payload["turn_id"],
                    item_id=approval_payload["item_id"],
                )
                if response_type == AGENT_MESSAGE_TOOL_CALL_APPROVE
                else RejectAgentToolCall(
                    type=AGENT_MESSAGE_TOOL_CALL_REJECT,
                    thread_id="thread-1",
                    turn_id=approval_payload["turn_id"],
                    item_id=approval_payload["item_id"],
                )
            )
        )
    )

    await asyncio.wait_for(adapter.approval_resolved.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert adapter.approval_decisions == [expected_decision]
    assert process._pending_tool_call_approvals == {}

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_clears_pending_approvals_when_turn_ends() -> None:
    adapter = _ApprovalCapableLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[{"type": "text", "text": "run tool"}],
            )
        )
    )

    await asyncio.wait_for(adapter.approval_requested.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)) == 1
    )

    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    assert set(process._pending_tool_call_approvals) == {"approval-1"}

    process.send(
        Message(
            data=TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                thread_id="thread-1",
                turn_id=started_payload["turn_id"],
            )
        )
    )

    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert process._pending_tool_call_approvals == {}
    assert adapter.approval_decisions == []

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_appends_remote_file_urls_as_image_and_file_inputs() -> (
    None
):
    session = _AttachmentRecordingSession()
    adapter = _RecordingLLMAdapter(session=session)
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[
                    {"type": "file", "url": "https://example.com/image.png"},
                    {"type": "file", "url": "https://example.com/report.pdf"},
                ],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert session.image_url_calls == ["https://example.com/image.png"]
    assert session.file_url_calls == [
        {"url": "https://example.com/report.pdf", "filename": None}
    ]
    assert session.image_message_calls == []
    assert session.file_message_calls == []
    assert adapter.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "image-url", "url": "https://example.com/image.png"}],
        },
        {
            "role": "user",
            "content": [{"type": "file-url", "url": "https://example.com/report.pdf"}],
        },
    ]

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_resolves_room_file_urls_before_appending_inputs() -> (
    None
):
    session = _AttachmentRecordingSession()
    adapter = _RecordingLLMAdapter(session=session)
    room = _DownloadRecordingRoom(
        files={
            "images/cat.png": FileContent(
                data=b"png-bytes",
                name="cat.png",
                mime_type="image/png",
            ),
            "docs/report.pdf": FileContent(
                data=b"%PDF-1.7",
                name="report.pdf",
                mime_type="application/pdf",
            ),
            "audio/prompt.wav": FileContent(
                data=b"wav-bytes",
                name="prompt.wav",
                mime_type="audio/wav",
            ),
        }
    )
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[
                    {"type": "file", "url": "room://images/cat.png"},
                    {"type": "file", "url": "room:///docs/report.pdf"},
                    {"type": "file", "url": "room:///audio/prompt.wav"},
                ],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert room.storage.download_calls == [
        "images/cat.png",
        "docs/report.pdf",
        "audio/prompt.wav",
    ]
    assert session.image_message_calls == [
        {"mime_type": "image/png", "data": b"png-bytes"}
    ]
    assert session.file_message_calls == [
        {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "data": b"%PDF-1.7",
        },
        {
            "filename": "prompt.wav",
            "mime_type": "audio/wav",
            "data": b"wav-bytes",
        },
    ]
    assert session.image_url_calls == []
    assert session.file_url_calls == []
    assert adapter.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image-bytes",
                    "mime_type": "image/png",
                    "size": len(b"png-bytes"),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file-bytes",
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                    "size": len(b"%PDF-1.7"),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file-bytes",
                    "filename": "prompt.wav",
                    "mime_type": "audio/wav",
                    "size": len(b"wav-bytes"),
                }
            ],
        },
    ]

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_handles_audio_file_attachments_as_files() -> None:
    session = _AttachmentRecordingSession()
    adapter = _RecordingLLMAdapter(session=session)
    room = _DownloadRecordingRoom(
        files={
            "audio/prompt.wav": FileContent(
                data=b"wav-bytes",
                name="prompt.wav",
                mime_type="audio/wav",
            ),
        }
    )
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[
                    {"type": "file", "url": "room:///audio/prompt.wav"},
                ],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert session.file_message_calls == [
        {
            "filename": "prompt.wav",
            "mime_type": "audio/wav",
            "data": b"wav-bytes",
        }
    ]

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_preserves_data_url_file_attachments() -> None:
    session = _AttachmentRecordingSession()
    adapter = _RecordingLLMAdapter(session=session)
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
    )

    await process.start(supervisor)

    data_url = "data:text/plain;base64,aGVsbG8="
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="thread-1",
                content=[
                    {"type": "file", "url": data_url, "name": "note.txt"},
                ],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert session.file_url_calls == [{"url": data_url, "filename": "note.txt"}]
    assert session.file_message_calls == []
    assert adapter.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "file-url", "url": data_url, "filename": "note.txt"}],
        }
    ]

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_realtime_audio_commit_then_turn_start_runs_one_turn() -> (
    None
):
    session = _RealtimeAudioRecordingSession()
    adapter = _AudioRecordingLLMAdapter(session=session)
    supervisor = _RecordingSupervisor()
    thread_status_publisher = _RecordingThreadStatusPublisher()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=thread_status_publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=AgentRealtimeAudioChunk(
                    type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
                    thread_id="thread-1",
                    message_id="audio-chunk-1",
                    data=b"pcm-bytes",
                    format=AgentAudioFormat(type="audio/pcm", sample_rate=24000),
                ),
            )
        )
        process.send(
            Message(
                data=AgentRealtimeAudioCommit(
                    type=AGENT_MESSAGE_REALTIME_AUDIO_COMMIT,
                    thread_id="thread-1",
                    message_id="audio-commit-1",
                    turn_id="turn-1",
                ),
            )
        )
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    message_id="audio-turn-start-1",
                    turn_id="turn-1",
                    output_modalities=["text"],
                ),
            )
        )

        await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )
    finally:
        await process.stop(supervisor)

    assert session.audio_chunk_calls == [
        {
            "mime_type": "audio/pcm",
            "data": b"pcm-bytes",
            "sample_rate": 24000,
            "bitrate": None,
        }
    ]
    assert session.operation_order == ["append", "commit"]
    assert session.commit_calls == 1
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["options"] == {"output_modalities": ["text"]}
    assert len(adapter.start_session_calls) == 1
    assert callable(adapter.start_session_calls[0]["event_handler"])
    assert adapter.start_session_calls[0]["metadata"]["turn_id"] == "turn-1"
    assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_START_ACCEPTED)) == 1
    assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)) == 1
    assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    assert thread_status_publisher.statuses[:3] == [
        {
            "status": "Listening",
            "mode": "busy",
            "pending_item_id": None,
            "total_bytes": None,
            "lines_added": None,
            "lines_removed": None,
        },
        {
            "status": "Processing audio",
            "mode": "busy",
            "pending_item_id": None,
            "total_bytes": None,
            "lines_added": None,
            "lines_removed": None,
        },
        {
            "status": "Thinking",
            "mode": None,
            "pending_item_id": None,
            "total_bytes": None,
            "lines_added": None,
            "lines_removed": None,
        },
    ]
    audio_transcriptions = supervisor.payloads(
        message_type=AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED
    )
    assert audio_transcriptions == [
        {
            "content_index": None,
            "item_id": "user-audio-1",
            "message_id": audio_transcriptions[0]["message_id"],
            "model": "default-model",
            "provider": "test-provider",
            "response_id": None,
            "role": "user",
            "sender_name": "assistant",
            "text": "hello from audio",
            "thread_id": "thread-1",
            "turn_id": supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0][
                "turn_id"
            ],
            "type": AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED,
        }
    ]
    emitted_types = [message.data.type for message in supervisor.sent]
    assert emitted_types.index(AGENT_EVENT_TURN_START_ACCEPTED) < emitted_types.index(
        AGENT_EVENT_TURN_STARTED
    )
    assert emitted_types.index(AGENT_EVENT_TURN_STARTED) < emitted_types.index(
        AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED
    )
    assert emitted_types.index(AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED) < (
        emitted_types.index(AGENT_EVENT_TURN_ENDED)
    )


@pytest.mark.asyncio
async def test_llm_agent_process_automatic_realtime_audio_turn_starts_on_speech() -> (
    None
):
    session = _RealtimeAudioRecordingSession()
    adapter = _AutomaticAudioRecordingLLMAdapter(session=session)
    supervisor = _RecordingSupervisor()
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        thread_id="thread-1",
        llm_adapter=adapter,
        toolkits=[Toolkit(name="storage", tools=[])],
    )

    await process.start(supervisor)
    try:
        caller = _ThreadParticipant(name="caller", participant_id="caller-id")
        process.send(
            Message(
                data=AgentRealtimeAudioChunk(
                    type=AGENT_MESSAGE_REALTIME_AUDIO_CHUNK,
                    thread_id="thread-1",
                    message_id="audio-chunk-1",
                    data=b"pcm-bytes",
                    format=AgentAudioFormat(type="audio/pcm", sample_rate=24000),
                ),
                sender=caller,
            )
        )

        await asyncio.wait_for(adapter.start_session_event.wait(), timeout=1)
        assert adapter.realtime_session_calls == [
            {
                "context": adapter.session,
                "caller": caller,
                "toolkits": ["storage"],
                "tool_choice": None,
                "model": "default-model",
                "options": None,
            }
        ]
        assert supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED) == []
        assert callable(session.event_handler)

        session.event_handler(
            {
                "type": "input_audio_buffer.speech_started",
                "item_id": "user-audio-1",
                "audio_start_ms": 10,
            }
        )
        started_payloads = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)
        assert len(started_payloads) == 1
        assert started_payloads[0]["source_message_id"] == "audio-chunk-1"

        session.event_handler(
            {
                "type": "input_audio_transcription.completed",
                "item_id": "user-audio-1",
                "text": "hello from audio",
            }
        )
        audio_transcriptions = supervisor.payloads(
            message_type=AGENT_EVENT_AUDIO_TRANSCRIPTION_COMPLETED
        )
        assert len(audio_transcriptions) == 1
        assert audio_transcriptions[0]["role"] == "user"
        assert audio_transcriptions[0]["sender_name"] == "caller"

        session.event_handler({"type": "response.done", "response": {"output": []}})
        ended_payloads = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)
        assert len(ended_payloads) == 1
        assert ended_payloads[0]["turn_id"] == started_payloads[0]["turn_id"]
    finally:
        await process.stop(supervisor)


def test_llm_agent_process_accepts_generic_thread_adapter() -> None:
    room = _ThreadRoom(document=_ThreadDocument())

    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        participant=room.local_participant,
        llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
        thread_adapter=_GenericThreadAdapter(
            room=room,
            path="/threads/test.thread",
        ),
    )

    assert process.thread_adapter is not None


def test_llm_agent_process_allows_storage_path_to_differ_from_thread_id() -> None:
    room = _ThreadRoom(document=_ThreadDocument())

    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        participant=room.local_participant,
        llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
        thread_storage=_GenericThreadAdapter(
            room=room,
            path="/threads/test",
        ),
    )

    assert process.thread_id == "/threads/test.thread"
    assert process.thread_storage is not None
    assert process.thread_storage.path == "/threads/test"


def test_llm_agent_process_handles_models_request_without_thread_id() -> None:
    room = _ThreadRoom(document=_ThreadDocument())
    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        participant=room.local_participant,
        llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
        thread_storage=_GenericThreadAdapter(
            room=room,
            path="/threads/test.thread",
        ),
    )

    assert process.handles(
        Message(data=ModelsRequest(type=AGENT_MESSAGE_MODELS_REQUEST))
    )


@pytest.mark.asyncio
async def test_llm_agent_process_defaults_to_first_provider() -> None:
    primary = _RecordingLLMAdapter(session=_LifecycleSession())
    secondary = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[
            LLMProvider(name="primary", adapter=primary),
            LLMProvider(name="secondary", adapter=secondary),
        ],
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await _wait_for(lambda: len(primary.calls) == 1)

        assert secondary.calls == []
        assert primary.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_defers_runtime_configuration_until_turn_done() -> None:
    original = _RecordingLLMAdapter(session=_LifecycleSession())
    replacement = _RecordingLLMAdapter(session=_LifecycleSession())
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[LLMProvider(name="original", adapter=original)],
    )
    turn_task = asyncio.create_task(asyncio.sleep(10))
    process._turn_task = turn_task
    process._turn_id = "turn-1"

    process.configure_runtime(
        llm_providers=[LLMProvider(name="replacement", adapter=replacement)],
        toolkits=[Toolkit(name="updated", tools=[])],
    )

    assert process.llm_adapter is original
    assert process.toolkits == []

    turn_task.cancel()
    with suppress(asyncio.CancelledError):
        await turn_task
    process._on_turn_done(turn_task)

    assert process.llm_adapter is replacement
    assert [toolkit.name for toolkit in process.toolkits] == ["updated"]


@pytest.mark.asyncio
async def test_llm_agent_process_uses_configured_default_provider() -> None:
    primary = _RecordingLLMAdapter(session=_LifecycleSession())
    secondary = _RecordingLLMAdapter(session=_LifecycleSession())
    secondary_provider = LLMProvider(name="secondary", adapter=secondary)
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[
            LLMProvider(name="primary", adapter=primary),
            secondary_provider,
        ],
        default_provider=secondary_provider,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await _wait_for(lambda: len(secondary.calls) == 1)

        assert primary.calls == []
        assert secondary.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_turn_output_modalities_change_emits_model_changed() -> (
    None
):
    adapter = _AudioRecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[LLMProvider(name="primary", adapter=adapter)],
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    output_modalities=["audio"],
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await _wait_for(lambda: len(adapter.calls) == 1)

        changed = supervisor.payloads(message_type=AGENT_EVENT_MODEL_CHANGED)
        assert len(changed) == 1
        assert changed[0]["provider"] == "primary"
        assert changed[0]["model"] == "default-model"
        assert changed[0]["output_modalities"] == ["audio"]
        assert adapter.calls[0]["options"] == {"output_modalities": ["audio"]}
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_thread_open_restores_last_persisted_model() -> None:
    primary = _RecordingLLMAdapter(session=_LifecycleSession())
    secondary = _RecordingLLMAdapter(session=_LifecycleSession())
    storage = _LifecycleThreadStorage(path="thread-1")
    storage.messages.extend(
        [
            AgentModelChanged(
                type=AGENT_EVENT_MODEL_CHANGED,
                thread_id="thread-1",
                provider="primary",
                provider_friendly_name="Primary",
                model="default-model",
            ),
            AgentModelChanged(
                type=AGENT_EVENT_MODEL_CHANGED,
                thread_id="thread-1",
                provider="secondary",
                provider_friendly_name="Secondary",
                model="default-model",
            ),
        ]
    )
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[
            LLMProvider(name="primary", adapter=primary),
            LLMProvider(name="secondary", adapter=secondary),
        ],
        thread_storage=storage,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="thread-1",
                )
            )
        )

        await _wait_for(lambda: len(secondary.start_session_calls) == 1)

        changed = supervisor.payloads(message_type=AGENT_EVENT_MODEL_CHANGED)
        assert len(changed) == 1
        assert changed[0]["provider"] == "secondary"
        assert changed[0]["model"] == "default-model"
        assert primary.start_session_calls == []
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_turn_provider_override_switches_and_restores() -> None:
    primary_session = _LifecycleSession()
    secondary_session = _LifecycleSession()
    primary = _RecordingLLMAdapter(session=primary_session)
    secondary = _RecordingLLMAdapter(session=secondary_session)
    storage = _ProviderRestoreRecordingThreadStorage(path="thread-1")
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[
            LLMProvider(name="primary", adapter=primary),
            LLMProvider(name="secondary", adapter=secondary),
        ],
        thread_storage=storage,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    provider="primary",
                    content=[{"type": "text", "text": "first"}],
                )
            )
        )
        await _wait_for(lambda: len(primary.calls) == 1)

        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    provider="secondary",
                    content=[{"type": "text", "text": "second"}],
                )
            )
        )
        await _wait_for(lambda: len(secondary.calls) == 1)

        assert len(storage.restore_calls) == 2
        assert storage.restore_calls[0]["context"] is primary_session
        assert storage.restore_calls[0]["llm_adapter"] is primary
        assert storage.restore_calls[1]["context"] is secondary_session
        assert storage.restore_calls[1]["llm_adapter"] is secondary
        assert primary.stop_session_calls == [{"context": primary_session}]
        assert primary_session.closed == 1
        assert process.session_context is secondary_session
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_rejects_unknown_turn_provider() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_providers=[LLMProvider(name="known", adapter=adapter)],
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    provider="missing",
                    content=[{"type": "text", "text": "hello"}],
                )
            )
        )

        await _wait_for(
            lambda: any(
                message.data.type == AGENT_EVENT_TURN_START_REJECTED
                for message in supervisor.sent
            )
        )

        rejected = [
            message.data
            for message in supervisor.sent
            if message.data.type == AGENT_EVENT_TURN_START_REJECTED
        ]
        assert len(rejected) == 1
        rejection = rejected[0]
        assert isinstance(rejection, TurnStartRejected)
        assert rejection.error is not None
        assert rejection.error.code == "unknown_provider"
        assert "missing" in rejection.error.message
        assert adapter.calls == []
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_optional_thread_status_publisher() -> None:
    adapter = _QueuedSteerLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "first"}],
                )
            )
        )
        await asyncio.wait_for(adapter.started_events[0].wait(), timeout=1)
        await _wait_for(
            lambda: any(status["status"] == "Thinking" for status in publisher.statuses)
        )

        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "second"}],
                )
            )
        )

        await _wait_for(lambda: len(publisher.pending_messages) > 0)
        assert publisher.turn_ids[0] is not None
        assert publisher.pending_messages[-1][0]["content"] == [
            {"type": "text", "text": "second"}
        ]

        adapter.release_events[0].set()
        adapter.release_events[1].set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
        assert publisher.clear_count >= 1
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_image_generation_status() -> None:
    adapter = _ImageGenerationStatusLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "make an image"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Generating image"
                and status["pending_item_id"] == "image-tool"
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.statuses[-1]["status"] == "Thinking")
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_writing_for_final_answer_text_status() -> (
    None
):
    adapter = _FinalAnswerTextStatusLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write me a story"}],
                )
            )
        )

        await _wait_for(
            lambda: any(status["status"] == "Writing" for status in publisher.statuses)
        )
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
        statuses = [status["status"] for status in publisher.statuses]
        writing_index = statuses.index("Writing")
        assert "Thinking" in statuses[:writing_index]
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_tool_status_from_agent_messages() -> None:
    adapter = _ShellStatusLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "read the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Reading src/app.py"
                and status["pending_item_id"] == "shell-tool"
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_tool_argument_snapshot_size() -> None:
    adapter = _ShellStatusLLMAdapter(
        command=[
            "python",
            "-c",
            "from pathlib import Path; "
            "Path('/tmp/meshagent_counter_probe.txt').write_text('x' * 256)",
        ]
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                isinstance(status["status"], str)
                and status["pending_item_id"] == "shell-tool"
                and isinstance(status["total_bytes"], int)
                and status["total_bytes"] > 100
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_preparing_command_argument_size() -> None:
    adapter = _ShellStatusLLMAdapter(command="", argument_bytes=150, pending=True)
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "run a command"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing"
                and status["pending_item_id"] == "shell-tool"
                and status["total_bytes"] == 150
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_increments_preparing_command_argument_size() -> None:
    adapter = _ShellStatusLLMAdapter(
        command="",
        pending=True,
        command_deltas=["x" * 120, "y" * 80],
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "run a command"}],
                )
            )
        )
        await _wait_for(
            lambda: (
                len(
                    [
                        status
                        for status in publisher.statuses
                        if status["status"] == "Preparing"
                        and status["pending_item_id"] == "shell-tool"
                        and isinstance(status["total_bytes"], int)
                        and status["total_bytes"] >= 120
                    ]
                )
                >= 2
            )
        )
        preparing_totals = [
            status["total_bytes"]
            for status in publisher.statuses
            if status["status"] == "Preparing"
            and status["pending_item_id"] == "shell-tool"
            and isinstance(status["total_bytes"], int)
        ]
        assert preparing_totals[-2:] == [120, 200]

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_tool_argument_delta_size() -> None:
    adapter = _ToolArgumentDeltaStatusLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing to write src/app.py"
                and status["pending_item_id"] == "write-tool"
                and status["total_bytes"] == 120
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_partial_json_argument_delta_for_status() -> None:
    adapter = _PartialToolArgumentDeltaStatusLLMAdapter(
        toolkit="storage",
        tool="write_file",
        arguments={},
        deltas=['{"path":"src/app.py","content":"'],
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing to write src/app.py"
                and status["pending_item_id"] == "partial-tool"
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_apply_patch_delta_for_status_counts() -> None:
    adapter = _PartialToolArgumentDeltaStatusLLMAdapter(
        toolkit="openai",
        tool="apply_patch",
        arguments={},
        deltas=[
            "*** Begin Patch\n*** Update File: app.ts\n",
            "@@\n-old\n+new\n+extra\n*** End Patch\n",
        ],
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "edit the app"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Editing app.ts"
                and status["pending_item_id"] == "partial-tool"
                and status["lines_added"] == 2
                and status["lines_removed"] == 1
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_joins_apply_patch_operation_args_with_delta_counts() -> (
    None
):
    adapter = _PartialToolArgumentDeltaStatusLLMAdapter(
        toolkit="openai",
        tool="apply_patch",
        arguments={
            "operation": {
                "type": "update_file",
                "path": "report.py",
                "diff": "@@\n-old\n+new\n+extra\n",
            }
        },
        deltas=["@@\n-old\n+new\n+extra\n"],
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "edit the report"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Editing report.py"
                and status["pending_item_id"] == "partial-tool"
                and status["lines_added"] == 2
                and status["lines_removed"] == 1
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_shell_command_delta_for_status() -> None:
    adapter = _ShellStatusLLMAdapter(
        command="",
        pending=True,
        command_deltas=["cat > /tmp/app.py <<'EOF'\n"],
    )
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing to write /tmp/app.py"
                and status["pending_item_id"] == "shell-tool"
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_keeps_tool_argument_delta_status_publish_order() -> (
    None
):
    adapter = _ToolArgumentDeltaStatusLLMAdapter()
    publisher = _DelayedPreparingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await asyncio.wait_for(publisher.first_preparing_started.wait(), timeout=1)
        await asyncio.sleep(0.05)
        assert not any(
            status["status"] == "Preparing to write src/app.py"
            and status["pending_item_id"] == "write-tool"
            and status["total_bytes"] == 120
            for status in publisher.statuses
        )

        publisher.release_first_preparing.set()
        await _wait_for(
            lambda: (
                len(publisher.statuses) >= 2
                and publisher.statuses[-1]["status"] == "Preparing to write src/app.py"
                and publisher.statuses[-1]["pending_item_id"] == "write-tool"
                and publisher.statuses[-1]["total_bytes"] == 120
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_tool_argument_delta_size_as_agent_message() -> (
    None
):
    adapter = _ToolArgumentDeltaStatusLLMAdapter()
    supervisor = _RecordingSupervisor()

    def publish_thread_status(message: AgentMessage) -> None:
        process.emit(sender=None, payload=message)

    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=AgentMessageThreadStatusPublisher(
            thread_id="thread-1",
            publish=publish_thread_status,
        ),
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "write the file"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing to write src/app.py"
                and status["pending_item_id"] == "write-tool"
                and status["total_bytes"] == 120
                for status in supervisor.payloads(
                    message_type=AGENT_EVENT_THREAD_STATUS
                )
            )
        )

        adapter.release.set()
        await _wait_for(
            lambda: any(
                status["status"] is None and status["total_bytes"] is None
                for status in supervisor.payloads(
                    message_type=AGENT_EVENT_THREAD_STATUS
                )
            )
        )
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_publishes_tool_event_argument_delta_size() -> None:
    adapter = _ToolEventArgumentDeltaStatusLLMAdapter()
    publisher = _RecordingThreadStatusPublisher()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=_DownloadRecordingRoom(),
        thread_id="thread-1",
        llm_adapter=adapter,
        thread_status_publisher=publisher,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[{"type": "text", "text": "run the command"}],
                )
            )
        )
        await _wait_for(
            lambda: any(
                status["status"] == "Preparing"
                and status["pending_item_id"] == "shell-tool"
                and status["total_bytes"] == 140
                for status in publisher.statuses
            )
        )

        adapter.release.set()
        await _wait_for(lambda: publisher.turn_ids[-1] is None)
    finally:
        await process.stop(supervisor)


def test_llm_agent_process_does_not_handle_clear_thread() -> None:
    room = _DownloadRecordingRoom()
    llm_adapter = _RecordingLLMAdapter()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
    )

    assert not process.handles(
        Message(
            data=ClearThread(
                type=AGENT_MESSAGE_THREAD_CLEAR,
                thread_id="/threads/test.thread",
            )
        )
    )


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_persists_channel_emitted_file_events(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    channel = _LifecycleChannel()
    process = AgentProcess(
        thread_id="/threads/test.thread",
        thread_adapter=MeshDocumentThreadStorage(
            room=room,
            path="/threads/test.thread",
        ),
    )
    supervisor = AgentSupervisor()
    supervisor.add_channel(channel)
    supervisor.add_process(process)

    await supervisor.start()
    try:
        await asyncio.wait_for(channel.start_event.wait(), timeout=1)

        channel.emit(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            payload=AgentFileContentStarted(
                type=AGENT_EVENT_FILE_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="file-1",
            ),
        )
        channel.emit(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            payload=AgentFileContentDelta(
                type=AGENT_EVENT_FILE_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="file-1",
                url="room:///docs/report.pdf",
            ),
        )
        channel.emit(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            payload=AgentFileContentEnded(
                type=AGENT_EVENT_FILE_CONTENT_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="file-1",
            ),
        )

        await _wait_for(lambda: len(room.sync.document.message_elements) == 1)

        assistant_message = room.sync.document.message_elements[0]
        assert assistant_message.get_attribute("author_name") == "assistant"
        assert assistant_message.get_attribute("role") == "agent"
        assert assistant_message.get_attribute("turn_id") == "turn-1"
        assert [
            child.get_attribute("path")
            for child in assistant_message.get_children_by_tag_name("file")
        ] == ["docs/report.pdf"]
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_persists_events_messages_and_status(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _ThreadPublishingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=adapter,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[
                        {"type": "text", "text": "hello from caller"},
                        {"type": "file", "url": "https://example.com/report.pdf"},
                    ],
                ),
            )
        )

        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )
        await _wait_for(lambda: len(room.sync.document.message_elements) == 2)
        await _wait_for(lambda: len(room.sync.document.event_elements) >= 2)

        user_message = room.sync.document.message_elements[0]
        assert user_message.get_attribute("author_name") == "caller"
        assert user_message.get_attribute("role") == "user"
        assert user_message.get_attribute("text") == "hello from caller"
        assert [
            child.get_attribute("path")
            for child in user_message.get_children_by_tag_name("file")
        ] == ["https://example.com/report.pdf"]

        assistant_message = room.sync.document.message_elements[1]
        assert assistant_message.get_attribute("author_name") == "assistant"
        assert assistant_message.get_attribute("role") == "agent"
        assert assistant_message.get_attribute("text") == "hello"
        assert llm_adapter.calls[0]["messages"][0]["role"] == "user"
        assert re.fullmatch(
            r"caller said at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z: hello from caller",
            llm_adapter.calls[0]["messages"][0]["content"],
        )
        assert llm_adapter.calls[0]["messages"][1] == {
            "role": "user",
            "content": "caller attached a file available at https://example.com/report.pdf",
        }

        tool_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        assert tool_event.get_attribute("kind") == "web"
        assert tool_event.get_attribute("state") == "completed"
        assert tool_event.get_attribute("preview") == ""

        turn_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id")
            == supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]["turn_id"]
        )
        assert turn_event.get_attribute("kind") == "turn"
        await _wait_for(lambda: turn_event.get_attribute("state") == "completed")
        assert turn_event.get_attribute("state") == "completed"

        turn_id = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0][
            "turn_id"
        ]
        assert user_message.get_attribute("turn_id") is None
        assert assistant_message.get_attribute("turn_id") == turn_id
        assert tool_event.get_attribute("turn_id") == turn_id
        assert turn_event.get_attribute("turn_id") == turn_id

        assert room.sync.document.member_names == ["assistant", "caller"]
        assert not any(
            name.startswith("thread.status")
            for name, _value in room.local_participant.set_attribute_calls
        )
    finally:
        await process.stop(supervisor)

    assert room.sync.close_calls == ["/threads/test.thread"]


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_inserts_accepted_turn_before_session_initializer_finishes(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    initializer_started = asyncio.Event()
    release_initializer = asyncio.Event()

    async def _session_initializer() -> AgentSessionContext:
        initializer_started.set()
        await release_initializer.wait()
        return _LifecycleSession()

    document = _ThreadDocument()
    document.root.messages.append_child(
        "message",
        {
            "text": "Earlier question",
            "created_at": "2026-03-11T00:00:00Z",
            "author_name": "caller",
            "role": "user",
        },
    )
    room = _ThreadRoom(document=document)
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=adapter,
        session_initializer=_session_initializer,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    message_id="00000000-0000-0000-0000-000000000101",
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "hello from caller"}],
                ),
            )
        )

        await asyncio.wait_for(initializer_started.wait(), timeout=1)
        await _wait_for(lambda: len(room.sync.document.message_elements) == 2)

        user_message = room.sync.document.message_elements[1]
        assert user_message.get_attribute("id") == (
            "00000000-0000-0000-0000-000000000101"
        )
        assert user_message.get_attribute("turn_id") is None
        assert user_message.get_attribute("author_name") == "caller"
        assert user_message.get_attribute("role") == "user"
        assert user_message.get_attribute("text") == "hello from caller"
        assert llm_adapter.calls == []

        release_initializer.set()
        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)

        assert llm_adapter.calls[0]["messages"][0] == {
            "role": "user",
            "content": "caller said at 2026-03-11T00:00:00Z: Earlier question",
        }
        live_messages = [
            message
            for message in llm_adapter.calls[0]["messages"]
            if message["role"] == "user"
            and isinstance(message["content"], str)
            and "hello from caller" in message["content"]
        ]
        assert len(live_messages) == 1
    finally:
        release_initializer.set()
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_accepts_turn_before_resolving_turn_instructions(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    async def _turn_instructions_provider(sender: Participant | None) -> str:
        del sender
        provider_started.set()
        await release_provider.wait()
        return "custom instructions"

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=adapter,
        turn_instructions_provider=_turn_instructions_provider,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "hello from caller"}],
                ),
            )
        )

        await asyncio.wait_for(provider_started.wait(), timeout=1)
        await _wait_for(lambda: len(room.sync.document.message_elements) == 1)
        assert len(supervisor.payloads(message_type=AGENT_EVENT_TURN_START_ACCEPTED))
        assert llm_adapter.calls == []

        release_provider.set()
        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)
        assert llm_adapter.calls[0]["context"].instructions == "custom instructions"
    finally:
        release_provider.set()
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_persists_turn_id_on_reasoning_messages_and_events(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            message=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="/threads/test.thread",
                message_id="turn-start-1",
                content=[{"type": "text", "text": "hello"}],
            ),
        )
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="turn-start-1",
            ),
        )
        adapter.push_message(
            message=AgentReasoningContentStarted(
                type=AGENT_EVENT_REASONING_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="reasoning-1",
            ),
        )
        adapter.push_message(
            message=AgentReasoningContentDelta(
                type=AGENT_EVENT_REASONING_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="reasoning-1",
                text="thinking",
            ),
        )
        adapter.push_message(
            message=AgentReasoningContentEnded(
                type=AGENT_EVENT_REASONING_CONTENT_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="reasoning-1",
            ),
        )
        adapter.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
            ),
        )
        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
                text="hi there",
            ),
        )
        adapter.push_message(
            message=AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
            ),
        )
        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="lookup",
                tool="lookup",
                arguments={"query": "meshagent"},
            ),
        )
        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                result=TextContent(text="ok"),
            ),
        )

        await real_sleep(0)
        await _wait_for(lambda: len(room.sync.document.message_elements) >= 2)
        await _wait_for(lambda: len(room.sync.document.reasoning_elements) == 1)
        await _wait_for(lambda: len(room.sync.document.event_elements) >= 2)

        user_message = room.sync.document.message_elements[0]
        assistant_message = room.sync.document.message_elements[1]
        reasoning = room.sync.document.reasoning_elements[0]
        tool_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        turn_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "turn-1"
        )

        assert user_message.get_attribute("turn_id") is None
        assert assistant_message.get_attribute("turn_id") == "turn-1"
        assert reasoning.get_attribute("turn_id") == "turn-1"
        assert tool_event.get_attribute("turn_id") == "turn-1"
        assert turn_event.get_attribute("turn_id") == "turn-1"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_does_not_persist_text_phase(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
                phase="commentary",
            ),
        )
        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
                text="checking",
            ),
        )
        adapter.push_message(
            message=AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-1",
            ),
        )
        adapter.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-2",
            ),
        )
        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-2",
                text="done",
            ),
        )
        adapter.push_message(
            message=AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="assistant-2",
                phase="final_answer",
            ),
        )

        await real_sleep(0)
        await _wait_for(lambda: len(room.sync.document.message_elements) == 2)

        commentary_message = room.sync.document.message_elements[0]
        final_answer_message = room.sync.document.message_elements[1]
        assert commentary_message.get_attribute("text") == "checking"
        assert final_answer_message.get_attribute("text") == "done"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_sets_status_from_streaming_output_deltas() -> (
    None
):
    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="commentary-1",
                phase="commentary",
            ),
        )
        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="commentary-1",
                text="checking",
            ),
        )
        await _wait_for(lambda: adapter._thread_status_value == "Planning")

        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="final-1",
                text="done",
                phase="final_answer",
            ),
        )
        await _wait_for(lambda: adapter._thread_status_value == "Writing")

        adapter.push_message(
            message=AgentAudioGenerationDelta(
                type=AGENT_EVENT_AUDIO_GENERATION_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="audio-1",
                data=b"pcm",
                mime_type="audio/pcm",
            ),
        )
        await _wait_for(lambda: adapter._thread_status_value == "Speaking")
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_sets_writing_from_final_answer_start() -> (
    None
):
    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")

        adapter.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="final-1",
                phase="final_answer",
            ),
        )
        await _wait_for(lambda: adapter._thread_status_value == "Writing")

        adapter.push_message(
            message=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="final-1",
                text="done",
            ),
        )
        await _wait_for(lambda: adapter._thread_status_value == "Writing")
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_surfaces_failed_turns_in_feed(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="turn-start-1",
            ),
        )
        adapter.push_message(
            message=TurnEnded(
                type=AGENT_EVENT_TURN_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                error=AgentError(
                    message=(
                        "Error from Anthropic: prompt is too long: 2368193 tokens > "
                        "1000000 maximum"
                    ),
                    code="RoomException",
                ),
            ),
        )

        await real_sleep(0)
        await _wait_for(lambda: len(room.sync.document.event_elements) == 1)

        turn_event = room.sync.document.event_elements[0]
        assert turn_event.get_attribute("item_id") == "turn-1"
        assert turn_event.get_attribute("turn_id") == "turn-1"
        assert turn_event.get_attribute("kind") == "message"
        assert turn_event.get_attribute("state") == "failed"
        assert (
            turn_event.get_attribute("headline")
            == "The model was not able to complete the request"
        )
        assert (
            turn_event.get_attribute("details")
            == "Error from Anthropic: prompt is too long: 2368193 tokens > "
            "1000000 maximum"
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_coalesces_shell_exploration_events_and_updates_status(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": "sed -n '1,20p' src/app.py"}},
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: adapter._thread_status_value == "Reading src/app.py")
        await _wait_for(
            lambda: (
                len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("kind") == "exec"
                    ]
                )
                == 1
            )
        )

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("kind") == "exec"
        )
        assert exec_event.get_attribute("state") == "in_progress"
        assert exec_event.get_attribute("headline") == "Reading src/app.py"
        assert exec_event.get_attribute("path") == ""
        assert exec_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                result=TextContent(text="line 1"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: exec_event.get_attribute("state") == "completed")
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-2",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": "cat src/app.py"}},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                exec_event.get_attribute("item_id") == "tool-2"
                and exec_event.get_attribute("state") == "in_progress"
            )
        )
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )
        assert exec_event.get_attribute("headline") == "Reading src/app.py"
        assert exec_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-2",
                result=TextContent(text="line 2"),
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                exec_event.get_attribute("item_id") == "tool-2"
                and exec_event.get_attribute("state") == "completed"
            )
        )
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=TurnEnded(
                type=AGENT_EVENT_TURN_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                error=None,
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: adapter._thread_status_value is None)
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_coalesces_repeated_web_searches_and_appends_queries(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    first_query = (
        "auto research agent implementation python planning report generation "
        "official docs examples"
    )
    second_query = (
        "openai deep research overview research agent browse synthesize report"
    )

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="ws_1",
                toolkit="openai",
                tool="web_search",
                arguments={"query": first_query},
            )
        )
        await real_sleep(0)

        web_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "ws_1"
        )
        await _wait_for(
            lambda: (
                web_event.get_attribute("state") == "in_progress"
                and web_event.get_attribute("headline") == "Searching the web"
            )
        )
        assert web_event.get_attribute("details") == first_query
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "web"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="ws_1",
                result=JsonContent(json={"results": [{"title": "Auto Research"}]}),
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                web_event.get_attribute("state") == "completed"
                and web_event.get_attribute("headline") == "Searched the web"
            )
        )
        assert web_event.get_attribute("details") == first_query

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="ws_2",
                toolkit="openai",
                tool="web_search",
                arguments={"query": second_query},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                web_event.get_attribute("item_id") == "ws_2"
                and web_event.get_attribute("state") == "in_progress"
                and web_event.get_attribute("headline") == "Searching the web"
            )
        )
        assert web_event.get_attribute("details") == (f"{first_query}\n{second_query}")
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "web"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_writes_preview_for_pending_and_started_running_command(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        command = "node scripts/custom.js --flag"
        adapter.push_message(
            message=AgentToolCallPending(
                type=AGENT_EVENT_TOOL_CALL_PENDING,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": command}},
            )
        )
        await real_sleep(0)

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        await _wait_for(
            lambda: (
                exec_event.get_attribute("state") == "pending"
                and exec_event.get_attribute("headline") == "Preparing"
            )
        )

        assert exec_event.get_attribute("kind") == "exec"
        assert exec_event.get_attribute("path") == ""
        assert exec_event.get_attribute("preview") == command

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": command}},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                exec_event.get_attribute("state") == "in_progress"
                and exec_event.get_attribute("headline") == "Running command"
            )
        )

        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_appends_and_prunes_event_logs(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-logs-1",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": "echo hello"}},
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallLogDelta(
                type=AGENT_EVENT_TOOL_CALL_LOG_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-logs-1",
                lines=[
                    AgentToolCallLogLine(source="stdout", text="line-1"),
                    AgentToolCallLogLine(source="stderr", text="line-2"),
                ],
            )
        )
        await real_sleep(0)

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-logs-1"
        )
        await _wait_for(lambda: len(exec_event.get_children_by_tag_name("log")) == 2)

        initial_logs = exec_event.get_children_by_tag_name("log")
        assert [log.get_attribute("source") for log in initial_logs] == [
            "stdout",
            "stderr",
        ]
        assert [log.get_attribute("text") for log in initial_logs] == [
            "line-1",
            "line-2",
        ]

        inserted_texts: list[str] = []
        log_counts_before_insert: list[int] = []
        original_append_child = exec_event.append_child

        def _tracking_append_child(
            tag_name: str,
            attributes: dict[str, Any] | None = None,
        ) -> _ThreadElement:
            if tag_name == "log":
                log_counts_before_insert.append(
                    len(exec_event.get_children_by_tag_name("log"))
                )
                if attributes is not None:
                    text = attributes.get("text")
                    if isinstance(text, str):
                        inserted_texts.append(text)
            return original_append_child(tag_name, attributes)

        monkeypatch.setattr(exec_event, "append_child", _tracking_append_child)

        batch_start = 3
        batch_stop = (
            batch_start + process_thread_adapter_module.EVENT_LOG_LINE_LIMIT + 5
        )
        adapter.push_message(
            message=AgentToolCallLogDelta(
                type=AGENT_EVENT_TOOL_CALL_LOG_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-logs-1",
                lines=[
                    AgentToolCallLogLine(source="stdout", text=f"line-{index}")
                    for index in range(batch_start, batch_stop)
                ],
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(exec_event.get_children_by_tag_name("log"))
                == process_thread_adapter_module.EVENT_LOG_LINE_LIMIT
            )
        )

        expected_inserted_texts = [
            f"line-{index}"
            for index in range(
                batch_stop - process_thread_adapter_module.EVENT_LOG_LINE_LIMIT,
                batch_stop,
            )
        ]
        assert inserted_texts == expected_inserted_texts
        assert max(log_counts_before_insert) <= (
            process_thread_adapter_module.EVENT_LOG_LINE_LIMIT - 1
        )

        pruned_logs = exec_event.get_children_by_tag_name("log")
        assert pruned_logs[0].get_attribute("text") == expected_inserted_texts[0]
        assert pruned_logs[-1].get_attribute("text") == expected_inserted_texts[-1]
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_coalesces_cd_prefixed_shell_exploration_commands(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": "cd /website && pwd && ls -la && find . -maxdepth 2 -type f | sed 's#^./##' | sort | head -200",
                    }
                },
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: adapter._thread_status_value == "Exploring /website")

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        assert exec_event.get_attribute("kind") == "exec"
        assert exec_event.get_attribute("headline") == "Exploring /website"
        assert exec_event.get_attribute("path") == ""
        assert exec_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                result=TextContent(text="/website"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: exec_event.get_attribute("state") == "completed")
        assert exec_event.get_attribute("headline") == "Explored /website"

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-2",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": "cd /website && sed -n '1,220p' public/index.html && printf '\\n---CSS---\\n' && sed -n '1,260p' public/styles.css && printf '\\n---JS---\\n' && sed -n '1,260p' public/app.js",
                    }
                },
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                exec_event.get_attribute("item_id") == "tool-2"
                and exec_event.get_attribute("state") == "in_progress"
            )
        )
        assert exec_event.get_attribute("headline") == "Exploring /website"
        assert exec_event.get_attribute("path") == ""
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_refines_shell_event_by_item_id_without_adding_duplicate(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={"action": {"command": "node scripts/custom.js --flag"}},
            )
        )
        await real_sleep(0)

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        await _wait_for(
            lambda: (
                exec_event.get_attribute("state") == "in_progress"
                and exec_event.get_attribute("headline") == "Running command"
            )
        )

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": "pwd && ls -la / && ls -la /workspace && find . -maxdepth 3 -type f | sed -n '1,120p'",
                    }
                },
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: exec_event.get_attribute("headline") == "Exploring .")
        assert exec_event.get_attribute("item_id") == "tool-1"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                result=TextContent(text="done"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: exec_event.get_attribute("state") == "completed")
        assert exec_event.get_attribute("headline") == "Explored ."
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_renders_cd_prefixed_shell_heredoc_write_as_file_event(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": "cd /website && cat > public/index.html <<'EOF'\n<!doctype html>\n<html></html>\nEOF",
                    }
                },
            )
        )
        await real_sleep(0)

        write_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "write-1"
        )
        assert write_event.get_attribute("kind") == "file"
        assert (
            write_event.get_attribute("headline")
            == "Writing /website/public/index.html"
        )
        assert write_event.get_attribute("path") == "/website/public/index.html"
        assert write_event.get_attribute("details") == ""
        assert write_event.get_attribute("preview") == (
            "cd /website && cat > public/index.html <<'EOF'\n"
            "<!doctype html>\n"
            "<html></html>\n"
            "EOF"
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                result=TextContent(text="ok"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: write_event.get_attribute("state") == "completed")
        assert (
            write_event.get_attribute("headline") == "Wrote /website/public/index.html"
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_renders_if_guarded_shell_heredoc_write_as_file_event(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-if-1",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": (
                            "mkdir -p /website/docs && cd /website/docs && "
                            "if [ ! -f index.html ]; then cat > index.html <<'EOF'\n"
                            "<!doctype html>\n"
                            "EOF\n"
                            "fi"
                        ),
                    }
                },
            )
        )
        await real_sleep(0)

        write_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "write-if-1"
        )
        assert write_event.get_attribute("kind") == "file"
        assert (
            write_event.get_attribute("headline") == "Writing /website/docs/index.html"
        )
        assert write_event.get_attribute("path") == "/website/docs/index.html"

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-if-1",
                result=TextContent(text="ok"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: write_event.get_attribute("state") == "completed")
        assert write_event.get_attribute("headline") == "Wrote /website/docs/index.html"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_groups_multi_file_shell_heredoc_writes(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-many-1",
                toolkit="openai",
                tool="shell",
                arguments={
                    "action": {
                        "command": "cd /website && mkdir -p src public dist && cat > package.json <<'EOF'\n{}\nEOF\ncat > tsconfig.json <<'EOF'\n{}\nEOF\ncat > src/main.tsx <<'EOF'\nconsole.log('hi')\nEOF\nnpm install\nnpm run build",
                    }
                },
            )
        )
        await real_sleep(0)

        write_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "write-many-1"
        )
        assert write_event.get_attribute("kind") == "file"
        assert write_event.get_attribute("headline") == "Writing files in /website"
        assert write_event.get_attribute("path") == "/website"
        assert write_event.get_attribute("details") == ""
        assert write_event.get_attribute("preview") == (
            "cd /website && mkdir -p src public dist && cat > package.json <<'EOF'\n"
            "{}\n"
            "EOF\n"
            "cat > tsconfig.json <<'EOF'\n"
            "{}\n"
            "..."
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-many-1",
                result=TextContent(text="ok"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: write_event.get_attribute("state") == "completed")
        assert write_event.get_attribute("headline") == "Wrote files in /website"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_renders_storage_read_write_and_grep_tools(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                toolkit="storage",
                tool="read_file",
                arguments={"path": "src/app.py", "offset": None},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("kind") == "exec"
                    ]
                )
                == 1
            )
        )
        read_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "read-1"
        )
        assert read_event.get_attribute("headline") == "Reading src/app.py"
        assert read_event.get_attribute("path") == "src/app.py"
        assert read_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                result=TextContent(text="print('hi')"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: read_event.get_attribute("state") == "completed")
        assert read_event.get_attribute("headline") == "Read src/app.py"
        assert read_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                toolkit="storage",
                tool="write_file",
                arguments={
                    "path": "src/app.py",
                    "text": "print('hello world')",
                    "overwrite": True,
                },
            )
        )
        await real_sleep(0)

        write_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "write-1"
        )
        assert write_event.get_attribute("kind") == "file"
        assert write_event.get_attribute("headline") == "Writing src/app.py"
        assert write_event.get_attribute("path") == "src/app.py"
        assert write_event.get_attribute("details") == ""
        assert write_event.get_attribute("preview") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                result=TextContent(text="the file was saved"),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: write_event.get_attribute("state") == "completed")
        assert write_event.get_attribute("headline") == "Wrote src/app.py"
        assert write_event.get_attribute("details") == ""
        assert write_event.get_attribute("preview") == ""

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="grep-1",
                toolkit="storage",
                tool="grep_file",
                arguments={
                    "path": "src/app.py",
                    "pattern": "hello",
                    "offset": None,
                    "before": None,
                    "after": None,
                },
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                read_event.get_attribute("item_id") == "grep-1"
                and read_event.get_attribute("state") == "in_progress"
            )
        )
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "exec"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("kind") == "file"
                ]
            )
            == 1
        )
        assert read_event.get_attribute("headline") == "Searching src/app.py"
        assert read_event.get_attribute("details") == "Pattern: hello"

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="grep-1",
                result=TextContent(text="1: print('hello world')"),
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                read_event.get_attribute("item_id") == "grep-1"
                and read_event.get_attribute("state") == "completed"
            )
        )
        assert read_event.get_attribute("headline") == "Searched src/app.py"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_marks_failed_storage_write_and_restores_thinking_status(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                toolkit="storage",
                tool="write_file",
                arguments={
                    "path": "src/app.py",
                    "text": "print('hello world')",
                    "overwrite": True,
                },
            )
        )
        await real_sleep(0)
        await _wait_for(lambda: adapter._thread_status_value == "Writing src/app.py")
        assert adapter._thread_status_pending_item_id_value == "write-1"

        write_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "write-1"
        )
        assert write_event.get_attribute("kind") == "file"
        assert write_event.get_attribute("preview") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="write-1",
                error=AgentError(
                    message="'text' is a required property",
                    code="tool_call_failed",
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: write_event.get_attribute("state") == "failed")
        assert (
            write_event.get_attribute("headline")
            == "Attempted to write file src/app.py"
        )
        assert write_event.get_attribute("details") == "'text' is a required property"
        assert write_event.get_attribute("preview") == ""
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "write-1"
                ]
            )
            == 1
        )
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")
        assert adapter._thread_status_pending_item_id_value is None
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_restores_thinking_status_on_turn_interrupt(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                toolkit="storage",
                tool="read_file",
                arguments={"path": "src/app.py", "offset": None},
            )
        )
        await real_sleep(0)
        await _wait_for(lambda: adapter._thread_status_value == "Reading src/app.py")
        assert adapter._thread_status_pending_item_id_value == "read-1"

        adapter.push_message(
            message=TurnInterrupted(
                type=AGENT_EVENT_TURN_INTERRUPTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="interrupt-1",
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: adapter._thread_status_value == "Thinking")
        assert adapter._thread_status_pending_item_id_value is None
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_uses_computer_startup_details_for_status(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        await adapter.handle_custom_event(
            event={
                "type": "agent.event",
                "source": "computer",
                "name": "computer.startup",
                "kind": "tool",
                "state": "in_progress",
                "method": "computer.startup",
                "correlation_key": "computer:start",
                "headline": "Starting computer...",
                "details": [
                    "Waiting for Playwright container to become ready.",
                ],
            },
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                adapter._thread_status_value
                == "Waiting for Playwright container to become ready."
            )
        )

        startup_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("name") == "computer.startup"
        )
        assert (
            startup_event.get_attribute("details")
            == "Waiting for Playwright container to become ready."
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_does_not_replace_thread_status_with_queued_steer(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)
        await _wait_for(lambda: adapter._thread_status_value == "Thinking")
        previous_status = adapter._thread_status_value
        adapter.push_message(
            message=TurnSteerAccepted(
                type=AGENT_EVENT_TURN_STEER_ACCEPTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="steer-1",
            )
        )
        await real_sleep(0)

        assert adapter._thread_status_value == previous_status
        assert not any(
            name.startswith("thread.status")
            for name, _value in room.local_participant.set_attribute_calls
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_message_thread_status_publisher_publishes_status_messages() -> (
    None
):
    published: list[AgentThreadStatus] = []

    def publish(message: AgentMessage) -> None:
        published.append(AgentThreadStatus.model_validate(message.model_dump()))

    publisher = AgentMessageThreadStatusPublisher(
        thread_id="/threads/test.thread",
        publish=publish,
    )

    await publisher.set_thread_turn_id(turn_id="turn-1")
    assert published[-1].type == AGENT_EVENT_THREAD_STATUS
    assert published[-1].thread_id == "/threads/test.thread"
    assert published[-1].turn_id == "turn-1"
    assert published[-1].status is None

    await publisher.set_thread_status(
        status=" Generating image ",
        pending_item_id=" image-1 ",
        total_bytes=240,
        lines_added=3,
        lines_removed=2,
    )
    active_status = published[-1]
    assert active_status.status == "Generating image"
    assert active_status.mode == "steerable"
    assert active_status.started_at is not None
    assert active_status.pending_item_id == "image-1"
    assert active_status.total_bytes == 240
    assert active_status.lines_added == 3
    assert active_status.lines_removed == 2

    await publisher.set_thread_status(
        status="Generating image",
        pending_item_id="image-1",
        total_bytes=240,
        lines_added=3,
        lines_removed=2,
    )
    assert published[-1] == active_status

    await publisher.set_thread_status(
        status="Generating image",
        pending_item_id="image-2",
        total_bytes=240,
        lines_added=3,
        lines_removed=2,
    )
    assert published[-1].pending_item_id == "image-2"

    await publisher.clear_thread_status()
    assert published[-1].status is None
    assert published[-1].mode is None
    assert published[-1].started_at is None
    assert published[-1].turn_id == "turn-1"
    assert published[-1].pending_item_id is None
    assert published[-1].total_bytes is None
    assert published[-1].lines_added is None
    assert published[-1].lines_removed is None


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_resets_started_at_when_status_changes(
    monkeypatch,
) -> None:
    timestamps = iter(
        [
            "2026-03-14T01:00:00Z",
            "2026-03-14T01:00:05Z",
            "2026-03-14T01:00:10Z",
        ]
    )
    monkeypatch.setattr(
        process_thread_adapter_module,
        "_now_iso",
        lambda: next(timestamps),
    )

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        await adapter.set_thread_status(status="Reading src/app.py")
        assert adapter._thread_status_started_at_value == "2026-03-14T01:00:00Z"

        await adapter.set_thread_status(status="Reading src/app.py")
        assert adapter._thread_status_started_at_value == "2026-03-14T01:00:00Z"

        await adapter.set_thread_status(status="Thinking")
        assert adapter._thread_status_started_at_value == "2026-03-14T01:00:05Z"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_replaces_generic_storage_read_and_grep_items(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                toolkit="storage",
                tool="read_file",
                arguments=None,
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("item_id") == "read-1"
                    ]
                )
                == 1
            )
        )
        read_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "read-1"
        )
        assert read_event.get_attribute("headline") == "Calling read file"

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                toolkit="storage",
                tool="read_file",
                arguments={"path": "src/app.py", "offset": None},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: read_event.get_attribute("headline") == "Reading src/app.py"
        )
        assert read_event.get_attribute("kind") == "exec"
        assert read_event.get_attribute("path") == "src/app.py"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "read-1"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="read-1",
                error=AgentError(
                    message="read failed",
                    code="tool_call_failed",
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: read_event.get_attribute("state") == "failed")
        assert (
            read_event.get_attribute("headline") == "Attempted to read file src/app.py"
        )
        assert read_event.get_attribute("details") == "read failed"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "read-1"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="grep-1",
                toolkit="storage",
                tool="grep_file",
                arguments=None,
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("item_id") == "grep-1"
                    ]
                )
                == 1
            )
        )
        grep_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "grep-1"
        )
        assert grep_event.get_attribute("headline") == "Calling grep file"

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="grep-1",
                toolkit="storage",
                tool="grep_file",
                arguments={
                    "path": "src/app.py",
                    "pattern": "hello",
                    "offset": None,
                    "before": None,
                    "after": None,
                },
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: grep_event.get_attribute("headline") == "Searching src/app.py"
        )
        assert grep_event.get_attribute("kind") == "exec"
        assert grep_event.get_attribute("path") == "src/app.py"
        assert grep_event.get_attribute("details") == "Pattern: hello"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "grep-1"
                ]
            )
            == 1
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="grep-1",
                error=AgentError(
                    message="search failed",
                    code="tool_call_failed",
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: grep_event.get_attribute("state") == "failed")
        assert (
            grep_event.get_attribute("headline")
            == "Attempted to search file src/app.py"
        )
        assert grep_event.get_attribute("details") == "Pattern: hello\nsearch failed"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "grep-1"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_omits_generic_tool_arguments_and_error_details(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="search",
                tool="lookup",
                arguments={"q": "meshagent"},
            )
        )
        await real_sleep(0)

        tool_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        assert tool_event.get_attribute("details") == ""

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                error=AgentError(
                    message="bad call",
                    code="tool_call_failed",
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: tool_event.get_attribute("state") == "failed")
        assert tool_event.get_attribute("headline") == "Attempted to call Lookup"
        assert tool_event.get_attribute("details") == "bad call"
        assert (
            len(
                [
                    event
                    for event in room.sync.document.event_elements
                    if event.get_attribute("item_id") == "tool-1"
                ]
            )
            == 1
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_omits_inline_computer_result_payloads(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                toolkit="openai",
                tool="computer",
                arguments={"action": "click"},
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="tool-1",
                result=JsonContent(
                    json={
                        "type": "computer_call_output",
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": "data:image/png;base64,ZmFrZS1pbWFnZQ==",
                        },
                    }
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("item_id") == "tool-1"
                    ]
                )
                == 1
            )
        )

        tool_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        assert tool_event.get_attribute("state") == "completed"
        assert tool_event.get_attribute("preview") == ""
        assert "Result:" not in (tool_event.get_attribute("details") or "")
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_persists_generated_images(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="image-1",
                toolkit="openai",
                tool="image_generation",
                arguments={
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                },
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="image-1",
                result=BinaryContent(
                    data=b"fake-image-bytes",
                    headers={
                        "mime_type": "image/png",
                        "output_format": "png",
                        "quality": "high",
                        "size": "1024x1024",
                    },
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                len(
                    [
                        message
                        for message in room.sync.document.message_elements
                        if message.get_attribute("id") == "image-1"
                    ]
                )
                == 1
                and len(
                    [
                        event
                        for event in room.sync.document.event_elements
                        if event.get_attribute("item_id") == "image-1"
                    ]
                )
                == 1
            )
        )

        image_message = next(
            message
            for message in room.sync.document.message_elements
            if message.get_attribute("id") == "image-1"
        )
        image = image_message.get_children_by_tag_name("image")[0]
        tool_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "image-1"
        )

        assert image_message.get_attribute("id") == "image-1"
        assert image_message.get_attribute("turn_id") == "turn-1"
        assert image.get_attribute("mime_type") == "image/png"
        assert image.get_attribute("width") == 1024
        assert image.get_attribute("height") == 1024
        assert image.get_attribute("status") == "completed"

        image_id = image.get_attribute("id")
        assert isinstance(image_id, str) and image_id != ""

        stored_rows = await room.datasets.search(
            table="images",
            where={"id": image_id},
            limit=1,
            select=["data", "mime_type", "created_by"],
        )
        assert stored_rows == [
            {
                "data": b"fake-image-bytes",
                "mime_type": "image/png",
                "created_by": "assistant",
            }
        ]
        stored_annotations = room.datasets.rows["images"][0]["annotations"]
        assert len(stored_annotations) == 3
        assert "fake-image-bytes" not in (tool_event.get_attribute("data") or "")
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_renders_new_thread_tool_as_thread_reference(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="new-thread-1",
                toolkit="chat",
                tool="new_thread",
                arguments={"message": {"text": "make a website thread"}},
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="new-thread-1",
                result=JsonContent(
                    json={
                        "path": "/threads/12345678-1234-5678-1234-567812345678.thread",
                    }
                ),
            )
        )
        await real_sleep(0)

        thread_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "new-thread-1"
        )
        await _wait_for(lambda: thread_event.get_attribute("state") == "completed")
        assert thread_event.get_attribute("kind") == "thread"
        assert thread_event.get_attribute("headline") == "New Thread"
        assert thread_event.get_attribute("details") == ""
        assert (
            thread_event.get_attribute("path")
            == "/threads/12345678-1234-5678-1234-567812345678.thread"
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_inserts_applied_steer_before_post_tool_response() -> (
    None
):
    room = _ThreadRoom(document=_ThreadDocument())
    thread_adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _ToolBoundaryThreadOrderingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=thread_adapter,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "make the website"}],
                ),
            )
        )

        await asyncio.wait_for(llm_adapter.call_started.wait(), timeout=1)

        started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
        turn_id = started_payload["turn_id"]
        process.send(
            Message(
                sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
                data=TurnSteer(
                    type=AGENT_MESSAGE_TURN_STEER,
                    message_id="00000000-0000-0000-0000-000000000041",
                    thread_id="/threads/test.thread",
                    turn_id=turn_id,
                    content=[{"type": "text", "text": "nevermind"}],
                ),
            )
        )

        await _wait_for(
            lambda: (
                len(supervisor.payloads(message_type=AGENT_EVENT_TURN_STEER_ACCEPTED))
                == 1
            )
        )
        await _wait_for(
            lambda: any(
                message.get_attribute("text") == "nevermind"
                for message in room.sync.document.message_elements
            )
        )
        llm_adapter.release_tool_boundary.set()
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )
        await _wait_for(lambda: len(room.sync.document.message_elements) >= 3)
        await _wait_for(lambda: len(room.sync.document.event_elements) >= 1)

        ordered_children = room.sync.document.root.get_children_by_tag_name("messages")[
            0
        ].get_children()
        steer_index = next(
            index
            for index, child in enumerate(ordered_children)
            if child.tag_name == "message"
            and child.get_attribute("text") == "nevermind"
        )
        assistant_index = next(
            index
            for index, child in enumerate(ordered_children)
            if child.tag_name == "message"
            and child.get_attribute("author_name") == "assistant"
            and child.get_attribute("text") == "steered reply"
        )
        assert steer_index < assistant_index
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_strips_room_scheme_from_turn_attachments(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            message=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="/threads/test.thread",
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "file", "url": "room:///audio/prompt.wav"},
                    {"type": "file", "url": "room:///docs/report.pdf"},
                    {"type": "file", "url": "room://images/cat.png"},
                    {"type": "file", "url": "https://example.com/report.pdf"},
                ],
            ),
        )

        await real_sleep(0)

        user_message = room.sync.document.message_elements[0]
        assert user_message.get_attribute("author_name") == "caller"
        assert user_message.get_attribute("turn_id") is None
        assert user_message.get_attribute("text") == "hello"
        assert [
            child.get_attribute("path")
            for child in user_message.get_children_by_tag_name("file")
        ] == [
            "audio/prompt.wav",
            "docs/report.pdf",
            "images/cat.png",
            "https://example.com/report.pdf",
        ]
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_renders_failed_apply_patch_as_attempt(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="patch-1",
                toolkit="patch",
                tool="apply_patch",
                arguments={
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Update File: src/app.py\n"
                        "@@\n"
                        "-old\n"
                        "+new\n"
                        "*** End Patch\n"
                    )
                },
            )
        )
        await real_sleep(0)

        patch_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "patch-1"
        )
        assert patch_event.get_attribute("kind") == "diff"
        assert patch_event.get_attribute("path") == "src/app.py"
        assert patch_event.get_attribute("headline") == "Editing src/app.py"
        assert patch_event.get_attribute("preview") == (
            "*** Begin Patch\n"
            "*** Update File: src/app.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )

        adapter.push_message(
            message=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="patch-1",
                error=AgentError(
                    message="patch rejected",
                    code="tool_call_failed",
                ),
            )
        )
        await real_sleep(0)

        await _wait_for(lambda: patch_event.get_attribute("state") == "failed")
        assert patch_event.get_attribute("headline") == "Attempted to patch src/app.py"
        assert patch_event.get_attribute("details") == "patch rejected"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_updates_apply_patch_from_argument_deltas(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            message=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="start-1",
            )
        )
        adapter.push_message(
            message=AgentToolCallPending(
                type=AGENT_EVENT_TOOL_CALL_PENDING,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="patch-1",
                toolkit="openai",
                tool="apply_patch",
                arguments={},
            )
        )
        await real_sleep(0)

        adapter.push_message(
            message=AgentToolCallArgumentsDelta(
                type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="patch-1",
                delta=("*** Begin Patch\n*** Update File: src/app.ts\n@@\n-old line\n"),
            )
        )
        await _wait_for(lambda: adapter._thread_status_value == "Editing src/app.ts")
        await _wait_for(lambda: adapter._thread_status_lines_removed_value == 1)

        adapter.push_message(
            message=AgentToolCallArgumentsDelta(
                type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                item_id="patch-1",
                delta=(
                    "+new line with enough content to pass the byte display threshold\n"
                    "+another new line\n"
                    "*** End Patch\n"
                ),
            )
        )

        await _wait_for(lambda: adapter._thread_status_lines_added_value == 2)
        await _wait_for(lambda: adapter._thread_status_total_bytes_value is not None)

        patch_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "patch-1"
        )
        assert patch_event.get_attribute("kind") == "diff"
        assert patch_event.get_attribute("path") == "src/app.ts"
        assert patch_event.get_attribute("headline") == "Editing src/app.ts"
        assert patch_event.get_attribute("preview") == (
            "*** Begin Patch\n"
            "*** Update File: src/app.ts\n"
            "@@\n"
            "-old line\n"
            "+new line with enough content to pass the byte display threshold\n"
            "+another new line\n"
            "*** End Patch"
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_restores_thread_state(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    document = _ThreadDocument()
    earlier_user_message = document.root.messages.append_child(
        "message",
        {
            "text": "Earlier question",
            "created_at": "2026-03-11T00:00:00Z",
            "author_name": "caller",
            "role": "user",
        },
    )
    earlier_user_message.append_child(
        "file",
        {"path": "docs/report.pdf"},
    )
    document.root.messages.append_child(
        "message",
        {
            "text": "Earlier answer",
            "created_at": "2026-03-11T00:01:00Z",
            "author_name": "assistant",
            "role": "agent",
        },
    )

    room = _ThreadRoom(document=document)
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=adapter,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "current"}],
                )
            )
        )

        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )

        assert llm_adapter.calls[0]["messages"] == [
            {
                "role": "user",
                "content": "caller said at 2026-03-11T00:00:00Z: Earlier question",
            },
            {
                "role": "user",
                "content": "caller attached a file available at docs/report.pdf",
            },
            {
                "role": "assistant",
                "content": "Earlier answer",
            },
            {
                "role": "user",
                "content": "current",
            },
        ]
    finally:
        await process.stop(supervisor)


@pytest.mark.asyncio
async def test_mesh_document_thread_storage_message_range_separates_messages() -> None:
    document = MeshDocument(schema=thread_schema, on_document_sync=None)
    messages = document.root.append_child("messages")
    messages.append_child(
        "message",
        {
            "text": "hi",
            "created_at": "2026-05-07T20:00:54.054542Z",
            "author_name": "jesse.ezell@timu.com",
            "role": "user",
        },
    )
    messages.append_child(
        "message",
        {
            "text": "hello",
            "created_at": "2026-05-07T20:01:00.000000Z",
            "author_name": "chatbot",
            "role": "agent",
        },
    )

    try:
        adapter = MeshDocumentThreadStorage(
            room=_ThreadRoom(document=document),
            path="/threads/test.thread",
        )
        adapter._thread = document

        result = await adapter.get_message_range.execute(
            context=ToolContext(caller=object()),
            start=0,
            end=2,
        )
    finally:
        document.close()

    assert isinstance(result, TextContent)
    assert result.text == (
        "matching messages:\n"
        "jesse.ezell@timu.com said at 2026-05-07T20:00:54.054542Z: hi\n"
        "chatbot said at 2026-05-07T20:01:00.000000Z: hello"
    )


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_restore_prefers_message_role(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay
        await real_sleep(0)

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    document = _ThreadDocument()
    document.root.messages.append_child(
        "message",
        {
            "text": "External update",
            "created_at": "2026-03-11T00:00:00Z",
            "author_name": "assistant",
            "role": "user",
        },
    )

    room = _ThreadRoom(document=document)
    adapter = MeshDocumentThreadStorage(room=room, path="/threads/test.thread")
    llm_adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=llm_adapter,
        thread_adapter=adapter,
    )

    await process.start(supervisor)
    try:
        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "current"}],
                )
            )
        )

        await asyncio.wait_for(llm_adapter.call_event.wait(), timeout=1)
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )

        assert llm_adapter.calls[0]["messages"] == [
            {
                "role": "user",
                "content": "assistant said at 2026-03-11T00:00:00Z: External update",
            },
            {
                "role": "user",
                "content": "current",
            },
        ]
    finally:
        await process.stop(supervisor)


def test_llm_agent_process_resolves_typed_turn_mcp_config() -> None:
    room = _DownloadRecordingRoom()
    process = _make_llm_agent_process(
        room=room,
        thread_id="/threads/test.thread",
        llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
    )

    options = process._resolve_turn_toolkit_client_options(
        turns=[
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="/threads/test.thread",
                mcp=TurnMCPConfig(
                    servers=[
                        {
                            "server_label": "docs",
                            "server_url": "https://mcp.example.test/mcp",
                            "authorization": "Bearer secret-token",
                        }
                    ]
                ),
                content=[{"type": "text", "text": "use docs"}],
            )
        ],
    )

    assert options == {
        "mcp": {
            "servers": [
                {
                    "server_label": "docs",
                    "server_url": "https://mcp.example.test/mcp",
                    "authorization": "Bearer secret-token",
                }
            ]
        }
    }


def test_dataset_thread_storage_strips_mcp_authorization_from_saved_rows() -> None:
    message = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="/threads/test.thread",
        mcp=TurnMCPConfig(
            servers=[
                {
                    "server_label": "docs",
                    "server_url": "https://mcp.example.test/mcp",
                    "authorization": "Bearer secret-token",
                    "headers": {"x-safe": "kept"},
                }
            ]
        ),
        content=[{"type": "text", "text": "use docs"}],
    )

    data, attachment = DatasetThreadStorage._message_row_data_and_attachment(
        message=message
    )

    assert attachment is None
    assert data["mcp"] == {
        "servers": [
            {
                "server_label": "docs",
                "server_url": "https://mcp.example.test/mcp",
                "headers": {"x-safe": "kept"},
            }
        ]
    }
    assert message.mcp is not None
    assert message.mcp.servers[0]["authorization"] == "Bearer secret-token"


def test_dataset_thread_storage_strips_legacy_mcp_authorization_from_saved_rows() -> (
    None
):
    message = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="/threads/test.thread",
        toolkits={
            "mcp": TurnToolkitConfig(
                client_options={
                    "servers": [
                        {
                            "server_label": "docs",
                            "server_url": "https://mcp.example.test/mcp",
                            "authorization": "Bearer secret-token",
                        }
                    ]
                }
            )
        },
        content=[{"type": "text", "text": "use docs"}],
    )

    data, attachment = DatasetThreadStorage._message_row_data_and_attachment(
        message=message
    )

    assert attachment is None
    assert data["toolkits"] == {
        "mcp": {
            "client_options": {
                "servers": [
                    {
                        "server_label": "docs",
                        "server_url": "https://mcp.example.test/mcp",
                    }
                ]
            }
        }
    }
    assert message.toolkits is not None
    assert message.toolkits["mcp"].client_options is not None
    assert (
        message.toolkits["mcp"].client_options["servers"][0]["authorization"]
        == "Bearer secret-token"
    )
