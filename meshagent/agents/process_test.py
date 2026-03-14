import asyncio
import uuid
from typing import Any, Literal


import pytest
from pydantic import BaseModel, ValidationError

import meshagent.agents.process_thread_adapter as process_thread_adapter_module
import meshagent.agents.thread_adapter as thread_adapter_module
from meshagent.agents import AgentProcessThreadAdapter
from meshagent.agents.adapter import LLMAdapter, ToolCallApprovalRequest
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.messages import (
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_REASONING_CONTENT_DELTA,
    AGENT_EVENT_REASONING_CONTENT_ENDED,
    AGENT_EVENT_REASONING_CONTENT_STARTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_MESSAGE_THREAD_CLEAR,
    AGENT_MESSAGE_TOOL_CALL_APPROVE,
    AGENT_MESSAGE_TOOL_CALL_REJECT,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    ApproveAgentToolCall,
    AgentError,
    AgentMessage,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallStarted,
    AgentToolCallEnded,
    ClearThread,
    RejectAgentToolCall,
    TurnStart,
    TurnEnded,
    TurnInterrupted,
    TurnInterrupt,
    TurnStarted,
    TurnSteerAccepted,
    TurnSteer,
)
from meshagent.agents.process import (
    AgentProcess,
    AgentSupervisor,
    Channel,
    LLMAgentProcess,
    Message,
)
from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.api import Participant
from meshagent.api.messaging import FileContent
from meshagent.api.messaging import JsonContent
from meshagent.api.messaging import TextContent
from meshagent.tools import ToolContext, Toolkit, ToolkitBuilder


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
            message.data.type in {AGENT_MESSAGE_TURN_START, AGENT_MESSAGE_THREAD_CLEAR}
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


class _PayloadMessage(AgentMessage):
    payload: str | None = None


class _ThreadCreatingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.created_processes: list[_ThreadRecordingProcess] = []

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        process = _ThreadRecordingProcess(thread_id=thread_id)
        self.created_processes.append(process)
        return process


class _GenericThreadAdapter(ThreadAdapter):
    async def handle_custom_event(
        self,
        *,
        messages,
        event: dict,
    ) -> None:
        del messages
        del event

    async def _process_llm_events(self) -> None:
        return None


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
        self.file_url_calls: list[str] = []

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

    def append_file_url(self, *, url: str) -> dict:
        self.file_url_calls.append(url)
        message = {
            "role": "user",
            "content": [{"type": "file-url", "url": url}],
        }
        self.messages.append(message)
        return message


class _ExampleToolkitConfig(BaseModel):
    name: Literal["example"] = "example"
    enabled: bool = False


class _ExampleToolkitBuilder(ToolkitBuilder):
    def __init__(self) -> None:
        super().__init__(name="example", type=_ExampleToolkitConfig)
        self.calls: list[dict[str, Any]] = []

    async def make(
        self,
        *,
        room,
        model: str,
        config: _ExampleToolkitConfig,
    ) -> Toolkit:
        self.calls.append({"room": room, "model": model, "config": config})
        return Toolkit(name="example", tools=[])


class _RecordingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self, *, session: _LifecycleSession | None = None) -> None:
        self.session = session if session is not None else _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self) -> AgentSessionContext:
        return self.session

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del output_schema
        del steering_callback
        del on_behalf_of
        del options
        self.calls.append(
            {
                "context": context,
                "room": room,
                "messages": [*context.messages],
                "metadata": dict(context.metadata),
                "toolkits": [toolkit.name for toolkit in toolkits],
                "model": model,
            }
        )
        if event_handler is not None:
            event_handler({"type": "adapter.event", "call_index": len(self.calls) - 1})
        self.call_event.set()
        return {"ok": True}


class _PublishingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
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

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del context
        del room
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

    def create_session(self) -> AgentSessionContext:
        return self.session

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
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
                "room": room,
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

    def create_session(self) -> AgentSessionContext:
        return self.session

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del room
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

    def create_session(self) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
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

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del context
        del room
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


class _DownloadRecordingRoom:
    def __init__(self, *, files: dict[str, FileContent] | None = None) -> None:
        self.storage = _DownloadRecordingStorage(files=files or {})


class _ThreadParticipant(Participant):
    def __init__(self, *, name: str, participant_id: str) -> None:
        super().__init__(id=participant_id, attributes={"name": name})


class _ThreadLocalParticipant(_ThreadParticipant):
    def __init__(self) -> None:
        super().__init__(name="assistant", participant_id="assistant-id")
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

    def create_session(self) -> AgentSessionContext:
        return self.session

    def set_tool_call_approval_handler(self, handler) -> None:
        self._approval_handler = handler

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
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
                room=room,
                caller=_ToolCallerParticipant(),  # type: ignore[arg-type]
            ),
            request,
        )
        self.approval_decisions.append(decision)
        self.approval_resolved.set()
        return {"approved": decision}


class _ThreadPublishingLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.call_event = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self) -> AgentSessionContext:
        return self.session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
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

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del room
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


class _ClearableLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.sessions: list[_LifecycleSession] = []
        self.calls: list[dict[str, Any]] = []
        self.first_call_started = asyncio.Event()
        self.first_call_cancelled = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self) -> AgentSessionContext:
        session = _LifecycleSession()
        self.sessions.append(session)
        return session

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback,
    ):
        def publish(event: dict[str, Any]) -> None:
            callback(
                AgentTextContentStarted(
                    type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=event["item_id"],
                )
            )
            callback(
                AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=event["item_id"],
                    text=event["text"],
                )
            )
            callback(
                AgentTextContentEnded(
                    type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=event["item_id"],
                )
            )

        return publish

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del room
        del toolkits
        del output_schema
        del model
        del steering_callback
        del on_behalf_of
        del options

        call_index = len(self.calls)
        self.calls.append({"messages": [*context.messages], "context": context})
        if event_handler is not None:
            event_handler(
                {
                    "item_id": f"assistant-{call_index + 1}",
                    "text": f"reply-{call_index + 1}",
                }
            )

        if call_index == 0:
            self.first_call_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.first_call_cancelled.set()
                raise

        return {"ok": True}


class _CancellationIgnoringLLMAdapter(LLMAdapter[dict[str, Any]]):
    def __init__(self) -> None:
        self.session = _LifecycleSession()
        self.calls: list[dict[str, Any]] = []
        self.first_call_started = asyncio.Event()
        self.first_call_cancelled = asyncio.Event()

    def default_model(self) -> str:
        return "default-model"

    def create_session(self) -> AgentSessionContext:
        return self.session

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room,
        toolkits: list[Toolkit],
        output_schema: dict | None = None,
        event_handler=None,
        steering_callback=None,
        model: str | None = None,
        on_behalf_of=None,
        options: dict | None = None,
    ) -> Any:
        del room
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
        }
    ]

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_continues_when_channel_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = AgentSupervisor()
    failing_channel = _FailingStartChannel()
    healthy_channel = _LifecycleChannel()
    supervisor.add_channel(failing_channel)
    supervisor.add_channel(healthy_channel)

    with caplog.at_level("ERROR", logger="agent-process"):
        await supervisor.start()

    await asyncio.wait_for(healthy_channel.start_event.wait(), timeout=1)

    assert failing_channel.state == "failed"
    assert failing_channel.supervisor is None
    assert healthy_channel.state == "started"
    assert healthy_channel.supervisor is supervisor
    assert "channel _FailingStartChannel failed during start; continuing" in caplog.text

    await supervisor.stop()


@pytest.mark.asyncio
async def test_agent_supervisor_continues_when_process_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = AgentSupervisor()
    failing_process = _FailingStartProcess()
    healthy_process = _RecordingProcess(handled_type="work")
    supervisor.add_process(failing_process)
    supervisor.add_process(healthy_process)

    with caplog.at_level("ERROR", logger="agent-process"):
        await supervisor.start()

    await asyncio.wait_for(healthy_process.start_event.wait(), timeout=1)

    assert failing_process.state == "failed"
    assert failing_process.supervisor is None
    assert healthy_process.state == "started"
    assert healthy_process.supervisor is supervisor
    assert "process _FailingStartProcess failed during start; continuing" in caplog.text

    await supervisor.stop()


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


@pytest.mark.asyncio
async def test_llm_agent_process_builds_requested_toolkits_for_turn_start() -> None:
    session = _LifecycleSession()
    adapter = _RecordingLLMAdapter(session=session)
    builder = _ExampleToolkitBuilder()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
        llm_adapter=adapter,
        toolkit_builders=[builder],
        toolkits=[Toolkit(name="static", tools=[])],
    )

    await process.start(supervisor)

    turn_start_message_id = "00000000-0000-0000-0000-000000000001"
    process.send(
        Message(
            data=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=turn_start_message_id,
                thread_id="thread-1",
                content=[{"type": "text", "text": "hello"}],
                toolkits=[{"name": "example", "enabled": True}],
                model="custom-model",
                instructions="be concise",
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert session.started == 1
    assert session.instructions is None
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert adapter.calls[0]["toolkits"] == ["static", "example"]
    assert adapter.calls[0]["model"] == "custom-model"
    assert len(builder.calls) == 1
    assert builder.calls[0]["model"] == "custom-model"
    assert builder.calls[0]["config"].enabled is True
    accepted_payload = supervisor.payloads(
        message_type=AGENT_EVENT_TURN_START_ACCEPTED
    )[0]
    uuid.UUID(accepted_payload["message_id"])
    assert accepted_payload["thread_id"] == "thread-1"
    assert accepted_payload["source_message_id"] == turn_start_message_id
    started_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0]
    uuid.UUID(started_payload["message_id"])
    assert started_payload["thread_id"] == "thread-1"
    assert started_payload["source_message_id"] == turn_start_message_id
    ended_payload = supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)[0]
    uuid.UUID(ended_payload["message_id"])
    assert ended_payload["thread_id"] == "thread-1"
    assert ended_payload["error"] is None

    await process.stop(supervisor)

    assert session.closed == 1


@pytest.mark.asyncio
async def test_llm_agent_process_passes_turn_id_to_restore_session_context() -> None:
    adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = _RestoringLLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_uses_adapter_agent_event_publisher() -> None:
    adapter = _PublishingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    assert delta_payload["turn_id"] == started_payload["turn_id"]
    assert delta_payload["text"] == "hello"
    assert ended_payload["turn_id"] == started_payload["turn_id"]

    await process.stop(supervisor)


@pytest.mark.asyncio
async def test_llm_agent_process_processes_queued_steer_messages_before_turn_end() -> (
    None
):
    adapter = _QueuedSteerLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    assert adapter.calls[1]["metadata"] == {}
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
async def test_llm_agent_process_calls_on_turn_steer_before_interrupt_continuation() -> (
    None
):
    adapter = _InterruptAwareQueuedSteerLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=object(),  # type: ignore[arg-type]
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=_DownloadRecordingRoom(),
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=_DownloadRecordingRoom(),
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
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=_DownloadRecordingRoom(),
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
    assert session.file_url_calls == ["https://example.com/report.pdf"]
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
        }
    )
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="thread-1",
        room=room,
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
                ],
            )
        )
    )

    await asyncio.wait_for(adapter.call_event.wait(), timeout=1)
    await _wait_for(
        lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
    )

    assert room.storage.download_calls == ["images/cat.png", "docs/report.pdf"]
    assert session.image_message_calls == [
        {"mime_type": "image/png", "data": b"png-bytes"}
    ]
    assert session.file_message_calls == [
        {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "data": b"%PDF-1.7",
        }
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
    ]

    await process.stop(supervisor)


def test_llm_agent_process_requires_agent_process_thread_adapter() -> None:
    room = _ThreadRoom(document=_ThreadDocument())

    with pytest.raises(TypeError, match="AgentProcessThreadAdapter"):
        LLMAgentProcess(
            thread_id="/threads/test.thread",
            room=room,
            llm_adapter=_RecordingLLMAdapter(session=_LifecycleSession()),
            thread_adapter=_GenericThreadAdapter(
                room=room,
                path="/threads/test.thread",
            ),
        )


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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")
    llm_adapter = _ThreadPublishingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        room=room,
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
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )
        await _wait_for(
            lambda: len(supervisor.payloads(message_type=AGENT_EVENT_TURN_ENDED)) == 1
        )
        await _wait_for(lambda: len(room.sync.document.message_elements) == 2)
        await _wait_for(lambda: len(room.sync.document.event_elements) >= 2)
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    None,
                )
                in room.local_participant.set_attribute_calls
            )
        )

        user_message = room.sync.document.message_elements[0]
        assert user_message.get_attribute("author_name") == "caller"
        assert user_message.get_attribute("text") == "hello from caller"
        assert [
            child.get_attribute("path")
            for child in user_message.get_children_by_tag_name("file")
        ] == ["https://example.com/report.pdf"]

        assistant_message = room.sync.document.message_elements[1]
        assert assistant_message.get_attribute("author_name") == "assistant"
        assert assistant_message.get_attribute("text") == "hello"

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
        assert turn_event.get_attribute("state") == "completed"

        turn_id = supervisor.payloads(message_type=AGENT_EVENT_TURN_STARTED)[0][
            "turn_id"
        ]
        assert user_message.get_attribute("turn_id") == turn_id
        assert assistant_message.get_attribute("turn_id") == turn_id
        assert tool_event.get_attribute("turn_id") == turn_id
        assert turn_event.get_attribute("turn_id") == turn_id

        assert room.sync.document.member_names == ["assistant", "caller"]
        assert (
            "thread.status.mode./threads/test.thread",
            "steerable",
        ) in room.local_participant.set_attribute_calls
    finally:
        await process.stop(supervisor)

    assert room.sync.close_calls == ["/threads/test.thread"]


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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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

        assert user_message.get_attribute("turn_id") == "turn-1"
        assert assistant_message.get_attribute("turn_id") == "turn-1"
        assert reasoning.get_attribute("turn_id") == "turn-1"
        assert tool_event.get_attribute("turn_id") == "turn-1"
        assert turn_event.get_attribute("turn_id") == "turn-1"
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
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
                arguments={"action": {"command": "sed -n '1,20p' src/app.py"}},
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Reading src/app.py",
                )
                in room.local_participant.set_attribute_calls
            )
        )
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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )

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

        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    None,
                )
                in room.local_participant.set_attribute_calls
            )
        )
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_agent_process_thread_adapter_writes_preview_for_fallback_running_command(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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

        exec_event = next(
            event
            for event in room.sync.document.event_elements
            if event.get_attribute("item_id") == "tool-1"
        )
        await _wait_for(
            lambda: (
                exec_event.get_attribute("state") == "in_progress"
                and exec_event.get_attribute("headline") == "Running Command"
            )
        )

        assert exec_event.get_attribute("kind") == "exec"
        assert exec_event.get_attribute("path") == ""
        assert exec_event.get_attribute("preview") == command
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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

        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Exploring /website",
                )
                in room.local_participant.set_attribute_calls
            )
        )

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
async def test_agent_process_thread_adapter_renders_cd_prefixed_shell_heredoc_write_as_file_event(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
            "..."
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
async def test_agent_process_thread_adapter_groups_multi_file_shell_heredoc_writes(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )

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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Writing src/app.py",
                )
                in room.local_participant.set_attribute_calls
            )
        )
        assert (
            room.local_participant.get_attribute(
                "thread.status.pending_item_id./threads/test.thread"
            )
            == "write-1"
        )

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
        assert write_event.get_attribute("details") == ""
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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )
        assert (
            room.local_participant.get_attribute(
                "thread.status.pending_item_id./threads/test.thread"
            )
            is None
        )
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
                (
                    "thread.status./threads/test.thread",
                    "Reading src/app.py",
                )
                in room.local_participant.set_attribute_calls
            )
        )
        assert (
            room.local_participant.get_attribute(
                "thread.status.pending_item_id./threads/test.thread"
            )
            == "read-1"
        )

        adapter.push_message(
            message=TurnInterrupted(
                type=AGENT_EVENT_TURN_INTERRUPTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="interrupt-1",
            )
        )
        await real_sleep(0)

        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )
        assert (
            room.local_participant.get_attribute(
                "thread.status.pending_item_id./threads/test.thread"
            )
            is None
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
        await _wait_for(
            lambda: (
                (
                    "thread.status./threads/test.thread",
                    "Thinking",
                )
                in room.local_participant.set_attribute_calls
            )
        )

        previous_calls = list(room.local_participant.set_attribute_calls)
        adapter.push_message(
            message=TurnSteerAccepted(
                type=AGENT_EVENT_TURN_STEER_ACCEPTED,
                thread_id="/threads/test.thread",
                turn_id="turn-1",
                source_message_id="steer-1",
            )
        )
        await real_sleep(0)

        assert (
            "thread.status./threads/test.thread",
            "Queued turn steering",
        ) not in room.local_participant.set_attribute_calls
        assert room.local_participant.set_attribute_calls == previous_calls
    finally:
        await adapter.stop()


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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        await adapter.set_thread_status(status="Reading src/app.py")
        assert (
            room.local_participant.get_attribute(
                "thread.status.started_at./threads/test.thread"
            )
            == "2026-03-14T01:00:00Z"
        )

        await adapter.set_thread_status(status="Reading src/app.py")
        assert (
            room.local_participant.get_attribute(
                "thread.status.started_at./threads/test.thread"
            )
            == "2026-03-14T01:00:00Z"
        )

        await adapter.set_thread_status(status="Thinking")
        assert (
            room.local_participant.get_attribute(
                "thread.status.started_at./threads/test.thread"
            )
            == "2026-03-14T01:00:05Z"
        )
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
        assert read_event.get_attribute("headline") == "Calling Read File"

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
        assert grep_event.get_attribute("headline") == "Calling Grep File"

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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
        assert tool_event.get_attribute("headline") == "Lookup Failed"
        assert tool_event.get_attribute("details") == ""
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
async def test_agent_process_thread_adapter_renders_new_thread_tool_as_thread_reference(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

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
                        "path": "/threads/generated.thread",
                        "name": "Anthropic Website",
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
        assert thread_event.get_attribute("headline") == "Anthropic Website"
        assert thread_event.get_attribute("details") == ""
        assert thread_event.get_attribute("path") == "/threads/generated.thread"
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_inserts_applied_steer_before_post_tool_response() -> (
    None
):
    room = _ThreadRoom(document=_ThreadDocument())
    thread_adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")
    llm_adapter = _ToolBoundaryThreadOrderingLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        room=room,
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
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")

    await adapter.start()
    try:
        adapter.push_message(
            sender=_ThreadParticipant(name="caller", participant_id="caller-id"),
            message=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id="/threads/test.thread",
                content=[
                    {"type": "text", "text": "hello"},
                    {"type": "file", "url": "room:///docs/report.pdf"},
                    {"type": "file", "url": "room://images/cat.png"},
                    {"type": "file", "url": "https://example.com/report.pdf"},
                ],
            ),
        )

        await real_sleep(0)

        user_message = room.sync.document.message_elements[0]
        assert user_message.get_attribute("author_name") == "caller"
        assert user_message.get_attribute("text") == "hello"
        assert [
            child.get_attribute("path")
            for child in user_message.get_children_by_tag_name("file")
        ] == [
            "docs/report.pdf",
            "images/cat.png",
            "https://example.com/report.pdf",
        ]
    finally:
        await adapter.stop()


@pytest.mark.asyncio
async def test_llm_agent_process_thread_adapter_restores_thread_state(
    monkeypatch,
) -> None:
    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    document = _ThreadDocument()
    earlier_user_message = document.root.messages.append_child(
        "message",
        {
            "text": "Earlier question",
            "created_at": "2026-03-11T00:00:00Z",
            "author_name": "caller",
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
        },
    )

    room = _ThreadRoom(document=document)
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")
    llm_adapter = _RecordingLLMAdapter(session=_LifecycleSession())
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        room=room,
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
async def test_llm_agent_process_clear_thread_resets_thread_and_session_context(
    monkeypatch,
) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(thread_adapter_module.asyncio, "sleep", _fast_sleep)

    async def _wait_for_real_sleep(
        predicate,
        *,
        timeout: float = 1,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise asyncio.TimeoutError()
            await real_sleep(0.01)

    room = _ThreadRoom(document=_ThreadDocument())
    adapter = AgentProcessThreadAdapter(room=room, path="/threads/test.thread")
    llm_adapter = _ClearableLLMAdapter()
    supervisor = _RecordingSupervisor()
    process = LLMAgentProcess(
        thread_id="/threads/test.thread",
        room=room,
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
                    content=[{"type": "text", "text": "first"}],
                )
            )
        )

        await asyncio.wait_for(llm_adapter.first_call_started.wait(), timeout=1)
        await _wait_for_real_sleep(
            lambda: len(room.sync.document.message_elements) == 2
        )
        await _wait_for_real_sleep(lambda: len(room.sync.document.event_elements) >= 1)

        clear_thread = ClearThread(
            type=AGENT_MESSAGE_THREAD_CLEAR,
            thread_id="/threads/test.thread",
        )
        process.send(
            Message(
                data=clear_thread,
            )
        )

        await asyncio.wait_for(llm_adapter.first_call_cancelled.wait(), timeout=1)
        await _wait_for_real_sleep(
            lambda: len(room.sync.document.message_elements) == 0
        )
        await _wait_for_real_sleep(lambda: len(room.sync.document.event_elements) == 0)
        await _wait_for_real_sleep(lambda: process.session_context is None)

        assert len(llm_adapter.sessions) == 1
        assert llm_adapter.sessions[0].closed == 1
        thread_cleared_events = supervisor.payloads(
            message_type=AGENT_EVENT_THREAD_CLEARED
        )
        assert len(thread_cleared_events) == 1
        assert thread_cleared_events[0]["thread_id"] == "/threads/test.thread"
        assert thread_cleared_events[0]["source_message_id"] == clear_thread.message_id

        process.send(
            Message(
                data=TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="/threads/test.thread",
                    content=[{"type": "text", "text": "second"}],
                )
            )
        )

        await _wait_for_real_sleep(lambda: len(llm_adapter.calls) == 2)
        await _wait_for_real_sleep(
            lambda: len(room.sync.document.message_elements) == 2
        )

        assert len(llm_adapter.sessions) == 2
        assert llm_adapter.calls[1]["messages"] == [
            {
                "role": "user",
                "content": "second",
            }
        ]

        assistant_message = room.sync.document.message_elements[1]
        assert assistant_message.get_attribute("text") == "reply-2"
    finally:
        await process.stop(supervisor)
