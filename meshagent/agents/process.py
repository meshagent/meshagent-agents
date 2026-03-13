from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable, Literal, Optional, TypeVar
from urllib.parse import urlparse

from meshagent.api import Participant, RoomClient
from meshagent.api.messaging import FileContent
from meshagent.agents.adapter import LLMAdapter, ToolCallApprovalRequest
from meshagent.agents.context import AgentSessionContext
from meshagent.tools import RemoteToolkit, ToolContext, Toolkit, ToolkitBuilder
from .process_thread_adapter import AgentProcessThreadAdapter
from .thread_adapter import ThreadAdapter
from .messages import (
    AGENT_EVENT_THREAD_CLEARED,
    AGENT_EVENT_TOOL_CALL_APPROVAL_REQUESTED,
    AGENT_EVENT_TURN_ENDED,
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
    AgentError,
    AgentFileContent,
    AgentMessage,
    AgentTextContent,
    AgentToolCallApprovalRequested,
    ApproveAgentToolCall,
    ClearThread,
    RejectAgentToolCall,
    ThreadCleared,
    TurnEnded,
    TurnInterrupt,
    TurnStartAccepted,
    TurnStart,
    TurnSteerAccepted,
    TurnSteered,
    TurnSteer,
    TurnSteerRejected,
    TurnStarted,
)

logger = logging.getLogger("agent-process")

LifecycleState = Literal["stopped", "starting", "started", "stopping", "failed"]
ChannelState = LifecycleState


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

    def get_agent_toolkits(self) -> list[Toolkit]:
        return []

    def get_exposed_toolkits(self) -> list[RemoteToolkit]:
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

    async def _route(self, message: Message) -> None:
        routed_message = message
        target_processes: list[AgentProcess] | None = None
        message_type = message.data.type
        if message_type == AGENT_MESSAGE_TURN_START:
            turn_start = _coerce_message_data(message.data, TurnStart)
            process = self._process_for_thread(thread_id=turn_start.thread_id)
            if process is None:
                process = self.create_thread_process(turn_start.thread_id)
                self.add_process(process)

            routed_message = self._copy_message(
                message,
                data=turn_start,
            )
            target_processes = [process]

        elif message_type == AGENT_MESSAGE_TURN_STEER:
            turn_steer = _coerce_message_data(message.data, TurnSteer)
            process = self._process_for_thread(thread_id=turn_steer.thread_id)
            routed_message = self._copy_message(
                message,
                data=turn_steer,
            )
            if process is not None and process.turn_id == turn_steer.turn_id:
                target_processes = [process]
            else:
                target_processes = []

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
                process = self.create_thread_process(clear_thread.thread_id)
                self.add_process(process)

            routed_message = self._copy_message(
                message,
                data=clear_thread,
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
        thread_adapter: ThreadAdapter | None = None,
    ) -> None:
        del supervisor
        if thread_adapter is not None:
            if thread_id is None:
                thread_id = thread_adapter.path
            elif thread_adapter.path != thread_id:
                raise ValueError("thread_adapter path must match thread_id")

        self._supervisor: AgentSupervisor | None = None
        self._thread_id = thread_id
        self._thread_adapter = thread_adapter
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

        if self.handles(message):
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueShutDown:
                logger.debug("dropping process message after queue shutdown")

    def handles(self, message: Message) -> bool:
        return False

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
    def thread_adapter(self) -> ThreadAdapter | None:
        return self._thread_adapter

    @property
    def turn_id(self) -> str | None:
        return None

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
                if self._thread_adapter is not None:
                    await self._thread_adapter.start()
                await self.on_start()
                self._run_task = asyncio.create_task(self.run())
                self._state = "started"
            except Exception:
                self._state = "failed"
                if self._thread_adapter is not None:
                    with contextlib.suppress(Exception):
                        await self._thread_adapter.stop()
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
                if self._thread_adapter is not None:
                    await self._thread_adapter.stop()
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


@dataclass(slots=True)
class _QueuedTurnMessage:
    sender: Participant | None
    request: TurnStart | TurnSteer


class LLMAgentProcess(AgentProcess):
    def __init__(
        self,
        *,
        thread_id: str,
        room: RoomClient,
        llm_adapter: LLMAdapter,
        toolkit_builders: Optional[list[ToolkitBuilder]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        thread_adapter: AgentProcessThreadAdapter | None = None,
    ) -> None:
        if thread_adapter is not None and not isinstance(
            thread_adapter, AgentProcessThreadAdapter
        ):
            raise TypeError("thread_adapter must be an AgentProcessThreadAdapter")

        super().__init__(thread_id=thread_id, thread_adapter=thread_adapter)
        self.llm_adapter = llm_adapter
        self._turn_id: str | None = None
        self._handlers: dict[str, Callable[[Message], Awaitable[None]]] = {
            AGENT_MESSAGE_TURN_START: self.on_turn_start,
            AGENT_MESSAGE_TURN_STEER: self.on_turn_steer,
            AGENT_MESSAGE_TURN_INTERRUPT: self.on_turn_interrupt,
            AGENT_MESSAGE_THREAD_CLEAR: self.on_clear_thread,
            AGENT_MESSAGE_TOOL_CALL_APPROVE: self.on_tool_call_approve,
            AGENT_MESSAGE_TOOL_CALL_REJECT: self.on_tool_call_reject,
        }
        self._session_context: AgentSessionContext | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._pending_turns: asyncio.Queue[_QueuedTurn] = asyncio.Queue()
        self._active_turn_queue: asyncio.Queue[_QueuedTurnMessage] | None = None
        self._toolkit_builders = list(toolkit_builders or [])
        self._toolkits = list(toolkits or [])
        self._room = room
        self._pending_tool_call_approvals: dict[str, asyncio.Future[bool]] = {}
        self._active_turn_sender: Participant | None = None
        self._pending_status_messages: list[_QueuedTurnMessage] = []
        self._interrupt_requested_turn_id: str | None = None
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
    def thread_adapter(self) -> AgentProcessThreadAdapter | None:
        adapter = super().thread_adapter
        if adapter is None:
            return None

        if not isinstance(adapter, AgentProcessThreadAdapter):
            raise TypeError("thread_adapter must be an AgentProcessThreadAdapter")

        return adapter

    @property
    def toolkits(self) -> list[Toolkit]:
        return self._toolkits

    @property
    def toolkit_builders(self) -> list[ToolkitBuilder]:
        return self._toolkit_builders

    @property
    def room(self) -> RoomClient:
        return self._room

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        thread_adapter = self.thread_adapter
        if thread_adapter is not None:
            thread_adapter.push_message(message=payload, sender=sender)

        super().emit(sender=sender, payload=payload)

    def handles(self, message: Message) -> bool:
        message_type = message.data.type
        if message_type not in self._handlers:
            return False

        return message.data.thread_id == self._thread_id

    async def on_session_context_created(self) -> None:
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

    async def ensure_session_context(self, *, turn_id: str) -> AgentSessionContext:
        if self._session_context is None:
            self._session_context = self.llm_adapter.create_session()
            await self.on_session_context_created()
            thread_adapter = self.thread_adapter
            if thread_adapter is not None:
                thread_adapter.restore_session_context(context=self._session_context)
            await self.on_restore_session_context(turn_id, self._session_context)
            await self._session_context.start()

        return self._session_context

    def _record_applied_turns(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
    ) -> None:
        thread_adapter = self.thread_adapter
        if thread_adapter is None:
            return

        for queued_message in queued_messages:
            thread_adapter.push_message(
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
            "content": content if isinstance(content, list) else [],
        }

    async def _sync_pending_status_messages(self) -> None:
        thread_adapter = self.thread_adapter
        if thread_adapter is None:
            return

        await thread_adapter.set_pending_messages(
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

    async def _resolve_requested_toolkits(
        self,
        *,
        model: str,
        requested_toolkits: list[dict[str, Any]],
    ) -> list[Toolkit]:
        toolkits: list[Toolkit] = []
        for raw_config in requested_toolkits:
            toolkit_name = raw_config.get("name")
            if not isinstance(toolkit_name, str):
                raise ValueError("toolkit config must include a string `name`")

            matching_builder: ToolkitBuilder | None = None
            for builder in self._toolkit_builders:
                if builder.name == toolkit_name:
                    matching_builder = builder
                    break

            if matching_builder is None:
                raise ValueError(f"tool cannot be configured: {raw_config}")

            typed_config = matching_builder.type.model_validate(raw_config)
            toolkits.append(
                await matching_builder.make(
                    room=self._room,
                    model=model,
                    config=typed_config,
                )
            )

        return toolkits

    async def _build_turn_toolkits(
        self,
        *,
        model: str,
        turns: list[TurnStart | TurnSteer],
    ) -> list[Toolkit]:
        combined_toolkits = [*self._toolkits]
        supervisor = self.supervisor
        if supervisor is not None:
            for channel in supervisor.channels:
                combined_toolkits.extend(channel.get_agent_toolkits())
        for turn in turns:
            if turn.toolkits is not None and len(turn.toolkits) > 0:
                combined_toolkits.extend(
                    await self._resolve_requested_toolkits(
                        model=model,
                        requested_toolkits=turn.toolkits,
                    )
                )
        return combined_toolkits

    @staticmethod
    def _normalize_room_storage_path(*, url: str) -> str:
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

    @staticmethod
    def _guess_url_mime_type(*, url: str) -> str | None:
        guessed_mime_type, _ = mimetypes.guess_type(urlparse(url).path)
        return guessed_mime_type

    def _append_downloaded_file_content(
        self,
        *,
        session: AgentSessionContext,
        file_content: FileContent,
        url: str,
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

        session.append_user_message(f"the user attached a file available at {url}")

    def _append_remote_file_content(
        self,
        *,
        session: AgentSessionContext,
        url: str,
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

        session.append_user_message(f"the user attached a file available at {url}")

    async def _append_file_content(
        self,
        *,
        session: AgentSessionContext,
        url: str,
    ) -> None:
        if urlparse(url).scheme == "room":
            room_path = self._normalize_room_storage_path(url=url)
            file_content = await self._room.storage.download(path=room_path)
            self._append_downloaded_file_content(
                session=session,
                file_content=file_content,
                url=url,
            )
            return

        self._append_remote_file_content(session=session, url=url)

    async def _append_turn_content(
        self,
        *,
        session: AgentSessionContext,
        turns: list[TurnStart | TurnSteer],
    ) -> None:
        for turn in turns:
            for item in turn.content:
                if isinstance(item, AgentTextContent):
                    session.append_user_message(item.text)
                elif isinstance(item, AgentFileContent):
                    await self._append_file_content(session=session, url=item.url)

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

    def _drain_pending_turns(self) -> list[_QueuedTurn]:
        drained_turns: list[_QueuedTurn] = []
        while True:
            try:
                drained_turns.append(self._pending_turns.get_nowait())
            except asyncio.QueueEmpty:
                break

        return drained_turns

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

    async def _execute_turn_batch(
        self,
        *,
        queued_messages: list[_QueuedTurnMessage],
        session: AgentSessionContext,
        model: str,
    ) -> None:
        turns = [queued_message.request for queued_message in queued_messages]
        sender = self._sender_for_turn_batch(queued_messages=queued_messages)
        self._record_applied_turns(queued_messages=queued_messages)
        await self._append_turn_content(session=session, turns=turns)
        self._emit_turn_steered_events(queued_messages=queued_messages)
        combined_toolkits = await self._build_turn_toolkits(model=model, turns=turns)
        turn_id = self._turn_id
        thread_id = self.thread_id
        if turn_id is None or thread_id is None:
            raise RuntimeError("turn publisher requested without an active turn")

        def publish_event(message: AgentMessage) -> None:
            if self._interrupt_requested_turn_id == turn_id:
                return
            self.emit(sender=sender, payload=message)

        handle_event = self.llm_adapter.make_agent_event_publisher(
            turn_id=turn_id,
            thread_id=thread_id,
            callback=publish_event,
        )

        self._active_turn_sender = sender
        try:
            await self.llm_adapter.next(
                context=session,
                toolkits=combined_toolkits,
                room=self._room,
                event_handler=handle_event,
                model=model,
                on_behalf_of=sender,
            )
        finally:
            self._active_turn_sender = None

    async def _run_next_turn(self) -> None:
        queued_turn = await self._pending_turns.get()
        await self._remove_pending_status_messages(
            queued_messages=[
                _QueuedTurnMessage(
                    sender=queued_turn.sender,
                    request=queued_turn.request,
                )
            ]
        )
        turn_id = str(uuid.uuid4())
        self._turn_id = turn_id
        self._interrupt_requested_turn_id = None
        active_turn_queue: asyncio.Queue[_QueuedTurnMessage] = asyncio.Queue()
        self._active_turn_queue = active_turn_queue
        active_turn_queue.put_nowait(
            _QueuedTurnMessage(
                sender=queued_turn.sender,
                request=queued_turn.request,
            )
        )
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
            original_instructions = session.instructions
            if queued_turn.request.instructions is not None:
                session.instructions = queued_turn.request.instructions

            while True:
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
                if self._interrupt_requested_turn_id == turn_id:
                    raise asyncio.CancelledError
                if active_turn_queue.empty():
                    break
        except asyncio.CancelledError:
            error = self._turn_error(message="turn cancelled", code="cancelled")
        except Exception as exc:
            logger.exception("turn failed")
            error_message = str(exc) if str(exc) != "" else exc.__class__.__name__
            error = self._turn_error(
                message=error_message,
                code=exc.__class__.__name__,
            )
        finally:
            self._interrupt_requested_turn_id = None
            self._active_turn_queue = None
            queued_turn_messages = self._drain_queued_turn_messages(
                active_turn_queue=active_turn_queue,
            )
            if len(queued_turn_messages) > 0:
                await self._remove_pending_status_messages(
                    queued_messages=queued_turn_messages
                )
                self._emit_rejected_queued_turn_steers(
                    queued_messages=queued_turn_messages,
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

    def _schedule_next_turn(self) -> None:
        if (
            self._turn_task is not None
            or self._stop.is_set()
            or self._pending_turns.empty()
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

        if not self._stop.is_set():
            self._schedule_next_turn()

    async def on_turn_start(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnStart)
        should_track_pending_status = (
            self._turn_task is not None or not self._pending_turns.empty()
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
                source_message_id=turn.message_id,
            ),
        )
        if should_track_pending_status:
            await self._add_pending_status_messages(
                queued_messages=[
                    _QueuedTurnMessage(
                        sender=message.sender,
                        request=turn,
                    )
                ]
            )
        self._schedule_next_turn()

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

        active_turn_queue.put_nowait(
            _QueuedTurnMessage(
                sender=message.sender,
                request=turn,
            )
        )
        await self._add_pending_status_messages(
            queued_messages=[
                _QueuedTurnMessage(
                    sender=message.sender,
                    request=turn,
                )
            ]
        )
        self.emit(
            sender=message.sender,
            payload=TurnSteerAccepted(
                type=AGENT_EVENT_TURN_STEER_ACCEPTED,
                thread_id=turn.thread_id,
                turn_id=turn.turn_id,
                source_message_id=turn.message_id,
            ),
        )

    async def on_turn_interrupt(self, message: Message) -> None:
        turn = _coerce_message_data(message.data, TurnInterrupt)
        if self._turn_id != turn.turn_id or self._turn_task is None:
            return

        self._interrupt_requested_turn_id = turn.turn_id
        self._active_turn_queue = None
        self._turn_task.cancel()

    async def on_clear_thread(self, message: Message) -> None:
        clear_thread = _coerce_message_data(message.data, ClearThread)
        if self.thread_id != clear_thread.thread_id:
            return

        turn_task = self._turn_task
        if turn_task is not None:
            turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn_task

        self._active_turn_queue = None
        self._active_turn_sender = None
        self._turn_id = None
        self._drain_pending_turns()
        self._cancel_pending_tool_call_approvals()
        await self._clear_pending_status_messages()

        if self._session_context is not None:
            await self._session_context.close()
            self._session_context = None

        thread_adapter = self.thread_adapter
        if thread_adapter is not None:
            await thread_adapter.clear_thread()

        self.emit(
            sender=message.sender,
            payload=ThreadCleared(
                type=AGENT_EVENT_THREAD_CLEARED,
                thread_id=clear_thread.thread_id,
                source_message_id=clear_thread.message_id,
            ),
        )

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
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._active_turn_queue = None
        self._active_turn_sender = None
        self._cancel_pending_tool_call_approvals()
        await self._clear_pending_status_messages()

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
