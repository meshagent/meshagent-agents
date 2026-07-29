from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Protocol

from meshagent.api import Participant

from .messages import (
    AGENT_MESSAGE_MESSAGES_INJECT,
    AgentMessage,
    AgentThreadMessage,
    CloseThread,
    InjectMessages,
    OpenThread,
    ParticipantDisconnect,
    StartThread,
    ThreadStarted,
    ThreadLoaded,
    TurnStartRejected,
)
from .process import AgentSupervisor, Channel, Message

logger = logging.getLogger("threaded-proxy-channel")


class ThreadChannelFactory(Protocol):
    async def start(
        self,
        *,
        request: StartThread,
        sender: Participant | None,
    ) -> Channel: ...

    async def open(
        self,
        *,
        request: OpenThread,
        sender: Participant | None,
    ) -> Channel: ...

    async def close(self, *, channel: Channel) -> None: ...


@dataclass(slots=True)
class _ThreadDestination:
    binding_id: int
    channel: Channel
    supervisor: AgentSupervisor
    clients: set[str]
    pending_message_id: str | None = None
    thread_id: str | None = None
    loading_source_message_id: str | None = None
    loaded_messages: list[AgentMessage] | None = None
    buffered_source_messages: list[Message] | None = None


@dataclass(frozen=True, slots=True)
class _ProxyEvent:
    source: bool
    message: Message
    binding_id: int | None = None


class _ProxyChildSupervisor(AgentSupervisor):
    def __init__(self, *, send_event) -> None:
        super().__init__()
        self._send_event = send_event

    def send(self, message: Message) -> None:
        self._send_event(message)


class ThreadedProxyChannel(Channel):
    """Bidirectionally proxy a source channel to one destination per open thread.

    ``StartThread`` creates a provisional destination through
    :meth:`ThreadChannelFactory.start`. The destination is promoted into the
    thread map when it emits the correlated ``ThreadStarted`` event.
    ``OpenThread`` creates or reuses a destination keyed by ``thread_id``.
    """

    def __init__(
        self,
        *,
        source: Channel,
        destination_factory: ThreadChannelFactory,
        inject_loaded_messages: bool = False,
    ) -> None:
        super().__init__()
        self._source = source
        self._destination_factory = destination_factory
        self._inject_loaded_messages = inject_loaded_messages
        self._bindings: dict[int, _ThreadDestination] = {}
        self._pending_starts: dict[str, int] = {}
        self._destinations: dict[str, int] = {}
        self._next_binding_id = 1
        self._events: asyncio.Queue[_ProxyEvent | None] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._source_supervisor: AgentSupervisor | None = None
        self._last_error: BaseException | None = None

    @property
    def source(self) -> Channel:
        return self._source

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def destination_for_thread(self, *, thread_id: str) -> Channel | None:
        binding_id = self._destinations.get(thread_id)
        binding = self._bindings.get(binding_id) if binding_id is not None else None
        return binding.channel if binding is not None else None

    def _enqueue_source(self, message: Message) -> None:
        if self._dispatcher_task is not None:
            self._events.put_nowait(_ProxyEvent(source=True, message=message))

    def _enqueue_destination(self, *, binding_id: int, message: Message) -> None:
        if self._dispatcher_task is not None:
            self._events.put_nowait(
                _ProxyEvent(
                    source=False,
                    binding_id=binding_id,
                    message=message,
                )
            )

    @staticmethod
    def _client_key(sender: Participant | None) -> str:
        if sender is None or sender.id.strip() == "":
            return "<anonymous>"
        return sender.id

    @staticmethod
    def _forwarded(message: Message) -> Message:
        return Message(
            data=message.data,
            sender=message.sender,
            to_participant_id=message.to_participant_id,
        )

    @staticmethod
    async def _deliver(*, channel: Channel, message: Message) -> bool:
        forwarded = ThreadedProxyChannel._forwarded(message)
        target_id = forwarded.to_participant_id
        if target_id is None:
            return await channel.send_and_wait(forwarded)
        participant = forwarded.sender
        if participant is None or participant.id != target_id:
            participant = Participant(id=target_id, attributes={})
        return await channel.send_agent_message_to_participant_and_wait(
            participant=participant,
            payload=forwarded.data,
        )

    async def _create_destination(
        self,
        *,
        request: StartThread | OpenThread,
        sender: Participant | None,
    ) -> _ThreadDestination:
        if isinstance(request, StartThread):
            channel = await self._destination_factory.start(
                request=request,
                sender=sender,
            )
        else:
            channel = await self._destination_factory.open(
                request=request,
                sender=sender,
            )

        binding_id = self._next_binding_id
        self._next_binding_id += 1
        supervisor = _ProxyChildSupervisor(
            send_event=lambda message: self._enqueue_destination(
                binding_id=binding_id,
                message=message,
            )
        )
        binding = _ThreadDestination(
            binding_id=binding_id,
            channel=channel,
            supervisor=supervisor,
            clients={self._client_key(sender)},
            pending_message_id=(
                request.message_id if isinstance(request, StartThread) else None
            ),
            thread_id=request.thread_id if isinstance(request, OpenThread) else None,
            loaded_messages=[],
            buffered_source_messages=[],
        )
        try:
            await channel.start(supervisor)
        except BaseException:
            with contextlib.suppress(Exception):
                await self._destination_factory.close(channel=channel)
            raise

        self._bindings[binding_id] = binding
        if binding.pending_message_id is not None:
            self._pending_starts[binding.pending_message_id] = binding_id
        if binding.thread_id is not None:
            self._destinations[binding.thread_id] = binding_id
        return binding

    async def _close_destination(self, binding: _ThreadDestination) -> None:
        self._bindings.pop(binding.binding_id, None)
        if binding.pending_message_id is not None:
            current = self._pending_starts.get(binding.pending_message_id)
            if current == binding.binding_id:
                self._pending_starts.pop(binding.pending_message_id, None)
        if binding.thread_id is not None:
            current = self._destinations.get(binding.thread_id)
            if current == binding.binding_id:
                self._destinations.pop(binding.thread_id, None)

        stop_error: BaseException | None = None
        if binding.channel.supervisor is binding.supervisor:
            try:
                await binding.channel.stop(binding.supervisor)
            except BaseException as exc:
                stop_error = exc
        try:
            await self._destination_factory.close(channel=binding.channel)
        except BaseException as exc:
            if stop_error is None:
                stop_error = exc
            else:
                logger.exception(
                    "destination factory cleanup also failed for binding %d",
                    binding.binding_id,
                )
        if stop_error is not None:
            raise stop_error

    async def _binding_for_start(
        self,
        *,
        request: StartThread,
        sender: Participant | None,
    ) -> _ThreadDestination:
        binding_id = self._pending_starts.get(request.message_id)
        if binding_id is not None:
            binding = self._bindings.get(binding_id)
            if binding is not None:
                binding.clients.add(self._client_key(sender))
                return binding
        return await self._create_destination(request=request, sender=sender)

    async def _binding_for_open(
        self,
        *,
        request: OpenThread,
        sender: Participant | None,
    ) -> _ThreadDestination:
        binding_id = self._destinations.get(request.thread_id)
        if binding_id is not None:
            binding = self._bindings.get(binding_id)
            if binding is not None:
                binding.clients.add(self._client_key(sender))
                return binding
        return await self._create_destination(request=request, sender=sender)

    async def _handle_source_message(self, message: Message) -> None:
        data = message.data
        if isinstance(data, AgentThreadMessage):
            binding_id = self._destinations.get(data.thread_id)
            loading_binding = (
                self._bindings.get(binding_id) if binding_id is not None else None
            )
            if (
                loading_binding is not None
                and loading_binding.loading_source_message_id is not None
            ):
                assert loading_binding.buffered_source_messages is not None
                loading_binding.buffered_source_messages.append(message)
                return

        if isinstance(data, StartThread):
            binding = await self._binding_for_start(
                request=data,
                sender=message.sender,
            )
            await self._deliver(channel=binding.channel, message=message)
            return

        if isinstance(data, OpenThread):
            binding = await self._binding_for_open(
                request=data,
                sender=message.sender,
            )
            if self._inject_loaded_messages and data.load is True:
                binding.loading_source_message_id = data.message_id
                assert binding.loaded_messages is not None
                binding.loaded_messages.clear()
            await self._deliver(channel=binding.channel, message=message)
            return

        if isinstance(data, CloseThread):
            binding_id = self._destinations.get(data.thread_id)
            binding = self._bindings.get(binding_id) if binding_id is not None else None
            if binding is None:
                return
            await self._deliver(channel=binding.channel, message=message)
            binding.clients.discard(self._client_key(message.sender))
            if len(binding.clients) == 0:
                await self._close_destination(binding)
            return

        if isinstance(data, ParticipantDisconnect):
            bindings = list(self._bindings.values())
            for binding in bindings:
                await self._deliver(channel=binding.channel, message=message)
                binding.clients.discard(data.participant_id)
                if len(binding.clients) == 0:
                    await self._close_destination(binding)
            return

        if isinstance(data, AgentThreadMessage):
            binding_id = self._destinations.get(data.thread_id)
            binding = self._bindings.get(binding_id) if binding_id is not None else None
            if binding is None:
                raise RuntimeError(
                    f"no destination channel is open for thread {data.thread_id!r}"
                )
            await self._deliver(channel=binding.channel, message=message)
            return

        for binding in list(self._bindings.values()):
            await self._deliver(channel=binding.channel, message=message)

    async def _promote_started_destination(
        self,
        *,
        binding: _ThreadDestination,
        started: ThreadStarted,
    ) -> None:
        if binding.pending_message_id != started.source_message_id:
            raise RuntimeError(
                "destination emitted ThreadStarted for a different StartThread"
            )
        existing_id = self._destinations.get(started.thread_id)
        if existing_id is not None and existing_id != binding.binding_id:
            await self._close_destination(binding)
            raise RuntimeError(
                f"destination already exists for thread {started.thread_id!r}"
            )
        self._pending_starts.pop(started.source_message_id, None)
        binding.pending_message_id = None
        binding.thread_id = started.thread_id
        self._destinations[started.thread_id] = binding.binding_id

    async def _handle_destination_message(
        self,
        *,
        binding_id: int,
        message: Message,
    ) -> None:
        binding = self._bindings.get(binding_id)
        if binding is None:
            return

        data = message.data
        if binding.loading_source_message_id is not None and isinstance(
            data, AgentThreadMessage
        ):
            if binding.thread_id is not None and data.thread_id != binding.thread_id:
                raise RuntimeError(
                    f"destination for thread {binding.thread_id!r} emitted "
                    f"message for thread {data.thread_id!r}"
                )
            assert binding.loaded_messages is not None
            binding.loaded_messages.append(data)
            if not isinstance(data, ThreadLoaded):
                return
            if data.source_message_id != binding.loading_source_message_id:
                raise RuntimeError(
                    "destination emitted ThreadLoaded for a different OpenThread"
                )

            injected = InjectMessages(
                type=AGENT_MESSAGE_MESSAGES_INJECT,
                thread_id=data.thread_id,
                messages=list(binding.loaded_messages),
            )
            await self._deliver(
                channel=self._source,
                message=Message(
                    data=injected,
                    sender=message.sender,
                    to_participant_id=message.to_participant_id,
                ),
            )
            binding.loading_source_message_id = None
            binding.loaded_messages.clear()
            assert binding.buffered_source_messages is not None
            buffered = list(binding.buffered_source_messages)
            binding.buffered_source_messages.clear()
            for buffered_message in buffered:
                await self._handle_source_message(buffered_message)
            return

        if isinstance(data, ThreadStarted):
            await self._promote_started_destination(
                binding=binding,
                started=data,
            )
        elif (
            isinstance(data, TurnStartRejected)
            and binding.pending_message_id == data.source_message_id
        ):
            await self._deliver(channel=self._source, message=message)
            await self._close_destination(binding)
            return
        elif (
            isinstance(data, AgentThreadMessage)
            and binding.thread_id is not None
            and data.thread_id != binding.thread_id
        ):
            raise RuntimeError(
                f"destination for thread {binding.thread_id!r} emitted "
                f"message for thread {data.thread_id!r}"
            )

        await self._deliver(channel=self._source, message=message)

        if isinstance(data, CloseThread):
            await self._close_destination(binding)

    async def _dispatch(self) -> None:
        while True:
            event = await self._events.get()
            if event is None:
                return
            try:
                if event.source:
                    await self._handle_source_message(event.message)
                elif event.binding_id is not None:
                    await self._handle_destination_message(
                        binding_id=event.binding_id,
                        message=event.message,
                    )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self._last_error = exc
                logger.exception("threaded proxy channel failed to forward a message")

    async def on_start(self) -> None:
        self._events = asyncio.Queue()
        self._last_error = None
        self._dispatcher_task = asyncio.create_task(self._dispatch())
        self._source_supervisor = _ProxyChildSupervisor(send_event=self._enqueue_source)
        try:
            await self._source.start(self._source_supervisor)
        except BaseException:
            self._events.put_nowait(None)
            await self._dispatcher_task
            self._dispatcher_task = None
            self._source_supervisor = None
            raise

    async def on_message(self, message: Message) -> None:
        await self._deliver(channel=self._source, message=message)

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        return self._source.send_agent_message_to_participant(
            participant=participant,
            payload=payload,
        )

    async def send_agent_message_to_participant_and_wait(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        return await self._source.send_agent_message_to_participant_and_wait(
            participant=participant,
            payload=payload,
        )

    def get_turn_toolkits(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
    ):
        toolkits = self._source.get_turn_toolkits(
            thread_id=thread_id,
            turn_id=turn_id,
        )
        destination = self.destination_for_thread(thread_id=thread_id)
        if destination is not None:
            toolkits.extend(
                destination.get_turn_toolkits(
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )
        return toolkits

    def get_exposed_toolkits(self):
        return self._source.get_exposed_toolkits()

    async def on_stop(self) -> None:
        source_supervisor = self._source_supervisor
        self._source_supervisor = None
        if (
            source_supervisor is not None
            and self._source.supervisor is source_supervisor
        ):
            with contextlib.suppress(Exception):
                await self._source.stop(source_supervisor)

        dispatcher_task = self._dispatcher_task
        self._dispatcher_task = None
        if dispatcher_task is not None:
            self._events.put_nowait(None)
            await dispatcher_task

        errors: list[BaseException] = []
        for binding in list(self._bindings.values()):
            try:
                await self._close_destination(binding)
            except BaseException as exc:
                errors.append(exc)
        self._bindings.clear()
        self._pending_starts.clear()
        self._destinations.clear()
        if errors:
            raise errors[0]
