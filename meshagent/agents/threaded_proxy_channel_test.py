from __future__ import annotations

import asyncio

import pytest

from meshagent.api import Participant

from .messages import (
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_MESSAGE_MESSAGES_INJECT,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_TURN_START,
    AgentError,
    AgentTextContent,
    CloseThread,
    OpenThread,
    InjectMessages,
    StartThread,
    ThreadStarted,
    ThreadLoaded,
    TurnStart,
    TurnStartRejected,
)
from .process import AgentSupervisor, Channel, Message
from .threaded_proxy_channel import ThreadedProxyChannel


async def _wait_until(predicate, *, message: str) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(message)


class _RecordingChannel(Channel):
    def __init__(self, *, fail_start: bool = False) -> None:
        super().__init__()
        self.fail_start = fail_start
        self.received: list[Message] = []
        self.started = 0
        self.stopped = 0

    async def on_start(self) -> None:
        if self.fail_start:
            raise RuntimeError("destination start failed")
        self.started += 1

    async def on_message(self, message: Message) -> None:
        self.received.append(message)

    async def on_stop(self) -> None:
        self.stopped += 1


class _BlockingReplayChannel(_RecordingChannel):
    def __init__(self) -> None:
        super().__init__()
        self.first_message_started = asyncio.Event()
        self.release_first_message = asyncio.Event()

    async def on_message(self, message: Message) -> None:
        self.received.append(message)
        if len(self.received) == 1:
            self.first_message_started.set()
            await self.release_first_message.wait()


class _RecordingFactory:
    def __init__(self) -> None:
        self.started: list[
            tuple[StartThread, Participant | None, _RecordingChannel]
        ] = []
        self.opened: list[tuple[OpenThread, Participant | None, _RecordingChannel]] = []
        self.closed: list[Channel] = []
        self.fail_next_start = False
        self.fail_next_open = False
        self.fail_next_channel_start = False

    async def start(
        self,
        *,
        request: StartThread,
        sender: Participant | None,
    ) -> Channel:
        if self.fail_next_start:
            self.fail_next_start = False
            raise RuntimeError("factory start failed")
        channel = _RecordingChannel(fail_start=self.fail_next_channel_start)
        self.fail_next_channel_start = False
        self.started.append((request, sender, channel))
        return channel

    async def open(
        self,
        *,
        request: OpenThread,
        sender: Participant | None,
    ) -> Channel:
        if self.fail_next_open:
            self.fail_next_open = False
            raise RuntimeError("factory open failed")
        channel = _RecordingChannel(fail_start=self.fail_next_channel_start)
        self.fail_next_channel_start = False
        self.opened.append((request, sender, channel))
        return channel

    async def close(self, *, channel: Channel) -> None:
        self.closed.append(channel)


@pytest.mark.asyncio
async def test_threaded_proxy_promotes_started_channel_and_forwards_both_directions() -> (
    None
):
    source = _RecordingChannel()
    factory = _RecordingFactory()
    proxy = ThreadedProxyChannel(source=source, destination_factory=factory)
    supervisor = AgentSupervisor()
    alice = Participant(id="alice", attributes={})

    await proxy.start(supervisor)
    try:
        start = StartThread(
            type=AGENT_MESSAGE_THREAD_START,
            message_id="start-1",
            content=[AgentTextContent(type="text", text="hello")],
        )
        source.emit(sender=alice, payload=start)
        await _wait_until(
            lambda: (
                len(factory.started) == 1 and len(factory.started[0][2].received) == 1
            ),
            message="start was not forwarded to its provisional destination",
        )
        destination = factory.started[0][2]
        assert destination.received[0].data is start

        started = ThreadStarted(
            type=AGENT_EVENT_THREAD_STARTED,
            source_message_id="start-1",
            thread_id="thread-1",
        )
        destination.emit(sender=alice, payload=started)
        await _wait_until(
            lambda: (
                proxy.destination_for_thread(thread_id="thread-1") is destination
                and any(message.data is started for message in source.received)
            ),
            message="started destination was not promoted or forwarded",
        )

        turn = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            content=[AgentTextContent(type="text", text="next")],
        )
        source.emit(sender=alice, payload=turn)
        await _wait_until(
            lambda: any(message.data is turn for message in destination.received),
            message="thread message was not forwarded to the destination",
        )

        close = CloseThread(
            type=AGENT_MESSAGE_THREAD_CLOSE,
            thread_id="thread-1",
        )
        source.emit(sender=alice, payload=close)
        await _wait_until(
            lambda: destination in factory.closed,
            message="destination was not released when the thread closed",
        )
        assert any(message.data is close for message in destination.received)
        assert destination.state == "stopped"
        assert proxy.destination_for_thread(thread_id="thread-1") is None
    finally:
        await proxy.stop(supervisor)


@pytest.mark.asyncio
async def test_threaded_proxy_reuses_open_destination_until_last_client_closes() -> (
    None
):
    source = _RecordingChannel()
    factory = _RecordingFactory()
    proxy = ThreadedProxyChannel(source=source, destination_factory=factory)
    supervisor = AgentSupervisor()
    alice = Participant(id="alice", attributes={})
    bob = Participant(id="bob", attributes={})

    await proxy.start(supervisor)
    try:
        source.emit(
            sender=alice,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            ),
        )
        source.emit(
            sender=bob,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            ),
        )
        await _wait_until(
            lambda: (
                len(factory.opened) == 1 and len(factory.opened[0][2].received) == 2
            ),
            message="opens did not reuse one destination",
        )
        destination = factory.opened[0][2]

        source.emit(
            sender=alice,
            payload=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            ),
        )
        await _wait_until(
            lambda: len(destination.received) == 3,
            message="first close was not forwarded",
        )
        assert factory.closed == []

        source.emit(
            sender=bob,
            payload=CloseThread(
                type=AGENT_MESSAGE_THREAD_CLOSE,
                thread_id="thread-1",
            ),
        )
        await _wait_until(
            lambda: factory.closed == [destination],
            message="last close did not release the destination",
        )
    finally:
        await proxy.stop(supervisor)


@pytest.mark.asyncio
async def test_threaded_proxy_recovers_after_factory_and_destination_start_errors() -> (
    None
):
    source = _RecordingChannel()
    factory = _RecordingFactory()
    proxy = ThreadedProxyChannel(source=source, destination_factory=factory)
    supervisor = AgentSupervisor()
    alice = Participant(id="alice", attributes={})

    await proxy.start(supervisor)
    try:
        factory.fail_next_open = True
        source.emit(
            sender=alice,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="factory-error",
            ),
        )
        await _wait_until(
            lambda: isinstance(proxy.last_error, RuntimeError),
            message="factory error was not recorded",
        )
        assert proxy.destination_for_thread(thread_id="factory-error") is None

        factory.fail_next_channel_start = True
        source.emit(
            sender=alice,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="start-error",
            ),
        )
        await _wait_until(
            lambda: len(factory.closed) == 1,
            message="failed destination start did not invoke factory cleanup",
        )
        assert proxy.destination_for_thread(thread_id="start-error") is None

        source.emit(
            sender=alice,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="healthy",
            ),
        )
        await _wait_until(
            lambda: proxy.destination_for_thread(thread_id="healthy") is not None,
            message="proxy did not recover after destination creation failures",
        )
    finally:
        await proxy.stop(supervisor)


@pytest.mark.asyncio
async def test_threaded_proxy_releases_rejected_start_and_rejects_wrong_thread_output() -> (
    None
):
    source = _RecordingChannel()
    factory = _RecordingFactory()
    proxy = ThreadedProxyChannel(source=source, destination_factory=factory)
    supervisor = AgentSupervisor()
    alice = Participant(id="alice", attributes={})

    await proxy.start(supervisor)
    try:
        source.emit(
            sender=alice,
            payload=StartThread(
                type=AGENT_MESSAGE_THREAD_START,
                message_id="rejected-start",
            ),
        )
        await _wait_until(
            lambda: len(factory.started) == 1,
            message="provisional destination was not created",
        )
        rejected_destination = factory.started[0][2]
        rejected = TurnStartRejected(
            type=AGENT_EVENT_TURN_START_REJECTED,
            source_message_id="rejected-start",
            thread_id="",
            error=AgentError(message="no", code="rejected"),
        )
        rejected_destination.emit(sender=alice, payload=rejected)
        await _wait_until(
            lambda: (
                rejected_destination in factory.closed
                and any(message.data is rejected for message in source.received)
            ),
            message="rejected start did not forward or release its destination",
        )

        source.emit(
            sender=alice,
            payload=OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id="thread-1",
            ),
        )
        await _wait_until(
            lambda: len(factory.opened) == 1,
            message="open destination was not created",
        )
        destination = factory.opened[0][2]
        wrong_thread = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-2",
            content=[],
        )
        received_before = len(source.received)
        destination.emit(sender=alice, payload=wrong_thread)
        await _wait_until(
            lambda: (
                proxy.last_error is not None and "thread-2" in str(proxy.last_error)
            ),
            message="wrong-thread output was not rejected",
        )
        assert len(source.received) == received_before
        assert proxy.destination_for_thread(thread_id="thread-1") is destination
    finally:
        await proxy.stop(supervisor)


@pytest.mark.asyncio
async def test_channel_inject_messages_is_an_ordered_barrier_before_next_message() -> (
    None
):
    channel = _BlockingReplayChannel()
    supervisor = AgentSupervisor()
    await channel.start(supervisor)
    replay_one = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="thread-1",
        content=[AgentTextContent(type="text", text="one")],
    )
    replay_two = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="thread-1",
        content=[AgentTextContent(type="text", text="two")],
    )
    live = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="thread-1",
        content=[AgentTextContent(type="text", text="live")],
    )

    injection = asyncio.create_task(
        channel.inject_messages(
            thread_id="thread-1",
            messages=[replay_one, replay_two],
        )
    )
    await channel.first_message_started.wait()
    channel.send(Message(data=live))
    await asyncio.sleep(0)
    assert [message.data for message in channel.received] == [replay_one]

    channel.release_first_message.set()
    assert await injection is True
    await _wait_until(
        lambda: len(channel.received) == 3,
        message="live message was not delivered after injection",
    )
    assert [message.data for message in channel.received] == [
        replay_one,
        replay_two,
        live,
    ]
    await channel.stop(supervisor)


@pytest.mark.asyncio
async def test_threaded_proxy_injects_loaded_replay_before_forwarding_buffered_live_message() -> (
    None
):
    source = _BlockingReplayChannel()
    factory = _RecordingFactory()
    proxy = ThreadedProxyChannel(
        source=source,
        destination_factory=factory,
        inject_loaded_messages=True,
    )
    supervisor = AgentSupervisor()
    alice = Participant(id="alice", attributes={})
    await proxy.start(supervisor)
    try:
        opened = OpenThread(
            type=AGENT_MESSAGE_THREAD_OPEN,
            message_id="open-1",
            thread_id="thread-1",
            load=True,
        )
        source.emit(sender=alice, payload=opened)
        await _wait_until(
            lambda: (
                len(factory.opened) == 1 and len(factory.opened[0][2].received) == 1
            ),
            message="load open was not forwarded",
        )
        destination = factory.opened[0][2]
        replay_one = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            content=[AgentTextContent(type="text", text="stored one")],
        )
        replay_two = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            content=[AgentTextContent(type="text", text="stored two")],
        )
        loaded = ThreadLoaded(
            type=AGENT_EVENT_THREAD_LOADED,
            thread_id="thread-1",
            source_message_id="open-1",
        )
        destination.emit(sender=alice, payload=replay_one)
        destination.emit(sender=alice, payload=replay_two)
        destination.emit(sender=alice, payload=loaded)
        await source.first_message_started.wait()

        live = TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            content=[AgentTextContent(type="text", text="live")],
        )
        source.emit(sender=alice, payload=live)
        await asyncio.sleep(0.02)
        assert [message.data for message in destination.received] == [opened]
        assert [message.data for message in source.received] == [replay_one]

        source.release_first_message.set()
        await _wait_until(
            lambda: len(source.received) == 3 and len(destination.received) == 2,
            message="replay injection or buffered live delivery did not complete",
        )
        assert [message.data for message in source.received] == [
            replay_one,
            replay_two,
            loaded,
        ]
        assert [message.data for message in destination.received] == [opened, live]
    finally:
        await proxy.stop(supervisor)


def test_inject_messages_round_trips_typed_child_messages() -> None:
    child = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="thread-1",
        content=[AgentTextContent(type="text", text="stored")],
    )
    injected = InjectMessages.model_validate(
        {
            "type": AGENT_MESSAGE_MESSAGES_INJECT,
            "thread_id": "thread-1",
            "messages": [child.model_dump(mode="json")],
        }
    )
    assert injected.messages == [child]
