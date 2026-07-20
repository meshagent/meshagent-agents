from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Callable, Sequence

from meshagent.api import Participant, RoomClient

from .chat_channel import MessagingChatChannel
from .external_process_channel import ExternalChannelConnection
from .mail_channel import MailChannel
from .process import AgentSupervisor, Channel, Message
from .queue_channel import QueueChannel
from .toolkit_channel import ToolkitChannel


class ExternalChannelBridgeSupervisor(AgentSupervisor):
    """Minimal supervisor that bridges one real channel to its parent process."""

    def __init__(self, *, connection: ExternalChannelConnection) -> None:
        super().__init__()
        self._connection = connection
        self._outbound: asyncio.Queue[Message | None] = asyncio.Queue()

    def send(self, message: Message) -> None:
        self._outbound.put_nowait(message)

    async def _write_messages(self) -> None:
        while True:
            message = await self._outbound.get()
            if message is None:
                return
            await self._connection.send(
                payload=message.data,
                sender=message.sender,
                to_participant_id=message.to_participant_id,
            )

    @staticmethod
    async def _deliver_to_channel(*, channel: Channel, message: Message) -> None:
        target_id = message.to_participant_id
        if target_id is None:
            channel.send(message)
            return
        participant = message.sender
        if participant is None or participant.id != target_id:
            participant = Participant(id=target_id)
        await channel.send_agent_message_to_participant_and_wait(
            participant=participant,
            payload=message.data,
        )

    async def run_channel(self, *, channel: Channel) -> None:
        await channel.start(self)
        writer_task = asyncio.create_task(self._write_messages())
        try:
            while True:
                try:
                    message = await self._connection.receive()
                except EOFError:
                    return
                await self._deliver_to_channel(channel=channel, message=message)
        finally:
            await channel.stop(self)
            self._outbound.put_nowait(None)
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await writer_task


async def run_external_channel(channel: Channel) -> None:
    connection = await ExternalChannelConnection.connect_from_environment()
    try:
        bridge = ExternalChannelBridgeSupervisor(connection=connection)
        await bridge.run_channel(channel=channel)
    finally:
        await connection.close()


async def _run_room_channel(channel_factory: Callable[[RoomClient], Channel]) -> None:
    connection = await ExternalChannelConnection.connect_from_environment()
    try:
        async with RoomClient() as room:
            bridge = ExternalChannelBridgeSupervisor(connection=connection)
            await bridge.run_channel(channel=channel_factory(room))
    finally:
        await connection.close()


def run_room_channel(channel_factory: Callable[[RoomClient], Channel]) -> None:
    """Run a room-backed channel as an external channel child process."""
    asyncio.run(_run_room_channel(channel_factory))


def _common_parser(*, program: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("--thread-dir")
    parser.add_argument("--threading-mode", default="default-new")
    return parser


def chat_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser(program="meshagent-channel-chat")
    options = parser.parse_args(argv)
    run_room_channel(
        lambda room: MessagingChatChannel(
            room=room,
            threading_mode=options.threading_mode,
            thread_dir=options.thread_dir,
        )
    )


def mail_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser(program="meshagent-channel-mail")
    parser.add_argument("email_address")
    parser.add_argument("--queue-name")
    parser.add_argument("--reply-all", action="store_true")
    options = parser.parse_args(argv)
    run_room_channel(
        lambda room: MailChannel(
            room=room,
            queue_name=options.queue_name or options.email_address,
            email_address=options.email_address,
            reply_all=options.reply_all,
            threading_mode=options.threading_mode,
            thread_dir=options.thread_dir,
        )
    )


def queue_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser(program="meshagent-channel-queue")
    parser.add_argument("queue_name")
    options = parser.parse_args(argv)
    run_room_channel(
        lambda room: QueueChannel(
            room=room,
            queue_name=options.queue_name,
            threading_mode=options.threading_mode,
            thread_dir=options.thread_dir,
        )
    )


def toolkit_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="meshagent-channel-toolkit")
    parser.add_argument("toolkit_name")
    parser.add_argument("--thread-dir")
    options = parser.parse_args(argv)
    run_room_channel(
        lambda room: ToolkitChannel(
            room=room,
            toolkit_name=options.toolkit_name,
            thread_dir=options.thread_dir,
        )
    )


def dispatch_main(*, executable_name: str, argv: Sequence[str] | None = None) -> bool:
    entrypoints = {
        "meshagent-channel-chat": chat_main,
        "meshagent-channel-mail": mail_main,
        "meshagent-channel-queue": queue_main,
        "meshagent-channel-toolkit": toolkit_main,
    }
    entrypoint = entrypoints.get(executable_name)
    if entrypoint is None:
        return False
    entrypoint(argv)
    return True
