from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import msgpack

from meshagent.api import Participant

from .messages import AgentMessage, parse_agent_message
from .process import Channel, Message


logger = logging.getLogger("external-process-channel")
EXTERNAL_CHANNEL_MAX_BUFFER_SIZE = 64 * 1024 * 1024


class ExternalChannelConnection:
    """Child-side connection for a language-neutral external channel process."""

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._unpacker = msgpack.Unpacker(
            raw=False,
            strict_map_key=False,
            max_buffer_size=EXTERNAL_CHANNEL_MAX_BUFFER_SIZE,
        )
        self._pending: deque[object] = deque()

    @classmethod
    async def connect_from_environment(cls) -> ExternalChannelConnection:
        transport_url = os.environ.get("MESHAGENT_CHANNEL_URL", "")
        capability = os.environ.get("MESHAGENT_CHANNEL_CAPABILITY", "")
        if transport_url == "" or capability == "":
            raise RuntimeError(
                "MESHAGENT_CHANNEL_URL and MESHAGENT_CHANNEL_CAPABILITY are required"
            )
        transport = urlparse(transport_url)
        if transport.scheme == "unix":
            if os.name == "nt":
                raise RuntimeError("Unix channel sockets are not supported on Windows")
            reader, writer = await asyncio.open_unix_connection(transport.path)
        elif transport.scheme == "tcp":
            if transport.hostname is None or transport.port is None:
                raise RuntimeError(
                    "external channel TCP URL must include a host and port"
                )
            reader, writer = await asyncio.open_connection(
                transport.hostname,
                transport.port,
            )
        else:
            raise RuntimeError(
                f"unsupported external channel transport: {transport.scheme}"
            )
        writer.write(msgpack.packb({"capability": capability}, use_bin_type=True))
        await writer.drain()
        return cls(reader=reader, writer=writer)

    async def send(
        self,
        *,
        payload: AgentMessage,
        sender: Participant | None = None,
        to_participant_id: str | None = None,
    ) -> None:
        message = Message(
            data=payload,
            sender=sender,
            to_participant_id=to_participant_id,
        )
        self._writer.write(
            msgpack.packb(
                ExternalProcessChannel._message_to_wire(message),
                use_bin_type=True,
            )
        )
        await self._writer.drain()

    async def receive(self) -> Message:
        while len(self._pending) == 0:
            chunk = await self._reader.read(64 * 1024)
            if chunk == b"":
                raise EOFError("external channel connection closed")
            self._unpacker.feed(chunk)
            self._pending.extend(self._unpacker)
        value = self._pending.popleft()
        return ExternalProcessChannel._message_from_wire(value, source=None)

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await self._writer.wait_closed()


class ExternalProcessChannel(Channel):
    """A channel hosted by a child process using MessagePack message envelopes.

    The parent opens a capability-protected loopback connection before launching the
    child and passes its address through the child environment. Messages are consecutive
    encoded ``Message`` objects without an additional framing layer. The parent always
    assigns the internal channel source. The child process is the authority for inbound
    ``sender`` values.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        environment: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        normalized_command = tuple(command)
        if len(normalized_command) == 0 or any(
            part == "" for part in normalized_command
        ):
            raise ValueError("external process channel command cannot be empty")
        self._command = normalized_command
        self._environment = dict(environment or {})
        self._process: asyncio.subprocess.Process | None = None
        self._write_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._server: asyncio.Server | None = None
        self._connection_event = asyncio.Event()
        self._connection_done = asyncio.Event()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._capability = ""
        self._runtime_dir: Path | None = None

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def environment(self) -> dict[str, str]:
        return dict(self._environment)

    @staticmethod
    def _participant_from_wire(value: object) -> Participant | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("external channel sender must be an object")
        participant_id = value.get("id")
        attributes = value.get("attributes", {})
        if not isinstance(participant_id, str) or participant_id.strip() == "":
            raise ValueError("external channel sender id must be a non-empty string")
        if not isinstance(attributes, dict):
            raise ValueError("external channel sender attributes must be an object")
        return Participant(id=participant_id, attributes=attributes)

    @staticmethod
    def _message_to_wire(message: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "data": message.data.model_dump(mode="json", exclude_none=True),
        }
        if message.sender is not None:
            wire["sender"] = {
                "id": message.sender.id,
                "attributes": message.sender.attributes,
            }
        if message.to_participant_id is not None:
            wire["to_participant_id"] = message.to_participant_id
        return wire

    @staticmethod
    def _message_from_wire(
        value: object,
        *,
        source: Channel | None,
    ) -> Message:
        if not isinstance(value, dict):
            raise ValueError("external channel message must decode to an object")
        data = value.get("data")
        if not isinstance(data, dict):
            raise ValueError("external channel message data must be an object")
        to_participant_id = value.get("to_participant_id")
        if to_participant_id is not None and not isinstance(to_participant_id, str):
            raise ValueError("external channel target participant id must be a string")
        return Message(
            data=parse_agent_message(data),
            sender=ExternalProcessChannel._participant_from_wire(value.get("sender")),
            source=source,
            to_participant_id=to_participant_id,
        )

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._writer is not None:
            writer.close()
            await writer.wait_closed()
            return
        unpacker = msgpack.Unpacker(
            raw=False,
            strict_map_key=False,
            max_buffer_size=EXTERNAL_CHANNEL_MAX_BUFFER_SIZE,
        )
        authenticated = False
        try:
            while True:
                chunk = await reader.read(64 * 1024)
                if chunk == b"":
                    return
                unpacker.feed(chunk)
                for value in unpacker:
                    if not authenticated:
                        if not isinstance(value, dict) or not secrets.compare_digest(
                            str(value.get("capability", "")),
                            self._capability,
                        ):
                            logger.warning(
                                "rejected external channel connection with invalid capability"
                            )
                            return
                        authenticated = True
                        self._reader = reader
                        self._writer = writer
                        self._connection_event.set()
                        continue
                    try:
                        message = self._message_from_wire(value, source=self)
                    except (TypeError, ValueError):
                        logger.exception(
                            "ignoring invalid message from external channel process"
                        )
                        continue
                    supervisor = self.supervisor
                    if supervisor is not None:
                        supervisor.send(message)
        finally:
            if self._writer is writer:
                self._reader = None
                self._writer = None
                self._connection_done.set()
            writer.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await writer.wait_closed()

    async def _write_messages(self) -> None:
        while True:
            message = await self._write_queue.get()
            if message is None:
                return
            writer = self._writer
            if writer is None:
                raise RuntimeError("external channel connection is not available")
            writer.write(msgpack.packb(message, use_bin_type=True))
            await writer.drain()

    def _enqueue(self, message: Message) -> bool:
        process = self._process
        if process is None or process.returncode is not None or self._writer is None:
            return False
        self._write_queue.put_nowait(self._message_to_wire(message))
        return True

    async def on_start(self) -> None:
        self._write_queue = asyncio.Queue()
        self._connection_event = asyncio.Event()
        self._connection_done = asyncio.Event()
        self._capability = secrets.token_urlsafe(32)
        if os.name == "nt":
            self._server = await asyncio.start_server(
                self._handle_connection,
                host="127.0.0.1",
                port=0,
            )
            sockets = self._server.sockets or []
            if len(sockets) != 1:
                raise RuntimeError(
                    "external channel listener did not bind exactly one socket"
                )
            host, port = sockets[0].getsockname()[:2]
            transport_url = f"tcp://{host}:{port}"
        else:
            self._runtime_dir = Path(tempfile.mkdtemp(prefix="meshagent-channel-"))
            self._runtime_dir.chmod(0o700)
            socket_path = self._runtime_dir / "channel.sock"
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=socket_path,
            )
            transport_url = f"unix://{socket_path}"
        environment = os.environ.copy()
        environment.update(self._environment)
        environment["MESHAGENT_CHANNEL_URL"] = transport_url
        environment["MESHAGENT_CHANNEL_CAPABILITY"] = self._capability
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                env=environment,
            )
            await asyncio.wait_for(self._connection_event.wait(), timeout=10)
        except BaseException:
            await self._stop_transport()
            raise
        self._writer_task = asyncio.create_task(self._write_messages())

    async def on_message(self, message: Message) -> None:
        if not self._enqueue(message):
            raise RuntimeError("external channel process is not running")

    def send_agent_message_to_participant(
        self,
        *,
        participant: Participant,
        payload: AgentMessage,
    ) -> bool:
        return self._enqueue(
            Message(
                data=payload,
                sender=participant,
                to_participant_id=participant.id,
            )
        )

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

    async def _stop_transport(self) -> None:
        process = self._process
        writer_task = self._writer_task
        self._writer_task = None

        if writer_task is not None:
            self._write_queue.put_nowait(None)
            with contextlib.suppress(
                BrokenPipeError, ConnectionResetError, RuntimeError
            ):
                await writer_task
        writer = self._writer
        if writer is not None:
            writer.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await writer.wait_closed()
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        runtime_dir = self._runtime_dir
        self._runtime_dir = None
        if runtime_dir is not None:
            (runtime_dir / "channel.sock").unlink(missing_ok=True)
            runtime_dir.rmdir()
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
        if process is not None:
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        self._process = None
        self._reader = None
        self._writer = None
        self._capability = ""

    async def on_stop(self) -> None:
        await self._stop_transport()
