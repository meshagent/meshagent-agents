import asyncio
from typing import Any

import pytest
from aiohttp import web

from meshagent.agents.chat_channel import MsgpackWebSocketChatEncoding
from meshagent.agents.chat_client import ChatThreadSession, WebSocketChatClient
from meshagent.agents.messages import (
    AGENT_EVENT_CONNECTION_STATUS,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_MESSAGE_THREAD_OPEN,
    AgentConnectionStatus,
    ThreadLoaded,
    TurnEnded,
    parse_agent_message,
)


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


async def _receive_until(
    session: ChatThreadSession,
    predicate,
    *,
    timeout: float = 1,
) -> dict[str, Any]:
    async with asyncio.timeout(timeout):
        while True:
            payload = await session.receive()
            if predicate(payload):
                return payload


@pytest.mark.asyncio
async def test_thread_loaded_message_round_trips() -> None:
    loaded = ThreadLoaded(
        type=AGENT_EVENT_THREAD_LOADED,
        thread_id="/threads/test.thread",
        source_message_id="open-1",
        since_turn="turn-1",
    )

    parsed = parse_agent_message(loaded.model_dump(mode="json"))

    assert isinstance(parsed, ThreadLoaded)
    assert parsed.thread_id == "/threads/test.thread"
    assert parsed.source_message_id == "open-1"
    assert parsed.since_turn == "turn-1"


@pytest.mark.asyncio
async def test_websocket_chat_client_reconnect_reopens_thread_with_load() -> None:
    encoding = MsgpackWebSocketChatEncoding()
    sockets: list[web.WebSocketResponse] = []
    payloads: list[dict[str, Any]] = []
    socket_connected = asyncio.Event()
    first_open_received = asyncio.Event()
    second_open_received = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(protocols=("meshagent-msgpack",))
        await websocket.prepare(request)
        connection_index = len(sockets)
        sockets.append(websocket)
        socket_connected.set()
        async for message in websocket:
            decoded = encoding.decode(message)
            payload = decoded.model_dump(mode="json", exclude_none=True)
            payloads.append(payload)
            if payload.get("type") != AGENT_MESSAGE_THREAD_OPEN:
                continue
            if connection_index == 0:
                first_open_received.set()
                await websocket.close(code=1001, message=b"test reconnect")
            elif connection_index == 1:
                second_open_received.set()
        return websocket

    app = web.Application()
    app.router.add_get("/messages", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]

    client = WebSocketChatClient(
        url=f"ws://127.0.0.1:{port}/messages",
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.01,
    )
    try:
        await client.start()
        await _wait_for(lambda: len(sockets) == 1)
        session = await client.open_thread("/threads/reconnect.thread")
        await asyncio.wait_for(first_open_received.wait(), timeout=1)
        session.add_agent_message(
            TurnEnded(
                type=AGENT_EVENT_TURN_ENDED,
                thread_id=session.thread_path,
                turn_id="turn-1",
            )
        )

        reconnecting = await _receive_until(
            session,
            lambda payload: (
                payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
                and payload.get("status") == "reconnecting"
            ),
        )
        assert AgentConnectionStatus.model_validate(reconnecting).status == (
            "reconnecting"
        )

        socket_connected.clear()
        await asyncio.wait_for(second_open_received.wait(), timeout=1)
        assert len(sockets) >= 2
        reopened = next(
            payload
            for payload in reversed(payloads)
            if payload.get("type") == AGENT_MESSAGE_THREAD_OPEN
        )
        assert reopened["thread_id"] == "/threads/reconnect.thread"
        assert reopened["load"] is True
        assert reopened["since_turn"] == "turn-1"

        reconnected = await _receive_until(
            session,
            lambda payload: (
                payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
                and payload.get("status") == "reconnected"
            ),
        )
        assert AgentConnectionStatus.model_validate(reconnected).status == (
            "reconnected"
        )
    finally:
        await client.stop()
        for websocket in sockets:
            await websocket.close()
        await runner.cleanup()
