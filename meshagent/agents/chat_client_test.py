import asyncio
from typing import Any

import pytest
from aiohttp import web

from meshagent.agents.chat_channel import MsgpackWebSocketChatEncoding
from meshagent.agents.chat_client import (
    BaseChatClient,
    ChatThreadSession,
    MessagingChatClient,
    WebSocketChatClient,
)
from meshagent.agents.messages import (
    AGENT_EVENT_CONNECTION_STATUS,
    AGENT_EVENT_THREAD_CREATED,
    AGENT_EVENT_THREAD_LISTED,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_MESSAGE_THREAD_LIST,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_THREAD_START,
    AGENT_MESSAGE_TURN_START,
    AgentError,
    AgentThreadListEntry,
    AgentMessage,
    AgentModelChanged,
    AgentUsageUpdated,
    AgentConnectionStatus,
    AgentTextContent,
    AgentThreadStatus,
    AgentTextContentDelta,
    ThreadCreated,
    ThreadStarted,
    ThreadLoaded,
    ThreadsListed,
    TurnEnded,
    TurnStart,
    TurnStartAccepted,
    TurnStarted,
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


class _RecordingChatClient(BaseChatClient):
    def __init__(self) -> None:
        super().__init__(timeout=1)
        self.sent: list[dict[str, Any]] = []

    @property
    def remote_participant_name(self) -> str:
        return "assistant"

    async def _start_transport(self) -> None:
        return None

    async def _stop_transport(self) -> None:
        return None

    async def _send_agent_message(self, payload: AgentMessage) -> None:
        self.sent.append(payload.model_dump(mode="json"))


def test_agent_message_serializes_created_at_by_default() -> None:
    payload = AgentMessage(type="meshagent.agent.test")

    dumped = payload.model_dump(mode="json")

    assert isinstance(dumped["created_at"], str)
    assert dumped["created_at"].endswith("Z")


def test_chat_thread_session_records_failed_turn_end_for_rendering() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    message = TurnEnded(
        type=AGENT_EVENT_TURN_ENDED,
        thread_id=session.thread_path,
        turn_id="turn-1",
        error=AgentError(
            code="RoomException",
            message="Error from OpenAI websocket: unknown parameter",
        ),
    )

    session.add_agent_message(message)

    assert session.messages == (message,)
    assert session.pending_inputs == ()


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


def test_usage_updates_tolerate_missing_context_window_token_counts() -> None:
    parsed = parse_agent_message(
        {
            "type": "meshagent.agent.usage.updated",
            "thread_id": "/threads/test.thread",
            "turn_id": "turn-1",
            "usage": {},
            "context_window": {"total_tokens": 128000},
        }
    )

    assert isinstance(parsed, AgentUsageUpdated)
    assert parsed.context_window.used_tokens == 0
    assert parsed.context_window.total_tokens == 128000


@pytest.mark.asyncio
async def test_chat_thread_session_lists_threads_with_agent_message() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path=None)

    async def respond() -> None:
        await _wait_for(lambda: len(client.sent) == 1)
        request = client.sent[0]
        assert request["type"] == AGENT_MESSAGE_THREAD_LIST
        client._handle_agent_payload(
            ThreadsListed(
                type=AGENT_EVENT_THREAD_LISTED,
                source_message_id=request["message_id"],
                threads=[
                    AgentThreadListEntry(
                        path="/threads/one.thread",
                        name="One",
                    )
                ],
                total=1,
                offset=0,
                limit=100,
            ).model_dump(mode="json")
        )

    response_task = asyncio.create_task(respond())
    try:
        response = await session.list_threads(limit=100, offset=0)
    finally:
        await response_task

    assert [thread.name for thread in response.threads] == ["One"]


@pytest.mark.asyncio
async def test_chat_thread_session_notifies_thread_list_event_listeners() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path=None)
    events: list[dict[str, Any]] = []

    unsubscribe = session.add_event_listener(events.append)
    client._handle_agent_payload(
        ThreadCreated(
            type=AGENT_EVENT_THREAD_CREATED,
            thread=AgentThreadListEntry(
                path="/threads/new.thread",
                name="New",
            ),
        ).model_dump(mode="json")
    )
    unsubscribe()
    client._handle_agent_payload(
        ThreadCreated(
            type=AGENT_EVENT_THREAD_CREATED,
            thread=AgentThreadListEntry(
                path="/threads/ignored.thread",
                name="Ignored",
            ),
        ).model_dump(mode="json")
    )

    assert [event["thread"]["name"] for event in events] == ["New"]


def test_chat_thread_session_ignores_duplicate_delta_messages() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    delta = AgentTextContentDelta(
        type="meshagent.agent.text_content.delta",
        message_id="delta-1",
        thread_id="/threads/test.thread",
        turn_id="turn-1",
        item_id="item-1",
        text="hello",
    )

    session.add_agent_message(delta)
    session.add_agent_message(delta)

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], AgentTextContentDelta)
    assert session.messages[0].text == "hello"


def test_chat_thread_session_merges_distinct_text_delta_messages_by_item_id() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    first = AgentTextContentDelta(
        type="meshagent.agent.text_content.delta",
        message_id="delta-1",
        thread_id="/threads/test.thread",
        turn_id="turn-1",
        item_id="item-1",
        text="hello",
    )
    second = AgentTextContentDelta(
        type="meshagent.agent.text_content.delta",
        message_id="delta-2",
        thread_id="/threads/test.thread",
        turn_id="turn-1",
        item_id="item-1",
        text=" world",
    )

    assert first.message_id != first.item_id
    assert second.message_id != second.item_id
    session.add_agent_message(first)
    session.add_agent_message(second)

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], AgentTextContentDelta)
    assert session.messages[0].text == "hello world"


@pytest.mark.asyncio
async def test_chat_thread_session_does_not_enqueue_duplicate_delta_messages() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    turn_start = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="/threads/test.thread",
    )
    await session.send(turn_start)
    client._handle_agent_payload(
        TurnStarted(
            type=AGENT_EVENT_TURN_STARTED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            source_message_id=turn_start.message_id,
        ).model_dump(mode="json")
    )
    delta = AgentTextContentDelta(
        type="meshagent.agent.text_content.delta",
        message_id="delta-1",
        thread_id="/threads/test.thread",
        turn_id="turn-1",
        item_id="item-1",
        text="hello",
    ).model_dump(mode="json")

    client._handle_agent_payload(delta)
    client._handle_agent_payload(delta)

    first = await _receive_until(
        session,
        lambda payload: payload.get("type") == "meshagent.agent.text_content.delta",
    )
    assert first["text"] == "hello"
    with pytest.raises(asyncio.TimeoutError):
        await _receive_until(
            session,
            lambda payload: payload.get("type") == "meshagent.agent.text_content.delta",
            timeout=0.01,
        )


@pytest.mark.asyncio
async def test_chat_thread_session_clears_pending_inputs_on_turn_end() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    turn_start = TurnStart(
        type=AGENT_MESSAGE_TURN_START,
        thread_id="/threads/test.thread",
    )

    await session.send(turn_start)
    client._handle_agent_payload(
        TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            source_message_id=turn_start.message_id,
        ).model_dump(mode="json")
    )

    assert [pending.message_id for pending in session.pending_inputs] == [
        turn_start.message_id
    ]

    client._handle_agent_payload(
        TurnEnded(
            type=AGENT_EVENT_TURN_ENDED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            error=None,
        ).model_dump(mode="json")
    )

    assert session.pending_inputs == ()


@pytest.mark.asyncio
async def test_chat_thread_session_tracks_active_turn_from_accepted_event() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")

    message_id = await session.send_text(text="hello")
    client._handle_agent_payload(
        TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            source_message_id=message_id,
        ).model_dump(mode="json")
    )

    assert session.interrupt()
    client._handle_agent_payload(
        TurnEnded(
            type=AGENT_EVENT_TURN_ENDED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            error=None,
        ).model_dump(mode="json")
    )

    assert not session.interrupt()


def test_chat_thread_session_appends_remote_accepted_input_with_content() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")

    client._handle_agent_payload(
        TurnStartAccepted(
            type=AGENT_EVENT_TURN_START_ACCEPTED,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            source_message_id="remote-message-1",
            content=[AgentTextContent(type="text", text="hello from someone else")],
            sender_name="teammate",
        ).model_dump(mode="json")
    )

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], TurnStartAccepted)
    assert session.messages[0].sender_name == "teammate"
    assert session.messages[0].content == [
        AgentTextContent(type="text", text="hello from someone else")
    ]


@pytest.mark.asyncio
async def test_chat_thread_session_send_text_uses_selected_backend_model() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")
    client._handle_agent_payload(
        AgentModelChanged(
            type="meshagent.agent.model.changed",
            thread_id="/threads/test.thread",
            backend="codex",
            provider="openai",
            model="gpt-5.5",
            output_modalities=["text"],
        ).model_dump(mode="json")
    )

    await session.send_text(text="hello")

    assert client.sent[-1]["type"] == AGENT_MESSAGE_TURN_START
    assert client.sent[-1]["backend"] == "codex"
    assert client.sent[-1]["provider"] == "openai"
    assert client.sent[-1]["model"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_chat_thread_session_starts_thread_before_turns_when_threadless() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path=None)

    start_task = asyncio.create_task(
        session.start_thread(text="hello", provider="openai", model="gpt-5.5")
    )
    await _wait_for(lambda: len(client.sent) == 1)

    sent = client.sent[0]
    assert sent["type"] == AGENT_MESSAGE_THREAD_START
    assert sent["provider"] == "openai"
    assert sent["model"] == "gpt-5.5"
    assert sent["content"][0]["text"] == "hello"
    message_id = sent["message_id"]
    assert session.pending_inputs[0].message_id == message_id

    client._handle_agent_payload(
        ThreadStarted(
            type=AGENT_EVENT_THREAD_STARTED,
            source_message_id=message_id,
            thread_id="/threads/created.thread",
        ).model_dump(mode="json")
    )

    assert await start_task == message_id
    assert session.thread_path == "/threads/created.thread"
    assert client.sent[1]["type"] == AGENT_MESSAGE_THREAD_OPEN
    assert client.sent[1]["load"] is False


@pytest.mark.asyncio
async def test_chat_thread_session_tracks_steerable_status_as_active_turn() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path="/threads/test.thread")

    client._handle_agent_payload(
        AgentThreadStatus(
            type=AGENT_EVENT_THREAD_STATUS,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            status="Writing",
            mode="steerable",
        ).model_dump(mode="json")
    )

    assert session.active_turn_id == "turn-1"
    assert session.interrupt()

    client._handle_agent_payload(
        AgentThreadStatus(
            type=AGENT_EVENT_THREAD_STATUS,
            thread_id="/threads/test.thread",
            turn_id="turn-1",
            status=None,
        ).model_dump(mode="json")
    )

    assert session.active_turn_id is None
    assert not session.interrupt()


@pytest.mark.asyncio
async def test_chat_client_thread_list_event_listener_survives_session_close() -> None:
    client = _RecordingChatClient()
    session = client._create_thread_session(thread_path=None)
    events: list[dict[str, Any]] = []

    unsubscribe = client.add_event_listener(events.append)
    await session.close(close_client=False)
    client._handle_agent_payload(
        ThreadCreated(
            type=AGENT_EVENT_THREAD_CREATED,
            thread=AgentThreadListEntry(
                path="/threads/new.thread",
                name="New",
            ),
        ).model_dump(mode="json")
    )
    unsubscribe()
    client._handle_agent_payload(
        ThreadCreated(
            type=AGENT_EVENT_THREAD_CREATED,
            thread=AgentThreadListEntry(
                path="/threads/ignored.thread",
                name="Ignored",
            ),
        ).model_dump(mode="json")
    )

    assert [event["thread"]["name"] for event in events] == ["New"]


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
    client_events: list[dict[str, Any]] = []
    client_event_task: asyncio.Task[None] | None = None
    try:

        async def collect_client_events() -> None:
            async for payload in client.events:
                client_events.append(payload)

        client_event_task = asyncio.create_task(collect_client_events())
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

        await _wait_for(
            lambda: any(
                payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
                and payload.get("status") == "reconnecting"
                for payload in client_events
            ),
        )
        reconnecting = next(
            payload
            for payload in client_events
            if payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
            and payload.get("status") == "reconnecting"
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

        await _wait_for(
            lambda: any(
                payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
                and payload.get("status") == "reconnected"
                for payload in client_events
            ),
        )
        reconnected = next(
            payload
            for payload in client_events
            if payload.get("type") == AGENT_EVENT_CONNECTION_STATUS
            and payload.get("status") == "reconnected"
        )
        assert AgentConnectionStatus.model_validate(reconnected).status == (
            "reconnected"
        )
    finally:
        if client_event_task is not None:
            client_event_task.cancel()
            await asyncio.gather(client_event_task, return_exceptions=True)
        await client.stop()
        for websocket in sockets:
            await websocket.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_messaging_chat_client_reconnect_tracks_participant_and_reopens_thread() -> (
    None
):
    class _Participant:
        id = "agent-1"

        def get_attribute(self, name: str):
            if name == "name":
                return "assistant"
            if name == "supports_agent_messages":
                return True
            return None

    class _Messaging:
        def __init__(self) -> None:
            self.is_enabled = True
            self.participants: list[_Participant] = [_Participant()]
            self.handlers: dict[str, list[Any]] = {}
            self.sent_payloads: list[dict[str, Any]] = []

        def on(self, event: str, handler) -> None:
            self.handlers.setdefault(event, []).append(handler)

        def off(self, event: str, handler) -> None:
            self.handlers.get(event, []).remove(handler)

        def get_participants(self) -> list[_Participant]:
            return list(self.participants)

        async def enable(self) -> None:
            self.is_enabled = True

        async def send_message(
            self,
            *,
            to,
            type: str,
            message: dict[str, Any],
            attachment,
        ) -> None:
            assert to is self.participants[0]
            assert type == "agent-message"
            assert attachment is None
            self.sent_payloads.append(message)

        def remove_agent(self) -> _Participant:
            participant = self.participants.pop()
            for handler in self.handlers.get("participant_removed", []):
                handler(participant=participant)
            return participant

        def add_agent(self, participant: _Participant) -> None:
            self.participants.append(participant)
            for handler in self.handlers.get("participant_added", []):
                handler(participant=participant)

    class _Room:
        def __init__(self) -> None:
            self.messaging = _Messaging()
            self.handlers: dict[str, list[Any]] = {}

        def on(self, event: str, handler) -> None:
            self.handlers.setdefault(event, []).append(handler)

        def off(self, event: str, handler) -> None:
            self.handlers.get(event, []).remove(handler)

    room = _Room()
    client = MessagingChatClient(room=room, participant_name="assistant", timeout=0.1)
    await client.start()
    session = await client.open_thread("/threads/reconnect.thread", load=False)
    assert session.thread_path == "/threads/reconnect.thread"

    assert client.connection_status is not None
    assert client.connection_status.status == "connected"

    participant = room.messaging.remove_agent()
    assert client.connection_status is not None
    assert client.connection_status.status == "reconnecting"

    room.messaging.add_agent(participant)
    await _wait_for(
        lambda: any(
            payload.get("type") == AGENT_MESSAGE_THREAD_OPEN
            and payload.get("thread_id") == "/threads/reconnect.thread"
            and payload.get("load") is True
            for payload in room.messaging.sent_payloads
        )
    )
    assert client.connection_status is not None
    assert client.connection_status.status == "reconnected"
    await client.close()
