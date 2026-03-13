import pytest

from meshagent.api import RoomException

from meshagent.agents.chat import ChatThreadContext
from meshagent.agents.context import AgentSessionContext, TaskContext
from meshagent.agents.agent import Agent


def test_agent_session_context_does_not_support_binary_inputs_by_default() -> None:
    context = AgentSessionContext(system_role=None)

    assert context.supports_images is False
    assert context.supports_files is False

    with pytest.raises(RoomException):
        context.append_image_message(mime_type="image/png", data=b"img")

    with pytest.raises(RoomException):
        context.append_image_url(url="https://example.com/image.png")

    with pytest.raises(RoomException):
        context.append_file_message(
            filename="file.txt",
            mime_type="text/plain",
            data=b"file",
        )

    with pytest.raises(RoomException):
        context.append_file_url(url="https://example.com/file.txt")


class _LifecycleContext(AgentSessionContext):
    def __init__(self) -> None:
        super().__init__(system_role=None)
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_agent_session_context_async_manager_calls_start_and_close() -> None:
    context = _LifecycleContext()

    assert context.started == 0
    assert context.closed == 0

    async with context:
        assert context.started == 1
        assert context.closed == 0

    assert context.started == 1
    assert context.closed == 1


class _LegacyContextAgent(Agent):
    async def init_chat_context(self) -> AgentSessionContext:
        ctx = AgentSessionContext(system_role=None)
        ctx.metadata["source"] = "legacy"
        return ctx


@pytest.mark.asyncio
async def test_agent_init_session_uses_legacy_init_chat_context_override() -> None:
    agent = _LegacyContextAgent()

    context = await agent.init_session()

    assert context.metadata["source"] == "legacy"


@pytest.mark.asyncio
async def test_task_context_async_manager_calls_session_start_and_close() -> None:
    session = _LifecycleContext()
    context = TaskContext(
        session=session,
        room=object(),  # type: ignore[arg-type]
        caller=None,
        on_behalf_of=None,
        toolkits=[],
    )

    async with context:
        assert session.started == 1
        assert session.closed == 0

    assert session.started == 1
    assert session.closed == 1


@pytest.mark.asyncio
async def test_chat_thread_context_async_manager_disposes_session() -> None:
    session = _LifecycleContext()
    context = ChatThreadContext(
        session=session,
        thread=object(),  # type: ignore[arg-type]
        path="/threads/test",
        participants=[],
        event_handler=None,
    )

    async with context:
        assert session.started == 1
        assert session.closed == 0

    assert session.started == 1
    assert session.closed == 1
