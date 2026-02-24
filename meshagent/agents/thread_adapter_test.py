import pytest

from meshagent.agents.thread_adapter import ThreadAdapter


class _FakeThreadAdapter(ThreadAdapter):
    def __init__(self) -> None:
        super().__init__(room=object(), path="/threads/test")  # type: ignore[arg-type]
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def handle_custom_event(self, *, messages, event) -> None:
        del messages
        del event

    async def _process_llm_events(self) -> None:
        return None


@pytest.mark.asyncio
async def test_thread_adapter_async_manager_calls_start_and_stop() -> None:
    adapter = _FakeThreadAdapter()

    async with adapter:
        assert adapter.started == 1
        assert adapter.stopped == 0

    assert adapter.started == 1
    assert adapter.stopped == 1
