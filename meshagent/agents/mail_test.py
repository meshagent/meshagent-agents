from typing import Optional

import pytest

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.mail import MailBot, NewEmailThreadWithAttachments


class _FakeLLMAdapter(LLMAdapter):
    def default_model(self) -> str:
        return "test-model"

    async def create_response(
        self,
        *,
        context,
        caller,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        options: Optional[dict] = None,
    ):
        del (
            context,
            caller,
            toolkits,
            output_schema,
            event_handler,
            steering_callback,
            model,
            on_behalf_of,
            options,
        )
        return "assistant response"


class _FakeRoom:
    pass


class _FakeHostedToolkit:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_mailbot_start_builds_room_bound_toolkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosted_toolkit = _FakeHostedToolkit()
    started: dict[str, object] = {}

    async def fake_worker_start(self, *, room) -> None:
        self._room = room

    async def fake_worker_stop(self) -> None:
        self._room = None

    async def fake_start_hosted_toolkit(*, room, toolkit):
        started["room"] = room
        started["toolkit"] = toolkit
        return hosted_toolkit

    monkeypatch.setattr("meshagent.agents.worker.Worker.start", fake_worker_start)
    monkeypatch.setattr("meshagent.agents.worker.Worker.stop", fake_worker_stop)
    monkeypatch.setattr(
        "meshagent.agents.mail.start_hosted_toolkit",
        fake_start_hosted_toolkit,
    )

    room = _FakeRoom()
    bot = MailBot(
        queue="inbox",
        llm_adapter=_FakeLLMAdapter(),
        email_address="mailbox@mail.meshagent.com",
        toolkit_name="paemail",
    )

    await bot.start(room=room)  # type: ignore[arg-type]

    toolkit = started["toolkit"]
    assert started["room"] is room
    assert bot._toolkit is toolkit
    assert toolkit.name == "paemail"
    assert len(toolkit.tools) == 1
    tool = toolkit.tools[0]
    assert isinstance(tool, NewEmailThreadWithAttachments)
    assert tool.room is room

    await bot.stop()

    assert hosted_toolkit.stopped is True
    assert bot._toolkit is None
