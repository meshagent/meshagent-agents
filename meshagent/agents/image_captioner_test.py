import pytest

from meshagent.api import RoomException
from typing import Optional
from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.image_captioner import LLMImageCaptioner


class _ImageCapableContext(AgentSessionContext):
    @property
    def supports_images(self) -> bool:
        return True

    @property
    def supports_files(self) -> bool:
        return False

    def append_image_message(self, *, mime_type: str, data: bytes) -> dict:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "mime_type": mime_type,
                    "size": len(data),
                }
            ],
        }
        self.messages.append(message)
        return message


class _FakeAdapter(LLMAdapter):
    def __init__(self, *, response: object, image_supported: bool = True):
        self._response = response
        self._image_supported = image_supported
        self.last_context: AgentSessionContext | None = None

    def default_model(self) -> str:
        return "test-model"

    def create_session(self) -> AgentSessionContext:
        if self._image_supported:
            return _ImageCapableContext(system_role=None)
        return AgentSessionContext(system_role=None)

    async def next(
        self,
        *,
        context,
        room,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        options: Optional[dict] = None,
    ):
        del room
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        self.last_context = context
        return self._response


class _DummyRoom:
    pass


@pytest.mark.asyncio
async def test_llm_image_captioner_uses_context_and_extracts_caption_key() -> None:
    adapter = _FakeAdapter(response={"caption": "A cat sleeping on a desk."})
    captioner = LLMImageCaptioner(
        llm_adapter=adapter,
        rules=["describe only the visual contents"],
        prompt="Caption this image in one sentence.",
    )

    caption = await captioner.caption(
        room=_DummyRoom(),
        image_data=b"png-bytes",
        mime_type="image/png",
    )

    assert caption == "A cat sleeping on a desk."
    assert adapter.last_context is not None
    assert adapter.last_context.messages[0]["role"] == "user"
    assert (
        adapter.last_context.messages[0]["content"]
        == "Caption this image in one sentence."
    )
    assert adapter.last_context.messages[1]["content"][0]["type"] == "input_image"


@pytest.mark.asyncio
async def test_llm_image_captioner_requires_image_capable_context() -> None:
    adapter = _FakeAdapter(response="unused", image_supported=False)
    captioner = LLMImageCaptioner(llm_adapter=adapter)

    with pytest.raises(RoomException):
        await captioner.caption(
            room=_DummyRoom(),
            image_data=b"img",
            mime_type="image/png",
        )
