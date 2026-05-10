from abc import ABC, abstractmethod
from typing import Optional

from meshagent.api import RoomClient, RoomException

from .adapter import LLMAdapter

_DEFAULT_IMAGE_CAPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["caption"],
    "properties": {
        "caption": {
            "type": "string",
            "description": "a concise human-readable caption for the image",
        }
    },
}

_DEFAULT_IMAGE_CAPTION_PROMPT = (
    "Write a concise caption for this image. Focus on the important visual content."
)


class ImageCaptioner(ABC):
    @abstractmethod
    async def caption(
        self,
        *,
        room: RoomClient,
        image_data: bytes,
        mime_type: str,
    ) -> str: ...


class LLMImageCaptioner(ImageCaptioner):
    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        output_schema: Optional[dict] = None,
        rules: Optional[list[str]] = None,
        prompt: Optional[str] = None,
    ):
        self._llm_adapter = llm_adapter
        self._output_schema = (
            output_schema
            if output_schema is not None
            else _DEFAULT_IMAGE_CAPTION_SCHEMA.copy()
        )
        self._rules = [*(rules or [])]
        self._prompt = (
            prompt
            if isinstance(prompt, str) and prompt.strip() != ""
            else _DEFAULT_IMAGE_CAPTION_PROMPT
        )

    @staticmethod
    def _extract_caption(response: object) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            for key in ("caption", "text", "description", "summary"):
                value = response.get(key)
                if isinstance(value, str) and value.strip() != "":
                    return value.strip()

        return str(response).strip()

    async def caption(
        self,
        *,
        room: RoomClient,
        image_data: bytes,
        mime_type: str,
    ) -> str:
        session_context = self._llm_adapter.create_session()
        async with session_context:
            if not session_context.supports_images:
                raise RoomException(
                    "llm adapter chat context does not support image inputs for captioning"
                )

            if len(self._rules) > 0:
                session_context.append_rules(self._rules)

            session_context.append_user_message(self._prompt)
            session_context.append_image_message(mime_type=mime_type, data=image_data)

            response = await self._llm_adapter.create_response(
                context=session_context,
                caller=room.local_participant,
                toolkits=[],
                output_schema=self._output_schema,
            )
            return self._extract_caption(response)
