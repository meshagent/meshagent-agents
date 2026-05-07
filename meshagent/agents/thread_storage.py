from __future__ import annotations

import posixpath
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from meshagent.api import Participant, RoomClient
from meshagent.tools import Toolkit

from .context import AgentSessionContext
from .messages import AgentThreadMessage

if TYPE_CHECKING:
    from .adapter import LLMAdapter


@runtime_checkable
class ThreadStorage(Protocol):
    @staticmethod
    async def allocate_thread_path(
        *,
        room: RoomClient,
        base_path: str,
        extension: str = ".thread",
    ) -> str:
        try:
            exists = await room.storage.exists(path=base_path)
        except Exception:
            return base_path

        if not exists:
            return base_path

        thread_dir, filename = posixpath.split(base_path)
        if extension != "" and filename.endswith(extension):
            base_name = filename[: -len(extension)]
        else:
            base_name = filename

        for index in range(2, 1000):
            candidate = posixpath.join(thread_dir, f"{base_name} {index}{extension}")
            try:
                if not await room.storage.exists(path=candidate):
                    return candidate
            except Exception:
                return candidate

        return posixpath.join(
            thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}{extension}"
        )

    @property
    def path(self) -> str: ...

    def push_message(
        self,
        *,
        message: AgentThreadMessage,
        sender: Participant | None = None,
    ) -> None: ...

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None: ...

    def make_toolkit(self) -> Toolkit: ...
