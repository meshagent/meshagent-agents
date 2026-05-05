from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from meshagent.api import Participant
from meshagent.tools import Toolkit

from .context import AgentSessionContext
from .messages import AgentMessage

if TYPE_CHECKING:
    from .adapter import LLMAdapter


@runtime_checkable
class ThreadStorage(Protocol):
    @property
    def path(self) -> str: ...

    def push_message(
        self,
        *,
        message: AgentMessage,
        sender: Participant | None = None,
    ) -> None: ...

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: "LLMAdapter[Any] | None" = None,
    ) -> None: ...

    def make_toolkit(self) -> Toolkit: ...
