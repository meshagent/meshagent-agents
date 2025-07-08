from .agent import SingleRoomAgent
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomClient
from meshagent.agents import AgentChatContext
from meshagent.tools import Toolkit
from .adapter import LLMAdapter, ToolResponseAdapter
import asyncio
from typing import Optional
import json
from meshagent.tools import ToolContext
import logging

logger = logging.getLogger("chat")


class Worker(SingleRoomAgent):
    def __init__(
        self,
        *,
        queue: str,
        name,
        title=None,
        description=None,
        requires=None,
        llm_adapter: LLMAdapter,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
        )

        self._queue = queue

        if toolkits is None:
            toolkits = []

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter

        self._message_channel = Chan[RoomMessage]()

        self._room: RoomClient | None = None
        self._toolkits = toolkits

        if rules is None:
            rules = []

        self._rules = rules
        self._done = False

    async def start(self, *, room: RoomClient):
        self._done = False

        await super().start(room=room)

        self._main_task = asyncio.create_task(self.run(room=room))

    async def stop(self):
        self._done = True

        await asyncio.gather(self._main_task)

        await super().stop()

    async def append_message_context(
        self, *, room: RoomClient, message: dict, chat_context: AgentChatContext
    ):
        chat_context.append_user_message(message=json.dumps(message))

    async def process_message(
        self,
        *,
        chat_context: AgentChatContext,
        room: RoomClient,
        message: dict,
        toolkits: list[Toolkit],
    ):
        return await self._llm_adapter.next(
            context=chat_context,
            room=room,
            toolkits=toolkits,
            tool_adapter=self._tool_adapter,
        )

    async def run(self, *, room: RoomClient):
        toolkits = [
            *await self.get_required_toolkits(
                ToolContext(room=room, caller=room.local_participant)
            ),
            *self._toolkits,
        ]

        while not self._done:
            message = await room.queues.receive(
                name=self._queue, create=True, wait=True
            )
            if message is not None:
                logger.info(f"received message on worker queue {message}")
                try:
                    chat_context = await self.init_chat_context()

                    chat_context.append_rules(
                        rules=[
                            *self._rules,
                        ]
                    )

                    await self.append_message_context(
                        room=room, message=message, chat_context=chat_context
                    )

                    await self.process_message(
                        chat_context=chat_context,
                        room=room,
                        message=message,
                        toolkits=toolkits,
                    )

                except Exception as e:
                    logger.error(f"Failed to process a message {message}", exc_info=e)
