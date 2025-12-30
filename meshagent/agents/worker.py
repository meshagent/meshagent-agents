from .agent import SingleRoomAgent
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomClient
from meshagent.agents import AgentChatContext
from meshagent.tools import RemoteToolkit, Tool, Toolkit, make_toolkits, ToolkitBuilder
from .adapter import LLMAdapter, ToolResponseAdapter
import asyncio
from typing import Optional
import json
from meshagent.tools import ToolContext
import logging

logger = logging.getLogger("worker")


class SubmitTask(Tool):
    def __init__(self, *, agent: "Worker", queue: str):
        self.queue = queue
        self.agent = agent
        super().__init__(
            name=f"queue_{agent.name}_task",
            title=f"Queue {agent.title} Task",
            description=f"Queues a new task to the worker -- {agent.description}",
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                    },
                },
            },
        )

    async def execute(self, context: ToolContext, *, prompt: str):
        await context.room.queues.send(
            name=self.queue,
            message={
                "prompt": prompt,
            },
            create=True,
        )
        return None


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
        toolkit_name: Optional[str] = None,
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

        if toolkit_name is not None:
            logger.info(f"worker will start toolkit {toolkit_name}")
            self._toolkit = RemoteToolkit(
                name=toolkit_name,
                tools=[
                    SubmitTask(queue=self._queue, agent=self),
                ],
            )
        else:
            self._toolkit = None

    async def start(self, *, room: RoomClient):
        self._done = False

        if self._toolkit is not None:
            await self._toolkit.start(room=room)

        await super().start(room=room)

        self._main_task = asyncio.create_task(self.run(room=room))

    async def stop(self):
        self._done = True

        await asyncio.gather(self._main_task)

        if self._toolkit is not None:
            await self._toolkit.stop()

        await super().stop()

    async def get_rules(self):
        return [*self._rules]

    async def append_message_context(
        self, *, message: dict, chat_context: AgentChatContext
    ):
        prompt = message.get("prompt")
        if prompt is None:
            logger.warning(
                "prompt property not found on worker message, inserting whole message into context"
            )
            prompt = json.dumps(message)

        chat_context.append_user_message(message=prompt)

    async def process_message(
        self,
        *,
        chat_context: AgentChatContext,
        message: dict,
        toolkits: list[Toolkit],
    ):
        await self.append_message_context(message=message, chat_context=chat_context)

        return await self._llm_adapter.next(
            context=chat_context,
            room=self.room,
            toolkits=toolkits,
            tool_adapter=self._tool_adapter,
        )

    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return []

    async def get_message_toolkits(self, *, message: dict) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                room=self.room,
                caller=self.room.local_participant,
                on_behalf_of=None,
            )
        )

        tool_providers = [*self.get_toolkit_builders()]

        model = message.get("model", self._llm_adapter.default_model())

        message_tools = message.get("tools")

        if message_tools is not None and len(message_tools) > 0:
            toolkits.extend(
                await make_toolkits(
                    room=self.room,
                    model=model,
                    providers=tool_providers,
                    tools=message_tools,
                )
            )
        return [*self._toolkits, *toolkits]

    async def run(self, *, room: RoomClient):
        backoff = 0
        while not self._done:
            try:
                message = await room.queues.receive(
                    name=self._queue, create=True, wait=True
                )

                backoff = 0
                if message is not None:
                    logger.info("received message on worker queue")
                    try:
                        chat_context = await self.init_chat_context()

                        chat_context.replace_rules(
                            rules=[
                                *await self.get_rules(),
                            ]
                        )

                        toolkits = await self.get_message_toolkits(message=message)

                        await self.process_message(
                            chat_context=chat_context,
                            message=message,
                            toolkits=toolkits,
                        )

                    except Exception as e:
                        logger.error(f"Failed to process: {e}\n{message}", exc_info=e)

            except Exception as e:
                logger.error(
                    f"Worker error while receiving: {e}, will retry", exc_info=e
                )

                await asyncio.sleep(0.1 * pow(2, backoff))
                backoff = backoff + 1
