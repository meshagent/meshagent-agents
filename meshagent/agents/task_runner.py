from typing import Optional
from meshagent.api.messaging import unpack_message, pack_message
from meshagent.tools import RemoteToolkit
from meshagent.agents.agent import RoomTool

from meshagent.api import (
    Participant,
    RemoteParticipant,
)
from meshagent.api.protocol import Protocol
from meshagent.tools import (
    Toolkit,
    Tool,
    ToolContext,
)
from meshagent.api.room_server_client import RoomClient
from jsonschema import validate
from .context import TaskContext
from meshagent.api.schema_util import no_arguments_schema
import logging
import asyncio

from meshagent.agents.agent import SingleRoomAgent

logger = logging.getLogger("agent")


class RunTaskTool(Tool):
    def __init__(self, *, agent: "TaskRunner"):
        self.agent = agent
        super().__init__(
            name=f"run_{agent.name}_task",
            title=f"Run {agent.title or agent.name} Task",
            description=agent.description,
            input_schema=agent.input_schema,
        )

    async def execute(
        self, context: ToolContext, *, attachment: Optional[bytes] = None, **kwargs
    ):
        chat_context = await self.agent.init_chat_context()
        call_context = TaskContext(
            chat=chat_context,
            room=context.room,
            caller=context.caller,
            on_behalf_of=context.on_behalf_of,
            toolkits=[],
        )
        return await self.agent.ask(
            context=call_context,
            arguments=kwargs,
            attachment=attachment,
        )


class TaskRunner(SingleRoomAgent):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires=None,
        supports_tools: Optional[bool] = None,
        input_schema: dict,
        output_schema: Optional[dict] = None,
        labels: Optional[list[str]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        annotations: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            labels=labels,
        )

        if toolkits is None:
            toolkits = []

        self._toolkits = toolkits

        self._registration_id = None

        if input_schema is None:
            input_schema = no_arguments_schema(
                description="execute the agent",
            )

        if supports_tools is None:
            supports_tools = False

        self._supports_tools = supports_tools
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._annotations = annotations

        self._worker_toolkit = None

    async def validate_arguments(self, arguments: dict):
        validate(arguments, self.input_schema)

    async def validate_response(self, response: dict):
        if self.output_schema is not None:
            validate(response, self.output_schema)

    async def ask(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ) -> dict:
        raise Exception("Not implemented")

    @property
    def supports_tools(self):
        return self._supports_tools

    @property
    def annotations(self):
        return self._annotations

    @property
    def input_schema(self):
        return self._input_schema

    @property
    def output_schema(self):
        return self._output_schema

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "requires": list(map(lambda x: x.to_json(), self.requires)),
            "supports_tools": self.supports_tools,
            "labels": self.labels,
            "annotations": self.annotations,
        }

    async def start(self, *, room: RoomClient):
        await super().start(room=room)

        self._worker_toolkit = RemoteToolkit(
            name=self.name,
            tools=[
                RunTaskTool(agent=self),
            ],
        )
        await self._worker_toolkit.start(room=room)

    async def stop(self):
        await self._worker_toolkit.stop()

        logger.info(
            f"disconnected '{self.name}' from room, this will automatically happen when all the users leave the room. agents will not keep the room open"
        )

        await super().stop()

    async def _ask(
        self, protocol: Protocol, message_id: int, msg_type: str, data: bytes
    ):
        async def worker():
            # Decode and parse the message
            message, attachment = unpack_message(data)
            logger.info("agent got message %s", message)
            args = message["arguments"]
            task_id = message["task_id"]
            toolkits_json = message["toolkits"]

            # context_json = message["context"]

            chat_context = None

            try:
                chat_context = await self.init_chat_context()

                caller: Participant | None = None
                on_behalf_of: Participant | None = None
                on_behalf_of_id = message.get("on_behalf_of_id", None)

                for participant in self._room.messaging.get_participants():
                    if message["caller_id"] == participant.id:
                        caller = participant
                        break

                    if on_behalf_of_id == participant.id:
                        on_behalf_of = participant
                        break

                if caller is None:
                    caller = RemoteParticipant(
                        id=message["caller_id"], role="user", attributes={}
                    )

                if on_behalf_of_id is not None and on_behalf_of is None:
                    on_behalf_of = RemoteParticipant(
                        id=message["on_behalf_of_id"], role="user", attributes={}
                    )

                tool_target = caller
                if on_behalf_of is not None:
                    tool_target = on_behalf_of

                toolkits = [
                    *self._toolkits,
                    *await self.get_required_toolkits(
                        context=ToolContext(
                            room=self.room,
                            caller=caller,
                            on_behalf_of=on_behalf_of,
                            caller_context={"chat": chat_context.to_json()},
                        )
                    ),
                ]

                context = TaskContext(
                    chat=chat_context,
                    room=self.room,
                    caller=caller,
                    on_behalf_of=on_behalf_of,
                    toolkits=toolkits,
                )

                for toolkit_json in toolkits_json:
                    tools = []
                    for tool_json in toolkit_json["tools"]:
                        tools.append(
                            RoomTool(
                                on_behalf_of_id=on_behalf_of_id,
                                participant_id=tool_target.id,
                                toolkit_name=toolkit_json["name"],
                                name=tool_json["name"],
                                title=tool_json["title"],
                                description=tool_json["description"],
                                input_schema=tool_json["input_schema"],
                                thumbnail_url=toolkit_json["thumbnail_url"],
                                defs=tool_json.get("defs", None),
                            )
                        )

                    context.toolkits.append(
                        Toolkit(
                            name=toolkit_json["name"],
                            title=toolkit_json["title"],
                            description=toolkit_json["description"],
                            thumbnail_url=toolkit_json["thumbnail_url"],
                            tools=tools,
                        )
                    )

                if attachment is not None and len(attachment) > 0:
                    response = await self.ask(
                        context=context, arguments=args, attachment=attachment
                    )
                else:
                    response = await self.ask(context=context, arguments=args)

                await protocol.send(
                    type="agent.ask_response",
                    data=pack_message(
                        {
                            "task_id": task_id,
                            "answer": response,
                            "caller_context": chat_context.to_json(),
                        }
                    ),
                )

            except Exception as e:
                logger.error("Task runner failed to complete task", exc_info=e)
                if chat_context is not None:
                    await protocol.send(
                        type="agent.ask_response",
                        data=pack_message(
                            {
                                "task_id": task_id,
                                "error": str(e),
                                "caller_context": chat_context.to_json(),
                            }
                        ),
                    )
                else:
                    await protocol.send(
                        type="agent.ask_response",
                        data=pack_message(
                            {
                                "task_id": task_id,
                                "error": str(e),
                            }
                        ),
                    )

        def on_done(task: asyncio.Task):
            task.result()

        task = asyncio.create_task(worker())
        task.add_done_callback(on_done)
