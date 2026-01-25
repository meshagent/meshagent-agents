from typing import Optional
from meshagent.tools import RemoteToolkit

from meshagent.tools import (
    Toolkit,
    Tool,
    ToolContext,
)
from meshagent.api import Participant
from meshagent.api.messaging import ensure_response
from meshagent.api.room_server_client import RoomClient
from jsonschema import validate
from .context import TaskContext
from meshagent.api.schema_util import no_arguments_schema
import logging
from meshagent.tools import Response

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
    ) -> Response | dict | str | None:
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
    ) -> Response | dict | str | None:
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

    async def run(
        self,
        *,
        room: RoomClient,
        arguments: dict,
        attachment: Optional[bytes] = None,
        caller: Optional[Participant] = None,
    ) -> Response:
        await super().start(room=room)
        try:
            runner = RunTaskTool(agent=self)
            response = await runner.execute(
                context=ToolContext(
                    caller=caller or room.local_participant,
                    room=room,
                ),
                attachment=attachment,
                **arguments,
            )

            return ensure_response(response)

        finally:
            await super().stop()

    async def stop(self):
        await self._worker_toolkit.stop()

        logger.info(
            f"disconnected '{self.name}' from room, this will automatically happen when all the users leave the room. agents will not keep the room open"
        )

        await super().stop()
