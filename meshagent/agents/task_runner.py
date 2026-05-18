from typing import Optional
from meshagent.tools import (
    Toolkit,
    LocalRoomTool,
    ToolContext,
    RoomToolContext,
)
from meshagent.tools.hosting import _RemoteToolkitWrapper, _start_hosted_toolkit
from meshagent.api import Participant
from meshagent.api.messaging import ensure_content
from meshagent.api.room_server_client import RoomClient
from jsonschema import validate
from .context import TaskContext
from .thread_adapter import ThreadAdapter
from meshagent.api.schema_util import no_arguments_schema
import logging
from meshagent.tools import Content

from meshagent.agents.agent import SingleRoomAgent

logger = logging.getLogger("agent")


class RunTaskTool(LocalRoomTool):
    def __init__(self, *, agent: "TaskRunner", room: RoomClient):
        self.agent = agent
        super().__init__(
            room=room,
            name=f"run_{agent.name}_task",
            title=f"Run {agent.title or agent.name} Task",
            description=agent.description,
            input_schema=agent.input_schema,
        )

    async def execute(
        self, context: ToolContext, *, attachment: Optional[bytes] = None, **kwargs
    ) -> Content | dict | str | None:
        session_context = await self.agent.init_session()
        call_context = TaskContext(
            session=session_context,
            room=self.room,
            caller=context.caller,
            on_behalf_of=context.on_behalf_of,
            toolkits=[],
        )
        async with call_context:
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
        annotations: Optional[list[str]] = None,
        toolkits: Optional[list[Toolkit]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            annotations=annotations,
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

        self._worker_toolkit: Toolkit | None = None
        self._hosted_worker_toolkit: _RemoteToolkitWrapper | None = None

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
    ) -> Content | dict | str | None:
        raise Exception("Not implemented")

    def create_thread_adapter(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ) -> ThreadAdapter | None:
        del context, arguments, attachment
        return None

    @property
    def supports_tools(self):
        return self._supports_tools

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
            "annotations": self.annotations,
        }

    async def start(self, *, room: RoomClient):
        await super().start(room=room)

        self._worker_toolkit = Toolkit(
            name=self.name,
            tools=[
                RunTaskTool(agent=self, room=room),
            ],
        )
        self._hosted_worker_toolkit = await _start_hosted_toolkit(
            room=room,
            toolkit=self._worker_toolkit,
        )

    async def run(
        self,
        *,
        room: RoomClient,
        arguments: dict,
        attachment: Optional[bytes] = None,
        caller: Optional[Participant] = None,
    ) -> Content:
        await super().start(room=room)
        try:
            runner = RunTaskTool(agent=self, room=room)
            response = await runner.execute(
                context=RoomToolContext(
                    room=room,
                    caller=caller or room.local_participant,
                ),
                attachment=attachment,
                **arguments,
            )

            return ensure_content(response)

        finally:
            await super().stop()

    async def stop(self):
        if self._hosted_worker_toolkit is not None:
            await self._hosted_worker_toolkit.stop()
            self._hosted_worker_toolkit = None

        logger.info(
            f"disconnected '{self.name}' from room, this will automatically happen when all the users leave the room. agents will not keep the room open"
        )

        await super().stop()
