from meshagent.api.schema import ValueProperty, ChildProperty
from meshagent.tools import Toolkit
from meshagent.agents.writer import Writer, WriterContext
from meshagent.agents.adapter import LLMAdapter
from meshagent.api.schema_util import prompt_schema, merge
from typing import Optional
from meshagent.api import Requirement, RoomClient

import logging

logger = logging.getLogger("planning_agent")


class SingleShotWriter(Writer):
    def __init__(
        self,
        name: str,
        llm_adapter: LLMAdapter,
        description: Optional[str] = None,
        title: Optional[str] = None,
        rules: Optional[list[str]] = None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        supports_tools: Optional[bool] = None,
        annotations: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            title=title,
            input_schema=merge(
                schema=prompt_schema(description="use a prompt to generate content"),
                additional_properties={"path": {"type": "string"}},
            ),
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [],
                "properties": {},
            },
            requires=requires,
            supports_tools=supports_tools,
            annotations=annotations,
        )
        self._rules = rules
        self._llm_adapter = llm_adapter
        if toolkits is None:
            toolkits = []
        self._toolkits = toolkits

    async def init_session(self):
        context = self._llm_adapter.create_session()
        context.append_rules(self._rules)
        return context

    def bind_runtime_credentials(self, *, room: RoomClient) -> None:
        super().bind_runtime_credentials(room=room)
        self._llm_adapter = self._llm_adapter.with_runtime_api_key(
            api_key=self.resolve_runtime_api_key(room=room)
        )

    async def write(self, writer_context: WriterContext, arguments: dict):
        arguments = arguments.copy()
        self.pop_path(arguments=arguments)

        prompt = arguments["prompt"]

        writer_context.call_context.session.append_user_message(message=prompt)

        toolkits = [*self._toolkits, *writer_context.call_context.toolkits]

        try:
            response = await self._llm_adapter.next(
                context=writer_context.call_context.session,
                caller=writer_context.room.local_participant,
                toolkits=toolkits,
                output_schema=writer_context.document.schema.to_json(),
            )

        except Exception as e:
            logger.error("Unable to execute reasoning completion task", exc_info=e)
            # retry
            raise (e)

        document = writer_context.document
        response = response[document.schema.root.tag_name]

        for p in document.schema.root.properties:
            if isinstance(p, ValueProperty):
                document.root[p.name] = response[p.name]
            elif isinstance(p, ChildProperty):
                for value in response[p.name]:
                    document.root.append_json(value)
            else:
                raise Exception("Unexpected property type")

        return {}
