
from meshagent.api.schema import ValueProperty, ChildProperty
from meshagent.tools import Toolkit
from meshagent.agents.writer import Writer, WriterContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.api.schema_util import prompt_schema, merge
from typing import Optional
from meshagent.api import Requirement

import logging
logging.basicConfig()
logger = logging.getLogger("planning_agent")
logger.setLevel(logging.INFO)


class SingleShotWriter(Writer):

    def __init__(self,  name: str,  llm_adapter: LLMAdapter, tool_adapter:  Optional[ToolResponseAdapter] = None, description: Optional[str] = None, title: Optional[str] = None, rules:Optional[list[str]] = None, requires: Optional[list[Requirement]] = None, toolkits : Optional[list[Toolkit]] = None, supports_tools: Optional[bool] = None, labels: Optional[list[str]] = None, ):
        super().__init__(
            name=name,
            description=description,
            title=title,
            input_schema=merge(
                schema=prompt_schema(description="use a prompt to generate content"),
                additional_properties={
                    "path" : { "type" : "string" }
                }),
                output_schema={
                    "type" : "object",
                    "additionalProperties" : False,
                    "required" : [],
                    "properties" : {
                    }
                },
            requires=requires,
            supports_tools=supports_tools,
        )
        self._rules = rules
        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter 
        if toolkits == None:
            toolkits = []
        self._toolkits = toolkits
        self._labels = labels
    

    async def init_chat_context(self):
        context = self._llm_adapter.create_chat_context()
        context.append_rules(self._rules)
        return context

    async def write(self, writer_context: WriterContext, arguments: dict):

        arguments = arguments.copy()
        self.pop_path(arguments=arguments)

        prompt = arguments["prompt"]

        writer_context.call_context.chat.append_user_message(message=prompt)
            
        toolkits = [
            *self._toolkits,
            *writer_context.call_context.toolkits
        ]
        
        try:
            logger.info("SINGLE SHOT COMPLETION STARTING")
            
            response = await self._llm_adapter.next(context=writer_context.call_context.chat, room=writer_context.room, toolkits=toolkits, tool_adapter=self._tool_adapter, output_schema=writer_context.document.schema.to_json())

            logger.info("SINGLE SHOT COMPLETION RESPONSE %s", response)
        
        except Exception as e:
            logger.error("Unable to execute reasoning completion task", exc_info=e)
            # retry
            raise(e)
        
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

 
