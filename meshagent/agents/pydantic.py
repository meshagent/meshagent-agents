from pydantic import BaseModel
from abc import abstractmethod

import pydantic_ai.models
from .agent import AgentCallContext
from .writer import Writer, WriterContext
import logging
from typing import Optional
from meshagent.api import RoomClient

from .agent import TaskRunner, RequiredToolkit, Requirement, RequiredSchema

from .adapter import  ToolResponseAdapter
from meshagent.tools.toolkit import Tool, Response, ToolContext
from meshagent.tools.pydantic import get_pydantic_ai_tool_definition

from typing import Sequence
import pydantic_ai

logger = logging.getLogger("pydantic_agent")
logger.setLevel(logging.INFO)


def get_pydantic_ai_tool(*, room: RoomClient, tool: Tool, response_adapter: ToolResponseAdapter) -> pydantic_ai.tools.Tool:
     async def prepare(ctx: pydantic_ai.RunContext, tool_def: pydantic_ai.tools.ToolDefinition):
         return get_pydantic_ai_tool_definition(tool=tool)
     
     async def execute(**kwargs):
         response = await tool.execute(context=ToolContext(room=room, caller=room.local_participant), **kwargs)
         return await response_adapter.to_plain_text(room=room, response=response)

     return pydantic_ai.Tool(
        name=tool.name,
        takes_ctx=False,
        description=tool.description,
        prepare=prepare,
        function=execute
    )

def get_pydantic_ai_tools_from_context(*, context: AgentCallContext, response_adapter: ToolResponseAdapter) -> list[pydantic_ai.tools.Tool]:

    tools = list[pydantic_ai.tools.Tool]()

    for toolkit in context.toolkits:
        
        for tool in toolkit.tools:

            tools.append(get_pydantic_ai_tool(room=context.room, tool=tool, response_adapter=response_adapter))

    return tools

class PydanticAgent[TInput:BaseModel, TOutput:BaseModel](TaskRunner):
    def __init__(self,
        *,
        name: str,
        input_model: TInput,
        output_model: TOutput,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[Requirement]] = None,
        supports_tools : Optional[bool] = None,
    ):
        self.input_model = input_model
        self.output_model = output_model
        
        super().__init__(
            name = name,
            description = description,
            title = title,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),   
            requires=requires,
            supports_tools=supports_tools
        )
    
    async def ask(self, *, context: AgentCallContext, arguments: dict):
        try:
            input = self.input_model.model_validate(arguments)
            output = await self.ask_model(context=context, arguments=input)
            return output.model_dump(mode='json')
        except Exception as e:
            logger.error("Unhandled exception in ask agent call", exc_info=e)
            raise

    @abstractmethod
    async def ask_model(self, context: AgentCallContext, arguments: TInput) -> TOutput:
        pass

class PydanticWriter[TInput:BaseModel, TOutput:BaseModel](Writer):
    def __init__(self, 
        name: str,
        input_model: TInput,
        output_model: TOutput,
        title: Optional[str] = None,
        description: Optional[str] = None
    ):

        self.input_model = input_model
        self.output_model = output_model

        super().__init__(           
            name=name,
            description=description,
            title=title, 
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
        )

    async def write(self, writer_context: WriterContext, arguments: dict):
        try:
            input = self.input_model.model_validate(arguments)
            output = await self.write_model(writer_context=writer_context, arguments=input)
            return output.model_dump(mode='json')
        except Exception as e:
            logger.error("Unhandled exception in write call", exc_info=e)
            raise
    
    @abstractmethod
    async def write_model(self, writer_context: WriterContext, arguments: TInput) -> TOutput:
        pass



class PydanicAgentHost[TInput:BaseModel, TOutput:BaseModel](PydanticAgent[TInput, TOutput]):
    
    def __init__(self, *, 
                model: pydantic_ai.models.KnownModelName, 
                system_prompt: str | Sequence[str],
                response_adapter: ToolResponseAdapter, 
                tools: Optional[list[pydantic_ai.Tool]] = None,
                name, input_model, output_model, title = None, description = None, requires = None, supports_tools = None):
        super().__init__(
            name=name, input_model=input_model, output_model=output_model, title=title, description=description, requires=requires, supports_tools=supports_tools, 
            
        )
        self._model = model
        self._system_prompt = system_prompt
        self._response_adapter = response_adapter
        if tools == None: 
            tools = []
            
        self._tools = tools

    def message_history(self, context: AgentCallContext) -> None | list:
        return None
    
    def to_prompt(self, *, arguments: TInput) -> str:
        return str(arguments)

    def create_agent(self, *, tools: list[pydantic_ai.Tool]) -> pydantic_ai.Agent[TInput,TOutput]:
        return pydantic_ai.Agent( 
            model=self._model,
            system_prompt=self._system_prompt,
            deps_type=self.input_model,
            result_type=self.output_model,
            tools=[
                *self._tools,
                *tools
            ]
        )

    @abstractmethod
    async def ask_model(self, context: AgentCallContext, arguments: TInput) -> TOutput:
        
        agent = self.create_agent(tools=get_pydantic_ai_tools_from_context(context=context, response_adapter=self._response_adapter))
        result = await agent.run(self.to_prompt(arguments=arguments), message_history=self.message_history(context), result_type=self.output_model, deps=arguments)
        return result.data