
from .adapter import LLMAdapter, Toolkit, ToolResponseAdapter
from meshagent.api.schema_util import prompt_schema
from .agent import AgentCallContext
from typing import Optional
from meshagent.agents import TaskRunner
from meshagent.api import RequiredToolkit

# An agent that takes a simple prompt and gets the result
class PromptAgent(TaskRunner):
    def __init__(self,
        *,
        name: str,
        output_schema: dict,
        llm_adapter: LLMAdapter,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        tools: list[Toolkit] = [],
        rules: list[str] = [],
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[RequiredToolkit]] = None,
        supports_tools: Optional[bool] = None,
        labels: Optional[list[str]] = None
    ):
        super().__init__(
            name=name,
            description=description,
            title=title,
            input_schema=prompt_schema(
                description=description
            ),
            output_schema=output_schema,
            requires=requires,
            supports_tools=supports_tools,
            labels=labels
        )
        self.rules = rules
        self.tools = tools
        self.llm_adapter = llm_adapter
        self.tool_adapter = tool_adapter

    async def init_chat_context(self):
        chat = self.llm_adapter.create_chat_context()
        chat.append_rules(self.rules)
        return chat

    async def ask(self, *, context: AgentCallContext, arguments: dict):
        context.chat.append_user_message(arguments["prompt"])

        toolkits = [
            *self.toolkits,
            *context.toolkits
        ]

        return await self.llm_adapter.next(context=context.chat, room=context.room, toolkits=toolkits, tool_adapter=self.tool_adapter, output_schema=self.output_schema)