from typing import Optional

from jsonschema import validate, ValidationError
from meshagent.api.schema_util import prompt_schema, merge
from meshagent.api import Requirement
from meshagent.tools import Toolkit
from meshagent.agents import TaskRunner
from meshagent.agents.agent import AgentCallContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.api.messaging import TextResponse


class LLMTaskRunner(TaskRunner):
    """
    A Task Runner that uses an LLM execution loop until the task is complete.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_adapter: LLMAdapter,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        toolkits: Optional[list[Toolkit]] = None,
        requires: Optional[list[Requirement]] = None,
        supports_tools: bool = True,
        input_prompt: bool = True,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        rules: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
    ):
        if input_schema is None:
            if input_prompt:
                input_schema = prompt_schema(
                    description="use a prompt to generate content"
                )
            else:
                input_schema = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {},
                }

        static_toolkits = list(toolkits or [])

        super().__init__(
            name=name,
            title=title,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            requires=requires,
            supports_tools=supports_tools,
            labels=labels,
            toolkits=static_toolkits,
        )

        self._extra_rules = rules or []
        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter
        self.toolkits = static_toolkits

    async def init_chat_context(self):
        chat = self._llm_adapter.create_chat_context()
        if self._extra_rules:
            chat.append_rules(self._extra_rules)
        return chat

    async def ask(self, *, context: AgentCallContext, arguments: dict):
        prompt = arguments.get("prompt")
        if prompt is None:
            raise ValueError("`prompt` is required")

        context.chat.append_user_message(prompt)

        combined_toolkits: list[Toolkit] = [*self.toolkits, *context.toolkits]

        resp = await self._llm_adapter.next(
            context=context.chat,
            room=context.room,
            toolkits=combined_toolkits,
            tool_adapter=self._tool_adapter,
            output_schema=self.output_schema,
        )

        # Validate the LLM output against the declared output schema if one was provided
        if self.output_schema:
            try:
                validate(instance=resp, schema=self.output_schema)
            except ValidationError as exc:
                raise RuntimeError("LLM output failed schema validation") from exc
        # If no output schema was provided return a TextResponse
        else:
            resp = TextResponse(text=resp)

        return resp


class DynamicLLMTaskRunner(LLMTaskRunner):
    """
    Same capabilities as LLMTaskRunner, but the caller supplies an arbitrary JSON-schema (`output_schema`) at runtime
    """

    def __init__(
        self,
        *,
        name: str,
        llm_adapter: LLMAdapter,
        supports_tools: bool = True,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
    ):
        input_schema = merge(
            schema=prompt_schema(description="use a prompt to generate content"),
            additional_properties={"output_schema": {"type": "object"}},
        )
        super().__init__(
            name=name,
            llm_adapter=llm_adapter,
            supports_tools=supports_tools,
            title=title,
            description=description,
            tool_adapter=tool_adapter,
            toolkits=toolkits,
            rules=rules,
            input_prompt=True,
            input_schema=input_schema,
            output_schema=None,
        )

    async def ask(self, *, context: AgentCallContext, arguments: dict):
        prompt = arguments.get("prompt")
        if prompt is None:
            raise ValueError("`prompt` is required")

        # Parse and pass JSON output schema provided at runtime
        output_schema_raw = arguments.get("output_schema")
        if output_schema_raw is None:
            raise ValueError("`output_schema` is required for DynamicLLMTaskRunner")

        # Make sure provided schema is a dict
        if not isinstance(output_schema_raw, dict):
            raise TypeError("`output_schema` must be a dict (JSON-schema object)")

        context.chat.append_user_message(prompt)

        combined_toolkits: list[Toolkit] = [*self.toolkits, *context.toolkits]

        resp = await self._llm_adapter.next(
            context=context.chat,
            room=context.room,
            toolkits=combined_toolkits,
            tool_adapter=self._tool_adapter,
            output_schema=output_schema_raw,
        )

        try:
            validate(instance=resp, schema=output_schema_raw)
        except ValidationError as exc:
            raise RuntimeError("LLM output failed caller schema validation") from exc

        return resp
