from typing import Optional

from jsonschema import validate, ValidationError
from meshagent.api.schema_util import prompt_schema, merge
from meshagent.api import Requirement
from meshagent.tools import Toolkit, make_toolkits, ToolkitBuilder
from meshagent.agents import TaskRunner
from meshagent.agents.agent import AgentCallContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
import tarfile
import io
import mimetypes


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
        annotations: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
    ):
        if input_schema is None:
            if input_prompt:
                input_schema = prompt_schema(
                    description="use a prompt to generate content"
                )

                toolkit_builders = self.get_toolkit_builders()
                if len(toolkit_builders) > 0:
                    toolkit_config_schemas = []

                    defs = None

                    for builder in toolkit_builders:
                        schema = builder.type.model_json_schema()
                        if schema.get("$defs") is not None:
                            if defs is None:
                                defs = {}

                            for k, v in schema["$defs"].items():
                                defs[k] = v

                        toolkit_config_schemas.append(schema)

                    input_schema = merge(
                        schema=input_schema,
                        additional_properties={
                            "tools": {
                                "type": "array",
                                "items": {
                                    "anyOf": toolkit_config_schemas,
                                },
                            },
                            "model": {"type": ["string", "null"]},
                        },
                    )

                    if defs is not None:
                        if input_schema.get("$defs") is None:
                            input_schema["$defs"] = {}

                        for k, v in defs.items():
                            input_schema["$defs"][k] = v

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
            annotations=annotations,
        )

        self._extra_rules = rules or []
        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter
        self.toolkits = static_toolkits
        self._client_rules = client_rules

    async def init_chat_context(self):
        chat = self._llm_adapter.create_chat_context()
        return chat

    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return []

    async def get_context_toolkits(self, *, context: AgentCallContext) -> list[Toolkit]:
        return []

    async def get_rules(self, *, context: AgentCallContext):
        rules = [*self._extra_rules]

        participant = context.caller
        client = participant.get_attribute("client")

        if self._client_rules is not None and client is not None:
            cr = self._client_rules.get(client)
            if cr is not None:
                rules.extend(cr)

        return rules

    async def ask(
        self,
        *,
        context: AgentCallContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ):
        prompt = arguments.get("prompt")
        if prompt is None:
            raise ValueError("`prompt` is required")

        message_tools = arguments.get("tools")
        model = arguments.get("model", self._llm_adapter.default_model())

        context.chat.append_rules(await self.get_rules(context=context))

        context.chat.append_user_message(prompt)

        if attachment is not None:
            buf = io.BytesIO(attachment)
            with tarfile.open(fileobj=buf, mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        mime_type, encoding = mimetypes.guess_type(member.name)
                        f = tar.extractfile(member)
                        content = f.read()
                        if mime_type.startswith("image/"):
                            context.chat.append_image_message(
                                data=content, mime_type=mime_type
                            )
                        else:
                            context.chat.append_file_message(
                                filename=member.name, data=content, mime_type=mime_type
                            )

        combined_toolkits: list[Toolkit] = [
            *self.toolkits,
            *context.toolkits,
            *await self.get_context_toolkits(context=context),
        ]

        if message_tools is not None and len(message_tools) > 0:
            combined_toolkits.extend(
                await make_toolkits(
                    room=self.room,
                    model=model,
                    providers=self.get_toolkit_builders(),
                    tools=message_tools,
                )
            )

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
        annotations: Optional[list[str]] = None,
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
            annotations=annotations,
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
