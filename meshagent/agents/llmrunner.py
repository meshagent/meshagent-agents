from typing import Optional

from jsonschema import validate, ValidationError
from meshagent.api.schema_util import prompt_schema, merge
from meshagent.api import Requirement
from meshagent.tools import Toolkit, make_toolkits, ToolkitBuilder
from meshagent.agents import TaskRunner
from meshagent.agents.task_runner import TaskContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.agents.thread_adapter import ThreadAdapter

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
        llm_adapter: LLMAdapter,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        toolkits: Optional[list[Toolkit]] = None,
        requires: Optional[list[Requirement]] = None,
        supports_tools: bool = True,
        input_prompt: bool = True,
        input_path: bool = False,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        allow_model_selection: bool = True,
        rules: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        annotations: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
    ):
        self.allow_model_selection = allow_model_selection
        self.input_path = input_path

        if input_schema is None:
            if input_prompt:
                input_schema = prompt_schema(
                    description="use a prompt to generate content"
                )

                if allow_model_selection:
                    input_schema = merge(
                        schema=input_schema,
                        additional_properties={
                            "model": {"type": ["string", "null"]},
                        },
                    )

                if input_path:
                    input_schema = merge(
                        schema=input_schema,
                        additional_properties={
                            "path": {"type": ["string", "null"]},
                        },
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

    async def get_context_toolkits(self, *, context: TaskContext) -> list[Toolkit]:
        return []

    async def get_rules(self, *, context: TaskContext):
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
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ):
        prompt = arguments.get("prompt")
        if prompt is None:
            raise ValueError("`prompt` is required")

        message_tools = arguments.get("tools")
        if self.allow_model_selection:
            model = arguments.get("model", self._llm_adapter.default_model())
        else:
            model = self._llm_adapter.default_model()

        path = None
        thread_adapter = None

        if self.input_path:
            path = arguments.get("path")

        if path is not None:
            thread_adapter = ThreadAdapter(
                room=self.room,
                path=path,
            )

            await thread_adapter.start()
            thread_adapter.append_messages(context=context.chat)
            thread_adapter.write_text_message(text=prompt, participant=context.caller)

        try:
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
                                    filename=member.name,
                                    data=content,
                                    mime_type=mime_type,
                                )

            combined_toolkits: list[Toolkit] = [
                *self.toolkits,
                *context.toolkits,
                *await self.get_context_toolkits(context=context),
                *await self.get_required_toolkits(context=context),
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

            def push(event: dict):
                if thread_adapter is not None:
                    thread_adapter.push(event=event)

            resp = await self._llm_adapter.next(
                context=context.chat,
                room=context.room,
                toolkits=combined_toolkits,
                tool_adapter=self._tool_adapter,
                output_schema=self.output_schema,
                event_handler=push,
            )

            # Validate the LLM output against the declared output schema if one was provided
            if self.output_schema:
                try:
                    validate(instance=resp, schema=self.output_schema)
                except ValidationError as exc:
                    raise RuntimeError("LLM output failed schema validation") from exc

            return resp

        finally:
            if thread_adapter is not None:
                await thread_adapter.stop()
