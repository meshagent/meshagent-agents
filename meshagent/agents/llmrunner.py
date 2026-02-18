import logging
import posixpath
import re
from datetime import datetime, timezone
from typing import Literal, Optional

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

logger = logging.getLogger(__name__)

ThreadingMode = Literal["auto", "manual", "none"]

DEFAULT_THREAD_NAME_RULES = [
    "generate a concise topic name for storing this task in a thread",
    "return only a thread_name value suitable for a file name",
    "thread_name should be 2-6 words, lowercase, and topic-focused",
    "do not include slashes or a .thread extension",
]


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
        threading_mode: Optional[ThreadingMode] = None,
        thread_dir: str = ".threads",
        thread_name_rules: Optional[list[str]] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        allow_model_selection: bool = True,
        rules: Optional[list[str]] = None,
        annotations: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
    ):
        self.allow_model_selection = allow_model_selection
        if threading_mode is None:
            resolved_threading_mode: ThreadingMode = "manual" if input_path else "none"
        else:
            resolved_threading_mode = threading_mode

        self.threading_mode = resolved_threading_mode
        self.input_path = resolved_threading_mode == "manual"
        self.thread_dir = thread_dir
        if thread_name_rules is not None and len(thread_name_rules) > 0:
            self.thread_name_rules = [*thread_name_rules]
        else:
            self.thread_name_rules = [*DEFAULT_THREAD_NAME_RULES]

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

                if self.threading_mode == "manual":
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
            annotations=annotations,
            toolkits=static_toolkits,
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

    def _sanitize_thread_name(self, *, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.endswith(".thread"):
            normalized = normalized[: -len(".thread")]

        normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        if normalized == "":
            normalized = "thread"
        return normalized[:64].strip("-") or "thread"

    def _fallback_thread_name(self, *, prompt: str) -> str:
        del prompt
        return f"thread-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    def _thread_path_for_name(self, *, thread_name: str) -> str:
        return posixpath.join(self.thread_dir, f"{thread_name}.thread")

    async def _generate_thread_path(
        self,
        *,
        context: TaskContext,
        prompt: str,
        model: str,
    ) -> str:
        cloned_context = context.chat.copy()
        cloned_context.replace_rules(rules=self.thread_name_rules)
        cloned_context.append_user_message(prompt)

        generated_name = self._fallback_thread_name(prompt=prompt)
        try:
            response = await self._llm_adapter.next(
                context=cloned_context,
                room=context.room,
                model=model,
                on_behalf_of=context.on_behalf_of,
                toolkits=[],
                output_schema={
                    "type": "object",
                    "required": ["thread_name"],
                    "additionalProperties": False,
                    "properties": {
                        "thread_name": {
                            "type": "string",
                            "description": "2-6 word topic name for the task thread",
                        },
                    },
                },
            )
            if isinstance(response, dict):
                thread_name = response.get("thread_name")
                if isinstance(thread_name, str):
                    generated_name = self._sanitize_thread_name(value=thread_name)
        except Exception as ex:
            logger.warning(
                "unable to auto-generate thread name, using fallback", exc_info=ex
            )

        return self._thread_path_for_name(thread_name=generated_name)

    async def resolve_thread_path(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        prompt: str,
        model: str,
    ) -> str | None:
        if self.threading_mode == "none":
            return None

        if self.threading_mode == "manual":
            path = arguments.get("path")
            if path is None:
                return None
            if not isinstance(path, str):
                raise ValueError("`path` must be a string or null")
            if path == "":
                return None
            return path

        return await self._generate_thread_path(
            context=context,
            prompt=prompt,
            model=model,
        )

    def create_thread_adapter(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ) -> ThreadAdapter | None:
        del attachment
        if self.threading_mode == "none":
            return None

        path = arguments.get("path")
        if path is None:
            return None
        if not isinstance(path, str):
            raise ValueError("`path` must be a string or null")
        if path == "":
            return None

        return ThreadAdapter(
            room=context.room,
            path=path,
        )

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

        thread_path = await self.resolve_thread_path(
            context=context,
            arguments=arguments,
            prompt=prompt,
            model=model,
        )
        adapter_arguments = arguments
        if thread_path is not None:
            adapter_arguments = {
                **arguments,
                "path": thread_path,
            }

        thread_adapter = self.create_thread_adapter(
            context=context,
            arguments=adapter_arguments,
            attachment=attachment,
        )

        if thread_adapter is not None:
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
