from typing import Optional

from jsonschema import validate, ValidationError
from meshagent.api.schema_util import prompt_schema, merge
from meshagent.api import Requirement
from meshagent.tools import Toolkit, make_toolkits, ToolkitBuilder
from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.completions_thread_adapter import CompletionsThreadAdapter
from meshagent.agents.task_runner import TaskContext
from meshagent.agents.responses_thread_adapter import ResponsesThreadAdapter
from meshagent.agents.threaded_task_runner import ThreadedTaskRunner, ThreadingMode
from meshagent.agents.toolkit_schema import build_tools_property_schema
from meshagent.openai.tools.completions_adapter import OpenAICompletionsAdapter

import tarfile
import io
import mimetypes

import logging

logger = logging.getLogger("llm-runner")

ThreadAdapter = ResponsesThreadAdapter
CompletionsAdapterThreadAdapter = CompletionsThreadAdapter


class LLMTaskRunner(ThreadedTaskRunner):
    """
    A Task Runner that uses an LLM execution loop until the task is complete.
    """

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        title: Optional[str] = None,
        description: Optional[str] = None,
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
        resolved_threading_mode = self.resolve_threading_mode(
            threading_mode=threading_mode,
            input_path=input_path,
        )

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

                input_schema = self.with_manual_thread_path_schema(
                    input_schema=input_schema,
                    threading_mode=resolved_threading_mode,
                )

                tools_schema, defs = build_tools_property_schema(
                    toolkit_builders=self.get_toolkit_builders()
                )
                if tools_schema is not None:
                    input_schema = merge(
                        schema=input_schema,
                        additional_properties={
                            "tools": tools_schema,
                        },
                    )

                    if len(defs) > 0:
                        if input_schema.get("$defs") is None:
                            input_schema["$defs"] = {}

                        for key, value in defs.items():
                            input_schema["$defs"][key] = value

            else:
                input_schema = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [],
                    "properties": {},
                }

        static_toolkits = list(toolkits or [])
        thread_adapter_type = ThreadAdapter
        if isinstance(llm_adapter, OpenAICompletionsAdapter):
            thread_adapter_type = CompletionsAdapterThreadAdapter

        super().__init__(
            title=title,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            requires=requires,
            supports_tools=supports_tools,
            annotations=annotations,
            toolkits=static_toolkits,
            input_path=input_path,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_name_rules=thread_name_rules,
            thread_name_adapter=llm_adapter,
            thread_adapter_type=thread_adapter_type,
        )

        self._extra_rules = rules or []
        self._llm_adapter = llm_adapter
        self.toolkits = static_toolkits
        self._client_rules = client_rules

    async def init_session(self):
        chat = self._llm_adapter.create_session()
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

    async def is_done(self, *, context: TaskContext):
        return context.turn_count > 0

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
            thread_adapter.append_messages(context=context.session)
            thread_adapter.write_text_message(text=prompt, participant=context.caller)

        try:
            context.session.append_rules(await self.get_rules(context=context))

            context.session.append_user_message(prompt)

            if attachment is not None:
                buf = io.BytesIO(attachment)
                with tarfile.open(fileobj=buf, mode="r:*") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            mime_type, encoding = mimetypes.guess_type(member.name)
                            del encoding
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            content = f.read()

                            normalized_mime_type = (
                                mime_type or "application/octet-stream"
                            )
                            if (
                                normalized_mime_type.startswith("image/")
                                and context.session.supports_images
                            ):
                                context.session.append_image_message(
                                    data=content, mime_type=normalized_mime_type
                                )
                            elif context.session.supports_files:
                                context.session.append_file_message(
                                    filename=member.name,
                                    data=content,
                                    mime_type=normalized_mime_type,
                                )
                            else:
                                context.session.append_user_message(
                                    f"the user attached a file named '{member.name}' with mime type '{normalized_mime_type}'"
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

            while not self.is_done(context=context):
                try:
                    resp = await self._llm_adapter.next(
                        context=context.session,
                        room=context.room,
                        toolkits=combined_toolkits,
                        output_schema=self.output_schema,
                        event_handler=push,
                    )

                except Exception as ex:
                    logger.error("unexpected error during task execution", exc_info=ex)

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

    def create_thread_adapter(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ):
        del attachment
        selected_path = self._selected_thread_path(arguments=arguments)
        if selected_path is None:
            return None

        if issubclass(self._thread_adapter_type, ResponsesThreadAdapter):
            return self._thread_adapter_type(
                room=context.room,
                path=selected_path,
            )

        return self._thread_adapter_type(
            room=context.room,
            path=selected_path,
        )
