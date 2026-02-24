import logging
import posixpath
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from meshagent.api.schema_util import merge
from meshagent.tools import Toolkit

from .adapter import LLMAdapter
from .context import TaskContext
from .task_runner import TaskRunner
from .thread_adapter import ThreadAdapter
from .responses_thread_adapter import ResponsesThreadAdapter

logger = logging.getLogger(__name__)

ThreadingMode = Literal["auto", "manual", "none"]

DEFAULT_THREAD_NAME_RULES = [
    "generate a concise topic name for storing this task in a thread",
    "return only a thread_name value suitable for a file name",
    "thread_name should be 2-6 words, lowercase, and topic-focused",
    "do not include slashes or a .thread extension",
]


class ThreadedTaskRunner(TaskRunner):
    @staticmethod
    def resolve_threading_mode(
        *,
        threading_mode: Optional[ThreadingMode],
        input_path: bool,
    ) -> ThreadingMode:
        if threading_mode is None:
            return "manual" if input_path else "none"
        return threading_mode

    @staticmethod
    def with_manual_thread_path_schema(
        *,
        input_schema: dict,
        threading_mode: ThreadingMode,
    ) -> dict:
        if threading_mode != "manual":
            return input_schema
        return merge(
            schema=input_schema,
            additional_properties={"path": {"type": ["string", "null"]}},
        )

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
        input_path: bool = False,
        threading_mode: Optional[ThreadingMode] = None,
        thread_dir: str = ".threads",
        thread_name_rules: Optional[list[str]] = None,
        thread_name_adapter: Optional[LLMAdapter] = None,
        thread_adapter_type: type[ThreadAdapter] = ResponsesThreadAdapter,
    ):
        resolved_threading_mode = self.resolve_threading_mode(
            threading_mode=threading_mode,
            input_path=input_path,
        )

        if resolved_threading_mode == "auto" and thread_name_adapter is None:
            raise ValueError(
                "`llm_adapter` is required when `threading_mode` is 'auto'"
            )

        self.threading_mode = resolved_threading_mode
        self.input_path = resolved_threading_mode == "manual"
        self.thread_dir = thread_dir
        if thread_name_rules is not None and len(thread_name_rules) > 0:
            self.thread_name_rules = [*thread_name_rules]
        else:
            self.thread_name_rules = [*DEFAULT_THREAD_NAME_RULES]
        self._thread_name_adapter = thread_name_adapter
        self._thread_adapter_type = thread_adapter_type

        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            supports_tools=supports_tools,
            input_schema=input_schema,
            output_schema=output_schema,
            annotations=annotations,
            toolkits=toolkits,
        )

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
        if self._thread_name_adapter is None:
            raise RuntimeError(
                "auto threading mode requires a configured llm adapter for thread naming"
            )

        cloned_context = context.chat.copy()
        generated_name = self._fallback_thread_name(prompt=prompt)
        async with cloned_context:
            cloned_context.replace_rules(rules=self.thread_name_rules)
            cloned_context.append_user_message(prompt)

            try:
                response = await self._thread_name_adapter.next(
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
            return self._selected_thread_path(arguments=arguments)

        return await self._generate_thread_path(
            context=context,
            prompt=prompt,
            model=model,
        )

    def _selected_thread_path(self, *, arguments: dict) -> str | None:
        if self.threading_mode == "none":
            return None

        path = arguments.get("path")
        if path is None:
            return None
        if not isinstance(path, str):
            raise ValueError("`path` must be a string or null")

        selected_path = path.strip()
        if selected_path == "":
            return None
        return selected_path

    def create_thread_adapter(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ) -> ThreadAdapter | None:
        del attachment
        selected_path = self._selected_thread_path(arguments=arguments)
        if selected_path is None:
            return None

        return self._thread_adapter_type(
            room=context.room,
            path=selected_path,
        )
