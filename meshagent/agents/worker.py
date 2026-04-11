from .agent import SingleRoomAgent
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomClient
from meshagent.agents import AgentSessionContext
from meshagent.agents.context import TaskContext
from meshagent.tools import (
    FunctionTool,
    Toolkit,
    make_toolkits,
    ToolkitBuilder,
)
from meshagent.tools.hosting import _RemoteToolkitWrapper, _start_hosted_toolkit
from .adapter import LLMAdapter
import asyncio
import contextlib
from typing import Literal, Optional
import json
from meshagent.tools import ToolContext
import logging

from pathlib import Path
from meshagent.agents.skills import to_prompt
from meshagent.openai.tools.completions_adapter import OpenAICompletionsAdapter
from meshagent.openai import OpenAIResponsesAdapter

from .completions_thread_adapter import CompletionsThreadAdapter
from .responses_thread_adapter import ResponsesThreadAdapter
from .thread_adapter import ThreadAdapter
from .threaded_task_runner import ThreadedTaskRunner, ThreadingMode

logger = logging.getLogger("worker")
InitialMessageMode = Literal["summary", "code", "none"]


def _summarize_worker_message(*, message: object) -> str:
    if not isinstance(message, dict):
        return f"type={type(message).__name__}"

    keys = sorted(str(key) for key in message.keys())
    if len(keys) > 8:
        shown_keys = [*keys[:8], f"+{len(keys) - 8} more"]
    else:
        shown_keys = keys

    summary: list[str] = [f"keys={shown_keys}"]

    prompt_value = message.get("prompt")
    if isinstance(prompt_value, str):
        prompt_preview = " ".join(prompt_value.split())
        if len(prompt_preview) > 120:
            prompt_preview = f"{prompt_preview[:117]}..."
        summary.append(f"prompt={prompt_preview!r}")

    body_value = message.get("body")
    if isinstance(body_value, str):
        summary.append(f"body_len={len(body_value)}")

    return ", ".join(summary)


class _WorkerThreadingHelper(ThreadedTaskRunner):
    def __init__(
        self,
        *,
        threading_mode: Optional[ThreadingMode],
        thread_dir: str,
        thread_name_rules: Optional[list[str]],
        thread_name_adapter: Optional[LLMAdapter],
        thread_adapter_type: type[ThreadAdapter],
    ):
        super().__init__(
            input_schema={
                "type": "object",
                "additionalProperties": True,
                "required": [],
                "properties": {},
            },
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_name_rules=thread_name_rules,
            thread_name_adapter=thread_name_adapter,
            thread_adapter_type=thread_adapter_type,
        )


class SubmitWork(FunctionTool):
    def __init__(self, *, agent: "Worker", queue: str):
        self.queue = queue
        self.agent = agent
        super().__init__(
            name=f"queue_{agent.name}_task",
            title=f"Queue {agent.title} Task",
            description=f"Queues a new task to the worker -- {agent.description}",
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                    },
                },
            },
        )

    async def execute(self, context: ToolContext, *, prompt: str):
        await context.room.queues.send(
            name=self.queue,
            message={
                "prompt": prompt,
            },
            create=True,
        )
        return None


class Worker(SingleRoomAgent):
    def __init__(
        self,
        *,
        queue: str,
        name=None,
        title=None,
        description=None,
        requires=None,
        llm_adapter: LLMAdapter,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        toolkit_name: Optional[str] = None,
        skill_dirs: Optional[list[str]] = None,
        annotations: Optional[list[str]] = None,
        threading_mode: Optional[ThreadingMode] = None,
        thread_dir: str = ".threads",
        thread_name_rules: Optional[list[str]] = None,
        thread_name_adapter: Optional[LLMAdapter] = None,
        initial_message_mode: InitialMessageMode = "code",
        initial_message_from: str = "worker",
        decision_model: Optional[str] = None,
        decision_llm_adapter: Optional[LLMAdapter] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            annotations=annotations,
        )

        self._skill_dirs = skill_dirs

        self._queue = queue

        if toolkits is None:
            toolkits = []

        self._llm_adapter = llm_adapter
        self._initial_message_mode: InitialMessageMode = initial_message_mode
        normalized_initial_message_from = initial_message_from.strip()
        if normalized_initial_message_from == "":
            raise ValueError("initial_message_from must not be empty")
        self._initial_message_from = normalized_initial_message_from
        self._decision_model = (
            decision_model.strip()
            if isinstance(decision_model, str) and decision_model.strip() != ""
            else None
        )
        if self._initial_message_mode == "summary":
            self._decision_llm_adapter = (
                decision_llm_adapter
                if decision_llm_adapter is not None
                else OpenAIResponsesAdapter()
            )
        else:
            self._decision_llm_adapter = decision_llm_adapter
        resolved_thread_name_adapter = (
            thread_name_adapter if thread_name_adapter is not None else llm_adapter
        )
        thread_adapter_type: type[ThreadAdapter] = ResponsesThreadAdapter
        if isinstance(llm_adapter, OpenAICompletionsAdapter):
            thread_adapter_type = CompletionsThreadAdapter
        self._threading_helper = _WorkerThreadingHelper(
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_name_rules=thread_name_rules,
            thread_name_adapter=resolved_thread_name_adapter,
            thread_adapter_type=thread_adapter_type,
        )

        self._message_channel = Chan[RoomMessage]()

        self._room: RoomClient | None = None
        self._toolkits = toolkits

        if rules is None:
            rules = []

        self._rules = rules
        self._done = False

        if toolkit_name is not None:
            logger.info(f"worker will start toolkit {toolkit_name}")
            self._worker_toolkit = Toolkit(
                name=toolkit_name,
                tools=[
                    SubmitWork(queue=self._queue, agent=self),
                ],
            )
        else:
            self._worker_toolkit = None
        self._hosted_worker_toolkit: _RemoteToolkitWrapper | None = None

    def _serialize_initial_message_payload(
        self,
        *,
        message: dict,
        prompt: str,
    ) -> tuple[str, str]:
        prompt_from_message = message.get("prompt")
        payload_value: object
        if isinstance(prompt_from_message, str) and prompt_from_message.strip() != "":
            payload_value = prompt_from_message
        elif prompt.strip() != "":
            payload_value = prompt
        else:
            payload_value = message

        if isinstance(payload_value, str):
            normalized_payload = payload_value.strip()
            if normalized_payload != "":
                try:
                    parsed_payload = json.loads(normalized_payload)
                except json.JSONDecodeError:
                    return payload_value, "text"

                return (
                    json.dumps(parsed_payload, indent=2, ensure_ascii=False),
                    "json",
                )

            return payload_value, "text"

        try:
            return (
                json.dumps(payload_value, indent=2, ensure_ascii=False, default=str),
                "json",
            )
        except TypeError:
            return str(payload_value), "text"

    def _format_initial_message_code_block(
        self,
        *,
        payload_text: str,
        language: str,
    ) -> str:
        normalized_language = language.strip().lower()
        if normalized_language == "":
            normalized_language = "text"
        return f"```{normalized_language}\n{payload_text.rstrip()}\n```"

    async def _summarize_initial_message_payload(
        self,
        *,
        payload_text: str,
        language: str,
    ) -> Optional[str]:
        if self._decision_llm_adapter is None:
            return None

        decision_context = self._decision_llm_adapter.create_session()
        async with decision_context:
            decision_context.append_rules(
                [
                    "Summarize worker queue requests, in a manner suitable for a conversation.",
                    "Keep it concise and factual.",
                    "Return only key details useful for later debugging.",
                    "Do not include markdown code fences in the summary.",
                ]
            )
            decision_context.append_user_message(
                (f"Summarize what this request is:\n```{language}\n{payload_text}\n```")
            )
            response = await self._decision_llm_adapter.next(
                context=decision_context,
                room=self.room,
                toolkits=[],
                model=self._decision_model
                or self._decision_llm_adapter.default_model(),
                output_schema={
                    "type": "object",
                    "required": ["summary"],
                    "additionalProperties": False,
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "concise summary, suitable for a conversation"
                            ),
                        },
                    },
                },
            )

        if not isinstance(response, dict):
            return None

        summary = response.get("summary")
        if not isinstance(summary, str):
            return None
        normalized_summary = summary.strip()
        if normalized_summary == "":
            return None
        return normalized_summary

    async def _build_initial_thread_message(
        self,
        *,
        message: dict,
        prompt: str,
    ) -> Optional[str]:
        if self._initial_message_mode == "none":
            return None

        payload_text, payload_language = self._serialize_initial_message_payload(
            message=message,
            prompt=prompt,
        )
        if payload_text.strip() == "":
            return None

        if self._initial_message_mode == "summary":
            try:
                summary = await self._summarize_initial_message_payload(
                    payload_text=payload_text,
                    language=payload_language,
                )
            except Exception as ex:
                logger.warning(
                    "unable to summarize initial worker payload", exc_info=ex
                )
                summary = None
            if summary is not None:
                return summary

        return self._format_initial_message_code_block(
            payload_text=payload_text,
            language=payload_language,
        )

    async def preflight_start(self, *, room: RoomClient) -> None:
        del room

    async def start(self, *, room: RoomClient):
        self._done = False

        worker_toolkit_started = False
        room_agent_started = False
        try:
            if self._worker_toolkit is not None:
                self._hosted_worker_toolkit = await _start_hosted_toolkit(
                    room=room,
                    toolkit=self._worker_toolkit,
                )
                worker_toolkit_started = True

            await super().start(room=room)
            room_agent_started = True

            await self.preflight_start(room=room)

            self._main_task = asyncio.create_task(self.run(room=room))
        except Exception:
            self._done = True

            if room_agent_started:
                with contextlib.suppress(Exception):
                    await super().stop()

            if worker_toolkit_started and self._hosted_worker_toolkit is not None:
                with contextlib.suppress(Exception):
                    await self._hosted_worker_toolkit.stop()
                self._hosted_worker_toolkit = None
            raise

    async def stop(self):
        self._done = True

        await asyncio.gather(self._main_task)

        if self._hosted_worker_toolkit is not None:
            await self._hosted_worker_toolkit.stop()
            self._hosted_worker_toolkit = None

        await super().stop()

    async def get_rules(self):
        rules = [*self._rules]

        if self._skill_dirs is not None and len(self._skill_dirs) > 0:
            rules.append(
                "You have access to to following skills which follow the agentskills spec:"
            )
            rules.append(await to_prompt([*(Path(p) for p in self._skill_dirs)]))
            rules.append(
                "Use the shell or storage tool to find out more about skills and execute them when they are required"
            )

        return rules

    def get_prompt_for_message(self, *, message: dict) -> str:
        prompt = message.get("prompt")
        if prompt is None:
            logger.warning(
                "prompt property not found on worker message, inserting whole message into context"
            )
            prompt = json.dumps(message)

        return prompt

    async def append_message_context(
        self, *, message: dict, chat_context: AgentSessionContext
    ):
        prompt = self.get_prompt_for_message(message=message)

        chat_context.append_user_message(message=prompt)

    async def process_message(
        self,
        *,
        chat_context: AgentSessionContext,
        message: dict,
        toolkits: list[Toolkit],
    ):
        prompt = self.get_prompt_for_message(message=message)
        model = message.get("model", self._llm_adapter.default_model())
        if not isinstance(model, str) or model.strip() == "":
            model = self._llm_adapter.default_model()

        task_context = TaskContext(
            session=chat_context,
            room=self.room,
            caller=None,
            on_behalf_of=None,
            toolkits=[],
        )

        adapter_arguments = message
        thread_path = await self._threading_helper.resolve_thread_path(
            context=task_context,
            arguments=message,
            prompt=prompt,
            model=model,
        )
        if thread_path is not None:
            adapter_arguments = {
                **message,
                "path": thread_path,
            }
            await self._threading_helper.record_thread_in_index(
                context=task_context,
                path=thread_path,
            )

        thread_adapter = self._threading_helper.create_thread_adapter(
            context=task_context,
            arguments=adapter_arguments,
            attachment=None,
        )

        if thread_adapter is not None:
            await thread_adapter.start()
            self._threading_helper.ensure_local_member_on_thread(
                context=task_context,
                thread_adapter=thread_adapter,
            )
            thread_adapter.append_messages(context=chat_context)
            initial_message = await self._build_initial_thread_message(
                message=message,
                prompt=prompt,
            )
            if initial_message is not None:
                thread_adapter.write_text_message(
                    text=initial_message,
                    participant=self._initial_message_from,
                )

        def push(event: dict) -> None:
            if thread_adapter is not None:
                thread_adapter.push(event=event)

        try:
            await self.append_message_context(
                message=message, chat_context=chat_context
            )

            return await self._llm_adapter.next(
                context=chat_context,
                room=self.room,
                toolkits=toolkits,
                event_handler=push if thread_adapter is not None else None,
                model=model,
            )
        finally:
            if thread_adapter is not None:
                with contextlib.suppress(Exception):
                    await thread_adapter.stop()

    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return []

    async def get_message_toolkits(self, *, message: dict) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                room=self.room,
                caller=self.room.local_participant,
                on_behalf_of=None,
            )
        )

        tool_providers = [*self.get_toolkit_builders()]

        model = message.get("model", self._llm_adapter.default_model())

        message_tools = message.get("tools")

        if message_tools is not None and len(message_tools) > 0:
            toolkits.extend(
                await make_toolkits(
                    room=self.room,
                    model=model,
                    providers=tool_providers,
                    tools=message_tools,
                )
            )
        return [*self._toolkits, *toolkits]

    def prepare_chat_context(self, *, chat_context: AgentSessionContext):
        pass

    async def init_session(self) -> AgentSessionContext:
        context = self._llm_adapter.create_session()
        context.append_rules(self._rules)
        return context

    async def run(self, *, room: RoomClient):
        backoff = 0
        while not self._done:
            try:
                message = await room.queues.receive(
                    name=self._queue, create=True, wait=True
                )

                backoff = 0
                if message is not None:
                    logger.info("received message on worker queue")
                    try:
                        session_context = await self.init_session()
                        async with session_context:
                            session_context.replace_rules(
                                rules=[
                                    *await self.get_rules(),
                                ]
                            )

                            toolkits = await self.get_message_toolkits(message=message)

                            self.prepare_chat_context(chat_context=session_context)

                            await self.process_message(
                                chat_context=session_context,
                                message=message,
                                toolkits=toolkits,
                            )

                    except Exception as e:
                        logger.error(
                            "Failed to process worker message: %s (%s)",
                            e,
                            _summarize_worker_message(message=message),
                            exc_info=e,
                        )

            except Exception as e:
                logger.error(
                    f"Worker error while receiving: {e}, will retry", exc_info=e
                )

                await asyncio.sleep(0.1 * pow(2, backoff))
                backoff = backoff + 1
