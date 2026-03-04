import base64
import logging
import posixpath
import re
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from meshagent.api.schema_util import merge
from meshagent.api import Element, MeshDocument
from meshagent.tools import Toolkit

from .adapter import LLMAdapter
from .context import TaskContext
from .task_runner import TaskRunner
from .thread_schema import thread_list_schema
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

    async def _next_available_thread_path(
        self,
        *,
        context: TaskContext,
        base_path: str,
    ) -> str:
        try:
            exists = await context.room.storage.exists(path=base_path)
        except Exception:
            return base_path

        if not exists:
            return base_path

        thread_dir, filename = posixpath.split(base_path)
        if filename.endswith(".thread"):
            base_name = filename[: -len(".thread")]
        else:
            base_name = filename

        for index in range(2, 1000):
            candidate = posixpath.join(thread_dir, f"{base_name} {index}.thread")
            try:
                if not await context.room.storage.exists(path=candidate):
                    return candidate
            except Exception:
                return candidate

        return posixpath.join(thread_dir, f"{base_name}-{uuid.uuid4().hex[:8]}.thread")

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _thread_list_index_path(self) -> str:
        return posixpath.join(self.thread_dir, "index.threadl")

    def _thread_list_entry_name_for_path(self, *, path: str) -> str:
        filename = posixpath.basename(path.strip())
        if filename.endswith(".thread"):
            filename = filename[: -len(".thread")]

        normalized = re.sub(r"[-_/]+", " ", filename)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")
        if normalized == "":
            return "Thread"
        if normalized == normalized.lower() or normalized == normalized.upper():
            normalized = normalized.title()
        return normalized[:64].strip() or "Thread"

    def _find_thread_list_entry(
        self,
        *,
        document: MeshDocument,
        path: str,
    ) -> Optional[Element]:
        for child in document.root.get_children():
            if not isinstance(child, Element):
                continue
            if child.tag_name != "thread":
                continue
            if child.get_attribute("path") == path:
                return child
        return None

    def _upsert_thread_list_entry(
        self,
        *,
        document: MeshDocument,
        path: str,
        name: Optional[str] = None,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
    ) -> None:
        now = self._utc_now_iso()
        provided_name = name.strip() if isinstance(name, str) else ""

        entry = self._find_thread_list_entry(document=document, path=path)
        if entry is None:
            resolved_name = (
                provided_name
                if provided_name != ""
                else self._thread_list_entry_name_for_path(path=path)
            )
            created_value = (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
            modified_value = (
                modified_at.strip()
                if isinstance(modified_at, str) and modified_at.strip() != ""
                else created_value
            )
            document.root.append_child(
                tag_name="thread",
                attributes={
                    "name": resolved_name,
                    "path": path,
                    "created_at": created_value,
                    "modified_at": modified_value,
                },
            )
            return

        if provided_name != "":
            entry.set_attribute("name", provided_name)
        else:
            existing_name = entry.get_attribute("name")
            if not isinstance(existing_name, str) or existing_name.strip() == "":
                entry.set_attribute(
                    "name",
                    self._thread_list_entry_name_for_path(path=path),
                )

        entry.set_attribute("path", path)

        existing_created_at = entry.get_attribute("created_at")
        created_value = (
            existing_created_at.strip()
            if isinstance(existing_created_at, str)
            and existing_created_at.strip() != ""
            else (
                created_at.strip()
                if isinstance(created_at, str) and created_at.strip() != ""
                else now
            )
        )
        entry.set_attribute("created_at", created_value)

        modified_value = (
            modified_at.strip()
            if isinstance(modified_at, str) and modified_at.strip() != ""
            else now
        )
        entry.set_attribute("modified_at", modified_value)

    async def record_thread_in_index(
        self,
        *,
        context: TaskContext,
        path: str,
    ) -> None:
        if self.threading_mode != "auto":
            return

        normalized_path = path.strip()
        if normalized_path == "":
            return

        index_path = self._thread_list_index_path()
        document = None
        try:
            document = await context.room.sync.open(
                path=index_path,
                schema=thread_list_schema,
            )
            now = self._utc_now_iso()
            self._upsert_thread_list_entry(
                document=document,
                path=normalized_path,
                created_at=now,
                modified_at=now,
            )
            state = document.get_state()
            if isinstance(state, bytes) and len(state) > 0:
                await context.room.sync.sync(
                    path=index_path,
                    data=base64.standard_b64encode(state),
                )
        except Exception as ex:
            logger.warning(
                "unable to update thread list document at %s",
                index_path,
                exc_info=ex,
            )
        finally:
            if document is not None:
                try:
                    await context.room.sync.close(path=index_path)
                except Exception as ex:
                    logger.warning(
                        "unable to close thread list document at %s",
                        index_path,
                        exc_info=ex,
                    )

    def ensure_local_member_on_thread(
        self,
        *,
        context: TaskContext,
        thread_adapter: ThreadAdapter,
    ) -> None:
        thread = thread_adapter.thread
        if thread is None:
            return

        members_elements = thread.root.get_children_by_tag_name("members")
        if len(members_elements) == 0:
            return

        members = members_elements[0]
        local_name = context.room.local_participant.get_attribute("name")
        if not isinstance(local_name, str):
            return

        normalized_local_name = local_name.strip()
        if normalized_local_name == "":
            return

        for child in members.get_children():
            if not isinstance(child, Element):
                continue
            if child.tag_name != "member":
                continue
            member_name = child.get_attribute("name")
            if isinstance(member_name, str) and member_name == normalized_local_name:
                return

        members.append_child(
            tag_name="member",
            attributes={"name": normalized_local_name},
        )

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

        cloned_context = context.session.copy()
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

        base_path = self._thread_path_for_name(thread_name=generated_name)
        return await self._next_available_thread_path(
            context=context,
            base_path=base_path,
        )

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
