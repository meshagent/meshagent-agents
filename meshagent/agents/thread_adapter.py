import asyncio
import base64
import logging
import shlex
import uuid
from datetime import datetime, timezone
from typing import Optional, Callable
from meshagent.api import RemoteParticipant

from opentelemetry import trace

from meshagent.agents.agent import AgentChatContext
from meshagent.agents.thread_schema import thread_schema
from meshagent.api import (
    RoomClient,
    Element,
    MeshDocument,
    RoomException,
)

tracer = trace.get_tracer("meshagent.thread_adapter")

logger = logging.getLogger("thread_adapter")


class ThreadAdapter:
    def __init__(self, *, room: RoomClient, path: str):
        self._room = room
        self._thread_path = path
        self._processor_task: Optional[asyncio.Task] = None
        self._llm_messages: asyncio.Queue = asyncio.Queue()
        self._thread: Optional[MeshDocument] = None

    async def start(self) -> None:
        self._thread = await self._room.sync.open(
            path=self._thread_path, schema=thread_schema
        )

        self._processor_task = asyncio.create_task(self._process_llm_events())

    @property
    def thread(self) -> Optional[MeshDocument]:
        return self._thread

    async def stop(self) -> None:
        self._llm_messages.shutdown()
        if self._processor_task is not None:
            await self._processor_task
            self._processor_task = None

        if self._thread is not None:
            await self._room.sync.close(path=self._thread_path)
            self._thread = None

    def push(self, *, event: dict) -> None:
        self._llm_messages.put_nowait(event)

    def append_messages(
        self,
        *,
        context: AgentChatContext,
        format_message: Optional[Callable[..., str]] = None,
    ) -> None:
        doc_messages = None

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop

                for element in doc_messages.get_children():
                    if isinstance(element, Element) and element.tag_name == "message":
                        msg = element["text"]
                        if element[
                            "author_name"
                        ] == self._room.local_participant.get_attribute("name"):
                            context.append_assistant_message(msg)
                        else:
                            if format_message is not None:
                                context.append_user_message(
                                    format_message(
                                        user_name=element["author_name"],
                                        message=msg,
                                        iso_timestamp=element["created_at"],
                                    )
                                )
                            else:
                                context.append_user_message(msg)

                        for child in element.get_children():
                            if child.tag_name == "file":
                                context.append_assistant_message(
                                    f"the user attached a file with the path '{child.get_attribute('path')}'"
                                )

                break

        if doc_messages is None:
            raise Exception("thread was not properly initialized")

    def write_text_message(
        self, *, text: str, participant: RemoteParticipant | str
    ) -> None:
        doc_messages: Element = self._thread.root.get_children_by_tag_name("messages")[
            0
        ]

        doc_messages.append_child(
            tag_name="message",
            attributes={
                "text": text,
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": participant.get_attribute("name")
                if participant is RemoteParticipant
                else participant,
            },
        )

    async def _process_llm_events(
        self,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        doc_messages = None
        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop
                break

        if doc_messages is None:
            raise RoomException("messages element is missing from thread document")

        updates = asyncio.Queue()
        content_element = None
        partial = ""

        # throttle updates so we don't send too many syncs over the wire at once
        async def update_thread():
            changes = {}
            try:
                while True:
                    try:
                        element, partial_text = updates.get_nowait()
                        changes[element] = partial_text

                    except asyncio.QueueEmpty:
                        for element, partial_text in changes.items():
                            element["text"] = partial_text

                        changes.clear()

                        element, partial_text = await updates.get()
                        changes[element] = partial_text

                        # await asyncio.sleep(0.1)

            except asyncio.QueueShutDown:
                # flush any pending changes
                for element, partial_text in changes.items():
                    element["text"] = partial_text

                changes.clear()
                pass

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await self._llm_messages.get()

                if evt["type"] == "response.content_part.added":
                    partial = ""

                    content_element = doc_messages.append_child(
                        tag_name="message",
                        attributes={
                            "text": "",
                            "created_at": datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "author_name": self._room.local_participant.get_attribute(
                                "name"
                            ),
                        },
                    )

                elif evt["type"] == "response.output_text.delta":
                    partial += evt["delta"]
                    updates.put_nowait((content_element, partial))

                elif evt["type"] == "response.output_text.done":
                    content_element = None
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attribute(
                            "from_participant_name",
                            self._room.local_participant.get_attribute("name"),
                        )
                        span.set_attribute("role", "assistant")
                        span.set_attributes({"text": evt["text"]})

                elif evt["type"] == "response.image_generation_call.partial_image":
                    await self.handle_image_generation_partial(event=evt)

                elif evt["type"] == "meshagent.handler.added":
                    item = evt["item"]
                    if item["type"] == "shell_call":
                        await self.handle_shell_call_output(item=item)

                    elif item["type"] == "local_shell_call":
                        await self.handle_local_shell_call_output(item=item)

        except asyncio.QueueShutDown:
            pass
        finally:
            updates.shutdown()

        await update_thread_task

    async def handle_image_generation_partial(
        self,
        *,
        event: dict,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        item_id = event["item_id"]
        partial_image_b64 = event["partial_image_b64"]
        output_format = event["output_format"]

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        if output_format is None:
            output_format = "png"

        image_name = f"{str(uuid.uuid4())}.{output_format}"

        handle = await self._room.storage.open(path=image_name)
        await self._room.storage.write(
            handle=handle, data=base64.b64decode(partial_image_b64)
        )
        await self._room.storage.close(handle=handle)

        messages = None

        logger.info(f"A partial was saved at the path {image_name}")

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        for child in messages.get_children():
            if child.get_attribute("id") == item_id:
                for file in child.get_children():
                    file.set_attribute("path", image_name)

                return

        message_element = messages.append_child(
            tag_name="message",
            attributes={
                "id": item_id,
                "text": "",
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": self._room.local_participant.get_attribute("name"),
            },
        )
        message_element.append_child(tag_name="file", attributes={"path": image_name})

    async def handle_local_shell_call_output(
        self,
        *,
        item: dict,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        command = action["command"]
        working_directory = action["working_directory"]

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        exec_element = messages.append_child(
            tag_name="exec",
            attributes={"command": shlex.join(command), "pwd": working_directory},
        )

        evt = await self._llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = evt.get("error")
        item = evt.get("item")

        if error is not None:
            pass

        if item is not None:
            if item["type"] != "local_shell_call_output":
                raise RoomException("expected local_shell_call_output")

            exec_element.set_attribute("result", item["output"])

    async def handle_shell_call_output(
        self,
        *,
        item: dict,
    ):
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        commands = action["commands"]

        exec_elements = []
        for command in commands:
            exec_element = messages.append_child(
                tag_name="exec",
                attributes={"command": command},
            )
            exec_elements.append(exec_element)

        evt = await self._llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = evt.get("error")
        item = evt.get("item")

        if error is not None:
            pass

        if item is not None:
            if item["type"] != "shell_call_output":
                raise RoomException("expected shell_call_output")

            results = item["output"]

            for i in range(0, len(results)):
                result = results[i]
                exec_element = exec_elements[i]
                if "exit_code" in result["outcome"]:
                    exec_element.set_attribute(
                        "exit_code", result["outcome"]["exit_code"]
                    )

                exec_element.set_attribute("outcome", result["outcome"]["type"])
                exec_element.set_attribute("stdout", result["stdout"])
                exec_element.set_attribute("stderr", result["stderr"])
