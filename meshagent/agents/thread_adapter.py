import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Optional
import uuid

from opentelemetry import trace

from meshagent.api import (
    Element,
    MeshDocument,
    Participant,
    RemoteParticipant,
    RoomClient,
    RoomException,
)
from meshagent.tools import Toolkit, tool

from meshagent.agents.agent import AgentChatContext
from meshagent.agents.thread_schema import thread_schema

tracer = trace.get_tracer("meshagent.thread_adapter")


def default_format_message(*, user_name: str, message: str, iso_timestamp: str) -> str:
    return f"{user_name} said at {iso_timestamp}: {message}"


class ThreadAdapter(ABC):
    def __init__(
        self,
        *,
        room: RoomClient,
        path: str,
        format_message: Optional[Callable] = None,
        max_append_message_count: int = 25,
    ):
        self._room = room
        self._thread_path = path
        self._processor_task: Optional[asyncio.Task] = None
        self._llm_messages: asyncio.Queue = asyncio.Queue()
        self._thread: Optional[MeshDocument] = None
        self._format_message = format_message or default_format_message
        self._max_append_message_count = max_append_message_count

    async def start(self) -> None:
        self._thread = await self._room.sync.open(
            path=self._thread_path,
            schema=thread_schema,
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
            # TODO: Wait for pending changes to sync
            await asyncio.sleep(3)
            await self._room.sync.close(path=self._thread_path)
            self._thread = None

    def push(self, *, event: dict) -> None:
        self._llm_messages.put_nowait(event)

    @abstractmethod
    async def handle_custom_event(
        self,
        *,
        messages: Element,
        event: dict,
    ) -> None: ...

    @abstractmethod
    async def _process_llm_events(self) -> None: ...

    def make_toolkit(self) -> Toolkit:
        return Toolkit(
            name="search",
            description="tools for searching conversation history",
            tools=[
                self.grep_tool,
                self.get_message_range,
                self.count_tool,
            ],
        )

    @tool(
        name="get_message_range",
        description="gets a range of messages, index 0 is the first message in the conversation",
    )
    def get_message_range(self, *, start: int, end: int) -> str:
        all_items = self._thread.root.get_children_by_tag_name("messages")[
            0
        ].get_children()
        messages = [
            item
            for item in all_items
            if isinstance(item, Element) and item.tag_name == "message"
        ]

        elements = messages[start:end]

        if len(elements) == 0:
            return "no messages were found within the specified range"

        response = "matching messages:\n"

        for element in elements:
            response = response + self._format_message(
                user_name=element["author_name"],
                message=element["text"],
                iso_timestamp=element["created_at"],
            )

        return response

    @tool(
        name="count_current_thread_messages",
        description="return the number of messages in the current thread (including those outside the context window)",
    )
    def count_tool(
        self,
        *,
        pattern: str,
        ignore_case: bool,
        messages_before: int,
        messages_after: int,
    ) -> str:
        del pattern
        del ignore_case
        del messages_before
        del messages_after

        messages = self._thread.root.get_children_by_tag_name("messages")[0]
        count = len(
            [
                item
                for item in messages.get_children()
                if isinstance(item, Element) and item.tag_name == "message"
            ]
        )
        return f"{count}"

    @tool(
        name="grep_current_thread",
        description="search the current thread for text, includes messages outside the current context window",
    )
    def grep_tool(
        self,
        *,
        pattern: str,
        ignore_case: bool,
        messages_before: int,
        messages_after: int,
    ) -> str:
        messages = self._thread.root.get_children_by_tag_name("messages")[0]
        elements = [
            item
            for item in messages.grep(
                pattern,
                ignore_case=ignore_case,
                before=messages_before,
                after=messages_after,
            )
            if isinstance(item, Element) and item.tag_name == "message"
        ]
        if len(elements) == 0:
            return "no messages were found with the specified pattern"

        response = "matching messages:\n"

        for element in elements:
            response = response + self._format_message(
                user_name=element["author_name"],
                message=element["text"],
                iso_timestamp=element["created_at"],
            )

        return response

    def append_messages(self, *, context: AgentChatContext) -> None:
        doc_messages = None

        for prop in self._thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop

                messages = [
                    element
                    for element in doc_messages.get_children()
                    if isinstance(element, Element) and element.tag_name == "message"
                ]
                if len(messages) > self._max_append_message_count:
                    first_message = len(messages) - self._max_append_message_count
                    messages = messages[first_message:]
                    context.append_assistant_message(
                        f"there are more messages outside the current context window, the index of the first message loaded is {first_message}"
                    )

                for element in messages:
                    if isinstance(element, Element) and element.tag_name == "message":
                        msg = element["text"]
                        if element[
                            "author_name"
                        ] == self._room.local_participant.get_attribute("name"):
                            context.append_assistant_message(msg)
                        else:
                            context.append_user_message(
                                self._format_message(
                                    user_name=element["author_name"],
                                    message=element["text"],
                                    iso_timestamp=element["created_at"],
                                )
                            )

                        for child in element.get_children():
                            if child.tag_name == "file":
                                context.append_assistant_message(
                                    f"the user attached a file with the path '{child.get_attribute('path')}'"
                                )

                break

        if doc_messages is None:
            raise Exception("thread was not properly initialized")

    def write_text_message(
        self,
        *,
        text: str,
        participant: RemoteParticipant | str,
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
                if isinstance(participant, Participant)
                else participant,
            },
        )

    def write_image(
        self,
        *,
        message_id: Optional[str],
        image_id: Optional[str] = None,
        mime_type: Optional[str] = None,
        created_at: Optional[str] = None,
        created_by: Optional[str] = None,
        width: Optional[int | float] = None,
        height: Optional[int | float] = None,
        status: Optional[str] = None,
        status_detail: Optional[str] = None,
    ) -> str:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages: Element = self._thread.root.get_children_by_tag_name("messages")[0]

        resolved_message_id = (
            message_id
            if isinstance(message_id, str) and message_id.strip() != ""
            else str(uuid.uuid4())
        )

        message = None
        for child in messages.get_children():
            if (
                child.tag_name == "message"
                and child.get_attribute("id") == resolved_message_id
            ):
                message = child
                break

        if message is None:
            author_name = (
                created_by
                if isinstance(created_by, str) and created_by.strip() != ""
                else self._room.local_participant.get_attribute("name")
            )
            if not isinstance(author_name, str):
                author_name = ""

            message = messages.append_child(
                tag_name="message",
                attributes={
                    "id": resolved_message_id,
                    "text": "",
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "author_name": author_name,
                },
            )

        image = None
        for child in message.get_children():
            if child.tag_name == "image":
                image = child
                break

        normalized_width: Optional[int] = None
        if isinstance(width, (int, float)):
            width_int = int(width)
            if width_int > 0:
                normalized_width = width_int

        normalized_height: Optional[int] = None
        if isinstance(height, (int, float)):
            height_int = int(height)
            if height_int > 0:
                normalized_height = height_int

        image_attributes: dict[str, str | int] = {}
        if isinstance(image_id, str) and image_id.strip() != "":
            image_attributes["id"] = image_id
        if isinstance(mime_type, str) and mime_type.strip() != "":
            image_attributes["mime_type"] = mime_type
        if isinstance(created_at, str) and created_at.strip() != "":
            image_attributes["created_at"] = created_at
        if isinstance(created_by, str) and created_by.strip() != "":
            image_attributes["created_by"] = created_by
        if normalized_width is not None:
            image_attributes["width"] = normalized_width
        if normalized_height is not None:
            image_attributes["height"] = normalized_height
        if isinstance(status, str) and status.strip() != "":
            image_attributes["status"] = status.strip()
        if isinstance(status_detail, str) and status_detail.strip() != "":
            image_attributes["status_detail"] = status_detail.strip()

        if image is None:
            message.append_child(tag_name="image", attributes=image_attributes)
            return resolved_message_id

        for key, value in image_attributes.items():
            image.set_attribute(key, value)

        return resolved_message_id


# Backwards-compatible import path for existing callers.
def response_event_to_agent_event(event: dict) -> Optional[dict]:
    from .responses_thread_adapter import response_event_to_agent_event as _convert

    return _convert(event)
