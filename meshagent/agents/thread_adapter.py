import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Optional
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

from meshagent.agents.agent import AgentSessionContext
from meshagent.agents.thread_schema import thread_schema

tracer = trace.get_tracer("meshagent.thread_adapter")
logger = logging.getLogger("thread_adapter")

_THREAD_SYNC_CLOSE_TIMEOUT_SEC = 5.0


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
        self._ensure_members_element()
        self._ensure_messages_element()
        self._processor_task = asyncio.create_task(self._process_llm_events())

    async def __aenter__(self) -> "ThreadAdapter":
        await self.start()
        return self

    @property
    def thread(self) -> Optional[MeshDocument]:
        return self._thread

    @property
    def path(self) -> str:
        return self._thread_path

    def format_message(
        self,
        *,
        user_name: str,
        message: str,
        iso_timestamp: str,
    ) -> str:
        return self._format_message(
            user_name=user_name,
            message=message,
            iso_timestamp=iso_timestamp,
        )

    async def stop(self) -> None:
        if self._processor_task is not None and not self._processor_task.done():
            # Give the processor a bounded window to consume pending events
            # before we shut down the queue.
            drain_deadline = asyncio.get_running_loop().time() + 2.0
            while not self._llm_messages.empty():
                if self._processor_task.done():
                    break
                if asyncio.get_running_loop().time() >= drain_deadline:
                    break
                await asyncio.sleep(0.01)

        self._llm_messages.shutdown()
        if self._processor_task is not None:
            await self._processor_task
            self._processor_task = None

        if self._thread is not None:
            final_state: bytes | None = None
            try:
                state = self._thread.get_state()
                if isinstance(state, bytes) and len(state) > 0:
                    final_state = state
            except Exception as ex:
                logger.warning(
                    "unable to collect final thread state for %s",
                    self._thread_path,
                    exc_info=ex,
                )

            if final_state is not None and not self._room.is_closed:
                try:
                    encoded_state = base64.standard_b64encode(final_state)
                    await self._room.sync.sync(
                        path=self._thread_path,
                        data=encoded_state,
                    )
                except Exception as ex:
                    if self._room.is_closed:
                        logger.debug(
                            "skipping final thread state flush for closed room %s",
                            self._thread_path,
                            exc_info=ex,
                        )
                    else:
                        logger.warning(
                            "unable to flush final thread state for %s",
                            self._thread_path,
                            exc_info=ex,
                        )

            if not self._room.is_closed:
                await asyncio.sleep(3)
            # Do not let a stalled close handshake block agent shutdown
            # indefinitely.
            try:
                await asyncio.wait_for(
                    self._room.sync.close(path=self._thread_path),
                    timeout=_THREAD_SYNC_CLOSE_TIMEOUT_SEC,
                )
            except TimeoutError:
                logger.warning(
                    "timed out closing thread sync stream for %s after %.1fs",
                    self._thread_path,
                    _THREAD_SYNC_CLOSE_TIMEOUT_SEC,
                )
            self._thread = None

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        await self.stop()

    def push(self, *, event: dict) -> None:
        try:
            self._llm_messages.put_nowait(event)
        except asyncio.QueueShutDown:
            logger.debug("dropping thread adapter event after queue shutdown")

    async def clear_thread(self) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._ensure_messages_element()

        for child in list(messages.get_children()):
            child.delete()

    @abstractmethod
    async def handle_custom_event(
        self,
        *,
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

    def append_messages(self, *, context: AgentSessionContext) -> None:
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
                                    f"the user attached a file at the path '{child.get_attribute('path')}'"
                                )

                break

        if doc_messages is None:
            doc_messages = self._ensure_messages_element()

    def _ensure_members_element(self) -> Element:
        if self._thread is None:
            raise RoomException("thread was not opened")

        members = self._thread.root.get_children_by_tag_name("members")
        if len(members) > 0:
            return members[0]

        return self._thread.root.append_child(tag_name="members")

    def _ensure_messages_element(self) -> Element:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")
        if len(messages) > 0:
            return messages[0]

        return self._thread.root.append_child(tag_name="messages")

    def ensure_member(self, *, participant: Participant | str) -> None:
        if isinstance(participant, Participant):
            participant_name = participant.get_attribute("name")
        else:
            participant_name = participant

        if not isinstance(participant_name, str):
            return

        normalized_name = participant_name.strip()
        if normalized_name == "":
            return

        members = self._ensure_members_element()
        for member in members.get_children():
            if member.tag_name != "member":
                continue

            if member.get_attribute("name") == normalized_name:
                return

        members.append_child(
            tag_name="member",
            attributes={"name": normalized_name},
        )

    def write_text_message(
        self,
        *,
        text: str,
        participant: Participant | str,
        message_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        role: Optional[str] = None,
    ) -> None:
        self.ensure_member(participant=participant)
        doc_messages = self._ensure_messages_element()

        author_name = ""
        if isinstance(participant, Participant):
            participant_name = participant.get_attribute("name")
            if isinstance(participant_name, str):
                author_name = participant_name
        else:
            author_name = participant

        attributes: dict[str, Any] = {
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "author_name": author_name,
        }

        normalized_role = role.strip() if isinstance(role, str) else ""
        if normalized_role == "" and isinstance(participant, RemoteParticipant):
            normalized_role = participant.role.strip()
        if normalized_role == "" and participant is self._room.local_participant:
            normalized_role = "agent"
        if normalized_role != "":
            attributes["role"] = normalized_role

        if isinstance(message_id, str) and message_id.strip() != "":
            attributes["id"] = message_id.strip()
        if isinstance(turn_id, str) and turn_id.strip() != "":
            attributes["turn_id"] = turn_id.strip()

        message = doc_messages.append_child(
            tag_name="message",
            attributes=attributes,
        )

        if attachments is None:
            return

        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            path = attachment.get("path")
            if not isinstance(path, str):
                continue
            normalized_path = path.strip()
            if normalized_path == "":
                continue
            message.append_child(
                tag_name="file",
                attributes={"path": normalized_path},
            )

    def write_image(
        self,
        *,
        message_id: Optional[str],
        turn_id: Optional[str] = None,
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

            message_attributes: dict[str, Any] = {
                "id": resolved_message_id,
                "text": "",
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": author_name,
                "role": "agent",
            }
            if isinstance(turn_id, str) and turn_id.strip() != "":
                message_attributes["turn_id"] = turn_id.strip()

            message = messages.append_child(
                tag_name="message",
                attributes=message_attributes,
            )
        else:
            message.set_attribute("role", "agent")
            if isinstance(turn_id, str) and turn_id.strip() != "":
                message.set_attribute("turn_id", turn_id.strip())

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
