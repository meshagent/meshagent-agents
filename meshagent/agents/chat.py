from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from meshagent.agents.agent import SingleRoomAgent, AgentChatContext
from meshagent.api.chan import Chan
from meshagent.api import (
    RoomMessage,
    RoomClient,
    RemoteParticipant,
    Participant,
    Requirement,
    Element,
    MeshDocument,
)
from meshagent.tools import (
    Toolkit,
    ToolContext,
    make_toolkits,
    ToolkitBuilder,
    ToolkitConfig,
    RemoteToolkit,
)
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.openai.tools.responses_adapter import (
    ReasoningTool,
    OpenAIResponsesAdapter,
)
from meshagent.agents.thread_adapter import (
    ThreadAdapter,
    response_event_to_agent_event,
)

import uuid
from datetime import datetime, timezone
import asyncio
from typing import Any, Optional, Callable, AsyncIterator, Awaitable
import logging
from asyncio import CancelledError
from meshagent.api import RoomException
from opentelemetry import trace
import json
from pydantic import BaseModel
from meshagent.tools import tool
from pathlib import Path
from meshagent.agents.skills import to_prompt


tracer = trace.get_tracer("meshagent.chatbot")

logger = logging.getLogger("chat")


class ChatBotReasoningTool(ReasoningTool):
    def __init__(self, *, room: RoomClient, thread_context: "ChatThreadContext"):
        super().__init__()
        self.thread_context = thread_context
        self.room = room

        self._reasoning_element = None
        self._reasoning_item = None

    def _get_messages_element(self):
        messages = self.thread_context.thread.root.get_children_by_tag_name("messages")
        if len(messages) > 0:
            return messages[0]
        return None

    async def on_reasoning_summary_part_added(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        el = self._get_messages_element()
        if el is None:
            logger.warning("missing messages element, cannot log reasoning")
        else:
            self._reasoning_element = el.append_child("reasoning", {"summary": ""})

    async def on_reasoning_summary_part_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        part: dict,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        self._reasoning_element = None

    async def on_reasoning_summary_text_delta(
        self,
        context: ToolContext,
        *,
        delta: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        el = self._reasoning_element
        el.set_attribute("summary", el.get_attribute("summary") + delta)

    async def on_reasoning_summary_text_done(
        self,
        context: ToolContext,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        summary_index: int,
        type: str,
        **extra,
    ):
        pass


def get_online_participants(
    *,
    room: RoomClient,
    thread: MeshDocument,
    exclude: Optional[list[Participant]] = None,
) -> list[RemoteParticipant]:
    results = list[RemoteParticipant]()

    for prop in thread.root.get_children():
        if prop.tag_name == "members":
            for member in prop.get_children():
                for online in room.messaging.get_participants():
                    if online.get_attribute("name") == member.get_attribute("name"):
                        if exclude is None or online not in exclude:
                            results.append(online)

    return results


class ChatThreadContext:
    def __init__(
        self,
        *,
        chat: AgentChatContext,
        thread: MeshDocument,
        path: str,
        participants: Optional[list[RemoteParticipant]] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
    ):
        self.thread = thread
        if participants is None:
            participants = []

        self.participants = participants
        self.chat = chat
        self.path = path
        self._event_handler = event_handler

    def emit(self, event: dict):
        if self._event_handler is not None:
            self._event_handler(event)

    @property
    def context_id(self) -> str:
        return self.chat.id

    def to_caller_context(self) -> dict:
        return {"chat": self.chat.to_json()}


class ChatBotClient:
    def __init__(
        self,
        *,
        room: RoomClient,
        participant_name: str,
        thread_path: str,
        timeout: float = 30,
    ):
        self.room = room
        self.participant_name = participant_name
        self.thread_path = thread_path
        self._messages: asyncio.Queue[str] = asyncio.Queue()
        self._doc = None
        self._participant: Optional[RemoteParticipant] = None
        self._timeout = timeout

    async def start(self) -> None:
        self._doc = await self.room.sync.open(path=self.thread_path)
        self.room.messaging.on("message", self._on_message)
        await asyncio.sleep(1)
        await self._wait_for_participant()
        if self._participant is not None:
            await self.room.messaging.send_message(
                to=self._participant,
                type="opened",
                message={"path": self.thread_path},
                attachment=None,
            )

    async def stop(self) -> None:
        await self.room.sync.close(path=self.thread_path)
        self.room.messaging.off("message", self._on_message)

    async def __aenter__(self) -> "ChatBotClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        await self.stop()

    def _on_message(self, message: RoomMessage) -> None:
        received_type = message.type
        received_message = message.message

        if received_type == "chat" and received_message["path"] == self.thread_path:
            self._messages.put_nowait(received_message["text"])

    async def _wait_for_participant(self) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                while self._participant is None:
                    for participant in self.room.messaging.get_participants():
                        if participant.get_attribute("name") == self.participant_name:
                            self._participant = participant
                            break

                    await asyncio.sleep(1)
        except asyncio.TimeoutError as exc:
            raise RoomException(
                f"timed out waiting for {self.participant_name}"
            ) from exc

    async def clear(self) -> None:
        if self._participant is None:
            return

        await self.room.messaging.send_message(
            to=self._participant,
            type="clear",
            message={"path": self.thread_path},
            attachment=None,
        )

    async def send(
        self,
        *,
        text: str,
        tools: Optional[list[ToolkitConfig]] = None,
        attachments: Optional[list[dict]] = None,
    ) -> None:
        if self._participant is None or self._doc is None:
            raise RoomException("chat client not started")

        messages = self._doc.root.get_elements_by_tag_name("messages")[0]
        messages.append_child(
            tag_name="message",
            attributes={
                "id": str(uuid.uuid4()),
                "text": text,
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": self.room.local_participant.get_attribute("name"),
            },
        )

        if attachments is not None:
            for attachment in attachments:
                messages.append_child(tag_name="file", attributes=attachment)

        tool_payload = [tool.model_dump(mode="json") for tool in tools or []]
        await self.room.messaging.send_message(
            to=self._participant,
            type="chat",
            message={
                "text": text,
                "path": self.thread_path,
                "tools": tool_payload,
            },
            attachment=None,
        )

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._messages.get()

    async def receive(self) -> str:
        return await self._messages.get()

    async def receive_nowait(self) -> str:
        return await self._messages.get_nowait()


@dataclass
class _QueuedChatMessage:
    type: str
    message: dict
    from_participant: RemoteParticipant
    result: asyncio.Future = field(default_factory=asyncio.Future)


class ChatBotBase(SingleRoomAgent, ABC):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        always_reply: Optional[bool] = None,
        skill_dirs: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            annotations=annotations,
        )

        if toolkits is None:
            toolkits = []

        if always_reply is None:
            always_reply = False

        self._always_reply = always_reply

        self._message_channels = dict[str, Chan[_QueuedChatMessage]]()

        self._room: RoomClient | None = None
        self._toolkits = toolkits
        self._client_rules = client_rules

        if rules is None:
            rules = []

        self._rules = rules
        self._is_typing = dict[str, asyncio.Task]()
        self._auto_greet_message = auto_greet_message

        if empty_state_title is None:
            empty_state_title = "How can I help you?"
        self._empty_state_title = empty_state_title

        self._thread_tasks = dict[str, asyncio.Task]()
        self._open_threads = {}
        self._thread_contexts: dict[str, ChatThreadContext] = {}

        self._skill_dirs = skill_dirs

    async def _clear_thread_status(self, *, path: str) -> None:
        del path

    def _clear_thread_status_nowait(self, *, path: str) -> None:
        del path

    def _update_thread_status_from_event(self, *, path: str, event: dict) -> None:
        del path
        del event

    async def _clear_all_thread_statuses(self) -> None:
        pass

    async def _send_and_save_chat(
        self,
        thread: MeshDocument,
        path: str,
        to: RemoteParticipant,
        id: str,
        text: str,
        thread_attributes: dict,
    ):
        messages = None

        for prop in thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        if messages is None:
            raise RoomException("messages element was not found in thread document")

        with tracer.start_as_current_span("chatbot.thread.message") as span:
            span.set_attributes(thread_attributes)
            span.set_attribute("role", "assistant")
            span.set_attribute(
                "from_participant_name",
                self.room.local_participant.get_attribute("name"),
            )
            span.set_attributes({"id": id, "text": text})
            await self.room.messaging.send_message(
                to=to, type="chat", message={"path": path, "text": text}
            )

            messages.append_child(
                tag_name="message",
                attributes={
                    "id": id,
                    "text": text,
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "author_name": self.room.local_participant.get_attribute("name"),
                },
            )

    async def _greet(
        self,
        *,
        thread: MeshDocument,
        path: str,
        thread_context: ChatThreadContext,
        participant: RemoteParticipant,
        thread_attributes: dict,
    ):
        if self._auto_greet_message is not None:
            thread_context.chat.append_user_message(self._auto_greet_message)
            await self._send_and_save_chat(
                id=str(uuid.uuid4()),
                to=RemoteParticipant(id=participant.id),
                thread=thread,
                path=path,
                text=self._auto_greet_message,
                thread_attributes=thread_attributes,
            )

    def get_requirements(self):
        return [
            *super().get_requirements(),
        ]

    async def get_online_participants(
        self, *, thread: MeshDocument, exclude: Optional[list[Participant]] = None
    ):
        return get_online_participants(room=self._room, thread=thread, exclude=exclude)

    def get_toolkit_builders(self) -> list[ToolkitBuilder]:
        return []

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                room=self.room,
                caller=self.room.local_participant,
                on_behalf_of=participant,
                caller_context=thread_context.to_caller_context(),
                event_handler=thread_context.emit,
            )
        )

        @tool(
            name="attach_file",
            description="attach a file to the thread so the user can see it",
        )
        async def attach_file(path: str):
            messages = thread_context.thread.root.get_elements_by_tag_name("messages")[
                0
            ]
            message = messages.append_child(
                tag_name="message",
                attributes={
                    "id": str(uuid.uuid4()),
                    "text": "",
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "author_name": self.room.local_participant.get_attribute("name"),
                },
            )
            message.append_child(tag_name="file", attributes={"path": path})

        toolkits.append(
            Toolkit(
                name="thread tools",
                tools=[attach_file],
            )
        )

        toolkit = self._open_threads[thread_context.path].make_toolkit()

        return [*self._toolkits, *toolkits, toolkit]

    @abstractmethod
    def default_model(self) -> str: ...

    @abstractmethod
    async def create_thread_context(
        self,
        *,
        path: str,
        thread: MeshDocument,
        participants: list[RemoteParticipant],
        event_handler: Callable[[dict], None],
    ) -> ChatThreadContext: ...

    def create_thread_adapter(self, *, path: str) -> ThreadAdapter:
        return ThreadAdapter(
            room=self.room,
            path=path,
            format_message=self.format_message,
        )

    async def open_thread(self, *, path: str) -> ThreadAdapter:
        logger.info(f"opening thread {path}")
        if path not in self._open_threads:
            adapter = self.create_thread_adapter(path=path)
            await adapter.start()
            self._open_threads[path] = adapter

        return self._open_threads[path]

    async def close_thread(self, *, path: str) -> None:
        logger.info(f"closing thread {path}")

        adapter = self._open_threads.pop(path, None)
        if adapter is not None:
            await adapter.stop()

    async def on_thread_open(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_clear(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_cancel(self, *, thread_context: ChatThreadContext):
        pass

    async def on_thread_close(self, *, thread_context: ChatThreadContext):
        pass

    async def on_approved(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        pass

    async def on_rejected(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        pass

    async def _safe_invoke_thread_event(
        self,
        *,
        event_name: str,
        thread_context: ChatThreadContext,
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        try:
            await handler(thread_context=thread_context)
        except Exception as e:
            logger.error(
                f"chatbot thread event hook '{event_name}' failed",
                exc_info=e,
            )

    async def _safe_invoke_chat_event(
        self,
        *,
        event_name: str,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict[str, Any],
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        try:
            await handler(
                thread_context=thread_context,
                from_participant=from_participant,
                message=message,
            )
        except Exception as e:
            logger.error(
                f"chatbot chat event hook '{event_name}' failed",
                exc_info=e,
            )

    def get_thread_members(self, *, thread: MeshDocument) -> list[str]:
        results = []

        for prop in thread.root.get_children():
            if prop.tag_name == "members":
                for member in prop.get_children():
                    results.append(member.get_attribute("name"))

        return results

    async def get_rules(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ):
        rules = [*self._rules]

        if self._skill_dirs is not None and len(self._skill_dirs) > 0:
            rules.append(
                "You have access to to following skills which follow the agentskills spec:"
            )
            rules.append(await to_prompt([*(Path(p) for p in self._skill_dirs)]))
            rules.append(
                "Use the shell or storage tool to find out more about skills and execute them when they are required"
            )

        client = participant.get_attribute("client")

        if self._client_rules is not None and client is not None:
            cr = self._client_rules.get(client)
            if cr is not None:
                rules.extend(cr)

        # Without this rule 5.2 / 5.1 like to start their messages with things like "I could say"
        rules.append("based on the previous transcript, take your turn and respond")

        return rules

    async def on_chat_received(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> Optional[str]:
        return None

    def format_message(self, *, user_name: str, message: str, iso_timestamp: str):
        return f"{user_name} said at {iso_timestamp}: {message}"

    async def _spawn_thread(self, path: str, messages: Chan[RoomMessage]):
        logger.debug("chatbot is starting a thread", extra={"path": path})
        opened = False

        current_file = None
        thread_context = None

        thread_attributes = None

        thread = None
        thread_adapter = None

        def handle_event(evt, participant: RemoteParticipant):
            if isinstance(evt, BaseModel):
                evt = evt.model_dump(mode="json")

            if evt.get("type") == "response.output_text.done" and thread is not None:
                online = get_online_participants(room=self._room, thread=thread)
                for online_participant in online:
                    if online_participant.id != self._room.local_participant.id:
                        logger.info(
                            f"replying to {online_participant.get_attribute('name')}"
                        )
                        self._room.messaging.send_message_nowait(
                            to=online_participant,
                            type="chat",
                            message={
                                "type": "chat",
                                "path": path,
                                "text": evt.get("text"),
                            },
                        )

                if participant not in online:
                    logger.info(f"replying to {participant.get_attribute('name')}")
                    self._room.messaging.send_message_nowait(
                        to=participant,
                        type="chat",
                        message={
                            "type": "chat",
                            "path": path,
                            "text": evt.get("text"),
                        },
                    )

            if thread_adapter is None:
                return

            thread_adapter.push(event=evt)
            self._update_thread_status_from_event(path=path, event=evt)

        try:
            received = None

            while True:
                logger.debug(f"waiting for message on thread {path}")
                received = await messages.recv()

                logger.debug(f"received message on thread {path}: {received.type}")

                chat_with_participant = received.from_participant

                thread_attributes = {
                    "agent_name": self.name,
                    "agent_participant_id": self.room.local_participant.id,
                    "agent_participant_name": self.room.local_participant.get_attribute(
                        "name"
                    ),
                    "remote_participant_id": chat_with_participant.id,
                    "remote_participant_name": chat_with_participant.get_attribute(
                        "name"
                    ),
                    "path": path,
                }

                if current_file != chat_with_participant.get_attribute("current_file"):
                    logger.info(
                        f"{chat_with_participant.get_attribute('name')} is now looking at {chat_with_participant.get_attribute('current_file')}"
                    )
                    current_file = chat_with_participant.get_attribute("current_file")

                if thread is None:
                    with tracer.start_as_current_span("chatbot.thread.open") as span:
                        span.set_attributes(thread_attributes)

                        thread_adapter = await self.open_thread(path=path)
                        thread = thread_adapter.thread
                        if thread is None:
                            raise RoomException("thread was not opened")

                        thread_context = await self.create_thread_context(
                            path=path,
                            thread=thread,
                            participants=get_online_participants(
                                room=self.room, thread=thread
                            ),
                            event_handler=lambda evt: handle_event(
                                evt, chat_with_participant
                            ),
                        )
                        self._thread_contexts[path] = thread_context

                        thread_adapter.append_messages(
                            context=thread_context.chat,
                        )

                if received.type == "opened":
                    if not opened:
                        opened = True

                        if thread_context is not None:
                            await self._safe_invoke_thread_event(
                                event_name="open",
                                thread_context=thread_context,
                                handler=self.on_thread_open,
                            )

                        await self._greet(
                            path=path,
                            thread_context=thread_context,
                            participant=chat_with_participant,
                            thread=thread,
                            thread_attributes=thread_attributes,
                        )
                elif received.type == "clear":
                    thread_context = await self.create_thread_context(
                        path=path,
                        thread=thread,
                        participants=get_online_participants(
                            room=self.room, thread=thread
                        ),
                        event_handler=lambda evt: handle_event(
                            evt, chat_with_participant
                        ),
                    )
                    self._thread_contexts[path] = thread_context
                    messages_element: Element = thread.root.get_children_by_tag_name(
                        "messages"
                    )[0]
                    for child in list(messages_element.get_children()):
                        child.delete()

                    if thread_context is not None:
                        await self._safe_invoke_thread_event(
                            event_name="clear",
                            thread_context=thread_context,
                            handler=self.on_thread_clear,
                        )

                elif received.type == "chat":
                    if thread is None:
                        logger.info("thread is not open", extra={"path": path})
                        break

                    logger.debug(
                        "chatbot received a chat",
                        extra={
                            "context": (
                                thread_context.context_id
                                if thread_context is not None
                                else None
                            ),
                            "participant_id": self.room.local_participant.id,
                            "participant_name": self.room.local_participant.get_attribute(
                                "name"
                            ),
                            "text": received.message["text"],
                        },
                    )

                if received is not None and received.type == "chat":
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attributes(thread_attributes)
                        span.set_attribute("role", "user")
                        span.set_attribute(
                            "from_participant_name",
                            chat_with_participant.get_attribute("name"),
                        )

                        attachments = received.message.get("attachments", [])
                        span.set_attribute("attachments", json.dumps(attachments))

                        text = received.message["text"]
                        span.set_attributes({"text": text})

                        try:
                            if thread_context is None:
                                thread_context = await self.create_thread_context(
                                    path=path,
                                    thread=thread,
                                    participants=get_online_participants(
                                        room=self.room, thread=thread
                                    ),
                                    event_handler=lambda evt: handle_event(
                                        evt, chat_with_participant
                                    ),
                                )
                                self._thread_contexts[path] = thread_context
                            else:
                                thread_context.participants = get_online_participants(
                                    room=self.room, thread=thread
                                )

                            for participant in get_online_participants(
                                room=self._room, thread=thread
                            ):
                                self._room.messaging.send_message_nowait(
                                    to=participant,
                                    type="thinking",
                                    message={"thinking": True, "path": path},
                                )

                            result = await self.on_chat_received(
                                thread_context=thread_context,
                                from_participant=chat_with_participant,
                                message=received.message,
                            )
                            received.result.set_result(result)

                        except Exception as e:
                            logger.error("An error was encountered", exc_info=e)
                            text = (
                                "An unexpected error occured. Please try again later."
                            )
                            await self._send_and_save_chat(
                                thread=thread,
                                to=chat_with_participant,
                                path=path,
                                id=str(uuid.uuid4()),
                                text=text,
                                thread_attributes=thread_attributes,
                            )
                            received.result.set_result(text)

                        finally:

                            async def cleanup():
                                for participant in get_online_participants(
                                    room=self._room, thread=thread
                                ):
                                    self._room.messaging.send_message_nowait(
                                        to=participant,
                                        type="thinking",
                                        message={"thinking": False, "path": path},
                                    )

                            asyncio.shield(cleanup())

        finally:

            async def cleanup():
                if self.room is not None:
                    logger.info(f"thread was ended {path}")
                    logger.info("chatbot thread ended", extra={"path": path})

                    thread_context = self._thread_contexts.pop(path, None)
                    if thread_context is not None:
                        await self._safe_invoke_thread_event(
                            event_name="close",
                            thread_context=thread_context,
                            handler=self.on_thread_close,
                        )

                    await self.close_thread(path=path)

            asyncio.shield(cleanup())

    def _get_message_channel(self, key: str) -> Chan[_QueuedChatMessage]:
        if key not in self._message_channels:
            chan = Chan[_QueuedChatMessage]()
            self._message_channels[key] = chan

        chan = self._message_channels[key]

        return chan

    async def stop(self):
        await super().stop()

        await self._clear_all_thread_statuses()

        for thread in self._thread_tasks.values():
            thread.cancel()

        self._thread_tasks.clear()

    async def _on_get_thread_toolkits_message(self, *, message: RoomMessage):
        path = message.message["path"]

        chat_with_participant = None
        for participant in self._room.messaging.get_participants():
            if participant.id == message.from_participant_id:
                chat_with_participant = participant
                break

        if chat_with_participant is None:
            logger.warning(
                "participant does not have messaging enabled, skipping message"
            )
            return

        tool_providers = self.get_toolkit_builders()
        self._room.messaging.send_message_nowait(
            to=chat_with_participant,
            type="set_thread_tool_providers",
            message={
                "path": path,
                "tool_providers": [{"name": t.name} for t in tool_providers],
            },
        )

    async def get_exposed_toolkits(self) -> list[RemoteToolkit]:
        exposed_toolkits = await super().get_exposed_toolkits()

        @tool(
            description=f"sends a chat to {self.room.local_participant.get_attribute('name')} and gets the response"
        )
        async def ask(context: ToolContext, *, path: str, text: str) -> str:
            qm = _QueuedChatMessage(
                type="chat",
                message={
                    "text": text,
                    "path": path,
                },
                from_participant=context.on_behalf_of or context.caller,
            )

            thread = await self.open_thread(path=path)

            thread.write_text_message(
                text=text, participant=context.on_behalf_of or context.caller
            )

            messages = self._ensure_thread(path=path)
            messages.send_nowait(qm)

            return await qm.result

        chatbot_toolkit = RemoteToolkit(
            name=f"{self.name}",
            description=f"tools for interacting with {self.name}",
            public=False,
            tools=[ask],
        )

        exposed_toolkits.append(chatbot_toolkit)
        return exposed_toolkits

    def _ensure_thread(self, path: str) -> Chan[_QueuedChatMessage]:
        messages = self._get_message_channel(path)
        if path not in self._thread_tasks or self._thread_tasks[path].done():

            def thread_done(task: asyncio.Task):
                self._thread_tasks.pop(path)
                self._message_channels.pop(path)
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"The chat thread ended with an error {e}", exc_info=e)

            logger.debug(f"spawning chat thread for {path}")
            task = asyncio.create_task(self._spawn_thread(messages=messages, path=path))
            task.add_done_callback(thread_done)

            self._thread_tasks[path] = task

        return messages

    def on_message(
        self, message: RoomMessage, from_participant: Optional[RemoteParticipant] = None
    ) -> _QueuedChatMessage | None:
        if message.type == "get_thread_toolkit_builders":
            task = asyncio.create_task(
                self._on_get_thread_toolkits_message(message=message)
            )

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except Exception as ex:
                    logger.error(f"unable to get tool providers {ex}", exc_info=ex)

            task.add_done_callback(on_done)

        if (
            message.type == "chat"
            or message.type == "opened"
            or message.type == "clear"
        ):
            path = message.message["path"]

            logger.debug(f"queued incoming message for thread {path}: {message.type}")

            if from_participant is None:
                for participant in self._room.messaging.get_participants():
                    if participant.id == message.from_participant_id:
                        from_participant = participant
                        break

                if from_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    return

            msg = _QueuedChatMessage(
                type=message.type,
                message=message.message,
                from_participant=from_participant,
            )

            messages = self._ensure_thread(path=path)

            messages.send_nowait(msg)

            return msg

        elif message.type == "approved" or message.type == "rejected":
            path = message.message["path"]

            if from_participant is None:
                for participant in self._room.messaging.get_participants():
                    if participant.id == message.from_participant_id:
                        from_participant = participant
                        break

                if from_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    return

            async def handle_approval_response():
                thread_context = self._thread_contexts.get(path)
                if thread_context is None:
                    logger.warning(
                        f"unable to process {message.type} for thread {path}: thread is not open"
                    )
                    return

                if message.type == "approved":
                    await self._safe_invoke_chat_event(
                        event_name="approved",
                        thread_context=thread_context,
                        from_participant=from_participant,
                        message=message.message,
                        handler=self.on_approved,
                    )
                else:
                    await self._safe_invoke_chat_event(
                        event_name="rejected",
                        thread_context=thread_context,
                        from_participant=from_participant,
                        message=message.message,
                        handler=self.on_rejected,
                    )

            task = asyncio.create_task(handle_approval_response())

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as ex:
                    logger.error(
                        f"unable to process {message.type} message for thread {path}",
                        exc_info=ex,
                    )

            task.add_done_callback(on_done)

        elif message.type == "cancel":
            path = message.message["path"]

            async def handle_cancel():
                thread_context = self._thread_contexts.get(path)
                if thread_context is not None:
                    await self._safe_invoke_thread_event(
                        event_name="cancel",
                        thread_context=thread_context,
                        handler=self.on_thread_cancel,
                    )

                if path in self._thread_tasks:
                    self._thread_tasks[path].cancel()

            task = asyncio.create_task(handle_cancel())

            def on_done(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception as ex:
                    logger.error(
                        f"unable to process cancel message for thread {path}",
                        exc_info=ex,
                    )

            task.add_done_callback(on_done)

        elif message.type == "typing":

            def callback(task: asyncio.Task):
                try:
                    task.result()
                except CancelledError:
                    pass
                except Exception:
                    pass

            async def remove_timeout(id: str):
                await asyncio.sleep(1)
                self._is_typing.pop(id)

            if message.from_participant_id in self._is_typing:
                self._is_typing[message.from_participant_id].cancel()

            timeout = asyncio.create_task(
                remove_timeout(id=message.from_participant_id)
            )
            timeout.add_done_callback(callback)

            self._is_typing[message.from_participant_id] = timeout

        return None

    async def start(self, *, room):
        await super().start(room=room)

        logger.debug("Starting chatbot")

        await self.room.local_participant.set_attribute(
            "empty_state_title", self._empty_state_title
        )

        room.messaging.on("message", self.on_message)

        if self._auto_greet_message is not None:

            def on_participant_added(participant: RemoteParticipant):
                # will spawn the initial thread
                self._get_message_channel(participant.id)

            room.messaging.on("participant_added", on_participant_added)

        logger.debug("Enabling chatbot messaging")
        await room.messaging.enable()


class ChatBot(ChatBotBase):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires: Optional[list[Requirement]] = None,
        llm_adapter: LLMAdapter,
        tool_adapter: Optional[ToolResponseAdapter] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        decision_model: Optional[str] = None,
        always_reply: Optional[bool] = None,
        skill_dirs: Optional[list[str]] = None,
    ):
        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter
        self._decision_model = (
            "gpt-4.1-mini" if decision_model is None else decision_model
        )
        self._thread_status_values: dict[str, str] = {}
        self._thread_status_keys: dict[str, str] = {}
        self._thread_status_locks: dict[str, asyncio.Lock] = {}

        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            toolkits=toolkits,
            rules=rules,
            client_rules=client_rules,
            auto_greet_message=auto_greet_message,
            empty_state_title=empty_state_title,
            annotations=annotations,
            always_reply=always_reply,
            skill_dirs=skill_dirs,
        )

    def default_model(self) -> str:
        return self._llm_adapter.default_model()

    def _thread_status_attribute_name(self, *, path: str) -> str:
        return f"thread.status.{path}"

    def _status_lock(self, *, path: str) -> asyncio.Lock:
        lock = self._thread_status_locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_status_locks[path] = lock
        return lock

    async def _set_thread_status(self, *, path: str, status: Optional[str]) -> None:
        if self._room is None or self._room.local_participant is None:
            return

        attribute_name = self._thread_status_attribute_name(path=path)
        if status is None:
            self._thread_status_values.pop(path, None)
            await self._room.local_participant.set_attribute(attribute_name, None)
            return

        normalized = status.strip()
        if normalized == "":
            self._thread_status_values.pop(path, None)
            await self._room.local_participant.set_attribute(attribute_name, None)
            return

        if self._thread_status_values.get(path) == normalized:
            return

        self._thread_status_values[path] = normalized
        await self._room.local_participant.set_attribute(attribute_name, normalized)

    async def _apply_thread_status(self, *, path: str, status: Optional[str]) -> None:
        lock = self._status_lock(path=path)
        async with lock:
            await self._set_thread_status(path=path, status=status)

    def _set_thread_status_nowait(self, *, path: str, status: Optional[str]) -> None:
        async def run() -> None:
            try:
                await self._apply_thread_status(path=path, status=status)
            except Exception as ex:
                logger.error(
                    f"unable to set thread status for {path}",
                    exc_info=ex,
                )

        asyncio.create_task(run())

    def _status_event_details(
        self, *, event: dict
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        normalized_event = None
        event_type = event.get("type")
        if event_type in ("agent.event", "codex.event"):
            normalized_event = event
        else:
            normalized_event = response_event_to_agent_event(event)

        if not isinstance(normalized_event, dict):
            return None, None, None

        kind = normalized_event.get("kind")
        if not isinstance(kind, str):
            kind = ""
        kind = kind.strip().lower()
        if kind not in (
            "exec",
            "tool",
            "web",
            "search",
            "diff",
            "image",
        ):
            return None, None, None

        state = normalized_event.get("state")
        if not isinstance(state, str):
            state = ""
        state = state.strip().lower()

        key = None
        for candidate in (
            normalized_event.get("correlation_key"),
            normalized_event.get("event_key"),
            normalized_event.get("item_id"),
            normalized_event.get("name"),
            normalized_event.get("method"),
        ):
            if isinstance(candidate, str) and candidate.strip() != "":
                key = candidate.strip()
                break

        text = None
        for candidate in (
            normalized_event.get("headline"),
            normalized_event.get("summary"),
            normalized_event.get("name"),
            normalized_event.get("method"),
        ):
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized != "":
                    text = normalized
                    break

        return key, state, text

    def _update_thread_status_from_event(self, *, path: str, event: dict) -> None:
        key, state, text = self._status_event_details(event=event)
        if state is None:
            return

        is_active = state in (
            "queued",
            "in_progress",
            "running",
            "pending",
            "searching",
        )
        if is_active:
            if text is None:
                return
            if key is not None:
                self._thread_status_keys[path] = key
            self._set_thread_status_nowait(path=path, status=text)
            return

        if key is not None:
            tracked = self._thread_status_keys.get(path)
            if tracked is not None and tracked == key:
                self._clear_thread_status_nowait(path=path)
            return

        if state in ("completed", "failed", "cancelled"):
            self._clear_thread_status_nowait(path=path)

    async def _clear_thread_status(self, *, path: str) -> None:
        self._thread_status_keys.pop(path, None)
        await self._apply_thread_status(path=path, status=None)

    def _clear_thread_status_nowait(self, *, path: str) -> None:
        self._thread_status_keys.pop(path, None)
        self._set_thread_status_nowait(path=path, status=None)

    async def _clear_all_thread_statuses(self) -> None:
        paths = {
            *self._thread_status_values.keys(),
            *self._thread_status_keys.keys(),
        }
        for path in paths:
            await self._set_thread_status(path=path, status=None)
        self._thread_status_keys.clear()
        self._thread_status_values.clear()
        self._thread_status_locks.clear()

    async def create_thread_context(
        self,
        *,
        path: str,
        thread: MeshDocument,
        participants: list[RemoteParticipant],
        event_handler: Callable[[dict], None],
    ) -> ChatThreadContext:
        return ChatThreadContext(
            path=path,
            thread=thread,
            participants=participants,
            event_handler=event_handler,
            chat=await self.init_chat_context(),
        )

    # Backwards compatibility for existing subclasses overriding init_chat_context.
    async def init_chat_context(self) -> AgentChatContext:
        context = self._llm_adapter.create_chat_context()
        context.append_rules(self._rules)
        return context

    async def prepare_llm_context(self, *, thread_context: ChatThreadContext):
        """
        called prior to sending the request to the LLM in case the agent needs to modify the context prior to sending
        """
        pass

    def prepare_chat_context(self, *, chat_context: AgentChatContext):
        pass

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ) -> list[Toolkit]:
        toolkits = await super().get_thread_toolkits(
            thread_context=thread_context, participant=participant
        )

        if isinstance(self._llm_adapter, OpenAIResponsesAdapter):
            toolkits.insert(
                0,
                Toolkit(
                    name="reasoning",
                    tools=[
                        ChatBotReasoningTool(
                            room=self._room,
                            thread_context=thread_context,
                        )
                    ],
                ),
            )

        return toolkits

    async def should_reply(
        self,
        *,
        context: ChatThreadContext,
        has_more_than_one_other_user: bool,
        toolkits: list[Toolkit],
        from_user: RemoteParticipant,
        online: list[Participant],
    ):
        if not has_more_than_one_other_user or self._always_reply:
            return True

        online_set = {}

        all_members = []
        online_members = []

        for m in self.get_thread_members(thread=context.thread):
            all_members.append(m)

        for o in online:
            if o.get_attribute("name") not in online_set:
                online_set[o.get_attribute("name")] = True
                online_members.append(o.get_attribute("name"))

        logger.info(
            "multiple participants detected, checking whether agent should reply to conversation"
        )

        toolkits_json = ""
        for toolkit in toolkits:
            toolkits_json = (
                toolkits_json
                + f"\n{toolkit.name} ({toolkit.title}): {toolkit.description}"
            )
            for t in toolkit.tools:
                toolkits_json = (
                    toolkits_json + f"\n - {t.name} ({t.title}): {t.description}"
                )

        print(toolkits_json)

        cloned_context = context.chat.copy()
        cloned_context.replace_rules(
            rules=[
                "examine the conversation so far and return whether the user is expecting a reply from you or another user as the next message in the conversation",
                f'your name (the assistant) is "{self.room.local_participant.get_attribute("name")}"',
                "if the user mentions a person with another name, they aren't talking to you unless they also mention you",
                "if the user poses a question to everyone, they are talking to you",
                "to help identify the different users in the conversation, every message in the thread will start with '{user_name} said at {time}'",
                f"members of thread are currently {all_members}",
                f"users online currently are {online_members}",
                "if in doubt, reply to the user",
                f"if the user is asking for something that these toolkits can do, they want an answer from you: {toolkits_json}",
                "if the user they appear to be talking to is offline, then they probably are talking to you",
            ]
        )
        response = await self._llm_adapter.next(
            context=cloned_context,
            room=self._room,
            model=self._decision_model or self._llm_adapter.default_model(),
            on_behalf_of=from_user,
            toolkits=[],
            output_schema={
                "type": "object",
                "required": ["reasoning", "expecting_assistant_reply", "next_user"],
                "additionalProperties": False,
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "explain why you think the user was or was not expecting you to reply",
                    },
                    "next_user": {
                        "type": "string",
                        "description": "who would be expected to send the next message in the conversation",
                    },
                    "expecting_assistant_reply": {"type": "boolean"},
                },
            },
        )

        logger.info(f"should reply check returned {response}")

        return response["expecting_assistant_reply"]

    async def on_thread_open(self, *, thread_context: ChatThreadContext):
        await self._clear_thread_status(path=thread_context.path)

    async def on_thread_clear(self, *, thread_context: ChatThreadContext):
        await self._clear_thread_status(path=thread_context.path)

    async def on_thread_cancel(self, *, thread_context: ChatThreadContext):
        await self._clear_thread_status(path=thread_context.path)

    async def on_thread_close(self, *, thread_context: ChatThreadContext):
        await self._clear_thread_status(path=thread_context.path)

    async def on_chat_received(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> Optional[str]:
        rules = await self.get_rules(
            thread_context=thread_context,
            participant=from_participant,
        )
        thread_context.chat.replace_rules(rules)

        attachments = message.get("attachments", [])
        for attachment in attachments:
            thread_context.chat.append_assistant_message(
                message=f"the user attached a file at the path '{attachment['path']}'"
            )

        text = message["text"]
        iso_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        thread_context.chat.append_user_message(
            message=self.format_message(
                user_name=from_participant.get_attribute("name"),
                message=text,
                iso_timestamp=iso_timestamp,
            )
        )

        model = message.get("model", self.default_model())

        with tracer.start_as_current_span("chatbot.llm"):
            with tracer.start_as_current_span("get_thread_toolkits"):
                thread_toolkits = await self.get_thread_toolkits(
                    thread_context=thread_context,
                    participant=from_participant,
                )

            with tracer.start_as_current_span("get_thread_toolkit_builders"):
                thread_tool_providers = self.get_toolkit_builders()

            await self.prepare_llm_context(thread_context=thread_context)

            message_toolkits = [*thread_toolkits]
            message_tools = message.get("tools")

            if message_tools is not None and len(message_tools) > 0:
                message_toolkits.extend(
                    await make_toolkits(
                        room=self.room,
                        model=model,
                        providers=thread_tool_providers,
                        tools=message_tools,
                    )
                )

        online = await self.get_online_participants(
            thread=thread_context.thread, exclude=[self.room.local_participant]
        )

        for participant in get_online_participants(
            room=self._room, thread=thread_context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": True, "path": thread_context.path},
            )

        has_more_than_one_other_user = False

        thread_participants = []

        for member_name in self.get_thread_members(thread=thread_context.thread):
            thread_participants.append(member_name)
            if member_name != self._room.local_participant.get_attribute(
                "name"
            ) and member_name != from_participant.get_attribute("name"):
                has_more_than_one_other_user = True
                break

        thread_context.chat.metadata["thread_participants"] = thread_participants

        reply = await self.should_reply(
            has_more_than_one_other_user=has_more_than_one_other_user,
            online=online,
            context=thread_context,
            toolkits=message_toolkits,
            from_user=from_participant,
        )

        for participant in get_online_participants(
            room=self._room, thread=thread_context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": False, "path": thread_context.path},
            )

        if not reply:
            return None

        self.prepare_chat_context(chat_context=thread_context.chat)

        await self._clear_thread_status(path=thread_context.path)
        try:
            return await self._llm_adapter.next(
                context=thread_context.chat,
                room=self._room,
                toolkits=message_toolkits,
                tool_adapter=self._tool_adapter,
                event_handler=thread_context.emit,
                model=model,
                on_behalf_of=from_participant,
            )
        finally:
            await self._clear_thread_status(path=thread_context.path)
