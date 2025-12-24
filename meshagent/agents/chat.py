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
    RequiredSchema,
)
from meshagent.tools import Toolkit, ToolContext, make_toolkits, ToolkitBuilder
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.openai.tools.responses_adapter import (
    ReasoningTool,
)
import asyncio
from typing import Optional, Callable
import logging
import uuid
import datetime
import base64
from asyncio import CancelledError
from meshagent.api import RoomException
from pydantic import BaseModel
from opentelemetry import trace
import shlex
import json

from meshagent.agents.thread_schema import thread_schema

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


class ChatBot(SingleRoomAgent):
    def __init__(
        self,
        *,
        name,
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
        labels: Optional[list[str]] = None,
        decision_model: Optional[str] = None,
        always_reply: Optional[bool] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            labels=labels,
        )

        if toolkits is None:
            toolkits = []

        self._decision_model = (
            "gpt-4.1-mini" if decision_model is None else decision_model
        )

        if always_reply is None:
            always_reply = False

        self._always_reply = always_reply

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter

        self._message_channels = dict[str, Chan[RoomMessage]]()

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
                    "created_at": datetime.datetime.now(datetime.timezone.utc)
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
        chat_context: AgentChatContext,
        participant: RemoteParticipant,
        thread_attributes: dict,
    ):
        if self._auto_greet_message is not None:
            chat_context.append_user_message(self._auto_greet_message)
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
            RequiredSchema(name="thread", schema=thread_schema),
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
                caller_context={"chat": thread_context.chat.to_json()},
                event_handler=thread_context.emit,
            )
        )

        toolkits.append(
            Toolkit(
                name="reasoning",
                tools=[
                    ChatBotReasoningTool(
                        room=self._room,
                        thread_context=thread_context,
                    )
                ],
            )
        )

        return [*self._toolkits, *toolkits]

    async def init_chat_context(self) -> AgentChatContext:
        context = self._llm_adapter.create_chat_context()
        context.append_rules(self._rules)
        return context

    async def open_thread(self, *, path: str):
        logger.info(f"opening thread {path}")
        if path not in self._open_threads:
            fut = asyncio.ensure_future(
                self.room.sync.open(path=path, schema=thread_schema)
            )
            self._open_threads[path] = fut

        return await self._open_threads[path]

    async def close_thread(self, *, path: str):
        logger.info(f"closing thread {path}")

        if path in self._open_threads:
            del self._open_threads[path]

        return await self.room.sync.close(path=path)

    async def load_thread_context(self, *, thread_context: ChatThreadContext):
        """
        load the thread from the thread document by inserting the current messages in the thread into the chat context
        """
        thread = thread_context.thread
        chat_context = thread_context.chat
        for prop in thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop

                for element in doc_messages.get_children():
                    if isinstance(element, Element) and element.tag_name == "message":
                        msg = element["text"]
                        if element[
                            "author_name"
                        ] == self.room.local_participant.get_attribute("name"):
                            chat_context.append_assistant_message(msg)
                        else:
                            chat_context.append_user_message(
                                self.format_message(
                                    user_name=element["author_name"],
                                    message=msg,
                                    iso_timestamp=element["created_at"],
                                )
                            )

                        for child in element.get_children():
                            if child.tag_name == "file":
                                chat_context.append_assistant_message(
                                    f"the user attached a file with the path '{child.get_attribute('path')}'"
                                )

                break

        if doc_messages is None:
            raise Exception("thread was not properly initialized")

    async def prepare_llm_context(self, *, thread_context: ChatThreadContext):
        """
        called prior to sending the request to the LLM in case the agent needs to modify the context prior to sending
        """
        pass

    async def _process_llm_events(
        self,
        *,
        thread_context: ChatThreadContext,
        llm_messages: asyncio.Queue,
        thread_attributes: dict,
    ):
        thread = thread_context.thread
        doc_messages = None
        for prop in thread.root.get_children():
            if prop.tag_name == "messages":
                doc_messages = prop
                break

        if doc_messages is None:
            raise RoomException("messages element is missing from thread document")

        context_message = None
        updates = asyncio.Queue()

        # throttle updates so we don't send too many syncs over the wire at once
        async def update_thread():
            try:
                changes = {}
                while True:
                    try:
                        element, partial = updates.get_nowait()
                        changes[element] = partial

                    except asyncio.QueueEmpty:
                        for e, p in changes.items():
                            e["text"] = p

                        changes.clear()

                        e, p = await updates.get()
                        changes[e] = p

                        await asyncio.sleep(0.1)

            except asyncio.QueueShutDown:
                # flush any pending changes
                for e, p in changes.items():
                    e["text"] = p

                changes.clear()
                pass

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await llm_messages.get()

                if evt["type"] == "response.content_part.added":
                    partial = ""

                    content_element = doc_messages.append_child(
                        tag_name="message",
                        attributes={
                            "text": "",
                            "created_at": datetime.datetime.now(datetime.timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "author_name": self.room.local_participant.get_attribute(
                                "name"
                            ),
                        },
                    )

                    context_message = {"role": "assistant", "content": ""}
                    thread_context.chat.messages.append(context_message)

                elif evt["type"] == "response.output_text.delta":
                    partial += evt["delta"]
                    updates.put_nowait((content_element, partial))
                    context_message["content"] = partial

                elif evt["type"] == "response.output_text.done":
                    content_element = None

                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attribute(
                            "from_participant_name",
                            self.room.local_participant.get_attribute("name"),
                        )
                        span.set_attribute("role", "assistant")
                        span.set_attributes(thread_attributes)
                        span.set_attributes({"text": evt["text"]})

                        for participant in get_online_participants(
                            room=self._room, thread=thread_context.thread
                        ):
                            if participant.id != self._room.local_participant.id:
                                logger.info(
                                    f"replying to {participant.get_attribute('name')}"
                                )
                                self._room.messaging.send_message_nowait(
                                    to=participant,
                                    type="chat",
                                    message={
                                        "type": "chat",
                                        "path": thread_context.path,
                                        "text": evt["text"],
                                    },
                                )
                elif evt["type"] == "response.image_generation_call.partial_image":
                    await self.handle_image_generation_partial(
                        thread_context=thread_context,
                        llm_messages=llm_messages,
                        event=evt,
                    )

                elif evt["type"] == "meshagent.handler.added":
                    item = evt["item"]
                    print(f"{item}", flush=True)
                    if item["type"] == "shell_call":
                        await self.handle_shell_call_output(
                            thread_context=thread_context,
                            llm_messages=llm_messages,
                            item=item,
                        )

                    elif item["type"] == "local_shell_call":
                        await self.handle_local_shell_call_output(
                            thread_context=thread_context,
                            llm_messages=llm_messages,
                            item=item,
                        )

        except asyncio.QueueShutDown:
            pass
        finally:
            updates.shutdown()

        await update_thread_task

    async def handle_image_generation_partial(
        self,
        *,
        thread_context: ChatThreadContext,
        llm_messages: asyncio.Queue,
        event: dict,
    ):
        item_id = event["item_id"]
        partial_image_b64 = event["partial_image_b64"]
        output_format = event["output_format"]

        messages = thread_context.thread.root.get_children_by_tag_name("messages")[0]

        if output_format is None:
            output_format = "png"

        image_name = f"{str(uuid.uuid4())}.{output_format}"

        handle = await self.room.storage.open(path=image_name)
        await self.room.storage.write(
            handle=handle, data=base64.b64decode(partial_image_b64)
        )
        await self.room.storage.close(handle=handle)

        messages = None

        logger.info(f"A partial was saved at the path {image_name}")

        for prop in thread_context.thread.root.get_children():
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
                "created_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": self.room.local_participant.get_attribute("name"),
            },
        )
        message_element.append_child(tag_name="file", attributes={"path": image_name})

    async def handle_local_shell_call_output(
        self,
        *,
        thread_context: ChatThreadContext,
        llm_messages: asyncio.Queue,
        item: dict,
    ):
        messages = thread_context.thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        command = action["command"]
        working_directory = action["working_directory"]

        for prop in thread_context.thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        exec_element = messages.append_child(
            tag_name="exec",
            attributes={"command": shlex.join(command), "pwd": working_directory},
        )

        evt = await llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = item.get("error")
        item = item.get("item")

        if error is not None:
            pass

        if item is not None:
            if item["type"] != "local_shell_call_output":
                raise RoomException("expected local_shell_call_output")

            exec_element.set_attribute("result", item["output"])

    async def handle_shell_call_output(
        self,
        *,
        thread_context: ChatThreadContext,
        llm_messages: asyncio.Queue,
        item: dict,
    ):
        messages = thread_context.thread.root.get_children_by_tag_name("messages")[0]

        action = item["action"]
        commands = action["commands"]

        exec_elements = []
        for command in commands:
            exec_element = messages.append_child(
                tag_name="exec",
                attributes={"command": command},
            )
            exec_elements.append(exec_element)

        evt = await llm_messages.get()

        if evt["type"] != "meshagent.handler.done":
            raise RoomException("expected meshagent.handler.done")

        error = item.get("error")
        item = item.get("item")

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

    def get_thread_members(self, *, thread: MeshDocument) -> list[str]:
        results = []

        for prop in thread.root.get_children():
            if prop.tag_name == "members":
                for member in prop.get_children():
                    results.append(member.get_attribute("name"))

        return results

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

        cloned_context = context.chat.copy()
        cloned_context.replace_rules(
            rules=[
                "examine the conversation so far and return whether the user is expecting a reply from you or another user as the next message in the conversation",
                f'your name (the assistant) is "{self.room.local_participant.get_attribute("name")}"',
                "if the user mentions a person with another name, they aren't talking to you unless they also mention you",
                "if the user poses a question to everyone, they are talking to you",
                f"members of thread are currently {all_members}",
                f"users online currently are {online_members}",
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
                        "description": "who would be expectd to send the next message in the conversation",
                    },
                    "expecting_assistant_reply": {"type": "boolean"},
                },
            },
        )

        logger.info(f"should reply check returned {response}")

        return response["expecting_assistant_reply"]

    async def handle_user_message(
        self,
        *,
        context: ChatThreadContext,
        toolkits: list[Toolkit],
        model: str,
        from_user: RemoteParticipant,
        event_handler,
    ):
        online = await self.get_online_participants(
            thread=context.thread, exclude=[self.room.local_participant]
        )

        for participant in get_online_participants(
            room=self._room, thread=context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": True, "path": context.path},
            )

        has_more_than_one_other_user = False

        for member_name in self.get_thread_members(thread=context.thread):
            if member_name != self._room.local_participant.get_attribute(
                "name"
            ) and member_name != from_user.get_attribute("name"):
                has_more_than_one_other_user = True
                break

        reply = await self.should_reply(
            has_more_than_one_other_user=has_more_than_one_other_user,
            online=online,
            context=context,
            toolkits=toolkits,
            from_user=from_user,
        )

        for participant in get_online_participants(
            room=self._room, thread=context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="listening",
                message={"listening": False, "path": context.path},
            )

        if not reply:
            return

        for participant in get_online_participants(
            room=self._room, thread=context.thread
        ):
            self._room.messaging.send_message_nowait(
                to=participant,
                type="thinking",
                message={"thinking": True, "path": context.path},
            )

        self.prepare_chat_context(chat_context=context.chat)

        await self._llm_adapter.next(
            context=context.chat,
            room=self._room,
            toolkits=toolkits,
            tool_adapter=self._tool_adapter,
            event_handler=event_handler,
            model=model,
            on_behalf_of=from_user,
        )

    async def get_rules(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ):
        rules = [*self._rules]
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
    ):
        rules = await self.get_rules(
            thread_context=thread_context, participant=from_participant
        )
        thread_context.chat.replace_rules(rules)

    def format_message(self, *, user_name: str, message: str, iso_timestamp: str):
        return f"{user_name} said at {iso_timestamp}: {message}"

    def prepare_chat_context(self, *, chat_context: ChatThreadContext):
        pass

    async def _spawn_thread(self, path: str, messages: Chan[RoomMessage]):
        logger.debug("chatbot is starting a thread", extra={"path": path})
        chat_context = await self.init_chat_context()
        opened = False

        current_file = None
        thread_context = None

        thread_attributes = None

        thread = None

        llm_messages = asyncio.Queue[dict]()
        llm_task = None

        def handle_event(evt):
            if isinstance(evt, BaseModel):
                evt = evt.model_dump(mode="json")
            llm_messages.put_nowait(evt)

        try:
            received = None

            while True:
                logger.debug(f"waiting for message on thread {path}")
                received = await messages.recv()
                logger.debug(f"received message on thread {path}: {received.type}")

                chat_with_participant = None
                for participant in self._room.messaging.get_participants():
                    if participant.id == received.from_participant_id:
                        chat_with_participant = participant
                        break

                if chat_with_participant is None:
                    logger.warning(
                        "participant does not have messaging enabled, skipping message"
                    )
                    continue

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
                        f"participant is now looking at {chat_with_participant.get_attribute('current_file')}"
                    )
                    current_file = chat_with_participant.get_attribute("current_file")

                if current_file is not None:
                    chat_context.append_assistant_message(
                        message=f"the user is currently viewing the file at the path: {current_file}"
                    )

                elif current_file is not None:
                    chat_context.append_assistant_message(
                        message="the user is not current viewing any files"
                    )

                if thread is None:
                    with tracer.start_as_current_span("chatbot.thread.open") as span:
                        span.set_attributes(thread_attributes)

                        thread = await self.open_thread(path=path)

                        thread_context = ChatThreadContext(
                            path=path,
                            chat=chat_context,
                            thread=thread,
                            participants=get_online_participants(
                                room=self.room, thread=thread
                            ),
                            event_handler=handle_event,
                        )

                        llm_task = asyncio.create_task(
                            self._process_llm_events(
                                thread_context=thread_context,
                                llm_messages=llm_messages,
                                thread_attributes=thread_attributes,
                            )
                        )

                        await self.load_thread_context(thread_context=thread_context)

                if received.type == "opened":
                    if not opened:
                        opened = True

                        await self._greet(
                            path=path,
                            chat_context=chat_context,
                            participant=chat_with_participant,
                            thread=thread,
                            thread_attributes=thread_attributes,
                        )
                elif received.type == "clear":
                    chat_context = await self.init_chat_context()
                    thread_context.chat = chat_context
                    messages_element: Element = thread.root.get_children_by_tag_name(
                        "messages"
                    )[0]
                    for child in list(messages_element.get_children()):
                        child.delete()

                elif received.type == "chat":
                    if thread is None:
                        logger.info("thread is not open", extra={"path": path})
                        break

                    logger.debug(
                        "chatbot received a chat",
                        extra={
                            "context": chat_context.id,
                            "participant_id": self.room.local_participant.id,
                            "participant_name": self.room.local_participant.get_attribute(
                                "name"
                            ),
                            "text": received.message["text"],
                        },
                    )

                    attachments = received.message.get("attachments", [])
                    text = received.message["text"]

                    for attachment in attachments:
                        chat_context.append_assistant_message(
                            message=f"the user attached a file at the path '{attachment['path']}'"
                        )

                    iso_timestamp = (
                        datetime.datetime.now(datetime.timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )

                    chat_context.append_user_message(
                        message=self.format_message(
                            user_name=chat_with_participant.get_attribute("name"),
                            message=text,
                            iso_timestamp=iso_timestamp,
                        )
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
                                thread_context = ChatThreadContext(
                                    path=path,
                                    chat=chat_context,
                                    thread=thread,
                                    participants=get_online_participants(
                                        room=self.room, thread=thread
                                    ),
                                )
                            else:
                                thread_context.participants = get_online_participants(
                                    room=self.room, thread=thread
                                )

                            await self.on_chat_received(
                                thread_context=thread_context,
                                from_participant=chat_with_participant,
                                message=received.message,
                            )

                            with tracer.start_as_current_span("chatbot.llm") as span:
                                try:
                                    with tracer.start_as_current_span(
                                        "get_thread_toolkits"
                                    ) as span:
                                        thread_toolkits = (
                                            await self.get_thread_toolkits(
                                                thread_context=thread_context,
                                                participant=chat_with_participant,
                                            )
                                        )

                                    with tracer.start_as_current_span(
                                        "get_thread_toolkit_builders"
                                    ) as span:
                                        thread_tool_providers = (
                                            self.get_toolkit_builders()
                                        )

                                    await self.prepare_llm_context(
                                        thread_context=thread_context
                                    )

                                    message_toolkits = [*thread_toolkits]

                                    model = received.message.get(
                                        "model", self._llm_adapter.default_model()
                                    )

                                    message_tools = received.message.get("tools")

                                    if (
                                        message_tools is not None
                                        and len(message_tools) > 0
                                    ):
                                        message_toolkits.extend(
                                            await make_toolkits(
                                                room=self.room,
                                                model=model,
                                                providers=thread_tool_providers,
                                                tools=message_tools,
                                            )
                                        )

                                    await self.handle_user_message(
                                        context=thread_context,
                                        toolkits=message_toolkits,
                                        event_handler=handle_event,
                                        model=model,
                                        from_user=chat_with_participant,
                                    )

                                except Exception as e:
                                    logger.error("An error was encountered", exc_info=e)
                                    await self._send_and_save_chat(
                                        thread=thread,
                                        to=chat_with_participant,
                                        path=path,
                                        id=str(uuid.uuid4()),
                                        text="There was an error while communicating with the LLM. Please try again later.",
                                        thread_attributes=thread_attributes,
                                    )

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
                llm_messages.shutdown()
                if llm_task is not None:
                    await llm_task

                if self.room is not None:
                    logger.info(f"thread was ended {path}")
                    logger.info("chatbot thread ended", extra={"path": path})

                    if thread is not None:
                        await self.close_thread(path=path)

            asyncio.shield(cleanup())

    def _get_message_channel(self, key: str) -> Chan[RoomMessage]:
        if key not in self._message_channels:
            chan = Chan[RoomMessage]()
            self._message_channels[key] = chan

        chan = self._message_channels[key]

        return chan

    async def stop(self):
        await super().stop()

        for thread in self._thread_tasks.values():
            thread.cancel()

        self._thread_tasks.clear()

    async def _on_get_thread_toolkits_message(self, *, message: RoomMessage):
        path = message.message["path"]

        thread_context = None
        if path in self._open_threads:
            thread = await self._open_threads[path]

            thread_context = ChatThreadContext(
                path=path,
                chat=AgentChatContext(),
                thread=thread,
                participants=get_online_participants(room=self.room, thread=thread),
            )

        if thread_context is None:
            logger.warning("thread toolkits requested for a thread that is not open")
            return

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

    async def start(self, *, room):
        await super().start(room=room)

        logger.debug("Starting chatbot")

        await self.room.local_participant.set_attribute(
            "empty_state_title", self._empty_state_title
        )

        def on_message(message: RoomMessage):
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

                messages = self._get_message_channel(path)

                logger.debug(
                    f"queued incoming message for thread {path}: {message.type}"
                )

                messages.send_nowait(message)

                if path not in self._thread_tasks or self._thread_tasks[path].done():

                    def thread_done(task: asyncio.Task):
                        self._thread_tasks.pop(path)
                        self._message_channels.pop(path)
                        try:
                            task.result()
                        except CancelledError:
                            pass
                        except Exception as e:
                            logger.error(
                                f"The chat thread ended with an error {e}", exc_info=e
                            )

                    logger.debug(f"spawning chat thread for {path}")
                    task = asyncio.create_task(
                        self._spawn_thread(messages=messages, path=path)
                    )
                    task.add_done_callback(thread_done)

                    self._thread_tasks[path] = task

            elif message.type == "cancel":
                path = message.message["path"]
                if path in self._thread_tasks:
                    self._thread_tasks[path].cancel()

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

        room.messaging.on("message", on_message)

        if self._auto_greet_message is not None:

            def on_participant_added(participant: RemoteParticipant):
                # will spawn the initial thread
                self._get_message_channel(participant.id)

            room.messaging.on("participant_added", on_participant_added)

        logger.debug("Enabling chatbot messaging")
        await room.messaging.enable()
