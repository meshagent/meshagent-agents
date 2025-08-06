from meshagent.agents.agent import SingleRoomAgent, AgentChatContext
from meshagent.api.chan import Chan
from meshagent.api import (
    RoomMessage,
    RoomClient,
    RemoteParticipant,
    RequiredSchema,
    Requirement,
    Element,
    MeshDocument,
)
from meshagent.tools import Toolkit, ToolContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.openai.tools.responses_adapter import ImageGenerationTool, LocalShellTool
import asyncio
from typing import Optional
import logging
from meshagent.tools import MultiToolkit
import uuid
import datetime
from typing import Literal
import base64
from openai.types.responses import ResponseStreamEvent
from asyncio import CancelledError
from meshagent.api import RoomException

from opentelemetry import trace
import shlex

tracer = trace.get_tracer("meshagent.chatbot")

logger = logging.getLogger("chat")


class ChatBotThreadLocalShellTool(LocalShellTool):
    def __init__(self, *, thread_context: "ChatThreadContext"):
        super().__init__()
        self.thread_context = thread_context

    async def execute_shell_command(
        self,
        context,
        *,
        command,
        env,
        type,
        timeout_ms=None,
        user=None,
        working_directory=None,
    ):
        messages = None

        for prop in self.thread_context.thread.root.get_children():
            if prop.tag_name == "messages":
                messages = prop
                break

        exec_element = messages.append_child(
            tag_name="exec",
            attributes={"command": shlex.join(command), "pwd": working_directory},
        )

        result = await super().execute_shell_command(
            context,
            command=command,
            env=env,
            type=type,
            timeout_ms=timeout_ms,
            user=user,
            working_directory=working_directory,
        )

        exec_element.set_attribute("result", result)

        return result


class ChatBotThreadOpenAIImageGenerationTool(ImageGenerationTool):
    def __init__(
        self,
        *,
        background: Literal["transparent", "opaque", "auto"] = None,
        input_image_mask_url: Optional[str] = None,
        model: Optional[str] = None,
        moderation: Optional[str] = None,
        output_compression: Optional[int] = None,
        output_format: Optional[Literal["png", "webp", "jpeg"]] = None,
        partial_images: Optional[int] = None,
        quality: Optional[Literal["auto", "low", "medium", "high"]] = None,
        size: Optional[Literal["1024x1024", "1024x1536", "1536x1024", "auto"]] = None,
        thread_context: "ChatThreadContext",
    ):
        super().__init__(
            background=background,
            input_image_mask_url=input_image_mask_url,
            model=model,
            moderation=moderation,
            output_compression=output_compression,
            output_format=output_format,
            partial_images=partial_images,
            quality=quality,
            size=size,
        )

        self.thread_context = thread_context

    async def on_image_generation_partial(
        self,
        context,
        *,
        item_id,
        output_index,
        sequence_number,
        type,
        partial_image_b64,
        partial_image_index,
        size,
        quality,
        background,
        output_format,
        **extra,
    ):
        output_format = self.output_format
        if output_format is None:
            output_format = "png"

        image_name = f"{str(uuid.uuid4())}.{output_format}"

        handle = await context.room.storage.open(path=image_name)
        await context.room.storage.write(
            handle=handle, data=base64.b64decode(partial_image_b64)
        )
        await context.room.storage.close(handle=handle)

        messages = None

        for prop in self.thread_context.thread.root.get_children():
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
                "author_name": context.room.local_participant.get_attribute("name"),
            },
        )
        message_element.append_child(tag_name="file", attributes={"path": image_name})

    async def on_image_generated(
        self,
        context: ToolContext,
        *,
        item_id: str,
        data: bytes,
        status: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        **extra,
    ):
        output_format = self.output_format
        if output_format is None:
            output_format = "png"

        image_name = f"{str(uuid.uuid4())}.{output_format}"

        handle = await context.room.storage.open(path=image_name)
        await context.room.storage.write(handle=handle, data=data)
        await context.room.storage.close(handle=handle)

        messages = None

        for prop in self.thread_context.thread.root.get_children():
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
                "author_name": context.room.local_participant.get_attribute("name"),
            },
        )
        message_element.append_child(tag_name="file", attributes={"path": image_name})

        self.thread_context.chat.append_assistant_message(
            f"An image was saved at the path {image_name} and displayed to the user"
        )


def get_thread_participants(
    *, room: RoomClient, thread: MeshDocument
) -> list[RemoteParticipant]:
    results = list[RemoteParticipant]()

    for prop in thread.root.get_children():
        if prop.tag_name == "members":
            for member in prop.get_children():
                for online in room.messaging.get_participants():
                    if online.get_attribute("name") == member.get_attribute("name"):
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
    ):
        self.thread = thread
        if participants is None:
            participants = []

        self.participants = participants
        self.chat = chat
        self.path = path


# todo: thread should stop when participant stops?
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
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        labels: Optional[str] = None,
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

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter

        self._message_channels = dict[str, Chan[RoomMessage]]()

        self._room: RoomClient | None = None
        self._toolkits = toolkits

        if rules is None:
            rules = []

        self._rules = rules
        self._is_typing = dict[str, asyncio.Task]()
        self._auto_greet_message = auto_greet_message

        if empty_state_title is None:
            empty_state_title = "How can I help you?"
        self._empty_state_title = empty_state_title

        self._thread_tasks = dict[str, asyncio.Task]()

    def init_requirements(self, requires: list[Requirement]):
        if requires is None:
            requires = [RequiredSchema(name="thread")]

        else:
            thread_schema = list(
                n
                for n in requires
                if (isinstance(n, RequiredSchema) and n.name == "thread")
            )
            if len(thread_schema) == 0:
                requires.append(RequiredSchema(name="thread"))

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

    async def get_thread_participants(self, *, thread: MeshDocument):
        return get_thread_participants(room=self._room, thread=thread)

    async def get_thread_toolkits(
        self, *, thread_context: ChatThreadContext, participant: RemoteParticipant
    ) -> list[Toolkit]:
        toolkits = await self.get_required_toolkits(
            context=ToolContext(
                room=self.room,
                caller=participant,
                caller_context={"chat": thread_context.chat.to_json()},
            )
        )
        toaster = None

        for toolkit in toolkits:
            if toolkit.name == "ui":
                for tool in toolkit.tools:
                    if tool.name == "show_toast":
                        toaster = tool

        if toaster is not None:

            def multi_tool(toolkit: Toolkit):
                if toaster in toolkit.tools:
                    return toolkit

                return MultiToolkit(required=[toaster], base_toolkit=toolkit)

            toolkits = list(map(multi_tool, toolkits))

        return [*self._toolkits, *toolkits]

    async def init_chat_context(self) -> AgentChatContext:
        context = self._llm_adapter.create_chat_context()
        context.append_rules(self._rules)
        return context

    async def open_thread(self, *, path: str):
        return await self.room.sync.open(path=path)

    async def close_thread(self, *, path: str):
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
                            chat_context.append_user_message(msg)

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
                for participant in self._room.messaging.get_participants():
                    logger.debug(
                        f"sending event {evt.type} to {participant.get_attribute('name')}"
                    )

                    # self.room.messaging.send_message_nowait(to=participant, type="llm.event", message=json.loads(evt.to_json()))

                if evt.type == "response.content_part.added":
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

                elif evt.type == "response.output_text.delta":
                    partial += evt.delta
                    updates.put_nowait((content_element, partial))
                    context_message["content"] = partial

                elif evt.type == "response.output_text.done":
                    content_element = None

                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attribute(
                            "from_participant_name",
                            self.room.local_participant.get_attribute("name"),
                        )
                        span.set_attribute("role", "assistant")
                        span.set_attributes(thread_attributes)
                        span.set_attributes({"text": evt.text})
        except asyncio.QueueShutDown:
            pass
        finally:
            updates.shutdown()
            await update_thread_task

    async def _spawn_thread(self, path: str, messages: Chan[RoomMessage]):
        logger.debug("chatbot is starting a thread", extra={"path": path})
        chat_context = await self.init_chat_context()
        opened = False

        current_file = None
        thread_context = None

        thread_attributes = None

        thread = None

        try:
            received = None

            while True:
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

                    if current_file != chat_with_participant.get_attribute(
                        "current_file"
                    ):
                        logger.info(
                            f"participant is now looking at {chat_with_participant.get_attribute('current_file')}"
                        )
                        current_file = chat_with_participant.get_attribute(
                            "current_file"
                        )

                    if current_file is not None:
                        chat_context.append_assistant_message(
                            message=f"the user is currently viewing the file at the path: {current_file}"
                        )

                    elif current_file is not None:
                        chat_context.append_assistant_message(
                            message="the user is not current viewing any files"
                        )

                    if thread is None:
                        with tracer.start_as_current_span(
                            "chatbot.thread.open"
                        ) as span:
                            span.set_attributes(thread_attributes)

                            thread = await self.open_thread(path=path)

                            thread_context = ChatThreadContext(
                                path=path,
                                chat=chat_context,
                                thread=thread,
                                participants=get_thread_participants(
                                    room=self.room, thread=thread
                                ),
                            )

                            await self.load_thread_context(
                                thread_context=thread_context
                            )

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

                    if received.type == "chat":
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

                        chat_context.append_user_message(message=text)

                        if messages.empty():
                            break

                if received is not None:
                    with tracer.start_as_current_span("chatbot.thread.message") as span:
                        span.set_attributes(thread_attributes)
                        span.set_attribute("role", "user")
                        span.set_attribute(
                            "from_participant_name",
                            chat_with_participant.get_attribute("name"),
                        )

                        attachments = received.message.get("attachments", [])
                        span.set_attribute("attachments", attachments)

                        text = received.message["text"]
                        span.set_attributes({"text": text})

                        try:
                            for participant in get_thread_participants(
                                room=self._room, thread=thread
                            ):
                                # TODO: async gather
                                self._room.messaging.send_message_nowait(
                                    to=participant,
                                    type="thinking",
                                    message={"thinking": True, "path": path},
                                )

                            if thread_context is None:
                                thread_context = ChatThreadContext(
                                    path=path,
                                    chat=chat_context,
                                    thread=thread,
                                    participants=get_thread_participants(
                                        room=self.room, thread=thread
                                    ),
                                )
                            else:
                                thread_context.participants = get_thread_participants(
                                    room=self.room, thread=thread
                                )

                            with tracer.start_as_current_span("chatbot.llm") as span:
                                try:
                                    with tracer.start_as_current_span(
                                        "get_thread_toolkits"
                                    ) as span:
                                        thread_toolkits = (
                                            await self.get_thread_toolkits(
                                                thread_context=thread_context,
                                                participant=participant,
                                            )
                                        )

                                    await self.prepare_llm_context(
                                        thread_context=thread_context
                                    )

                                    llm_messages = asyncio.Queue[ResponseStreamEvent]()

                                    def handle_event(evt):
                                        llm_messages.put_nowait(evt)

                                    llm_task = asyncio.create_task(
                                        self._process_llm_events(
                                            thread_context=thread_context,
                                            llm_messages=llm_messages,
                                            thread_attributes=thread_attributes,
                                        )
                                    )

                                    await self._llm_adapter.next(
                                        context=chat_context,
                                        room=self._room,
                                        toolkits=thread_toolkits,
                                        tool_adapter=self._tool_adapter,
                                        event_handler=handle_event,
                                    )

                                    llm_messages.shutdown()
                                    await llm_task

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
                                for participant in get_thread_participants(
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

    async def start(self, *, room):
        await super().start(room=room)

        logger.debug("Starting chatbot")

        await self.room.local_participant.set_attribute(
            "empty_state_title", self._empty_state_title
        )

        def on_message(message: RoomMessage):
            if message.type == "chat" or message.type == "opened":
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
