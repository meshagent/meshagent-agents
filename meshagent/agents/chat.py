from .agent import SingleRoomAgent, AgentChatContext, AgentCallContext
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomException, RoomClient, RemoteParticipant, RequiredSchema, Requirement, Element, MeshDocument
from meshagent.tools import Toolkit
from .adapter import LLMAdapter, ToolResponseAdapter
import asyncio
from typing import Optional
import logging
from meshagent.tools import MultiToolkit
import urllib
import uuid
import datetime

from openai.types.responses import ResponseStreamEvent

logging.basicConfig()
logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)


# todo: thread should stop when participant stops?

def get_thread_participants(*, room: RoomClient, thread: MeshDocument) -> list[RemoteParticipant]:

    results = list[RemoteParticipant]()

    for prop in thread.root.get_children():

        if prop.tag_name == "members":

            for member in prop.get_children():

                for online in room.messaging.get_participants():

                    if online.get_attribute("name") == member.get_attribute("name"):

                        results.append(online)

    return results


class ChatThreadContext:
    def __init__(self, *, chat: AgentChatContext, thread: MeshDocument, toolkits: list[Toolkit], participants: Optional[list[RemoteParticipant]] = None):
        self.thread = thread
        self.toolkits = toolkits
        if participants == None:
            participants = []

        self.participants = participants
        self.chat = chat

class ChatBot(SingleRoomAgent):
    def __init__(self, *, name, title = None, description = None, requires : Optional[list[Requirement]] = None, llm_adapter: LLMAdapter, tool_adapter: Optional[ToolResponseAdapter] = None, toolkits: Optional[list[Toolkit]] = None, rules : Optional[list[str]] = None, auto_greet_message : Optional[str] = None,  empty_state_title : Optional[str] = None, labels: Optional[str] = None):
        
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            labels=labels
        )

        if toolkits == None:
            toolkits = []

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter

        self._message_channels = dict[str, Chan[RoomMessage]]()

        self._room : RoomClient | None = None
        self._toolkits = toolkits

        if rules == None:
            rules = []

        self._rules = rules
        self._is_typing = dict[str,asyncio.Task]()
        self._auto_greet_message = auto_greet_message
        
        if empty_state_title == None:
            empty_state_title = "How can I help you?"
        self._empty_state_title = empty_state_title

        self._thread_tasks = dict[str,asyncio.Task]()

    def init_requirements(self, requires: list[Requirement]):
        if requires == None:

            requires = [
                RequiredSchema(
                    name="thread"
                )
            ]

        else:
            
            thread_schema = list(n for n in requires if (isinstance(n, RequiredSchema) and n.name == "thread"))
            if len(thread_schema) == 0:
                requires.append(
                    RequiredSchema(
                        name="thread"
                    )
                )

    async def _send_and_save_chat(self, messages: Element, path: str, to: RemoteParticipant, id: str, text: str):

        await self.room.messaging.send_message(to=to, type="chat", message={  "path" : path, "text" : text })

        messages.append_child(tag_name="message", attributes={
            "id" : id,
            "text" : text,
            "created_at" : datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),
            "author_name" : self.room.local_participant.get_attribute("name"),
        })
        
     
    async def greet(self, *, messages: Element, path: str, chat_context: AgentChatContext, participant: RemoteParticipant):

        if self._auto_greet_message != None:
            chat_context.append_user_message(self._auto_greet_message)
            await self._send_and_save_chat(id=str(uuid.uuid4()), to=RemoteParticipant(id=participant.id), messages=messages, path=path, text= self._auto_greet_message)
           

    async def get_thread_participants(self, *, thread: MeshDocument):
        return get_thread_participants(room=self._room, thread=thread)
    
    async def init_thread_context(self, *, thread_context: ChatThreadContext) -> list[Toolkit]:

        toaster = None
        
        for toolkit in thread_context.toolkits:

            if toolkit.name == "meshagent.ui":

                for tool in toolkit.tools:

                    if tool.name == "show_toast":

                        toaster = tool

        if toaster != None:

            def multi_tool(toolkit: Toolkit):
                if toaster in toolkit.tools:
                    return toolkit
                
                return MultiToolkit(required=[ toaster ], base_toolkit=toolkit )

            toolkits = list(map(multi_tool, thread_context.toolkits))
        
            thread_context.toolkits = toolkits

    
    async def init_chat_context(self) -> AgentChatContext:
        context =  self._llm_adapter.create_chat_context()
        context.append_rules(self._rules)
        return context

    async def open_thread(self, *, path: str):


        return await self.room.sync.open(path=path)
    
    async def close_thread(self, *, path: str):

        return await self.room.sync.close(path=path)


    async def _spawn_thread(self, path: str, messages: Chan[RoomMessage]):

        self.room.developer.log_nowait(type="chatbot.thread.started", data={ "path" : path })
        chat_context = await self.init_chat_context()
        opened = False
        thread = None
        doc_messages = None
        current_file = None
        llm_messages = Chan[ResponseStreamEvent]()
        
        def done_processing_llm_events(task: asyncio.Task):
            try:
                task.result()
            except Exception as e:
                logger.error("error sending delta", exc_info=e)

        async def process_llm_events():

            partial = ""
            content_element = None
            context_message = None

            async for evt in llm_messages:
                
                #await self.room.messaging.send_message(to=chat_with_participant, type="llm.event", message=evt)

                if evt.type == "response.content_part.added":
                    partial = ""
                    content_element = doc_messages.append_child(tag_name="message", attributes={
                        "text" : "",
                        "created_at" : datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),
                        "author_name" : self.room.local_participant.get_attribute("name"),
                    })

                    context_message = {
                        "role" : "assistant",
                        "content" : ""
                    }
                    chat_context.messages.append(context_message)
                 
                elif evt.type == "response.output_text.delta":
                    partial += evt.delta
                    content_element["text"] = partial
                    context_message["content"] = partial
                    
                elif evt.type == "response.output_text.done":
                    content_element = None

        llm_task = asyncio.create_task(process_llm_events())
        llm_task.add_done_callback(done_processing_llm_events)

        try:
            while True:
                
                while True:

                    received = await messages.recv()

                    chat_with_participant = None
                    for participant in self._room.messaging.get_participants():
                        if participant.id == received.from_participant_id:
                            chat_with_participant = participant
                            break
                        
                    if chat_with_participant == None:
                        logger.warning("participant does not have messaging enabled, skipping message")
                        continue
                
                    if current_file != chat_with_participant.get_attribute("current_file"):
                        logger.info(f"participant is now looking at {chat_with_participant.get_attribute("current_file")}")
                        current_file = chat_with_participant.get_attribute("current_file")
                        
                    if current_file != None:
                        chat_context.append_assistant_message(message=f"the user is currently viewing the file at the path: {current_file}")

                    elif current_file != None:
                        chat_context.append_assistant_message(message=f"the user is not current viewing any files")

                
                    if thread == None:
                        thread = await self.open_thread(path=path)
                        
                        for prop in thread.root.get_children():
                            
                            if prop.tag_name == "messages":

                                doc_messages = prop
                                
                                for element in doc_messages.get_children():

                                    if isinstance(element, Element):
                                        
                                        msg = element["text"]
                                        if element["author_name"] == self.room.local_participant.get_attribute("name"):
                                            chat_context.append_assistant_message(msg)
                                        else:
                                            chat_context.append_user_message(msg)
                        
                        if doc_messages == None:
                            raise Exception("thread was not properly initialized")


                    if received.type == "opened":
                        
                        if opened == False:
                            
                            opened = True
                            
                            await self.greet(path=path, chat_context=chat_context, participant=chat_with_participant, messages=doc_messages)

                    if received.type == "chat":
                        
                        if thread == None:

                            self.room.developer.log_nowait(type="thread is not open", data={})

                            break


                        text = received.message["text"]
                        

                        for participant in get_thread_participants(room=self._room, thread=thread):
                            # TODO: async gather
                            await self._room.messaging.send_message(to=participant, type="thinking", message={"thinking":True, "path": path})

                        if chat_with_participant.id == received.from_participant_id:
                            self.room.developer.log_nowait(type="llm.message", data={ "context" : chat_context.id, "participant_id" : self.room.local_participant.id, "participant_name" : self.room.local_participant.get_attribute("name"), "message" : { "content" : {  "role" : "user", "text" : received.message["text"] } } })
                    
                        
                            attachments = received.message.get("attachments", [])

                            for attachment in attachments:

                                chat_context.append_assistant_message(message=f"the user attached a file '{attachment["filename"]}' with the content: '{attachment["content"]}'")
                                

                            chat_context.append_user_message(message=text)
                                

                        # if user is typing, wait for typing to stop
                        while True:
                            
                            if chat_with_participant.id not in self._is_typing:
                                break
                        
                            await asyncio.sleep(.5)

                        if messages.empty() == True:
                            break
            

                try:
                
                    toolkits = [
                        *self._toolkits,
                        *await self.get_required_tools(participant_id=chat_with_participant.id)
                    ]

                    thread_context = ChatThreadContext(
                        chat=chat_context,
                        thread=thread,
                        toolkits=toolkits,
                    )

                    await self.init_thread_context(thread_context=thread_context)

                    def handle_event(evt):
                        llm_messages.send_nowait(evt)
                        
                    try:
                        response = await self._llm_adapter.next(
                            context=chat_context,
                            room=self._room,
                            toolkits=toolkits,
                            tool_adapter=self._tool_adapter,
                            event_handler=handle_event
                        )
                    except Exception as e:
                        logger.error("An error was encountered", exc_info=e)
                        await self._send_and_save_chat(messages=doc_messages, to=chat_with_participant, path=path, id=str(uuid.uuid4()), text="There was an error while communicating with the LLM. Please try again later.")
    
                    
                finally:
                    for participant in get_thread_participants(room=self._room, thread=thread):
                        # TODO: async gather
                        await self._room.messaging.send_message(to=participant, type="thinking", message={"thinking":False, "path" : path})

                   
        finally:

            self.room.developer.log_nowait(type="chatbot.thread.ended", data={ "path" : path })
    
            llm_messages.close()

            if thread != None:
                await self.close_thread(path=path)
   

    def _get_message_channel(self, participant_id: str) -> Chan[RoomMessage]:
        if participant_id not in self._message_channels:
            chan = Chan[RoomMessage]()
            self._message_channels[participant_id] = chan

        chan = self._message_channels[participant_id]
        
        return chan
    
    async def stop(self):
        await super().stop()

        for thread in self._thread_tasks.values():
            thread.cancel()
        
        self._thread_tasks.clear()

    async def start(self, *, room):

        await super().start(room=room)
        
        await self.room.local_participant.set_attribute("empty_state_title", self._empty_state_title)

        def on_message(message: RoomMessage):

            messages = self._get_message_channel(participant_id=message.from_participant_id)
            if message.type == "chat" or message.type == "opened":
                messages.send_nowait(message)

                path = message.message["path"]
                logger.info(f"received message for thread {path}")
                
                if path not in self._thread_tasks or self._thread_tasks[path].cancelled:
                     
                    def thread_done(task: asyncio.Task):

                        self._message_channels.pop(message.from_participant_id)
                        try:
                            task.result()
                        except Exception as e:
                            logger.error(f"The chat thread ended with an error {e}", exc_info=e)
                    
                    
                    task = asyncio.create_task(self._spawn_thread(messages=messages, path=path))
                    task.add_done_callback(thread_done)

                    self._thread_tasks[path] = task

            elif message.type == "typing":
                def callback(task: asyncio.Task):
                    try:
                        task.result()
                    except:
                        pass
                
                async def remove_timeout(id: str):
                    await asyncio.sleep(1)
                    self._is_typing.pop(id)

                if message.from_participant_id in self._is_typing:
                    self._is_typing[message.from_participant_id].cancel()

                timeout = asyncio.create_task(remove_timeout(id=message.from_participant_id))
                timeout.add_done_callback(callback)

                self._is_typing[message.from_participant_id] = timeout

        room.messaging.on("message", on_message)
        
        if self._auto_greet_message != None:
            def on_participant_added(participant:RemoteParticipant):
                
                # will spawn the initial thread
                self._get_message_channel(participant_id=participant.id)
               

            room.messaging.on("participant_added", on_participant_added)

        await room.messaging.enable()

