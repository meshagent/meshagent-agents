from .agent import SingleRoomAgent, AgentChatContext
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomException, RoomClient, RemoteParticipant
from meshagent.tools import Toolkit
from .adapter import LLMAdapter, ToolResponseAdapter
import asyncio
from typing import Optional
import logging
from meshagent.tools import MultiToolkit
import urllib

logging.basicConfig()
logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)


# todo: thread should stop when participant stops?

class ChatBot(SingleRoomAgent):
    def __init__(self, *, name, title = None, description = None, requires = None,  llm_adapter: LLMAdapter, tool_adapter: Optional[ToolResponseAdapter] = None, toolkits: Optional[list[Toolkit]] = None, rules : Optional[list[str]] = None, auto_greet_prompt : Optional[str] = None, auto_greet_message : Optional[str] = None,  empty_state_title : Optional[str] = None, labels: Optional[str] = None):
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
        self._auto_greet_prompt = auto_greet_prompt
        self._auto_greet_message = auto_greet_message
        
        if empty_state_title == None:
            empty_state_title = "How can I help you?"
        self._empty_state_title = empty_state_title


    async def greet(self, *, chat_context: AgentChatContext, messages: Chan[RoomMessage], participant: RemoteParticipant):

        if self._auto_greet_prompt != None:
            messages.send_nowait(RoomMessage(from_participant_id=participant.id, type="chat",  message={"text": self._auto_greet_prompt }))
        
        if self._auto_greet_message != None:
            chat_context.append_user_message(self._auto_greet_message)
            
            await self.room.messaging.send_message(to=RemoteParticipant(id=participant.id), type="chat", message={ "text" : self._auto_greet_message  })


    async def finalize_toolkits(self, toolkits) -> list[Toolkit]:

        toaster = None
        
        for toolkit in toolkits:

            if toolkit.name == "meshagent.ui":

                for tool in toolkit.tools:

                    if tool.name == "show_toast":

                        toaster = tool

        if toaster != None:

            def multi_tool(toolkit: Toolkit):
                if toaster in toolkit.tools:
                    return toolkit
                
                return MultiToolkit(required=[ toaster ], base_toolkit=toolkit )

            toolkits = list(map(multi_tool, toolkits))
        
        return toolkits

    async def _spawn_thread(self, participant_id: str, messages: Chan[RoomMessage]):

        
        chat_context = await self.init_chat_context()
        

        chat_context.append_rules(
            rules=[
                *self._rules,
                "think step by step",
           ]
        )

        opened = False
        chat_with_participant = None

        for participant in self._room.messaging.get_participants():
            if participant.id == participant_id:
                chat_with_participant = participant
                break

        if chat_with_participant == None:
            raise RoomException(f"caller did not have messaging turned on")
        
        messaging = self._room.messaging

        current_file = None
        
        step_schema = {
            "type" : "object",
            "required" : ["text","finished"],
            "additionalProperties" : False, 
            "description" : "execute a step",
            "properties" : {
                "text" : {
                    "description" : "a reply to the user or status to display during an intermediate step",
                    "type" : "string"
                },
                "finished" : {
                    "description" : "whether the agent has finished answering the user's last message. you MUST set this to true if there are no more tool calls to be made or you are stuck in a loop.",
                    "type" : "boolean"
                }
            }
        }

        installed = False

        while True:
            
            while True:

             

                received = await messages.recv()

                

                if current_file != chat_with_participant.get_attribute("current_file"):
                    logger.info(f"participant is now looking at {chat_with_participant.get_attribute("current_file")}")
                    current_file = chat_with_participant.get_attribute("current_file")
                    
                if current_file != None:
                    chat_context.append_assistant_message(message=f"the user is currently viewing the file at the path: {current_file}")

                elif current_file != None:
                    chat_context.append_assistant_message(message=f"the user is not current viewing any files")



                if installed == False:
                    installed = True
                    try:
                        await self.install_requirements(participant_id=participant_id)
                    except Exception as e:
                        self.room.developer.log_nowait("error", { "text" : f"unable to install requirements: {e}" })

                        error = "I was unable to install the tools I require to operate in the room, I may not function properly."
                        chat_context.append_user_message(message=error)
                        await self._room.messaging.send_message(
                            to=chat_with_participant,
                            type="chat",
                            message={
                                "text": error
                            }
                        )

                if received.type == "opened":
                    
                    
                    if opened == False:
                        
                        opened = True
                        
                        await self.greet(chat_context=chat_context, participant=chat_with_participant, messages=messages)

                if received.type == "chat":


                    await self._room.messaging.send_message(to=chat_with_participant, type="thinking", message={"thinking":True})
                
                    if chat_with_participant.id == received.from_participant_id:
                        self.room.developer.log_nowait(type="llm.message", data={ "context" : chat_context.id, "participant_id" : self.room.local_participant.id, "participant_name" : self.room.local_participant.get_attribute("name"), "message" : { "content" : {  "role" : "user", "text" : received.message["text"] } } })
                
                        text = received.message["text"]
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
                while True:

                    toolkits = [
                        *self._toolkits,
                        *await self.get_required_tools(participant_id=chat_with_participant.id)
                    ]

                    toolkits = await self.finalize_toolkits(toolkits)

                    response = await self._llm_adapter.next(
                        context=chat_context,
                        room=self._room,
                        toolkits=toolkits,
                        tool_adapter=self._tool_adapter,
                        output_schema=step_schema,
                    )

                    text = response["text"]
                    
                    
                    if response["finished"] or len(toolkits) == 0:
                        await self._room.messaging.send_message(
                            to=chat_with_participant,
                            type="chat",
                            message={
                                "text": text
                            }
                        )
                        break
                    else:
                        await self._room.messaging.send_message(
                            to=chat_with_participant,
                            type="status",
                            message={
                                "text": text
                            }
                        )
                        chat_context.append_user_message(message="proceed to the next step if you are ready")
                
            finally:

                await self._room.messaging.send_message(to=chat_with_participant, type="thinking", message={"thinking":False})

   

    def _get_message_channel(self, participant_id: str) -> Chan[RoomMessage]:
        if participant_id not in self._message_channels:
            chan = Chan[RoomMessage]()
            self._message_channels[participant_id] = chan

            def thread_done(task: asyncio.Task):

                self._message_channels.pop(participant_id)
                try:
                    task.result()
                    logger.info("ending chat thread")
                except Exception as e:
                    logger.error("chat thread error", exc_info=e)

            task = asyncio.create_task(self._spawn_thread(participant_id=participant_id, messages=chan))
            task.add_done_callback(thread_done)
        
        chan = self._message_channels[participant_id]
        
        return chan
    
    async def start(self, *, room):

        await super().start(room=room)


        await self.room.local_participant.set_attribute("empty_state_title", self._empty_state_title)

        def on_message(message: RoomMessage):
            messages = self._get_message_channel(participant_id=message.from_participant_id)
            if message.type == "chat" or message.type == "opened":
                messages.send_nowait(message)
    
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
        
        if self._auto_greet_prompt != None or self._auto_greet_message != None:
            def on_participant_added(participant:RemoteParticipant):
                
                # will spawn the initial thread
                self._get_message_channel(participant_id=participant.id)
               

            room.messaging.on("participant_added", on_participant_added)

        await room.messaging.enable()

