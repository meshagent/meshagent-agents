from .agent import TaskRunner, AgentCallContext
from meshagent.api.chan import Chan
from meshagent.api import RoomMessage, RoomException, RoomClient, RemoteParticipant
from meshagent.tools import Toolkit
from .adapter import LLMAdapter, ToolResponseAdapter
import asyncio
from typing import Optional
import json

import logging

logging.basicConfig()
logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)


# todo: thread should stop when participant stops?

class Worker(TaskRunner):
    def __init__(self, *, name, title = None, description = None, requires = None, llm_adapter: LLMAdapter, tool_adapter:  Optional[ToolResponseAdapter] = None, toolkits: Optional[list[Toolkit]] = None, rules : Optional[list[str]] = None, supports_tools: bool = True):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            input_schema={
                "type" : "object",
                "additionalProperties" : False,
                "required" : [ "prompt", "queue" ],
                "properties" : {
                    "prompt" : {
                        "type" : "string"
                    },
                    "queue" : {
                        "type" : "string"
                    }
                }
            },
            output_schema=None,
            supports_tools=supports_tools
        )

        if toolkits == None:
            toolkits = []

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter

        self._message_channel = Chan[RoomMessage]()

        self._room : RoomClient | None = None
        self._toolkits = toolkits

        if rules == None:
            rules = []

        self._rules = rules


    async def ask(self, *, context: AgentCallContext, arguments: dict):
        
        queue = arguments["queue"]
        prompt = arguments["prompt"]

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
                    "description" : "whether the agent has finished answering the user's last message, also should be set to true if we get stuck in a loop, or if the user did not make a request",
                    "type" : "boolean"
                }
            }
        }
        
        # todo: add graceful exit 

        while True:

            message = await self.room.queues.receive(name=queue, create=True, wait=True)
            if message != None:
                
                # for each message, create a new chat context

                chat_context = await self.init_chat_context()

            
                chat_context.append_rules(
                    rules=[
                        *self._rules,
                    ]
                )
                
                chat_context.append_user_message(message=prompt)
                chat_context.append_user_message(message=json.dumps(message))
                
                try:
                    while True:

                        toolkits = [
                            *self._toolkits,
                            *context.toolkits,
                            *await self.get_required_tools(participant_id=self.room.local_participant.id)
                        ]
                        response = await self._llm_adapter.next(
                            context=chat_context,
                            room=self._room,
                            toolkits=toolkits,
                            tool_adapter=self._tool_adapter,
                            output_schema=step_schema,
                        )

                        if response["finished"] or len(toolkits) == 0:
                            break
                        else:
                            chat_context.append_user_message(message="proceed to the next step if you are ready")
                    
                except Exception as e:

                    logger.error(f"Failed to process a message {message}", exc_info=e)
