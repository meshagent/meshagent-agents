
import logging
import asyncio
from typing import Optional

from meshagent.agents import TaskRunner
from meshagent.api.schema_document import Element,Text
from meshagent.api.room_server_client import RoomClient, MeshDocument
from meshagent.api.websocket_protocol import WebSocketClientProtocol

from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from meshagent.agents.agent import TaskRunner, AgentChatContext, AgentCallContext

class ListenerContext:
    def __init__(self, document: MeshDocument, room: RoomClient, call_context: AgentCallContext):
        self.document = document
        self.call_context = call_context
        self.room = room


# Notifies of new nodes or changed nodes in a document, the document must already exist
class Listener(TaskRunner):

    def __init__(self, *, name: str, wait_for_synchronize: bool = True, title: Optional[str] = None, description: Optional[str] = None):
        super().__init__(
            name=name,
            description=description,
            title=title,
            input_schema={
                "type" : "object",
                "required" : ["path"],
                "additionalProperties" : False,
                "properties" : {
                    "path" : {
                        "type" : "string",
                        "description" : "the path of the document to proofread"
                    }
                }
            },
            output_schema={
                "type" : "object",
                "additionalProperties" : False,
                "required" : [],
                "properties" : {
                }
            }
        )
        self.wait_for_synchronize = wait_for_synchronize
    
    async def on_listening_started(self, listener_context: ListenerContext):
        pass

    async def on_element_inserted(self, listener_context: ListenerContext, element: Element) -> bool:
        return False

    async def on_attribute_changed(self, listener_context: ListenerContext, element: Element, attribute: Optional[str]) -> bool:
        return False
    
    async def ask(self, *, context: AgentCallContext, arguments: dict):

        output_path = arguments["path"]
        room = context.room
        logger.info("Visitor connecting to %s", output_path)
        doc = await room.sync.open(path=output_path, create=True)
        try:
            listener_context = ListenerContext(
                document=doc,
                room=room,
                call_context=context,
            )
            logger.info("Visitor connected to %s", output_path)

            change_queue = list[Element | Text]()
            
            def append_children(node:Element):
                for child in node.get_children():
                    if child not in change_queue:
                        change_queue.append([child])
                    if isinstance(child, Element):
                        append_children(child)
            
            if self.wait_for_synchronize == False:
                change_queue.append([doc.root])
                append_children(doc.root)
            else:
                await doc.synchronized


            await self.on_listening_started(listener_context=listener_context)
                
            wait_for_changes = asyncio.Future()
            
            @doc.on("inserted")
            def on_inserted(e: Element):
                logger.info("element inserted %s", e.tag_name)
                if e not in change_queue:
                    change_queue.append([e])
                    append_children(e)
                
                if wait_for_changes.done() == False:
                    wait_for_changes.set_result(True)


            @doc.on("updated")
            def on_updated(e: Element, attribute: str):
                logger.info("element updated %s", e.tag_name)
                if e not in change_queue:                            
                    change_queue.append([e, attribute])
                    #
                    # append_children(e)
                
                if wait_for_changes.done() == False:
                    wait_for_changes.set_result(True)


            waiting_for_end = True
            while waiting_for_end:
                await wait_for_changes
            
                while len(change_queue) > 0:

                    change = change_queue.pop(0)
                    content = change[0]
                    if len(change) > 1:
                        done = await self.on_attribute_changed(listener_context, content, change[1])
                        if done:
                            waiting_for_end = False

                    else:
                        done = await self.on_element_inserted(listener_context, content)
                        if done:
                            waiting_for_end = False
                        
                    if content in change_queue:
                        change_queue.remove(content)

                    
                
                wait_for_changes = asyncio.Future()
        except Exception as e:
            logger.error("Failed to visit", exc_info=e)                
            raise

        finally:

            logger.info("vistor done")

            await asyncio.sleep(5)
            await room.sync.close(output_path)
            
            return {}
