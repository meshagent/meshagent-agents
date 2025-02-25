from meshagent.agents.agent import  AgentCallContext
from meshagent.api.room_server_client import RoomClient
from meshagent.api.schema_document import Document
from meshagent.api import RoomClient, RequiredToolkit, RequiredSchema, Requirement
#from meshagent.tools.document_tools import DocumentToolkit
from typing import Optional
import asyncio
import logging
from meshagent.agents.hosting import TaskRunner

logging.basicConfig()
logger = logging.getLogger("writer_agent")
logger.setLevel(logging.INFO)

class WriterContext:
    def __init__(self, room: RoomClient, document: Document, call_context: AgentCallContext, path: str):
        self._room = room
        self._document = document
        self._call_context = call_context 
        self._path = path

    @property
    def room(self) -> RoomClient:
        return self._room
    
    @property
    def path(self) -> str:
        return self._path
    
    @property
    def document(self) -> Document:
        return self._document
    
    @property
    def call_context(self) -> AgentCallContext:
        return self._call_context
    

# writes to a path, the document must already exist, optionally create the document if it does not exist
class Writer(TaskRunner):
    def __init__(self, name: str, input_schema:dict, output_schema:dict, create: bool = True, title: Optional[str] = None, description: Optional[str] = None, requires: Optional[list[Requirement]] = None, supports_tools: Optional[bool] = None):
        super().__init__(
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            title=title,
            requires=requires,
            supports_tools=supports_tools
        )
        
        self._create = create

        if "path" not in input_schema["properties"]:
            raise Exception("input schema must include a 'path' property")
        
    def pop_path(self, arguments: dict):
        return arguments["path"]
    
    async def write(self, writer_context: WriterContext, arguments: dict) -> dict:
        pass
 
    async def ask(self, *, context: AgentCallContext, arguments: dict):
    
        path = arguments["path"]
        
        client = context.room
        document = await client.sync.open(path=path, create=self._create)

        writer_context = WriterContext(
            room=client,
            document=document,
            call_context=context,
            path=path
        )
    
        try:
            return await self.write(writer_context, arguments)
        finally:
            # TODO: need a way to wait for changes to be synchronized
            await asyncio.sleep(1)
            await client.sync.close(path=path)