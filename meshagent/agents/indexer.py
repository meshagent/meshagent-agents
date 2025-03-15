from meshagent.agents import TaskRunner, RequiredToolkit, SingleRoomAgent
from meshagent.tools import Toolkit, Tool, ToolContext
from meshagent.api.room_server_client import TextDataType, VectorDataType, FloatDataType, IntDataType
from openai import AsyncOpenAI
from typing import Optional
from meshagent.api.chan import Chan

import hashlib
import chonkie
import asyncio
import logging

from functools import wraps

import os

# TODO: install chonkie, chonkie[semantic], openai

from concurrent.futures import ThreadPoolExecutor, as_completed




def _async_debounce(wait):
    def decorator(func):
        task = None

        @wraps(func)
        async def debounced(*args, **kwargs):
            nonlocal task

            async def call_func():
                await asyncio.sleep(wait)
                await func(*args, **kwargs)

            if task and not task.done():
                task.cancel()

            task = asyncio.create_task(call_func())
            return task

        return debounced

    return decorator

logger = logging.getLogger("indexer")
logger.setLevel(logging.INFO)

class Chunk:
    def __init__(self, *, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

class Chunker:

    async def chunk(self, *, text: str, max_length: Optional[int] = None) -> list[Chunk]:
        pass

class ChonkieChunker(Chunker):
    def __init__(self, chunker: Optional[chonkie.BaseChunker] = None):
        super().__init__()

        if chunker == None:
            chunker = chonkie.SemanticChunker()

        self._chunker = chunker

    async def chunk(self, *, text: str, max_length: Optional[int] = None) -> list[Chunk]:
        chunks = await asyncio.to_thread(self._chunker.chunk, text=text)
        mapped = []
        for chunk in chunks:
            mapped.append(Chunk(text=chunk.text, start=chunk.start_index, end=chunk.end_index))
        return mapped
    

class Embedder:
    def __init__(self, *, size: int, max_length: int):
        self.size = size
        self.max_length = max_length
    
    async def embed(self, *, text: str) -> list[float]:
        pass

class OpenAIEmbedder(Embedder):
    def __init__(self, *, size: int, max_length: int, model: str,  openai: Optional[AsyncOpenAI] = None, ):
        if openai == None:
            openai = AsyncOpenAI()
    
        self._openai = openai
        self._model = model

        super().__init__(size=size, max_length=max_length)

    
    async def embed(self, *, text):
        return (await self._openai.embeddings.create(input=text, model=self._model, encoding_format="float")).data[0].embedding
    


class RagTool(Tool):
    def __init__(self, *, name = "rag_search", table: str, title = "RAG search", description = "perform a RAG search", rules = None, thumbnail_url = None, embedder: Optional[Embedder] = None):
        
        self.table = table

        super().__init__(
            name=name,
            input_schema={
                "type":"object",
                "additionalProperties" : False,
                "required" : [
                    "query"
                ],
                "properties" : {
                    "query" : {
                        "type" : "string"
                    }
                }
            },
            title=title,
            description=description,
            rules=rules, thumbnail_url=thumbnail_url)
        
        self._embedder = embedder

    async def execute(self, context: ToolContext, query: str):
        
        if self._embedder == None:
            results = await context.room.database.search(
                table=self.table,
                text=query,
                limit=10
            )
        else:
            embedding = await self._embedder.embed(text=query)
            results = await context.room.database.search(
                table=self.table,
                text=query,
                vector=embedding,
                limit=10
            )

        results = list(map(lambda r: f"from {r["url"]}: {r["text"]}", results))

        return {
            "results" : results
        }
        

def open_ai_embedding_3_small():
    return OpenAIEmbedder(model="text-embedding-3-small", max_length=8191, size=1536)

def open_ai_embedding_3_large():
    return OpenAIEmbedder(model="text-embedding-3-large", max_length=8191, size=3072)

def open_ai_embedding_ada_2():
    return OpenAIEmbedder(model="text-embedding-ada-002", max_length=8191, size=1536)


class RagToolkit(Toolkit):
    def __init__(self, table: str, embedder:Optional[Embedder] = None):

        if embedder == None:
            embedder = open_ai_embedding_3_large()

        super().__init__(
            name="meshagent.rag",
            title="RAG",
            description="Searches against an index",
            tools=[
                RagTool(table=table, embedder=embedder)
            ]
        )


class FileIndexEvent:
    def __init__(self, *, path: str, deleted: bool):
        self.path = path
        self.deleted = deleted

class StorageIndexer(SingleRoomAgent):

    def __init__(
        self,
        *,
        name,
        title=None,
        description=None,
        requires=None,
        labels = None, 
        chunker: Optional[Chunker] = None,
        embedder:Optional[Embedder] = None,
        table: str = "storage_index",
      ):
        super().__init__(name=name, title=title, description=description, requires=requires, labels=labels)

        self._chan = Chan[FileIndexEvent]()
        
        if chunker == None:
            chunker = ChonkieChunker()

        if embedder == None:
            embedder = open_ai_embedding_3_large()

        self.chunker = chunker
        self.embedder = embedder
        self.table = table
        self._vector_index_created = False
        self._fts_created = False

    async def read_file(self, *, path: str) -> str | None:
        pass
    
    @_async_debounce(10)
    async def refresh_index(self):

        self.room.developer.log_nowait(type="indexer.rebuild", data={})

        indexes = await self.room.database.list_indexes(table=self.table)

        logger.info(f"existing indexes {indexes}")
        
        for index in indexes:

            if "embedding" in index["columns"]:
                self._vector_index_created = True
                
            if "text" in index["columns"]:
                self._fts_created  = True

        if self._vector_index_created == False:
            try:
                logger.info("attempting to create embedding index")
                await self.room.database.create_vector_index(table=self.table, column="embedding", replace=False)
                self._vector_index_created = True
            except Exception as e:
                # Will fail if there aren't enough rows
                pass

        if self._fts_created == False:
            try:
                logger.info("attempting to create fts index")
                await self.room.database.create_full_text_search_index(table=self.table, column="text", replace=False)
                self._fts_created = True
            except Exception as e:
                # Will fail if there aren't enough rows
                pass

        if self._fts_created == True or self._vector_index_created == True:
            logger.info("optimizing existing index")
            await self.room.database.optimize(table=self.table)
        

    async def start(self, *, room):
        await super().start(room=room)

        room.storage.on("file.updated", self._on_file_updated)
        room.storage.on("file.deleted", self._on_file_deleted)

        await room.database.create_table_with_schema(
            name=self.table,
            schema={
                "url" : TextDataType(),
                "text" : TextDataType(),
                "embedding" : VectorDataType(
                    size=self.embedder.size,
                    element_type=FloatDataType()
                ),
                "sha" : TextDataType(),
            },
            mode="create_if_not_exists",
            data=None
        )

      
        def index_task(task: asyncio.Task):

            try:
                result = task.result()
            except Exception as e:
                logger.error("Index task failed", exc_info=e)
            

        self._index_task = asyncio.create_task(self._indexer())
        self._index_task.add_done_callback(index_task)
    
    async def stop(self):
        await super().stop()
        await self._chan.close()
        
           
    async def _indexer(self):

        async for e in self._chan:

            try:
                if e.deleted:

                    # todo: consider using sql_alchemy or a library to do the escaping
                    def escape_sql_string(value):
                        if not isinstance(value, str):
                            raise TypeError("Input must be a string")
                        return value.replace("'", "''")

                    self.room.developer.log_nowait(type="indexer.delete", data={"path": e.path})
                    await self.room.database.delete(table=self.table, where=f"url='{escape_sql_string(e.path)}'")
                    

                else:

                    self.room.developer.log_nowait(type="indexer.index", data={"path": e.path})
                    

                    async def lookup_or_embed(*, sha: str, text: str) -> list[float]:

                        # if we already indexed this chunk, lets use the existing embedding instead of generating a new one
                        results = await self.room.database.search(
                            table=self.table,
                            where={
                                "sha" : sha,
                            },
                            limit=1
                        )

                        if len(results) != 0:
                            logger.info(f"chunk found from {e.path} {sha}, reusing embedding")
                            return results[0]["embedding"]
                            
                        logger.info(f"chunk not found from {e.path} {sha}, generating embedding")
                                
                        return await self.embedder.embed(text=text)

                    basename = os.path.basename(e.path)

                    chunk_sha = hashlib.sha256(basename.encode("utf-8")).hexdigest()

                    rows = []
                    # let's make the filename it's own chunk
                    rows.append(
                        {
                            "url" : e.path,
                            "text" : basename,
                            "sha" : chunk_sha,
                            "embedding" : await lookup_or_embed(sha=chunk_sha, text=basename)
                        }
                    )
                        
                    
                    text = await self.read_file(path=e.path)
                    if text != None:
                        
                        # the content will be transformed into additional chunks
                        for chunk in await self.chunker.chunk(text=text, max_length = self.embedder.max_length):
                            logger.info(f"processing chunk from {e.path}: {chunk.start}")
                            chunk_sha = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                            rows.append(
                                {
                                    "url" : e.path,
                                    "text" : chunk.text,
                                    "embedding" : await lookup_or_embed(sha=chunk_sha, text=chunk.text),
                                    "sha" : chunk_sha,
                                }
                            )
                    await self.room.database.merge(table=self.table, on="sha", records=rows)
                    await self.refresh_index()

                    
                
            except Exception as e:
                logger.error("error while indexing", exc_info=e)


    def _on_file_deleted(self, path: str):
        self._chan.send_nowait(FileIndexEvent(path=path, deleted=True))
        
    def _on_file_updated(self, path: str):
        self._chan.send_nowait(FileIndexEvent(path=path, deleted=False))

        
    


class SiteIndexer(TaskRunner):

    def __init__(self,
         *, 
        name,
        chunker: Optional[Chunker] = None,
        embedder:Optional[Embedder] = None,
        title=None,
        description=None,
        requires=None,
        supports_tools = None,
        labels: Optional[list[str]] = None
     
    ):
        
        if chunker == None:
            chunker = ChonkieChunker()

        if embedder == None:
            embedder = open_ai_embedding_3_large()

        self.chunker = chunker
        self.embedder = embedder

        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=[
                RequiredToolkit(
                    name="meshagent.firecrawl",
                    tools=[
                        "firecrawl_queue"
                    ]
                ),
            ],
            supports_tools=supports_tools,
            input_schema={
                "type" : "object",
                "required" : [
                    "queue", "table", "url"
                ],
                "additionalProperties" : False,
                "properties" : {
                    "queue" : {
                        "type" : "string",
                        "description" : "default: firecrawl"
                    },
                    "table" : {
                        "type" : "string",
                        "description" : "default: index"
                    },
                    "url" : {
                        "type" : "string",
                        "description" : "default: index"
                    }
                }
            },
            output_schema={
                "type" : "object",
                "required" : [],
                "additionalProperties" : False,
                "properties" : {},
            },
            labels=labels
        )


    async def ask(self, *, context, arguments):
        
        queue = arguments["queue"]
        table = arguments["table"]
        url = arguments["url"]

        tables = await context.room.database.list_tables()

        exists = False
        try:
            exists = tables.index(table)
        except ValueError:
            pass


        async def lookup_or_embed(*, sha: str, text: str) -> list[float]:

            # if we already indexed this chunk, lets use the existing embedding instead of generating a new one
            if exists:

                results = await self.room.database.search(
                    table=self.table,
                    where={
                        "sha" : sha,
                    },
                    limit=1
                )


                if len(results) != 0:
                    logger.info(f"chunk found from {url} {sha}, reusing embedding")
                    return results[0]["embedding"]
                
            logger.info(f"chunk not found from {url} {sha}, generating embedding")
                    
            return await self.embedder.embed(text=text)
            
        
        async def crawl():
            logger.info(f"starting to crawl: {url}")
            await context.room.agents.invoke_tool(
                toolkit="meshagent.firecrawl",
                tool="firecrawl_queue",
                arguments={
                    "url" : url,
                    "queue": queue,
                    "limit" : 100
                })
            
            logger.info(f"done with crawl: {url}")
            await context.room.queues.send(name=queue, message={ "done" : True })
            
        def crawl_done(task: asyncio.Task):
            try:
                task.result()
            except Exception as e:
                logger.error("crawl failed", exc_info=e)


        crawl_task = asyncio.create_task(crawl())
        crawl_task.add_done_callback(crawl_done)
        
        rows = []

        id = 0
        
        while True:
            message = await context.room.queues.receive(name=queue, create=True, wait=True)
           
            if message == None:
                break

            if message.get("type", None) == "crawl.completed":
                break
            
            if "data" in message:
                for data in message["data"]:
                    try:
                        url : str  = data["metadata"]["url"]
                        text : str = data["markdown"]
                        title : str  = data["metadata"]["title"]
                        title_sha : str  = hashlib.sha256(text.encode("utf-8")).hexdigest()

                        logger.info(f"processing crawled page: {url}")
                        
                        # let's make the title it's own chunk
                        rows.append(
                                {
                                    "id" : id,
                                    "url" : url,
                                    "text" : title,
                                    "sha" : title_sha,
                                    "embedding" : await lookup_or_embed(sha=title_sha, text=title)
                                }
                            )
                            
                        id = id + 1
                        
                        # the content will be transformed into additional chunks
                        for chunk in await self.chunker.chunk(text=text, max_length = self.embedder.max_length):
                            logger.info(f"processing chunk from {url}: {chunk.text}")
                            chunk_sha = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                            rows.append(
                                {
                                    "id" : id,
                                    "url" : url,
                                    "text" : chunk.text,
                                    "embedding" : await lookup_or_embed(sha=chunk_sha, text=chunk.text)
                                }
                            )
                            
                            id = id + 1

                    except Exception as e:
                        logger.error(f"failed to process: {url}", exc_info=e)

        logger.info(f"saving crawl: {url}")
            
        await context.room.database.create_table_with_schema(
            name=table,
            schema={
                "id" : IntDataType(),
                "url" : TextDataType(),
                "text" : TextDataType(),
                "embedding" : VectorDataType(
                    size=self.embedder.size,
                    element_type=FloatDataType()
                ),
                "sha" : TextDataType(),
            },
            mode="overwrite",
            data=rows
        )

        if len(rows) > 255:
            await context.room.database.create_vector_index(table=table, column="embedding")

        await context.room.database.create_full_text_search_index(table=table, column="text")

        return {}