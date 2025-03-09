from meshagent.agents import TaskRunner, RequiredToolkit
from meshagent.tools import Toolkit, Tool, ToolContext
from meshagent.api.room_server_client import TextDataType, VectorDataType, FloatDataType, IntDataType
from openai import AsyncOpenAI
from typing import Optional
import hashlib
import chonkie
import asyncio
import logging

# TODO: install chonkie, chonkie[semantic], openai

from concurrent.futures import ThreadPoolExecutor, as_completed

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

                results = await context.room.database.search(where={
                    "table" : table,
                    "sha" : sha,
                    "limit" : 1
                })

                if len(results) != 0:
                    logger.info(f"chunk found from {url} {sha}: {text}, reusing embedding")
                    return results[0]["embedding"]
                
            logger.info(f"chunk not found from {url} {sha}: {text}, generating embedding")
                    
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