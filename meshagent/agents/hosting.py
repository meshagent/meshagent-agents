import logging

from meshagent.api import websocket_protocol, RoomMessage
from meshagent.api.webhooks import WebhookServer, RoomStartedEvent, RoomEndedEvent, WebSocketClientProtocol, CallEvent
from meshagent.api.room_server_client import RoomClient
from meshagent.agents import SingleRoomAgent
from aiohttp import web
import asyncio

import logging
from typing import Callable, Optional, TypeVar

from .agent import TaskRunner

logger = logging.getLogger("hosting")
logger.setLevel(logging.INFO)


class RemoteTaskRunnerServer[T:TaskRunner](WebhookServer):
    def __init__(self, *, cls: Optional[T] = None, path: Optional[str] = None, app: Optional[web.Application] = None, host = None, port = None, webhook_secret = None, create_agent: Optional[Callable[[dict],TaskRunner]] = None, validate_webhook_secret: Optional[bool] = None):
        super().__init__(path=path, app=app, host=host, port=port, webhook_secret=webhook_secret, validate_webhook_secret=validate_webhook_secret)
        
        if create_agent == None:
            def default_create_agent(arguments: dict) -> TaskRunner:
                return cls(**arguments)
            
            create_agent = default_create_agent

        self._create_agent = create_agent

    async def _spawn(self, *, room_name: str, room_url: str, token: str, arguments: Optional[dict] = None):
        agent = self._create_agent(arguments=arguments)
        
        async def run():
            async with RoomClient(protocol=WebSocketClientProtocol(url= room_url, token=token)) as room:
    
                dismissed = asyncio.Future()

                def on_message(message: RoomMessage):
                    if message.type == "dismiss":
                        logger.info(f"dismissed by {message.from_participant_id}")
                        dismissed.set_result(True)

                room.messaging.on("message", on_message)
        
                await agent.start(room=room)

                done, pending = await asyncio.wait([
                    dismissed,
                    asyncio.ensure_future(room.protocol.wait_for_close())
                ], return_when=asyncio.FIRST_COMPLETED)
                

                await agent.stop()

        def on_done(task: asyncio.Task):
            try:
                result = task.result()
            except Exception as e:
                logger.error("agent encountered an error", exc_info=e)
                
        task = asyncio.create_task(run())
        task.add_done_callback(on_done)
    
    async def on_call(self, event: CallEvent):
        await self._spawn(room_name=event.room_name, room_url=event.room_url, token=event.token, arguments=event.arguments)


class RemoteAgentServer[T:SingleRoomAgent](WebhookServer):
    def __init__(self, *, cls: Optional[T] = None, path: Optional[str] = None, app: Optional[web.Application] = None,  host = None, port = None, webhook_secret = None, create_agent: Optional[Callable[[dict], SingleRoomAgent]] = None, validate_webhook_secret: Optional[bool] = None):
        super().__init__(path=path, app=app, host=host, port=port, webhook_secret=webhook_secret, validate_webhook_secret=validate_webhook_secret)
        
        if create_agent == None:
            def default_create_agent(arguments: dict) -> SingleRoomAgent:
                return cls(**arguments)
            
            create_agent = default_create_agent

        self._create_agent = create_agent
    
    async def _spawn(self, *, room_name: str, room_url: str, token: str, arguments: Optional[dict] = None):

        logger.info(f"room: {room_name} url: {room_url} token: {token} arguments: {arguments}")
        agent = self._create_agent(arguments=arguments)

        async def run():
            async with RoomClient(protocol=WebSocketClientProtocol(url=room_url, token=token)) as room:
    
                dismissed = asyncio.Future()

                def on_message(message: RoomMessage):
                    if message.type == "dismiss":
                        logger.info(f"dismissed by {message.from_participant_id}")
                        dismissed.set_result(True)

                room.messaging.on("message", on_message)
            
                await agent.start(room=room)

                done, pending = await asyncio.wait([
                    dismissed,
                    asyncio.ensure_future(room.protocol.wait_for_close())
                ], return_when=asyncio.FIRST_COMPLETED)
                
                await agent.stop()

        def on_done(task: asyncio.Task):
            try:
                result = task.result()
            except Exception as e:
                logger.error("agent encountered an error", exc_info=e)

        task = asyncio.create_task(run())
        task.add_done_callback(on_done)
        
    async def on_call(self, event: CallEvent):
        await self._spawn(room_name=event.room_name, room_url=event.room_url, token=event.token, arguments=event.arguments)
