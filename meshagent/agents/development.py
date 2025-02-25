from .agent import SingleRoomAgent
from meshagent.api import websocket_protocol, RoomClient, MeshSchema, RoomException
import asyncio
import signal
import json

from aiohttp import web


async def connect_development_agent(*, room_name:str, agent: SingleRoomAgent):
    
    async with RoomClient(
        protocol = websocket_protocol(participant_name=agent.name, room_name=room_name, role="agent")) as room:
        
        await agent.start(room=room)

        try:
            
            term = asyncio.Future()

            def clean_termination(signal, frame):
                term.set_result(True)
            
            signal.signal(signal.SIGTERM, clean_termination)
            signal.signal(signal.SIGABRT, clean_termination)

            await asyncio.wait([asyncio.ensure_future(room.protocol.wait_for_close()), term], return_when=asyncio.FIRST_COMPLETED)

        finally:

            await agent.stop()

