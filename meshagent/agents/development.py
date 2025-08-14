from .agent import SingleRoomAgent
from meshagent.api import websocket_protocol, RoomClient
import asyncio
import signal
from warnings import deprecated


@deprecated("use ServiceHost and the cli to connect agents to the room")
async def connect_development_agent(*, room_name: str, agent: SingleRoomAgent):
    async with RoomClient(
        protocol=websocket_protocol(
            participant_name=agent.name, room_name=room_name, role="agent"
        )
    ) as room:
        await agent.start(room=room)

        try:
            term = asyncio.Future()

            def clean_termination(signal, frame):
                term.set_result(True)

            signal.signal(signal.SIGTERM, clean_termination)
            signal.signal(signal.SIGABRT, clean_termination)

            await asyncio.wait(
                [asyncio.create_task(room.protocol.wait_for_close()), term],
                return_when=asyncio.FIRST_COMPLETED,
            )

        finally:
            await agent.stop()
