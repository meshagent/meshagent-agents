from meshagent.api import RoomClient, Requirement, RoomException, Participant
import json
from typing import Optional
import asyncio


async def generate_json(
    *,
    on_behalf_of: Optional[Participant] = None,
    room: RoomClient,
    prompt: Optional[str] = None,
    output_schema: dict,
    requires: Optional[list[Requirement]] = None,
) -> dict:
    # make sure agent is in the room before proceeding
    agent = None
    for i in range(10):
        agents = await room.agents.list_agents()
        for a in agents:
            if a.name == "meshagent.schema_planner":
                agent = a
                break

        if agent is not None:
            break

        await asyncio.sleep(1)

    if agent is None:
        raise RoomException(
            "unable to locate required agent (meshagent.schema_planner)"
        )

    if prompt is None:
        prompt = f"ask me a series of questions to completely fill out the data structure described by this JSON schema ${json.dumps(output_schema)}. If you need to ask multiple questions, try to include all of them in a single form."

    return await room.agents.ask(
        on_behalf_of=on_behalf_of,
        agent="meshagent.schema_planner",
        requires=requires,
        arguments={
            "prompt": prompt,
            "output_schema": output_schema,
        },
    )
