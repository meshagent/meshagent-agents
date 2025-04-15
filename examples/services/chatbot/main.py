from meshagent.api import RequiredToolkit
from meshagent.agents.hosting import RemoteAgentServer
from meshagent.agents.chat import ChatBot
from meshagent.openai import OpenAIResponsesAdapter
import asyncio
import os


class GenericChatBot(ChatBot):
    def __init__(self):
        super().__init__(
            name="meshagent.chatbot",
            title="chatbot",
            description="an simple chatbot",
            rules=[
            ],
            llm_adapter = OpenAIResponsesAdapter(
                parallel_tool_calls=None
            ),
            labels=[ "chatbot" ],
            requires=[
                RequiredToolkit(
                    name="meshagent.markitdown",
                    tools=[ "markitdown_from_user", "markitdown_from_file" ]
                ),
            ]
        )
        

async def server():

    remote_agent_server = RemoteAgentServer(
        cls=GenericChatBot,
        path="/webhook",
        validate_webhook_secret=False,
        port=int(os.getenv("MESHAGENT_PORT"))
    )
    await remote_agent_server.run()


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.get_event_loop().run_until_complete(server())