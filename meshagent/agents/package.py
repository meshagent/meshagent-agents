from typing import Optional

# Fluent configuration for agent test / deploy
class AgentPackage:

    def __init__(self, *, name: str):
        self.name = name

    def chat_channel(self) -> 'AgentPackage':
        pass
    
    def mail_channel(self, email: str) -> 'AgentPackage':
        pass

    def queue_channel(self, queue: str) -> 'AgentPackage':
        pass
    
    def heartbeat(self, cron: str, prompt: str) -> 'AgentPackage':
        pass

    def files(self, source: str,  *, dest: Optional[str] = None, read_only: bool = False) -> 'AgentPackage':
        pass

    def skills(self, source: str, *, dest: Optional[str] = None, read_only: bool = False) -> 'AgentPackage':
        pass

    def instructions(self, source: str, *, dest:  Optional[str] = None, read_only: bool = False) -> 'AgentPackage':
        pass

    def enable_shell(self, *, image: str) -> 'AgentPackage':
        pass

    def enable_web_fetch(self) -> 'AgentPackage':
        pass

    def enable_web_search(self) -> 'AgentPackage':
        pass
    
    def enable_mcp(self) -> 'AgentPackage':
        pass

    def use_model(self, model: str) -> 'AgentPackage':
        pass

    def mount(self, source: str, *, dest: str) -> 'AgentPackage':
        pass
    
    # Pack files, instructions, and skills into an image, load the container into the room's registry, and deploy a service mounts each of the paths and runs the meshagent process cli with the meshagent/cli:default image
    # use --instructions to point at each of the mounted instructions
    # use --skill-dir to point at each of the mounted skill dirs
    # mounts should map to room storage mounts in the service, as well as shell and storage mounts on the cli flags
    # each channel should map to the respective channel flag on the cli command
    # heartbeat should map to the agent heartbeat in the agent spec (and a {agent_name}.heartbeat queue should be added)
    # email should map to the agent
    async def deploy(self, *, room: str) -> None:
        pass

    # Run a local agent, should mirror deploy in terms of setup, but use non container shell / local mounts instead of room / image mounts so it doesn't have to build images. Run a LLMProcess agent similar to how the CLI does, run a turn with the message, and return the result of the turn.
    async def test(self, prompt: str, attachments: Optional[list[str]] = None) -> str:
        pass
