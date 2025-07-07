### MeshAgent Agents

The ``meshagent.agents`` package provides higher-level agent classes that orchestrate tools and tasks. The primary agents you will build will use the ``TaskRunner``, ``Worker``, and ``ChatBot`` agent types. These agents extend the base ``Agent`` and ``SingleRoomAgent`` classes which setup the fundamentals for working with Agents in MeshAgent Rooms.

Note that agents use LLMAdapters and ToolResponseAdapters to translate between language model calls and tool executions. They also use the ServiceHost to run. 

### Agent
The ``Agent`` base class handles static info such as the agent name, description, and requirements. This class is not used directly, but is the foundation for specialized agents. 

### SingleRoomAgent
The ``SingleRoomAgent``extends the ``Agent`` class, connects to a ``RoomClient``, and installs any declared schemas or toolkits when the agent starts up. All other MeshAgent Agent types extend the ``SingleRoomAgent`` class with additional functionality.

### TaskRunner
The ``TaskRunner`` agent is useful when you want to invoke an agent with a well-defined JSON schemas for input and output. This is important for running agents-as-tools or running remote tasks. Often you will define a ``TaskRunner`` and pass it to a ``ChatBot`` or ``VoiceBot`` as a tool for that agent to use. 

### Worker
The ``Worker`` is a queue-based ``SingleRoomAgent`` that processes queued messages with an LLM and optional tools. This is particularly helpful for running asynchronous jobs. With the ``Worker`` agent you can create a set of tasks that need to run in a Room and the ``Worker`` will execute all of the tasks in the queue. 

### ChatBot
The ``ChatBot`` is a conversational agent derived from the ``SingleRoomAgent``. It wires an LLMAdapter, optoinal tools, and manages chat threads for each user. This means multiple users can be in the same room interacting with a chat agent, but each user will have private messages with the agent. Check out the [Build and Deploy a Chat Agent](https://docs.meshagent.com/agents/standard/buildanddeploychatbot) example to learn how to create a simple Chat Agent without tools then add built-in MeshAgent tools and custom tools to the agent.

---
### Learn more about MeshAgent on our website or check out the docs for additional examples!

**Website**: [www.meshagent.com](https://www.meshagent.com/)

**Documentation**: [docs.meshagent.com](https://docs.meshagent.com/)

---