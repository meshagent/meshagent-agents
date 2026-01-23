from typing import Optional
import json
from meshagent.api.room_server_client import (
    RoomException,
    RequiredToolkit,
    Requirement,
    RequiredSchema,
    RequiredTable,
)
from meshagent.api import (
    ToolDescription,
    ToolkitDescription,
    StorageEntry,
)
from meshagent.tools import (
    Toolkit,
    Tool,
    ToolContext,
)
from meshagent.api.room_server_client import RoomClient
from .context import AgentChatContext
import logging
import asyncio

logger = logging.getLogger("agent")


class AgentException(RoomException):
    pass


class RoomTool(Tool):
    def __init__(
        self,
        *,
        toolkit_name: str,
        name,
        input_schema,
        title=None,
        description=None,
        rules=None,
        thumbnail_url=None,
        participant_id: Optional[str] = None,
        on_behalf_of_id: Optional[str] = None,
        defs: Optional[dict] = None,
    ):
        self._toolkit_name = toolkit_name
        self._participant_id = participant_id
        self._on_behalf_of_id = on_behalf_of_id

        super().__init__(
            name=name,
            input_schema=input_schema,
            title=title,
            description=description,
            rules=rules,
            thumbnail_url=thumbnail_url,
            defs=defs,
        )

    async def execute(self, context, **kwargs):
        return await context.room.agents.invoke_tool(
            toolkit=self._toolkit_name,
            tool=self.name,
            participant_id=self._participant_id,
            on_behalf_of_id=self._on_behalf_of_id,
            arguments=kwargs,
        )


class Agent:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[Requirement]] = None,
        labels: Optional[list[str]] = None,
        skills_dirs: Optional[list[str]] = None,
    ):
        if name is not None:
            logger.warning(
                f"agent name property is deprecated and will be removed in a future version {name}"
            )

        if title is None:
            title = name
        self._title = title
        if description is None:
            description = ""

        self._description = description
        if requires is None:
            requires = []

        self._requires = requires

        if labels is None:
            labels = []

        self._labels = labels
        self._skills_dirs = None

    def get_requirements(self) -> list[Requirement]:
        return self._requires

    @property
    def description(self):
        return self._description

    @property
    def title(self):
        return self._title

    @property
    def requires(self):
        return self._requires

    @property
    def labels(self):
        return self._labels

    async def init_chat_context(self) -> AgentChatContext:
        return AgentChatContext()

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "requires": list(map(lambda x: x.to_json(), self.requires)),
            "labels": self.labels,
        }


async def install_required_table(*, room: RoomClient, table: RequiredTable):
    await room.database.create_table_with_schema(
        name=table.name,
        mode="create_if_not_exists",
        schema=table.schema,
        namespace=table.namespace,
    )

    indexes = await room.database.list_indexes(
        table=table.name, namespace=table.namespace
    )

    def index_exists(column: str):
        for i in indexes:
            if column in i.columns:
                return True

        return False

    for vi in table.vector_indexes or []:
        if not index_exists(vi):
            try:
                await room.database.create_vector_index(
                    table=table.name,
                    column=vi,
                    namespace=table.namespace,
                    replace=True,
                )
            except Exception as e:
                logger.warning(f"unable to create vector index {e}", exec_info=e)

    for ti in table.full_text_search_indexes or []:
        if not index_exists(ti):
            try:
                await room.database.create_full_text_search_index(
                    table=table.name,
                    column=ti,
                    namespace=table.namespace,
                    replace=True,
                )
            except Exception as e:
                logger.warning(
                    f"unable to create full text search index {e}",
                    exec_info=e,
                )

    for si in table.scalar_indexes or []:
        if not index_exists(si):
            try:
                await room.database.create_scalar_index(
                    table=table.name,
                    column=si,
                    namespace=table.namespace,
                    replace=True,
                )
            except Exception as e:
                logger.warning(f"unable to create scalar index {e}", exec_info=e)

    logger.info(f"optimizing table {table.name} in {table.namespace}")

    # TODO: use index_stats to determine when indexes need to be updated
    await room.database.optimize(table=table.name, namespace=table.namespace)


class SingleRoomAgent(Agent):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires=None,
        labels: Optional[list[str]] = None,
    ):
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            labels=labels,
        )
        self._room = None

    async def start(self, *, room: RoomClient) -> None:
        if self._room is not None:
            raise RoomException("agent is already started")

        self._room = room

        await self.install_requirements()

    async def stop(self) -> None:
        self._room = None
        pass

    @property
    def room(self):
        return self._room

    @property
    def name(self):
        return self._room.local_participant.get_attribute("name")

    async def install_requirements(self, participant_id: Optional[str] = None):
        schemas_by_name = dict[str, StorageEntry]()
        toolkits_by_name = dict[str, ToolkitDescription]()

        async def refresh_schemas():
            schemas = await self._room.storage.list(path=".schemas")

            for schema in schemas:
                schemas_by_name[schema.name] = schema

        async def refresh_tools():
            toolkits_by_name.clear()

            visible_tools = await self._room.agents.list_toolkits(
                participant_id=participant_id
            )
            for toolkit_description in visible_tools:
                toolkits_by_name[toolkit_description.name] = toolkit_description

        installed = False

        await refresh_tools()
        await refresh_schemas()

        builtin_agents_url = "http://localhost:8080"

        for requirement in self.get_requirements():
            if isinstance(requirement, RequiredToolkit):
                if requirement.name == "ui":
                    # TODO: maybe requirements can be marked as non installable?
                    continue

                if requirement.name not in toolkits_by_name:
                    if not requirement.callable:
                        if requirement.timeout == 0:
                            logger.info(
                                f"{self.name} not waiting for toolkit {requirement.name}"
                            )
                            continue

                        async with asyncio.timeout(requirement.timeout):
                            logger.info(
                                f"{self.name} waiting for toolkit {requirement.name}"
                            )

                            while requirement.name not in toolkits_by_name:
                                await refresh_tools()
                                await asyncio.sleep(1)

                    else:
                        installed = True
                        logger.info(
                            f"{self.name} calling required tool into room {requirement.name}"
                        )

                        if requirement.name.startswith(
                            "https://"
                        ) or requirement.name.startswith("http://"):
                            url = requirement.name
                        else:
                            url = f"{builtin_agents_url}/toolkits/{requirement.name}"

                        await self._room.agents.make_call(
                            url=url, name=requirement.name, arguments={}
                        )

            elif isinstance(requirement, RequiredSchema):
                if requirement.schema is not None:
                    logger.info(
                        f"{self.name} installing required schema {requirement.name} from json"
                    )
                    handle = await self._room.storage.open(
                        path=f".schemas/{requirement.name}.json", overwrite=True
                    )
                    await self._room.storage.write(
                        handle=handle,
                        data=json.dumps(requirement.schema.to_json()).encode(),
                    )
                    await self._room.storage.close(handle=handle)

                elif requirement.name not in schemas_by_name:
                    installed = True

                    if not requirement.callable:
                        if requirement.timeout == 0:
                            logger.info(
                                f"{self.name} not waiting for schema {requirement.name}"
                            )
                            continue

                        async with asyncio.timeout(requirement.timeout):
                            logger.info(
                                f"{self.name} waiting for schema {requirement.name}"
                            )

                            while requirement.name not in schemas_by_name:
                                await refresh_schemas()
                                await asyncio.sleep(1)

                    else:
                        logger.info(
                            f"{self.name} installing required schema {requirement.name} from registry"
                        )

                        if requirement.name.startswith(
                            "https://"
                        ) or requirement.name.startswith("http://"):
                            url = requirement.name
                        else:
                            url = f"{builtin_agents_url}/schemas/{requirement.name}"

                        await self._room.agents.make_call(
                            url=url, name=requirement.name, arguments={}
                        )

            elif isinstance(requirement, RequiredTable):
                logger.info(
                    f"ensuring required table exists {requirement.name} in {requirement.namespace}"
                )

                await install_required_table(room=self.room, table=requirement)

            else:
                raise RoomException("unsupported requirement")

        if installed:
            await asyncio.sleep(5)

    async def get_toolkits(
        self, context: ToolContext, remote_toolkits: list[RequiredToolkit]
    ):
        tool_target = context.caller
        if context.on_behalf_of is not None:
            tool_target = context.on_behalf_of

        toolkits_by_name = dict[str, ToolkitDescription]()

        toolkits = list[Toolkit]()

        visible_tools = await self._room.agents.list_toolkits(
            participant_id=tool_target.id
        )

        for toolkit_description in visible_tools:
            toolkits_by_name[toolkit_description.name] = toolkit_description

        for required_toolkit in remote_toolkits:
            if isinstance(required_toolkit, RequiredToolkit):
                toolkit = toolkits_by_name.get(required_toolkit.name, None)
                if toolkit is None:
                    if context.on_behalf_of is not None:
                        raise RoomException(
                            f"unable to get toolkit {required_toolkit.name} on behalf of {context.on_behalf_of}"
                        )
                    else:
                        raise RoomException(
                            f"unable to get toolkit {required_toolkit.name} for caller {context.caller.id}"
                        )

                room_tools = list[RoomTool]()

                if required_toolkit.tools is None:
                    for tool_description in toolkit.tools:
                        tool = RoomTool(
                            on_behalf_of_id=tool_target.id,
                            toolkit_name=toolkit.name,
                            name=tool_description.name,
                            description=tool_description.description,
                            input_schema=tool_description.input_schema,
                            title=tool_description.title,
                            thumbnail_url=tool_description.thumbnail_url,
                            participant_id=tool_target.id,
                            defs=tool_description.defs,
                        )
                        room_tools.append(tool)

                else:
                    tools_by_name = dict[str, ToolDescription]()
                    for tool_description in toolkit.tools:
                        tools_by_name[tool_description.name] = tool_description

                    for required_tool in required_toolkit.tools:
                        tool_description = tools_by_name.get(required_tool, None)
                        if tool_description is None:
                            raise RoomException(
                                f"unable to locate required tool {required_tool} in toolkit {required_toolkit.name}"
                            )

                        tool = RoomTool(
                            on_behalf_of_id=tool_target.id,
                            toolkit_name=toolkit.name,
                            name=tool_description.name,
                            description=tool_description.description,
                            input_schema=tool_description.input_schema,
                            title=tool_description.title,
                            thumbnail_url=tool_description.thumbnail_url,
                            participant_id=tool_target.id,
                            defs=tool_description.defs,
                        )
                        room_tools.append(tool)

                toolkits.append(
                    Toolkit(
                        name=toolkit.name,
                        title=toolkit.title,
                        description=toolkit.description,
                        thumbnail_url=toolkit.thumbnail_url,
                        tools=room_tools,
                    )
                )

        return toolkits

    async def get_required_toolkits(self, context: ToolContext) -> list[Toolkit]:
        return await self.get_toolkits(context, self.requires)
