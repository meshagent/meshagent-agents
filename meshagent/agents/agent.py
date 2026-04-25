from collections.abc import AsyncIterable
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
    ToolContentSpec,
    ToolDescription,
    ToolkitDescription,
    StorageEntry,
)
from meshagent.tools import (
    Toolkit,
    LocalRoomTool,
    ToolContext,
)
from meshagent.tools.hosting import _RemoteToolkitWrapper, _start_hosted_toolkit

from meshagent.api.room_server_client import RoomClient
from .context import AgentSessionContext
import logging
import asyncio
import warnings

logger = logging.getLogger("agent")
_legacy_init_chat_context_warned: set[type] = set()


class AgentException(RoomException):
    pass


class RemoteRoomTool(LocalRoomTool):
    def __init__(
        self,
        *,
        room: RoomClient,
        toolkit_name: str,
        name,
        input_schema,
        output_spec: Optional[ToolContentSpec] = None,
        output_schema: Optional[dict] = None,
        title=None,
        description=None,
        rules=None,
        thumbnail_url=None,
        pricing: Optional[str] = None,
        participant_id: Optional[str] = None,
        on_behalf_of_id: Optional[str] = None,
        defs: Optional[dict] = None,
        strict: Optional[bool] = None,
    ):
        self._toolkit_name = toolkit_name
        self._participant_id = participant_id
        self._on_behalf_of_id = on_behalf_of_id
        if input_schema is None:
            input_schema = {
                "type": "object",
                "additionalProperties": True,
                "properties": {},
            }

        super().__init__(
            room=room,
            name=name,
            input_schema=input_schema,
            output_spec=output_spec,
            output_schema=output_schema,
            title=title,
            description=description,
            rules=rules,
            thumbnail_url=thumbnail_url,
            pricing=pricing,
            defs=defs,
            strict=True if strict is None else strict,
        )

    async def execute(self, context: ToolContext, **kwargs):
        result = await self.room.agents.invoke_tool(
            toolkit=self._toolkit_name,
            tool=self.name,
            participant_id=self._participant_id,
            on_behalf_of_id=self._on_behalf_of_id,
            input=kwargs,
            caller_context=context.caller_context,
        )

        if isinstance(result, AsyncIterable):
            raise RoomException(
                f"tool '{self._toolkit_name}.{self.name}' returned an iterable stream, which is not supported by RemoteRoomTool"
            )

        return result


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


class SingleRoomAgent:
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[Requirement]] = None,
        annotations: Optional[list[str]] = None,
    ) -> None:
        if name is not None:
            logger.warning(
                "agent name property is deprecated and will be removed in a future version %s",
                name,
            )

        if title is None:
            title = name
        if description is None:
            description = ""
        if requires is None:
            requires = []
        if annotations is None:
            annotations = []

        self._name = name
        self._title = title
        self._description = description
        self._requires = requires
        self._annotations = annotations
        self._room = None
        self._exposed_toolkits: list[Toolkit] = []
        self._hosted_exposed_toolkits: list[_RemoteToolkitWrapper] = []

    def get_requirements(self) -> list[Requirement]:
        return self._requires

    @property
    def description(self) -> str:
        return self._description

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def requires(self) -> list[Requirement]:
        return self._requires

    @property
    def annotations(self) -> list[str]:
        return self._annotations

    async def init_session(self) -> AgentSessionContext:
        legacy_initializer = type(self).init_chat_context
        if legacy_initializer is not SingleRoomAgent.init_chat_context:
            cls = type(self)
            if cls not in _legacy_init_chat_context_warned:
                warnings.warn(
                    (
                        f"{cls.__name__}.init_chat_context() is deprecated and will be removed in a future release. "
                        "Override init_session() instead."
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                _legacy_init_chat_context_warned.add(cls)
            return await legacy_initializer(self)
        return AgentSessionContext()

    # Backwards compatibility for existing subclasses overriding init_chat_context.
    async def init_chat_context(self) -> AgentSessionContext:
        warnings.warn(
            "init_chat_context() is deprecated and will be removed in a future release. Use init_session() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AgentSessionContext()

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "requires": [requirement.to_json() for requirement in self.requires],
            "annotations": self.annotations,
        }

    async def get_exposed_toolkits(self) -> list[Toolkit]:
        return []

    @staticmethod
    def resolve_runtime_api_key(*, room: RoomClient) -> str | None:
        token = room.protocol.token
        if token is None:
            return None

        normalized_token = token.strip()
        return normalized_token or None

    def bind_runtime_credentials(self, *, room: RoomClient) -> None:
        del room

    async def start(self, *, room: RoomClient) -> None:
        if self._room is not None:
            raise RoomException("agent is already started")

        self._room = room
        self.bind_runtime_credentials(room=room)

        await self.install_requirements()

        self._exposed_toolkits = await self.get_exposed_toolkits()
        self._hosted_exposed_toolkits = []
        try:
            for toolkit in self._exposed_toolkits:
                hosted_toolkit = await _start_hosted_toolkit(
                    room=room,
                    toolkit=toolkit,
                )
                self._hosted_exposed_toolkits.append(hosted_toolkit)
        except Exception:
            for hosted_toolkit in reversed(self._hosted_exposed_toolkits):
                await hosted_toolkit.stop()
            self._hosted_exposed_toolkits = []
            self._exposed_toolkits = []
            self._room = None
            raise

    async def stop(self) -> None:
        for hosted_toolkit in reversed(self._hosted_exposed_toolkits):
            await hosted_toolkit.stop()

        self._hosted_exposed_toolkits = []
        self._exposed_toolkits = []
        self._room = None

    @property
    def room(self) -> RoomClient:
        return self._room

    @property
    def name(self) -> str:
        room = self._room
        if room is not None:
            room_name = room.local_participant.get_attribute("name")
            if isinstance(room_name, str) and room_name.strip() != "":
                return room_name

        if isinstance(self._name, str) and self._name.strip() != "":
            return self._name

        raise RoomException("agent name is only available after the agent starts")

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
                    await self._room.storage.upload(
                        path=f".schemas/{requirement.name}.json",
                        overwrite=True,
                        data=json.dumps(requirement.schema.to_json()).encode(),
                    )

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

    async def get_required_toolkits(self, context: ToolContext) -> list[Toolkit]:
        tool_target = context.caller
        if context.on_behalf_of is not None:
            tool_target = context.on_behalf_of

        toolkits_by_name = dict[str, ToolkitDescription]()
        toolkits_by_participant = dict[str, list[ToolkitDescription]]()
        toolkits = list[Toolkit]()

        visible_tools = await self._room.agents.list_toolkits(
            participant_id=tool_target.id
        )

        for toolkit_description in visible_tools:
            toolkits_by_name[toolkit_description.name] = toolkit_description

        for required_toolkit in self.requires:
            if isinstance(required_toolkit, RequiredToolkit):
                toolkit = None
                if required_toolkit.participant_name is None:
                    toolkit = toolkits_by_name.get(required_toolkit.name, None)
                else:
                    if required_toolkit.participant_name not in toolkits_by_participant:
                        toolkits_by_participant[
                            required_toolkit.participant_name
                        ] = await self._room.agents.list_toolkits(
                            participant_name=required_toolkit.participant_name
                        )

                    for tk in toolkits_by_participant[
                        required_toolkit.participant_name
                    ]:
                        if tk.name == required_toolkit.name:
                            toolkit = tk
                            break

                if toolkit is None:
                    if context.on_behalf_of is not None:
                        raise RoomException(
                            f"unable to get toolkit {required_toolkit.name} on behalf of {context.on_behalf_of}"
                        )
                    else:
                        raise RoomException(
                            f"unable to get toolkit {required_toolkit.name} for caller {context.caller.id}"
                        )

                remote_room_tools = list[RemoteRoomTool]()

                if required_toolkit.tools is None:
                    for tool_description in toolkit.tools:
                        tool = RemoteRoomTool(
                            room=self.room,
                            on_behalf_of_id=tool_target.id,
                            toolkit_name=toolkit.name,
                            name=tool_description.name,
                            description=tool_description.description,
                            input_schema=tool_description.input_schema,
                            output_spec=tool_description.output_spec,
                            output_schema=tool_description.output_schema,
                            title=tool_description.title,
                            thumbnail_url=tool_description.thumbnail_url,
                            participant_id=tool_target.id,
                            defs=tool_description.defs,
                            pricing=tool_description.pricing,
                            strict=tool_description.strict,
                        )
                        remote_room_tools.append(tool)

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

                        tool = RemoteRoomTool(
                            room=self.room,
                            on_behalf_of_id=tool_target.id,
                            toolkit_name=toolkit.name,
                            name=tool_description.name,
                            description=tool_description.description,
                            input_schema=tool_description.input_schema,
                            output_spec=tool_description.output_spec,
                            output_schema=tool_description.output_schema,
                            title=tool_description.title,
                            thumbnail_url=tool_description.thumbnail_url,
                            participant_id=tool_description.participant_id,
                            defs=tool_description.defs,
                            pricing=tool_description.pricing,
                            strict=tool_description.strict,
                        )
                        remote_room_tools.append(tool)

                required_toolkit_instance = Toolkit(
                    name=toolkit.name,
                    title=toolkit.title,
                    description=toolkit.description,
                    thumbnail_url=toolkit.thumbnail_url,
                    room=self.room,
                    tools=remote_room_tools,
                )
                toolkits.append(required_toolkit_instance)

        return toolkits
