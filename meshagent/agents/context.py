from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
from meshagent.api import RoomException
from meshagent.api import RoomClient
from meshagent.tools import Toolkit
from meshagent.api.participant import Participant

import uuid


@dataclass(frozen=True, slots=True)
class SessionUsage:
    model: str
    usage: dict[str, float]
    context_window_used: int | None = None
    context_window_size: int | None = None


SessionUsageCallback = Callable[[SessionUsage], None]


class AgentSessionContext:
    def __init__(
        self,
        *,
        messages: Optional[list[dict]] = None,
        system_role: Optional[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[dict] = None,
        turn_count: Optional[float] = None,
        last_usage: SessionUsage | None = None,
        usage_callback: SessionUsageCallback | None = None,
    ):
        self.id = str(uuid.uuid4())
        if messages is None:
            messages = list[dict]()
        self._messages = messages.copy()
        self._system_role = system_role
        self._metadata = metadata or {}

        self.instructions = instructions

        self.turn_count = turn_count or 0
        self.last_usage = last_usage
        self._usage_callback = usage_callback

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def set_usage_callback(self, callback: SessionUsageCallback | None) -> None:
        self._usage_callback = callback

    def emit_usage_updated(self, usage: SessionUsage) -> None:
        self.last_usage = usage
        callback = self._usage_callback
        if callback is not None:
            callback(usage)

    async def __aenter__(self) -> "AgentSessionContext":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb
        await self.close()

    @property
    def metadata(self):
        return self._metadata

    @property
    def messages(self):
        return self._messages

    @property
    def system_role(self):
        return self._system_role

    @property
    def supports_images(self) -> bool:
        return False

    @property
    def supports_files(self) -> bool:
        return False

    @property
    def supports_realtime_audio(self) -> bool:
        return False

    def replace_rules(self, rules: list[str]):
        system_message = None

        if self.system_role is not None:
            for m in self.messages:
                if m.get("role") == self.system_role:
                    system_message = m
                    break

            if system_message is None:
                system_message = {"role": self.system_role, "content": ""}
                self.messages.insert(0, system_message)

        if len(rules) > 0:
            plan = "\n".join(rules)
        else:
            plan = ""

        if self.system_role is not None:
            system_message["content"] = plan
        else:
            self.instructions = plan

    def append_image_message(self, *, mime_type: str, data: bytes) -> dict:
        del mime_type
        del data
        raise RoomException("this chat context does not support image inputs")

    def append_image_url(self, *, url: str) -> dict:
        del url
        raise RoomException("this chat context does not support image URL inputs")

    def append_file_message(
        self, *, filename: str, mime_type: str, data: bytes
    ) -> dict:
        del filename
        del mime_type
        del data
        raise RoomException("this chat context does not support file inputs")

    def append_file_url(self, *, url: str, filename: str | None = None) -> dict:
        del url
        del filename
        raise RoomException("this chat context does not support file URL inputs")

    async def append_realtime_audio_chunk(
        self,
        *,
        mime_type: str,
        data: bytes,
        sample_rate: int | None = None,
        bitrate: int | None = None,
    ) -> None:
        del mime_type
        del data
        del sample_rate
        del bitrate
        raise RoomException("this chat context does not support realtime audio inputs")

    async def commit_realtime_audio(self) -> None:
        raise RoomException("this chat context does not support realtime audio inputs")

    def append_rules(self, rules: list[str]):
        system_message = None

        if self.system_role is not None:
            for m in self.messages:
                if m["role"] == self.system_role:
                    system_message = m
                    break

            if system_message is None:
                system_message = {"role": self.system_role, "content": ""}
                self.messages.insert(0, system_message)

        if len(rules) > 0:
            plan = "\n".join(rules)
        else:
            plan = ""

        if self.system_role is not None:
            system_message["content"] = system_message["content"] + plan
        else:
            instructions = self.instructions

            if len(plan) > 0:
                if instructions is not None:
                    instructions = instructions + "\n" + plan
                else:
                    instructions = plan
            self.instructions = instructions

    def get_system_instructions(self) -> None | str:
        if self.system_role is not None:
            system_message = None

            for m in self.messages:
                if m["role"] == self.system_role:
                    content = m.get("content")
                    if content is not None:
                        if system_message is None:
                            system_message = content
                        else:
                            system_message += "\n" + content

            return system_message

        else:
            return self.instructions

    def append_assistant_message(self, message: str) -> dict:
        m = {"role": "assistant", "content": message}
        self.messages.append(m)
        return m

    def append_user_message(self, message: str) -> dict:
        m = {"role": "user", "content": message}
        self.messages.append(m)
        return m

    def append_user_image(self, url: str) -> dict:
        m = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
            ],
        }
        self.messages.append(m)
        return m

    def copy(self) -> "AgentSessionContext":
        return self.__class__(
            messages=deepcopy(self.messages),
            system_role=self._system_role,
            turn_count=self.turn_count,
            last_usage=deepcopy(self.last_usage),
            usage_callback=self._usage_callback,
        )

    def to_json(self) -> dict:
        return {
            "messages": self.messages,
            "system_role": self.system_role,
        }

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            messages=json["messages"],
            system_role=json.get("system_role", None),
        )


class TaskContext:
    def __init__(
        self,
        *,
        session: AgentSessionContext,
        room: RoomClient,
        toolkits: Optional[list[Toolkit]] = None,
        caller: Optional[Participant] = None,
        on_behalf_of: Optional[Participant] = None,
    ):
        self._room = room
        if toolkits is None:
            toolkits = list[Toolkit]()
        self._toolkits = toolkits
        self._session = session
        self._caller = caller
        self._on_behalf_of = on_behalf_of

    async def __aenter__(self) -> "TaskContext":
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._session.__aexit__(exc_type, exc, tb)

    @property
    def toolkits(self):
        return self._toolkits

    @property
    def session(self):
        return self._session

    @property
    def caller(self):
        return self._caller

    @property
    def on_behalf_of(self):
        return self._on_behalf_of

    @property
    def room(self):
        return self._room
