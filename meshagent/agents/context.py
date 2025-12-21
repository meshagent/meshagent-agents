from typing import Optional
from copy import deepcopy
from meshagent.api import RoomClient
from meshagent.tools import Toolkit
from meshagent.api.participant import Participant
import base64

import uuid


class AgentChatContext:
    def __init__(
        self,
        *,
        messages: Optional[list[dict]] = None,
        system_role: Optional[str] = None,
        previous_messages: Optional[list[dict]] = None,
        previous_response_id: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        if messages is None:
            messages = list[dict]()
        self._messages = messages.copy()
        self._system_role = system_role

        if previous_messages is None:
            previous_messages = list[dict]()

        self._previous_response_id = previous_response_id
        self._previous_messages = previous_messages
        self._deferred_messages = []

        self.instructions = instructions

    @property
    def messages(self):
        return self._messages

    @property
    def system_role(self):
        return self._system_role

    @property
    def deferred_messages(self):
        return self._deferred_messages

    @property
    def previous_messages(self):
        return self._previous_messages

    @property
    def previous_response_id(self):
        return self._previous_response_id

    def track_response(self, id: str):
        self._previous_response_id = id
        self._previous_messages.extend(self.messages)
        self.messages.clear()
        self.messages.extend(self._deferred_messages)
        self._deferred_messages.clear()

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
        m = {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64.b64encode(data).decode()}",
                },
            ],
        }
        self.messages.append(m)
        return m

    def append_file_message(
        self, *, filename: str, mime_type: str, data: bytes
    ) -> dict:
        m = {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": f"data:{mime_type or 'text/plain'};base64,{base64.b64encode(data).decode()}",
                }
            ],
        }
        self.messages.append(m)
        return m

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
            self.instructions = (self.instructions or "") + "\n" + plan

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

    def append_assistant_message(self, message: str, deferred: bool = False) -> dict:
        if deferred:
            m = {"role": "assistant", "content": message}
            self.deferred_messages.append(m)
            return m
        else:
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

    def copy(self) -> "AgentChatContext":
        return AgentChatContext(
            messages=deepcopy(self.messages), system_role=self._system_role
        )

    def to_json(self) -> dict:
        return {
            "messages": self.messages,
            "system_role": self.system_role,
            "previous_messages": self.previous_messages,
            "previous_response_id": self.previous_response_id,
        }

    @staticmethod
    def from_json(json: dict):
        return AgentChatContext(
            messages=json["messages"],
            system_role=json.get("system_role", None),
            previous_messages=json.get("previous_messages", None),
            previous_response_id=json.get("previous_response_id", None),
        )


class AgentCallContext:
    def __init__(
        self,
        *,
        chat: AgentChatContext,
        room: RoomClient,
        toolkits: Optional[list[Toolkit]] = None,
        caller: Optional[Participant] = None,
        on_behalf_of: Optional[Participant] = None,
    ):
        self._room = room
        if toolkits is None:
            toolkits = list[Toolkit]()
        self._toolkits = toolkits
        self._chat = chat
        self._caller = caller
        self._on_behalf_of = on_behalf_of

    @property
    def toolkits(self):
        return self._toolkits

    @property
    def chat(self):
        return self._chat

    @property
    def caller(self):
        return self._caller

    @property
    def on_behalf_of(self):
        return self._on_behalf_of

    @property
    def room(self):
        return self._room
