from typing import Optional
from copy import deepcopy
from meshagent.api import RoomClient
from meshagent.tools.toolkit import Toolkit
from meshagent.api.participant import Participant

import uuid


class AgentChatContext:
    def __init__(
        self,
        *,
        messages: Optional[list[dict]] = None,
        system_role: Optional[str] = None,
        previous_messages: Optional[list[dict]] = None,
        previous_response_id: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        if messages is None:
            messages = list[dict]()
        self._messages = messages.copy()
        if system_role is None:
            system_role = "system"
        self._system_role = system_role

        if previous_messages is None:
            previous_messages = list[dict]()

        self._previous_response_id = previous_response_id
        self._previous_messages = previous_messages

    @property
    def messages(self):
        return self._messages

    @property
    def system_role(self):
        return self._system_role

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

    def append_rules(self, rules: list[str]):
        system_message = None

        for m in self.messages:
            if m["role"] == self.system_role:
                system_message = m
                break

        if system_message is None:
            system_message = {"role": self.system_role, "content": ""}
            self.messages.insert(0, system_message)

        plan = f"Rules:\n-{'\n-'.join(rules)}\n"
        system_message["content"] = system_message["content"] + plan

    def append_assistant_message(self, message: str) -> None:
        self.messages.append({"role": "assistant", "content": message})

    def append_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def append_user_image(self, url: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
                ],
            }
        )

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
