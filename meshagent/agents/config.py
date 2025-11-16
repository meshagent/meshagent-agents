from pydantic import BaseModel
from typing import Optional


class RulesConfig(BaseModel):
    # Rules that apply for every client
    rules: Optional[list[str]] = None

    # Rules that only apply when participant's client attribute matches the given key
    client_rules: Optional[dict[str, list[str]]] = None

    @staticmethod
    def parse(text: str):
        rules = []
        client_rules = {}

        client = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue

            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("[") and line.endswith("]"):
                client = line.strip("[]")
                client_rules[client] = []
            else:
                if client is None:
                    rules.append(line)
                else:
                    client_rules[client].append(line)

        return RulesConfig(
            rules=rules,
            client_rules=client_rules,
        )
