from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MCPHeader(BaseModel):
    name: str
    value: str


class MCPServerConfig(BaseModel):
    server_label: str
    server_url: str | None = None
    allowed_tools: list[str] | None = None
    authorization: str | None = None
    headers: list[MCPHeader] | None = None
    require_approval: Literal["always", "never"] | None = None
    always_require_approval: list[str] | None = None
    never_require_approval: list[str] | None = None
    openai_connector_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_headers(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        headers = value.get("headers")
        if not isinstance(headers, dict):
            return value

        normalized = dict(value)
        normalized["headers"] = [
            {"name": str(key), "value": str(header_value)}
            for key, header_value in headers.items()
        ]
        return normalized


class MCPToolkitClientOptions(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)
