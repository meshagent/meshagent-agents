from __future__ import annotations

from typing import Literal
from urllib.parse import urlencode

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
    use_proxy_secret: str | None = None

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


class MeshagentProxyConfig(BaseModel):
    api_url: str
    api_key: str
    user: str | None = None


class MCPToolkitClientOptions(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)
    meshagent_proxy_config: MeshagentProxyConfig | None = None


def apply_mcp_proxy_config(
    *,
    server: MCPServerConfig,
    proxy_config: MeshagentProxyConfig | None,
    authorization_mode: Literal["header", "token"] = "header",
) -> MCPServerConfig:
    secret_id = server.use_proxy_secret.strip() if server.use_proxy_secret else None
    if (
        server.server_url is None
        or secret_id is None
        or secret_id == ""
        or proxy_config is None
    ):
        return server

    api_url = proxy_config.api_url.strip().rstrip("/")
    api_key = proxy_config.api_key.strip()
    if api_url == "" or api_key == "":
        raise ValueError("meshagent_proxy_config requires api_url and api_key")

    query = {
        "url": server.server_url,
        "secret-id": secret_id,
    }
    if proxy_config.user is not None and proxy_config.user.strip() != "":
        query["user"] = proxy_config.user.strip()

    headers = list(server.headers or [])
    authorization: str | None = None
    if authorization_mode == "header":
        headers.append(MCPHeader(name="Authorization", value=f"Bearer {api_key}"))
    else:
        authorization = api_key

    return server.model_copy(
        update={
            "server_url": f"{api_url}/proxy-request?{urlencode(query)}",
            "headers": headers,
            "authorization": authorization,
        }
    )
