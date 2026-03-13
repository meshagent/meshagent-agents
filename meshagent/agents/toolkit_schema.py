from typing import Any, Optional

from meshagent.tools import ToolkitBuilder
from meshagent.tools.strict_schema import ensure_strict_json_schema


def build_tools_property_schema(
    *, toolkit_builders: list[ToolkitBuilder]
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    toolkit_config_schemas: list[dict[str, Any]] = []
    defs: dict[str, Any] = {}

    for builder in toolkit_builders:
        schema = ensure_strict_json_schema(builder.type.model_json_schema())
        builder_defs = schema.get("$defs")
        if isinstance(builder_defs, dict):
            for key, value in builder_defs.items():
                defs[key] = value

        toolkit_config_schemas.append(schema)

    if len(toolkit_config_schemas) == 0:
        return None, defs

    if len(toolkit_config_schemas) == 1:
        items_schema: dict[str, Any] = toolkit_config_schemas[0]
    else:
        items_schema = {
            "type": "object",
            "anyOf": toolkit_config_schemas,
        }

    return (
        {
            "anyOf": [
                {
                    "type": "array",
                    "items": items_schema,
                },
                {"type": "null"},
            ],
        },
        defs,
    )
