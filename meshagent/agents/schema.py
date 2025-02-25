from copy import deepcopy

def validation_schema(description: str):
    return {
        "type" : "object",
        "description" : description,
        "required" : [ "is_valid", "message" ],
        "additionalProperties" : False,
        "properties" : {
            "is_valid" : {
                "type" : "boolean",
            },
            "message" : {
                "type" : "string",
            },
        }
    }

def prompt_schema(description: str):
    return {
        "type" : "object",
        "description" : description,
        "required" : [ "prompt" ],
        "additionalProperties" : False,
        "properties" : {
            "prompt" : {
                "description"  : "a prompt that will be used by the agent to create a response",
                "type" : "string",
            }
        }
    }

def no_arguments_schema(description: str):
    return {
        "description" : description,
        "type" : "object",
        "required" : [],
        "additionalProperties" : False,
        "properties" : {
        }
    }

def merge(*, schema: dict, additional_properties: dict) -> dict:
    schema = deepcopy(schema)

    for k,v in additional_properties.items():
        schema["required"].append(k)
        schema["properties"][k] = v

    return schema
