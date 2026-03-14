from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty


thread_list_schema = MeshSchema(
    root_tag_name="thread_list",
    elements=[
        ElementType(
            tag_name="thread_list",
            description="an index of chat threads",
            properties=[
                ChildProperty(
                    name="threads",
                    description="threads known to this chatbot",
                    child_tag_names=["thread"],
                )
            ],
        ),
        ElementType(
            tag_name="thread",
            description="a thread entry in the index",
            properties=[
                ValueProperty(
                    name="name",
                    description="a readable name for the thread",
                    type="string",
                ),
                ValueProperty(
                    name="path",
                    description="the sync path to the thread document",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the date the thread was created in ISO format",
                    type="string",
                ),
                ValueProperty(
                    name="modified_at",
                    description="the date the thread was last modified in ISO format",
                    type="string",
                ),
            ],
        ),
    ],
)

thread_schema = MeshSchema(
    root_tag_name="thread",
    elements=[
        ElementType(
            tag_name="thread",
            description="a thread of messages",
            properties=[
                ValueProperty(
                    name="name", description="name of the tread", type="string"
                ),
                ChildProperty(
                    name="properties",
                    description="the messages in the thread",
                    ordered=True,
                    child_tag_names=["members", "messages"],
                ),
            ],
        ),
        ElementType(
            tag_name="members",
            description="the members of this thread",
            properties=[
                ChildProperty(
                    name="items",
                    child_tag_names=["member"],
                    description="the messages in this thread",
                )
            ],
        ),
        ElementType(
            tag_name="messages",
            description="the messages of this thread",
            properties=[
                ValueProperty(
                    name="external_thread_id",
                    description="optional external backend thread id used for resume",
                    type="string",
                ),
                ChildProperty(
                    name="items",
                    child_tag_names=[
                        "message",
                        "exec",
                        "ui",
                        "reasoning",
                        "event",
                    ],
                    description="the messages in this thread",
                ),
            ],
        ),
        ElementType(
            tag_name="member",
            description="a member of this thread",
            properties=[
                ValueProperty(
                    name="name", description="the name of the member", type="string"
                ),
            ],
        ),
        ElementType(
            tag_name="file",
            description="a file attachment",
            properties=[
                ValueProperty(
                    name="path",
                    description="the path of the file in the room",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="image",
            description="an image attachment stored in the images database",
            properties=[
                ValueProperty(
                    name="status",
                    description="the image generation status (for example generating/completed/failed)",
                    type="string",
                ),
                ValueProperty(
                    name="status_detail",
                    description="optional human-readable status text for image generation",
                    type="string",
                ),
                ValueProperty(
                    name="width",
                    description="the pixel width of the generated image when known",
                    type="number",
                ),
                ValueProperty(
                    name="height",
                    description="the pixel height of the generated image when known",
                    type="number",
                ),
                ValueProperty(
                    name="id",
                    description="the image id in the room images table",
                    type="string",
                ),
                ValueProperty(
                    name="mime_type",
                    description="the image mime type",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the date that the image was generated in ISO format",
                    type="string",
                ),
                ValueProperty(
                    name="created_by",
                    description="the participant who generated the image",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="reasoning",
            description="a reasoning trace",
            properties=[
                ValueProperty(
                    name="turn_id",
                    description="the agent turn id associated with this reasoning trace",
                    type="string",
                ),
                ValueProperty(
                    name="summary",
                    description="a summary of the reasoning",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the time that the reasoning started",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="exec",
            description="a command execution",
            properties=[
                ValueProperty(
                    name="turn_id",
                    description="the agent turn id associated with this command execution",
                    type="string",
                ),
                ValueProperty(
                    name="command",
                    description="a command that was executed",
                    type="string",
                ),
                ValueProperty(
                    name="result",
                    description="the result of the command",
                    type="string",
                ),
                ValueProperty(
                    name="outcome",
                    description="the outcome of the command",
                    type="string",
                ),
                ValueProperty(
                    name="stderr",
                    description="the standard error text of the command",
                    type="string",
                ),
                ValueProperty(
                    name="stdout",
                    description="the standard output text of the command",
                    type="string",
                ),
                ValueProperty(
                    name="exit_code",
                    description="the exit code of the command",
                    type="number",
                ),
                ValueProperty(
                    name="pwd",
                    description="the working directory the command was executed in",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the time that the reasoning started",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="ui",
            description="custom user interface data",
            properties=[
                ValueProperty(
                    name="turn_id",
                    description="the agent turn id associated with this UI payload",
                    type="string",
                ),
                ValueProperty(
                    name="data",
                    description="raw data to be renderered, usually in JSON format",
                    type="string",
                ),
                ValueProperty(
                    name="renderer",
                    description="the renderer to use",
                    type="string",
                ),
                ValueProperty(
                    name="widget",
                    description="the type of widget",
                    type="string",
                ),
                ValueProperty(
                    name="width",
                    description="the width of widget",
                    type="number",
                ),
                ValueProperty(
                    name="height",
                    description="the height of widget",
                    type="number",
                ),
                ValueProperty(
                    name="created_at",
                    description="the time that the reasoning started",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="event",
            description="a structured event emitted by an agent backend",
            properties=[
                ValueProperty(
                    name="turn_id",
                    description="the agent turn id associated with this event",
                    type="string",
                ),
                ValueProperty(
                    name="id",
                    description="a unique id for this event",
                    type="string",
                ),
                ValueProperty(
                    name="source",
                    description="backend source, for example codex/openai/anthropic/chatkit",
                    type="string",
                ),
                ValueProperty(
                    name="name",
                    description="normalized event name",
                    type="string",
                ),
                ValueProperty(
                    name="kind",
                    description="high-level event category, for example turn/item/plan/diff/exec",
                    type="string",
                ),
                ValueProperty(
                    name="state",
                    description="lifecycle state, for example queued/in_progress/completed/failed/cancelled/info",
                    type="string",
                ),
                ValueProperty(
                    name="method",
                    description="source event or rpc method",
                    type="string",
                ),
                ValueProperty(
                    name="item_id",
                    description="item id for this event, when available",
                    type="string",
                ),
                ValueProperty(
                    name="item_type",
                    description="item type for this event, when available",
                    type="string",
                ),
                ValueProperty(
                    name="path",
                    description="optional path associated with the event, typically a file path",
                    type="string",
                ),
                ValueProperty(
                    name="summary",
                    description="short human readable summary",
                    type="string",
                ),
                ValueProperty(
                    name="headline",
                    description="primary event headline for compact rendering",
                    type="string",
                ),
                ValueProperty(
                    name="details",
                    description="optional newline-delimited detail lines for structured rendering",
                    type="string",
                ),
                ValueProperty(
                    name="preview",
                    description="optional structured preview body for compact rendering, for example file contents",
                    type="string",
                ),
                ValueProperty(
                    name="data",
                    description="serialized backend payload",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the date this event was emitted in ISO format",
                    type="string",
                ),
                ValueProperty(
                    name="updated_at",
                    description="the date this event was last updated in ISO format",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="reaction",
            description="a unicode reaction from a user on a message or specific attachment",
            properties=[
                ValueProperty(
                    name="user_name",
                    description="the display name of the user who reacted",
                    type="string",
                ),
                ValueProperty(
                    name="value",
                    description="the reaction as a single unicode character",
                    type="string",
                ),
                ValueProperty(
                    name="target",
                    description="reaction target type (message or attachment)",
                    type="string",
                ),
                ValueProperty(
                    name="attachment_ref",
                    description="the attachment element id when target is attachment",
                    type="string",
                ),
                ValueProperty(
                    name="created_at",
                    description="the date that the reaction was added in ISO format",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="message",
            description="a message sent in the conversation",
            properties=[
                ValueProperty(
                    name="id", description="the id of the message", type="string"
                ),
                ValueProperty(
                    name="turn_id",
                    description="the agent turn id associated with this message",
                    type="string",
                ),
                ValueProperty(
                    name="text", description="the text of the message", type="string"
                ),
                ValueProperty(
                    name="created_at",
                    description="the date that the message was sent in ISO format",
                    type="string",
                ),
                ValueProperty(
                    name="author_name",
                    description="the name of the author of the post",
                    type="string",
                ),
                ValueProperty(
                    name="author_ref",
                    description="a reference to author identity in another system",
                    type="string",
                ),
                ChildProperty(
                    name="attachments",
                    child_tag_names=["file", "image", "reaction"],
                    description="a list of message attachments",
                ),
            ],
        ),
    ],
)
