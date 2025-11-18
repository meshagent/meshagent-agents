from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

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
                ChildProperty(
                    name="items",
                    child_tag_names=["message", "exec", "ui", "reasoning"],
                    description="the messages in this thread",
                )
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
            tag_name="reasoning",
            description="a reasoning trace",
            properties=[
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
            tag_name="message",
            description="a message sent in the conversation",
            properties=[
                ValueProperty(
                    name="id", description="the id of the message", type="string"
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
                    child_tag_names=["file"],
                    description="a list of message attachments",
                ),
            ],
        ),
    ],
)
