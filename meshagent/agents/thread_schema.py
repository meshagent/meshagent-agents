from meshagent.api.schema import MeshSchema, ElementType,  ChildProperty, ValueProperty

thread_schema = MeshSchema(
    root_tag_name="thread",
    elements=[
        ElementType(
            tag_name="thread", 
            description="a thread of messages",
            properties=[
                ChildProperty(name="properties", description="the messages in the thread", ordered=True, child_tag_names=[
                    "members", "messages"
                ]),
            ]
        ),
        ElementType(
            tag_name="members", 
            description="the members of this thread",
            properties=[
                ChildProperty(name="items", child_tag_names=["member"], description="the messages in this thread")
            ]
        ),
        ElementType(
            tag_name="messages", 
            description="the messages of this thread",
            properties=[
                ChildProperty(name="items", child_tag_names=["message"], description="the messages in this thread")
            ]
        ),
        ElementType(
            tag_name="member", 
            description="a member of this thread",
            properties=[
                ValueProperty(name="name", description="the name of the member", type="string"),
                ValueProperty(name="type", description="the type of member", type="string", enum=[
                    "user", "agent"
                ]),
            ]
        ),
        ElementType(
            tag_name="file", 
            description="a file attachment",
            properties=[
                ValueProperty(name="path", description="the path of the file in the room", type="string"),
            ]
        ),
        ElementType(
            tag_name="message", 
            description="a message sent in the conversation",
            properties=[
                ValueProperty(name="id", description="the id of the message", type="string"),
                ValueProperty(name="text", description="the text of the message", type="string"),
                ValueProperty(name="created_at", description="the date that the message was sent in ISO format", type="string"),
                ValueProperty(name="author_name", description="the name of the author of the post", type="string"),
                ValueProperty(name="author_ref", description="a reference to author identity in another system", type="string"),
                ChildProperty(name="attachments", child_tag_names=["file"], description="a list of message attachments")
            ]
        ),          
    ]
)
