from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

document_schema = MeshSchema(
    root_tag_name="document",
    elements=[
        ElementType(
            tag_name="document",
            description="a document",
            properties=[
                ChildProperty(
                    name="content",
                    description="the content in the document",
                    child_tag_names=["heading", "body", "file"],
                ),
            ],
        ),
        ElementType(
            tag_name="heading",
            description="heading",
            properties=[
                ValueProperty(
                    name="text", description="the text of the heading", type="string"
                ),
            ],
        ),
        ElementType(
            tag_name="body",
            description="text",
            properties=[
                ValueProperty(
                    name="text",
                    description="body content in markdown format",
                    type="string",
                ),
            ],
        ),
        ElementType(
            tag_name="file",
            description="an image",
            properties=[
                ValueProperty(
                    name="name",
                    description="reference a file that was output from one of the steps. use to display the content of a file or image.",
                    type="string",
                ),
            ],
        ),
    ],
)
