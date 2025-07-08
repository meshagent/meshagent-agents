from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

super_editor_document_schema = MeshSchema(
    root_tag_name="document",
    elements=[
        ElementType(
            tag_name="document",
            description="A SuperEditor document for use with Flutter apps",
            properties=[
                ChildProperty(
                    name="children",
                    child_tag_names=[
                        "block",
                        "horizontalRule",
                        "image",
                    ],
                ),
            ],
        ),
        ElementType(
            tag_name="block",
            description="Block element - this can be a paragraph, list-item or tasks",
            properties=[
                ValueProperty(name="blockType", type="string"),
                ValueProperty(name="textAlign", type="string"),
                ValueProperty(name="itemType", type="string"),
                ValueProperty(name="indent", type="number"),
                ChildProperty(name="children", child_tag_names=["text"]),
            ],
        ),
        ElementType(
            tag_name="image",
            description="Image element",
            properties=[
                ValueProperty(name="imageUrl", type="string"),
                ValueProperty(name="width", type="number"),
                ValueProperty(name="height", type="number"),
            ],
        ),
        ElementType(
            tag_name="horizontalRule",
            description="Horizontal rule element",
            properties=[],
        ),
        ElementType(
            tag_name="text",
            description="Text element",
            properties=[
                ValueProperty(name="text", type="string"),
            ],
        ),
    ],
)
