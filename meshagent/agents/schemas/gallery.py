from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

gallery_schema = MeshSchema(
    root_tag_name="gallery",
    elements=[
        ElementType(
            tag_name="gallery",
            description="a media gallery",
            properties=[
                ChildProperty(
                    name="content",
                    description="the content in the gallery",
                    child_tag_names=["file"],
                ),
            ],
        ),
        ElementType(
            tag_name="file",
            description="an image",
            properties=[
                ValueProperty(
                    name="path",
                    description="reference a file that was output from one of the steps. use to display the content of a file or image.",
                    type="string",
                ),
                ValueProperty(
                    name="description",
                    description="a description of the image",
                    type="string",
                ),
            ],
        ),
    ],
)
