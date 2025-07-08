from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

presentation_schema = MeshSchema(
    root_tag_name="presentation",
    elements=[
        ElementType(
            tag_name="presentation",
            description="a presentation",
            properties=[
                ChildProperty(
                    name="slides",
                    description="the slides for the presentation",
                    child_tag_names=["slide"],
                ),
            ],
        ),
        ElementType(
            tag_name="slide",
            description="a slide",
            properties=[
                ValueProperty(
                    name="title", description="a title for the slide", type="string"
                ),
                ValueProperty(
                    name="background",
                    description="an file to use for an background of the slide, must be a filename output from a previous step",
                    type="string",
                ),
                ChildProperty(
                    name="lines",
                    description="the slides for the presentation",
                    child_tag_names=["line"],
                ),
            ],
        ),
        ElementType(
            tag_name="line",
            description="a line of text",
            properties=[
                ValueProperty(
                    name="line",
                    description="the text of a bullet point for the slide",
                    type="string",
                ),
            ],
        ),
    ],
)
