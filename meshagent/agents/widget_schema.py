from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

widget_names = {
    "button",
    "box",
    "canvas",
    "text",
    "image",
    "video",
    "hstack",
    "vstack",
    "zstack",
    "input",
    "textarea",
    "form",
    "select",
    "checkbox",
    "radiogroup",
    "spacer",
    "divider",
    "spinner",
    "markdown",
    "datepicker",
    "scrollview",
    "format",
    "datagrid",
    "chat",
    "tabs",
}


def text_style_properties():
    return [
        ValueProperty(
            name="color", description="the color of the text #rrggbbaa", type="string"
        ),
        ValueProperty(
            name="emphasis",
            description="the emphasis for the text",
            type="string",
            enum=["none", "italic"],
        ),
        ValueProperty(
            name="decoration",
            description="the decoration for the text",
            type="string",
            enum=["none", "underline", "strikethrough"],
        ),
        ValueProperty(
            name="font", description="the name of a font to render with", type="string"
        ),
        ValueProperty(
            name="weight",
            description="the font weight to use",
            type="string",
            enum=[
                "bold",
                "normal",
                "100",
                "200",
                "300",
                "400",
                "500",
                "600",
                "700",
                "800",
                "900",
            ],
        ),
        ValueProperty(
            name="align",
            description="the text alignment to use",
            type="string",
            enum=["left", "right", "center", "none"],
        ),
        ValueProperty(
            name="letterSpacing", description="the letter spacing to use", type="number"
        ),
        ValueProperty(name="lineHeight", description="the line height", type="number"),
        ValueProperty(name="size", description="the text size", type="number"),
    ]


def box_properties():
    return [
        ValueProperty(
            name="topLeftRadius",
            description="the top left radius for the box",
            type="number",
        ),
        ValueProperty(
            name="topRightRadius",
            description="the top right radius for the box",
            type="number",
        ),
        ValueProperty(
            name="bottomLeftRadius",
            description="the bottom left radius for the box",
            type="number",
        ),
        ValueProperty(
            name="bottomRightRadius",
            description="the bottom right radius for the box",
            type="number",
        ),
        ValueProperty(
            name="marginTop", description="the top margin for the box", type="number"
        ),
        ValueProperty(
            name="marginBottom",
            description="the bottom margin for the box",
            type="number",
        ),
        ValueProperty(
            name="marginLeft", description="the left margin for the box", type="number"
        ),
        ValueProperty(
            name="marginRight",
            description="the right margin for the box",
            type="number",
        ),
        ValueProperty(
            name="paddingTop", description="the top padding for the box", type="number"
        ),
        ValueProperty(
            name="paddingBottom",
            description="the bottom padding for the box",
            type="number",
        ),
        ValueProperty(
            name="paddingLeft",
            description="the left padding for the box",
            type="number",
        ),
        ValueProperty(
            name="paddingRight",
            description="the right padding for the box",
            type="number",
        ),
        ValueProperty(
            name="borderColor",
            description="the border color for the box",
            type="number",
        ),
        ValueProperty(
            name="borderTop", description="the top border for the box", type="number"
        ),
        ValueProperty(
            name="borderBottom",
            description="the bottom border for the box",
            type="number",
        ),
        ValueProperty(
            name="borderLeft", description="the left border for the box", type="number"
        ),
        ValueProperty(
            name="borderRight",
            description="the right border for the box",
            type="number",
        ),
        ValueProperty(
            name="borderTopColor",
            description="the top border color for the box #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="borderBottomColor",
            description="the bottom border color for the box #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="borderLeftColor",
            description="the left border color for the box #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="borderRightColor",
            description="the right border color for the box #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="backgroundColor",
            description="the background color for the box #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="backgroundImageUrl", description="a background image", type="string"
        ),
        ValueProperty(
            name="backgroundImageFit",
            description="how to position the background image",
            type="string",
            enum=[
                "fit",
                "contain",
                "cover",
                "fitWidth",
                "fitHeight",
                "scaleDown",
                "none",
            ],
        ),
        ValueProperty(
            name="linearGradient",
            description="a linear gradient for the background",
            type="string",
        ),
        ValueProperty(
            name="radialGradient",
            description="a radial gradient for the background",
            type="string",
        ),
        ValueProperty(name="width", description="the width for the box", type="number"),
        ValueProperty(
            name="height", description="the height for the box", type="number"
        ),
    ]


def widget_properties():
    return [
        ValueProperty(
            name="name",
            description="the name of the widget, must be unique in this widget",
            type="string",
        ),
        ValueProperty(
            name="script",
            description="a script that contains the logic for this button",
            type="string",
        ),
        ValueProperty(
            name="visible",
            description="the whether the widget is initially visible",
            type="boolean",
        ),
    ]


def child_widgets():
    return [
        ChildProperty(
            name="children",
            description="child widgets",
            child_tag_names=widget_names,
        )
    ]


button = ElementType(
    tag_name="button",
    description="a button",
    properties=[
        *widget_properties(),
        *box_properties(),
        *child_widgets(),
    ],
)

box = ElementType(
    tag_name="box",
    description="a box",
    properties=[
        *widget_properties(),
        *box_properties(),
        *child_widgets(),
    ],
)


canvas = ElementType(
    tag_name="canvas",
    description="a canvas",
    properties=[
        *widget_properties(),
        *box_properties(),
        *child_widgets(),
    ],
)

text = ElementType(
    tag_name="text",
    description="text",
    properties=[
        ValueProperty(name="text", description="the text to display", type="string"),
        *widget_properties(),
        *text_style_properties(),
    ],
)

markdown = ElementType(
    tag_name="markdown",
    description="markdown",
    properties=[
        *widget_properties(),
        ValueProperty(
            name="text", description="the markdown to display", type="string"
        ),
    ],
)

image = ElementType(
    tag_name="image",
    description="image",
    properties=[
        *widget_properties(),
        ValueProperty(name="url", description="the url of an image", type="string"),
        ValueProperty(
            name="alt", description="a description of the image", type="string"
        ),
        ValueProperty(
            name="fit",
            description="how to position the image",
            type="string",
            enum=[
                "fit",
                "contain",
                "cover",
                "fitWidth",
                "fitHeight",
                "scaleDown",
                "none",
            ],
        ),
        *box_properties(),
    ],
)

video = ElementType(
    tag_name="video",
    description="video",
    properties=[
        *widget_properties(),
        ValueProperty(name="url", description="the url of a video", type="string"),
        ValueProperty(
            name="alt", description="a description of the video", type="string"
        ),
        ValueProperty(
            name="fit",
            description="how to position the video",
            type="string",
            enum=[
                "fit",
                "contain",
                "cover",
                "fitWidth",
                "fitHeight",
                "scaleDown",
                "none",
            ],
        ),
        *box_properties(),
    ],
)

scrollview = ElementType(
    tag_name="scrollview",
    description="scrollview",
    properties=[
        *widget_properties(),
        *box_properties(),
        ValueProperty(
            name="spacing", description="spacing between the items", type="number"
        ),
        ValueProperty(
            name="direction",
            description="spacing between the items",
            type="string",
            enum=["horizontal", "vertical"],
            required=True,
        ),
        *child_widgets(),
    ],
)


hstack = ElementType(
    tag_name="hstack",
    description="hstack",
    properties=[
        *widget_properties(),
        ValueProperty(
            name="spacing", description="spacing between the items", type="number"
        ),
        *child_widgets(),
    ],
)

vstack = ElementType(
    tag_name="vstack",
    description="vstack",
    properties=[
        ValueProperty(
            name="spacing", description="spacing between the items", type="number"
        ),
        *widget_properties(),
        *child_widgets(),
    ],
)

zstack = ElementType(
    tag_name="zstack",
    description="zstack",
    properties=[
        *widget_properties(),
        *child_widgets(),
    ],
)

form = ElementType(
    tag_name="form",
    description="form",
    properties=[
        *widget_properties(),
        *child_widgets(),
    ],
)

input = ElementType(
    tag_name="input",
    description="input",
    properties=[
        ValueProperty(
            name="placeholder",
            description="a placeholder for the field",
            type="string",
        ),
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
        ),
        ValueProperty(
            name="obscure",
            description="whether to obscure the input",
            type="boolean",
        ),
        *widget_properties(),
        *box_properties(),
        *text_style_properties(),
    ],
)

select = ElementType(
    tag_name="select",
    description="select",
    properties=[
        ValueProperty(
            name="placeholder",
            description="a placeholder for the field",
            type="string",
        ),
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
        ),
        ChildProperty(name="options", child_tag_names=["option"]),
        *widget_properties(),
        *box_properties(),
    ],
)

textarea = ElementType(
    tag_name="textarea",
    description="textarea",
    properties=[
        ValueProperty(
            name="placeholder",
            description="a placeholder for the field",
            type="string",
        ),
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
        ),
        *widget_properties(),
        *text_style_properties(),
        *box_properties(),
    ],
)

checkbox = ElementType(
    tag_name="checkbox",
    description="checkbox",
    properties=[
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
            enum=["checked", "unchecked"],
        ),
        *child_widgets(),
    ],
)

datepicker = ElementType(
    tag_name="datepicker",
    description="datepicker",
    properties=[
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
        ),
        *widget_properties(),
        *box_properties(),
        *text_style_properties(),
    ],
)

option = ElementType(
    tag_name="option",
    description="option",
    properties=[
        ValueProperty(
            name="value",
            description="the initial value of the input field",
            type="string",
        ),
        *child_widgets(),
    ],
)

radio = ElementType(
    tag_name="radio",
    description="radio",
    properties=[
        ValueProperty(
            name="value",
            description="the initial value of the input field",
            type="string",
        ),
        *child_widgets(),
    ],
)

radiogroup = ElementType(
    tag_name="radiogroup",
    description="radiogroup",
    properties=[
        ValueProperty(
            name="initialValue",
            description="the initial value of the input field",
            type="string",
        ),
        ChildProperty(name="radios", child_tag_names=["radio"]),
        *widget_properties(),
    ],
)

spacer = ElementType(
    tag_name="spacer",
    description="spacer",
    properties=[
        *widget_properties(),
    ],
)

divider = ElementType(
    tag_name="divider",
    description="divider",
    properties=[
        ValueProperty(
            name="color",
            description="the color of the divider #rrggbbaa",
            type="string",
        ),
        ValueProperty(name="marginTop", description="the top margin", type="number"),
        ValueProperty(
            name="marginBottom", description="the bottom margin", type="number"
        ),
        ValueProperty(name="marginLeft", description="the left margin", type="number"),
        ValueProperty(
            name="marginRight", description="the right margin", type="number"
        ),
        *widget_properties(),
    ],
)

spinner = ElementType(
    tag_name="spinner",
    description="spinner",
    properties=[
        ValueProperty(
            name="color",
            description="the color of the spinner #rrggbbaa",
            type="string",
        ),
        ValueProperty(
            name="width", description="the height of the spinner", type="number"
        ),
        ValueProperty(
            name="height", description="the width of the spinner", type="number"
        ),
        *widget_properties(),
    ],
)

widgets = ElementType(
    tag_name="widgets",
    description="widgets",
    properties=[*child_widgets()],
)

widget = ElementType(
    tag_name="widget",
    description="widget",
    properties=[
        ChildProperty(
            name="children",
            description="child widgets",
            ordered=True,
            child_tag_names=[
                "widgets",
            ],
        )
    ],
)


chat = ElementType(
    tag_name="chat",
    description="widget",
    properties=[
        ValueProperty(
            name="path",
            description="the path to the chat thread",
            type="string",
        ),
        ValueProperty(
            name="agent",
            description="the name of an agent to chat with on the thread",
            type="string",
        ),
        ValueProperty(
            name="initalStyle",
            description="how should the chat look before the user interacts with the agent",
            type="string",
            enum=["centered", "feed"],
        ),
        *box_properties(),
    ],
)


gridcolumn = ElementType(
    tag_name="gridcolumn",
    description="data grid column",
    properties=[
        ValueProperty(
            name="label",
            description="a friendly label for the column",
            type="string",
        ),
        ValueProperty(
            name="name",
            description="the name of the column",
            type="string",
        ),
        ValueProperty(
            name="width",
            description="the width of the column",
            type="number",
        ),
        ValueProperty(
            name="height",
            description="the width of the column",
            type="number",
        ),
        ValueProperty(
            name="cursor",
            description="the width of the column",
            type="string",
            enum=["none", "click"],
        ),
        *text_style_properties(),
    ],
)


datagrid = ElementType(
    tag_name="datagrid",
    description="data grid",
    properties=[
        *widget_properties(),
        *box_properties(),
        ValueProperty(
            name="table",
            description="the table to query for data",
            type="string",
        ),
        ValueProperty(
            name="filter",
            description="a filter statement",
            type="string",
        ),
        ChildProperty(
            name="columns",
            description="columns in the grid",
            ordered=True,
            child_tag_names=[
                "gridcolumn",
            ],
        ),
    ],
)

tab = ElementType(
    tag_name="tab",
    description="tab",
    properties=[
        *box_properties(),
        *widget_properties(),
        *child_widgets(),
        ValueProperty(
            name="activeBackgroundColor",
            description="the color of the background when the tab is active #rrggbbaa",
            type="string",
        ),
    ],
)

tabs = ElementType(
    tag_name="tabs",
    description="tabs",
    properties=[
        *box_properties(),
        *widget_properties(),
        ValueProperty(
            name="spacing", description="spacing between the items", type="number"
        ),
        ValueProperty(
            name="initialTab", description="the initially selected tab", type="string"
        ),
        ChildProperty(
            name="tabs",
            description="columns in the grid",
            ordered=True,
            child_tag_names=[
                "tab",
            ],
        ),
    ],
)


format = ElementType(
    tag_name="format",
    description="format text using an expression",
    properties=[
        *text_style_properties(),
        *widget_properties(),
    ],
)

widget_schema = MeshSchema(
    root_tag_name="widget",
    elements=[
        widget,
        canvas,
        box,
        button,
        text,
        video,
        image,
        hstack,
        vstack,
        zstack,
        input,
        textarea,
        form,
        select,
        checkbox,
        radiogroup,
        datepicker,
        spacer,
        divider,
        spinner,
        markdown,
        radio,
        option,
        widgets,
        scrollview,
        datagrid,
        gridcolumn,
        format,
        chat,
        tabs,
        tab,
    ],
)

if __name__ == "__main__":
    import json

    print(json.dumps(widget_schema.to_json()))
