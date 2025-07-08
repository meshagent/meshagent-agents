from meshagent.api.schema import (
    ElementProperty,
    ValueProperty,
    MeshSchema,
    ChildProperty,
    ElementType,
)
from meshagent.api.schema_document import Document

schema_schema = MeshSchema(
    root_tag_name="schema",
    elements=[
        ElementType(
            tag_name="schema",
            description="defines an xml document representing a json schema",
            properties=[
                ValueProperty(
                    name="root_tag_name",
                    description="the root tag name in this document (must match one of the elements defined in the schema)",
                    type="string",
                ),
                ChildProperty(
                    name="elements",
                    description="the configuration of this sandbox",
                    child_tag_names=["element"],
                ),
            ],
        ),
        ElementType(
            tag_name="element",
            description="an element inside the document",
            properties=[
                ValueProperty(
                    name="tag_name",
                    description="the tag name used by this element. Element names are case-sensitive. Element names cannot contain spaces. Element names must start with a letter or underscore. Element names cannot start with the letters xml (or XML, or Xml, etc) Element names can contain letters, digits, hyphens, underscores, and periods.",
                    type="string",
                ),
                ChildProperty(
                    name="schema",
                    description="the attributes of this node",
                    child_tag_names=["attribute", "child_elements"],
                ),
                ValueProperty(name="description", type="string"),
            ],
        ),
        ElementType(
            tag_name="attribute",
            properties=[
                ValueProperty(
                    name="name",
                    description="the attribute name. Attribute names are case-sensitive. Attribute names cannot contain spaces. Attribute names must start with a letter or underscore. Attribute names cannot start with the letters xml (or XML, or Xml, etc) Attribute names can contain letters, digits, hyphens, underscores, and periods.",
                    type="string",
                ),
                ValueProperty(
                    name="type",
                    description="the type of value stored by this attribute",
                    type="string",
                    enum=["string", "boolean", "number"],
                ),
                ValueProperty(name="description", type="string"),
            ],
        ),
        ElementType(
            tag_name="child_elements",
            properties=[
                ValueProperty(
                    name="name",
                    description="the name of the property to store the children in. Propterty names cannot contain spaces. Propterty names must start with a letter or underscore. Propterty names cannot start with the letters xml (or XML, or Xml, etc) Propterty names can contain letters, digits, hyphens, and underscores.",
                    type="string",
                ),
                ValueProperty(name="description", type="string"),
                ChildProperty(
                    name="children",
                    description="the tag name of an element. The element must have a corresponding element definition under the schema elements",
                    child_tag_names=["element_reference"],
                ),
            ],
        ),
        ElementType(
            tag_name="element_reference",
            description="an allowed child element for an element, should only used when a schema can contain multiple items of a specific type",
            properties=[
                ValueProperty(
                    name="tag_name",
                    description="the tag name of an element. The element must have a corresponding element definition under the schema elements",
                    type="string",
                ),
            ],
        ),
    ],
)


def schema_from_schema_document(document: Document):
    root_tag_name = document.root["root_tag_name"]

    elements = list[ElementType]()

    for element in document.root.get_children():
        properties = list[ElementProperty]()

        for prop in element.get_children():
            name = prop["name"]
            description = prop["description"]
            type = prop["type"]

            if prop.tag_name == "attribute":
                properties.append(
                    ValueProperty(name=name, type=type, description=description)
                )
            elif prop.tag_name == "child_elements":
                child_tag_names = []

                for child in prop.get_children():
                    child_tag_names.append(child["tag_name"])

                properties.append(
                    ChildProperty(name=name, child_tag_names=child_tag_names)
                )

        et = ElementType(
            tag_name=element["tag_name"],
            properties=properties,
            description=element["description"],
        )
        elements.append(et)

    doc = MeshSchema(root_tag_name=root_tag_name, elements=elements)
    return doc
