import pytest

from meshagent.api import RoomException

from meshagent.agents.context import AgentChatContext


def test_agent_chat_context_does_not_support_binary_inputs_by_default() -> None:
    context = AgentChatContext(system_role=None)

    assert context.supports_images is False
    assert context.supports_files is False

    with pytest.raises(RoomException):
        context.append_image_message(mime_type="image/png", data=b"img")

    with pytest.raises(RoomException):
        context.append_file_message(
            filename="file.txt",
            mime_type="text/plain",
            data=b"file",
        )
