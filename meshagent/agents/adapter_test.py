from typing import Any

import pytest

from meshagent.agents.adapter import (
    DEFAULT_MAX_TOOL_CALL_LENGTH,
    DEFAULT_MAX_TOOL_CALL_LINES,
    ToolResponseAdapter,
)
from meshagent.api.messaging import FileContent, JsonContent, TextContent


class _FakeRoom:
    pass


class _FakeToolResponseAdapter(ToolResponseAdapter):
    async def to_plain_text(self, *, response):
        del response
        return ""

    async def create_messages(
        self,
        *,
        context,
        tool_call: Any,
        response,
    ) -> list:
        del context
        del tool_call
        del response
        return []


def test_tool_response_adapter_defaults_match_opencode() -> None:
    adapter = _FakeToolResponseAdapter()

    assert adapter.max_tool_call_lines == DEFAULT_MAX_TOOL_CALL_LINES == 2000
    assert adapter.max_tool_call_length == DEFAULT_MAX_TOOL_CALL_LENGTH == 50 * 1024


def test_truncate_returns_text_content_for_long_text() -> None:
    adapter = _FakeToolResponseAdapter(max_tool_call_length=16, max_tool_call_lines=2)

    truncated = adapter.truncate(
        content=TextContent(text="line1\nline2\nline3\nline4"),
    )

    assert isinstance(truncated, TextContent)
    assert "line1\nline2" in truncated.text
    assert "line3" not in truncated.text
    assert "The tool call returned too much data and was truncated." in truncated.text


def test_truncate_returns_text_content_for_long_json() -> None:
    adapter = _FakeToolResponseAdapter(max_tool_call_length=12, max_tool_call_lines=10)

    truncated = adapter.truncate(
        content=JsonContent(json={"message": "x" * 50}),
    )

    assert isinstance(truncated, TextContent)
    assert "The tool call returned too much data and was truncated." in truncated.text


def test_truncate_leaves_file_content_unchanged() -> None:
    adapter = _FakeToolResponseAdapter(max_tool_call_length=8, max_tool_call_lines=1)
    content = FileContent(
        data=b"abcdefghijklmnopqrstuvwxyz",
        name="big.txt",
        mime_type="text/plain",
    )

    assert adapter.truncate(content=content) is content


@pytest.mark.asyncio
async def test_file_content_to_text_content_decodes_declared_text_file() -> None:
    adapter = _FakeToolResponseAdapter()
    content = FileContent(
        data="hello\nworld".encode("utf-8"),
        name="notes.txt",
        mime_type="text/plain; charset=utf-8",
    )

    text_content = await adapter.file_content_to_text_content(
        content=content,
    )

    assert isinstance(text_content, TextContent)
    assert text_content.text == "hello\nworld"


@pytest.mark.asyncio
async def test_file_content_to_text_content_decodes_utf8_text_without_text_mime() -> (
    None
):
    adapter = _FakeToolResponseAdapter()
    content = FileContent(
        data="plain utf8".encode("utf-8"),
        name="README",
        mime_type="application/octet-stream",
    )

    text_content = await adapter.file_content_to_text_content(
        content=content,
    )

    assert isinstance(text_content, TextContent)
    assert text_content.text == "plain utf8"


@pytest.mark.asyncio
async def test_file_content_to_text_content_rejects_binary_utf8_file() -> None:
    adapter = _FakeToolResponseAdapter()
    content = FileContent(
        data=b"\x00\x01\x02",
        name="binary.bin",
        mime_type="application/octet-stream",
    )

    assert (
        await adapter.file_content_to_text_content(
            content=content,
        )
        is None
    )


def test_tool_response_adapter_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        _FakeToolResponseAdapter(max_tool_call_length=0)

    with pytest.raises(ValueError):
        _FakeToolResponseAdapter(max_tool_call_lines=0)
