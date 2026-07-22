import pytest

from meshagent.api import RoomException
from meshagent.anthropic.event_publisher import make_anthropic_agent_event_publisher
from meshagent.openai.tools.event_publisher import make_openai_agent_event_publisher
from meshagent.agents.messages import (
    AgentMessage,
    AgentImageGenerationCompleted,
    AgentImageGenerationStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallArgumentsDelta,
    AgentToolCallEnded,
    AgentToolCallInProgress,
    AgentToolCallPending,
    AgentToolCallStarted,
)
from meshagent.agents.responses_thread_adapter import (
    _headline_for_response_event,
    ResponsesThreadAdapter,
    _extract_image_dimensions,
    response_event_to_agent_event,
)
from meshagent.agents.thread_adapter import ThreadAdapter


class _FakeParticipant:
    def __init__(self, *, name: str):
        self._name = name

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant(name="assistant")


class _FakeElement:
    def __init__(self, tag_name: str) -> None:
        self.tag_name = tag_name


class _FakeRoot:
    def __init__(self, children: list[_FakeElement]) -> None:
        self._children = children

    def get_children(self) -> list[_FakeElement]:
        return self._children


class _FakeThread:
    def __init__(self, children: list[_FakeElement]) -> None:
        self.root = _FakeRoot(children)


@pytest.mark.asyncio
async def test_responses_stop_awaits_base_stop_before_clearing_runtime_maps(
    monkeypatch,
) -> None:
    adapter = object.__new__(ResponsesThreadAdapter)
    active_event = object()
    adapter._active_events_by_key = {"tool-1": active_event}
    adapter._saved_image_ids_by_item_id = {"item-1": "image-1"}
    adapter._saved_image_stage_by_item_id = {"item-1": "final"}
    calls: list[dict] = []

    async def _fake_base_stop(self):
        calls.append(
            {
                "active_events": dict(self._active_events_by_key),
                "saved_image_ids": dict(self._saved_image_ids_by_item_id),
                "saved_image_stages": dict(self._saved_image_stage_by_item_id),
            }
        )

    monkeypatch.setattr(ThreadAdapter, "stop", _fake_base_stop)

    await adapter.stop()

    assert calls == [
        {
            "active_events": {"tool-1": active_event},
            "saved_image_ids": {"item-1": "image-1"},
            "saved_image_stages": {"item-1": "final"},
        }
    ]
    assert adapter._active_events_by_key == {}
    assert adapter._saved_image_ids_by_item_id == {}
    assert adapter._saved_image_stage_by_item_id == {}


@pytest.mark.asyncio
async def test_responses_handle_custom_event_resolves_messages_element() -> None:
    adapter = object.__new__(ResponsesThreadAdapter)
    messages = _FakeElement("messages")
    adapter._thread = _FakeThread([_FakeElement("members"), messages])

    calls: list[dict] = []

    async def _fake_handle_custom_event_for_messages(*, messages, event):
        calls.append({"messages": messages, "event": event})

    adapter._handle_custom_event_for_messages = (  # type: ignore[method-assign]
        _fake_handle_custom_event_for_messages
    )

    event = {"type": "agent.event", "kind": "tool"}
    await adapter.handle_custom_event(event=event)

    assert calls == [{"messages": messages, "event": event}]


@pytest.mark.asyncio
async def test_responses_handle_custom_event_requires_messages_element() -> None:
    adapter = object.__new__(ResponsesThreadAdapter)
    adapter._thread = _FakeThread([_FakeElement("members")])

    with pytest.raises(
        RoomException,
        match="messages element is missing from thread document",
    ):
        await adapter.handle_custom_event(event={"type": "agent.event", "kind": "tool"})


def test_openai_event_publisher_preserves_commentary_message_phase() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "message-1",
            "delta": "checking",
            "phase": "commentary",
        }
    )
    publisher(
        {
            "type": "response.output_text.done",
            "item_id": "message-1",
            "text": "checking",
            "phase": "commentary",
        }
    )

    assert isinstance(messages[0], AgentTextContentStarted)
    assert messages[0].phase == "commentary"
    assert isinstance(messages[1], AgentTextContentDelta)
    assert messages[1].phase == "commentary"
    assert isinstance(messages[2], AgentTextContentEnded)
    assert messages[2].phase == "commentary"


def test_openai_event_publisher_applies_output_item_phase_to_text_delta() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "message-1",
                "type": "message",
                "phase": "final_answer",
            },
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "answer",
        }
    )

    assert isinstance(messages[0], AgentTextContentStarted)
    assert messages[0].phase == "final_answer"
    assert isinstance(messages[1], AgentTextContentDelta)
    assert messages[1].phase == "final_answer"


def test_openai_event_publisher_emits_distinct_text_delta_message_ids() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "message-1",
            "delta": "hel",
        }
    )
    publisher(
        {
            "type": "response.output_text.delta",
            "item_id": "message-1",
            "delta": "lo",
        }
    )

    deltas = [
        message for message in messages if isinstance(message, AgentTextContentDelta)
    ]
    assert [delta.text for delta in deltas] == ["hel", "lo"]
    assert len({delta.message_id for delta in deltas}) == 2
    assert all(delta.message_id != delta.item_id for delta in deltas)


def test_openai_event_publisher_emits_tool_argument_delta() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item_id": "tool-1",
            "item": {
                "id": "tool-1",
                "type": "function_call",
                "name": "write_file",
                "call_id": "call-1",
                "arguments": "",
            },
        }
    )
    publisher(
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": '{"path":"src/app.py"',
        }
    )

    assert isinstance(messages[0], AgentToolCallPending)
    assert isinstance(messages[1], AgentToolCallArgumentsDelta)
    assert messages[1].item_id == "tool-1"
    assert messages[1].call_id == "call-1"
    assert messages[1].delta == '{"path":"src/app.py"'
    assert "toolkit" not in messages[1].model_dump()
    assert "tool" not in messages[1].model_dump()


def _assert_tool_call_lifecycle_shape(
    *,
    messages: list[AgentMessage],
    item_id: str,
    toolkit: str,
    tool: str,
    call_id: str | None,
    delta: str,
) -> None:
    _assert_tool_lifecycle(
        messages=messages,
        item_id=item_id,
        toolkit=toolkit,
        tool=tool,
        call_id=call_id,
        delta=delta,
        require_in_progress=True,
    )


def _assert_tool_lifecycle(
    *,
    messages: list[AgentMessage],
    item_id: str,
    toolkit: str,
    tool: str,
    call_id: str | None,
    delta: str | None = None,
    require_in_progress: bool = False,
) -> None:
    pending_index = _index_of_tool_event(
        messages, AgentToolCallPending, item_id=item_id
    )
    delta_index = _index_of_tool_event(
        messages, AgentToolCallArgumentsDelta, item_id=item_id
    )
    started_index = _index_of_tool_event(
        messages, AgentToolCallStarted, item_id=item_id
    )
    ended_index = _index_of_tool_event(messages, AgentToolCallEnded, item_id=item_id)

    assert pending_index < delta_index < started_index < ended_index

    if require_in_progress:
        in_progress_index = _index_of_tool_event(
            messages, AgentToolCallInProgress, item_id=item_id
        )
        assert pending_index < in_progress_index < delta_index

    pending = messages[pending_index]
    assert isinstance(pending, AgentToolCallPending)
    _assert_tool_event_identity(
        pending, item_id=item_id, toolkit=toolkit, tool=tool, call_id=call_id
    )

    arguments_delta = messages[delta_index]
    _assert_tool_arguments_delta(
        message=arguments_delta,
        item_id=item_id,
        call_id=call_id,
        delta=delta,
    )

    started = messages[started_index]
    assert isinstance(started, AgentToolCallStarted)
    _assert_tool_event_identity(
        started, item_id=item_id, toolkit=toolkit, tool=tool, call_id=call_id
    )

    ended = messages[ended_index]
    assert isinstance(ended, AgentToolCallEnded)
    _assert_tool_event_identity(
        ended, item_id=item_id, toolkit=toolkit, tool=tool, call_id=call_id
    )


def _index_of_tool_event(
    messages: list[AgentMessage],
    event_type: type[AgentMessage],
    *,
    item_id: str,
) -> int:
    for index, message in enumerate(messages):
        if isinstance(message, event_type) and message.item_id == item_id:
            return index
    raise AssertionError(f"missing {event_type.__name__} for {item_id}")


def _assert_tool_event_identity(
    message: AgentMessage,
    *,
    item_id: str,
    toolkit: str,
    tool: str,
    call_id: str | None,
) -> None:
    assert message.thread_id == "thread-1"
    assert message.turn_id == "turn-1"
    assert message.item_id == item_id
    assert message.toolkit == toolkit
    assert message.tool == tool
    assert message.call_id == call_id


def _assert_tool_arguments_delta(
    *,
    message: AgentMessage,
    item_id: str,
    call_id: str | None,
    delta: str | None,
) -> None:
    assert isinstance(message, AgentToolCallArgumentsDelta)
    assert message.thread_id == "thread-1"
    assert message.turn_id == "turn-1"
    assert message.item_id == item_id
    assert message.call_id == call_id
    if delta is not None:
        assert message.delta == delta
    dumped = message.model_dump()
    assert "toolkit" not in dumped
    assert "tool" not in dumped


def test_openai_event_publisher_emits_apply_patch_lifecycle_from_stream_events() -> (
    None
):
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.apply_patch_call.in_progress",
            "output_index": 0,
            "item_id": "patch-1",
            "sequence_number": 1,
        }
    )
    publisher(
        {
            "type": "response.apply_patch_call_operation_diff.delta",
            "output_index": 0,
            "item_id": "patch-1",
            "delta": "*** Begin Patch\n*** Update File: app.ts\n",
            "sequence_number": 2,
        }
    )
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "id": "patch-1",
                "type": "apply_patch_call",
                "status": "in_progress",
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item_id": "patch-1",
            "result": "done",
        }
    )

    _assert_tool_call_lifecycle_shape(
        messages=messages,
        item_id="patch-1",
        toolkit="openai",
        tool="apply_patch",
        call_id=None,
        delta="*** Begin Patch\n*** Update File: app.ts\n",
    )


def test_openai_event_publisher_preserves_cancelled_handler_error_code() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "id": "tool-1",
                "call_id": "call-1",
                "type": "function_call",
                "name": "client.pick_color",
                "arguments": "{}",
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item_id": "tool-1",
            "error": "client toolkit call cancelled: participant_disconnected",
            "error_code": "cancelled",
        }
    )

    ended = next(
        message for message in messages if isinstance(message, AgentToolCallEnded)
    )
    assert ended.error is not None
    assert ended.error.code == "cancelled"


def test_openai_event_publisher_emits_apply_patch_fallback_delta_from_handler() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.apply_patch_call.in_progress",
            "output_index": 0,
            "item_id": "patch-1",
            "sequence_number": 1,
        }
    )
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "id": "patch-1",
                "type": "apply_patch_call",
                "call_id": "call-1",
                "status": "in_progress",
                "operation": {
                    "type": "update_file",
                    "path": "app.ts",
                    "diff": "@@\n-old\n+new\n",
                },
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "patch-1"})

    assert [type(message) for message in messages] == [
        AgentToolCallPending,
        AgentToolCallInProgress,
        AgentToolCallArgumentsDelta,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    _assert_tool_arguments_delta(
        message=messages[2],
        item_id="patch-1",
        call_id="call-1",
        delta="@@\n-old\n+new\n",
    )


@pytest.mark.parametrize(
    ("delta_event_type", "tool_item", "delta", "toolkit", "tool", "call_id"),
    [
        (
            "response.function_call_arguments.delta",
            {
                "id": "function-1",
                "type": "function_call",
                "name": "write_file",
                "call_id": "call-function",
                "arguments": "",
            },
            '{"path":"app.ts"}',
            "function",
            "write_file",
            "call-function",
        ),
        (
            "response.shell_call_command.delta",
            {
                "id": "shell-1",
                "type": "shell_call",
                "call_id": "call-shell",
                "status": "in_progress",
            },
            "python report.py",
            "openai",
            "shell",
            "call-shell",
        ),
        (
            "response.custom_tool_call_input.delta",
            {
                "id": "custom-1",
                "type": "custom_tool_call",
                "call_id": "call-custom",
                "status": "in_progress",
            },
            "raw custom input",
            "openai",
            "custom_tool",
            "call-custom",
        ),
    ],
)
def test_openai_event_publisher_normalizes_builtin_tool_delta_lifecycle(
    delta_event_type: str,
    tool_item: dict[str, object],
    delta: str,
    toolkit: str,
    tool: str,
    call_id: str,
) -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )
    item_id = str(tool_item["id"])

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": tool_item,
        }
    )
    publisher(
        {
            "type": delta_event_type,
            "output_index": 0,
            "delta": delta,
        }
    )
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                **tool_item,
                "arguments": (
                    '{"path":"app.ts"}'
                    if tool_item["type"] == "function_call"
                    else {"path": "app.ts"}
                ),
                "input": "raw custom input"
                if tool_item["type"] == "custom_tool_call"
                else None,
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": item_id})

    _assert_tool_lifecycle(
        messages=messages,
        item_id=item_id,
        toolkit=toolkit,
        tool=tool,
        call_id=call_id,
        delta=delta,
    )


def test_openai_event_publisher_normalizes_mcp_builtin_delta_lifecycle() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "mcp-1",
                "type": "mcp_call",
                "server_label": "docs",
                "name": "search",
                "call_id": "call-mcp",
                "arguments": {},
            },
        }
    )
    publisher(
        {
            "type": "response.mcp_call_arguments.delta",
            "output_index": 0,
            "delta": '{"query":"meshagent"}',
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "mcp-1",
                "type": "mcp_call",
                "server_label": "docs",
                "name": "search",
                "call_id": "call-mcp",
                "arguments": {"query": "meshagent"},
                "output": "done",
                "status": "completed",
            },
        }
    )
    _assert_tool_lifecycle(
        messages=messages,
        item_id="mcp-1",
        toolkit="docs",
        tool="search",
        call_id="call-mcp",
        delta='{"query":"meshagent"}',
    )
    final_started = next(
        message
        for message in reversed(messages[:-1])
        if isinstance(message, AgentToolCallStarted)
    )
    assert final_started.arguments == {"query": "meshagent"}


def test_openai_event_publisher_normalizes_code_interpreter_delta_lifecycle() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "code-1",
                "type": "code_interpreter_call",
                "call_id": "call-code",
                "status": "in_progress",
            },
        }
    )
    publisher(
        {
            "type": "response.code_interpreter_call_code.delta",
            "output_index": 0,
            "delta": "print('hello')",
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "code-1",
                "type": "code_interpreter_call",
                "call_id": "call-code",
                "status": "completed",
                "code": "print('hello')",
                "outputs": [],
            },
        }
    )
    _assert_tool_lifecycle(
        messages=messages,
        item_id="code-1",
        toolkit="openai",
        tool="code_interpreter",
        call_id="call-code",
        delta="print('hello')",
    )
    final_started = next(
        message
        for message in reversed(messages[:-1])
        if isinstance(message, AgentToolCallStarted)
    )
    assert final_started.arguments == {"code": "print('hello')", "outputs": []}


@pytest.mark.parametrize(
    ("tool_item", "toolkit", "tool", "result"),
    [
        (
            {
                "id": "web-1",
                "type": "web_search_call",
                "call_id": "call-web",
                "status": "in_progress",
                "query": "meshagent",
            },
            "openai",
            "web_search",
            {"results": [{"title": "MeshAgent"}]},
        ),
        (
            {
                "id": "file-1",
                "type": "file_search_call",
                "call_id": "call-file",
                "status": "in_progress",
                "queries": ["report.py"],
            },
            "openai",
            "file_search",
            {"results": [{"filename": "report.py"}]},
        ),
        (
            {
                "id": "computer-1",
                "type": "computer_call",
                "call_id": "call-computer",
                "status": "in_progress",
                "action": {"type": "screenshot"},
            },
            "openai",
            "computer",
            {"output": [{"type": "screenshot"}]},
        ),
        (
            {
                "id": "local-shell-1",
                "type": "local_shell_call",
                "call_id": "call-local-shell",
                "status": "in_progress",
                "command": "pwd",
            },
            "openai",
            "local_shell",
            {"output": "done"},
        ),
        (
            {
                "id": "mcp-list-1",
                "type": "mcp_list_tools",
                "server_label": "docs",
                "call_id": "call-mcp-list",
                "status": "in_progress",
            },
            "docs",
            "list_tools",
            {"tools": [{"name": "search"}]},
        ),
    ],
)
def test_openai_event_publisher_emits_fallback_delta_for_builtin_tool_lifecycle(
    tool_item: dict[str, object],
    toolkit: str,
    tool: str,
    result: dict[str, object],
) -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": tool_item,
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                **tool_item,
                "status": "completed",
                "output": result,
            },
        }
    )
    if tool_item["type"] in {"computer_call", "local_shell_call"}:
        publisher({"type": "meshagent.handler.added", "item": tool_item})
        publisher(
            {
                "type": "meshagent.handler.done",
                "item_id": tool_item["id"],
                "result": result,
            }
        )

        _assert_tool_lifecycle(
            messages=messages,
            item_id=str(tool_item["id"]),
            toolkit=toolkit,
            tool=tool,
            call_id=str(tool_item["call_id"]),
        )
        return

    _assert_tool_lifecycle(
        messages=messages,
        item_id=str(tool_item["id"]),
        toolkit=toolkit,
        tool=tool,
        call_id=str(tool_item["call_id"]),
    )


def test_anthropic_event_publisher_emits_tool_argument_delta_lifecycle() -> None:
    messages: list[AgentMessage] = []
    publisher = make_anthropic_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "message_start",
            "event": {"message": {"id": "msg-1"}},
        }
    )
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 0,
                "content_block": {
                    "id": "tool-1",
                    "type": "tool_use",
                    "name": "write_file",
                    "input": {},
                },
            },
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"path":"report.py"}',
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 0}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "id": "tool-1",
                "type": "tool_use",
                "name": "write_file",
                "input": {"path": "report.py"},
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "tool-1"})

    _assert_tool_lifecycle(
        messages=messages,
        item_id="tool-1",
        toolkit="function",
        tool="write_file",
        call_id="tool-1",
        delta='{"path":"report.py"}',
    )


def test_openai_event_publisher_normalizes_image_generation_lifecycle() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.image_generation_call.generating",
            "item_id": "image-1",
            "call_id": "call-image",
            "prompt": "chart",
            "size": "1024x1024",
        }
    )
    publisher(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "image-1",
                "type": "image_generation_call",
                "call_id": "call-image",
                "status": "completed",
                "prompt": "chart",
                "size": "1024x1024",
                "result": "aW1hZ2U=",
            },
        }
    )

    assert isinstance(messages[0], AgentImageGenerationStarted)
    assert messages[0].toolkit == "openai"
    assert messages[0].tool == "image_generation"
    assert isinstance(messages[-1], AgentImageGenerationCompleted)
    assert messages[-1].toolkit == "openai"
    assert messages[-1].tool == "image_generation"
    assert len(messages[-1].images) == 1


def test_openai_event_publisher_emits_commentary_from_completed_snapshot() -> None:
    messages: list[AgentMessage] = []
    publisher = make_openai_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=messages.append,
    )

    publisher(
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "id": "message-1",
                        "type": "message",
                        "phase": "commentary",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "checking",
                            }
                        ],
                    }
                ]
            },
        }
    )

    assert isinstance(messages[0], AgentTextContentStarted)
    assert messages[0].phase == "commentary"
    assert isinstance(messages[1], AgentTextContentDelta)
    assert messages[1].text == "checking"
    assert messages[1].phase == "commentary"
    assert isinstance(messages[2], AgentTextContentEnded)
    assert messages[2].phase == "commentary"


def test_extract_image_dimensions_prefers_explicit_fields_then_size():
    width, height = _extract_image_dimensions(
        item={"width": 1536, "height": 1024},
        event={"size": "1024x1024"},
    )
    assert width == 1536
    assert height == 1024

    width, height = _extract_image_dimensions(
        item={"size": "512x768"},
        event=None,
    )
    assert width == 512
    assert height == 768


@pytest.mark.asyncio
async def test_emit_image_status_event_updates_image_element_state():
    adapter = object.__new__(ResponsesThreadAdapter)
    adapter._room = _FakeRoom()

    events: list[dict] = []
    writes: list[dict] = []

    async def _fake_handle_custom_event_for_messages(*, messages, event):
        del messages
        events.append(event)

    def _fake_write_image(**kwargs):
        writes.append(kwargs)
        return kwargs.get("message_id", "")

    adapter._handle_custom_event_for_messages = (  # type: ignore[method-assign]
        _fake_handle_custom_event_for_messages
    )
    adapter.write_image = _fake_write_image  # type: ignore[assignment]

    await adapter._emit_image_status_event(
        messages=object(),
        item_id="img-item-1",
        state="in_progress",
        headline="Generating image",
        turn_id="turn-1",
        width=1024,
        height=768,
    )

    assert len(events) == 1
    assert events[0]["state"] == "in_progress"
    assert events[0]["turn_id"] == "turn-1"
    assert events[0]["item_id"] == "img-item-1"

    assert len(writes) == 1
    assert writes[0]["message_id"] == "img-item-1"
    assert writes[0]["turn_id"] == "turn-1"
    assert writes[0]["status"] == "generating"
    assert writes[0]["width"] == 1024
    assert writes[0]["height"] == 768


def test_computer_call_events_are_not_classified_as_exec():
    event = {
        "type": "response.output_item.added",
        "turn_id": "turn-1",
        "item_id": "item_123",
        "item": {
            "id": "item_123",
            "type": "computer_call",
            "status": "in_progress",
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["turn_id"] == "turn-1"
    assert normalized["headline"] == "Using computer"


def test_computer_call_click_event_headline_is_friendly_without_coordinates():
    event = {
        "type": "response.output_item.added",
        "item_id": "item_456",
        "item": {
            "id": "item_456",
            "type": "computer_call",
            "status": "in_progress",
            "action": {
                "type": "click",
                "x": 140,
                "y": 320,
            },
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["headline"] == "Clicking on page"
    assert normalized["details"] == []


def test_computer_call_scroll_event_uses_human_friendly_direction():
    event = {
        "type": "response.output_item.done",
        "item_id": "item_789",
        "item": {
            "id": "item_789",
            "type": "computer_call",
            "status": "completed",
            "action": {
                "type": "scroll",
                "scroll_x": 0,
                "scroll_y": -640,
            },
        },
    }

    normalized = response_event_to_agent_event(event)
    assert isinstance(normalized, dict)
    assert normalized["kind"] == "tool"
    assert normalized["headline"] == "Scrolled page"
    assert normalized["details"] == ["Direction: up"]


def test_computer_startup_event_uses_startup_specific_headline():
    event = {
        "type": "response.output_item.done",
        "item_id": "startup_1",
        "name": "computer.startup",
        "state": "completed",
    }

    headline = _headline_for_response_event(event=event, kind="tool", state="completed")
    assert headline == "Computer ready"
