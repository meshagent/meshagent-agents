from __future__ import annotations

from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.messages import (
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_MESSAGE_TURN_START,
    AgentContextCompacted,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentThreadEvent,
    AgentToolCallEnded,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallStarted,
    TurnStart,
)
from meshagent.api.messaging import JsonContent


def test_default_agent_event_reader_roundtrips_core_agent_messages() -> None:
    context = AgentSessionContext()
    reader = LLMAdapter().make_agent_event_reader(context=context)

    reader.consume(
        TurnStart(
            type=AGENT_MESSAGE_TURN_START,
            thread_id="thread-1",
            message_id="user-1",
            content=[AgentTextContent(type="text", text="hello")],
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="hel",
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="lo",
        )
    )
    reader.consume(
        AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
        )
    )
    reader.consume(
        AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            toolkit="openai",
            tool="web_search",
            arguments={"query": "meshagent"},
        )
    )
    reader.consume(
        AgentToolCallLogDelta(
            type=AGENT_EVENT_TOOL_CALL_LOG_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            lines=[AgentToolCallLogLine(source="stdout", text="searching\n")],
        )
    )
    reader.consume(
        AgentToolCallEnded(
            type=AGENT_EVENT_TOOL_CALL_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            result=JsonContent(json={"results": [{"title": "MeshAgent"}]}),
        )
    )
    reader.consume(
        AgentThreadEvent(
            type=AGENT_EVENT_THREAD_EVENT,
            thread_id="thread-1",
            event={"kind": "shell", "cmd": "pwd"},
        )
    )
    reader.consume(
        AgentContextCompacted(
            type=AGENT_EVENT_CONTEXT_COMPACTED,
            thread_id="thread-1",
            checkpoint_id="checkpoint-1",
            path="dataset://threads/main",
            through_sequence=42,
            created_at="2026-05-05T00:00:00Z",
        )
    )
    reader.finalize()

    assert context.messages[0] == {"role": "user", "content": "hello"}
    assert context.messages[1] == {"role": "assistant", "content": "hello"}
    tool_call = context.messages[2]["content"][0]
    assert tool_call == {
        "type": "tool_call",
        "item_id": "tool-1",
        "namespace": "openai.responses",
        "call_id": "call-1",
        "toolkit": "openai",
        "tool": "web_search",
        "arguments": {"query": "meshagent"},
        "result": {"type": "json", "json": {"results": [{"title": "MeshAgent"}]}},
        "error": None,
        "logs": [{"source": "stdout", "text": "searching\n"}],
    }
    assert context.messages[3] == {
        "role": "assistant",
        "content": [{"type": "event", "event": {"kind": "shell", "cmd": "pwd"}}],
    }
    assert context.metadata["last_compaction"] == {
        "checkpoint_id": "checkpoint-1",
        "path": "dataset://threads/main",
        "through_sequence": 42,
        "created_at": "2026-05-05T00:00:00Z",
    }
    assert [event["type"] for event in context.metadata["agent_events"]] == [
        AGENT_MESSAGE_TURN_START,
        AGENT_EVENT_TEXT_CONTENT_DELTA,
        AGENT_EVENT_TEXT_CONTENT_DELTA,
        AGENT_EVENT_TEXT_CONTENT_ENDED,
        AGENT_EVENT_TOOL_CALL_STARTED,
        AGENT_EVENT_TOOL_CALL_LOG_DELTA,
        AGENT_EVENT_TOOL_CALL_ENDED,
        AGENT_EVENT_THREAD_EVENT,
        AGENT_EVENT_CONTEXT_COMPACTED,
    ]
