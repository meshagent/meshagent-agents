from __future__ import annotations

from typing import Any

from meshagent.agents.agent_event_reader import (
    AccumulatingAgentEventReader,
    AgentEventReaderCallbacks,
    _BufferedToolCall,
)
from meshagent.agents.context import AgentSessionContext, SessionUsage
from meshagent.agents.messages import (
    AGENT_EVENT_CONTEXT_COMPACTED,
    AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_LOG_DELTA,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_MESSAGE_TURN_START,
    AgentContextCompacted,
    AgentGeneratedImage,
    AgentImageGenerationCompleted,
    AgentUsageUpdated,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentThreadEvent,
    AgentToolCallArgumentsDelta,
    AgentToolCallEnded,
    AgentToolCallLogDelta,
    AgentToolCallLogLine,
    AgentToolCallStarted,
    TurnStart,
)
from meshagent.api.messaging import JsonContent


class _TestAgentEventReader(AccumulatingAgentEventReader):
    def _append_user_text(self, text: str) -> None:
        self._emit_context_message({"role": "user", "content": text})

    def _append_user_content(self, content: list[dict[str, Any]]) -> None:
        self._emit_context_message({"role": "user", "content": content})

    def _append_assistant_text(self, *, text: str, phase: str | None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if phase is not None:
            message["phase"] = phase
        self._emit_context_message(message)

    def _append_assistant_reasoning(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        self._emit_context_message({"role": "assistant", "reasoning": text})

    def _append_assistant_file(self, *, url: str) -> None:
        self._emit_context_message({"role": "assistant", "file": url})

    def _append_thread_event(self, *, event: dict[str, Any]) -> None:
        self._emit_context_message({"role": "assistant", "event": event})

    def _append_tool_call(
        self,
        *,
        tool_call: _BufferedToolCall,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "tool_call": {
                    "item_id": tool_call.item_id,
                    "namespace": tool_call.namespace,
                    "call_id": tool_call.call_id,
                    "toolkit": tool_call.toolkit,
                    "tool": tool_call.tool,
                    "arguments_json": tool_call.arguments_json(),
                    "arguments": tool_call.arguments_dict(),
                    "result": result,
                    "error": error,
                    "logs": tool_call.logs,
                },
            }
        )

    def _append_image_generation_event(
        self,
        *,
        event_type: str,
        turn_id: str,
        item_id: str,
        call_id: str | None,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        images: list[dict[str, Any]],
        status: str,
    ) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "image_generation": {
                    "event_type": event_type,
                    "turn_id": turn_id,
                    "item_id": item_id,
                    "call_id": call_id,
                    "toolkit": toolkit,
                    "tool": tool,
                    "arguments": arguments,
                    "images": images,
                    "status": status,
                },
            }
        )

    def _append_audio_generation_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {"role": "assistant", "audio_generation": message.model_dump(mode="json")}
        )

    def _append_audio_transcription_event(self, *, message: Any) -> None:
        self._emit_context_message(
            {
                "role": "assistant",
                "audio_transcription": message.model_dump(mode="json"),
            }
        )

    def _restore_compacted_messages(self, *, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self._emit_context_message(message)


def _reader_for_context(context: AgentSessionContext) -> _TestAgentEventReader:
    def record_event(message: Any) -> None:
        context.metadata.setdefault("agent_events", []).append(
            message.model_dump(mode="json")
        )

    def update_usage(message: AgentUsageUpdated) -> None:
        context.last_usage = SessionUsage(
            model="",
            usage=dict(message.usage),
            context_window_used=message.context_window.used_tokens,
            context_window_size=message.context_window.total_tokens,
        )

    def restore_compacted_context(message: AgentContextCompacted) -> None:
        context.metadata["last_compaction"] = {
            "checkpoint_id": message.checkpoint_id,
            "path": message.path,
            "through_sequence": message.through_sequence,
            "created_at": message.created_at,
        }
        if message.messages is not None:
            context.messages.clear()

    return _TestAgentEventReader(
        emit_message=context.messages.append,
        callbacks=AgentEventReaderCallbacks(
            record_event=record_event,
            update_usage=update_usage,
            restore_compacted_context=restore_compacted_context,
        ),
    )


def test_accumulating_agent_event_reader_preserves_image_generation_turn_id() -> None:
    messages: list[dict[str, Any]] = []
    reader = _TestAgentEventReader(emit_message=messages.append)

    reader(
        AgentImageGenerationCompleted(
            type=AGENT_EVENT_IMAGE_GENERATION_COMPLETED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="image-1",
            call_id="call-image",
            toolkit="openai",
            tool="image_generation",
            arguments={"size": "512x512"},
            images=[
                AgentGeneratedImage(
                    uri="dataset://images?id=image-1",
                    status="completed",
                )
            ],
        )
    )

    image_generation = messages[0]["image_generation"]
    assert image_generation["turn_id"] == "turn-1"
    assert image_generation["item_id"] == "image-1"


def test_accumulating_agent_event_reader_roundtrips_core_agent_messages() -> None:
    context = AgentSessionContext()
    reader = _reader_for_context(context)

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
        AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            delta='{"query"',
        )
    )
    reader.consume(
        AgentToolCallArgumentsDelta(
            type=AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="openai.responses",
            call_id="call-1",
            delta=':"meshagent"}',
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
    tool_call = context.messages[2]["tool_call"]
    assert tool_call == {
        "item_id": "tool-1",
        "namespace": "openai.responses",
        "call_id": "call-1",
        "toolkit": "openai",
        "tool": "web_search",
        "arguments_json": '{"query":"meshagent"}',
        "arguments": {"query": "meshagent"},
        "result": {"type": "json", "json": {"results": [{"title": "MeshAgent"}]}},
        "error": None,
        "logs": [{"source": "stdout", "text": "searching\n"}],
    }
    assert context.messages[3] == {
        "role": "assistant",
        "event": {"kind": "shell", "cmd": "pwd"},
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
        AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
        AGENT_EVENT_TOOL_CALL_ARGUMENTS_DELTA,
        AGENT_EVENT_TOOL_CALL_STARTED,
        AGENT_EVENT_TOOL_CALL_LOG_DELTA,
        AGENT_EVENT_TOOL_CALL_ENDED,
        AGENT_EVENT_THREAD_EVENT,
        AGENT_EVENT_CONTEXT_COMPACTED,
    ]


def test_accumulating_agent_event_reader_preserves_commentary_phase() -> None:
    context = AgentSessionContext()
    reader = _reader_for_context(context)

    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="checking",
            phase="commentary",
        )
    )
    reader.consume(
        AgentTextContentEnded(
            type=AGENT_EVENT_TEXT_CONTENT_ENDED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            phase="commentary",
        )
    )

    assert context.messages == [
        {"role": "assistant", "content": "checking", "phase": "commentary"}
    ]


def test_accumulating_agent_event_reader_cancels_incomplete_tool_call() -> None:
    messages: list[dict[str, Any]] = []
    reader = _TestAgentEventReader(emit_message=messages.append)

    reader.consume(
        AgentToolCallStarted(
            type=AGENT_EVENT_TOOL_CALL_STARTED,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="tool-1",
            namespace="meshagent",
            call_id="call-1",
            toolkit="test",
            tool="blocking_tool",
            arguments={"marker": "cancel-me"},
        )
    )
    reader.finalize()

    assert messages[0]["tool_call"]["error"] == {
        "message": "tool call was cancelled before completion",
        "code": "cancelled",
    }


def test_accumulating_agent_event_reader_treats_final_text_snapshot_as_replacement() -> (
    None
):
    context = AgentSessionContext()
    reader = _reader_for_context(context)

    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi",
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text=" there",
        )
    )
    reader.consume(
        AgentTextContentDelta(
            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
            thread_id="thread-1",
            turn_id="turn-1",
            item_id="text-1",
            text="Hi there",
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

    assert context.messages == [{"role": "assistant", "content": "Hi there"}]
