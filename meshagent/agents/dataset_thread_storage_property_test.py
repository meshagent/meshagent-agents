from __future__ import annotations

import asyncio
import os
import string
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase

from meshagent.agents.context import AgentSessionContext
from meshagent.agents.dataset_thread_storage import DatasetThreadStorage
from meshagent.agents.dataset_thread_storage_test import (
    _FakeRoom,
    _participant,
    _row_data,
    _test_llm_adapter,
)
from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_MESSAGE_TURN_START,
    AgentError,
    AgentTextContent,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallEnded,
    AgentToolCallStarted,
    TurnEnded,
    TurnInterrupted,
    TurnStart,
    TurnStartAccepted,
)


def _property_settings() -> settings:
    profile = os.getenv("MESHAGENT_AGENT_PROPERTY_PROFILE", "pr").strip().lower()
    max_examples = 500 if profile == "full" else 40
    database_path = Path(
        os.getenv(
            "MESHAGENT_AGENT_PROPERTY_DATABASE",
            ".hypothesis/meshagent-agent-storage",
        )
    )
    return settings(
        max_examples=max_examples,
        deadline=None,
        database=DirectoryBasedExampleDatabase(database_path),
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow],
    )


_TURN_SPEC = st.fixed_dictionaries(
    {
        "text": st.text(
            alphabet=string.ascii_lowercase + string.digits + " ",
            min_size=1,
            max_size=24,
        ),
        "delta_count": st.integers(min_value=1, max_value=4),
        "tool_outcome": st.sampled_from(("none", "success", "failure", "cancel")),
        "flush_after": st.booleans(),
    }
)


def _prefixes(text: str, count: int) -> list[str]:
    return [
        text[: max(1, (len(text) * index + count - 1) // count)]
        for index in range(1, count + 1)
    ]


async def _run_append_log_scenario(turns: list[dict[str, Any]]) -> None:
    path = "dataset://threads/property"
    room = _FakeRoom()
    storage = DatasetThreadStorage(room=room, path=path)
    await storage.start()
    await storage.wait_until_ready()

    for index, spec in enumerate(turns):
        turn_id = f"turn-{index}"
        source_message_id = f"user-{index}"
        text_item_id = f"text-{index}"
        storage.push_message(
            message=TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                message_id=source_message_id,
                thread_id=path,
                content=[AgentTextContent(type="text", text=spec["text"])],
            ),
            sender=_participant(f"user-{index % 2}"),
        )
        storage.push_message(
            message=TurnStartAccepted(
                type=AGENT_EVENT_TURN_START_ACCEPTED,
                message_id=f"accepted-{index}",
                thread_id=path,
                turn_id=turn_id,
                source_message_id=source_message_id,
            )
        )
        storage.push_message(
            message=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                message_id=f"text-started-{index}",
                thread_id=path,
                turn_id=turn_id,
                item_id=text_item_id,
            )
        )
        for delta_index, text in enumerate(
            _prefixes(spec["text"], spec["delta_count"])
        ):
            storage.push_message(
                message=AgentTextContentDelta(
                    type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                    message_id=f"text-delta-{index}-{delta_index}",
                    thread_id=path,
                    turn_id=turn_id,
                    item_id=text_item_id,
                    text=text,
                )
            )
        storage.push_message(
            message=AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                message_id=f"text-ended-{index}",
                thread_id=path,
                turn_id=turn_id,
                item_id=text_item_id,
            )
        )

        tool_outcome = spec["tool_outcome"]
        if tool_outcome != "none":
            tool_item_id = f"tool-{index}"
            storage.push_message(
                message=AgentToolCallStarted(
                    type=AGENT_EVENT_TOOL_CALL_STARTED,
                    message_id=f"tool-started-{index}",
                    thread_id=path,
                    turn_id=turn_id,
                    item_id=tool_item_id,
                    call_id=f"call-{index}",
                    toolkit="property",
                    tool="probe",
                    arguments={"turn": index},
                )
            )
            if tool_outcome in {"success", "failure"}:
                storage.push_message(
                    message=AgentToolCallEnded(
                        type=AGENT_EVENT_TOOL_CALL_ENDED,
                        message_id=f"tool-ended-{index}",
                        thread_id=path,
                        turn_id=turn_id,
                        item_id=tool_item_id,
                        call_id=f"call-{index}",
                        toolkit="property",
                        tool="probe",
                        error=(
                            AgentError(message="generated failure", code="generated")
                            if tool_outcome == "failure"
                            else None
                        ),
                    )
                )
            else:
                storage.push_message(
                    message=TurnInterrupted(
                        type=AGENT_EVENT_TURN_INTERRUPTED,
                        message_id=f"interrupted-{index}",
                        thread_id=path,
                        turn_id=turn_id,
                        source_message_id=f"interrupt-{index}",
                    )
                )

        storage.push_message(
            message=TurnEnded(
                type=AGENT_EVENT_TURN_ENDED,
                message_id=f"turn-ended-{index}",
                thread_id=path,
                turn_id=turn_id,
                error=None,
            )
        )
        if spec["flush_after"]:
            await storage.flush()

    await storage.stop()

    rows = room.datasets.rows[(("threads",), "property")]
    sequences = [row["sequence"] for row in rows]
    assert sequences == list(range(len(rows)))
    assert len(sequences) == len(set(sequences))
    assert room.datasets.merge_calls == []
    assert room.datasets.update_calls == []
    inserted_rows = [
        record for call in room.datasets.insert_calls for record in call["records"]
    ]
    assert [record["sequence"] for record in inserted_rows] == sequences

    row_data = [_row_data(row) for row in rows]
    for index, spec in enumerate(turns):
        turn_id = f"turn-{index}"
        turn_rows = [row for row in row_data if row.get("turn_id") == turn_id]
        turn_types = [row["type"] for row in turn_rows]
        assert AGENT_EVENT_TURN_START_ACCEPTED in turn_types
        assert AGENT_EVENT_TURN_ENDED in turn_types
        if spec["tool_outcome"] != "none":
            started_index = turn_types.index(AGENT_EVENT_TOOL_CALL_STARTED)
            ended_index = turn_types.index(AGENT_EVENT_TOOL_CALL_ENDED)
            assert started_index < ended_index
            ended = turn_rows[ended_index]
            if spec["tool_outcome"] == "cancel":
                assert ended["error"]["code"] == "cancelled"
                assert AGENT_EVENT_TURN_INTERRUPTED in turn_types

    restored = DatasetThreadStorage(room=room, path=path)
    await restored.start()
    await restored.wait_until_ready()
    try:
        restored_types = [message.type for message in restored.agent_messages()]
        assert restored_types == [row["type"] for row in row_data]
        contexts: list[list[dict[str, Any]]] = []
        for _ in range(2):
            context = AgentSessionContext(system_role=None)
            restored.restore_session_context(
                context=context,
                llm_adapter=_test_llm_adapter(),
            )
            contexts.append(context.messages)
        assert contexts[0] == contexts[1]
    finally:
        await restored.stop()


@_property_settings()
@given(turns=st.lists(_TURN_SPEC, min_size=1, max_size=8))
def test_dataset_thread_storage_generated_append_restore_invariants(
    turns: list[dict[str, Any]],
) -> None:
    asyncio.run(_run_append_log_scenario(turns))
