from meshagent.agents.tool_call_accumulator import ToolCallAccumulator


def test_tool_call_accumulator_uses_stable_apply_patch_text_before_path() -> None:
    accumulator = ToolCallAccumulator()

    pending = accumulator.upsert_lifecycle(
        item_id="patch-1",
        toolkit="openai",
        tool="apply_patch",
        arguments={},
        state="pending",
    )

    assert pending.text == "Preparing patch"

    streaming = accumulator.append_delta(
        item_id="patch-1",
        delta="@@\n-old\n+new\n",
    )

    assert streaming is not None
    assert streaming.text == "Preparing patch"
    assert streaming.lines_added == 1
    assert streaming.lines_removed == 1


def test_tool_call_accumulator_tracks_apply_patch_operation_and_deltas() -> None:
    accumulator = ToolCallAccumulator()

    pending = accumulator.upsert_lifecycle(
        item_id="patch-1",
        toolkit="openai",
        tool="apply_patch",
        arguments={
            "operation": {
                "type": "update_file",
                "path": "report.py",
                "diff": "",
            }
        },
        state="pending",
    )

    assert pending.text == "Editing report.py"
    assert pending.lines_added is None
    assert pending.lines_removed is None

    streaming = accumulator.append_delta(
        item_id="patch-1",
        delta="@@\n-old\n+new\n+extra\n",
    )

    assert streaming is not None
    assert streaming.text == "Editing report.py"
    assert streaming.lines_added == 2
    assert streaming.lines_removed == 1
    assert accumulator.get("patch-1").state == "pending"

    completed = accumulator.complete(item_id="patch-1")
    assert completed is not None
    assert completed.text == "Edited report.py"
    assert accumulator.get("patch-1").state == "completed"


def test_tool_call_accumulator_keeps_delta_messages_lean_and_correlates_by_item() -> (
    None
):
    accumulator = ToolCallAccumulator()

    assert (
        accumulator.append_delta(
            item_id="shell-1",
            delta="x" * 120,
        )
        is None
    )

    status = accumulator.upsert_lifecycle(
        item_id="shell-1",
        toolkit="openai",
        tool="shell",
        arguments={},
        state="pending",
    )

    assert status.text == "Preparing"
    assert status.total_bytes is not None
    assert status.total_bytes >= 120
