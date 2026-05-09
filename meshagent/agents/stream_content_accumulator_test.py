from meshagent.agents.stream_content_accumulator import (
    FileContentAccumulator,
    TextContentAccumulator,
)


def test_text_content_accumulator_coalesces_streamed_text() -> None:
    accumulator = TextContentAccumulator()

    accumulator.upsert(item_id="msg-1", turn_id="turn-1", phase="commentary")
    first = accumulator.append_delta(item_id="msg-1", delta="hello ")
    second = accumulator.append_delta(
        item_id="msg-1",
        delta="world",
        sender_name="assistant",
    )

    assert first.text == "hello "
    assert second.text == "hello world"
    assert second.turn_id == "turn-1"
    assert second.status == "in_progress"
    assert second.sender_name == "assistant"
    assert second.phase == "commentary"
    completed = accumulator.complete(item_id="msg-1")
    assert completed is not None
    assert completed.status == "completed"
    assert accumulator.remove("msg-1") == completed
    assert accumulator.get("msg-1") is None


def test_file_content_accumulator_coalesces_urls_without_duplicates() -> None:
    accumulator = FileContentAccumulator()

    accumulator.upsert(item_id="file-1", turn_id="turn-1")
    first = accumulator.append_url(
        item_id="file-1", url=" mesh://one ", sender_name="assistant"
    )
    second = accumulator.append_url(item_id="file-1", url="mesh://two")
    duplicate = accumulator.append_url(item_id="file-1", url="mesh://two")

    assert first.urls == ("mesh://one",)
    assert second.urls == ("mesh://one", "mesh://two")
    assert duplicate.urls == second.urls
    assert duplicate.latest_url == "mesh://two"
    assert duplicate.turn_id == "turn-1"
    assert duplicate.status == "in_progress"
    assert duplicate.sender_name == "assistant"
    completed = accumulator.complete(item_id="file-1")
    assert completed is not None
    assert completed.status == "completed"
