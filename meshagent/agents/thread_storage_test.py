import pytest

from meshagent.agents.thread_storage import thread_dir_for_namespace


def test_thread_dir_for_namespace_returns_base_thread_dir_without_namespace() -> None:
    assert (
        thread_dir_for_namespace(thread_dir=" /threads/ ", namespace=None) == "threads"
    )


def test_thread_dir_for_namespace_appends_namespace() -> None:
    assert (
        thread_dir_for_namespace(
            thread_dir="threads",
            namespace="jesse.ezell@timu.com",
        )
        == "threads/jesse.ezell@timu.com"
    )


def test_thread_dir_for_namespace_rejects_empty_thread_dir() -> None:
    with pytest.raises(ValueError, match="thread_dir must not be empty"):
        thread_dir_for_namespace(thread_dir=" / ", namespace=None)


def test_thread_dir_for_namespace_rejects_relative_namespace_parts() -> None:
    with pytest.raises(
        ValueError,
        match="namespace must not contain empty or relative path parts",
    ):
        thread_dir_for_namespace(thread_dir="threads", namespace="../other")
