from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_core import from_json as pydantic_core_from_json

from .shell_semantics import analyze_shell_command

_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
ToolCallState = Literal["pending", "in_progress", "completed", "failed", "cancelled"]
_APPLY_PATCH_PATH_RES = (
    re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (?P<path>.+)$", re.MULTILINE),
    re.compile(r"^(?:\+\+\+ b/|--- a/)(?P<path>.+)$", re.MULTILINE),
)


@dataclass(frozen=True, slots=True)
class ToolCallStatusSnapshot:
    text: str
    item_id: str
    total_bytes: int | None = None
    lines_added: int | None = None
    lines_removed: int | None = None


@dataclass(frozen=True, slots=True)
class AccumulatedToolCall:
    item_id: str
    toolkit: str
    tool: str
    arguments: dict[str, Any] | None
    state: ToolCallState
    argument_bytes: int = 0
    argument_delta_text: str = ""


def status_total_bytes(total_bytes: int) -> int | None:
    return total_bytes if total_bytes > 100 else None


def _merge_tool_arguments(
    *, current: dict[str, Any] | None, update: dict[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(current) if current is not None else {}
    for key, value in update.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_tool_arguments(current=existing, update=value)
        else:
            merged[key] = value
    return merged


def _partial_json_tool_arguments(text: str) -> dict[str, Any] | None:
    if text.strip() == "":
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        try:
            decoded = pydantic_core_from_json(text.encode("utf-8"), allow_partial=True)
        except ValueError:
            return None
    return decoded if isinstance(decoded, dict) else None


def _command_text(*, value: Any, multiline: bool = False) -> str:
    if isinstance(value, str):
        return value.strip() if not multiline else value.strip("\n")
    if isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
        return "\n".join(values).strip("\n") if multiline else " ".join(values).strip()
    if isinstance(value, dict):
        for key in ("command", "cmd", "script"):
            nested = value.get(key)
            text = _command_text(value=nested, multiline=multiline or key == "commands")
            if text != "":
                return text
        nested = value.get("content")
        if nested is not None:
            return _command_text(value=nested, multiline=multiline)
    return ""


def first_nested_text(*, value: Any, keys: tuple[str, ...]) -> str:
    key_set = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() not in key_set:
                continue
            text = _command_text(value=nested, multiline=key.endswith("s"))
            if text != "":
                return text
        for nested in value.values():
            text = first_nested_text(value=nested, keys=keys)
            if text != "":
                return text
    if isinstance(value, list):
        for nested in value:
            text = first_nested_text(value=nested, keys=keys)
            if text != "":
                return text
    return ""


def _shell_delta_arguments(
    *, current: dict[str, Any] | None, command: str
) -> dict[str, Any]:
    merged = deepcopy(current) if current is not None else {}
    action = merged.get("action")
    if isinstance(action, dict):
        existing_commands = action.get("commands")
        if isinstance(existing_commands, list):
            action["commands"] = [command]
        else:
            action["command"] = command
        return merged
    merged["command"] = command
    return merged


def tool_arguments_from_delta_text(
    *, tool: str, current: dict[str, Any] | None, text: str
) -> dict[str, Any] | None:
    partial_arguments = _partial_json_tool_arguments(text)
    if partial_arguments is not None:
        return _merge_tool_arguments(current=current, update=partial_arguments)

    normalized_tool = tool.strip().lower()
    if normalized_tool == "apply_patch":
        patch = text.strip()
        if patch != "":
            return _merge_tool_arguments(current=current, update={"patch": patch})

    if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
        command = text.strip()
        if command != "":
            return _shell_delta_arguments(current=current, command=command)
    return None


def _extract_tool_command(*, tool: str, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    normalized_tool = tool.strip().lower()
    if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
        return first_nested_text(
            value=arguments,
            keys=("command", "commands", "cmd", "script"),
        )
    return ""


def _extract_apply_patch_text(*, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    return first_nested_text(value=arguments, keys=("patch", "input", "diff"))


def _extract_apply_patch_path(*, arguments: dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    return first_nested_text(value=arguments, keys=("path",))


def _apply_patch_path(*, patch: str) -> str:
    for pattern in _APPLY_PATCH_PATH_RES:
        match = pattern.search(patch)
        if match is None:
            continue
        path = match.group("path").strip()
        if path != "":
            return path
    return ""


def _patch_line_counts(*, patch: str) -> tuple[int | None, int | None]:
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if added == 0 and removed == 0:
        return None, None
    return added, removed


def _apply_patch_line_counts(
    *, tool: str, arguments: dict[str, Any] | None
) -> tuple[int | None, int | None]:
    if tool.strip().lower() != "apply_patch":
        return None, None
    return _patch_line_counts(patch=_extract_apply_patch_text(arguments=arguments))


def tool_status_text(
    *, state: str, toolkit: str, tool: str, arguments: dict[str, Any] | None
) -> str:
    normalized_tool = tool.strip().lower()
    if normalized_tool in {"shell", "local_shell", "code_interpreter"}:
        command = _extract_tool_command(tool=normalized_tool, arguments=arguments)
        if state == "pending" and command == "":
            return "Preparing"
        return (
            analyze_shell_command(command=command)
            .display.phase_for_state(state=state)
            .headline
        )

    if normalized_tool == "apply_patch":
        patch = _extract_apply_patch_text(arguments=arguments)
        path = _extract_apply_patch_path(arguments=arguments) or _apply_patch_path(
            patch=patch
        )
        if path != "":
            if state == "pending":
                return f"Editing {path}"
            if state == "failed":
                return f"Attempted to patch {path}"
            if state == "cancelled":
                return f"Patch cancelled: {path}"
            if state in _ACTIVE_STATES:
                return f"Editing {path}"
            return f"Edited {path}"
        if state == "pending":
            return "Preparing patch"
        if state == "failed":
            return "Attempted to patch"
        if state == "cancelled":
            return "Patch cancelled"
        return "Applying patch" if state in _ACTIVE_STATES else "Applied patch"

    humanized = tool.strip().replace("_", " ").replace("-", " ")
    if state == "pending":
        return f"Preparing {humanized}" if humanized != "" else "Preparing tool call"
    if state in _ACTIVE_STATES:
        return f"Calling {humanized}" if humanized != "" else "Calling tool"
    if state == "failed":
        return (
            f"Attempted to call {humanized}"
            if humanized != ""
            else "Attempted to call tool"
        )
    if state == "cancelled":
        return f"{humanized} cancelled" if humanized != "" else "Tool call cancelled"
    return f"Called {humanized}" if humanized != "" else "Called tool"


class ToolCallAccumulator:
    def __init__(self) -> None:
        self._calls: dict[str, AccumulatedToolCall] = {}
        self._delta_bytes_by_item_id: dict[str, int] = {}
        self._delta_text_by_item_id: dict[str, str] = {}

    def clear(self) -> None:
        self._calls.clear()
        self._delta_bytes_by_item_id.clear()
        self._delta_text_by_item_id.clear()

    def get(self, item_id: str) -> AccumulatedToolCall | None:
        return self._calls.get(item_id)

    def total_bytes(self, item_id: str) -> int:
        return self._delta_bytes_by_item_id.get(item_id, 0)

    def remove(self, item_id: str) -> AccumulatedToolCall | None:
        self._delta_bytes_by_item_id.pop(item_id, None)
        self._delta_text_by_item_id.pop(item_id, None)
        return self._calls.pop(item_id, None)

    def upsert_lifecycle(
        self,
        *,
        item_id: str,
        toolkit: str,
        tool: str,
        arguments: dict[str, Any] | None,
        state: Literal["pending", "in_progress"],
        argument_bytes: int | None = None,
    ) -> ToolCallStatusSnapshot:
        existing = self._calls.get(item_id)
        delta_text = self._delta_text_by_item_id.get(item_id, "")
        merged_arguments = (
            arguments
            if arguments is not None
            else existing.arguments
            if existing
            else None
        )
        if delta_text != "":
            updated = tool_arguments_from_delta_text(
                tool=tool,
                current=merged_arguments,
                text=delta_text,
            )
            if updated is not None:
                merged_arguments = updated
        total_bytes = max(
            self._delta_bytes_by_item_id.get(item_id, 0),
            _json_snapshot_bytes(merged_arguments),
            argument_bytes or 0,
        )
        if total_bytes > 0:
            self._delta_bytes_by_item_id[item_id] = total_bytes
        call = AccumulatedToolCall(
            item_id=item_id,
            toolkit=toolkit,
            tool=tool,
            arguments=merged_arguments,
            state=state,
            argument_bytes=total_bytes,
            argument_delta_text=delta_text,
        )
        self._calls[item_id] = call
        return self.status_for(item_id)

    def append_delta(
        self,
        *,
        item_id: str,
        delta: str,
    ) -> ToolCallStatusSnapshot | None:
        if delta == "":
            return None
        self._delta_bytes_by_item_id[item_id] = self._delta_bytes_by_item_id.get(
            item_id, 0
        ) + len(delta.encode("utf-8"))
        delta_text = self._delta_text_by_item_id.get(item_id, "") + delta
        self._delta_text_by_item_id[item_id] = delta_text
        call = self._calls.get(item_id)
        if call is None:
            return None
        updated_arguments = tool_arguments_from_delta_text(
            tool=call.tool,
            current=call.arguments,
            text=delta_text,
        )
        next_call = AccumulatedToolCall(
            item_id=item_id,
            toolkit=call.toolkit,
            tool=call.tool,
            arguments=updated_arguments
            if updated_arguments is not None
            else call.arguments,
            state=call.state,
            argument_bytes=self._delta_bytes_by_item_id[item_id],
            argument_delta_text=delta_text,
        )
        self._calls[item_id] = next_call
        return self.status_for(item_id)

    def complete(
        self,
        *,
        item_id: str,
        state: Literal["completed", "failed", "cancelled"] = "completed",
    ) -> ToolCallStatusSnapshot | None:
        call = self._calls.get(item_id)
        if call is None:
            return None
        next_call = AccumulatedToolCall(
            item_id=item_id,
            toolkit=call.toolkit,
            tool=call.tool,
            arguments=call.arguments,
            state=state,
            argument_bytes=call.argument_bytes,
            argument_delta_text=call.argument_delta_text,
        )
        self._calls[item_id] = next_call
        return self.status_for(item_id)

    def status_for(self, item_id: str) -> ToolCallStatusSnapshot:
        call = self._calls[item_id]
        lines_added, lines_removed = _apply_patch_line_counts(
            tool=call.tool,
            arguments=call.arguments,
        )
        return ToolCallStatusSnapshot(
            text=tool_status_text(
                state=call.state,
                toolkit=call.toolkit,
                tool=call.tool,
                arguments=call.arguments,
            ),
            item_id=item_id,
            total_bytes=status_total_bytes(
                self._delta_bytes_by_item_id.get(item_id, 0)
            ),
            lines_added=lines_added,
            lines_removed=lines_removed,
        )


def _json_snapshot_bytes(arguments: dict[str, Any] | None) -> int:
    if arguments is None or len(arguments) == 0:
        return 0
    return len(
        json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
