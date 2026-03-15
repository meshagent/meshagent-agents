from __future__ import annotations

import ast
import contextlib
import posixpath
import shlex
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Literal

from .shell_parser import (
    ShellCommand,
    extract_heredoc_bodies,
    ShellParseError,
    ShellPipeline,
    ShellProgram,
    parse_shell_script,
)

ShellDisplayPhaseName = Literal["pending", "active", "completed", "failed", "cancelled"]
ShellEventKind = Literal["exec", "file"]
ShellOpKind = Literal[
    "build",
    "dev",
    "download",
    "edit",
    "explore",
    "install",
    "lint",
    "request",
    "run",
    "search",
    "script",
    "test",
    "write",
]
ShellExploreMode = Literal["explore", "list", "read"]

_ACTIVE_STATES = {"queued", "in_progress", "running", "pending", "searching"}
_DETAIL_LIMIT = 2000
_SUMMARY_LIMIT = 280
_SEARCH_COMMANDS = {"egrep", "fgrep", "grep", "rg"}
_READ_COMMANDS = {"bat", "cat", "head", "less", "more", "sed", "tail"}
_LIST_COMMANDS = {"ls", "tree"}
_EXPLORE_ONLY_COMMANDS = {"echo", "find", "printf", "pwd", "stat"}
_SEARCH_QUERY_FLAGS = {"-e", "--regexp"}
_SEARCH_SHORT_VALUE_FLAGS = {
    "-A",
    "-B",
    "-C",
    "-D",
    "-M",
    "-T",
    "-f",
    "-g",
    "-j",
    "-m",
    "-t",
}
_READ_VALUE_FLAGS = {"-c", "-e", "-f", "-n"}
_LIST_VALUE_FLAGS = {"-I", "--ignore"}
_SEARCH_LONG_VALUE_FLAGS = {
    "--binary-files",
    "--color",
    "--colors",
    "--context",
    "--context-separator",
    "--devices",
    "--directories",
    "--encoding",
    "--engine",
    "--exclude",
    "--exclude-dir",
    "--file",
    "--glob",
    "--include",
    "--label",
    "--max-columns",
    "--max-count",
    "--max-filesize",
    "--path-separator",
    "--pre",
    "--pre-glob",
    "--replace",
    "--sort",
    "--sortr",
    "--threads",
    "--type",
    "--type-add",
    "--type-not",
}
_DOWNLOAD_COMMANDS = {"curl", "wget"}
_DIRECT_WRITE_COMMANDS = {"cat", "echo", "printf"}
_COMBINED_CONTEXTUAL_EXEC_KINDS: tuple[ShellOpKind, ...] = (
    "install",
    "build",
    "lint",
    "test",
    "dev",
)
_BUILD_TOOLS = {
    "astro",
    "cmake",
    "esbuild",
    "grunt",
    "gulp",
    "next",
    "nuxt",
    "nx",
    "parcel",
    "rollup",
    "rspack",
    "swc",
    "tsc",
    "turbo",
    "vite",
    "webpack",
    "webpack-cli",
}
_DEV_TOOLS = {
    "astro",
    "browser-sync",
    "next",
    "nuxt",
    "nx",
    "parcel",
    "rollup",
    "rspack",
    "tsc",
    "turbo",
    "vite",
    "webpack",
    "webpack-cli",
}
_TEST_TOOLS = {
    "ava",
    "c8",
    "cargo",
    "cypress",
    "dotnet",
    "go",
    "jest",
    "mocha",
    "mvn",
    "playwright",
    "pytest",
    "tap",
    "uvu",
    "vitest",
}
_LINT_TOOLS = {
    "biome",
    "black",
    "cargo",
    "clippy",
    "eslint",
    "flake8",
    "golangci-lint",
    "isort",
    "markdownlint",
    "mypy",
    "prettier",
    "pylint",
    "ruff",
    "shellcheck",
    "stylelint",
}
_LISTING_SEARCH_TOOLS = {"fd", "fd-find"}
_GIT_EXPLORE_SUBCOMMANDS = {"diff", "log", "show", "status"}
_GIT_SEARCH_SUBCOMMANDS = {"grep"}
_INSTALLING_SUBCOMMANDS = {"add", "ci", "i", "install"}
_TESTING_SUBCOMMANDS = {"test"}
_LINTING_SUBCOMMANDS = {"check", "fmt", "format", "lint"}
_DEV_SUBCOMMANDS = {"dev", "serve", "start", "watch"}
_BUILD_SUBCOMMANDS = {"build"}
_PYTHON_LAUNCHERS = {"py", "python", "python3"}
_NODE_LAUNCHERS = {"node"}
_SCRIPT_TOOL_NAMES = {
    "deno": "Deno",
    "lua": "Lua",
    "node": "Node.js",
    "perl": "Perl",
    "php": "PHP",
    "py": "Python",
    "python": "Python",
    "python3": "Python",
    "ruby": "Ruby",
}


@dataclass(frozen=True, slots=True)
class ShellDisplayPhase:
    headline: str
    summary: str


@dataclass(frozen=True, slots=True)
class ShellDisplay:
    event_kind: ShellEventKind
    path: str = ""
    details: tuple[str, ...] = ()
    preview: str = ""
    coalesce_path: str = ""
    pending: ShellDisplayPhase = ShellDisplayPhase("", "")
    active: ShellDisplayPhase = ShellDisplayPhase("", "")
    completed: ShellDisplayPhase = ShellDisplayPhase("", "")
    failed: ShellDisplayPhase = ShellDisplayPhase("", "")
    cancelled: ShellDisplayPhase = ShellDisplayPhase("", "")

    def phase_for_state(self, *, state: str) -> ShellDisplayPhase:
        phase = phase_name_for_state(state=state)
        if phase == "pending":
            return self.pending
        if phase == "failed":
            return self.failed
        if phase == "cancelled":
            return self.cancelled
        if phase == "completed":
            return self.completed
        return self.active


@dataclass(frozen=True, slots=True)
class ShellOp:
    kind: ShellOpKind
    path: str = ""
    paths: tuple[str, ...] = ()
    command: str = ""
    query: str = ""
    append: bool = False
    multi: bool = False
    mode: ShellExploreMode | None = None
    tool: str = ""


@dataclass(frozen=True, slots=True)
class ShellCommandAnalysis:
    command: str
    script: str
    cwd: str | None
    operations: tuple[ShellOp, ...]
    display: ShellDisplay


@dataclass(frozen=True, slots=True)
class _CommandView:
    executable: str
    argv: tuple[str, ...]
    effective_executable: str
    effective_argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PipelineContext:
    pipeline: ShellPipeline
    cwd: str | None


def phase_name_for_state(*, state: str) -> ShellDisplayPhaseName:
    if state == "pending":
        return "pending"
    if state == "failed":
        return "failed"
    if state == "cancelled":
        return "cancelled"
    if state == "completed":
        return "completed"
    if state in _ACTIVE_STATES:
        return "active"
    return "completed"


def analyze_shell_command(*, command: str) -> ShellCommandAnalysis:
    script = _extract_exec_script(command=command).strip()
    if script == "":
        display = _run_display(command=command, cwd=None)
        return ShellCommandAnalysis(
            command=command,
            script="",
            cwd=None,
            operations=(),
            display=display,
        )

    try:
        program = parse_shell_script(script=script)
    except ShellParseError:
        display = _run_display(command=script, cwd=None)
        return ShellCommandAnalysis(
            command=command,
            script=script,
            cwd=None,
            operations=(),
            display=display,
        )

    pipeline_contexts, cwd = _compile_pipeline_contexts(program=program)
    operations: list[ShellOp] = []

    write_op = _collect_write_operation(
        pipeline_contexts=pipeline_contexts, script=script
    )
    if write_op is not None:
        operations.append(write_op)

    if write_op is None and _should_coalesce_exploration(
        pipeline_contexts=pipeline_contexts
    ):
        explore_path = pipeline_contexts[0].cwd or ""
        operations.append(
            ShellOp(
                kind="explore",
                mode="explore",
                path=explore_path,
                command=script,
            )
        )
    else:
        for context in pipeline_contexts:
            if _pipeline_is_write(pipeline=context.pipeline, cwd=context.cwd):
                continue
            op = _classify_pipeline(pipeline=context.pipeline, cwd=context.cwd)
            if op is not None:
                operations.append(op)

    normalized_operations = tuple(_dedupe_operations(operations=operations))
    display = _display_for_operations(
        command=script,
        cwd=cwd,
        operations=normalized_operations,
    )
    display = replace(display, preview=_command_preview(command=script))
    return ShellCommandAnalysis(
        command=command,
        script=script,
        cwd=cwd,
        operations=normalized_operations,
        display=display,
    )


def _truncate_text(*, text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _command_preview(*, command: str) -> str:
    normalized = command.strip()
    if normalized == "":
        return ""

    lines = normalized.splitlines()
    if len(lines) > 5:
        normalized = "\n".join(lines[:5]) + "\n..."

    return _truncate_text(text=normalized, limit=_DETAIL_LIMIT)


def _extract_exec_script(*, command: str) -> str:
    normalized = command.strip()
    if normalized == "":
        return ""

    with contextlib.suppress(ValueError):
        argv = shlex.split(normalized)
        shell_executable = Path(argv[0]).name.lower() if len(argv) > 0 else ""
        if shell_executable in {"bash", "dash", "ksh", "sh", "zsh"}:
            for index, token in enumerate(argv[1:-1], start=1):
                if not _is_shell_command_flag(token=token):
                    continue
                script = argv[index + 1]
                if isinstance(script, str) and script.strip() != "":
                    return script

    return normalized


def _is_shell_command_flag(*, token: str) -> bool:
    if len(token) <= 1 or not token.startswith("-"):
        return False

    flags = token[1:]
    return "c" in flags and all(flag in {"c", "l"} for flag in flags)


def _resolve_path(*, path: str, cwd: str | None) -> str:
    normalized = _unquote_shell_token(token=path)
    if normalized == "":
        return ""

    if cwd is not None and cwd != "":
        if normalized == ".":
            return cwd
        if not PurePosixPath(normalized).is_absolute():
            normalized = str(PurePosixPath(cwd) / normalized)

    return posixpath.normpath(normalized)


def _unquote_shell_token(*, token: str) -> str:
    normalized = token.strip()
    if normalized == "":
        return ""

    with contextlib.suppress(ValueError):
        parts = shlex.split(normalized)
        if len(parts) == 1:
            return parts[0]

    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        if normalized[0] in {"'", '"'}:
            return normalized[1:-1]

    return normalized


def _compile_pipeline_contexts(
    *,
    program: ShellProgram,
) -> tuple[list[_PipelineContext], str | None]:
    current_cwd: str | None = None
    pipeline_contexts: list[_PipelineContext] = []

    for item in program.items:
        updated_cwd = _pipeline_cd_target(pipeline=item.pipeline, cwd=current_cwd)
        if updated_cwd is not None:
            current_cwd = updated_cwd
            continue
        pipeline_contexts.append(
            _PipelineContext(pipeline=item.pipeline, cwd=current_cwd)
        )

    return pipeline_contexts, current_cwd


def _pipeline_cd_target(*, pipeline: ShellPipeline, cwd: str | None) -> str | None:
    if len(pipeline.commands) != 1:
        return None

    command = pipeline.first_command
    argv = command.argv
    if len(argv) < 2:
        return None
    if Path(argv[0]).name.lower() != "cd":
        return None
    return _resolve_path(path=argv[1], cwd=cwd)


def _should_coalesce_exploration(
    *,
    pipeline_contexts: list[_PipelineContext],
) -> bool:
    if len(pipeline_contexts) <= 1:
        return False
    if any(context.cwd in {None, ""} for context in pipeline_contexts):
        return False
    first_cwd = pipeline_contexts[0].cwd
    if any(context.cwd != first_cwd for context in pipeline_contexts[1:]):
        return False
    return all(
        _pipeline_is_exploration(pipeline=context.pipeline)
        for context in pipeline_contexts
    )


def _pipeline_is_exploration(*, pipeline: ShellPipeline) -> bool:
    executable = _command_executable(command=pipeline.first_command)
    return executable in {*_READ_COMMANDS, *_LIST_COMMANDS, *_EXPLORE_ONLY_COMMANDS}


def _pipeline_is_write(*, pipeline: ShellPipeline, cwd: str | None) -> bool:
    return any(
        _command_write_paths(command=command, cwd=cwd)[0]
        for command in pipeline.commands
    )


def _command_write_targets(*, command: ShellCommand) -> tuple[str, ...]:
    return tuple(
        redirection.target
        for redirection in command.redirections
        if redirection.operator in {">", ">>"}
    )


def _collect_write_operation(
    *,
    pipeline_contexts: list[_PipelineContext],
    script: str,
) -> ShellOp | None:
    discovered_paths: list[str] = []
    saw_append = False
    saw_multi = False
    heredoc_bodies = iter(extract_heredoc_bodies(script=script))

    for context in pipeline_contexts:
        for command in context.pipeline.commands:
            paths, append = _command_write_paths(command=command, cwd=context.cwd)
            if len(paths) == 0:
                body = (
                    next(heredoc_bodies, "")
                    if _command_has_heredoc(command=command)
                    else ""
                )
                embedded_paths, embedded_append, embedded_multi = (
                    _embedded_script_write_paths(
                        command=command,
                        body=body,
                        cwd=context.cwd,
                    )
                )
                if len(embedded_paths) == 0:
                    continue
                discovered_paths.extend(embedded_paths)
                saw_append = saw_append or embedded_append
                saw_multi = saw_multi or embedded_multi
                continue

            discovered_paths.extend(paths)
            saw_append = saw_append or append
            if _command_has_heredoc(command=command):
                next(heredoc_bodies, "")

    normalized_paths = tuple(
        dict.fromkeys(path for path in discovered_paths if path != "")
    )
    if len(normalized_paths) == 0:
        return None

    target_path = (
        posixpath.commonpath(normalized_paths)
        if saw_multi or len(normalized_paths) > 1
        else normalized_paths[0]
    )
    return ShellOp(
        kind="write",
        path=target_path,
        paths=normalized_paths,
        append=saw_append,
        multi=saw_multi,
        command=script,
    )


def _command_write_paths(
    *,
    command: ShellCommand,
    cwd: str | None,
) -> tuple[tuple[str, ...], bool]:
    executable = _command_executable(command=command)
    heredoc_present = any(
        redirection.operator in {"<<", "<<-"} for redirection in command.redirections
    )
    output_targets = tuple(
        _resolve_path(path=target, cwd=cwd)
        for target in _command_write_targets(command=command)
        if target.strip() != ""
    )

    if executable in _DIRECT_WRITE_COMMANDS and len(output_targets) > 0:
        append = any(
            redirection.operator == ">>" for redirection in command.redirections
        )
        return output_targets, append

    if heredoc_present and len(output_targets) > 0:
        append = any(
            redirection.operator == ">>" for redirection in command.redirections
        )
        return output_targets, append

    if executable == "tee":
        tokens = _tee_output_targets(command=command)
        append = "-a" in command.argv or "--append" in command.argv
        resolved = tuple(
            _resolve_path(path=token, cwd=cwd)
            for token in tokens
            if token.strip() != ""
        )
        return resolved, append

    return (), False


def _command_has_heredoc(*, command: ShellCommand) -> bool:
    return any(
        redirection.operator in {"<<", "<<-"} for redirection in command.redirections
    )


def _embedded_script_write_paths(
    *,
    command: ShellCommand,
    body: str,
    cwd: str | None,
) -> tuple[tuple[str, ...], bool, bool]:
    executable = _command_executable(command=command)
    if executable not in _PYTHON_LAUNCHERS or body.strip() == "":
        return (), False, False

    return _python_heredoc_write_paths(body=body, cwd=cwd)


def _python_heredoc_write_paths(
    *,
    body: str,
    cwd: str | None,
) -> tuple[tuple[str, ...], bool, bool]:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return (), False, False

    collector = _PythonWriteCollector(cwd=cwd)
    collector.visit(tree)

    normalized_paths = tuple(
        dict.fromkeys(path for path in collector.paths if path != "")
    )
    if not collector.saw_write:
        return (), False, False

    if collector.saw_dynamic_path:
        fallback_path = cwd or (
            posixpath.commonpath(normalized_paths) if len(normalized_paths) > 0 else ""
        )
        if fallback_path != "":
            normalized_paths = tuple(dict.fromkeys((*normalized_paths, fallback_path)))
        return normalized_paths, collector.saw_append, True

    return normalized_paths, collector.saw_append, len(normalized_paths) > 1


class _PythonWriteCollector(ast.NodeVisitor):
    def __init__(self, *, cwd: str | None) -> None:
        self.cwd = cwd
        self.paths: list[str] = []
        self.saw_write = False
        self.saw_dynamic_path = False
        self.saw_append = False

    def visit_Call(self, node: ast.Call) -> None:
        if _is_python_open_call(node):
            mode = _python_open_mode(node)
            if _is_python_write_mode(mode=mode):
                self.saw_write = True
                raw_path = (
                    _python_string_literal(node.args[0])
                    if len(node.args) >= 1
                    else None
                )
                if raw_path is None:
                    self.saw_dynamic_path = True
                else:
                    self.paths.append(_resolve_path(path=raw_path, cwd=self.cwd))
                if "a" in mode:
                    self.saw_append = True

        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if attribute in {"write_text", "write_bytes"}:
            self.saw_write = True
            raw_path = _python_path_receiver_literal(node.func.value)
            if raw_path is None:
                self.saw_dynamic_path = True
            else:
                self.paths.append(_resolve_path(path=raw_path, cwd=self.cwd))

        self.generic_visit(node)


def _is_python_open_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "open"


def _python_open_mode(node: ast.Call) -> str:
    if len(node.args) >= 2:
        mode = _python_string_literal(node.args[1])
        if mode is not None:
            return mode

    for keyword in node.keywords:
        if keyword.arg != "mode":
            continue
        mode = _python_string_literal(keyword.value)
        if mode is not None:
            return mode

    return "r"


def _is_python_write_mode(*, mode: str) -> bool:
    return any(flag in mode for flag in {"w", "a", "x"})


def _python_string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _python_path_receiver_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if not _is_python_path_factory(node.func):
            return None
        if len(node.args) == 0:
            return None
        return _python_string_literal(node.args[0])
    return None


def _is_python_path_factory(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"Path", "PurePath", "PosixPath"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"Path", "PurePath", "PosixPath"}
    return False


def _tee_output_targets(*, command: ShellCommand) -> tuple[str, ...]:
    argv = command.argv
    if len(argv) <= 1:
        return ()

    targets: list[str] = []
    after_separator = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if not after_separator and token == "--":
            after_separator = True
            index += 1
            continue
        if not after_separator and token in {
            "-a",
            "-i",
            "--append",
            "--ignore-interrupts",
        }:
            index += 1
            continue
        if not after_separator and token.startswith("-") and token != "-":
            index += 1
            continue
        targets.append(token)
        index += 1

    return tuple(targets)


def _classify_pipeline(*, pipeline: ShellPipeline, cwd: str | None) -> ShellOp | None:
    command = pipeline.first_command
    command_text = pipeline.text
    view = _command_view(command=command)

    op = _classify_search(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_build(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_dev(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_test(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_lint(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_install(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    op = _classify_edit(command=command, cwd=cwd, command_text=command_text)
    if op is not None:
        return op

    op = _classify_download(command=command, cwd=cwd, command_text=command_text)
    if op is not None:
        return op

    op = _classify_script(command=command, cwd=cwd, command_text=command_text)
    if op is not None:
        return op

    op = _classify_exploration(
        command=command,
        view=view,
        cwd=cwd,
        command_text=command_text,
    )
    if op is not None:
        return op

    return ShellOp(kind="run", path=cwd or "", command=command_text.strip())


def _classify_install(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    del command
    argv = view.argv
    executable = view.executable
    effective_executable = view.effective_executable
    effective_argv = view.effective_argv

    if executable == "npm" and _command_uses_script(
        argv=argv, names=_INSTALLING_SUBCOMMANDS
    ):
        return ShellOp(kind="install", path=cwd or "", tool="npm", command=command_text)
    if executable == "pnpm" and _command_uses_script(
        argv=argv, names=_INSTALLING_SUBCOMMANDS
    ):
        return ShellOp(
            kind="install", path=cwd or "", tool="pnpm", command=command_text
        )
    if executable == "yarn" and _command_uses_script(
        argv=argv, names={"add", "install"}
    ):
        return ShellOp(
            kind="install", path=cwd or "", tool="yarn", command=command_text
        )
    if executable == "bun" and _command_uses_script(
        argv=argv, names={"add", "install"}
    ):
        return ShellOp(kind="install", path=cwd or "", tool="bun", command=command_text)
    if executable in {"pip", "pip3"} and len(argv) >= 2 and argv[1] == "install":
        return ShellOp(
            kind="install", path=cwd or "", tool=executable, command=command_text
        )
    if (
        effective_executable in {"pip", "pip3"}
        and len(effective_argv) >= 2
        and effective_argv[1] == "install"
    ):
        return ShellOp(
            kind="install",
            path=cwd or "",
            tool=effective_executable,
            command=command_text,
        )
    if (
        executable == "uv"
        and len(argv) >= 3
        and argv[1] == "pip"
        and argv[2] == "install"
    ):
        return ShellOp(
            kind="install", path=cwd or "", tool="uv pip", command=command_text
        )
    if executable == "poetry" and len(argv) >= 2 and argv[1] in {"add", "install"}:
        return ShellOp(
            kind="install", path=cwd or "", tool="poetry", command=command_text
        )
    if executable == "pipenv" and len(argv) >= 2 and argv[1] == "install":
        return ShellOp(
            kind="install", path=cwd or "", tool="pipenv", command=command_text
        )
    if executable == "cargo" and len(argv) >= 2 and argv[1] in {"add", "install"}:
        return ShellOp(
            kind="install", path=cwd or "", tool="cargo", command=command_text
        )
    if executable == "go" and len(argv) >= 2 and argv[1] in {"get", "install"}:
        return ShellOp(kind="install", path=cwd or "", tool="go", command=command_text)
    if (
        executable == "dotnet"
        and len(argv) >= 3
        and argv[1] == "add"
        and argv[2] == "package"
    ):
        return ShellOp(
            kind="install", path=cwd or "", tool="dotnet", command=command_text
        )
    if (
        executable == "composer"
        and len(argv) >= 2
        and argv[1] in {"install", "require"}
    ):
        return ShellOp(
            kind="install", path=cwd or "", tool="composer", command=command_text
        )
    if executable == "bundle" and len(argv) >= 2 and argv[1] == "install":
        return ShellOp(
            kind="install", path=cwd or "", tool="bundle", command=command_text
        )
    if executable == "gem" and len(argv) >= 2 and argv[1] == "install":
        return ShellOp(kind="install", path=cwd or "", tool="gem", command=command_text)
    if executable in {"apt", "apt-get", "brew"} and "install" in argv[1:]:
        return ShellOp(
            kind="install", path=cwd or "", tool=executable, command=command_text
        )

    return None


def _classify_build(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    argv = view.argv
    executable = view.executable
    effective_executable = view.effective_executable
    effective_argv = view.effective_argv

    if executable == "npm" and _command_uses_script(
        argv=argv, names=_BUILD_SUBCOMMANDS
    ):
        return ShellOp(kind="build", path=cwd or "", tool="npm", command=command_text)
    if executable == "pnpm" and _command_uses_script(
        argv=argv, names=_BUILD_SUBCOMMANDS
    ):
        return ShellOp(kind="build", path=cwd or "", tool="pnpm", command=command_text)
    if executable == "yarn" and _command_uses_script(
        argv=argv, names=_BUILD_SUBCOMMANDS
    ):
        return ShellOp(kind="build", path=cwd or "", tool="yarn", command=command_text)
    if executable == "bun" and _command_uses_script(
        argv=argv, names=_BUILD_SUBCOMMANDS
    ):
        return ShellOp(kind="build", path=cwd or "", tool="bun", command=command_text)
    if executable == "cargo" and len(argv) >= 2 and argv[1] == "build":
        return ShellOp(kind="build", path=cwd or "", tool="cargo", command=command_text)
    if executable == "go" and len(argv) >= 2 and argv[1] == "build":
        return ShellOp(kind="build", path=cwd or "", tool="go", command=command_text)
    if executable == "dotnet" and len(argv) >= 2 and argv[1] == "build":
        return ShellOp(
            kind="build", path=cwd or "", tool="dotnet", command=command_text
        )
    if (
        executable == "python"
        and len(argv) >= 3
        and argv[1] == "-m"
        and argv[2] == "build"
    ):
        return ShellOp(
            kind="build", path=cwd or "", tool="python", command=command_text
        )
    if effective_executable in _BUILD_TOOLS and _is_build_effective_command(
        executable=effective_executable,
        argv=effective_argv,
    ):
        return ShellOp(
            kind="build",
            path=cwd or "",
            tool=effective_executable,
            command=command_text,
        )
    if executable == "make" and (
        len(argv) == 1 or (len(argv) >= 2 and argv[1] == "build")
    ):
        return ShellOp(kind="build", path=cwd or "", tool="make", command=command_text)
    if executable == "mvn" and any(
        token in {"package", "install", "compile"} for token in argv[1:]
    ):
        return ShellOp(kind="build", path=cwd or "", tool="mvn", command=command_text)
    if executable in {"gradle", "gradlew"} and any(
        token == "build" for token in argv[1:]
    ):
        return ShellOp(
            kind="build", path=cwd or "", tool="gradle", command=command_text
        )

    return None


def _classify_dev(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    del command
    argv = view.argv
    executable = view.executable
    effective_executable = view.effective_executable
    effective_argv = view.effective_argv

    if executable == "npm" and _command_uses_script(argv=argv, names=_DEV_SUBCOMMANDS):
        return ShellOp(kind="dev", path=cwd or "", tool="npm", command=command_text)
    if executable == "pnpm" and _command_uses_script(argv=argv, names=_DEV_SUBCOMMANDS):
        return ShellOp(kind="dev", path=cwd or "", tool="pnpm", command=command_text)
    if executable == "yarn" and _command_uses_script(argv=argv, names=_DEV_SUBCOMMANDS):
        return ShellOp(kind="dev", path=cwd or "", tool="yarn", command=command_text)
    if executable == "bun" and _command_uses_script(argv=argv, names=_DEV_SUBCOMMANDS):
        return ShellOp(kind="dev", path=cwd or "", tool="bun", command=command_text)
    if effective_executable in _DEV_TOOLS and _is_dev_effective_command(
        executable=effective_executable,
        argv=effective_argv,
    ):
        return ShellOp(
            kind="dev",
            path=cwd or "",
            tool=effective_executable,
            command=command_text,
        )
    return None


def _classify_test(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    del command
    argv = view.argv
    executable = view.executable
    effective_executable = view.effective_executable
    effective_argv = view.effective_argv

    if executable == "npm" and _command_uses_script(
        argv=argv, names=_TESTING_SUBCOMMANDS
    ):
        return ShellOp(kind="test", path=cwd or "", tool="npm", command=command_text)
    if executable == "pnpm" and _command_uses_script(
        argv=argv, names=_TESTING_SUBCOMMANDS
    ):
        return ShellOp(kind="test", path=cwd or "", tool="pnpm", command=command_text)
    if executable == "yarn" and _command_uses_script(
        argv=argv, names=_TESTING_SUBCOMMANDS
    ):
        return ShellOp(kind="test", path=cwd or "", tool="yarn", command=command_text)
    if executable == "bun" and _command_uses_script(
        argv=argv, names=_TESTING_SUBCOMMANDS
    ):
        return ShellOp(kind="test", path=cwd or "", tool="bun", command=command_text)
    if effective_executable in _TEST_TOOLS and _is_test_effective_command(
        executable=effective_executable,
        argv=effective_argv,
    ):
        return ShellOp(
            kind="test",
            path=cwd or "",
            tool=effective_executable,
            command=command_text,
        )
    return None


def _classify_lint(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    del command
    argv = view.argv
    executable = view.executable
    effective_executable = view.effective_executable
    effective_argv = view.effective_argv

    if executable == "npm" and _command_uses_script(
        argv=argv, names=_LINTING_SUBCOMMANDS
    ):
        return ShellOp(kind="lint", path=cwd or "", tool="npm", command=command_text)
    if executable == "pnpm" and _command_uses_script(
        argv=argv, names=_LINTING_SUBCOMMANDS
    ):
        return ShellOp(kind="lint", path=cwd or "", tool="pnpm", command=command_text)
    if executable == "yarn" and _command_uses_script(
        argv=argv, names=_LINTING_SUBCOMMANDS
    ):
        return ShellOp(kind="lint", path=cwd or "", tool="yarn", command=command_text)
    if executable == "bun" and _command_uses_script(
        argv=argv, names=_LINTING_SUBCOMMANDS
    ):
        return ShellOp(kind="lint", path=cwd or "", tool="bun", command=command_text)
    if effective_executable in _LINT_TOOLS and _is_lint_effective_command(
        executable=effective_executable,
        argv=effective_argv,
    ):
        return ShellOp(
            kind="lint",
            path=cwd or "",
            tool=effective_executable,
            command=command_text,
        )
    return None


def _classify_edit(
    *,
    command: ShellCommand,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    argv = command.argv
    executable = _command_executable(command=command)
    path = ""

    if executable == "sed" and "-i" in argv:
        tokens = _positional_tokens(argv=argv, value_flags={"-e", "-f"})
        if len(tokens) >= 1:
            path = _resolve_path(path=tokens[-1], cwd=cwd)
    elif executable == "perl" and "-pi" in argv:
        tokens = _positional_tokens(argv=argv, value_flags={"-e"})
        if len(tokens) >= 1:
            path = _resolve_path(path=tokens[-1], cwd=cwd)

    if path == "":
        return None

    return ShellOp(kind="edit", path=path, tool=executable, command=command_text)


def _classify_download(
    *,
    command: ShellCommand,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    argv = command.argv
    executable = _command_executable(command=command)
    if executable not in _DOWNLOAD_COMMANDS:
        return None

    destination = ""
    if executable == "curl":
        for index, token in enumerate(argv[1:], start=1):
            if token in {"-o", "--output"} and index + 1 < len(argv):
                destination = argv[index + 1]
                break
            if token.startswith("--output="):
                destination = token.partition("=")[2]
                break
    elif executable == "wget":
        for index, token in enumerate(argv[1:], start=1):
            if token in {"-O", "--output-document"} and index + 1 < len(argv):
                destination = argv[index + 1]
                break
            if token.startswith("--output-document="):
                destination = token.partition("=")[2]
                break

    if destination == "":
        url = _download_url_from_tokens(tokens=argv)
        if url is None:
            return None
        return ShellOp(
            kind="request",
            query=url,
            tool=executable,
            command=command_text,
        )

    path = _resolve_path(path=destination, cwd=cwd)
    return ShellOp(kind="download", path=path, tool=executable, command=command_text)


def _classify_script(
    *,
    command: ShellCommand,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    executable = _command_executable(command=command)
    if executable not in _SCRIPT_TOOL_NAMES:
        return None

    argv = command.argv
    uses_inline_script = _command_has_heredoc(command=command) or (
        len(argv) >= 2 and argv[1] in {"-c", "-e"}
    )
    if not uses_inline_script:
        return None

    return ShellOp(
        kind="script",
        path=cwd or "",
        tool=_SCRIPT_TOOL_NAMES[executable],
        command=command_text,
    )


def _download_url_from_tokens(*, tokens: tuple[str, ...]) -> str | None:
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        if token.startswith(("http://", "https://")):
            return token
    return None


def _classify_search(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    argv = command.argv
    executable = view.executable
    if executable == "git" and len(argv) >= 2 and argv[1] in _GIT_SEARCH_SUBCOMMANDS:
        return _classify_git_search(argv=argv, cwd=cwd, command_text=command_text)

    if executable not in _SEARCH_COMMANDS:
        return None

    query = ""
    paths: list[str] = []
    after_separator = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if not after_separator and token == "--":
            after_separator = True
            index += 1
            continue

        flag, attached_value = _search_option_parts(token=token)
        if not after_separator and flag in _SEARCH_QUERY_FLAGS:
            if attached_value is not None and attached_value != "":
                if query == "":
                    query = attached_value
                index += 1
                continue
            if index + 1 >= len(argv):
                return None
            if query == "":
                query = argv[index + 1]
            index += 2
            continue

        if not after_separator and token.startswith("-e") and token != "-e":
            if query == "":
                query = token[2:]
            index += 1
            continue

        if not after_separator and flag in _SEARCH_LONG_VALUE_FLAGS:
            index += 1 if attached_value is not None else 2
            continue

        if not after_separator and _is_search_short_value_flag(token=token):
            index += 2
            continue

        if not after_separator and token.startswith("-") and token != "-":
            index += 1
            continue

        if query == "":
            query = token
        else:
            paths.append(token)
        index += 1

    normalized_query = query.strip()
    normalized_paths = tuple(
        _resolve_path(path=path, cwd=cwd)
        for path in paths
        if isinstance(path, str) and path.strip() != ""
    )
    if normalized_query == "":
        return None

    path = normalized_paths[0] if len(normalized_paths) == 1 else ""
    return ShellOp(
        kind="search",
        path=path,
        paths=normalized_paths,
        query=normalized_query,
        command=command_text,
    )


def _classify_exploration(
    *,
    command: ShellCommand,
    view: _CommandView,
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    argv = command.argv
    executable = view.executable

    if executable == "git" and len(argv) >= 2 and argv[1] in _GIT_EXPLORE_SUBCOMMANDS:
        return ShellOp(
            kind="explore",
            mode="explore",
            path=cwd or "",
            command=command_text,
        )

    if executable in _LISTING_SEARCH_TOOLS:
        path = _fd_path_from_tokens(tokens=argv, cwd=cwd)
        if path != "":
            return ShellOp(
                kind="explore",
                mode="explore",
                path=path,
                command=command_text,
            )

    path = _read_path_from_tokens(tokens=argv)
    if path != "":
        return ShellOp(
            kind="explore",
            mode="read",
            path=_resolve_path(path=path, cwd=cwd),
            command=command_text,
        )

    path = _list_path_from_tokens(tokens=argv, cwd=cwd)
    if path != "":
        return ShellOp(
            kind="explore",
            mode="list",
            path=path,
            command=command_text,
        )

    if executable == "stat":
        path = _stat_path_from_tokens(tokens=argv, cwd=cwd)
        if path != "":
            return ShellOp(
                kind="explore",
                mode="read",
                path=path,
                command=command_text,
            )

    if executable == "find":
        path = _find_path_from_tokens(tokens=argv, cwd=cwd)
        if path != "":
            return ShellOp(
                kind="explore",
                mode="explore",
                path=path,
                command=command_text,
            )

    if cwd is not None and executable in {
        "pwd",
        "find",
        *_LIST_COMMANDS,
        "echo",
        "printf",
    }:
        return ShellOp(
            kind="explore",
            mode="explore",
            path=cwd,
            command=command_text,
        )

    return None


def _command_view(*, command: ShellCommand) -> _CommandView:
    argv = command.argv
    executable = _command_executable(command=command)
    effective_argv = argv
    effective_executable = executable

    if executable in _PYTHON_LAUNCHERS and len(argv) >= 3 and argv[1] == "-m":
        module = _normalize_tool_name(token=argv[2])
        if module != "":
            effective_executable = module
            effective_argv = (module, *argv[3:])
    elif executable in _NODE_LAUNCHERS and len(argv) >= 2:
        node_tool = _node_wrapped_tool_name(token=argv[1])
        if node_tool is not None:
            effective_executable = node_tool
            effective_argv = (node_tool, *argv[2:])
    else:
        wrapped_argv = _wrapped_effective_argv(argv=argv, executable=executable)
        if wrapped_argv is not None:
            effective_argv = wrapped_argv
            effective_executable = _normalize_tool_name(token=wrapped_argv[0])

    return _CommandView(
        executable=executable,
        argv=argv,
        effective_executable=effective_executable,
        effective_argv=effective_argv,
    )


def _command_executable(*, command: ShellCommand) -> str:
    argv = command.argv
    if len(argv) == 0:
        return ""
    return _normalize_tool_name(token=argv[0])


def _normalize_tool_name(*, token: str) -> str:
    name = Path(token).name.lower()
    for suffix in (".cmd", ".cjs", ".js", ".mjs", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _node_wrapped_tool_name(*, token: str) -> str | None:
    normalized = token.replace("\\", "/")
    if "node_modules/" not in normalized:
        return None
    return _normalize_tool_name(token=normalized)


def _wrapped_effective_argv(
    *,
    argv: tuple[str, ...],
    executable: str,
) -> tuple[str, ...] | None:
    if executable in {"npx", "bunx", "uvx"}:
        tokens = _positional_tokens(argv=argv, value_flags={"--package", "-p"})
        if len(tokens) >= 1:
            return tuple(tokens)
        return None

    if (
        executable in {"npm", "pnpm"}
        and len(argv) >= 2
        and argv[1] in {"exec", "x", "dlx"}
    ):
        tokens = _tokens_after_subcommand(
            argv=argv,
            subcommand_index=1,
            value_flags={"--call", "--package", "--workspace", "-c", "-p", "-w"},
        )
        if len(tokens) >= 1:
            return tuple(tokens)
        return None

    if executable in {"yarn", "bun"} and len(argv) >= 2 and argv[1] in {"dlx", "x"}:
        tokens = _tokens_after_subcommand(
            argv=argv,
            subcommand_index=1,
            value_flags={"--cwd"},
        )
        if len(tokens) >= 1:
            return tuple(tokens)
        return None

    if (
        executable in {"uv", "poetry", "pipenv", "hatch"}
        and len(argv) >= 2
        and argv[1] == "run"
    ):
        tokens = _tokens_after_subcommand(
            argv=argv,
            subcommand_index=1,
            value_flags={"--directory", "--project", "--with", "-C"},
        )
        if len(tokens) >= 1:
            return tuple(tokens)
        return None

    return None


def _command_uses_script(*, argv: tuple[str, ...], names: set[str]) -> bool:
    if len(argv) < 2:
        return False
    executable = _normalize_tool_name(token=argv[0])
    if (
        executable in {"npm", "pnpm", "yarn", "bun"}
        and len(argv) >= 3
        and argv[1] == "run"
    ):
        return argv[2] in names
    return argv[1] in names


def _is_build_effective_command(*, executable: str, argv: tuple[str, ...]) -> bool:
    if executable in {"rollup", "rspack", "tsc", "esbuild", "swc"}:
        return True
    if executable in {"vite", "astro", "next", "nuxt", "parcel"}:
        return len(argv) >= 2 and argv[1] in {"build", "export", "preview"}
    if executable in {"webpack", "webpack-cli"}:
        return len(argv) == 1 or (len(argv) >= 2 and argv[1] not in {"serve", "watch"})
    if executable in {"turbo", "nx"}:
        return len(argv) >= 2 and argv[1] == "build"
    if executable == "cmake":
        return len(argv) >= 2 and argv[1] == "--build"
    if executable in {"grunt", "gulp"}:
        return len(argv) == 1 or (len(argv) >= 2 and argv[1] == "build")
    return False


def _is_dev_effective_command(*, executable: str, argv: tuple[str, ...]) -> bool:
    if executable in {"vite", "parcel"}:
        return len(argv) == 1 or (len(argv) >= 2 and argv[1] in {"dev", "serve"})
    if executable in {"rollup", "tsc"}:
        return any(token in {"-w", "--watch"} for token in argv[1:])
    if executable in {"webpack", "webpack-cli", "rspack"}:
        return len(argv) >= 2 and argv[1] in {"serve", "watch"}
    if executable in {"astro", "next", "nuxt"}:
        return len(argv) >= 2 and argv[1] in {"dev", "start"}
    if executable in {"turbo", "nx"}:
        return len(argv) >= 2 and argv[1] in {"dev", "serve", "start", "watch"}
    if executable in {"grunt", "gulp"}:
        return len(argv) >= 2 and argv[1] in {"dev", "serve", "watch"}
    return False


def _is_test_effective_command(*, executable: str, argv: tuple[str, ...]) -> bool:
    if executable in {
        "ava",
        "c8",
        "cypress",
        "jest",
        "mocha",
        "playwright",
        "pytest",
        "tap",
        "uvu",
        "vitest",
    }:
        return True
    if executable in {"cargo", "go", "dotnet"}:
        return len(argv) >= 2 and argv[1] == "test"
    if executable == "mvn":
        return any(token == "test" for token in argv[1:])
    return False


def _is_lint_effective_command(*, executable: str, argv: tuple[str, ...]) -> bool:
    if executable in {
        "biome",
        "black",
        "eslint",
        "flake8",
        "golangci-lint",
        "isort",
        "markdownlint",
        "mypy",
        "prettier",
        "pylint",
        "shellcheck",
        "stylelint",
    }:
        return True
    if executable == "ruff":
        return len(argv) == 1 or (len(argv) >= 2 and argv[1] in {"check", "format"})
    if executable == "cargo":
        return len(argv) >= 2 and argv[1] in {"clippy", "fmt"}
    return False


def _positional_tokens(
    *,
    argv: tuple[str, ...],
    value_flags: set[str],
) -> list[str]:
    tokens: list[str] = []
    after_separator = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if not after_separator and token == "--":
            after_separator = True
            index += 1
            continue

        if not after_separator and token in value_flags:
            index += 2
            continue

        if not after_separator and token.startswith("--") and "=" in token:
            index += 1
            continue

        if not after_separator and token.startswith("-") and token != "-":
            index += 1
            continue

        tokens.append(token)
        index += 1

    return tokens


def _tokens_after_subcommand(
    *,
    argv: tuple[str, ...],
    subcommand_index: int,
    value_flags: set[str],
) -> list[str]:
    if subcommand_index + 1 >= len(argv):
        return []
    synthetic_argv = ("wrapper", *argv[subcommand_index + 1 :])
    return _positional_tokens(argv=synthetic_argv, value_flags=value_flags)


def _read_path_from_tokens(*, tokens: tuple[str, ...]) -> str:
    if len(tokens) == 0:
        return ""

    executable = Path(tokens[0]).name.lower()
    if executable == "sed":
        values = _positional_tokens(argv=tokens, value_flags={"-e", "-f"})
        if len(values) >= 2:
            return values[-1]
        return ""

    if executable in {"head", "tail"}:
        values = _positional_tokens(argv=tokens, value_flags=_READ_VALUE_FLAGS)
        if len(values) == 1:
            return values[0]
        return ""

    if executable in _READ_COMMANDS:
        values = _positional_tokens(argv=tokens, value_flags=set())
        if len(values) == 1:
            return values[0]
        return ""

    return ""


def _list_path_from_tokens(*, tokens: tuple[str, ...], cwd: str | None) -> str:
    if len(tokens) == 0:
        return ""

    executable = Path(tokens[0]).name.lower()
    if executable in _LIST_COMMANDS:
        values = _positional_tokens(argv=tokens, value_flags=_LIST_VALUE_FLAGS)
        if len(values) == 1:
            return _resolve_path(path=values[0], cwd=cwd)
        if executable == "ls" and len(values) == 0 and cwd is not None:
            return cwd
        return ""

    return ""


def _stat_path_from_tokens(*, tokens: tuple[str, ...], cwd: str | None) -> str:
    values = _positional_tokens(argv=tokens, value_flags={"-c", "--format"})
    resolved_paths = [
        _resolve_path(path=token, cwd=cwd)
        for token in values
        if isinstance(token, str) and token.strip() != ""
    ]
    if len(resolved_paths) == 0:
        return ""
    if len(resolved_paths) == 1:
        return resolved_paths[0]
    return posixpath.commonpath(resolved_paths)


def _find_path_from_tokens(*, tokens: tuple[str, ...], cwd: str | None) -> str:
    if len(tokens) <= 1:
        return cwd or ""
    token = tokens[1]
    if token.startswith("-"):
        return cwd or ""
    return _resolve_path(path=token, cwd=cwd)


def _fd_path_from_tokens(*, tokens: tuple[str, ...], cwd: str | None) -> str:
    values = _positional_tokens(argv=tokens, value_flags=set())
    if len(values) >= 2:
        return _resolve_path(path=values[1], cwd=cwd)
    if cwd is not None:
        return cwd
    return ""


def _classify_git_search(
    *,
    argv: tuple[str, ...],
    cwd: str | None,
    command_text: str,
) -> ShellOp | None:
    query = ""
    paths: list[str] = []
    after_separator = False
    index = 2
    while index < len(argv):
        token = argv[index]
        if not after_separator and token == "--":
            after_separator = True
            index += 1
            continue
        if not after_separator and token.startswith("-") and token != "-":
            index += 1
            continue
        if query == "":
            query = token
        else:
            paths.append(token)
        index += 1

    if query == "":
        return None

    normalized_paths = tuple(
        _resolve_path(path=path, cwd=cwd)
        for path in paths
        if isinstance(path, str) and path.strip() != ""
    )
    path = normalized_paths[0] if len(normalized_paths) == 1 else ""
    return ShellOp(
        kind="search",
        path=path,
        paths=normalized_paths,
        query=query,
        command=command_text,
    )


def _search_option_parts(*, token: str) -> tuple[str, str | None]:
    if token.startswith("--") and "=" in token:
        flag, _, value = token.partition("=")
        return flag, value
    return token, None


def _is_search_short_value_flag(*, token: str) -> bool:
    if len(token) <= 2 or not token.startswith("-") or token.startswith("--"):
        return False
    return token[:2] in _SEARCH_SHORT_VALUE_FLAGS


def _dedupe_operations(*, operations: list[ShellOp]) -> list[ShellOp]:
    deduped: list[ShellOp] = []
    seen = set[tuple[object, ...]]()
    for op in operations:
        key = (
            op.kind,
            op.path,
            op.paths,
            op.query,
            op.append,
            op.multi,
            op.mode,
            op.tool,
            op.command,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(op)
    return deduped


def _display_for_operations(
    *,
    command: str,
    cwd: str | None,
    operations: tuple[ShellOp, ...],
) -> ShellDisplay:
    combined_display = _combined_contextual_exec_display(operations=operations)
    if combined_display is not None:
        return combined_display

    primary = _primary_operation(operations=operations)
    if primary is None:
        return _run_display(command=command, cwd=cwd)

    if primary.kind == "write":
        return _write_display(primary)
    if primary.kind == "search":
        return _search_display(primary)
    if primary.kind == "explore":
        return _explore_display(primary)
    if primary.kind == "install":
        return _install_display(primary)
    if primary.kind == "build":
        return _build_display(primary)
    if primary.kind == "dev":
        return _dev_display(primary)
    if primary.kind == "edit":
        return _edit_display(primary)
    if primary.kind == "download":
        return _download_display(primary)
    if primary.kind == "request":
        return _request_display(primary)
    if primary.kind == "script":
        return _script_display(primary)
    if primary.kind == "lint":
        return _lint_display(primary)
    if primary.kind == "test":
        return _test_display(primary)
    return _run_display(command=primary.command or command, cwd=cwd)


def _primary_operation(*, operations: tuple[ShellOp, ...]) -> ShellOp | None:
    priority: tuple[ShellOpKind, ...] = (
        "write",
        "search",
        "build",
        "dev",
        "test",
        "lint",
        "install",
        "edit",
        "download",
        "request",
        "script",
        "explore",
        "run",
    )
    for kind in priority:
        matching = [op for op in operations if op.kind == kind]
        if matching:
            return matching[-1]
    return None


def _phase(*, headline: str, summary: str | None = None) -> ShellDisplayPhase:
    resolved_summary = summary if summary is not None else headline
    return ShellDisplayPhase(
        headline=_truncate_text(text=headline, limit=_SUMMARY_LIMIT),
        summary=_truncate_text(text=resolved_summary, limit=_SUMMARY_LIMIT),
    )


def _combined_contextual_exec_display(
    *,
    operations: tuple[ShellOp, ...],
) -> ShellDisplay | None:
    if any(op.kind in {"write", "edit", "download", "search"} for op in operations):
        return None

    contextual_ops = [
        op for op in operations if op.kind in _COMBINED_CONTEXTUAL_EXEC_KINDS
    ]
    if len(contextual_ops) < 2:
        return None

    deduped_ops: list[ShellOp] = []
    seen = set[tuple[ShellOpKind, str]]()
    for op in contextual_ops:
        key = (op.kind, op.path)
        if key in seen:
            continue
        seen.add(key)
        deduped_ops.append(op)

    if len(deduped_ops) < 2:
        return None

    common_target = _combined_contextual_target(operations=tuple(deduped_ops))
    if common_target == "":
        return None

    return ShellDisplay(
        event_kind="exec",
        path=common_target,
        pending=_phase(
            headline=_combined_contextual_headline(
                operations=tuple(deduped_ops),
                phase="pending",
                target=common_target,
            )
        ),
        active=_phase(
            headline=_combined_contextual_headline(
                operations=tuple(deduped_ops),
                phase="active",
                target=common_target,
            )
        ),
        completed=_phase(
            headline=_combined_contextual_headline(
                operations=tuple(deduped_ops),
                phase="completed",
                target=common_target,
            )
        ),
        failed=_phase(
            headline=_combined_contextual_headline(
                operations=tuple(deduped_ops),
                phase="failed",
                target=common_target,
            )
        ),
        cancelled=_phase(
            headline=_combined_contextual_headline(
                operations=tuple(deduped_ops),
                phase="cancelled",
                target=common_target,
            )
        ),
    )


def _combined_contextual_target(*, operations: tuple[ShellOp, ...]) -> str:
    paths = tuple(op.path for op in operations if op.path != "")
    if len(paths) != len(operations) or len(paths) == 0:
        return ""
    if any(path != paths[0] for path in paths[1:]):
        return ""
    return paths[0]


def _combined_contextual_headline(
    *,
    operations: tuple[ShellOp, ...],
    phase: ShellDisplayPhaseName,
    target: str,
) -> str:
    fragments = [
        _combined_contextual_fragment(kind=op.kind, phase=phase) for op in operations
    ]
    sentence = _join_human_list(fragments=tuple(fragments))
    if sentence == "":
        return ""
    return f"{sentence} in {target}"


def _combined_contextual_fragment(
    *,
    kind: ShellOpKind,
    phase: ShellDisplayPhaseName,
) -> str:
    fragment_map: dict[ShellOpKind, dict[ShellDisplayPhaseName, str]] = {
        "install": {
            "pending": "Preparing to install packages",
            "active": "Installing packages",
            "completed": "Installed packages",
            "failed": "Attempted to install packages",
            "cancelled": "Cancelled installing packages",
        },
        "build": {
            "pending": "Preparing to build project",
            "active": "Building project",
            "completed": "Built project",
            "failed": "Attempted to build project",
            "cancelled": "Cancelled building project",
        },
        "lint": {
            "pending": "Preparing to check code",
            "active": "Checking code",
            "completed": "Checked code",
            "failed": "Attempted to check code",
            "cancelled": "Cancelled checking code",
        },
        "test": {
            "pending": "Preparing to run tests",
            "active": "Running tests",
            "completed": "Ran tests",
            "failed": "Attempted to run tests",
            "cancelled": "Cancelled tests",
        },
        "dev": {
            "pending": "Preparing to start dev command",
            "active": "Starting dev command",
            "completed": "Ran dev command",
            "failed": "Attempted to start dev command",
            "cancelled": "Cancelled dev command",
        },
    }
    return fragment_map[kind][phase]


def _join_human_list(*, fragments: tuple[str, ...]) -> str:
    if len(fragments) == 0:
        return ""
    if len(fragments) == 1:
        return fragments[0]
    if len(fragments) == 2:
        return f"{fragments[0]} and {fragments[1].lower()}"
    head = ", ".join(fragment.lower() for fragment in fragments[1:-1])
    return f"{fragments[0]}, {head}, and {fragments[-1].lower()}"


def _write_display(op: ShellOp) -> ShellDisplay:
    is_multi_file = op.multi or len(op.paths) > 1
    target = f"files in {op.path}" if is_multi_file else op.path
    if op.append:
        pending = f"Preparing to append {target}"
        active = f"Appending {target}"
        completed = f"Appended {target}"
        failed = (
            f"Attempted to append {target}"
            if is_multi_file
            else f"Attempted to append file {target}"
        )
        cancelled = (
            f"Cancelled appending {target}"
            if is_multi_file
            else f"Cancelled appending file {target}"
        )
    else:
        pending = f"Preparing to write {target}"
        active = f"Writing {target}"
        completed = f"Wrote {target}"
        failed = (
            f"Attempted to write {target}"
            if is_multi_file
            else f"Attempted to write file {target}"
        )
        cancelled = (
            f"Cancelled writing {target}"
            if is_multi_file
            else f"Cancelled writing file {target}"
        )

    return ShellDisplay(
        event_kind="file",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=pending),
        active=_phase(headline=active),
        completed=_phase(headline=completed),
        failed=_phase(headline=failed),
        cancelled=_phase(headline=cancelled),
    )


def _search_display(op: ShellOp) -> ShellDisplay:
    uses_query_as_target = op.path == "" and len(op.paths) == 0
    if op.path != "":
        target = op.path
        active = f"Searching {target}"
        completed = f"Searched {target}"
        failed = f"Attempted to search file {target}"
        cancelled = f"Cancelled searching file {target}"
    elif len(op.paths) > 1:
        target = f"{len(op.paths)} paths"
        active = f"Searching {target}"
        completed = f"Searched {target}"
        failed = f"Attempted to search {target}"
        cancelled = f"Cancelled searching {target}"
    else:
        target = op.query
        active = f"Searching for {target}"
        completed = f"Searched for {target}"
        failed = f"Attempted to search for {target}"
        cancelled = f"Cancelled searching for {target}"

    details: list[str] = []
    if op.path != "" or len(op.paths) > 1:
        details.append(_truncate_text(text=f"Pattern: {op.query}", limit=_DETAIL_LIMIT))
    if len(op.paths) > 1:
        details.append(
            _truncate_text(text="Paths: " + ", ".join(op.paths), limit=_DETAIL_LIMIT)
        )

    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        details=tuple(details),
        preview=_command_preview(command=op.command),
        coalesce_path=op.path,
        pending=_phase(
            headline=(
                f"Preparing to search for {target}"
                if uses_query_as_target
                else f"Preparing to search {target}"
            )
        ),
        active=_phase(headline=active),
        completed=_phase(headline=completed),
        failed=_phase(headline=failed),
        cancelled=_phase(headline=cancelled),
    )


def _explore_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    if op.mode == "read":
        active = f"Reading {target}"
        completed = f"Read {target}"
        failed = f"Attempted to read file {target}"
        cancelled = f"Cancelled reading file {target}"
    elif op.mode == "list":
        active = f"Listing {target}"
        completed = f"Listed {target}"
        failed = f"Attempted to list {target}"
        cancelled = f"Cancelled listing {target}"
    else:
        active = f"Exploring {target}"
        completed = f"Explored {target}"
        failed = f"Attempted to explore {target}"
        cancelled = f"Cancelled exploring {target}"

    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        coalesce_path=op.path,
        pending=_phase(
            headline=(
                f"Preparing to read {target}"
                if op.mode == "read"
                else (
                    f"Preparing to list {target}"
                    if op.mode == "list"
                    else f"Preparing to explore {target}"
                )
            )
        ),
        active=_phase(headline=active),
        completed=_phase(headline=completed),
        failed=_phase(headline=failed),
        cancelled=_phase(headline=cancelled),
    )


def _install_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to install packages in {target}"),
        active=_phase(headline=f"Installing packages in {target}"),
        completed=_phase(headline=f"Installed packages in {target}"),
        failed=_phase(headline=f"Attempted to install packages in {target}"),
        cancelled=_phase(headline=f"Cancelled installing packages in {target}"),
    )


def _build_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to build {target}"),
        active=_phase(headline=f"Building {target}"),
        completed=_phase(headline=f"Built {target}"),
        failed=_phase(headline=f"Attempted to build {target}"),
        cancelled=_phase(headline=f"Cancelled building {target}"),
    )


def _dev_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to start dev command in {target}"),
        active=_phase(headline=f"Starting dev command in {target}"),
        completed=_phase(headline=f"Ran dev command in {target}"),
        failed=_phase(headline=f"Attempted to start dev command in {target}"),
        cancelled=_phase(headline=f"Cancelled dev command in {target}"),
    )


def _edit_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "file"
    return ShellDisplay(
        event_kind="file",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to edit {target}"),
        active=_phase(headline=f"Editing {target}"),
        completed=_phase(headline=f"Edited {target}"),
        failed=_phase(headline=f"Attempted to edit file {target}"),
        cancelled=_phase(headline=f"Cancelled editing file {target}"),
    )


def _download_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "file"
    return ShellDisplay(
        event_kind="file",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to download {target}"),
        active=_phase(headline=f"Downloading {target}"),
        completed=_phase(headline=f"Downloaded {target}"),
        failed=_phase(headline=f"Attempted to download {target}"),
        cancelled=_phase(headline=f"Cancelled downloading {target}"),
    )


def _request_display(op: ShellOp) -> ShellDisplay:
    return ShellDisplay(
        event_kind="exec",
        details=(op.query,) if op.query != "" else (),
        preview=_command_preview(command=op.command),
        pending=_phase(headline="Preparing web request"),
        active=_phase(headline="Making web request"),
        completed=_phase(headline="Made web request"),
        failed=_phase(headline="Web request failed"),
        cancelled=_phase(headline="Cancelled web request"),
    )


def _script_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    tool_name = op.tool if op.tool != "" else "script"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to run {tool_name} script in {target}"),
        active=_phase(headline=f"Running {tool_name} script in {target}"),
        completed=_phase(headline=f"Ran {tool_name} script in {target}"),
        failed=_phase(headline=f"Attempted to run {tool_name} script in {target}"),
        cancelled=_phase(headline=f"Cancelled {tool_name} script in {target}"),
    )


def _lint_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to check code in {target}"),
        active=_phase(headline=f"Checking code in {target}"),
        completed=_phase(headline=f"Checked code in {target}"),
        failed=_phase(headline=f"Attempted to check code in {target}"),
        cancelled=_phase(headline=f"Cancelled code checks in {target}"),
    )


def _test_display(op: ShellOp) -> ShellDisplay:
    target = op.path if op.path != "" else "project"
    return ShellDisplay(
        event_kind="exec",
        path=op.path,
        preview=_command_preview(command=op.command),
        pending=_phase(headline=f"Preparing to run tests in {target}"),
        active=_phase(headline=f"Running tests in {target}"),
        completed=_phase(headline=f"Ran tests in {target}"),
        failed=_phase(headline=f"Attempted to run tests in {target}"),
        cancelled=_phase(headline=f"Cancelled tests in {target}"),
    )


def _run_display(*, command: str, cwd: str | None) -> ShellDisplay:
    normalized_command = command.strip()
    preview = _command_preview(command=normalized_command)
    if normalized_command == "":
        return ShellDisplay(
            event_kind="exec",
            path=cwd or "",
            pending=_phase(headline="Preparing Command"),
            active=_phase(headline="Running Command"),
            completed=_phase(headline="Ran Command"),
            failed=_phase(headline="Command Failed"),
            cancelled=_phase(headline="Command Cancelled"),
        )

    return ShellDisplay(
        event_kind="exec",
        path=cwd or "",
        preview=preview,
        pending=_phase(
            headline="Preparing Command",
            summary=f"Prepare {normalized_command}",
        ),
        active=_phase(headline="Running Command", summary=f"Run {normalized_command}"),
        completed=_phase(headline="Ran Command", summary=normalized_command),
        failed=_phase(headline="Command Failed", summary="Command Failed"),
        cancelled=_phase(headline="Command Cancelled", summary="Command Cancelled"),
    )
