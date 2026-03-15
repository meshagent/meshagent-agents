from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lark import Lark, Token, Transformer, UnexpectedInput

ShellSeparator = Literal["&&", "||", ";", "\n"]

_GRAMMAR_TEXT = r"""
start: sequence

sequence: pipeline (separator pipeline)* separator?

separator: AND_IF | OR_IF | SEMICOLON | NEWLINE_SEPARATOR

pipeline: command (_PIPE command)*

command: command_part+

?command_part: redirection | WORD

redirection: IO_NUMBER? REDIR WORD

AND_IF: "&&"
OR_IF: "||"
SEMICOLON: ";"
_PIPE: "|"
IO_NUMBER.10: /\d+(?=(?:<<-|<<|>&|<&|>>|>|<))/
REDIR: "&>>" | "&>" | "<<-" | "<<" | ">&" | "<&" | ">>" | ">" | "<"
WORD: /(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^ \t\r\n;&|<>])+/
NEWLINE_SEPARATOR: /(\r?\n)+/

%ignore /[ \t]+/
"""
_HEREDOC_MARKER_RE = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?P<marker>[A-Za-z0-9_:\-]+)(?P=quote)"
)


class ShellParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShellRedirection:
    operator: str
    target: str


ShellCommandPart = str | ShellRedirection


@dataclass(frozen=True, slots=True)
class ShellCommand:
    parts: tuple[ShellCommandPart, ...]

    @property
    def argv(self) -> tuple[str, ...]:
        return tuple(part for part in self.parts if isinstance(part, str))

    @property
    def redirections(self) -> tuple[ShellRedirection, ...]:
        return tuple(part for part in self.parts if isinstance(part, ShellRedirection))

    @property
    def executable(self) -> str:
        argv = self.argv
        return argv[0] if len(argv) > 0 else ""

    @property
    def text(self) -> str:
        return " ".join(_command_part_text(part=part) for part in self.parts)


@dataclass(frozen=True, slots=True)
class ShellPipeline:
    commands: tuple[ShellCommand, ...]

    @property
    def first_command(self) -> ShellCommand:
        return self.commands[0]

    @property
    def text(self) -> str:
        return " | ".join(command.text for command in self.commands)


@dataclass(frozen=True, slots=True)
class ShellSequenceItem:
    pipeline: ShellPipeline
    separator_before: ShellSeparator | None


@dataclass(frozen=True, slots=True)
class ShellProgram:
    items: tuple[ShellSequenceItem, ...]


class _ShellTransformer(Transformer):
    def WORD(self, token: Token) -> str:
        return str(token)

    def redirection(self, children: list[Token | str]) -> ShellRedirection:
        if len(children) == 2:
            operator, target = children
            return ShellRedirection(operator=str(operator), target=str(target))

        io_number, operator, target = children
        return ShellRedirection(
            operator=f"{io_number}{operator}",
            target=str(target),
        )

    def command(self, children: list[ShellCommandPart]) -> ShellCommand:
        return ShellCommand(parts=tuple(children))

    def pipeline(self, children: list[ShellCommand]) -> ShellPipeline:
        return ShellPipeline(commands=tuple(children))

    def separator(self, children: list[Token]) -> ShellSeparator:
        value = str(children[0])
        if "\n" in value:
            return "\n"
        if value in {"&&", "||", ";"}:
            return value
        raise AssertionError(f"unexpected separator {value!r}")

    def sequence(
        self,
        children: list[ShellPipeline | ShellSeparator],
    ) -> ShellProgram:
        items: list[ShellSequenceItem] = []
        separator_before: ShellSeparator | None = None
        for child in children:
            if isinstance(child, ShellPipeline):
                items.append(
                    ShellSequenceItem(
                        pipeline=child,
                        separator_before=separator_before,
                    )
                )
                separator_before = None
            else:
                separator_before = child
        return ShellProgram(items=tuple(items))

    def start(self, children: list[ShellProgram]) -> ShellProgram:
        return children[0]


_PARSER = Lark(_GRAMMAR_TEXT, parser="lalr", maybe_placeholders=False)
_TRANSFORMER = _ShellTransformer()


def parse_shell_script(*, script: str) -> ShellProgram:
    normalized = script.replace("\r\n", "\n").strip()
    if normalized == "":
        return ShellProgram(items=())

    stripped = _strip_heredoc_bodies(script=normalized)
    if stripped.strip() == "":
        return ShellProgram(items=())

    try:
        tree = _PARSER.parse(stripped)
    except UnexpectedInput as exc:
        raise ShellParseError(f"unable to parse shell script: {exc}") from exc
    return _TRANSFORMER.transform(tree)


def extract_heredoc_bodies(*, script: str) -> tuple[str, ...]:
    normalized = script.replace("\r\n", "\n").strip()
    if normalized == "":
        return ()

    lines = normalized.split("\n")
    bodies: list[str] = []
    index = 0

    while index < len(lines):
        markers = _heredoc_markers_in_line(line=lines[index])
        if len(markers) == 0:
            index += 1
            continue

        index += 1
        for marker in markers:
            body_lines: list[str] = []
            while index < len(lines) and lines[index] != marker:
                body_lines.append(lines[index])
                index += 1
            if index < len(lines) and lines[index] == marker:
                index += 1
            bodies.append("\n".join(body_lines))

    return tuple(bodies)


def _command_part_text(*, part: ShellCommandPart) -> str:
    if isinstance(part, ShellRedirection):
        return f"{part.operator} {part.target}"
    return part


def _strip_heredoc_bodies(*, script: str) -> str:
    lines = script.split("\n")
    output: list[str] = []
    pending_markers: list[str] = []

    for line in lines:
        if len(pending_markers) > 0:
            if line == pending_markers[0]:
                pending_markers.pop(0)
            continue

        output.append(line)
        pending_markers = _heredoc_markers_in_line(line=line)

    return "\n".join(output)


def _heredoc_markers_in_line(*, line: str) -> list[str]:
    return [match.group("marker") for match in _HEREDOC_MARKER_RE.finditer(line)]
