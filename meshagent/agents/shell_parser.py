from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ShellSeparator = Literal["&&", "||", ";", "\n", "&"]


class ShellParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShellRedirection:
    operator: str
    target: str
    heredoc_body: str | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class ShellCompound:
    kind: str
    programs: tuple[ShellProgram, ...]
    text: str


@dataclass(frozen=True, slots=True)
class ShellCommand:
    argv: tuple[str, ...]
    redirections: tuple[ShellRedirection, ...] = ()
    text: str = ""
    compound: ShellCompound | None = None

    @property
    def executable(self) -> str:
        return self.argv[0] if len(self.argv) > 0 else ""


@dataclass(frozen=True, slots=True)
class ShellPipeline:
    commands: tuple[ShellCommand, ...]
    text: str = ""

    @property
    def first_command(self) -> ShellCommand:
        return self.commands[0]


@dataclass(frozen=True, slots=True)
class ShellSequenceItem:
    pipeline: ShellPipeline
    separator_before: ShellSeparator | None


@dataclass(frozen=True, slots=True)
class ShellProgram:
    items: tuple[ShellSequenceItem, ...]


@dataclass(slots=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int
    quoted: bool = False
    heredoc_body: str | None = None


@dataclass(slots=True)
class _PendingHeredoc:
    delimiter: str
    strip_tabs: bool
    token: _Token


_REDIRECTION_OPERATORS = (
    "&>>",
    "&>",
    "<<-",
    "<<",
    ">&",
    "<&",
    ">>",
    ">|",
    "<>",
    ">",
    "<",
)
_CONTROL_OPERATORS = {
    "&&": "AND_IF",
    "||": "OR_IF",
    ";;": "DSEMI",
    ";": "SEMI",
    "&": "AMP",
    "|": "PIPE",
    "(": "LPAREN",
    ")": "RPAREN",
    "{": "LBRACE",
    "}": "RBRACE",
}
_SHELL_OPERATOR_CHARS = "&|;(){}<>"
_RESERVED_WORDS = {
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "for",
    "while",
    "until",
    "do",
    "done",
    "in",
    "case",
    "esac",
    "select",
    "function",
    "time",
}
_MAX_COMPOUND_PARSE_DEPTH = 128


class _ShellLexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.index = 0
        self.tokens: list[_Token] = []
        self._pending_heredoc_redirection: str | None = None
        self._pending_heredocs: list[_PendingHeredoc] = []

    def tokenize(self) -> tuple[_Token, ...]:
        while self.index < self.length:
            start_index = self.index
            char = self.source[self.index]
            if char in {" ", "\t", "\r"}:
                self.index += 1
            elif self._starts_comment():
                self._consume_comment()
            elif char == "\n":
                self._consume_newline()
            else:
                token = self._lex_operator_or_redirection()
                if token is not None:
                    self.tokens.append(token)
                    if token.kind == "REDIR" and token.text.endswith(("<<", "<<-")):
                        self._pending_heredoc_redirection = token.text
                    else:
                        self._pending_heredoc_redirection = None
                else:
                    token = self._lex_word()
                    self.tokens.append(token)
                    if self._pending_heredoc_redirection is not None:
                        self._pending_heredocs.append(
                            _PendingHeredoc(
                                delimiter=_unquote_heredoc_delimiter(token.text),
                                strip_tabs=self._pending_heredoc_redirection.endswith(
                                    "<<-"
                                ),
                                token=token,
                            )
                        )
                        self._pending_heredoc_redirection = None

            if self.index <= start_index:
                raise ShellParseError(
                    f"shell lexer made no progress at index {self.index}"
                )

        self.tokens.append(
            _Token(kind="EOF", text="", start=self.length, end=self.length)
        )
        return tuple(self.tokens)

    def _consume_comment(self) -> None:
        while self.index < self.length and self.source[self.index] != "\n":
            self.index += 1

    def _consume_newline(self) -> None:
        start = self.index
        self.index += 1
        self.tokens.append(
            _Token(kind="NEWLINE", text="\n", start=start, end=self.index)
        )
        if self._pending_heredocs:
            self._consume_pending_heredocs()

    def _consume_pending_heredocs(self) -> None:
        while self._pending_heredocs:
            pending = self._pending_heredocs.pop(0)
            pending.token.heredoc_body = self._read_heredoc_body(
                delimiter=pending.delimiter,
                strip_tabs=pending.strip_tabs,
            )

    def _read_heredoc_body(self, *, delimiter: str, strip_tabs: bool) -> str:
        lines: list[str] = []
        while self.index < self.length:
            line_start = self.index
            while self.index < self.length and self.source[self.index] != "\n":
                self.index += 1
            line_end = self.index
            line = self.source[line_start:line_end]
            comparable = line.lstrip("\t") if strip_tabs else line
            if comparable == delimiter:
                if self.index < self.length and self.source[self.index] == "\n":
                    self.index += 1
                return "\n".join(lines)
            lines.append(line.lstrip("\t") if strip_tabs else line)
            if self.index < self.length and self.source[self.index] == "\n":
                self.index += 1
        return "\n".join(lines)

    def _lex_operator_or_redirection(self) -> _Token | None:
        start = self.index
        io_number_end = self.index
        while io_number_end < self.length and self.source[io_number_end].isdigit():
            io_number_end += 1

        if io_number_end > self.index:
            operator = self._match_redirection(self.source[io_number_end:])
            if operator is not None:
                self.index = io_number_end + len(operator)
                return _Token(
                    kind="REDIR",
                    text=self.source[start : self.index],
                    start=start,
                    end=self.index,
                )

        operator = self._match_redirection(self.source[self.index :])
        if operator is not None:
            self.index += len(operator)
            return _Token(kind="REDIR", text=operator, start=start, end=self.index)

        for operator_text, kind in sorted(
            _CONTROL_OPERATORS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if self.source.startswith(operator_text, self.index):
                self.index += len(operator_text)
                return _Token(
                    kind=kind, text=operator_text, start=start, end=self.index
                )
        return None

    def _lex_word(self) -> _Token:
        start = self.index
        quoted = False
        while self.index < self.length:
            char = self.source[self.index]
            if char in {" ", "\t", "\r", "\n"}:
                break
            if char == "\\":
                quoted = True
                self.index += 1
                if self.index < self.length:
                    self.index += 1
                continue
            if char == "'":
                quoted = True
                self._consume_single_quoted()
                continue
            if char == '"':
                quoted = True
                self._consume_double_quoted()
                continue
            if self._starts_operator():
                break
            self.index += 1

        if self.index == start:
            raise ShellParseError(f"unable to lex shell token at index {self.index}")

        return _Token(
            kind="WORD",
            text=self.source[start : self.index],
            start=start,
            end=self.index,
            quoted=quoted,
        )

    def _consume_single_quoted(self) -> None:
        self.index += 1
        while self.index < self.length:
            if self.source[self.index] == "'":
                self.index += 1
                return
            self.index += 1

    def _consume_double_quoted(self) -> None:
        self.index += 1
        while self.index < self.length:
            char = self.source[self.index]
            if char == "\\":
                self.index += 1
                if self.index < self.length:
                    self.index += 1
                continue
            if char == '"':
                self.index += 1
                return
            self.index += 1

    def _starts_comment(self) -> bool:
        if self.source[self.index] != "#":
            return False
        if self.index == 0:
            return True
        return self.source[self.index - 1] in {
            " ",
            "\t",
            "\r",
            "\n",
            ";",
            "&",
            "|",
            "(",
            ")",
            "{",
            "}",
        }

    def _starts_operator(self) -> bool:
        char = self.source[self.index]
        return char in _SHELL_OPERATOR_CHARS

    @staticmethod
    def _match_redirection(text: str) -> str | None:
        for operator in _REDIRECTION_OPERATORS:
            if text.startswith(operator):
                return operator
        return None


class _ShellParser:
    def __init__(self, *, source: str, tokens: tuple[_Token, ...]) -> None:
        self.source = source
        self.tokens = tokens
        self.index = 0

    def parse(self) -> ShellProgram:
        return self._parse_program(
            stop_words=frozenset(),
            stop_kinds=frozenset(),
            depth=0,
        )

    def _parse_program(
        self,
        *,
        stop_words: frozenset[str],
        stop_kinds: frozenset[str],
        depth: int,
    ) -> ShellProgram:
        if depth > _MAX_COMPOUND_PARSE_DEPTH:
            raise ShellParseError("shell parse nesting too deep")
        items: list[ShellSequenceItem] = []
        separator_before: ShellSeparator | None = None
        self._skip_newlines()

        while not self._at_program_end(stop_words=stop_words, stop_kinds=stop_kinds):
            start_index = self.index
            pipeline = self._parse_pipeline_command(
                stop_words=stop_words,
                stop_kinds=stop_kinds,
                depth=depth,
            )
            items.append(
                ShellSequenceItem(
                    pipeline=pipeline,
                    separator_before=separator_before,
                )
            )
            separator_before = None

            separator = self._consume_separator()
            if separator is None:
                if self.index <= start_index:
                    raise ShellParseError(
                        f"shell parser made no progress at token {self._current().kind}"
                    )
                break
            separator_before = separator
            self._skip_newlines()
            if self.index <= start_index:
                raise ShellParseError(
                    f"shell parser made no progress at token {self._current().kind}"
                )

        return ShellProgram(items=tuple(items))

    def _parse_pipeline_command(
        self,
        *,
        stop_words: frozenset[str],
        stop_kinds: frozenset[str],
        depth: int,
    ) -> ShellPipeline:
        start = self._current().start
        self._consume_prefix_words({"!", "time", "-p"})
        commands = [
            self._parse_command(
                stop_words=stop_words,
                stop_kinds=stop_kinds,
                depth=depth,
            )
        ]
        while self._match(kind="PIPE"):
            self._skip_newlines()
            commands.append(
                self._parse_command(
                    stop_words=stop_words,
                    stop_kinds=stop_kinds,
                    depth=depth,
                )
            )
        end = (
            commands[-1].text_end
            if isinstance(commands[-1], _ParsedShellCommand)
            else self._previous().end
        )
        text = self.source[start:end].strip()
        return ShellPipeline(
            commands=tuple(_strip_internal_command(command) for command in commands),
            text=text,
        )

    def _parse_command(
        self,
        *,
        stop_words: frozenset[str],
        stop_kinds: frozenset[str],
        depth: int,
    ) -> _ParsedShellCommand:
        token = self._current()
        if token.kind == "LPAREN":
            command = self._parse_subshell(depth=depth + 1)
        elif token.kind == "LBRACE":
            command = self._parse_group_command(depth=depth + 1)
        elif self._is_function_definition():
            command = self._parse_function_definition(depth=depth + 1)
        elif token.kind == "WORD" and not token.quoted:
            word = token.text
            if word == "if":
                command = self._parse_if_command(depth=depth + 1)
            elif word == "for":
                command = self._parse_for_command(depth=depth + 1)
            elif word == "while":
                command = self._parse_loop_command(kind="while", depth=depth + 1)
            elif word == "until":
                command = self._parse_loop_command(kind="until", depth=depth + 1)
            elif word == "select":
                command = self._parse_select_command(depth=depth + 1)
            elif word == "case":
                command = self._parse_case_command(depth=depth + 1)
            else:
                command = self._parse_simple_command()
        else:
            command = self._parse_simple_command()

        trailing_redirections: list[ShellRedirection] = []
        while self._current().kind == "REDIR":
            trailing_redirections.append(self._parse_redirection())

        if trailing_redirections:
            return _ParsedShellCommand(
                argv=command.argv,
                redirections=tuple((*command.redirections, *trailing_redirections)),
                text=self.source[command.text_start : self._previous().end].strip(),
                compound=command.compound,
                text_start=command.text_start,
                text_end=self._previous().end,
            )
        return command

    def _parse_simple_command(self) -> _ParsedShellCommand:
        start = self._current().start
        argv: list[str] = []
        redirections: list[ShellRedirection] = []
        while True:
            token = self._current()
            if token.kind == "REDIR":
                redirections.append(self._parse_redirection())
                continue
            if token.kind == "WORD":
                argv.append(self._advance().text)
                continue
            break

        if not argv and not redirections:
            raise ShellParseError(
                f"expected shell command near {self._current().text!r}"
            )

        end = self._previous().end
        return _ParsedShellCommand(
            argv=tuple(argv),
            redirections=tuple(redirections),
            text=self.source[start:end].strip(),
            compound=None,
            text_start=start,
            text_end=end,
        )

    def _parse_redirection(self) -> ShellRedirection:
        operator = self._expect(kind="REDIR")
        target = self._expect(kind="WORD")
        return ShellRedirection(
            operator=operator.text,
            target=target.text,
            heredoc_body=target.heredoc_body,
        )

    def _parse_if_command(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect_word("if").start
        self._skip_newlines()
        programs: list[ShellProgram] = [
            self._parse_program(
                stop_words=frozenset({"then"}),
                stop_kinds=frozenset(),
                depth=depth,
            )
        ]
        self._expect_word("then")
        self._skip_newlines()
        programs.append(
            self._parse_program(
                stop_words=frozenset({"elif", "else", "fi"}),
                stop_kinds=frozenset(),
                depth=depth,
            )
        )

        while self._is_word("elif"):
            self._advance()
            self._skip_newlines()
            programs.append(
                self._parse_program(
                    stop_words=frozenset({"then"}),
                    stop_kinds=frozenset(),
                    depth=depth,
                )
            )
            self._expect_word("then")
            self._skip_newlines()
            programs.append(
                self._parse_program(
                    stop_words=frozenset({"elif", "else", "fi"}),
                    stop_kinds=frozenset(),
                    depth=depth,
                )
            )

        if self._is_word("else"):
            self._advance()
            self._skip_newlines()
            programs.append(
                self._parse_program(
                    stop_words=frozenset({"fi"}),
                    stop_kinds=frozenset(),
                    depth=depth,
                )
            )

        end = self._expect_word("fi").end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("if",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="if", programs=tuple(programs), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_for_command(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect_word("for").start
        if self._current().kind == "WORD":
            self._advance()
        self._skip_newlines()
        if self._is_word("in"):
            self._advance()
            while not self._at_list_terminator() and not self._is_word("do"):
                if self._current().kind == "WORD":
                    self._advance()
                    continue
                if self._current().kind == "NEWLINE":
                    break
                break
        if self._current().kind in {"SEMI", "NEWLINE"}:
            self._consume_list_terminator()
        self._skip_newlines()
        body = self._parse_do_group(stop_word="done", depth=depth)
        end = self._previous().end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("for",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="for", programs=(body,), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_select_command(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect_word("select").start
        if self._current().kind == "WORD":
            self._advance()
        self._skip_newlines()
        if self._is_word("in"):
            self._advance()
            while not self._at_list_terminator() and not self._is_word("do"):
                if self._current().kind == "WORD":
                    self._advance()
                    continue
                break
        if self._current().kind in {"SEMI", "NEWLINE"}:
            self._consume_list_terminator()
        self._skip_newlines()
        body = self._parse_do_group(stop_word="done", depth=depth)
        end = self._previous().end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("select",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="select", programs=(body,), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_loop_command(
        self, *, kind: Literal["while", "until"], depth: int
    ) -> _ParsedShellCommand:
        start = self._expect_word(kind).start
        self._skip_newlines()
        condition = self._parse_program(
            stop_words=frozenset({"do"}),
            stop_kinds=frozenset(),
            depth=depth,
        )
        self._expect_word("do")
        self._skip_newlines()
        body = self._parse_program(
            stop_words=frozenset({"done"}),
            stop_kinds=frozenset(),
            depth=depth,
        )
        end = self._expect_word("done").end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=(kind,),
            redirections=(),
            text=text,
            compound=ShellCompound(kind=kind, programs=(condition, body), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_case_command(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect_word("case").start
        if self._current().kind == "WORD":
            self._advance()
        self._skip_newlines()
        self._expect_word("in")
        self._skip_newlines()

        programs: list[ShellProgram] = []
        while not self._is_word("esac") and self._current().kind != "EOF":
            self._skip_newlines()
            if self._current().kind == "LPAREN":
                self._advance()
            while self._current().kind != "RPAREN" and not self._at_program_end(
                stop_words=frozenset({"esac"}),
                stop_kinds=frozenset(),
            ):
                self._advance()
            if self._current().kind == "RPAREN":
                self._advance()
            self._skip_newlines()
            programs.append(
                self._parse_program(
                    stop_words=frozenset({"esac"}),
                    stop_kinds=frozenset({"DSEMI"}),
                    depth=depth,
                )
            )
            if self._current().kind == "DSEMI":
                self._advance()
            self._skip_newlines()

        end = self._expect_word("esac").end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("case",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="case", programs=tuple(programs), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_subshell(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect(kind="LPAREN").start
        program = self._parse_program(
            stop_words=frozenset(),
            stop_kinds=frozenset({"RPAREN"}),
            depth=depth,
        )
        end = self._expect(kind="RPAREN").end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("(",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="subshell", programs=(program,), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_group_command(self, *, depth: int) -> _ParsedShellCommand:
        start = self._expect(kind="LBRACE").start
        self._skip_newlines()
        program = self._parse_program(
            stop_words=frozenset(),
            stop_kinds=frozenset({"RBRACE"}),
            depth=depth,
        )
        end = self._expect(kind="RBRACE").end
        text = self.source[start:end].strip()
        return _ParsedShellCommand(
            argv=("{",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="group", programs=(program,), text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_function_definition(self, *, depth: int) -> _ParsedShellCommand:
        start = self._current().start
        if self._is_word("function"):
            self._advance()
            if self._current().kind == "WORD":
                self._advance()
            if self._current().kind == "LPAREN":
                self._advance()
                self._expect(kind="RPAREN")
        else:
            self._advance()
            self._expect(kind="LPAREN")
            self._expect(kind="RPAREN")

        self._skip_newlines()
        if self._current().kind == "LBRACE":
            body_command = self._parse_group_command(depth=depth)
        elif self._current().kind == "LPAREN":
            body_command = self._parse_subshell(depth=depth)
        else:
            body_command = self._parse_command(
                stop_words=frozenset(),
                stop_kinds=frozenset(),
                depth=depth,
            )
        end = body_command.text_end
        text = self.source[start:end].strip()
        programs = body_command.compound.programs if body_command.compound else ()
        return _ParsedShellCommand(
            argv=("function",),
            redirections=(),
            text=text,
            compound=ShellCompound(kind="function", programs=programs, text=text),
            text_start=start,
            text_end=end,
        )

    def _parse_do_group(self, *, stop_word: str, depth: int) -> ShellProgram:
        if self._is_word("do"):
            self._advance()
        elif self._current().kind == "LBRACE":
            self._advance()
            self._skip_newlines()
            program = self._parse_program(
                stop_words=frozenset(),
                stop_kinds=frozenset({"RBRACE"}),
                depth=depth,
            )
            self._expect(kind="RBRACE")
            return program
        else:
            raise ShellParseError(f"expected do-group near {self._current().text!r}")

        self._skip_newlines()
        program = self._parse_program(
            stop_words=frozenset({stop_word}),
            stop_kinds=frozenset(),
            depth=depth,
        )
        self._expect_word(stop_word)
        return program

    def _consume_prefix_words(self, words: set[str]) -> None:
        while self._current().kind == "WORD" and not self._current().quoted:
            if self._current().text not in words:
                break
            self._advance()
            self._skip_newlines()

    def _consume_separator(self) -> ShellSeparator | None:
        token = self._current()
        if token.kind == "AND_IF":
            self._advance()
            return "&&"
        if token.kind == "OR_IF":
            self._advance()
            return "||"
        if token.kind == "SEMI":
            self._advance()
            return ";"
        if token.kind == "AMP":
            self._advance()
            return "&"
        if token.kind == "NEWLINE":
            self._advance()
            while self._current().kind == "NEWLINE":
                self._advance()
            return "\n"
        return None

    def _consume_list_terminator(self) -> None:
        if self._current().kind in {"SEMI", "NEWLINE"}:
            self._advance()
            while self._current().kind == "NEWLINE":
                self._advance()

    def _at_list_terminator(self) -> bool:
        return self._current().kind in {"SEMI", "NEWLINE", "EOF"}

    def _at_program_end(
        self,
        *,
        stop_words: frozenset[str],
        stop_kinds: frozenset[str],
    ) -> bool:
        token = self._current()
        if token.kind == "EOF":
            return True
        if token.kind in stop_kinds:
            return True
        if token.kind == "WORD" and not token.quoted and token.text in stop_words:
            return True
        return False

    def _is_function_definition(self) -> bool:
        token = self._current()
        if token.kind == "WORD" and not token.quoted and token.text == "function":
            return True
        if token.kind != "WORD" or token.quoted:
            return False
        next_token = self._peek(1)
        next_next_token = self._peek(2)
        return next_token.kind == "LPAREN" and next_next_token.kind == "RPAREN"

    def _skip_newlines(self) -> None:
        while self._current().kind == "NEWLINE":
            self._advance()

    def _is_word(self, text: str) -> bool:
        token = self._current()
        return token.kind == "WORD" and not token.quoted and token.text == text

    def _expect_word(self, text: str) -> _Token:
        token = self._current()
        if token.kind != "WORD" or token.quoted or token.text != text:
            raise ShellParseError(f"expected {text!r}, found {token.text!r}")
        self.index += 1
        return token

    def _match(self, *, kind: str) -> bool:
        if self._current().kind != kind:
            return False
        self.index += 1
        return True

    def _expect(self, *, kind: str) -> _Token:
        token = self._current()
        if token.kind != kind:
            raise ShellParseError(f"expected {kind}, found {token.kind}")
        self.index += 1
        return token

    def _advance(self) -> _Token:
        token = self._current()
        self.index += 1
        return token

    def _current(self) -> _Token:
        return self.tokens[self.index]

    def _peek(self, offset: int) -> _Token:
        index = min(self.index + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def _previous(self) -> _Token:
        if self.index == 0:
            return self.tokens[0]
        return self.tokens[self.index - 1]


@dataclass(frozen=True, slots=True)
class _ParsedShellCommand:
    argv: tuple[str, ...]
    redirections: tuple[ShellRedirection, ...]
    text: str
    compound: ShellCompound | None
    text_start: int
    text_end: int

    @property
    def executable(self) -> str:
        return self.argv[0] if len(self.argv) > 0 else ""


def parse_shell_script(*, script: str) -> ShellProgram:
    normalized = script.replace("\r\n", "\n").strip()
    if normalized == "":
        return ShellProgram(items=())

    tokens = _ShellLexer(normalized).tokenize()
    parser = _ShellParser(source=normalized, tokens=tokens)
    return parser.parse()


def extract_heredoc_bodies(*, script: str) -> tuple[str, ...]:
    normalized = script.replace("\r\n", "\n").strip()
    if normalized == "":
        return ()

    tokens = _ShellLexer(normalized).tokenize()
    bodies: list[str] = []
    for token in tokens:
        if token.kind == "WORD" and token.heredoc_body is not None:
            bodies.append(token.heredoc_body)
    return tuple(bodies)


def _strip_internal_command(command: _ParsedShellCommand) -> ShellCommand:
    return ShellCommand(
        argv=command.argv,
        redirections=command.redirections,
        text=command.text,
        compound=command.compound,
    )


def _unquote_heredoc_delimiter(token: str) -> str:
    normalized = token.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        if normalized[0] in {"'", '"'}:
            return normalized[1:-1]
    return normalized
