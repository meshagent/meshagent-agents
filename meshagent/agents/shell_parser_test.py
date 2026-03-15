from __future__ import annotations

from .shell_parser import ShellRedirection, extract_heredoc_bodies, parse_shell_script


def test_parse_shell_script_parses_command_lists_and_pipelines() -> None:
    program = parse_shell_script(
        script=(
            "cd /website && npm install tslib @types/react @types/react-dom "
            "--cache /agents/webmaster/.npm && npm run build && "
            "find dist -maxdepth 2 -type f | sort"
        )
    )

    assert len(program.items) == 4
    assert program.items[0].separator_before is None
    assert program.items[1].separator_before == "&&"
    assert program.items[2].separator_before == "&&"
    assert program.items[3].separator_before == "&&"
    assert program.items[0].pipeline.first_command.argv == ("cd", "/website")
    assert program.items[1].pipeline.first_command.argv == (
        "npm",
        "install",
        "tslib",
        "@types/react",
        "@types/react-dom",
        "--cache",
        "/agents/webmaster/.npm",
    )
    assert len(program.items[3].pipeline.commands) == 2
    assert program.items[3].pipeline.commands[0].argv == (
        "find",
        "dist",
        "-maxdepth",
        "2",
        "-type",
        "f",
    )
    assert program.items[3].pipeline.commands[1].argv == ("sort",)


def test_parse_shell_script_preserves_quoted_words() -> None:
    program = parse_shell_script(
        script=(
            "stat -c '%A %n' node_modules/rollup/dist/bin/rollup "
            "node_modules/.bin/rollup && "
            'sed -i \'s#"build": "rollup -c"#"build": '
            '"node ./node_modules/rollup/dist/bin/rollup -c"#\' package.json'
        )
    )

    assert len(program.items) == 2
    assert program.items[0].pipeline.first_command.argv == (
        "stat",
        "-c",
        "'%A %n'",
        "node_modules/rollup/dist/bin/rollup",
        "node_modules/.bin/rollup",
    )
    assert program.items[1].pipeline.first_command.argv == (
        "sed",
        "-i",
        '\'s#"build": "rollup -c"#"build": "node ./node_modules/rollup/dist/bin/rollup -c"#\'',
        "package.json",
    )


def test_parse_shell_script_strips_heredoc_bodies_and_keeps_redirections() -> None:
    program = parse_shell_script(
        script=(
            "cd /website && cat > public/index.html <<'EOF'\n"
            "<!doctype html>\n"
            "<html></html>\n"
            "EOF\n"
            "npm run build"
        )
    )

    assert len(program.items) == 3
    cat_command = program.items[1].pipeline.first_command
    assert cat_command.argv == ("cat",)
    assert cat_command.redirections == (
        ShellRedirection(operator=">", target="public/index.html"),
        ShellRedirection(operator="<<", target="'EOF'"),
    )
    assert program.items[2].pipeline.first_command.argv == ("npm", "run", "build")


def test_parse_shell_script_supports_dash_heredoc_operator() -> None:
    program = parse_shell_script(
        script=("python - <<-'PY'\nprint('hi')\nPY\necho done")
    )

    assert len(program.items) == 2
    command = program.items[0].pipeline.first_command
    assert command.argv == ("python", "-")
    assert command.redirections == (ShellRedirection(operator="<<-", target="'PY'"),)


def test_parse_shell_script_supports_file_descriptor_redirections() -> None:
    program = parse_shell_script(
        script=(
            "cd /data/pythontest && "
            "python3 -m pip install ruff >/tmp/ruff_install.log 2>&1 && "
            "python3 -m ruff check ."
        )
    )

    assert len(program.items) == 3
    install_command = program.items[1].pipeline.first_command
    assert install_command.argv == ("python3", "-m", "pip", "install", "ruff")
    assert install_command.redirections == (
        ShellRedirection(operator=">", target="/tmp/ruff_install.log"),
        ShellRedirection(operator="2>&", target="1"),
    )
    lint_command = program.items[2].pipeline.first_command
    assert lint_command.argv == ("python3", "-m", "ruff", "check", ".")


def test_extract_heredoc_bodies_returns_embedded_script_bodies_in_order() -> None:
    bodies = extract_heredoc_bodies(
        script=("python - <<'PY'\nprint('one')\nPY\ncat > out.txt <<'EOF'\nhello\nEOF")
    )

    assert bodies == ("print('one')", "hello")


def test_parse_shell_script_supports_if_shell_command() -> None:
    program = parse_shell_script(
        script="if test -f app.py; then echo ok; else echo nope; fi"
    )

    assert len(program.items) == 1
    command = program.items[0].pipeline.first_command
    assert command.executable == "if"
    assert "then" in command.text
    assert "else" in command.text
    assert "fi" in command.text


def test_parse_shell_script_supports_if_test_command_with_bang() -> None:
    program = parse_shell_script(
        script="if [ ! -f index.html ]; then cat > index.html <<'EOF'\nhi\nEOF\nfi"
    )

    assert len(program.items) == 1
    command = program.items[0].pipeline.first_command
    assert command.executable == "if"
    assert "[ ! -f index.html ]" in command.text
    assert command.compound is not None
    assert len(command.compound.programs) >= 1


def test_parse_shell_script_supports_for_shell_command() -> None:
    program = parse_shell_script(
        script="for file in README.md app.py; do echo $file; done"
    )

    assert len(program.items) == 1
    command = program.items[0].pipeline.first_command
    assert command.executable == "for"
    assert "in README.md app.py" in command.text
    assert "done" in command.text
