from __future__ import annotations

import pytest

from .shell_semantics import analyze_shell_command


def test_analyze_shell_command_coalesces_cd_prefixed_exploration_chain() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && pwd && ls -la && find . -maxdepth 2 -type f | "
            "sed 's#^./##' | sort | head -200"
        )
    )

    assert [op.kind for op in analysis.operations] == ["explore"]
    assert analysis.operations[0].mode == "explore"
    assert analysis.operations[0].path == "/website"
    assert analysis.display.event_kind == "exec"
    assert analysis.display.path == "/website"
    assert analysis.display.coalesce_path == "/website"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to explore /website"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Exploring /website"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Explored /website"
    )


def test_analyze_shell_command_coalesces_cd_prefixed_read_chain() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && sed -n '1,220p' public/index.html && "
            "printf '\\n---CSS---\\n' && sed -n '1,260p' public/styles.css && "
            "printf '\\n---JS---\\n' && sed -n '1,260p' public/app.js"
        )
    )

    assert [op.kind for op in analysis.operations] == ["explore"]
    assert analysis.operations[0].path == "/website"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to explore /website"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Exploring /website"
    )


def test_analyze_shell_command_renders_single_heredoc_write() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && cat > public/index.html <<'EOF'\n"
            "<!doctype html>\n"
            "<html></html>\n"
            "EOF"
        )
    )

    assert [op.kind for op in analysis.operations] == ["write"]
    assert analysis.operations[0].paths == ("/website/public/index.html",)
    assert analysis.display.event_kind == "file"
    assert analysis.display.path == "/website/public/index.html"
    assert analysis.display.preview == (
        "cd /website && cat > public/index.html <<'EOF'\n"
        "<!doctype html>\n"
        "<html></html>\n"
        "EOF"
    )
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to write /website/public/index.html"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Writing /website/public/index.html"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Wrote /website/public/index.html"
    )
    assert analysis.display.phase_for_state(state="failed").headline == (
        "Attempted to write file /website/public/index.html"
    )


def test_analyze_shell_command_preview_truncates_after_five_lines() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && cat > public/index.html <<'EOF'\n"
            "line 1\n"
            "line 2\n"
            "line 3\n"
            "line 4\n"
            "line 5\n"
            "line 6\n"
            "EOF"
        )
    )

    assert analysis.display.preview == (
        "cd /website && cat > public/index.html <<'EOF'\n"
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "..."
    )


def test_analyze_shell_command_groups_multi_file_heredoc_writes() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && mkdir -p src public dist && "
            "cat > package.json <<'EOF'\n{}\nEOF\n"
            "cat > tsconfig.json <<'EOF'\n{}\nEOF\n"
            "cat > src/main.tsx <<'EOF'\nconsole.log('hi')\nEOF\n"
            "npm install\n"
            "npm run build"
        )
    )

    assert analysis.operations[0].kind == "write"
    assert analysis.operations[0].path == "/website"
    assert analysis.operations[0].paths == (
        "/website/package.json",
        "/website/tsconfig.json",
        "/website/src/main.tsx",
    )
    assert analysis.display.event_kind == "file"
    assert analysis.display.preview == (
        "cd /website && mkdir -p src public dist && cat > package.json <<'EOF'\n"
        "{}\n"
        "EOF\n"
        "cat > tsconfig.json <<'EOF'\n"
        "{}\n"
        "..."
    )
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to write files in /website"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Writing files in /website"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Wrote files in /website"
    )


def test_analyze_shell_command_treats_python_heredoc_file_generation_as_write() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /data/docs && python - <<'PY'\n"
            "for name in ['index.html', 'faq.html']:\n"
            "    with open(name, 'w', encoding='utf-8') as f:\n"
            "        f.write('hello')\n"
            "PY\n"
            "ls -1 /data/docs"
        )
    )

    assert analysis.operations[0].kind == "write"
    assert analysis.operations[0].path == "/data/docs"
    assert analysis.operations[0].multi is True
    assert analysis.operations[-1].kind == "explore"
    assert analysis.display.event_kind == "file"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to write files in /data/docs"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Writing files in /data/docs"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Wrote files in /data/docs"
    )


def test_analyze_shell_command_treats_if_guarded_heredoc_write_as_write() -> None:
    analysis = analyze_shell_command(
        command=(
            "mkdir -p /website/docs && cd /website/docs && "
            "if [ ! -f index.html ]; then cat > index.html <<'EOF'\n"
            "<!doctype html>\n"
            "EOF\n"
            "fi"
        )
    )

    assert analysis.operations[0].kind == "write"
    assert analysis.operations[0].path == "/website/docs/index.html"
    assert analysis.operations[0].paths == ("/website/docs/index.html",)
    assert analysis.display.event_kind == "file"
    assert analysis.display.path == "/website/docs/index.html"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to write /website/docs/index.html"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Writing /website/docs/index.html"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Wrote /website/docs/index.html"
    )


def test_analyze_shell_command_treats_for_loop_grep_as_search_in_cwd() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /tmp/meshsite && for f in home.html meshagent-sdk.html "
            "meshagent-server.html meshagent-studio.html powerboards.html "
            "faqs.html; do echo '===== '$f; grep -oiE "
            "'agent|sdk|server|studio|powerboards|room|api|monitor|orchestrat|"
            "deploy|runtime|workflow|realtime|tool|memory|observ|trace|mcp|"
            'integration|enterprise\' "$f" | sort | uniq -c | sort -nr | '
            "head -40; done"
        )
    )

    assert [op.kind for op in analysis.operations] == ["explore", "search"]
    assert analysis.operations[-1].path == "/tmp/meshsite"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to search /tmp/meshsite"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Searching /tmp/meshsite"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Searched /tmp/meshsite"
    )


def test_analyze_shell_command_falls_back_to_shell_script_for_complex_parse_failures() -> (
    None
):
    analysis = analyze_shell_command(
        command="cd /tmp/meshsite && for f in a b; do echo $f"
    )

    assert [op.kind for op in analysis.operations] == ["script"]
    assert analysis.operations[0].path == "/tmp/meshsite"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to run shell script in /tmp/meshsite"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Running shell script in /tmp/meshsite"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Ran shell script in /tmp/meshsite"
    )


def test_analyze_shell_command_tracks_install_build_and_exploration_sequence() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && npm install tslib @types/react @types/react-dom "
            "--cache /agents/webmaster/.npm && npm run build && "
            "find dist -maxdepth 2 -type f | sort"
        )
    )

    assert [op.kind for op in analysis.operations] == ["install", "build", "explore"]
    assert analysis.operations[0].path == "/website"
    assert analysis.operations[1].path == "/website"
    assert analysis.operations[2].path == "/website/dist"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to install packages and preparing to build project in /website"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Installing packages and building project in /website"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Installed packages and built project in /website"
    )


def test_analyze_shell_command_combines_install_and_lint_in_same_directory() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /data/pythontest && "
            "python3 -m pip install ruff >/tmp/ruff_install.log 2>&1 && "
            "python3 -m ruff check ."
        )
    )

    assert [op.kind for op in analysis.operations] == ["install", "lint"]
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to install packages and preparing to check code in /data/pythontest"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Installing packages and checking code in /data/pythontest"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Installed packages and checked code in /data/pythontest"
    )
    assert analysis.display.preview == (
        "cd /data/pythontest && python3 -m pip install ruff >/tmp/ruff_install.log 2>&1 && python3 -m ruff check ."
    )


def test_analyze_shell_command_tracks_edit_then_build_sequence() -> None:
    analysis = analyze_shell_command(
        command=(
            "cd /website && stat -c '%A %n' node_modules/rollup/dist/bin/rollup "
            "node_modules/.bin/rollup && sed -i "
            '\'s#"build": "rollup -c"#"build": '
            '"node ./node_modules/rollup/dist/bin/rollup -c"#\' package.json && '
            'sed -i \'s#"dev": "rollup -c -w"#"dev": '
            '"node ./node_modules/rollup/dist/bin/rollup -c -w"#\' package.json && '
            "npm run build"
        )
    )

    assert [op.kind for op in analysis.operations] == [
        "explore",
        "edit",
        "edit",
        "build",
    ]
    assert analysis.operations[0].path == "/website/node_modules"
    assert analysis.operations[1].path == "/website/package.json"
    assert analysis.operations[2].path == "/website/package.json"
    assert analysis.operations[3].path == "/website"
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Building /website"
    )


def test_analyze_shell_command_recognizes_search_display() -> None:
    analysis = analyze_shell_command(
        command="cd /website && rg --ignore-case hero public/index.html"
    )

    assert [op.kind for op in analysis.operations] == ["search"]
    assert analysis.operations[0].query == "hero"
    assert analysis.operations[0].path == "/website/public/index.html"
    assert analysis.display.path == "/website/public/index.html"
    assert analysis.display.coalesce_path == "/website/public/index.html"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to search /website/public/index.html"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Searching /website/public/index.html"
    )
    assert analysis.display.details == ("Pattern: hero",)


def test_analyze_shell_command_recognizes_download_display() -> None:
    analysis = analyze_shell_command(
        command="cd /website && curl -L https://example.com/logo.svg -o public/logo.svg"
    )

    assert [op.kind for op in analysis.operations] == ["download"]
    assert analysis.operations[0].path == "/website/public/logo.svg"
    assert analysis.display.event_kind == "file"
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing to download /website/public/logo.svg"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Downloading /website/public/logo.svg"
    )
    assert analysis.display.phase_for_state(state="completed").headline == (
        "Downloaded /website/public/logo.svg"
    )
    assert analysis.display.preview == (
        "cd /website && curl -L https://example.com/logo.svg -o public/logo.svg"
    )


def test_analyze_shell_command_treats_curl_without_output_as_web_request() -> None:
    analysis = analyze_shell_command(command="curl -sS http://127.0.0.1:8080/")

    assert [op.kind for op in analysis.operations] == ["request"]
    assert analysis.display.phase_for_state(state="pending").headline == (
        "Preparing web request"
    )
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Making web request"
    )
    assert analysis.display.preview == "curl -sS http://127.0.0.1:8080/"
    assert analysis.display.details == ("http://127.0.0.1:8080/",)


def test_analyze_shell_command_falls_back_to_run_display() -> None:
    analysis = analyze_shell_command(command="node scripts/custom.js --flag")

    assert [op.kind for op in analysis.operations] == ["run"]
    assert analysis.display.event_kind == "exec"
    assert analysis.display.path == ""
    assert analysis.display.phase_for_state(state="pending").headline == ("Preparing")
    assert analysis.display.phase_for_state(state="in_progress").headline == (
        "Running command"
    )
    assert analysis.display.phase_for_state(state="failed").headline == (
        "Attempted to run command"
    )
    assert analysis.display.preview == "node scripts/custom.js --flag"


@pytest.mark.parametrize(
    ("command", "expected_kind", "expected_headline"),
    [
        ("npx vite build", "build", "Building project"),
        ("bunx vite build", "build", "Building project"),
        ("astro build", "build", "Building project"),
        (
            "npm exec webpack -- --mode production",
            "build",
            "Building project",
        ),
        (
            "node ./node_modules/rollup/dist/bin/rollup -c",
            "build",
            "Building project",
        ),
        ("npm run dev", "dev", "Starting dev command in project"),
        ("next dev", "dev", "Starting dev command in project"),
        ("webpack serve", "dev", "Starting dev command in project"),
        ("uv run pytest -q", "test", "Running tests in project"),
        ("python -m pytest -q", "test", "Running tests in project"),
        ("pnpm exec vitest run", "test", "Running tests in project"),
        ("npx playwright test", "test", "Running tests in project"),
        ("cargo test", "test", "Running tests in project"),
        ("uv run ruff check .", "lint", "Checking code in project"),
        ("npx prettier --write .", "lint", "Checking code in project"),
        ("pnpm dlx prettier --write .", "lint", "Checking code in project"),
        ("cargo clippy", "lint", "Checking code in project"),
        ("poetry add requests", "install", "Installing packages in project"),
        ("composer install", "install", "Installing packages in project"),
        ("git grep hero src", "search", "Searching src"),
        ("git status --short", "explore", "Exploring project"),
        ("fd main src", "explore", "Exploring src"),
    ],
)
def test_analyze_shell_command_supports_common_dev_clis_and_utilities(
    *,
    command: str,
    expected_kind: str,
    expected_headline: str,
) -> None:
    analysis = analyze_shell_command(command=command)

    assert analysis.operations[0].kind == expected_kind
    assert (
        analysis.display.phase_for_state(state="in_progress").headline
        == expected_headline
    )
