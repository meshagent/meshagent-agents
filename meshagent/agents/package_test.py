import os
import sys
from pathlib import Path

import pytest

import meshagent.agents as agents_module
import meshagent.agents.package as package_module
import meshagent.cli.helper as helper_module
from meshagent.api import ApiScope, ParticipantToken
from meshagent.agents import (
    DebianPackage,
    MeshagentPackage,
    Package,
    deploy_package,
    run_package,
)
from meshagent.agents.package import Package as AgentPackage
from meshagent.agents.package import DebianPackage as PackagedDebianPackage
from meshagent.agents.package import MeshagentPackage as PackagedMeshagentPackage
from meshagent.agents.package import PythonPackage


def _write_runtime_module(tmp_path: Path, *, body: str = "print('hello')\n") -> Path:
    module_path = tmp_path / "agent.py"
    module_path.write_text(body, encoding="utf-8")
    return module_path


def test_agent_package_skills_validation_is_lazy(tmp_path: Path) -> None:
    file_path = tmp_path / "skill.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    package = Package(name="assistant").skills(str(file_path))

    with pytest.raises(ValueError, match="directory"):
        package._resolve_deploy_assets()


def test_agent_package_instructions_validation_is_lazy(tmp_path: Path) -> None:
    dir_path = tmp_path / "rules"
    dir_path.mkdir()

    package = Package(name="assistant").instructions(str(dir_path))

    with pytest.raises(ValueError, match="file"):
        package._resolve_deploy_assets()


def test_root_agents_module_exports_package_and_agent() -> None:
    assert agents_module.Package is AgentPackage
    assert agents_module.DebianPackage is PackagedDebianPackage
    assert agents_module.MeshagentPackage is PackagedMeshagentPackage
    assert agents_module.deploy_package is package_module.deploy_package
    assert agents_module.run_package is package_module.run_package


def test_package_meshagent_exposes_meshagent_package() -> None:
    package = Package.meshagent(name="assistant").chat_channel()

    assert isinstance(package, MeshagentPackage)
    assert isinstance(package, Package)
    assert Package.meshagent is MeshagentPackage
    assert package.name == "assistant"
    assert package._channels == ["chat"]


def test_package_meshagent_tool_methods_configure_runtime_flags() -> None:
    package = (
        Package.meshagent(name="assistant")
        .shell()
        .advanced_shell()
        .web_fetch()
        .web_search()
        .image_gen(model="gpt-image-1")
        .apply_patch()
        .storage(read_only=True)
        .table_read(tables=["users"], namespace="prod::analytics")
        .table_write(tables=["events"])
        .time()
        .uuid()
        .memory(path="assistant/memories", model="gpt-5.4-mini")
        .document_authoring()
        .discovery()
        .computer_use(starting_url="https://example.com", allow_goto_url=True)
        .mcp()
    )

    assert package._shell_enabled is True
    assert package._advanced_shell_enabled is True
    assert package._web_fetch_enabled is True
    assert package._web_search_enabled is True
    assert package._image_gen_enabled is True
    assert package._image_gen_model == "gpt-image-1"
    assert package._apply_patch_enabled is True
    assert package._storage_enabled is True
    assert package._storage_read_only is True
    assert package._table_read == ["users"]
    assert package._table_write == ["events"]
    assert package._dataset_namespace == ["prod", "analytics"]
    assert package._time_enabled is True
    assert package._uuid_enabled is True
    assert package._memory_config == package_module._MemoryToolConfig(
        name="memories",
        namespace=["assistant"],
        model="gpt-5.4-mini",
    )
    assert package._document_authoring_enabled is True
    assert package._discovery_enabled is True
    assert package._computer_use_config == package_module._ComputerUseConfig(
        starting_url="https://example.com",
        allow_goto_url=True,
    )
    assert package._mcp_enabled is True


def test_package_python_exposes_python_package() -> None:
    package = Package.python(name="assistant")

    assert isinstance(package, PythonPackage)
    assert isinstance(package, DebianPackage)
    assert isinstance(package, Package)
    assert Package.debian is DebianPackage
    assert Package.python is PythonPackage
    assert "install" not in Package.__dict__
    assert "apt_get_install" not in Package.__dict__
    assert "apt_get_install" in DebianPackage.__dict__
    assert "install" in PythonPackage.__dict__
    assert "chat_channel" not in Package.__dict__
    assert "chat_channel" not in PythonPackage.__dict__
    assert "chat_channel" in MeshagentPackage.__dict__


def test_package_include_workspace_defaults_by_package_type() -> None:
    assert Package(name="base")._include_workspace is False
    assert Package.debian(name="debian")._include_workspace is False
    assert Package.python(name="python")._include_workspace is True
    assert Package.meshagent(name="meshagent")._include_workspace is True


def test_agent_package_resolves_paths_lazily_from_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    instruction_path = project_dir / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")
    monkeypatch.chdir(project_dir)

    package = Package(name="assistant").instructions("rules.txt")

    assert package._instructions[0].source == Path("rules.txt")
    assert package._instructions[0].base_path is None
    deploy_assets = package._resolve_deploy_assets()
    assert deploy_assets[0].asset.source == instruction_path.resolve()


def test_agent_package_resolves_paths_lazily_from_explicit_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    instruction_path = project_dir / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    package = Package(name="assistant").instructions("rules.txt")

    deploy_assets = package._resolve_deploy_assets(root_path=project_dir)
    assert deploy_assets[0].asset.source == instruction_path.resolve()


def test_runtime_module_deploy_assets_include_filtered_workspace_files(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / ".dockerignore").write_text("docker-only.txt\n", encoding="utf-8")
    module_path = _write_runtime_module(
        tmp_path,
        body="import helper\nprint(helper.VALUE)\n",
    )
    (tmp_path / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "docker-only.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"ignored")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "local.py").write_text("ignored\n", encoding="utf-8")

    package = Package.meshagent(name="assistant")

    runtime_context, runtime_assets = package._runtime_module_deploy_assets(
        module_path=module_path
    )

    assert runtime_context.module_name is None
    assert runtime_context.runtime_command == "python agent.py"
    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/agent.py",
        "/package/helper.py",
    ]


def test_runtime_module_deploy_assets_ignore_venv_symlinks_outside_workspace(
    tmp_path: Path,
) -> None:
    module_path = _write_runtime_module(tmp_path)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    os.symlink(sys.executable, venv_bin / "python")

    package = Package.meshagent(name="assistant")

    _, runtime_assets = package._runtime_module_deploy_assets(module_path=module_path)

    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/agent.py",
    ]


def test_runtime_module_deploy_assets_can_disable_workspace_packaging(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    module_path = package_dir / "agent.py"
    module_path.write_text(
        "from .helper import VALUE\nprint(VALUE)\n", encoding="utf-8"
    )

    package = Package.meshagent(name="assistant").include_workspace(False)

    runtime_context, runtime_assets = package._runtime_module_deploy_assets(
        module_path=module_path
    )

    assert runtime_context.module_name == "pkg.agent"
    assert (
        package._runtime_command(runtime_context=runtime_context)
        == "python pkg/agent.py"
    )
    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/pkg/agent.py",
    ]


def test_runtime_module_deploy_assets_support_package_entrypoints(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    module_path = package_dir / "agent.py"
    module_path.write_text(
        "from .helper import VALUE\nprint(VALUE)\n", encoding="utf-8"
    )

    package = Package.meshagent(name="assistant")

    runtime_context, runtime_assets = package._runtime_module_deploy_assets(
        module_path=module_path
    )

    assert runtime_context.module_name == "pkg.agent"
    assert runtime_context.runtime_command == "python -m pkg.agent"
    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/pkg/__init__.py",
        "/package/pkg/agent.py",
        "/package/pkg/helper.py",
    ]


def test_runtime_module_deploy_assets_mirror_declared_assets_relative_to_module(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    subpackage_dir = package_dir / "subpkg"
    subpackage_dir.mkdir()
    (subpackage_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "rules.txt").write_text("Always be concise.\n", encoding="utf-8")
    module_path = subpackage_dir / "agent.py"
    module_path.write_text("print('hello')\n", encoding="utf-8")

    package = Package.meshagent(name="assistant").instructions(
        str(package_dir / "rules.txt")
    )

    _, runtime_assets = package._runtime_module_deploy_assets(module_path=module_path)

    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/pkg/__init__.py",
        "/package/pkg/rules.txt",
        "/package/pkg/subpkg/__init__.py",
        "/package/pkg/subpkg/agent.py",
    ]


def test_runtime_module_deploy_assets_add_entrypoint_for_bound_export(
    tmp_path: Path,
) -> None:
    module_path = _write_runtime_module(tmp_path)
    package = Package.meshagent(name="assistant")
    package._bind_module_export(export_name="main", export_is_factory=False)

    runtime_context, runtime_assets = package._runtime_module_deploy_assets(
        module_path=module_path,
        temp_dir=tmp_path / "runtime",
    )

    assert package._runtime_command(runtime_context=runtime_context) == (
        "python __meshagent_entrypoint__.py"
    )
    assert sorted(asset.asset.dest.as_posix() for asset in runtime_assets) == [
        "/package/__meshagent_entrypoint__.py",
        "/package/agent.py",
    ]


def test_packaged_file_entries_expand_directory_assets(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    skill_dir = package_dir / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    nested_dir = skill_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "helper.txt").write_text("helper\n", encoding="utf-8")
    module_path = package_dir / "agent.py"
    module_path.write_text("print('hello')\n", encoding="utf-8")

    package = (
        Package.meshagent(name="assistant")
        .include_workspace(False)
        .skills(str(skill_dir))
    )

    _, runtime_assets = package._runtime_module_deploy_assets(module_path=module_path)
    deploy_assets = package._resolve_deploy_assets()
    file_entries = package._packaged_file_entries(
        deploy_assets=deploy_assets,
        runtime_assets=runtime_assets,
    )

    assert [(entry.category, entry.dest.as_posix()) for entry in file_entries] == [
        ("mount", "/skills/skill/SKILL.md"),
        ("mount", "/skills/skill/nested/helper.txt"),
        ("runtime", "/package/pkg/agent.py"),
        ("runtime", "/package/pkg/skill/SKILL.md"),
        ("runtime", "/package/pkg/skill/nested/helper.txt"),
    ]


def test_runtime_module_deploy_assets_allow_overlapping_workspace_and_directory_asset(
    tmp_path: Path,
) -> None:
    module_path = _write_runtime_module(tmp_path)
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    (skill_dir / "helper.txt").write_text("helper\n", encoding="utf-8")

    package = Package.meshagent(name="assistant").skills(str(skill_dir))

    _, runtime_assets = package._runtime_module_deploy_assets(module_path=module_path)

    assert [asset.asset.dest.as_posix() for asset in runtime_assets] == [
        "/package/agent.py",
        "/package/skill/SKILL.md",
        "/package/skill/helper.txt",
    ]


def test_agent_package_env_rejects_reserved_meshagent_token() -> None:
    with pytest.raises(ValueError, match="MESHAGENT_TOKEN"):
        Package(name="assistant").env({"MESHAGENT_TOKEN": "override"})


def test_agent_package_env_adds_service_environment(tmp_path: Path) -> None:
    package = Package.meshagent(name="assistant").env(
        {
            "OPENAI_API_KEY": "api-key",
            "MY_SETTING": "enabled",
        }
    )
    runtime_context = package_module._runtime_module_context(
        module_path=_write_runtime_module(tmp_path)
    )

    spec = package._build_service_spec(
        room="demo-room",
        deploy_assets=[],
        runtime_assets=[],
        runtime_context=runtime_context,
        container_image="meshagent/python-sdk-slim:default",
    )

    assert spec.container.environment == [
        package_module.EnvironmentVariable(
            name="OPENAI_API_KEY",
            value="api-key",
        ),
        package_module.EnvironmentVariable(
            name="MY_SETTING",
            value="enabled",
        ),
        package_module.EnvironmentVariable(
            name="MESHAGENT_ROOM",
            value="demo-room",
        ),
        package_module.EnvironmentVariable(
            name="MESHAGENT_TOKEN",
            token=package_module.TokenValue(
                identity="assistant",
                api=ApiScope.agent_default(),
                role="agent",
            ),
        ),
    ]


def test_package_run_commands_render_in_dockerfile_order() -> None:
    package = (
        Package.python(name="assistant")
        .run("echo preparing")
        .install("requests")
        .run("mkdir -p /tmp/demo")
    )

    dockerfile = package_module._package_dockerfile_text(
        package=package,
        base_image="meshagent/cli:resolved-esgz",
    )

    assert (
        dockerfile == "FROM meshagent/cli:resolved-esgz\n"
        "RUN echo preparing\n"
        "RUN uv pip install requests\n"
        "RUN mkdir -p /tmp/demo\n"
    )


def test_package_optimization_defaults_to_enabled() -> None:
    package = Package(name="assistant")

    assert package._build_optimization_enabled() is True


def test_package_optimization_allows_disabling() -> None:
    package = Package(name="assistant").optimization(False)

    assert package._build_optimization_enabled() is False


def test_package_meshagent_default_base_image_renders_docker_args() -> None:
    package = Package.meshagent(name="assistant").install("requests")

    dockerfile = package_module._package_dockerfile_text(
        package=package,
        base_image="meshagent/python-sdk-slim:default",
    )

    assert (
        dockerfile
        == f"ARG MESHAGENT_IMAGE_PREFIX={package_module._DEFAULT_MESHAGENT_IMAGE_PREFIX}\n"
        f"FROM ${{MESHAGENT_IMAGE_PREFIX}}python-sdk-slim:{package_module.__version__}\n"
        "RUN uv pip install requests\n"
    )


def test_package_shell_default_base_image_keeps_estargz_tag() -> None:
    dockerfile = package_module._package_dockerfile_text(
        package=Package(name="assistant"),
        base_image="meshagent/shell-codex:default",
    )

    assert (
        dockerfile
        == f"ARG MESHAGENT_IMAGE_PREFIX={package_module._DEFAULT_MESHAGENT_IMAGE_PREFIX}\n"
        f"FROM ${{MESHAGENT_IMAGE_PREFIX}}shell-codex:{package_module.__version__}-esgz\n"
    )


def test_package_apt_get_install_wraps_update_and_cleanup_in_order() -> None:
    package = (
        Package.python(name="assistant")
        .apt_get_install("curl")
        .install("requests")
        .apt_get_install("git")
    )

    dockerfile = package_module._package_dockerfile_text(
        package=package,
        base_image="meshagent/cli:resolved-esgz",
    )

    assert (
        dockerfile == "FROM meshagent/cli:resolved-esgz\n"
        "RUN apt-get update\n"
        "RUN apt-get install -y curl\n"
        "RUN uv pip install requests\n"
        "RUN apt-get install -y git\n"
        "RUN rm -rf /var/lib/apt/lists/*\n"
    )


def test_package_apt_get_install_requires_separate_package_arguments() -> None:
    with pytest.raises(ValueError, match="separate arguments"):
        Package.debian(name="assistant").apt_get_install("curl git")


def test_package_install_normalizes_to_uv_pip_install() -> None:
    package = Package.python(name="assistant").install("requests>=2.0")

    dockerfile = package_module._package_dockerfile_text(
        package=package,
        base_image="meshagent/cli:resolved-esgz",
    )

    assert (
        dockerfile
        == "FROM meshagent/cli:resolved-esgz\nRUN uv pip install 'requests>=2.0'\n"
    )


def test_package_install_rejects_shell_command_syntax() -> None:
    with pytest.raises(ValueError, match="use run\\(\\) for commands"):
        Package.python(name="assistant").install("uv pip install requests")


def test_package_install_appends_optional_version() -> None:
    package = Package.python(name="assistant").install("requests", "2.32.3")

    dockerfile = package_module._package_dockerfile_text(
        package=package,
        base_image="meshagent/cli:resolved-esgz",
    )

    assert (
        dockerfile
        == "FROM meshagent/cli:resolved-esgz\nRUN uv pip install requests==2.32.3\n"
    )


def test_package_install_rejects_optional_version_for_existing_specifier() -> None:
    with pytest.raises(ValueError, match="already-versioned requirement"):
        Package.python(name="assistant").install("requests>=2.0", "2.32.3")


@pytest.mark.asyncio
async def test_build_package_image_uses_room_image_build_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import image as image_module

    captured: dict[str, object] = {}
    monkeypatch.delenv("MESHAGENT_ARCH", raising=False)

    def _fake_parse_build_tag(tag: str) -> str:
        captured["tag"] = tag
        return "parsed-tag"

    async def _fake_run_image_build_stage(**kwargs) -> None:
        captured.update(kwargs)
        captured["dockerfile"] = (Path(str(kwargs["pack"])) / "Dockerfile").read_text(
            encoding="utf-8"
        )

    monkeypatch.setattr(image_module, "_parse_build_tag", _fake_parse_build_tag)
    monkeypatch.setattr(
        image_module, "_run_image_build_stage", _fake_run_image_build_stage
    )

    package = Package.python(name="assistant").install("requests").run("echo hello")

    image_tag = await package_module._build_package_image(
        package=package,
        resolved_project_id="project-123",
        resolved_room="demo-room",
    )

    assert image_tag == "registry.meshagent.com/packages/assistant:latest"
    assert captured["tag"] == image_tag
    assert captured["resolved_project_id"] == "project-123"
    assert captured["resolved_room"] == "demo-room"
    assert captured["parsed_tag"] == "parsed-tag"
    assert captured["builder_name"] == "package-assistant"
    assert captured["arch"] == "amd64"
    assert captured["optimize"] is True
    assert (
        captured["dockerfile"]
        == f"ARG MESHAGENT_IMAGE_PREFIX={package_module._DEFAULT_MESHAGENT_IMAGE_PREFIX}\n"
        f"FROM ${{MESHAGENT_IMAGE_PREFIX}}cli:{package_module.__version__}\n"
        "RUN uv pip install requests\n"
        "RUN echo hello\n"
    )


@pytest.mark.asyncio
async def test_build_package_image_respects_package_optimization_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import image as image_module

    captured: dict[str, object] = {}

    def _fake_parse_build_tag(tag: str) -> str:
        captured["tag"] = tag
        return "parsed-tag"

    async def _fake_run_image_build_stage(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(image_module, "_parse_build_tag", _fake_parse_build_tag)
    monkeypatch.setattr(
        image_module, "_run_image_build_stage", _fake_run_image_build_stage
    )

    package = Package.python(name="assistant").install("requests").optimization(False)

    await package_module._build_package_image(
        package=package,
        resolved_project_id="project-123",
        resolved_room="demo-room",
    )

    assert captured["optimize"] is False


@pytest.mark.asyncio
async def test_build_meshagent_package_image_uses_python_sdk_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import image as image_module

    captured: dict[str, object] = {}

    def _fake_parse_build_tag(tag: str) -> str:
        captured["tag"] = tag
        return "parsed-tag"

    async def _fake_run_image_build_stage(**kwargs) -> None:
        captured.update(kwargs)
        captured["dockerfile"] = (Path(str(kwargs["pack"])) / "Dockerfile").read_text(
            encoding="utf-8"
        )

    monkeypatch.setattr(image_module, "_parse_build_tag", _fake_parse_build_tag)
    monkeypatch.setattr(
        image_module, "_run_image_build_stage", _fake_run_image_build_stage
    )

    package = Package.meshagent(name="assistant").install("requests")

    await package_module._build_package_image(
        package=package,
        resolved_project_id="project-123",
        resolved_room="demo-room",
    )

    assert captured["builder_name"] == "package-assistant"
    assert (
        captured["dockerfile"]
        == f"ARG MESHAGENT_IMAGE_PREFIX={package_module._DEFAULT_MESHAGENT_IMAGE_PREFIX}\n"
        f"FROM ${{MESHAGENT_IMAGE_PREFIX}}python-sdk-slim:{package_module.__version__}\n"
        "RUN uv pip install requests\n"
    )


@pytest.mark.asyncio
async def test_build_package_image_uses_meshagent_arch_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import image as image_module

    captured: dict[str, object] = {}

    def _fake_parse_build_tag(tag: str) -> str:
        captured["tag"] = tag
        return "parsed-tag"

    async def _fake_run_image_build_stage(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setenv("MESHAGENT_ARCH", "arm64")
    monkeypatch.setattr(image_module, "_parse_build_tag", _fake_parse_build_tag)
    monkeypatch.setattr(
        image_module, "_run_image_build_stage", _fake_run_image_build_stage
    )

    package = Package.python(name="assistant").install("requests")

    await package_module._build_package_image(
        package=package,
        resolved_project_id="project-123",
        resolved_room="demo-room",
    )

    assert captured["builder_name"] == "package-assistant"
    assert captured["arch"] == "arm64"


@pytest.mark.asyncio
async def test_build_package_image_allows_builder_name_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import image as image_module

    captured: dict[str, object] = {}

    def _fake_parse_build_tag(tag: str) -> str:
        captured["tag"] = tag
        return "parsed-tag"

    async def _fake_run_image_build_stage(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(image_module, "_parse_build_tag", _fake_parse_build_tag)
    monkeypatch.setattr(
        image_module, "_run_image_build_stage", _fake_run_image_build_stage
    )

    package = Package.python(name="assistant").install("requests")

    await package_module._build_package_image(
        package=package,
        resolved_project_id="project-123",
        resolved_room="demo-room",
        builder_name="custom-builder",
    )

    assert captured["builder_name"] == "custom-builder"


@pytest.mark.asyncio
async def test_meshagent_package_serve_adds_image_generation_toolkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self, *, turn_toolkits_builder, **kwargs) -> None:
            del kwargs
            self.turn_toolkits_builder = turn_toolkits_builder

        def register_content_scheme(self, scheme) -> None:
            del scheme
            return None

    class _FakeThreadAdapter:
        def __init__(self, *, room, path: str) -> None:
            del room, path

        def make_toolkit(self):
            return package_module.Toolkit(name="thread", tools=[])

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.local_participant = object()
            self.protocol = type(
                "_Protocol", (), {"wait_for_close": staticmethod(_wait_for_close)}
            )()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    async def _wait_for_close() -> None:
        return None

    async def _fake_start(self) -> None:
        self._state = "started"
        process = self.create_thread_process("thread")
        captured["toolkits"] = await process.turn_toolkits_builder(None, "gpt-5.4", [])

    async def _fake_stop(self) -> None:
        self._state = "stopped"

    monkeypatch.setenv("MESHAGENT_TOKEN", "test-token")
    monkeypatch.setattr(package_module, "LLMAgentProcess", _FakeProcess)
    monkeypatch.setattr(package_module, "MeshDocumentThreadStorage", _FakeThreadAdapter)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)
    monkeypatch.setattr(
        package_module.MeshagentPackage,
        "_storage_toolkit",
        lambda self, **kwargs: package_module.Toolkit(name="storage", tools=[]),
    )
    monkeypatch.setattr(package_module.AgentSupervisor, "start", _fake_start)
    monkeypatch.setattr(package_module.AgentSupervisor, "stop", _fake_stop)

    package = Package.meshagent(name="assistant").image_gen(model="gpt-image-1")

    await package._serve_async(room="demo-room")

    toolkits = captured["toolkits"]
    assert isinstance(toolkits, list)
    image_generation_toolkit = next(
        toolkit for toolkit in toolkits if toolkit.name == "image_generation"
    )
    assert len(image_generation_toolkit.tools) == 1
    assert image_generation_toolkit.tools[0].name == "image_generation"
    assert image_generation_toolkit.tools[0].model == "gpt-image-1"


@pytest.mark.asyncio
async def test_meshagent_package_serve_adds_other_requested_toolkits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meshagent.computers.agent as computer_agent_module
    import meshagent.openai.tools.responses_adapter as responses_adapter_module
    import meshagent.tools.container_shell as container_shell_module
    import meshagent.tools.dataset as datasets_module
    import meshagent.tools.datetime as datetime_tools_module
    import meshagent.tools.discovery as discovery_module
    import meshagent.tools.document_tools as document_tools_module
    import meshagent.tools.memories as memories_module
    import meshagent.tools.uuid as uuid_tools_module

    captured: dict[str, object] = {}
    instruction_path = tmp_path / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")

    class _FakeProcess:
        def __init__(self, *, turn_toolkits_builder, **kwargs) -> None:
            del kwargs
            self.turn_toolkits_builder = turn_toolkits_builder

        def register_content_scheme(self, scheme) -> None:
            del scheme
            return None

    class _FakeThreadAdapter:
        def __init__(self, *, room, path: str) -> None:
            del room, path

        def make_toolkit(self):
            return package_module.Toolkit(name="thread", tools=[])

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.local_participant = object()
            self.protocol = type(
                "_Protocol", (), {"wait_for_close": staticmethod(_wait_for_close)}
            )()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeApplyPatchTool:
        def __init__(self, *, storage, name: str = "apply_patch") -> None:
            self.name = name
            self.storage = storage

    async def _wait_for_close() -> None:
        return None

    async def _fake_start(self) -> None:
        self._state = "started"
        process = self.create_thread_process("thread")
        captured["toolkits"] = await process.turn_toolkits_builder(None, "gpt-5.4", [])

    async def _fake_stop(self) -> None:
        self._state = "stopped"

    async def _fake_make_dataset_toolkit(
        *,
        room,
        tables,
        read_only: bool,
        namespace,
    ):
        del room
        suffix = "read" if read_only else "write"
        namespace_suffix = "::".join(namespace) if namespace is not None else "none"
        return package_module.Toolkit(
            name=f"dataset.{suffix}.{','.join(tables)}.{namespace_suffix}",
            tools=[],
        )

    monkeypatch.setenv("MESHAGENT_TOKEN", "test-token")
    monkeypatch.setattr(package_module, "LLMAgentProcess", _FakeProcess)
    monkeypatch.setattr(package_module, "MeshDocumentThreadStorage", _FakeThreadAdapter)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)
    monkeypatch.setattr(
        package_module.MeshagentPackage,
        "_storage_toolkit",
        lambda self, **kwargs: package_module.Toolkit(name="storage", tools=[]),
    )
    monkeypatch.setattr(package_module.AgentSupervisor, "start", _fake_start)
    monkeypatch.setattr(package_module.AgentSupervisor, "stop", _fake_stop)
    monkeypatch.setattr(responses_adapter_module, "ApplyPatchTool", _FakeApplyPatchTool)
    monkeypatch.setattr(
        container_shell_module,
        "ContainerToolkit",
        lambda **kwargs: package_module.Toolkit(name="container", tools=[]),
    )
    monkeypatch.setattr(
        datetime_tools_module,
        "DatetimeToolkit",
        lambda: package_module.Toolkit(name="datetime", tools=[]),
    )
    monkeypatch.setattr(
        uuid_tools_module,
        "UUIDToolkit",
        lambda: package_module.Toolkit(name="uuid", tools=[]),
    )
    monkeypatch.setattr(
        memories_module,
        "MemoriesToolkit",
        lambda **kwargs: package_module.Toolkit(name="memories", tools=[]),
    )
    monkeypatch.setattr(
        discovery_module,
        "DiscoveryToolkit",
        lambda **kwargs: package_module.Toolkit(name="discovery", tools=[]),
    )
    monkeypatch.setattr(
        document_tools_module,
        "DocumentAuthoringToolkit",
        lambda **kwargs: package_module.Toolkit(
            name="meshagent.document_authoring",
            tools=[],
        ),
    )
    monkeypatch.setattr(
        document_tools_module,
        "DocumentTypeAuthoringToolkit",
        lambda **kwargs: package_module.Toolkit(
            name="meshagent.document_authoring.widget",
            tools=[],
        ),
    )
    monkeypatch.setattr(
        computer_agent_module,
        "ComputerToolkit",
        lambda **kwargs: package_module.Toolkit(
            name="meshagent.openai.computer",
            tools=[],
        ),
    )
    monkeypatch.setattr(
        datasets_module,
        "make_dataset_toolkit",
        _fake_make_dataset_toolkit,
    )

    package = (
        Package.meshagent(name="assistant")
        .instructions(str(instruction_path))
        .apply_patch()
        .advanced_shell()
        .table_read(tables=["users"], namespace="prod::analytics")
        .table_write(tables=["events"])
        .time()
        .uuid()
        .memory(path="assistant/memories", model="gpt-5.4-mini")
        .document_authoring()
        .discovery()
        .computer_use(starting_url="https://example.com", allow_goto_url=True)
    )

    await package._serve_async(room="demo-room")

    toolkits = captured["toolkits"]
    assert isinstance(toolkits, list)
    toolkit_names = [toolkit.name for toolkit in toolkits]
    assert "apply_patch" in toolkit_names
    assert "container" in toolkit_names
    assert "dataset.read.users.prod::analytics" in toolkit_names
    assert "dataset.write.events.prod::analytics" in toolkit_names
    assert "datetime" in toolkit_names
    assert "uuid" in toolkit_names
    assert "memories" in toolkit_names
    assert "meshagent.document_authoring" in toolkit_names
    assert "meshagent.document_authoring.widget" in toolkit_names
    assert "discovery" in toolkit_names
    assert "meshagent.openai.computer" in toolkit_names


@pytest.mark.asyncio
async def test_meshagent_package_serve_resolves_instructions_from_root_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    instruction_path = project_dir / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)
    captured: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self, *, turn_instructions_provider, **kwargs) -> None:
            del kwargs
            self.turn_instructions_provider = turn_instructions_provider

        def register_content_scheme(self, scheme) -> None:
            del scheme
            return None

    class _FakeThreadAdapter:
        def __init__(self, *, room, path: str) -> None:
            del room, path

        def make_toolkit(self):
            return package_module.Toolkit(name="thread", tools=[])

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.local_participant = object()
            self.protocol = type(
                "_Protocol", (), {"wait_for_close": staticmethod(_wait_for_close)}
            )()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    async def _wait_for_close() -> None:
        return None

    async def _fake_start(self) -> None:
        self._state = "started"
        process = self.create_thread_process("thread")
        captured["instructions"] = await process.turn_instructions_provider(None)

    async def _fake_stop(self) -> None:
        self._state = "stopped"

    monkeypatch.setenv("MESHAGENT_TOKEN", "test-token")
    monkeypatch.setattr(package_module, "LLMAgentProcess", _FakeProcess)
    monkeypatch.setattr(package_module, "MeshDocumentThreadStorage", _FakeThreadAdapter)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)
    monkeypatch.setattr(package_module.AgentSupervisor, "start", _fake_start)
    monkeypatch.setattr(package_module.AgentSupervisor, "stop", _fake_stop)

    package = Package.meshagent(name="assistant").instructions("rules.txt")

    await package._serve_async(room="demo-room", root_path=project_dir)

    assert captured["instructions"] == (
        "Always be concise.\n"
        "based on the previous transcript, take your turn and respond"
    )
    assert package._instructions[0].source == Path("rules.txt")
    assert package._instructions[0].base_path is None
    assert instruction_path.resolve() != (cwd_dir / "rules.txt").resolve()


@pytest.mark.asyncio
async def test_meshagent_package_serve_rejects_image_generation_for_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESHAGENT_TOKEN", "test-token")
    package = (
        Package.meshagent(name="assistant").use_model("claude-opus-4-6").image_gen()
    )

    with pytest.raises(
        ValueError, match="image generation is only supported by openai models"
    ):
        await package._serve_async(room="demo-room")


@pytest.mark.asyncio
async def test_meshagent_package_serve_rejects_apply_patch_for_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESHAGENT_TOKEN", "test-token")
    instruction_path = tmp_path / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")
    package = (
        Package.meshagent(name="assistant")
        .use_model("claude-opus-4-6")
        .instructions(str(instruction_path))
        .apply_patch()
    )

    with pytest.raises(
        ValueError, match="apply patch is only supported by openai models"
    ):
        await package._serve_async(room="demo-room")


@pytest.mark.asyncio
async def test_agent_package_run_uploads_assets_and_starts_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: list[tuple[str, bytes, bool]] = []
    captured: dict[str, object] = {}
    statuses: list[str] = []

    instruction_path = tmp_path / "rules.txt"
    instruction_path.write_text("Always be concise.\n", encoding="utf-8")
    module_path = _write_runtime_module(tmp_path)

    class _FakeContainers:
        async def run(
            self,
            *,
            image: str,
            command: str | None = None,
            working_dir: str | None = None,
            env: dict[str, str] | None = None,
            mounts=None,
            **kwargs,
        ) -> str:
            captured["image"] = image
            captured["command"] = command
            captured["working_dir"] = working_dir
            captured["env"] = env
            captured["mounts"] = mounts
            captured["extra"] = kwargs
            return "container-123"

    class _FakeRoomStorage:
        async def upload(self, *, path: str, data: bytes, overwrite: bool) -> None:
            uploaded.append((path, data, overwrite))

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.storage = _FakeRoomStorage()
            self.containers = _FakeContainers()

        def on(self, event_name: str, func) -> None:
            del event_name, func
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeAccountClient:
        async def connect_room(self, *, project_id: str, room: str):
            assert project_id == "project-123"
            assert room == "demo-room"
            return type(
                "_Connection",
                (),
                {
                    "jwt": "room-jwt",
                    "room_url": "ws://example.test/rooms/demo-room",
                },
            )()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(helper_module, "resolve_room", lambda room: room)
    monkeypatch.setenv("MESHAGENT_SECRET", "local-signing-secret")

    async def _fake_resolve_project_id(*, project_id: str | None) -> str:
        assert project_id == "project-123"
        return "project-123"

    async def _fake_resolve_key(*, project_id: str | None, key: str | None) -> None:
        assert project_id == "project-123"
        assert key is None
        return None

    async def _fake_get_client():
        return _FakeAccountClient()

    monkeypatch.setattr(helper_module, "resolve_project_id", _fake_resolve_project_id)
    monkeypatch.setattr(helper_module, "resolve_key", _fake_resolve_key)
    monkeypatch.setattr(helper_module, "get_client", _fake_get_client)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)

    package = (
        Package.meshagent(name="assistant")
        .instructions(str(instruction_path))
        .env({"OPENAI_API_KEY": "api-key"})
        .chat_channel()
    )
    package._bind_module_path(module_path=module_path)

    container_id = await run_package(
        package=package,
        room="demo-room",
        project_id="project-123",
        status_callback=statuses.append,
    )

    assert container_id == "container-123"
    assert uploaded == [
        (".agents/assistant/mounts/0/rules.txt", b"Always be concise.\n", True),
        (".agents/assistant/runtime/0/agent.py", b"print('hello')\n", True),
        (".agents/assistant/runtime/0/rules.txt", b"Always be concise.\n", True),
    ]
    assert captured["image"] == "meshagent/python-sdk-slim:default"
    assert captured["command"] == "python agent.py"
    assert captured["working_dir"] == "/package"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_API_KEY"] == "api-key"
    assert env["MESHAGENT_ROOM"] == "demo-room"
    minted_token = ParticipantToken.from_jwt(
        env["MESHAGENT_TOKEN"],
        token="local-signing-secret",
    )
    assert minted_token.name == "assistant"
    assert minted_token.role == "agent"
    assert minted_token.grant_scope("room") == "demo-room"
    assert minted_token.get_api_grant() == ApiScope.agent_default()
    assert captured["extra"] == {
        "participant_name": "assistant",
        "role": "agent",
    }
    assert statuses == [
        "Connecting to room demo-room",
        "Uploading 3 packaged assets to room demo-room",
        f"Uploading [1/3] {instruction_path} -> /instructions/rules.txt",
        f"Uploading [2/3] {module_path} -> /package/agent.py",
        f"Uploading [3/3] {instruction_path} -> /package/rules.txt",
        "Starting container from image meshagent/python-sdk-slim:default",
    ]
    mounts = captured["mounts"]
    assert mounts is not None
    assert mounts.room == [
        package_module.RoomStorageMountSpec(
            path="/instructions",
            subpath=".agents/assistant/mounts/0",
            read_only=False,
        ),
        package_module.RoomStorageMountSpec(
            path="/package",
            subpath=".agents/assistant/runtime/0",
            read_only=True,
        ),
    ]


@pytest.mark.asyncio
async def test_run_package_uses_built_image_when_commands_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    module_path = _write_runtime_module(tmp_path)

    class _FakeContainers:
        async def run(
            self,
            *,
            image: str,
            command: str | None = None,
            working_dir: str | None = None,
            **kwargs,
        ) -> str:
            captured["image"] = image
            captured["command"] = command
            captured["working_dir"] = working_dir
            captured["kwargs"] = kwargs
            return "container-123"

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory

            class _Storage:
                async def upload(
                    self,
                    *,
                    path: str,
                    data: bytes,
                    overwrite: bool,
                ) -> None:
                    del path, data, overwrite
                    return None

            self.storage = _Storage()
            self.containers = _FakeContainers()

        def on(self, event_name: str, func) -> None:
            del event_name, func
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class _FakeAccountClient:
        async def connect_room(self, *, project_id: str, room: str):
            assert project_id == "project-123"
            assert room == "demo-room"
            return type(
                "_Connection",
                (),
                {
                    "jwt": "room-jwt",
                    "room_url": "ws://example.test/rooms/demo-room",
                },
            )()

        async def close(self) -> None:
            return None

    async def _fake_resolve_project_id(*, project_id: str | None) -> str:
        assert project_id == "project-123"
        return "project-123"

    async def _fake_resolve_key(*, project_id: str | None, key: str | None) -> None:
        assert project_id == "project-123"
        assert key is None
        return None

    async def _fake_get_client():
        return _FakeAccountClient()

    async def _fake_build_package_image(**kwargs) -> str:
        captured["build_kwargs"] = kwargs
        return "registry.meshagent.com/packages/assistant:latest"

    monkeypatch.setattr(helper_module, "resolve_room", lambda room: room)
    monkeypatch.setattr(helper_module, "resolve_project_id", _fake_resolve_project_id)
    monkeypatch.setattr(helper_module, "resolve_key", _fake_resolve_key)
    monkeypatch.setattr(helper_module, "get_client", _fake_get_client)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)
    monkeypatch.setattr(
        package_module, "_build_package_image", _fake_build_package_image
    )
    monkeypatch.setenv("MESHAGENT_SECRET", "local-signing-secret")

    package = Package.meshagent(name="assistant").run("echo warmup").chat_channel()
    package._bind_module_path(module_path=module_path)

    container_id = await run_package(
        package=package,
        room="demo-room",
        project_id="project-123",
    )

    assert container_id == "container-123"
    assert captured["build_kwargs"] == {
        "package": package,
        "resolved_project_id": "project-123",
        "resolved_room": "demo-room",
        "builder_name": None,
        "status_callback": None,
    }
    assert captured["image"] == "registry.meshagent.com/packages/assistant:latest"
    assert captured["command"] == "python agent.py"
    assert captured["working_dir"] == "/package"


@pytest.mark.asyncio
async def test_agent_package_deploy_builds_service_spec_with_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    module_path = _write_runtime_module(tmp_path)

    class _FakeAccountClient:
        async def connect_room(self, *, project_id: str, room: str):
            assert project_id == "project-123"
            assert room == "demo-room"
            return type(
                "_Connection",
                (),
                {
                    "jwt": "room-jwt",
                    "room_url": "ws://example.test/rooms/demo-room",
                },
            )()

        async def list_room_services(self, *, project_id: str, room_name: str):
            raise AssertionError(
                "list_room_services should not be called on create path"
            )

        async def create_room_service(
            self, *, project_id: str, service, room_name: str
        ):
            captured["project_id"] = project_id
            captured["room_name"] = room_name
            captured["service"] = service
            return "service-123"

        async def close(self) -> None:
            return None

    class _FakeRoomStorage:
        async def upload(self, *, path: str, data: bytes, overwrite: bool) -> None:
            del path, data, overwrite
            return None

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.storage = _FakeRoomStorage()

        def on(self, event_name: str, func) -> None:
            del event_name, func
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    monkeypatch.setattr(helper_module, "resolve_room", lambda room: room)

    async def _fake_resolve_project_id(*, project_id: str | None) -> str:
        assert project_id == "project-123"
        return "project-123"

    async def _fake_get_client():
        return _FakeAccountClient()

    monkeypatch.setattr(helper_module, "resolve_project_id", _fake_resolve_project_id)
    monkeypatch.setattr(helper_module, "get_client", _fake_get_client)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)

    package = Package.meshagent(name="assistant")
    package._bind_module_path(module_path=module_path)

    service_id = await deploy_package(
        package=package,
        room="demo-room",
        project_id="project-123",
    )

    assert service_id == "service-123"
    service = captured["service"]
    assert service.kind == "Service"
    assert service.version == "v1"
    assert service.metadata.annotations == {
        package_module.ANNOTATION_SERVICE_ID: "assistant"
    }
    assert service.container.image == "meshagent/python-sdk-slim:default"
    assert service.container.command == "python agent.py"
    assert service.container.working_dir == "/package"
    assert service.container.storage.room == [
        package_module.RoomStorageMountSpec(
            path="/package",
            subpath=".agents/assistant/runtime/0",
            read_only=True,
        )
    ]


@pytest.mark.asyncio
async def test_deploy_package_uses_built_image_when_commands_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    module_path = _write_runtime_module(tmp_path)

    class _FakeAccountClient:
        async def connect_room(self, *, project_id: str, room: str):
            assert project_id == "project-123"
            assert room == "demo-room"
            return type(
                "_Connection",
                (),
                {
                    "jwt": "room-jwt",
                    "room_url": "ws://example.test/rooms/demo-room",
                },
            )()

        async def create_room_service(
            self, *, project_id: str, service, room_name: str
        ):
            captured["project_id"] = project_id
            captured["room_name"] = room_name
            captured["service"] = service
            return "service-123"

        async def close(self) -> None:
            return None

    class _FakeRoomStorage:
        async def upload(self, *, path: str, data: bytes, overwrite: bool) -> None:
            del path, data, overwrite
            return None

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.storage = _FakeRoomStorage()

        def on(self, event_name: str, func) -> None:
            del event_name, func
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    async def _fake_resolve_project_id(*, project_id: str | None) -> str:
        assert project_id == "project-123"
        return "project-123"

    async def _fake_get_client():
        return _FakeAccountClient()

    async def _fake_build_package_image(**kwargs) -> str:
        captured["build_kwargs"] = kwargs
        return "registry.meshagent.com/packages/assistant:latest"

    monkeypatch.setattr(helper_module, "resolve_room", lambda room: room)
    monkeypatch.setattr(helper_module, "resolve_project_id", _fake_resolve_project_id)
    monkeypatch.setattr(helper_module, "get_client", _fake_get_client)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)
    monkeypatch.setattr(
        package_module, "_build_package_image", _fake_build_package_image
    )

    package = Package.meshagent(name="assistant").install("requests")
    package._bind_module_path(module_path=module_path)

    service_id = await deploy_package(
        package=package,
        room="demo-room",
        project_id="project-123",
    )

    assert service_id == "service-123"
    assert captured["build_kwargs"] == {
        "package": package,
        "resolved_project_id": "project-123",
        "resolved_room": "demo-room",
        "builder_name": None,
        "status_callback": None,
    }
    assert (
        captured["service"].container.image
        == "registry.meshagent.com/packages/assistant:latest"
    )
    assert captured["service"].container.command == "python agent.py"
    assert captured["service"].container.working_dir == "/package"


@pytest.mark.asyncio
async def test_agent_package_deploy_updates_existing_service_by_service_id_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    module_path = _write_runtime_module(tmp_path)

    class _FakeAccountClient:
        async def connect_room(self, *, project_id: str, room: str):
            assert project_id == "project-123"
            assert room == "demo-room"
            return type(
                "_Connection",
                (),
                {
                    "jwt": "room-jwt",
                    "room_url": "ws://example.test/rooms/demo-room",
                },
            )()

        async def create_room_service(
            self, *, project_id: str, service, room_name: str
        ):
            del project_id, service, room_name
            raise package_module.ConflictError("service already exists")

        async def list_room_services(self, *, project_id: str, room_name: str):
            assert project_id == "project-123"
            assert room_name == "demo-room"
            return [
                package_module.ServiceSpec(
                    kind="Service",
                    version="v1",
                    id="service-456",
                    metadata=package_module.ServiceMetadata(
                        name="different-service-name",
                        annotations={package_module.ANNOTATION_SERVICE_ID: "assistant"},
                    ),
                )
            ]

        async def update_room_service(
            self, *, project_id: str, service_id: str, service, room_name: str
        ) -> None:
            captured["project_id"] = project_id
            captured["service_id"] = service_id
            captured["service"] = service
            captured["room_name"] = room_name

        async def close(self) -> None:
            return None

    class _FakeRoomStorage:
        async def upload(self, *, path: str, data: bytes, overwrite: bool) -> None:
            del path, data, overwrite
            return None

    class _FakeRoomClient:
        def __init__(self, *, protocol_factory) -> None:
            del protocol_factory
            self.storage = _FakeRoomStorage()

        def on(self, event_name: str, func) -> None:
            del event_name, func
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    monkeypatch.setattr(helper_module, "resolve_room", lambda room: room)

    async def _fake_resolve_project_id(*, project_id: str | None) -> str:
        assert project_id == "project-123"
        return "project-123"

    async def _fake_get_client():
        return _FakeAccountClient()

    monkeypatch.setattr(helper_module, "resolve_project_id", _fake_resolve_project_id)
    monkeypatch.setattr(helper_module, "get_client", _fake_get_client)
    monkeypatch.setattr(package_module, "RoomClient", _FakeRoomClient)

    package = Package.meshagent(name="assistant")
    package._bind_module_path(module_path=module_path)

    service_id = await deploy_package(
        package=package,
        room="demo-room",
        project_id="project-123",
    )

    assert service_id == "service-456"
    assert captured["project_id"] == "project-123"
    assert captured["service_id"] == "service-456"
    assert captured["room_name"] == "demo-room"
    service = captured["service"]
    assert service.id == "service-456"
    assert service.metadata.name == "assistant"
    assert service.metadata.annotations == {
        package_module.ANNOTATION_SERVICE_ID: "assistant"
    }
    cast_package = captured.get("tag")
    del cast_package
