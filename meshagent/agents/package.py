from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import ClassVar, Literal, Optional

from meshagent.api import (
    ApiScope,
    FileContent,
    ParticipantToken,
    RoomClient,
    WebSocketClientProtocol,
)
from meshagent.api.agent_content import AGENT_CONTENT_TYPE_TEXT
from meshagent.api.client import ConflictError
from meshagent.api.helpers import websocket_room_url
from meshagent.api.specs.service import (
    ANNOTATION_AGENT_TYPE,
    ANNOTATION_SERVICE_ID,
    AgentSpec,
    ContainerMountSpec,
    ContainerSpec,
    EnvironmentVariable,
    HeartbeatSpec,
    RoomStorageMountSpec,
    ServiceMetadata,
    ServiceSpec,
    TokenValue,
)
from meshagent.agents.images_dataset import ImagesDataset
from meshagent.agents.messages import (
    AGENT_EVENT_THREAD_EVENT,
    AgentTextContent,
    AgentThreadEvent,
)
from meshagent.tools import Toolkit
from meshagent.tools.hosting import _RemoteToolkitWrapper, start_hosted_toolkit
from meshagent.tools.storage import (
    StorageToolLocalMount,
    StorageToolRoomMount,
    StorageToolkit,
)

from .chat_channel import MessagingChatChannel
from .config import RulesConfig
from .mail_channel import MailChannel
from .process import AgentSupervisor, ContentScheme, LLMAgentProcess, Message
from .process_thread_adapter import MeshDocumentThreadStorage
from .queue_channel import QueueChannel
from .thread_status_publisher import AgentMessageThreadStatusPublisher
from .toolkit_channel import ToolkitChannel
from .skills import to_prompt
from .version import __version__


AssetKind = Literal["mount", "file", "skill", "instruction"]
_AGENT_BASE_PATH: ContextVar[Path | None] = ContextVar("_AGENT_BASE_PATH", default=None)
_RESERVED_ENV_NAMES = {"MESHAGENT_TOKEN", "MESHAGENT_ROOM"}
_DEFAULT_PACKAGE_IMAGE = "meshagent/cli:default"
_DEFAULT_MESHAGENT_PACKAGE_BUILD_IMAGE = "meshagent/python-sdk-slim:default"
_DEFAULT_MESHAGENT_IMAGE_PREFIX = "us-central1-docker.pkg.dev/meshagent-public/images/"
_DEFAULT_DATABASE_NAMESPACE = (".datasets",)
_PACKAGE_RUNTIME_DIR = PurePosixPath("/package")
_PACKAGE_ENTRYPOINT_NAME = "__meshagent_entrypoint__.py"

logger = logging.getLogger("package")


def _meshagent_default_image_tag_for_repository(*, repository: str) -> str:
    if repository.startswith("shell-"):
        return f"{__version__}-esgz"
    return __version__


@dataclass(frozen=True, slots=True)
class _Asset:
    kind: AssetKind
    source: Path
    dest: PurePosixPath
    read_only: bool
    base_path: Path | None = None

    def resolve(self, *, root_path: Path | None = None) -> _ResolvedAsset:
        resolved_source = _resolve_asset_source(
            source=self.source,
            configured_base_path=self.base_path,
            root_path=root_path,
        )
        if self.kind == "instruction" and not resolved_source.is_file():
            raise ValueError("instructions() requires a file source path")
        if self.kind == "skill" and not resolved_source.is_dir():
            raise ValueError("skills() requires a directory source path")
        return _ResolvedAsset(
            kind=self.kind,
            source=resolved_source,
            dest=self.dest,
            read_only=self.read_only,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedAsset:
    kind: AssetKind
    source: Path
    dest: PurePosixPath
    read_only: bool

    @property
    def is_file(self) -> bool:
        return self.source.is_file()

    @property
    def is_dir(self) -> bool:
        return self.source.is_dir()


@dataclass(frozen=True, slots=True)
class _Heartbeat:
    cron: str
    prompt: str


@dataclass(frozen=True, slots=True)
class _MemoryToolConfig:
    name: str
    namespace: list[str] | None
    model: str | None


@dataclass(frozen=True, slots=True)
class _ComputerUseConfig:
    starting_url: str | None
    allow_goto_url: bool


@dataclass(frozen=True, slots=True)
class _MountGroup:
    mount_path: PurePosixPath
    room_subpath: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class _DeployAsset:
    asset: _ResolvedAsset
    room_path: str
    mount_group: _MountGroup


@dataclass(frozen=True, slots=True)
class _PackagedFileEntry:
    category: Literal["mount", "runtime"]
    source: Path
    dest: PurePosixPath


@dataclass(frozen=True, slots=True)
class _RuntimeModuleContext:
    module_path: Path
    import_root: Path
    module_name: str | None

    @property
    def runtime_entry_relpath(self) -> PurePosixPath:
        return PurePosixPath(self.module_path.relative_to(self.import_root).as_posix())

    @property
    def runtime_entry_dest(self) -> PurePosixPath:
        return _PACKAGE_RUNTIME_DIR / self.runtime_entry_relpath

    @property
    def runtime_module_dir(self) -> PurePosixPath:
        relative_parent = self.runtime_entry_relpath.parent
        if str(relative_parent) == ".":
            return _PACKAGE_RUNTIME_DIR
        return _PACKAGE_RUNTIME_DIR / relative_parent

    @property
    def runtime_command(self) -> str:
        if self.module_name is not None:
            return f"python -m {self.module_name}"
        return f"python {shlex.quote(self.runtime_entry_relpath.as_posix())}"


@dataclass(frozen=True, slots=True)
class _RunBuildStep:
    command: str


@dataclass(frozen=True, slots=True)
class _AptGetInstallBuildStep:
    packages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PythonInstallBuildStep:
    requirements: tuple[str, ...]


type _ImageBuildStep = _RunBuildStep | _AptGetInstallBuildStep | _PythonInstallBuildStep
type _StatusCallback = Callable[[str], None]


@contextmanager
def _agent_base_path_scope(base_path: Path):
    token = _AGENT_BASE_PATH.set(base_path.resolve())
    try:
        yield
    finally:
        _AGENT_BASE_PATH.reset(token)


def _normalize_source_path(source: str | Path, *, base_path: Path) -> Path:
    resolved = Path(source).expanduser()
    if not resolved.is_absolute():
        resolved = base_path / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"local path not found: {source}")
    return resolved


def _normalize_root_path(
    root_path: str | Path | None,
    *,
    field_name: str = "root_path",
) -> Path | None:
    if root_path is None:
        return None

    resolved = Path(root_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} not found: {root_path}")
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must be a directory")
    return resolved


def _resolve_asset_source(
    *,
    source: Path,
    configured_base_path: Path | None,
    root_path: Path | None,
) -> Path:
    resolved_root = (
        root_path
        if root_path is not None
        else configured_base_path
        if configured_base_path is not None
        else Path.cwd().resolve()
    )
    return _normalize_source_path(source, base_path=resolved_root)


def _normalize_dest_path(dest: str) -> PurePosixPath:
    cleaned = dest.strip()
    if cleaned == "":
        raise ValueError("destination path must not be empty")

    normalized = PurePosixPath(cleaned)
    if not normalized.is_absolute():
        normalized = PurePosixPath("/") / normalized

    if any(part in {".", ".."} for part in normalized.parts):
        raise ValueError(f"invalid destination path: {dest}")

    return normalized


def _default_asset_dest(*, kind: AssetKind, source: Path) -> PurePosixPath:
    if kind == "skill":
        return PurePosixPath("/skills") / source.name
    if kind == "instruction":
        return PurePosixPath("/instructions") / source.name
    return PurePosixPath("/") / source.name


def _normalize_asset(
    *,
    kind: AssetKind,
    source: str,
    dest: str | None,
    read_only: bool,
    base_path: Path | None,
) -> _Asset:
    normalized_source = Path(source).expanduser()
    if str(normalized_source).strip() == "":
        raise ValueError("source path must not be empty")
    resolved_dest = (
        _normalize_dest_path(dest)
        if dest is not None
        else _default_asset_dest(kind=kind, source=normalized_source)
    )

    return _Asset(
        kind=kind,
        source=normalized_source,
        dest=resolved_dest,
        read_only=read_only,
        base_path=base_path.resolve() if base_path is not None else None,
    )


def _slugify_segment(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.strip()
    ).strip("-")
    return normalized or "agent"


def _canonical_channel(channel: str) -> str:
    normalized = channel.strip()
    if normalized == "":
        raise ValueError("channel must not be empty")

    lowered = normalized.casefold()
    if lowered == "chat":
        return "chat"
    if lowered.startswith("mail:"):
        address = normalized[5:].strip()
        if address == "":
            raise ValueError("mail channel requires an email address")
        return f"mail:{address}"
    if lowered.startswith("queue:"):
        queue_name = normalized[6:].strip()
        if queue_name == "":
            raise ValueError("queue channel requires a queue name")
        return f"queue:{queue_name}"
    if lowered.startswith("toolkit:"):
        toolkit_name = normalized[8:].strip()
        if toolkit_name == "":
            raise ValueError("toolkit channel requires a toolkit name")
        return f"toolkit:{toolkit_name}"

    raise ValueError(f"unsupported channel: {channel}")


def _heartbeat_minutes(schedule: str) -> int:
    normalized = schedule.strip().lower()
    if normalized == "":
        raise ValueError("heartbeat schedule must not be empty")

    aliases = {
        "@hourly": 60,
        "@daily": 24 * 60,
        "@weekly": 7 * 24 * 60,
    }
    alias_minutes = aliases.get(normalized)
    if alias_minutes is not None:
        return alias_minutes

    parts = normalized.split()
    if len(parts) != 5:
        raise ValueError(
            "heartbeat schedule must use @hourly/@daily/@weekly or a simple cron pattern"
        )

    minute, hour, day, month, weekday = parts
    if hour == day == month == weekday == "*" and minute.startswith("*/"):
        return int(minute[2:])
    if minute.isdigit() and day == month == weekday == "*" and hour.startswith("*/"):
        return int(hour[2:]) * 60
    if minute.isdigit() and hour == day == month == weekday == "*":
        return 60
    if minute.isdigit() and hour.isdigit() and day == month == weekday == "*":
        return 24 * 60

    raise ValueError(
        "unsupported heartbeat schedule; use @hourly/@daily/@weekly, '*/N * * * *', 'M * * * *', or 'M */N * * *'"
    )


def _group_mount_path(asset: _ResolvedAsset) -> PurePosixPath:
    if asset.is_file:
        parent = asset.dest.parent
        if str(parent) == ".":
            return PurePosixPath("/")
        return parent
    return asset.dest


def _annotation_service_id(annotations: dict[str, str] | None) -> str | None:
    if annotations is None:
        return None
    value = annotations.get(ANNOTATION_SERVICE_ID)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_storage_rules_path(path: str) -> str:
    normalized = Path(path)
    if not normalized.is_absolute():
        normalized = Path("/") / normalized
    return normalized.as_posix()


def _runtime_module_context(*, module_path: Path) -> _RuntimeModuleContext:
    resolved_module_path = module_path.expanduser().resolve()
    module_directory = resolved_module_path.parent
    package_parts: list[str] = []
    cursor = module_directory
    while (cursor / "__init__.py").is_file():
        package_parts.append(cursor.name)
        cursor = cursor.parent

    import_root = cursor.resolve()
    module_name: str | None = None
    if len(package_parts) > 0:
        if resolved_module_path.name == "__init__.py":
            module_parts = list(reversed(package_parts))
        else:
            module_parts = [*reversed(package_parts), resolved_module_path.stem]
        module_name = ".".join(module_parts)
    elif resolved_module_path.name == "__init__.py":
        raise ValueError(
            "__init__.py cannot be used as a package entrypoint outside of a package"
        )

    return _RuntimeModuleContext(
        module_path=resolved_module_path,
        import_root=import_root,
        module_name=module_name,
    )


def _is_special_ignored_runtime_path(*, relative_path: PurePosixPath) -> bool:
    ignored_directories = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
        "node_modules",
    }
    if any(part in ignored_directories for part in relative_path.parts):
        return True

    filename = relative_path.name
    if filename in {".DS_Store", ".gitignore", ".dockerignore"}:
        return True
    if filename.endswith((".pyc", ".pyo")):
        return True

    return False


def _workspace_ignore_patterns(*, workspace_root: Path) -> list[str]:
    patterns: list[str] = []
    for ignore_name in (".gitignore", ".dockerignore"):
        ignore_path = workspace_root / ignore_name
        if not ignore_path.is_file():
            continue
        patterns.extend(ignore_path.read_text(encoding="utf-8").splitlines())
    return patterns


def _workspace_runtime_code_assets(
    *,
    runtime_context: _RuntimeModuleContext,
) -> list[_ResolvedAsset]:
    import pathspec

    workspace_root = runtime_context.import_root
    ignore_spec = pathspec.PathSpec.from_lines(
        "gitignore",
        _workspace_ignore_patterns(workspace_root=workspace_root),
    )

    runtime_assets_by_dest: dict[str, _ResolvedAsset] = {}
    for source_path in sorted(workspace_root.rglob("*")):
        if not source_path.is_file():
            continue

        relative_source = PurePosixPath(
            source_path.relative_to(workspace_root).as_posix()
        )
        if _is_special_ignored_runtime_path(relative_path=relative_source):
            continue
        if (
            source_path.resolve() != runtime_context.module_path
            and ignore_spec.match_file(relative_source.as_posix())
        ):
            continue

        resolved_source_path = source_path.resolve()
        try:
            resolved_source_path.relative_to(workspace_root)
        except ValueError:
            # Ignore symlinks that escape the packaged workspace.
            continue

        runtime_dest = _PACKAGE_RUNTIME_DIR / relative_source
        runtime_assets_by_dest[runtime_dest.as_posix()] = _ResolvedAsset(
            kind="file",
            source=resolved_source_path,
            dest=runtime_dest,
            read_only=True,
        )

    runtime_assets_by_dest.setdefault(
        runtime_context.runtime_entry_dest.as_posix(),
        _ResolvedAsset(
            kind="file",
            source=runtime_context.module_path,
            dest=runtime_context.runtime_entry_dest,
            read_only=True,
        ),
    )

    return [
        runtime_assets_by_dest[path] for path in sorted(runtime_assets_by_dest.keys())
    ]


def _runtime_entrypoint_source(
    *,
    runtime_context: _RuntimeModuleContext,
    export_name: str,
    export_is_factory: bool,
    include_workspace: bool,
) -> str:
    runtime_module_dir = runtime_context.runtime_module_dir.as_posix()
    runtime_entry_relpath = runtime_context.runtime_entry_relpath.as_posix()
    if include_workspace and runtime_context.module_name is not None:
        load_module_lines = [
            "import importlib",
            f"module = importlib.import_module({runtime_context.module_name!r})",
        ]
    else:
        load_module_lines = [
            "import importlib.util",
            (
                "spec = importlib.util.spec_from_file_location("
                f"'_meshagent_package_module', {runtime_entry_relpath!r})"
            ),
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError('unable to load packaged module entrypoint')",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
        ]

    resolve_export_lines = (
        [f"package = module.{export_name}()"]
        if export_is_factory
        else [f"package = module.{export_name}"]
    )
    body_lines = [
        "from meshagent.agents import MeshagentPackage",
        "import meshagent.agents.package as package_module",
        "",
        (
            "with package_module._agent_base_path_scope("
            f"package_module.Path({runtime_module_dir!r})):"
        ),
        *[f"    {line}" for line in load_module_lines],
        *[f"    {line}" for line in resolve_export_lines],
        "    if not isinstance(package, MeshagentPackage):",
        "        raise TypeError('package export must resolve to MeshagentPackage')",
        "    package.serve()",
        "",
    ]
    return "\n".join(body_lines)


def _room_content_scheme(*, room: RoomClient) -> ContentScheme:
    async def _download(url: str) -> FileContent:
        if not url.startswith("room://"):
            raise ValueError(f"unsupported room file url: {url}")
        path = PurePosixPath("/" + url.removeprefix("room://").lstrip("/")).as_posix()
        if path == "/":
            raise ValueError("room file url must reference a non-root storage path")
        return await room.storage.download(path=path.lstrip("/"))

    return ContentScheme(prefix="room://", download=_download)


def _emit_status(
    *,
    status_callback: _StatusCallback | None,
    message: str,
) -> None:
    if status_callback is not None:
        status_callback(message)


def _bind_room_status_callback(
    *,
    room_client: RoomClient,
    status_callback: _StatusCallback | None,
) -> None:
    if status_callback is None:
        return

    def _on_room_status(**kwargs: object) -> None:
        status = kwargs.get("status")
        message = kwargs.get("message")
        status_text = status.strip() if isinstance(status, str) else ""
        message_text = message.strip() if isinstance(message, str) else ""
        if status_text != "" and message_text != "":
            _emit_status(
                status_callback=status_callback,
                message=f"Room status {status_text}: {message_text}",
            )
            return
        if message_text != "":
            _emit_status(
                status_callback=status_callback,
                message=f"Room status {message_text}",
            )
            return
        if status_text != "":
            _emit_status(
                status_callback=status_callback,
                message=f"Room status {status_text}",
            )

    room_client.on("room.status", _on_room_status)


def _has_chat_channel(*, channels: list[str]) -> bool:
    return "chat" in channels


def _normalize_environment_values(values: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, value in values.items():
        normalized_name = name.strip()
        if normalized_name == "":
            raise ValueError("environment variable name must not be empty")
        if normalized_name in _RESERVED_ENV_NAMES:
            raise ValueError(f"{normalized_name} is reserved")
        normalized[normalized_name] = value
    return normalized


def _normalize_container_command(*, command: str, method_name: str) -> str:
    normalized = command.strip()
    if normalized == "":
        raise ValueError(f"{method_name} command must not be empty")
    return normalized


def _normalize_apt_package_name(package: str) -> str:
    normalized = package.strip()
    if normalized == "":
        raise ValueError("apt_get_install package must not be empty")
    if any(character.isspace() for character in normalized):
        raise ValueError(
            "apt_get_install packages must be passed as separate arguments"
        )
    return normalized


def _normalize_apt_install_packages(packages: tuple[str, ...]) -> tuple[str, ...]:
    if len(packages) == 0:
        raise ValueError("apt_get_install requires at least one package")
    return tuple(_normalize_apt_package_name(package) for package in packages)


def _normalize_install_arguments(requirement: str) -> tuple[str, ...]:
    normalized = requirement.strip()
    if normalized == "":
        raise ValueError("install requirement must not be empty")
    arguments = tuple(shlex.split(normalized))
    if len(arguments) == 0:
        raise ValueError("install requirement must not be empty")
    if len(arguments) >= 3 and arguments[:3] == ("uv", "pip", "install"):
        raise ValueError(
            "install() expects package requirements, not shell commands; use run() for commands"
        )
    if len(arguments) >= 2 and arguments[:2] == ("pip", "install"):
        raise ValueError(
            "install() expects package requirements, not shell commands; use run() for commands"
        )
    return arguments


def _normalize_install_version(version: str) -> str:
    normalized = version.strip()
    if normalized == "":
        raise ValueError("install version must not be empty")
    arguments = tuple(shlex.split(normalized))
    if len(arguments) != 1:
        raise ValueError("install version must be a single value")
    return arguments[0]


def _normalize_optional_string(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_named_values(
    values: list[str] | tuple[str, ...],
    *,
    field_name: str,
) -> list[str]:
    normalized_values: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized == "":
            raise ValueError(f"{field_name} must not contain empty values")
        normalized_values.append(normalized)
    if len(normalized_values) == 0:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_values


def _parse_dataset_namespace(namespace: str | None) -> list[str]:
    if namespace is None:
        return [*_DEFAULT_DATABASE_NAMESPACE]
    normalized = _normalize_optional_string(namespace, field_name="namespace")
    assert normalized is not None
    return normalized.split("::")


def _parse_memory_path(path: str) -> tuple[str, list[str] | None]:
    normalized = path.strip()
    if normalized == "":
        raise ValueError("memory path must not be empty")
    segments = [segment.strip() for segment in normalized.split("/")]
    if any(segment == "" for segment in segments):
        raise ValueError(
            "memory path must be '<name>' or '<namespace>/<name>' with no empty segments"
        )
    return segments[-1], segments[:-1] or None


def _versioned_install_arguments(
    *,
    requirement: str,
    version: str | None,
) -> tuple[str, ...]:
    arguments = _normalize_install_arguments(requirement)
    if version is None:
        return arguments

    if len(arguments) != 1:
        raise ValueError("install version requires a single package requirement")

    normalized_version = _normalize_install_version(version)
    requirement_argument = arguments[0]
    if any(
        operator in requirement_argument
        for operator in ("==", "!=", ">=", "<=", "~=", "<", ">", "@", ";")
    ):
        raise ValueError(
            "install version cannot be combined with an already-versioned requirement"
        )

    return (f"{requirement_argument}=={normalized_version}",)


class Package:
    debian: ClassVar[type[DebianPackage]]
    python: ClassVar[type[PythonPackage]]
    meshagent: ClassVar[type[MeshagentPackage]]

    def __init__(self, *, name: str, include_workspace: bool = False):
        normalized_name = name.strip()
        if normalized_name == "":
            raise ValueError("package name must not be empty")

        self.name = normalized_name
        configured_base_path = _AGENT_BASE_PATH.get()
        self._base_path = (
            configured_base_path.resolve() if configured_base_path is not None else None
        )
        self._mounts: list[_Asset] = []
        self._files: list[_Asset] = []
        self._skills: list[_Asset] = []
        self._instructions: list[_Asset] = []
        self._env: dict[str, str] = {}
        self._image_build_steps: list[_ImageBuildStep] = []
        self._optimize_image = True
        self._include_workspace = include_workspace
        self._module_path: Path | None = None
        self._module_export_name: str | None = None
        self._module_export_is_factory = False

    def _all_assets(self) -> list[_Asset]:
        return [
            *self._mounts,
            *self._files,
            *self._skills,
            *self._instructions,
        ]

    def _instruction_paths(self) -> list[PurePosixPath]:
        return [asset.dest for asset in self._instructions]

    def _skill_paths(self) -> list[PurePosixPath]:
        return [asset.dest for asset in self._skills]

    def env(self, values: dict[str, str]) -> Package:
        self._env.update(_normalize_environment_values(values))
        return self

    def run(self, command: str) -> Package:
        self._image_build_steps.append(
            _RunBuildStep(
                command=_normalize_container_command(
                    command=command,
                    method_name="run",
                )
            )
        )
        return self

    def include_workspace(self, enabled: bool = True) -> Package:
        self._include_workspace = enabled
        return self

    def optimization(self, enabled: bool = True) -> Package:
        self._optimize_image = enabled
        return self

    def files(
        self,
        source: str,
        *,
        dest: Optional[str] = None,
        read_only: bool = False,
    ) -> Package:
        self._files.append(
            _normalize_asset(
                kind="file",
                source=source,
                dest=dest,
                read_only=read_only,
                base_path=self._base_path,
            )
        )
        return self

    def skills(
        self,
        source: str,
        *,
        dest: Optional[str] = None,
        read_only: bool = False,
    ) -> Package:
        self._skills.append(
            _normalize_asset(
                kind="skill",
                source=source,
                dest=dest,
                read_only=read_only,
                base_path=self._base_path,
            )
        )
        return self

    def instructions(
        self,
        source: str,
        *,
        dest: Optional[str] = None,
        read_only: bool = False,
    ) -> Package:
        self._instructions.append(
            _normalize_asset(
                kind="instruction",
                source=source,
                dest=dest,
                read_only=read_only,
                base_path=self._base_path,
            )
        )
        return self

    def mount(self, source: str, *, dest: str) -> Package:
        self._mounts.append(
            _normalize_asset(
                kind="mount",
                source=source,
                dest=dest,
                read_only=False,
                base_path=self._base_path,
            )
        )
        return self

    def _require_meshagent_package(self) -> MeshagentPackage:
        if not isinstance(self, MeshagentPackage):
            raise ValueError("deploy/run currently requires Package.meshagent(...)")
        return self

    def _bind_module_path(self, *, module_path: Path) -> None:
        self._module_path = module_path.resolve()

    def _bind_module_export(
        self,
        *,
        export_name: str,
        export_is_factory: bool,
    ) -> None:
        normalized_export_name = export_name.strip()
        if normalized_export_name == "":
            raise ValueError("export_name must not be empty")
        self._module_export_name = normalized_export_name
        self._module_export_is_factory = export_is_factory

    def _requires_custom_image(self) -> bool:
        return len(self._image_build_steps) > 0

    def _custom_image_tag(self) -> str:
        slug = _slugify_segment(self.name)
        return f"registry.meshagent.com/packages/{slug}:latest"

    def _custom_builder_name(self) -> str:
        slug = _slugify_segment(self.name)
        return f"package-{slug}"

    def _ordered_image_build_steps(self) -> tuple[_ImageBuildStep, ...]:
        return tuple(self._image_build_steps)

    def _custom_build_base_image(self) -> str:
        return _DEFAULT_PACKAGE_IMAGE

    def _runtime_base_image(self) -> str:
        return _DEFAULT_PACKAGE_IMAGE

    def _runtime_command(self, *, runtime_context: _RuntimeModuleContext) -> str:
        if self._module_export_name is not None:
            return f"python {shlex.quote(_PACKAGE_ENTRYPOINT_NAME)}"
        if not self._include_workspace:
            return f"python {shlex.quote(runtime_context.runtime_entry_relpath.as_posix())}"
        return runtime_context.runtime_command

    def _resolved_assets(
        self,
        *,
        root_path: str | Path | None = None,
    ) -> list[_ResolvedAsset]:
        resolved_root_path = _normalize_root_path(root_path)
        return [
            asset.resolve(root_path=resolved_root_path) for asset in self._all_assets()
        ]

    def _build_optimization_enabled(self) -> bool:
        return self._optimize_image

    def _resolve_assets_to_deploy_assets(
        self,
        *,
        assets: list[_ResolvedAsset],
        room_subpath_prefix: str,
    ) -> list[_DeployAsset]:
        groups: dict[PurePosixPath, list[_ResolvedAsset]] = {}
        for asset in assets:
            group_path = _group_mount_path(asset)
            grouped_assets = groups.setdefault(group_path, [])
            grouped_assets.append(asset)

        mount_groups: dict[PurePosixPath, _MountGroup] = {}
        for index, (mount_path, grouped_assets) in enumerate(
            sorted(groups.items(), key=lambda item: item[0].as_posix())
        ):
            dir_assets = [asset for asset in grouped_assets if asset.is_dir]
            if len(dir_assets) > 1:
                raise ValueError(
                    f"multiple directory assets cannot target the same destination: {mount_path.as_posix()}"
                )
            if len(dir_assets) == 1 and len(grouped_assets) > 1:
                raise ValueError(
                    f"directory and file assets cannot share the same destination: {mount_path.as_posix()}"
                )

            mount_groups[mount_path] = _MountGroup(
                mount_path=mount_path,
                room_subpath=f"{room_subpath_prefix}/{index}",
                read_only=all(asset.read_only for asset in grouped_assets),
            )

        resolved_assets: list[_DeployAsset] = []
        seen_destinations: set[str] = set()
        for asset in assets:
            destination_key = asset.dest.as_posix()
            if destination_key in seen_destinations:
                raise ValueError(f"duplicate destination path: {destination_key}")
            seen_destinations.add(destination_key)

            mount_group = mount_groups[_group_mount_path(asset)]
            if asset.is_file:
                room_path = f"{mount_group.room_subpath}/{asset.dest.name}"
            else:
                room_path = mount_group.room_subpath
            resolved_assets.append(
                _DeployAsset(
                    asset=asset,
                    room_path=room_path,
                    mount_group=mount_group,
                )
            )

        return resolved_assets

    def _resolve_deploy_assets(
        self,
        *,
        root_path: str | Path | None = None,
    ) -> list[_DeployAsset]:
        slug = _slugify_segment(self.name)
        return self._resolve_assets_to_deploy_assets(
            assets=self._resolved_assets(root_path=root_path),
            room_subpath_prefix=f".agents/{slug}/mounts",
        )

    def _resolved_runtime_module_path(
        self,
        *,
        module_path: str | Path | None,
    ) -> Path:
        if module_path is None:
            if self._module_path is None:
                raise ValueError(
                    "package deploy/run requires a source module path so the runtime module can be uploaded"
                )
            return self._module_path
        return Path(module_path).expanduser().resolve()

    def _runtime_entrypoint_asset(
        self,
        *,
        runtime_context: _RuntimeModuleContext,
        temp_dir: Path | None,
    ) -> _ResolvedAsset | None:
        if self._module_export_name is None:
            return None
        if temp_dir is None:
            raise ValueError(
                "runtime packaging requires a temp directory when an export entrypoint is bound"
            )

        temp_dir.mkdir(parents=True, exist_ok=True)
        entrypoint_source_path = temp_dir / _PACKAGE_ENTRYPOINT_NAME
        entrypoint_source_path.write_text(
            _runtime_entrypoint_source(
                runtime_context=runtime_context,
                export_name=self._module_export_name,
                export_is_factory=self._module_export_is_factory,
                include_workspace=self._include_workspace,
            ),
            encoding="utf-8",
        )
        return _ResolvedAsset(
            kind="file",
            source=entrypoint_source_path,
            dest=_PACKAGE_RUNTIME_DIR / _PACKAGE_ENTRYPOINT_NAME,
            read_only=True,
        )

    def _runtime_module_deploy_assets(
        self,
        *,
        module_path: Path,
        temp_dir: Path | None = None,
        root_path: str | Path | None = None,
    ) -> tuple[_RuntimeModuleContext, list[_DeployAsset]]:
        runtime_context = _runtime_module_context(module_path=module_path)
        workspace_root = runtime_context.import_root.resolve()
        if self._include_workspace:
            runtime_assets = _workspace_runtime_code_assets(
                runtime_context=runtime_context
            )
        else:
            runtime_assets = [
                _ResolvedAsset(
                    kind="file",
                    source=runtime_context.module_path,
                    dest=runtime_context.runtime_entry_dest,
                    read_only=True,
                )
            ]
        entrypoint_asset = self._runtime_entrypoint_asset(
            runtime_context=runtime_context,
            temp_dir=temp_dir,
        )
        if entrypoint_asset is not None:
            runtime_assets.append(entrypoint_asset)
        seen_destinations = {asset.dest.as_posix() for asset in runtime_assets}
        for asset in self._resolved_assets(root_path=root_path):
            try:
                relative_source = asset.source.relative_to(workspace_root)
            except ValueError as exc:
                raise ValueError(
                    "package asset source must be inside the packaged workspace for runtime execution: "
                    f"{asset.source}"
                ) from exc
            runtime_dest = _PACKAGE_RUNTIME_DIR / PurePosixPath(
                relative_source.as_posix()
            )
            if asset.is_dir:
                for local_path in sorted(asset.source.rglob("*")):
                    if not local_path.is_file():
                        continue
                    relative_file_path = PurePosixPath(
                        local_path.relative_to(asset.source).as_posix()
                    )
                    file_dest = runtime_dest / relative_file_path
                    destination_key = file_dest.as_posix()
                    if destination_key in seen_destinations:
                        continue
                    seen_destinations.add(destination_key)
                    runtime_assets.append(
                        _ResolvedAsset(
                            kind=asset.kind,
                            source=local_path,
                            dest=file_dest,
                            read_only=True,
                        )
                    )
                continue
            destination_key = runtime_dest.as_posix()
            if destination_key in seen_destinations:
                continue
            seen_destinations.add(destination_key)
            runtime_assets.append(
                _ResolvedAsset(
                    kind=asset.kind,
                    source=asset.source,
                    dest=runtime_dest,
                    read_only=True,
                )
            )

        slug = _slugify_segment(self.name)
        return runtime_context, self._resolve_assets_to_deploy_assets(
            assets=runtime_assets,
            room_subpath_prefix=f".agents/{slug}/runtime",
        )

    @staticmethod
    def _service_room_mounts(
        *,
        deploy_assets: list[_DeployAsset],
    ) -> list[RoomStorageMountSpec]:
        mount_groups: dict[str, _MountGroup] = {}
        for deploy_asset in deploy_assets:
            mount_groups[deploy_asset.mount_group.mount_path.as_posix()] = (
                deploy_asset.mount_group
            )
        return [
            RoomStorageMountSpec(
                path=mount_group.mount_path.as_posix(),
                subpath=mount_group.room_subpath,
                read_only=mount_group.read_only,
            )
            for mount_group in mount_groups.values()
        ]

    @staticmethod
    async def _upload_asset(
        *,
        room: RoomClient,
        deploy_asset: _DeployAsset,
    ) -> None:
        asset = deploy_asset.asset
        if asset.is_file:
            await room.storage.upload(
                path=deploy_asset.room_path,
                data=asset.source.read_bytes(),
                overwrite=True,
            )
            return

        uploaded_file = False
        for local_path in sorted(asset.source.rglob("*")):
            if not local_path.is_file():
                continue
            uploaded_file = True
            relative_path = local_path.relative_to(asset.source).as_posix()
            await room.storage.upload(
                path=f"{deploy_asset.room_path}/{relative_path}",
                data=local_path.read_bytes(),
                overwrite=True,
            )

        if not uploaded_file:
            raise ValueError(
                f"directory is empty and cannot be deployed: {asset.source}"
            )

    async def _upload_deploy_assets(
        self,
        *,
        room: RoomClient,
        deploy_assets: list[_DeployAsset],
        status_callback: _StatusCallback | None = None,
    ) -> None:
        total = len(deploy_assets)
        for index, deploy_asset in enumerate(deploy_assets, start=1):
            _emit_status(
                status_callback=status_callback,
                message=(
                    f"Uploading [{index}/{total}] "
                    f"{deploy_asset.asset.source} -> {deploy_asset.asset.dest.as_posix()}"
                ),
            )
            await self._upload_asset(
                room=room,
                deploy_asset=deploy_asset,
            )

    @staticmethod
    def _packaged_file_entries_for_assets(
        *,
        category: Literal["mount", "runtime"],
        deploy_assets: list[_DeployAsset],
    ) -> list[_PackagedFileEntry]:
        entries: list[_PackagedFileEntry] = []
        for deploy_asset in deploy_assets:
            asset = deploy_asset.asset
            if asset.is_file:
                entries.append(
                    _PackagedFileEntry(
                        category=category,
                        source=asset.source,
                        dest=asset.dest,
                    )
                )
                continue

            for local_path in sorted(asset.source.rglob("*")):
                if not local_path.is_file():
                    continue
                relative_path = PurePosixPath(
                    local_path.relative_to(asset.source).as_posix()
                )
                entries.append(
                    _PackagedFileEntry(
                        category=category,
                        source=local_path,
                        dest=asset.dest / relative_path,
                    )
                )

        return entries

    @classmethod
    def _packaged_file_entries(
        cls,
        *,
        deploy_assets: list[_DeployAsset],
        runtime_assets: list[_DeployAsset],
    ) -> list[_PackagedFileEntry]:
        entries = [
            *cls._packaged_file_entries_for_assets(
                category="mount",
                deploy_assets=deploy_assets,
            ),
            *cls._packaged_file_entries_for_assets(
                category="runtime",
                deploy_assets=runtime_assets,
            ),
        ]
        return sorted(
            entries,
            key=lambda entry: (
                entry.category,
                entry.dest.as_posix(),
                entry.source.as_posix(),
            ),
        )

    def _service_environment(self, *, room: str) -> list[EnvironmentVariable]:
        meshagent_package = self._require_meshagent_package()
        environment = [
            EnvironmentVariable(name=name, value=value)
            for name, value in self._env.items()
        ]
        environment.append(
            EnvironmentVariable(
                name="MESHAGENT_ROOM",
                value=room,
            )
        )
        environment.append(
            EnvironmentVariable(
                name="MESHAGENT_TOKEN",
                token=TokenValue(
                    identity=meshagent_package.name,
                    api=ApiScope.agent_default(),
                    role="agent",
                ),
            )
        )
        return environment

    def _runtime_environment(self, *, agent_jwt: str, room: str) -> dict[str, str]:
        environment = dict(self._env)
        environment["MESHAGENT_TOKEN"] = agent_jwt
        environment["MESHAGENT_ROOM"] = room
        return environment

    def _build_service_spec(
        self,
        *,
        room: str,
        deploy_assets: list[_DeployAsset],
        runtime_assets: list[_DeployAsset],
        runtime_context: _RuntimeModuleContext,
        container_image: str,
    ) -> ServiceSpec:
        meshagent_package = self._require_meshagent_package()
        service_room_mounts = self._service_room_mounts(
            deploy_assets=[*deploy_assets, *runtime_assets]
        )

        return ServiceSpec(
            kind="Service",
            version="v1",
            metadata=ServiceMetadata(
                name=self.name,
                annotations={ANNOTATION_SERVICE_ID: self.name},
            ),
            agents=[meshagent_package._build_agent_spec()],
            container=ContainerSpec(
                image=container_image,
                command=self._runtime_command(runtime_context=runtime_context),
                working_dir=_PACKAGE_RUNTIME_DIR.as_posix(),
                environment=self._service_environment(room=room),
                storage=ContainerMountSpec(
                    room=service_room_mounts or None,
                ),
            ),
        )


class DebianPackage(Package):
    def __init__(self, *, name: str):
        super().__init__(name=name, include_workspace=False)

    def apt_get_install(self, *packages: str) -> DebianPackage:
        self._image_build_steps.append(
            _AptGetInstallBuildStep(packages=_normalize_apt_install_packages(packages))
        )
        return self


Package.debian = DebianPackage


class PythonPackage(DebianPackage):
    def __init__(self, *, name: str):
        super().__init__(name=name)
        self._include_workspace = True

    def install(self, requirement: str, version: str | None = None) -> PythonPackage:
        self._image_build_steps.append(
            _PythonInstallBuildStep(
                requirements=_versioned_install_arguments(
                    requirement=requirement,
                    version=version,
                )
            )
        )
        return self


Package.python = PythonPackage


class MeshagentPackage(PythonPackage):
    def __init__(self, *, name: str):
        super().__init__(name=name)
        self.model = "gpt-5.4"
        self._channels: list[str] = []
        self._heartbeat: _Heartbeat | None = None
        self._shell_image: str | None = None
        self._shell_enabled = False
        self._advanced_shell_image: str | None = None
        self._advanced_shell_enabled = False
        self._web_fetch_enabled = False
        self._web_search_enabled = False
        self._image_gen_enabled = False
        self._image_gen_model: str | None = None
        self._apply_patch_enabled = False
        self._storage_enabled: bool | None = None
        self._storage_read_only = False
        self._table_read: list[str] = []
        self._table_write: list[str] = []
        self._dataset_namespace: list[str] | None = None
        self._time_enabled = False
        self._uuid_enabled = False
        self._memory_config: _MemoryToolConfig | None = None
        self._document_authoring_enabled = False
        self._discovery_enabled = False
        self._computer_use_config: _ComputerUseConfig | None = None
        self._mcp_enabled = False

    def _append_channel(self, channel: str) -> None:
        normalized = _canonical_channel(channel)
        if normalized not in self._channels:
            self._channels.append(normalized)

    def chat_channel(self) -> MeshagentPackage:
        self._append_channel("chat")
        return self

    def mail_channel(self, email: str) -> MeshagentPackage:
        self._append_channel(f"mail:{email}")
        return self

    def queue_channel(self, queue: str) -> MeshagentPackage:
        self._append_channel(f"queue:{queue}")
        return self

    def heartbeat(self, cron: str, prompt: str) -> MeshagentPackage:
        normalized_prompt = prompt.strip()
        if normalized_prompt == "":
            raise ValueError("heartbeat prompt must not be empty")
        _heartbeat_minutes(cron)
        self._heartbeat = _Heartbeat(cron=cron, prompt=normalized_prompt)
        return self

    def shell(
        self,
        *,
        enable: bool = True,
        image: Optional[str] = None,
    ) -> MeshagentPackage:
        if not enable:
            self._shell_enabled = False
            self._shell_image = None
            return self

        resolved_image = _normalize_optional_string(image, field_name="shell image")
        if resolved_image is None:
            resolved_image = self._shell_image or _DEFAULT_PACKAGE_IMAGE

        self._shell_enabled = True
        self._shell_image = resolved_image
        return self

    def advanced_shell(
        self,
        *,
        enable: bool = True,
        image: Optional[str] = None,
    ) -> MeshagentPackage:
        if not enable:
            self._advanced_shell_enabled = False
            self._advanced_shell_image = None
            return self

        resolved_image = _normalize_optional_string(
            image, field_name="advanced shell image"
        )
        if resolved_image is None:
            resolved_image = self._advanced_shell_image or self._shell_image
        self._advanced_shell_enabled = True
        self._advanced_shell_image = resolved_image
        return self

    def web_fetch(self, *, enable: bool = True) -> MeshagentPackage:
        self._web_fetch_enabled = enable
        return self

    def web_search(self, *, enable: bool = True) -> MeshagentPackage:
        self._web_search_enabled = enable
        return self

    def image_gen(
        self,
        *,
        enable: bool = True,
        model: str | None = None,
    ) -> MeshagentPackage:
        if not enable:
            self._image_gen_enabled = False
            self._image_gen_model = None
            return self

        self._image_gen_enabled = True
        self._image_gen_model = _normalize_optional_string(
            model,
            field_name="image generation model",
        )
        return self

    def apply_patch(self, *, enable: bool = True) -> MeshagentPackage:
        self._apply_patch_enabled = enable
        return self

    def storage(
        self,
        *,
        enable: bool = True,
        read_only: bool = False,
    ) -> MeshagentPackage:
        self._storage_enabled = enable
        self._storage_read_only = read_only if enable else False
        return self

    def table_read(
        self,
        *,
        enable: bool = True,
        tables: list[str] | tuple[str, ...],
        namespace: str | None = None,
    ) -> MeshagentPackage:
        if not enable:
            self._table_read = []
            return self

        self._table_read = _normalize_named_values(tables, field_name="tables")
        if namespace is not None or self._dataset_namespace is None:
            self._dataset_namespace = _parse_dataset_namespace(namespace)
        return self

    def table_write(
        self,
        *,
        enable: bool = True,
        tables: list[str] | tuple[str, ...],
        namespace: str | None = None,
    ) -> MeshagentPackage:
        if not enable:
            self._table_write = []
            return self

        self._table_write = _normalize_named_values(tables, field_name="tables")
        if namespace is not None or self._dataset_namespace is None:
            self._dataset_namespace = _parse_dataset_namespace(namespace)
        return self

    def time(self, *, enable: bool = True) -> MeshagentPackage:
        self._time_enabled = enable
        return self

    def uuid(self, *, enable: bool = True) -> MeshagentPackage:
        self._uuid_enabled = enable
        return self

    def memory(
        self,
        *,
        enable: bool = True,
        path: str = "graph",
        model: str | None = None,
    ) -> MeshagentPackage:
        if not enable:
            self._memory_config = None
            return self

        memory_name, namespace = _parse_memory_path(path)
        self._memory_config = _MemoryToolConfig(
            name=memory_name,
            namespace=namespace,
            model=_normalize_optional_string(model, field_name="memory model"),
        )
        return self

    def document_authoring(self, *, enable: bool = True) -> MeshagentPackage:
        self._document_authoring_enabled = enable
        return self

    def discovery(self, *, enable: bool = True) -> MeshagentPackage:
        self._discovery_enabled = enable
        return self

    def computer_use(
        self,
        *,
        enable: bool = True,
        starting_url: str | None = None,
        allow_goto_url: bool = False,
    ) -> MeshagentPackage:
        if not enable:
            self._computer_use_config = None
            return self

        self._computer_use_config = _ComputerUseConfig(
            starting_url=_normalize_optional_string(
                starting_url,
                field_name="starting url",
            ),
            allow_goto_url=allow_goto_url,
        )
        return self

    def mcp(self, *, enable: bool = True) -> MeshagentPackage:
        self._mcp_enabled = enable
        return self

    def _resolved_advanced_shell_image(self) -> str:
        return self._advanced_shell_image or self._shell_image or _DEFAULT_PACKAGE_IMAGE

    def _requires_storage_assets(
        self, *, feature_name: str, deploy_assets: list[_DeployAsset]
    ) -> None:
        if len(deploy_assets) == 0:
            raise ValueError(
                f"{feature_name} requires at least one package file, skill, instruction, or mount"
            )

    def _requires_openai_model(self, *, feature_name: str) -> None:
        if self.model.startswith("claude-"):
            raise ValueError(f"{feature_name} is only supported by openai models")

    def _validate_model_tool_compatibility(
        self, *, deploy_assets: list[_DeployAsset]
    ) -> None:
        if self._image_gen_enabled:
            self._requires_openai_model(feature_name="image generation")
        if self._apply_patch_enabled:
            self._requires_openai_model(feature_name="apply patch")
            self._requires_storage_assets(
                feature_name="apply patch",
                deploy_assets=deploy_assets,
            )
        if self._computer_use_config is not None:
            self._requires_openai_model(feature_name="computer use")

    def use_model(self, model: str) -> MeshagentPackage:
        normalized = model.strip()
        if normalized == "":
            raise ValueError("model must not be empty")
        self.model = normalized
        return self

    def _heartbeat_queue(self) -> str | None:
        if self._heartbeat is None:
            return None
        return f"{self.name}.heartbeat"

    def _process_channels(self) -> list[str]:
        channels = [*self._channels]
        heartbeat_queue = self._heartbeat_queue()
        if heartbeat_queue is not None:
            heartbeat_channel = f"queue:{heartbeat_queue}"
            if heartbeat_channel not in channels:
                channels.append(heartbeat_channel)
        return channels

    def _build_heartbeat_spec(self) -> HeartbeatSpec | None:
        if self._heartbeat is None:
            return None
        return HeartbeatSpec(
            queue=self._heartbeat_queue(),
            thread_id="/threads/heartbeats/{YYYY}/{MM}/{DD}/{HH}/{mm}/heartbeat.thread",
            prompt=[
                AgentTextContent(
                    type=AGENT_CONTENT_TYPE_TEXT,
                    text=self._heartbeat.prompt,
                )
            ],
            minutes=_heartbeat_minutes(self._heartbeat.cron),
        )

    def _build_agent_spec(self) -> AgentSpec:
        channels = self._process_channels()
        annotations: dict[str, str] | None = None
        if _has_chat_channel(channels=channels):
            annotations = {ANNOTATION_AGENT_TYPE: "ChatBot"}
        return AgentSpec(
            name=self.name,
            annotations=annotations,
            heartbeat=self._build_heartbeat_spec(),
        )

    def _custom_build_base_image(self) -> str:
        return _DEFAULT_MESHAGENT_PACKAGE_BUILD_IMAGE

    def _runtime_base_image(self) -> str:
        return _DEFAULT_MESHAGENT_PACKAGE_BUILD_IMAGE

    def _rules_storage_toolkit(
        self,
        *,
        deploy_assets: list[_DeployAsset],
    ) -> StorageToolkit | None:
        return self._storage_toolkit(
            deploy_assets=[
                deploy_asset
                for deploy_asset in deploy_assets
                if deploy_asset.asset.kind in {"instruction", "skill"}
            ],
            read_only=True,
        )

    @staticmethod
    async def _load_rules_from_storage(
        *,
        path: str,
        storage_toolkit: StorageToolkit,
        participant,
    ) -> list[str]:
        rules: list[str] = []
        normalized_path = _normalize_storage_rules_path(path)
        try:
            instructions_file = await storage_toolkit.read_file(path=normalized_path)
        except Exception as exc:
            logger.warning("unable to load instructions from %s: %s", path, exc)
            return rules

        rules_txt = instructions_file.data.decode()
        rules_config = RulesConfig.parse(rules_txt)
        if rules_config.rules is not None:
            rules.extend(rules_config.rules)

        if participant is not None:
            client = participant.get_attribute("client")
            if rules_config.client_rules is not None and client is not None:
                client_rules = rules_config.client_rules.get(client)
                if client_rules is not None:
                    rules.extend(client_rules)

        return rules

    def _shell_mount_spec(
        self,
        *,
        deploy_assets: list[_DeployAsset],
    ) -> ContainerMountSpec | None:
        room_mounts = self._service_room_mounts(deploy_assets=deploy_assets)
        if len(room_mounts) == 0:
            return None
        return ContainerMountSpec(room=room_mounts)

    def _storage_toolkit(
        self,
        *,
        deploy_assets: list[_DeployAsset],
        read_only: bool = False,
        room: RoomClient | None = None,
    ) -> StorageToolkit | None:
        if len(deploy_assets) == 0:
            return None

        if room is None:
            return StorageToolkit(
                read_only=read_only,
                mounts=[
                    StorageToolLocalMount(
                        path=deploy_asset.asset.dest.as_posix(),
                        local_path=str(deploy_asset.asset.source),
                        read_only=read_only or deploy_asset.asset.read_only,
                    )
                    for deploy_asset in deploy_assets
                ],
            )

        return StorageToolkit(
            read_only=read_only,
            mounts=[
                StorageToolRoomMount(
                    path=deploy_asset.asset.dest.as_posix(),
                    subpath=deploy_asset.room_path,
                    room=room,
                    read_only=read_only or deploy_asset.asset.read_only,
                )
                for deploy_asset in deploy_assets
            ],
        )

    def _llm_adapter(self):
        from meshagent.anthropic.openai_responses_stream_adapter import (
            AnthropicOpenAIResponsesStreamAdapter,
        )
        from meshagent.openai.tools.responses_adapter import OpenAIResponsesAdapter

        if self.model.startswith("claude-"):
            return AnthropicOpenAIResponsesStreamAdapter(model=self.model)
        return OpenAIResponsesAdapter(model=self.model)

    def serve(
        self,
        *,
        room: str | None = None,
        root_path: str | Path | None = None,
    ) -> None:
        asyncio.run(self._serve_async(room=room, root_path=root_path))

    async def _serve_async(
        self,
        *,
        room: str | None = None,
        root_path: str | Path | None = None,
    ) -> None:
        from meshagent.anthropic.mcp import AnthropicMessagesMCPToolkit
        from meshagent.anthropic.web_fetch import WebFetchTool as AnthropicWebFetchTool
        from meshagent.anthropic.web_search import (
            WebSearchTool as AnthropicWebSearchTool,
        )
        from meshagent.computers.agent import ComputerToolkit
        from meshagent.openai.tools.responses_adapter import (
            ApplyPatchTool,
            ImageGenerationTool,
            OpenAIResponsesAdapter,
            OpenAIResponsesMCPToolkit,
            ShellTool,
            WebSearchTool,
        )
        from meshagent.tools.container_shell import (
            ContainerShellTool,
            ContainerToolkit,
            ProcessShellTool,
        )
        from meshagent.tools.web_toolkit import WebFetchTool
        from meshagent.tools.dataset import make_dataset_toolkit
        from meshagent.tools.datetime import DatetimeToolkit
        from meshagent.tools.discovery import DiscoveryToolkit
        from meshagent.tools.document_tools import (
            DocumentAuthoringToolkit,
            DocumentTypeAuthoringToolkit,
        )
        from meshagent.tools.memories import MemoriesToolkit
        from meshagent.tools.uuid import UUIDToolkit
        from meshagent.agents.widget_schema import widget_schema

        resolved_room = room or os.getenv("MESHAGENT_ROOM")
        if resolved_room is None or resolved_room.strip() == "":
            raise ValueError(
                "MeshagentPackage.serve() requires a room or MESHAGENT_ROOM"
            )

        token = os.getenv("MESHAGENT_TOKEN")
        if token is None or token.strip() == "":
            raise ValueError(
                "MeshagentPackage.serve() requires MESHAGENT_TOKEN to be set"
            )
        deploy_assets = self._resolve_deploy_assets(root_path=root_path)
        self._validate_model_tool_compatibility(deploy_assets=deploy_assets)
        shell_mount_spec = self._shell_mount_spec(deploy_assets=deploy_assets)
        process_llm_adapter = self._llm_adapter()
        channel_llm_adapter = self._llm_adapter()
        rules_storage_toolkit = self._rules_storage_toolkit(deploy_assets=deploy_assets)
        self_package = self

        class _PackageSupervisor(AgentSupervisor):
            def __init__(self, *, room: RoomClient) -> None:
                super().__init__()
                self._room = room

            def create_thread_process(self, thread_id: str) -> LLMAgentProcess:
                thread_storage = MeshDocumentThreadStorage(
                    room=self._room,
                    path=thread_id,
                )

                async def _turn_instructions_provider(participant) -> str | None:
                    rules: list[str] = []
                    skill_paths = self_package._skill_paths()
                    if len(skill_paths) > 0:
                        assert rules_storage_toolkit is not None
                        rules.append(
                            "You have access to to following skills which follow the agentskills spec:"
                        )
                        rules.append(
                            await to_prompt(
                                [Path(path.as_posix()) for path in skill_paths],
                                storage_toolkit=rules_storage_toolkit,
                            )
                        )
                        rules.append(
                            "Use the shell or storage tool to find out more about skills and execute them when they are required"
                        )
                    for instruction_path in self_package._instruction_paths():
                        assert rules_storage_toolkit is not None
                        rules.extend(
                            await self_package._load_rules_from_storage(
                                path=instruction_path.as_posix(),
                                storage_toolkit=rules_storage_toolkit,
                                participant=participant,
                            )
                        )
                    rules.append(
                        "based on the previous transcript, take your turn and respond"
                    )
                    return "\n".join(rules) if len(rules) > 0 else None

                async def _turn_toolkits_builder(participant, model, turns):
                    del participant
                    toolkits: list[Toolkit] = []
                    storage_toolkit = (
                        self_package._storage_toolkit(
                            deploy_assets=deploy_assets,
                            read_only=self_package._storage_read_only,
                        )
                        if self_package._storage_enabled is not False
                        else None
                    )
                    if storage_toolkit is not None:
                        toolkits.append(storage_toolkit)
                    if self_package._shell_enabled:
                        if model.startswith("gpt-"):
                            toolkits.append(
                                Toolkit(
                                    name="shell",
                                    tools=[
                                        ShellTool(
                                            room=self._room,
                                            name="shell",
                                            image=self_package._shell_image,
                                            mounts=shell_mount_spec,
                                        )
                                    ],
                                )
                            )
                        else:
                            shell_tool = (
                                ProcessShellTool(name="shell")
                                if self_package._shell_image is None
                                else ContainerShellTool(
                                    room=self._room,
                                    name="shell",
                                    image=self_package._shell_image,
                                    mounts=shell_mount_spec,
                                )
                            )
                            toolkits.append(Toolkit(name="shell", tools=[shell_tool]))
                    if self_package._web_fetch_enabled:
                        toolkits.append(
                            Toolkit(
                                name="web_fetch",
                                tools=[
                                    AnthropicWebFetchTool()
                                    if model.startswith("claude-")
                                    else WebFetchTool()
                                ],
                            )
                        )
                    if self_package._web_search_enabled:
                        toolkits.append(
                            Toolkit(
                                name="web_search",
                                tools=[
                                    AnthropicWebSearchTool()
                                    if model.startswith("claude-")
                                    else WebSearchTool()
                                ],
                            )
                        )
                    if self_package._image_gen_enabled:
                        toolkits.append(
                            Toolkit(
                                name="image_generation",
                                tools=[
                                    ImageGenerationTool(
                                        model=self_package._image_gen_model,
                                    )
                                ],
                            )
                        )
                    if self_package._apply_patch_enabled:
                        apply_patch_storage = self_package._storage_toolkit(
                            deploy_assets=deploy_assets,
                        )
                        assert apply_patch_storage is not None
                        toolkits.append(
                            Toolkit(
                                name="apply_patch",
                                tools=[ApplyPatchTool(storage=apply_patch_storage)],
                            )
                        )
                    if self_package._advanced_shell_enabled:
                        toolkits.append(
                            ContainerToolkit(
                                room=self._room,
                                default_image=self_package._resolved_advanced_shell_image(),
                                mounts=shell_mount_spec,
                            )
                        )
                    if self_package._mcp_enabled:
                        toolkits.append(
                            AnthropicMessagesMCPToolkit()
                            if model.startswith("claude-")
                            else OpenAIResponsesMCPToolkit()
                        )
                    if len(self_package._table_read) > 0:
                        toolkits.append(
                            await make_dataset_toolkit(
                                room=self._room,
                                tables=self_package._table_read,
                                read_only=True,
                                namespace=self_package._dataset_namespace,
                            )
                        )
                    if self_package._time_enabled:
                        toolkits.append(DatetimeToolkit())
                    if self_package._uuid_enabled:
                        toolkits.append(UUIDToolkit())
                    if self_package._memory_config is not None:
                        toolkits.append(
                            MemoriesToolkit(
                                room=self._room,
                                memory_name=self_package._memory_config.name,
                                namespace=self_package._memory_config.namespace,
                                llm_model=self_package._memory_config.model,
                            )
                        )
                    if len(self_package._table_write) > 0:
                        toolkits.append(
                            await make_dataset_toolkit(
                                room=self._room,
                                tables=self_package._table_write,
                                read_only=False,
                                namespace=self_package._dataset_namespace,
                            )
                        )
                    if self_package._document_authoring_enabled:
                        toolkits.append(DocumentAuthoringToolkit(room=self._room))
                        toolkits.append(
                            DocumentTypeAuthoringToolkit(
                                room=self._room,
                                schema=widget_schema,
                                document_type="widget",
                            )
                        )
                    if self_package._discovery_enabled:
                        toolkits.append(DiscoveryToolkit(room=self._room))
                    if self_package._computer_use_config is not None:
                        images_dataset = ImagesDataset(self._room.datasets)
                        computer_toolkit: ComputerToolkit | None = None

                        async def render_screen(image_bytes: bytes) -> None:
                            created_by = self._room.local_participant.get_attribute(
                                "name"
                            )
                            if not isinstance(created_by, str):
                                created_by = ""

                            try:
                                saved_image = await images_dataset.save(
                                    data=image_bytes,
                                    mime_type="image/png",
                                    created_by=created_by,
                                    annotations={
                                        "source": "computer_toolkit",
                                        "thread_path": thread_id,
                                    },
                                )
                            except Exception as ex:
                                logger.error(
                                    "failed to persist computer screenshot",
                                    exc_info=ex,
                                )
                                return

                            width: int | float | None = None
                            height: int | float | None = None
                            if computer_toolkit is not None:
                                width, height = computer_toolkit.computer.dimensions

                            thread_storage.push_message(
                                message=AgentThreadEvent(
                                    type=AGENT_EVENT_THREAD_EVENT,
                                    thread_id=thread_id,
                                    event={
                                        "type": "computer.screenshot",
                                        "uri": f"dataset://{ImagesDataset.TABLE_NAME}?id={saved_image.id}",
                                        "mime_type": saved_image.mime_type,
                                        "created_at": saved_image.created_at,
                                        "created_by": saved_image.created_by,
                                        "width": width,
                                        "height": height,
                                        "status": "completed",
                                    },
                                )
                            )

                        computer_toolkit = ComputerToolkit(
                            room=self._room,
                            render_screen=render_screen,
                            starting_url=self_package._computer_use_config.starting_url,
                            include_goto_tool=self_package._computer_use_config.allow_goto_url,
                        )
                        toolkits.append(computer_toolkit)
                    toolkits.append(thread_storage.make_toolkit())
                    return toolkits

                def publish_thread_status(message) -> None:
                    self.send(Message(data=message, source=process))

                process = LLMAgentProcess(
                    thread_id=thread_id,
                    participant=self._room.local_participant,
                    llm_adapter=process_llm_adapter,
                    thread_storage=thread_storage,
                    thread_status_publisher=AgentMessageThreadStatusPublisher(
                        thread_id=thread_id,
                        publish=publish_thread_status,
                    ),
                    turn_instructions_provider=_turn_instructions_provider,
                    turn_toolkits_builder=_turn_toolkits_builder,
                )
                process.register_content_scheme(_room_content_scheme(room=self._room))
                return process

        async with RoomClient(
            protocol_factory=WebSocketClientProtocol(
                url=websocket_room_url(room_name=resolved_room),
                token=token,
            ).create_factory()
        ) as room_client:
            if self._image_gen_enabled and isinstance(
                process_llm_adapter, OpenAIResponsesAdapter
            ):
                process_llm_adapter.set_images_dataset(
                    ImagesDataset(room_client.datasets)
                )
            channels: list[
                MessagingChatChannel | MailChannel | QueueChannel | ToolkitChannel
            ] = []
            for channel_spec in self._process_channels():
                lowered = channel_spec.casefold()
                if lowered == "chat":
                    channels.append(
                        MessagingChatChannel(
                            room=room_client,
                            llm_adapter=channel_llm_adapter,
                        )
                    )
                    continue
                if lowered.startswith("mail:"):
                    email_address = channel_spec[5:].strip()
                    channels.append(
                        MailChannel(
                            room=room_client,
                            queue_name=email_address,
                            email_address=email_address,
                            llm_adapter=channel_llm_adapter,
                        )
                    )
                    continue
                if lowered.startswith("queue:"):
                    queue_name = channel_spec[6:].strip()
                    channels.append(
                        QueueChannel(
                            room=room_client,
                            queue_name=queue_name,
                            llm_adapter=channel_llm_adapter,
                        )
                    )
                    continue
                if lowered.startswith("toolkit:"):
                    toolkit_name = channel_spec[8:].strip()
                    channels.append(
                        ToolkitChannel(
                            room=room_client,
                            toolkit_name=toolkit_name,
                        )
                    )
                    continue
                raise ValueError(f"unsupported package channel: {channel_spec}")

            hosted_toolkits: list[_RemoteToolkitWrapper] = []
            supervisor = _PackageSupervisor(room=room_client)
            try:
                for channel in channels:
                    supervisor.add_channel(channel)
                await supervisor.start()
                for channel in channels:
                    if channel.state != "started":
                        continue
                    for toolkit in channel.get_exposed_toolkits():
                        hosted_toolkits.append(
                            await start_hosted_toolkit(
                                room=room_client, toolkit=toolkit
                            )
                        )
                await room_client.protocol.wait_for_close()
            finally:
                for hosted_toolkit in reversed(hosted_toolkits):
                    await hosted_toolkit.stop()
                if supervisor.state == "started":
                    await supervisor.stop()


Package.meshagent = MeshagentPackage


def _package_dockerfile_base_lines(*, base_image: str) -> list[str]:
    if base_image.startswith("meshagent/") and base_image.endswith(":default"):
        repository = base_image.removeprefix("meshagent/").removesuffix(":default")
        default_tag = _meshagent_default_image_tag_for_repository(repository=repository)
        return [
            f"ARG MESHAGENT_IMAGE_PREFIX={_DEFAULT_MESHAGENT_IMAGE_PREFIX}",
            f"FROM ${{MESHAGENT_IMAGE_PREFIX}}{repository}:{default_tag}",
        ]

    return [f"FROM {base_image}"]


def _package_dockerfile_text(*, package: Package, base_image: str) -> str:
    lines = _package_dockerfile_base_lines(base_image=base_image)
    build_steps = package._ordered_image_build_steps()
    first_apt_step_index: int | None = None
    last_apt_step_index: int | None = None
    for index, build_step in enumerate(build_steps):
        if isinstance(build_step, _AptGetInstallBuildStep):
            if first_apt_step_index is None:
                first_apt_step_index = index
            last_apt_step_index = index

    for index, build_step in enumerate(build_steps):
        if index == first_apt_step_index:
            lines.append("RUN apt-get update")

        if isinstance(build_step, _RunBuildStep):
            lines.append(f"RUN {build_step.command}")
        elif isinstance(build_step, _AptGetInstallBuildStep):
            lines.append(
                f"RUN {shlex.join(('apt-get', 'install', '-y', *build_step.packages))}"
            )
        elif isinstance(build_step, _PythonInstallBuildStep):
            lines.append(
                f"RUN {shlex.join(('uv', 'pip', 'install', *build_step.requirements))}"
            )
        else:
            raise AssertionError(f"unsupported image build step: {type(build_step)}")

        if index == last_apt_step_index:
            lines.append("RUN rm -rf /var/lib/apt/lists/*")

    return "\n".join(lines) + "\n"


async def _build_package_image(
    *,
    package: Package,
    resolved_project_id: str | None,
    resolved_room: str,
    builder_name: str | None = None,
    status_callback: _StatusCallback | None = None,
) -> str:
    from meshagent.cli import image as image_module

    image_tag = package._custom_image_tag()
    parsed_tag = image_module._parse_build_tag(image_tag)
    resolved_builder_name = (
        builder_name if builder_name is not None else package._custom_builder_name()
    )
    _emit_status(
        status_callback=status_callback,
        message=f"Building package image {image_tag} with builder {resolved_builder_name}",
    )
    with tempfile.TemporaryDirectory(prefix="meshagent-package-") as temp_dir:
        context_dir = Path(temp_dir)
        (context_dir / "Dockerfile").write_text(
            _package_dockerfile_text(
                package=package,
                base_image=package._custom_build_base_image(),
            ),
            encoding="utf-8",
        )
        await image_module._run_image_build_stage(
            resolved_project_id=resolved_project_id,
            resolved_room=resolved_room,
            parsed_tag=parsed_tag,
            context_path=None,
            dockerfile_path=None,
            pack=str(context_dir),
            arch=image_module.default_pack_architecture(),
            builder_name=resolved_builder_name,
            private=False,
            optimize=package._build_optimization_enabled(),
            cred=[],
        )
    _emit_status(
        status_callback=status_callback,
        message=f"Built package image {image_tag}",
    )
    return image_tag


async def deploy_package(
    *,
    package: Package,
    room: str,
    project_id: str | None = None,
    builder_name: str | None = None,
    status_callback: _StatusCallback | None = None,
) -> str:
    from meshagent.cli.helper import get_client, resolve_project_id, resolve_room

    resolved_room = resolve_room(room)
    if resolved_room is None:
        raise ValueError("room is required")

    resolved_project_id = await resolve_project_id(project_id=project_id)
    container_image = package._runtime_base_image()
    if package._requires_custom_image():
        container_image = await _build_package_image(
            package=package,
            resolved_project_id=resolved_project_id,
            resolved_room=resolved_room,
            builder_name=builder_name,
            status_callback=status_callback,
        )

    account_client = await get_client()
    try:
        _emit_status(
            status_callback=status_callback,
            message=f"Connecting to room {resolved_room}",
        )
        connection = await account_client.connect_room(
            project_id=resolved_project_id,
            room=resolved_room,
        )
        room_client = RoomClient(
            protocol_factory=WebSocketClientProtocol(
                url=connection.room_url,
                token=connection.jwt,
            ).create_factory()
        )
        _bind_room_status_callback(
            room_client=room_client,
            status_callback=status_callback,
        )
        module_path = package._resolved_runtime_module_path(module_path=None)
        deploy_assets = package._resolve_deploy_assets()
        with tempfile.TemporaryDirectory(
            prefix="meshagent-package-runtime-"
        ) as temp_dir:
            runtime_context, runtime_assets = package._runtime_module_deploy_assets(
                module_path=module_path,
                temp_dir=Path(temp_dir),
            )
            spec = package._build_service_spec(
                room=resolved_room,
                deploy_assets=deploy_assets,
                runtime_assets=runtime_assets,
                runtime_context=runtime_context,
                container_image=container_image,
            )
            async with room_client:
                _emit_status(
                    status_callback=status_callback,
                    message=(
                        f"Uploading {len(deploy_assets) + len(runtime_assets)} "
                        f"packaged assets to room {resolved_room}"
                    ),
                )
                await package._upload_deploy_assets(
                    room=room_client,
                    deploy_assets=[*deploy_assets, *runtime_assets],
                    status_callback=status_callback,
                )

        try:
            _emit_status(
                status_callback=status_callback,
                message=f"Creating service {package.name}",
            )
            return await account_client.create_room_service(
                project_id=resolved_project_id,
                service=spec,
                room_name=resolved_room,
            )
        except ConflictError:
            pass

        target_service_annotation_id = _annotation_service_id(spec.metadata.annotations)
        service_id: str | None = None
        services = await account_client.list_room_services(
            project_id=resolved_project_id,
            room_name=resolved_room,
        )
        for service in services:
            if (
                target_service_annotation_id is not None
                and _annotation_service_id(service.metadata.annotations)
                == target_service_annotation_id
            ):
                service_id = service.id
                break
        if service_id is None:
            for service in services:
                if service.metadata.name == spec.metadata.name:
                    service_id = service.id
                    break

        if service_id is None:
            unresolved_service_id = target_service_annotation_id or package.name
            raise ValueError(f"service id already in use: {unresolved_service_id}")

        spec.id = service_id
        _emit_status(
            status_callback=status_callback,
            message=f"Updating existing service {service_id}",
        )
        await account_client.update_room_service(
            project_id=resolved_project_id,
            service_id=service_id,
            service=spec,
            room_name=resolved_room,
        )
        return service_id
    finally:
        await account_client.close()


async def run_package(
    *,
    package: Package,
    room: str,
    project_id: str | None = None,
    builder_name: str | None = None,
    status_callback: _StatusCallback | None = None,
) -> str:
    from meshagent.cli.helper import (
        get_client,
        resolve_key,
        resolve_project_id,
        resolve_room,
    )

    resolved_room = resolve_room(room)
    if resolved_room is None:
        raise ValueError("room is required")

    resolved_project_id = await resolve_project_id(project_id=project_id)
    container_image = package._runtime_base_image()
    if package._requires_custom_image():
        container_image = await _build_package_image(
            package=package,
            resolved_project_id=resolved_project_id,
            resolved_room=resolved_room,
            builder_name=builder_name,
            status_callback=status_callback,
        )

    meshagent_package = package._require_meshagent_package()
    token = ParticipantToken(name=meshagent_package.name)
    token.add_api_grant(ApiScope.agent_default())
    token.add_role_grant(role="agent")
    token.add_room_grant(resolved_room)
    agent_jwt = token.to_jwt(
        api_key=await resolve_key(project_id=resolved_project_id, key=None)
    )
    account_client = await get_client()
    try:
        _emit_status(
            status_callback=status_callback,
            message=f"Connecting to room {resolved_room}",
        )
        connection = await account_client.connect_room(
            project_id=resolved_project_id,
            room=resolved_room,
        )
        room_client = RoomClient(
            protocol_factory=WebSocketClientProtocol(
                url=connection.room_url,
                token=connection.jwt,
            ).create_factory()
        )
        _bind_room_status_callback(
            room_client=room_client,
            status_callback=status_callback,
        )
        module_path = package._resolved_runtime_module_path(module_path=None)
        deploy_assets = package._resolve_deploy_assets()
        with tempfile.TemporaryDirectory(
            prefix="meshagent-package-runtime-"
        ) as temp_dir:
            runtime_context, runtime_assets = package._runtime_module_deploy_assets(
                module_path=module_path,
                temp_dir=Path(temp_dir),
            )
            spec = package._build_service_spec(
                room=resolved_room,
                deploy_assets=deploy_assets,
                runtime_assets=runtime_assets,
                runtime_context=runtime_context,
                container_image=container_image,
            )

            async with room_client:
                _emit_status(
                    status_callback=status_callback,
                    message=(
                        f"Uploading {len(deploy_assets) + len(runtime_assets)} "
                        f"packaged assets to room {resolved_room}"
                    ),
                )
                await package._upload_deploy_assets(
                    room=room_client,
                    deploy_assets=[*deploy_assets, *runtime_assets],
                    status_callback=status_callback,
                )
                _emit_status(
                    status_callback=status_callback,
                    message=f"Starting container from image {spec.container.image}",
                )
                return await room_client.containers.run(
                    image=spec.container.image,
                    command=spec.container.command,
                    working_dir=spec.container.working_dir,
                    env=package._runtime_environment(
                        agent_jwt=agent_jwt,
                        room=resolved_room,
                    ),
                    participant_name=meshagent_package.name,
                    role="agent",
                    mounts=spec.container.storage,
                )
    finally:
        await account_client.close()
