from pathlib import Path
import importlib.util
import sys
import types

import pytest


class _AsyncTextReader:
    def __init__(self, path: Path, *, encoding: str):
        self._path = path
        self._encoding = encoding

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type
        del exc
        del tb

    async def read(self) -> str:
        return self._path.read_text(encoding=self._encoding)


def _install_aiofiles_stub() -> None:
    aiofiles_module = types.ModuleType("aiofiles")
    aiofiles_ospath_module = types.ModuleType("aiofiles.ospath")

    def open_file(path, mode="r", encoding=None):
        if mode != "r":
            raise NotImplementedError("test aiofiles stub only supports read mode")
        return _AsyncTextReader(Path(path), encoding=encoding or "utf-8")

    async def exists(path) -> bool:
        return Path(path).exists()

    aiofiles_module.open = open_file
    aiofiles_ospath_module.exists = exists
    aiofiles_module.ospath = aiofiles_ospath_module
    sys.modules["aiofiles"] = aiofiles_module
    sys.modules["aiofiles.ospath"] = aiofiles_ospath_module


def _load_skills_module():
    module_name = "meshagent.agents.skills"
    module_path = (
        Path(__file__).resolve().parents[1] / "meshagent" / "agents" / "skills.py"
    )
    _install_aiofiles_stub()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


skills = _load_skills_module()


@pytest.mark.asyncio
async def test_to_prompt_includes_missing_skill_and_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing_skill_dir = tmp_path / "eli-authentication"
    missing_skill_dir.mkdir()

    caplog.set_level("WARNING")

    prompt = await skills.to_prompt([missing_skill_dir])

    assert "<available_skills>" in prompt
    assert "<name>" in prompt
    assert "eli-authentication" in prompt
    assert "missing SKILL.md" in prompt
    assert str(missing_skill_dir) in prompt
    assert "SKILL.md not found in" in caplog.text
    assert str(missing_skill_dir) in caplog.text


@pytest.mark.asyncio
async def test_to_prompt_raises_for_missing_skill_when_missing_ok_is_false(
    tmp_path: Path,
) -> None:
    missing_skill_dir = tmp_path / "eli-authentication"
    missing_skill_dir.mkdir()

    with pytest.raises(skills.ParseError, match="SKILL.md not found in"):
        await skills.to_prompt([missing_skill_dir], missing_ok=False)


@pytest.mark.asyncio
async def test_to_prompt_keeps_valid_skill_entries_unchanged(tmp_path: Path) -> None:
    skill_dir = tmp_path / "pdf-reader"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: pdf-reader\ndescription: Read PDF files\n---\nBody\n",
        encoding="utf-8",
    )

    prompt = await skills.to_prompt([skill_dir])

    assert "<name>\npdf-reader\n</name>" in prompt
    assert "<description>\nRead PDF files\n</description>" in prompt
    assert str(skill_md) in prompt


@pytest.mark.asyncio
async def test_to_prompt_can_retarget_skill_locations(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "pdf-reader"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: pdf-reader\ndescription: Read PDF files\n---\nBody\n",
        encoding="utf-8",
    )

    prompt = await skills.to_prompt(
        [skills_root],
        location_mapper=lambda location: location.as_posix().replace(
            skills_root.as_posix(),
            "/skills",
            1,
        ),
    )

    assert "<name>\npdf-reader\n</name>" in prompt
    assert "/skills/pdf-reader/SKILL.md" in prompt
    assert str(skill_md) not in prompt


@pytest.mark.asyncio
async def test_to_prompt_includes_unloadable_skill_error_when_frontmatter_is_invalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: broken-skill\ndescription: [unterminated\n---\nBody\n",
        encoding="utf-8",
    )

    caplog.set_level("WARNING")

    prompt = await skills.to_prompt([skill_dir])

    assert "<name>\nbroken-skill\n</name>" in prompt
    assert "Configured skill could not be loaded:" in prompt
    assert "Invalid YAML in frontmatter:" in prompt
    assert str(skill_md) in prompt
    assert f"unable to load skill from {skill_md}" in caplog.text


@pytest.mark.asyncio
async def test_discover_skill_folders_returns_immediate_skill_children(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    valid_skill = skills_root / "pdf-reader"
    valid_skill.mkdir()
    (valid_skill / "SKILL.md").write_text(
        "---\nname: pdf-reader\ndescription: Read PDF files\n---\nBody\n",
        encoding="utf-8",
    )
    empty_child = skills_root / "notes"
    empty_child.mkdir()
    (skills_root / "README.md").write_text("hello", encoding="utf-8")

    discovered = await skills.discover_skill_folders(skills_root)

    assert discovered == [valid_skill]


@pytest.mark.asyncio
async def test_to_prompt_discovers_subskills_when_parent_has_no_skill_md(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    first_skill = skills_root / "pdf-reader"
    first_skill.mkdir()
    first_skill_md = first_skill / "SKILL.md"
    first_skill_md.write_text(
        "---\nname: pdf-reader\ndescription: Read PDF files\n---\nBody\n",
        encoding="utf-8",
    )

    second_skill = skills_root / "csv-reader"
    second_skill.mkdir()
    second_skill_md = second_skill / "skill.md"
    second_skill_md.write_text(
        "---\nname: csv-reader\ndescription: Read CSV files\n---\nBody\n",
        encoding="utf-8",
    )

    prompt = await skills.to_prompt([skills_root])
    resolved_second_skill_md = await skills.find_skill_md(second_skill)

    assert "<name>\ncsv-reader\n</name>" in prompt
    assert "<name>\npdf-reader\n</name>" in prompt
    assert str(first_skill_md) in prompt
    assert resolved_second_skill_md is not None
    assert str(resolved_second_skill_md) in prompt


@pytest.mark.asyncio
async def test_to_prompt_keeps_discovered_valid_skills_when_one_subskill_is_invalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    valid_skill = skills_root / "pdf-reader"
    valid_skill.mkdir()
    valid_skill_md = valid_skill / "SKILL.md"
    valid_skill_md.write_text(
        "---\nname: pdf-reader\ndescription: Read PDF files\n---\nBody\n",
        encoding="utf-8",
    )

    invalid_skill = skills_root / "broken-skill"
    invalid_skill.mkdir()
    invalid_skill_md = invalid_skill / "SKILL.md"
    invalid_skill_md.write_text(
        "---\nname: broken-skill\ndescription: [unterminated\n---\nBody\n",
        encoding="utf-8",
    )

    caplog.set_level("WARNING")

    prompt = await skills.to_prompt([skills_root])

    assert "<name>\npdf-reader\n</name>" in prompt
    assert str(valid_skill_md) in prompt
    assert "<name>\nbroken-skill\n</name>" in prompt
    assert "Configured skill could not be loaded:" in prompt
    assert "Invalid YAML in frontmatter:" in prompt
    assert str(invalid_skill_md) in prompt
    assert f"unable to load skill from {invalid_skill_md}" in caplog.text
