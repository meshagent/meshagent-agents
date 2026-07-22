import sys
from pathlib import Path

from meshagent.agents.package_manifest import (
    MANIFEST_PREFIX,
    PackageManifestError,
    PackageManifestSuccess,
    main,
)


def _manifest_payload(output: str) -> str:
    manifest_line = output.splitlines()[-1]
    assert manifest_line.startswith(MANIFEST_PREFIX)
    return manifest_line.removeprefix(MANIFEST_PREFIX)


def test_package_manifest_emitter_preserves_module_output_and_typed_package(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text(
        "from meshagent.agents import Package\n"
        "print('module output')\n"
        "main = Package.meshagent(name='assistant').web_fetch()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["package_manifest", str(module_path), "main"],
    )

    main()

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith("module output\n")
    result = PackageManifestSuccess.model_validate_json(_manifest_payload(output.out))
    assert result.package.kind == "meshagent"
    assert result.package.name == "assistant"
    assert result.package.module_path == module_path.resolve()
    assert result.package.module_export_name == "main"
    assert result.package.module_export_is_factory is False
    assert result.package.web_fetch_enabled is True


def test_package_manifest_emitter_returns_typed_export_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["package_manifest", str(module_path), "main"],
    )

    main()

    output = capsys.readouterr()
    assert output.err == ""
    result = PackageManifestError.model_validate_json(_manifest_payload(output.out))
    assert result.message == f"{module_path.resolve()} does not define main"
