from __future__ import annotations

import argparse
from typing import Literal

from pydantic import BaseModel

from meshagent.agents.package import PackageManifest, load_package


MANIFEST_PREFIX = "MESHAGENT_PACKAGE_MANIFEST="


class PackageManifestSuccess(BaseModel):
    type: Literal["success"] = "success"
    package: PackageManifest


class PackageManifestError(BaseModel):
    type: Literal["error"] = "error"
    message: str


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("module")
    parser.add_argument("export_name")
    args = parser.parse_args()

    try:
        package = load_package(module_path=args.module, export_name=args.export_name)
        result: PackageManifestSuccess | PackageManifestError = PackageManifestSuccess(
            package=package.to_manifest()
        )
    except Exception as exc:
        result = PackageManifestError(message=str(exc))
    print(f"{MANIFEST_PREFIX}{result.model_dump_json()}")


if __name__ == "__main__":
    main()
