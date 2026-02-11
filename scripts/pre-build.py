from copy import deepcopy
from pathlib import Path
from typing import Optional

import tomli_w
import tomllib


def _get_version(data) -> Optional[str]:
    if "project" in data:
        project = data["project"]
        if "version" in project:
            return project["version"]
    if "tool" in data:
        tool = data["tool"]
        if "poetry" in tool:
            poetry = tool["poetry"]
            if "version" in poetry:
                return poetry["version"]


def _convert_project_dependencies(data: dict, pyproject: Path) -> dict:
    project = data["project"]
    if "dependencies" not in project:
        return data

    dependencies: list[str] = project["dependencies"]
    new_data = deepcopy(data)
    new_deps = []

    for dep in dependencies:
        if "@ file:///" not in dep:
            new_deps.append(dep)
            continue

        package, path = dep.split(" @ file:///")
        other_pyproject = Path(path).resolve() / "pyproject.toml"
        with open(other_pyproject, "rb") as f:
            other_data = tomllib.load(f)

        new_version = _get_version(other_data)
        if new_version is None:
            new_deps.append(dep)
            continue

        next_major = int(new_version.split(".")[0]) + 1
        package_version_caret = f"{package} (>={new_version},<{next_major}.0.0)"
        new_deps.append(package_version_caret)

    new_data["project"]["dependencies"] = new_deps

    return new_data


def _convert_poetry_dependencies(data: dict, pyproject: Path) -> dict:
    poetry = data["tool"]["poetry"]
    if "dependencies" not in poetry:
        return data

    dependencies = poetry["dependencies"]
    new_data = deepcopy(data)

    for k, v in dependencies.items():
        if "path" not in v:
            continue

        other_pyproject = (
            pyproject.parent / Path(v["path"])
        ).resolve() / "pyproject.toml"
        with open(other_pyproject, "rb") as f:
            other_data = tomllib.load(f)

        new_version = _get_version(other_data)
        if new_version is None:
            continue

        new_data["tool"]["poetry"]["dependencies"][k] = f"^{new_version}"

    return new_data


def main(root_dir_path: str | Path):
    pyproject = Path(root_dir_path) / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    if "project" in data:
        new_data = _convert_project_dependencies(data, pyproject)

    if "tool" in data and "poetry" in data["tool"]:
        new_data = _convert_poetry_dependencies(data, pyproject)

    with open(pyproject, "wb") as f:
        tomli_w.dump(new_data, f)


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
