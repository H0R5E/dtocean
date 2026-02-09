from copy import deepcopy
from pathlib import Path

import tomli_w
import tomllib

ROOT_DIR = Path(__file__).parents[1]


def _get_version(data):
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


def main():
    pyproject = ROOT_DIR / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    dependencies = data["tool"]["poetry"]["dependencies"]
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
        new_data["tool"]["poetry"]["dependencies"][k] = f"^{new_version}"

    with open(pyproject, "wb") as f:
        tomli_w.dump(new_data, f)


if __name__ == "__main__":
    main()
