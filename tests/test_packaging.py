from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from conftest import REPO_ROOT


def test_default_install_includes_gui_dependencies_and_no_gui_extra():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert "nicegui" in dependencies
    assert "plotly" in dependencies
    assert "gui" not in optional


def test_dependency_declarations_are_unversioned():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    declarations = list(pyproject["build-system"]["requires"])
    declarations.extend(pyproject["project"]["dependencies"])
    for group in pyproject["project"]["optional-dependencies"].values():
        declarations.extend(group)

    for declaration in declarations:
        package_name = declaration.split(";", maxsplit=1)[0].strip()
        assert all(operator not in package_name for operator in ("<", ">", "=", "~", "!"))
