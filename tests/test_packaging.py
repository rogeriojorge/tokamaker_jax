from __future__ import annotations

import re
from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from conftest import REPO_ROOT

from tokamaker_jax.examples import available_examples, example_text


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


def test_package_metadata_is_release_ready():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    docs_conf = (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert project["name"] == "tokamaker-jax"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", project["version"])
    assert f'release = "{project["version"]}"' in docs_conf
    assert f"version: {project['version']}" in citation
    assert 'date-released: "2026-05-13"' in citation
    assert project["description"]
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.10"
    assert project["license"]["text"] == "LGPL-3.0-only"
    assert project["authors"]
    assert project["keywords"]
    assert project["classifiers"]
    assert "tokamaker-jax" in project["scripts"]

    urls = project["urls"]
    assert urls["Homepage"] == "https://github.com/rogeriojorge/tokamaker_jax"
    assert urls["Documentation"] == "https://tokamaker-jax.readthedocs.io"
    assert urls["Issues"] == "https://github.com/rogeriojorge/tokamaker_jax/issues"

    wheel_target = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel_target["packages"] == ["src/tokamaker_jax"]


def test_packaged_fixed_boundary_example_is_available_for_pypi_installs():
    resource = resources.files("tokamaker_jax.example_data").joinpath("fixed_boundary.toml")

    assert "fixed-boundary" in available_examples()
    assert resource.is_file()
    text = example_text("fixed-boundary")
    assert "[grid]" in text
    assert "pressure_scale = 5000.0" in text
    assert text == (REPO_ROOT / "examples" / "fixed_boundary.toml").read_text(encoding="utf-8")
