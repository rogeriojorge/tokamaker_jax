from pathlib import Path

from conftest import REPO_ROOT


def read_workflow(name: str) -> str:
    return (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_ci_workflow_uploads_benchmark_artifacts():
    workflow = read_workflow("ci.yml")

    assert "benchmarks:" in workflow
    assert "python examples/benchmark_report.py" in workflow
    assert "--thresholds docs/validation/benchmark_thresholds.json" in workflow
    assert "outputs/benchmark_report.json" in workflow
    assert "outputs/benchmark_threshold_report.json" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_benchmark_threshold_file_is_tracked():
    path = Path(REPO_ROOT / "docs" / "validation" / "benchmark_thresholds.json")

    assert path.exists()


def test_publish_workflow_only_runs_for_github_releases():
    workflow = read_workflow("publish.yml")

    assert "on:" in workflow
    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "\n  push:" not in workflow
    assert "\n  pull_request:" not in workflow
    assert "\n  workflow_dispatch:" not in workflow


def test_publish_workflow_uses_trusted_publishing_permissions():
    workflow = read_workflow("publish.yml")

    assert "permissions:" in workflow
    assert "id-token: write" in workflow
    assert "contents: read" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "password:" not in workflow
    assert "api-token:" not in workflow


def test_publish_workflow_builds_and_checks_distributions_before_publish():
    workflow = read_workflow("publish.yml")

    build_index = workflow.index("python -m build")
    check_index = workflow.index("python -m twine check --strict dist/*")
    upload_index = workflow.index("actions/upload-artifact@v4")
    download_index = workflow.index("actions/download-artifact@v4")
    publish_index = workflow.index("pypa/gh-action-pypi-publish@release/v1")

    assert "python -m pip install build twine" in workflow
    assert "python -m build --sdist --wheel" in workflow
    assert "python -m pip install --force-reinstall dist/*.whl" in workflow
    assert build_index < check_index < upload_index < download_index < publish_index


def test_publish_workflow_separates_build_and_oidc_publish_jobs():
    workflow = read_workflow("publish.yml")

    assert "release-test:" in workflow
    assert "release-build:" in workflow
    assert "pypi:" in workflow
    assert "needs: release-test" in workflow
    assert "needs: release-build" in workflow
    assert "if: github.event_name == 'release'" in workflow
    assert "environment:" in workflow
    assert "name: pypi" in workflow
    assert "url: https://pypi.org/p/tokamaker-jax" in workflow


def test_publish_workflow_runs_release_validation_before_building():
    workflow = read_workflow("publish.yml")

    test_index = workflow.index("pytest --cov=tokamaker_jax --cov-fail-under=95")
    docs_index = workflow.index("sphinx-build -W -b html docs docs/_build/html")
    build_index = workflow.index("python -m build --sdist --wheel")

    assert 'python -m pip install -e ".[dev,docs]"' in workflow
    assert "ruff format --check ." in workflow
    assert "ruff check ." in workflow
    assert test_index < build_index
    assert docs_index < build_index


def test_publish_workflow_has_release_concurrency():
    workflow = read_workflow("publish.yml")

    assert "concurrency:" in workflow
    assert "group: publish-${{ github.event.release.tag_name }}" in workflow
    assert "cancel-in-progress: false" in workflow
