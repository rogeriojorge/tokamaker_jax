from pathlib import Path

from conftest import REPO_ROOT


def test_ci_workflow_uploads_benchmark_artifacts():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "benchmarks:" in workflow
    assert "python examples/benchmark_report.py" in workflow
    assert "--thresholds docs/validation/benchmark_thresholds.json" in workflow
    assert "outputs/benchmark_report.json" in workflow
    assert "outputs/benchmark_threshold_report.json" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_benchmark_threshold_file_is_tracked():
    path = Path(REPO_ROOT / "docs" / "validation" / "benchmark_thresholds.json")

    assert path.exists()
