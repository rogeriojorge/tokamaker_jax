import json
from pathlib import Path

from conftest import REPO_ROOT


def test_expanded_docs_are_wired_into_toctree():
    docs_dir = Path(REPO_ROOT / "docs")
    index = (docs_dir / "index.md").read_text(encoding="utf-8")

    for page in (
        "equations",
        "design_decisions",
        "comparisons",
        "upstream_fixtures",
        "io_contract",
        "case_manifest",
    ):
        assert f"\n{page}\n" in index
        assert (docs_dir / f"{page}.md").exists()


def test_equation_docs_record_core_physics_contracts():
    equations = (REPO_ROOT / "docs" / "equations.md").read_text(encoding="utf-8")

    assert "Grad-Shafranov equation" in equations
    assert "\\Delta^*\\psi" in equations
    assert "OpenFUSIONToolkit/TokaMaker `eval_green`" in equations
    assert "\\psi_\\mathrm{OFT}\\approx-\\psi_\\mathrm{JAX}" in equations
    assert "Differentiability Policy" in equations


def test_comparison_docs_and_report_declare_no_full_parity_claim():
    comparisons = (REPO_ROOT / "docs" / "comparisons.md").read_text(encoding="utf-8")
    report_path = REPO_ROOT / "docs" / "_static" / "upstream_comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "Full TokaMaker replacement status remains future work." in comparisons
    assert report["artifact_id"] == "upstream-literature-comparison-matrix"
    assert report["status_levels"]["kernel_parity"].startswith("scalar/vector")
    assert any(row["reference"] == "OFT/TokaMaker" for row in report["rows"])


def test_publication_and_io_assets_exist():
    static_dir = REPO_ROOT / "docs" / "_static"
    for filename in (
        "publication_validation_panel.png",
        "upstream_comparison_matrix.png",
        "upstream_fixture_mesh_sizes.png",
        "io_artifact_map.png",
        "case_manifest_status.png",
    ):
        path = static_dir / filename
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG")


def test_case_manifest_docs_and_json_are_consistent():
    case_docs = (REPO_ROOT / "docs" / "case_manifest.md").read_text(encoding="utf-8")
    report_path = REPO_ROOT / "docs" / "_static" / "case_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "tokamaker-jax cases" in case_docs
    assert report["artifact_id"] == "tokamaker-jax-case-manifest"
    assert any(entry["case_id"] == "fixed-boundary-seed" for entry in report["entries"])
    assert any(entry["status"] == "planned_upstream_fixture" for entry in report["entries"])


def test_upstream_fixture_docs_and_json_record_exact_mesh_inventory():
    fixture_docs = (REPO_ROOT / "docs" / "upstream_fixtures.md").read_text(encoding="utf-8")
    report_path = REPO_ROOT / "docs" / "_static" / "upstream_fixture_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_id = {entry["fixture_id"]: entry for entry in report["entries"]}

    assert "fixture inventory, not a full equilibrium parity claim" in fixture_docs
    assert report["artifact_id"] == "upstream-tokamaker-fixture-summary"
    assert report["claim"] == "mesh_geometry_inventory_only"
    assert by_id["iter"]["mesh"]["n_nodes"] == 4757
    assert by_id["iter"]["mesh"]["n_cells"] == 9400
    assert by_id["diiid"]["mesh"]["n_regions"] == 85
    assert by_id["diiid"]["geometry"]["coil_count"] == 20
