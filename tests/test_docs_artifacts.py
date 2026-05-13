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
        "fixed_boundary_upstream",
        "io_contract",
        "case_manifest",
        "browser_explorer",
    ):
        assert f"\n{page}\n" in index
        assert (docs_dir / f"{page}.md").exists()


def test_static_browser_explorer_is_self_contained_and_linked():
    explorer_docs = (REPO_ROOT / "docs" / "browser_explorer.md").read_text(encoding="utf-8")
    explorer = (REPO_ROOT / "docs" / "_static" / "tokamaker_jax_explorer.html").read_text(
        encoding="utf-8"
    )

    assert "tokamaker_jax_explorer.html" in explorer_docs
    assert "tokamaker-jax browser equilibrium explorer" in explorer
    assert "static HTML" in explorer
    assert "navigator.clipboard.writeText" in explorer
    assert "tokamaker-jax init-example fixed-boundary" in explorer
    assert "https://" not in explorer
    assert "<script src=" not in explorer


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
        "fixed_boundary_upstream_geqdsk.png",
        "io_artifact_map.png",
        "case_manifest_status.png",
        "tokamaker_jax_explorer_screenshot.png",
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


def test_fixed_boundary_evidence_docs_manifest_and_json_are_bounded():
    validation_docs = (REPO_ROOT / "docs" / "validation.md").read_text(encoding="utf-8")
    comparisons = (REPO_ROOT / "docs" / "comparisons.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "docs" / "validation" / "physics_gates_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (REPO_ROOT / "docs" / "validation" / "fixed_boundary_upstream_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    gates = {gate["id"]: gate for gate in manifest["gates"]}

    assert "Upstream Fixed-Boundary Evidence" in validation_docs
    assert (
        "does not prove that `tokamaker-jax` reproduces those solved equilibria" in validation_docs
    )
    assert "fixed_boundary_upstream_evidence.json" in comparisons
    assert gates["fixed-boundary-upstream-evidence"]["status"] == "implemented_source_evidence"
    assert gates["fixed-boundary-upstream-evidence"]["pass_rule"]["numeric_parity_claim"] is False
    assert report["artifact_id"] == "fixed-boundary-upstream-evidence"
    assert report["numeric_parity_claim"] is False


def test_fixed_boundary_upstream_docs_and_json_record_evidence():
    docs_text = (REPO_ROOT / "docs" / "fixed_boundary_upstream.md").read_text(encoding="utf-8")
    report_path = REPO_ROOT / "docs" / "_static" / "fixed_boundary_upstream_evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "not a full parity claim" in docs_text
    assert report["artifact_id"] == "upstream-fixed-boundary-evidence"
    assert report["claim"] == "source_evidence_only"
    assert report["numeric_parity_claim"] is False
    assert report["geqdsk"]["nr"] == 129
    assert report["geqdsk"]["nz"] == 129
    assert report["geqdsk"]["psi_shape"] == [129, 129]
    assert any(notebook["uses_geqdsk"] for notebook in report["notebooks"])
    assert any(notebook["free_boundary_assignments"] > 0 for notebook in report["notebooks"])
