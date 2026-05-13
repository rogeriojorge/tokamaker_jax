import json

import pytest
from conftest import REPO_ROOT

from tokamaker_jax.cases import (
    case_manifest_to_json,
    case_source_preview,
    case_table_rows,
    default_case_manifest,
    write_case_manifest,
)


def test_default_case_manifest_is_json_ready_and_explicit_about_parity_levels():
    manifest = default_case_manifest(REPO_ROOT)
    payload = manifest.to_dict()

    json.dumps(payload)
    assert payload["artifact_id"] == "tokamaker-jax-case-manifest"
    assert payload["status_counts"]["runnable"] >= 3
    assert payload["status_counts"]["planned_upstream_fixture"] >= 4

    fixed = manifest.by_id("fixed-boundary-seed")
    assert fixed.runnable
    assert fixed.path == "examples/fixed_boundary.toml"
    assert fixed.parity_level == "manufactured_validation"
    assert fixed.to_dict(root=REPO_ROOT)["path_exists"] is True

    iter_case = manifest.by_id("iter-baseline-upstream")
    assert not iter_case.runnable
    assert iter_case.parity_level == "source_audit"
    assert any("ITER_baseline_ex.ipynb" in source for source in iter_case.upstream_sources)


def test_case_manifest_filters_and_table_rows():
    manifest = default_case_manifest(REPO_ROOT)
    runnable = manifest.filter(runnable_only=True)
    planned = manifest.filter(status="planned_upstream_fixture")
    rows = case_table_rows(runnable)

    assert all(entry.runnable for entry in runnable.entries)
    assert all(entry.status == "planned_upstream_fixture" for entry in planned.entries)
    assert {row["case_id"] for row in rows} >= {
        "fixed-boundary-seed",
        "case-manifest-browser",
        "cpc-seed-family",
    }


def test_case_source_preview_reads_local_toml_and_reports_missing_file(tmp_path):
    manifest = default_case_manifest(REPO_ROOT)
    preview = case_source_preview("fixed-boundary-seed", manifest=manifest, root=REPO_ROOT)

    assert preview["exists"] is True
    assert preview["truncated"] is False
    assert "[grid]" in preview["source"]
    assert "pressure_scale" in preview["source"]

    missing_manifest = default_case_manifest(tmp_path)
    missing = case_source_preview("fixed-boundary-seed", manifest=missing_manifest, root=tmp_path)

    assert missing["exists"] is False
    assert "not present" in missing["message"]


def test_case_source_preview_rejects_unknown_case():
    with pytest.raises(KeyError, match="unknown case id"):
        case_source_preview("not-a-case")


def test_write_case_manifest_roundtrips_json(tmp_path):
    manifest = default_case_manifest(REPO_ROOT).filter(runnable_only=True)
    output = write_case_manifest(tmp_path / "case_manifest.json", manifest=manifest)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == len(manifest.entries)
    assert all(entry["runnable"] for entry in payload["entries"])
    assert case_manifest_to_json(manifest).endswith("\n")
