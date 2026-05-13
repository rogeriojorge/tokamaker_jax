import json

import numpy as np

from tokamaker_jax.mesh import mesh_from_arrays, save_gs_mesh
from tokamaker_jax.upstream_fixtures import (
    UpstreamFixture,
    summarize_upstream_fixture,
    summarize_upstream_fixtures,
    upstream_fixture_report_to_json,
    upstream_fixture_rows,
    write_upstream_fixture_summary,
)


def test_summarize_upstream_fixture_reads_mesh_and_geometry(tmp_path):
    root = tmp_path / "OpenFUSIONToolkit"
    mesh_path = root / "src/examples/TokaMaker/TINY/TINY_mesh.h5"
    geometry_path = root / "src/examples/TokaMaker/TINY/TINY_geom.json"
    example_path = root / "src/examples/TokaMaker/TINY/TINY_eq_ex.ipynb"
    mesh = mesh_from_arrays(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.asarray([[0, 1, 2], [1, 3, 2]]),
        np.asarray([1, 2]),
        coil_dict={"PF": {"reg_id": 2}},
        cond_dict={"VAC": {"reg_id": 1, "vac_id": 1}, "VV": {"reg_id": 2, "cond_id": 1}},
    )
    save_gs_mesh(mesh, mesh_path)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(
        json.dumps(
            {
                "limiter": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                "coils": {"PF": {"r": 1.0, "z": 0.5, "outline": [[0.8, 0.4], [1.2, 0.6]]}},
                "vv": {"VV": [[0.0, 0.0], [0.0, 1.0]]},
            }
        ),
        encoding="utf-8",
    )
    example_path.write_text("{}", encoding="utf-8")
    fixture = UpstreamFixture(
        fixture_id="tiny",
        title="Tiny",
        category="test",
        mesh_path="src/examples/TokaMaker/TINY/TINY_mesh.h5",
        geometry_path="src/examples/TokaMaker/TINY/TINY_geom.json",
        example_paths=("src/examples/TokaMaker/TINY/TINY_eq_ex.ipynb",),
    )

    summary = summarize_upstream_fixture(fixture, root=root)

    assert summary["available"] is True
    assert summary["claim"] == "fixture_inventory_only"
    assert summary["mesh"]["n_nodes"] == 4
    assert summary["mesh"]["n_cells"] == 2
    assert summary["mesh"]["region_cell_counts"] == {"1": 1, "2": 1}
    assert summary["mesh"]["n_coils"] == 1
    assert summary["mesh"]["n_conductors"] == 1
    assert summary["mesh"]["n_vacuum_regions"] == 1
    assert summary["geometry"]["limiter_points"] == 3
    assert summary["geometry"]["coil_count"] == 1
    assert summary["geometry"]["vv_count"] == 1
    assert summary["examples"][0]["exists"] is True
    assert len(summary["mesh"]["sha256"]) == 64
    assert len(summary["geometry"]["sha256"]) == 64


def test_upstream_fixture_report_handles_missing_checkout(tmp_path):
    report = summarize_upstream_fixtures(root=tmp_path / "missing", fixtures=())

    assert report["checkout_exists"] is False
    assert report["fixture_count"] == 0
    assert report["available_fixture_count"] == 0
    assert upstream_fixture_rows(report) == []
    assert upstream_fixture_report_to_json(report).endswith("\n")


def test_write_upstream_fixture_summary_roundtrips_json(tmp_path):
    fixture = UpstreamFixture(
        fixture_id="missing",
        title="Missing",
        category="test",
        mesh_path="missing_mesh.h5",
        geometry_path="missing_geom.json",
    )

    output = write_upstream_fixture_summary(
        tmp_path / "summary.json",
        root=tmp_path,
        fixtures=(fixture,),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_id"] == "upstream-tokamaker-fixture-summary"
    assert payload["entries"][0]["available"] is False
    assert payload["entries"][0]["mesh"] is None
    assert upstream_fixture_rows(payload)[0]["mesh"] == "missing"
