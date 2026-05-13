import hashlib
import json

import numpy as np
from conftest import REPO_ROOT

from tokamaker_jax.mesh import mesh_from_arrays, save_gs_mesh
from tokamaker_jax.upstream_fixtures import (
    UpstreamFixture,
    summarize_upstream_fixture,
    summarize_upstream_fixtures,
    upstream_fixture_report_to_json,
    upstream_fixture_rows,
    write_upstream_fixture_summary,
)

_EXPECTED_COMMITTED_FIXTURE_ORDER = (
    "nstxu-isoflux-controller",
    "cute",
    "diiid",
    "dipole",
    "hbt",
    "iter",
    "ltx",
    "manta",
)

_EXPECTED_COMMITTED_FIXTURE_SNAPSHOT = {
    "nstxu-isoflux-controller": {
        "category": "control",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_mesh.h5",
            "a3d3eca8ccc39ddee6d298e7837659157a5ef62041a35f1107274015e50c0072",
            16122,
            32138,
            40,
            30,
            8,
            1,
            104,
        ),
        "geometry": (
            "src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_geom.json",
            "f7d783c8ad8f479451272ca4548c6976fbd71933e704ac22e201c6278d72f638",
            47,
            30,
            18,
            260,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_shape_control_simulator.ipynb",
                True,
                "01552bb53012cc1ae2b9fc611f89eedbba1d31525dac11c12449a6a67425678f",
            ),
            (
                "src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_shape_generator.ipynb",
                True,
                "b26a32a7069c37bc60ac64631be70c661404dd0321b6db8c3ac9d3083635771b",
            ),
        ),
    },
    "cute": {
        "category": "time-dependent",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/CUTE/CUTE_mesh.h5",
            "1de4c9ffad84d45a77ec10f035fced12f02cdb39f918c6d48009956674449eb2",
            5796,
            11488,
            31,
            28,
            1,
            1,
            102,
        ),
        "geometry": (
            "src/examples/TokaMaker/CUTE/CUTE_geom.json",
            "c63000a07ca02d1a8f25b9d66abeafa58d667082dd115f3d3b528f6e88e51f12",
            0,
            28,
            3,
            162,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/CUTE/CUTE_VDE_ex.ipynb",
                True,
                "68e9c98b38c451e188c65c19d8ed1c13a1de997126074913ede630b210f6a366",
            ),
        ),
    },
    "diiid": {
        "category": "reconstruction",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/DIIID/DIIID_mesh.h5",
            "31c1c52cbd2f7bdedae74380a7a0a02426917179953544e7faea69606219001c",
            8911,
            17660,
            85,
            58,
            24,
            2,
            160,
        ),
        "geometry": (
            "src/examples/TokaMaker/DIIID/DIIID_geom.json",
            "e2b8a56fde1336190a07c27b03073e440a58112541cb5b29f50e297b3c70612a",
            73,
            20,
            24,
            401,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/DIIID/DIIID_baseline_ex.ipynb",
                True,
                "51d9e96bad041507fd22b8c13d532092af63127f78ea99f944696f8ec7372924",
            ),
            (
                "src/examples/TokaMaker/DIIID/g192185.02440",
                True,
                "6f33a01935847f25aea6edc45e939268ab910570ea6e8adb87528056a77520d5",
            ),
        ),
    },
    "dipole": {
        "category": "non-tokamak",
        "available": True,
        "mesh_available": True,
        "geometry_available": False,
        "mesh": (
            "src/examples/TokaMaker/Dipole/dipole_mesh.h5",
            "67437e54695a00782e3a741180833c21897e0bf933b35e42064bc12b6d00649f",
            8546,
            16912,
            6,
            2,
            1,
            2,
            178,
        ),
        "geometry": None,
        "examples": (
            (
                "src/examples/TokaMaker/Dipole/dipole_eq_ex.ipynb",
                True,
                "897607dcbd323b36918a474da7010883b583625eaadd5ae26bc47994c49ade54",
            ),
        ),
    },
    "hbt": {
        "category": "free-boundary",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/HBT/HBT_mesh.h5",
            "d197135f421b77c9b8ccf6c4ddd5854427f515a2b303fb9a3ef9ae73b1f3d5ba",
            3736,
            7352,
            35,
            30,
            2,
            2,
            118,
        ),
        "geometry": (
            "src/examples/TokaMaker/HBT/HBT_geom.json",
            "c6dbbec20dae385b63f6382ca799152fc3b58deef4373b90fc625cdf8fa7225c",
            128,
            30,
            4,
            1581,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/HBT/HBT_eq_ex.ipynb",
                True,
                "1e81085d809702267e68247dddf30bd42c6388ce4cfd04e37f74cc0d0fcb99b1",
            ),
            (
                "src/examples/TokaMaker/HBT/HBT_vac_coils.ipynb",
                True,
                "ccc9b06ca2714941710d845a32871dfa3004d14c3aa9d0762dc2b5c29de70c83",
            ),
        ),
    },
    "iter": {
        "category": "free-boundary",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/ITER/ITER_mesh.h5",
            "7f67f6b83770b6f92fb0afd94290870def0254163aaaa5f2e16b8ef79ac95c09",
            4757,
            9400,
            20,
            14,
            2,
            3,
            112,
        ),
        "geometry": (
            "src/examples/TokaMaker/ITER/ITER_geom.json",
            "2b25cfe0744a94fe5edd8ad7b2e158293b20e87d525b48954a3a350a2a134bb2",
            54,
            14,
            0,
            454,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/ITER/ITER_baseline_ex.ipynb",
                True,
                "472be5c46c1e2cf14657d532c1aa71646d0060049a1921b559e9847b391c1175",
            ),
            (
                "src/examples/TokaMaker/ITER/ITER_Hmode_ex.ipynb",
                True,
                "b99bc68339cf28b48ecab87bf2bdf2b66d0b8724d9f33b27acb2bbabae625ed8",
            ),
        ),
    },
    "ltx": {
        "category": "free-boundary",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/LTX/LTX_mesh.h5",
            "a2aad95a179c9b4a28c0bba9648830629b2b8c0f8b9f3fb2351cff38a1f33f13",
            3128,
            6114,
            28,
            17,
            9,
            1,
            140,
        ),
        "geometry": (
            "src/examples/TokaMaker/LTX/LTX_geom.json",
            "e7f281cd45aac1d3f81078cb31f3765ae007a984600ceacc22887caaeb4bfba1",
            207,
            17,
            7,
            445,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/LTX/LTX_eq_ex.ipynb",
                True,
                "3705f2b213f590551ac217c148e98ec31e769a863c0976b3bf91417f60f9ba1e",
            ),
        ),
    },
    "manta": {
        "category": "free-boundary",
        "available": True,
        "mesh_available": True,
        "geometry_available": True,
        "mesh": (
            "src/examples/TokaMaker/MANTA/MANTA_mesh.h5",
            "c844efed6698ac841ae0c17cc2b971b7f40c17a5f9e7e984efa76f6d1f862253",
            8001,
            15766,
            19,
            12,
            4,
            2,
            234,
        ),
        "geometry": (
            "src/examples/TokaMaker/MANTA/MANTA_geom.json",
            "e08bae8a2000783ee7373428f0f38df00d32403fd3841f894eefc76c01467f6e",
            39,
            12,
            0,
            482,
        ),
        "examples": (
            (
                "src/examples/TokaMaker/MANTA/MANTA_baseline.ipynb",
                True,
                "7e539806bad55beb7d0878a74d6c9de811f2bd2cb1c096ca431bd603b09e5264",
            ),
        ),
    },
}

_EXPECTED_COMMITTED_COUNT_HASH_DIGEST = (
    "74dca646b527265ebbd870342d8b2c1478731c64ab1a58f36ada36bd5c6a841a"
)

_COUNT_HASH_CONTAINER_KEYS = {"entries", "examples", "mesh", "geometry"}
_COUNT_HASH_KEYS = {
    "artifact_id",
    "schema_version",
    "fixture_count",
    "available_fixture_count",
    "claim",
    "fixture_id",
    "category",
    "available",
    "mesh_available",
    "geometry_available",
    "mesh_path",
    "geometry_path",
    "path",
    "exists",
    "sha256",
    "n_nodes",
    "n_cells",
    "n_regions",
    "n_coils",
    "n_conductors",
    "n_vacuum_regions",
    "n_boundary_edges",
    "limiter_points",
    "coil_count",
    "vv_count",
    "coordinate_pair_count",
    "region_cell_counts",
}


def test_committed_upstream_fixture_snapshot_records_exact_counts_and_hashes():
    report_path = REPO_ROOT / "docs" / "_static" / "upstream_fixture_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["artifact_id"] == "upstream-tokamaker-fixture-summary"
    assert report["schema_version"] == 1
    assert report["claim"] == "mesh_geometry_inventory_only"
    assert report["fixture_count"] == 8
    assert report["available_fixture_count"] == 8
    assert (
        tuple(entry["fixture_id"] for entry in report["entries"])
        == _EXPECTED_COMMITTED_FIXTURE_ORDER
    )
    assert {
        entry["fixture_id"]: _fixture_count_hash_row(entry) for entry in report["entries"]
    } == _EXPECTED_COMMITTED_FIXTURE_SNAPSHOT
    assert _count_hash_snapshot_digest(report) == _EXPECTED_COMMITTED_COUNT_HASH_DIGEST


def _fixture_count_hash_row(entry):
    mesh = entry["mesh"]
    geometry = entry["geometry"]
    return {
        "category": entry["category"],
        "available": entry["available"],
        "mesh_available": entry["mesh_available"],
        "geometry_available": entry["geometry_available"],
        "mesh": None
        if mesh is None
        else (
            mesh["path"],
            mesh["sha256"],
            mesh["n_nodes"],
            mesh["n_cells"],
            mesh["n_regions"],
            mesh["n_coils"],
            mesh["n_conductors"],
            mesh["n_vacuum_regions"],
            mesh["n_boundary_edges"],
        ),
        "geometry": None
        if geometry is None
        else (
            geometry["path"],
            geometry["sha256"],
            geometry["limiter_points"],
            geometry["coil_count"],
            geometry["vv_count"],
            geometry["coordinate_pair_count"],
        ),
        "examples": tuple(
            (example["path"], example["exists"], example["sha256"]) for example in entry["examples"]
        ),
    }


def _count_hash_snapshot_digest(report):
    snapshot = _count_hash_snapshot(report)
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_hash_snapshot(value, parent_key=None):
    if isinstance(value, dict):
        if parent_key == "region_cell_counts":
            return {key: int(value[key]) for key in sorted(value, key=lambda item: int(item))}
        return {
            key: _count_hash_snapshot(item, key)
            for key, item in sorted(value.items())
            if _is_count_hash_snapshot_key(key)
        }
    if isinstance(value, list):
        return [_count_hash_snapshot(item, parent_key) for item in value]
    return value


def _is_count_hash_snapshot_key(key):
    return (
        key in _COUNT_HASH_CONTAINER_KEYS
        or key in _COUNT_HASH_KEYS
        or key.endswith("_count")
        or key.endswith("_counts")
        or key.startswith("n_")
        or key == "sha256"
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
