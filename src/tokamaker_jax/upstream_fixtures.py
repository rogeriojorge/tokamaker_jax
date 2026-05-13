"""Summaries for upstream OpenFUSIONToolkit/TokaMaker example fixtures."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tokamaker_jax.mesh import load_gs_mesh

DEFAULT_OPENFUSIONTOOLKIT_ROOT = Path(
    os.environ.get("OPENFUSIONTOOLKIT_ROOT", "/Users/rogeriojorge/local/OpenFUSIONToolkit")
)


@dataclass(frozen=True)
class UpstreamFixture:
    """One upstream TokaMaker example fixture to inventory."""

    fixture_id: str
    title: str
    category: str
    mesh_path: str | None = None
    geometry_path: str | None = None
    example_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready fixture description."""

        return {
            "fixture_id": self.fixture_id,
            "title": self.title,
            "category": self.category,
            "mesh_path": self.mesh_path,
            "geometry_path": self.geometry_path,
            "example_paths": list(self.example_paths),
            "notes": list(self.notes),
        }


def default_upstream_fixtures() -> tuple[UpstreamFixture, ...]:
    """Return the upstream fixture inventory tracked by docs and tests."""

    return _DEFAULT_UPSTREAM_FIXTURES


def summarize_upstream_fixture(
    fixture: UpstreamFixture,
    *,
    root: str | Path = DEFAULT_OPENFUSIONTOOLKIT_ROOT,
) -> dict[str, Any]:
    """Summarize one upstream fixture without solving it."""

    root_path = Path(root)
    mesh_summary = None
    geometry_summary = None
    mesh_available = False
    geometry_available = False

    if fixture.mesh_path is not None:
        mesh_file = root_path / fixture.mesh_path
        mesh_available = mesh_file.exists()
        if mesh_available:
            mesh = load_gs_mesh(mesh_file)
            mesh_summary = {
                **mesh.summary(),
                "region_cell_counts": _int_key_dict(mesh.region_cell_counts()),
                "region_areas": _float_key_dict(mesh.region_areas()),
                "sha256": _sha256_file(mesh_file),
                "path": fixture.mesh_path,
            }

    if fixture.geometry_path is not None:
        geometry_file = root_path / fixture.geometry_path
        geometry_available = geometry_file.exists()
        if geometry_available:
            geometry_summary = {
                **_summarize_geometry_json(geometry_file),
                "sha256": _sha256_file(geometry_file),
                "path": fixture.geometry_path,
            }

    examples = [
        {
            "path": path,
            "exists": (root_path / path).exists(),
            "sha256": _sha256_file(root_path / path) if (root_path / path).is_file() else None,
        }
        for path in fixture.example_paths
    ]

    return {
        **fixture.to_dict(),
        "available": bool(
            mesh_available or geometry_available or any(row["exists"] for row in examples)
        ),
        "mesh_available": mesh_available,
        "geometry_available": geometry_available,
        "mesh": mesh_summary,
        "geometry": geometry_summary,
        "examples": examples,
        "parity_level": "source_audit",
        "claim": "fixture_inventory_only",
    }


def summarize_upstream_fixtures(
    *,
    root: str | Path = DEFAULT_OPENFUSIONTOOLKIT_ROOT,
    fixtures: tuple[UpstreamFixture, ...] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready summary report for upstream fixtures."""

    root_path = Path(root)
    fixture_list = _DEFAULT_UPSTREAM_FIXTURES if fixtures is None else fixtures
    summaries = [summarize_upstream_fixture(fixture, root=root_path) for fixture in fixture_list]
    return {
        "schema_version": 1,
        "artifact_id": "upstream-tokamaker-fixture-summary",
        "checkout_path": str(root_path),
        "checkout_exists": root_path.exists(),
        "fixture_count": len(summaries),
        "available_fixture_count": sum(1 for item in summaries if item["available"]),
        "claim": "mesh_geometry_inventory_only",
        "entries": summaries,
    }


def upstream_fixture_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    """Return compact table rows for fixture summaries."""

    rows = []
    for entry in report.get("entries", []):
        mesh = entry.get("mesh")
        geometry = entry.get("geometry")
        rows.append(
            {
                "fixture_id": str(entry["fixture_id"]),
                "category": str(entry["category"]),
                "available": "yes" if entry.get("available") else "no",
                "mesh": _mesh_label(mesh),
                "geometry": _geometry_label(geometry),
                "claim": str(entry.get("claim", "")),
            }
        )
    return rows


def upstream_fixture_report_to_json(report: dict[str, Any]) -> str:
    """Serialize an upstream fixture report with stable formatting."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_upstream_fixture_summary(
    path: str | Path,
    *,
    root: str | Path = DEFAULT_OPENFUSIONTOOLKIT_ROOT,
    fixtures: tuple[UpstreamFixture, ...] | None = None,
) -> Path:
    """Write an upstream fixture summary JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        upstream_fixture_report_to_json(summarize_upstream_fixtures(root=root, fixtures=fixtures)),
        encoding="utf-8",
    )
    return output_path.resolve()


def _summarize_geometry_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"geometry JSON must be an object: {path}")
    pairs = np.asarray(_coordinate_pairs(raw), dtype=np.float64)
    bounds = None
    if pairs.size:
        bounds = {
            "r_min": float(np.min(pairs[:, 0])),
            "r_max": float(np.max(pairs[:, 0])),
            "z_min": float(np.min(pairs[:, 1])),
            "z_max": float(np.max(pairs[:, 1])),
        }
    return {
        "top_level_keys": sorted(raw),
        "limiter_points": _coordinate_pair_count(raw.get("limiter")),
        "coil_count": _object_count(raw.get("coils")),
        "coil_names": _object_names(raw.get("coils")),
        "vv_count": _object_count(raw.get("vv")),
        "coordinate_pair_count": int(pairs.shape[0]) if pairs.size else 0,
        "bounds": bounds,
    }


def _coordinate_pair_count(value: Any) -> int:
    return len(_coordinate_pairs(value))


def _coordinate_pairs(value: Any) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    if _is_coordinate_pair(value):
        return [(float(value[0]), float(value[1]))]
    if isinstance(value, dict):
        for item in value.values():
            pairs.extend(_coordinate_pairs(item))
    elif isinstance(value, list | tuple):
        for item in value:
            pairs.extend(_coordinate_pairs(item))
    return pairs


def _is_coordinate_pair(value: Any) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    return all(isinstance(item, int | float) and np.isfinite(float(item)) for item in value)


def _object_count(value: Any) -> int:
    if isinstance(value, dict | list | tuple):
        return len(value)
    return 0


def _object_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    return []


def _int_key_dict(value: dict[int, int]) -> dict[str, int]:
    return {str(key): int(value[key]) for key in sorted(value)}


def _float_key_dict(value: dict[int, float]) -> dict[str, float]:
    return {str(key): float(value[key]) for key in sorted(value)}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mesh_label(mesh: Any) -> str:
    if not isinstance(mesh, dict):
        return "missing"
    return (
        f"{mesh['n_nodes']} nodes, {mesh['n_cells']} cells, "
        f"{mesh['n_regions']} regions, {mesh['n_coils']} coils"
    )


def _geometry_label(geometry: Any) -> str:
    if not isinstance(geometry, dict):
        return "missing"
    return (
        f"{geometry['coordinate_pair_count']} points, "
        f"{geometry['coil_count']} coils, {geometry['vv_count']} vv entries"
    )


_EXAMPLE_ROOT = "src/examples/TokaMaker"

_DEFAULT_UPSTREAM_FIXTURES = (
    UpstreamFixture(
        fixture_id="nstxu-isoflux-controller",
        title="NSTX-U isoflux controller",
        category="control",
        mesh_path=f"{_EXAMPLE_ROOT}/AdvancedWorkflows/IsofluxController/NSTXU_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/AdvancedWorkflows/IsofluxController/NSTXU_geom.json",
        example_paths=(
            f"{_EXAMPLE_ROOT}/AdvancedWorkflows/IsofluxController/NSTXU_shape_control_simulator.ipynb",
            f"{_EXAMPLE_ROOT}/AdvancedWorkflows/IsofluxController/NSTXU_shape_generator.ipynb",
        ),
    ),
    UpstreamFixture(
        fixture_id="cute",
        title="CUTE equilibrium and VDE",
        category="time-dependent",
        mesh_path=f"{_EXAMPLE_ROOT}/CUTE/CUTE_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/CUTE/CUTE_geom.json",
        example_paths=(f"{_EXAMPLE_ROOT}/CUTE/CUTE_VDE_ex.ipynb",),
    ),
    UpstreamFixture(
        fixture_id="diiid",
        title="DIII-D baseline",
        category="reconstruction",
        mesh_path=f"{_EXAMPLE_ROOT}/DIIID/DIIID_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/DIIID/DIIID_geom.json",
        example_paths=(
            f"{_EXAMPLE_ROOT}/DIIID/DIIID_baseline_ex.ipynb",
            f"{_EXAMPLE_ROOT}/DIIID/g192185.02440",
        ),
    ),
    UpstreamFixture(
        fixture_id="dipole",
        title="Dipole equilibrium",
        category="non-tokamak",
        mesh_path=f"{_EXAMPLE_ROOT}/Dipole/dipole_mesh.h5",
        example_paths=(f"{_EXAMPLE_ROOT}/Dipole/dipole_eq_ex.ipynb",),
        notes=("No upstream geometry JSON is shipped for this fixture.",),
    ),
    UpstreamFixture(
        fixture_id="hbt",
        title="HBT equilibrium",
        category="free-boundary",
        mesh_path=f"{_EXAMPLE_ROOT}/HBT/HBT_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/HBT/HBT_geom.json",
        example_paths=(
            f"{_EXAMPLE_ROOT}/HBT/HBT_eq_ex.ipynb",
            f"{_EXAMPLE_ROOT}/HBT/HBT_vac_coils.ipynb",
        ),
    ),
    UpstreamFixture(
        fixture_id="iter",
        title="ITER baseline",
        category="free-boundary",
        mesh_path=f"{_EXAMPLE_ROOT}/ITER/ITER_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/ITER/ITER_geom.json",
        example_paths=(
            f"{_EXAMPLE_ROOT}/ITER/ITER_baseline_ex.ipynb",
            f"{_EXAMPLE_ROOT}/ITER/ITER_Hmode_ex.ipynb",
        ),
    ),
    UpstreamFixture(
        fixture_id="ltx",
        title="LTX equilibrium",
        category="free-boundary",
        mesh_path=f"{_EXAMPLE_ROOT}/LTX/LTX_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/LTX/LTX_geom.json",
        example_paths=(f"{_EXAMPLE_ROOT}/LTX/LTX_eq_ex.ipynb",),
    ),
    UpstreamFixture(
        fixture_id="manta",
        title="MANTA baseline",
        category="free-boundary",
        mesh_path=f"{_EXAMPLE_ROOT}/MANTA/MANTA_mesh.h5",
        geometry_path=f"{_EXAMPLE_ROOT}/MANTA/MANTA_geom.json",
        example_paths=(f"{_EXAMPLE_ROOT}/MANTA/MANTA_baseline.ipynb",),
    ),
)
