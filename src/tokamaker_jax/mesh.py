"""Triangular mesh containers and TokaMaker mesh IO."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

MeshDict = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TriMesh:
    """Unstructured triangular mesh in axisymmetric ``R, Z`` coordinates.

    The mesh topology is intentionally NumPy-backed and static. Later FEM and
    solver kernels can pass coordinates into JAX separately when a workflow
    needs differentiable geometry with fixed connectivity.
    """

    nodes: NDArray[np.float64]
    triangles: NDArray[np.int32]
    regions: NDArray[np.int32]
    coil_dict: MeshDict = field(default_factory=dict)
    cond_dict: MeshDict = field(default_factory=dict)
    source_path: str | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int32)
        regions = np.asarray(self.regions, dtype=np.int32)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "coil_dict", _copy_mesh_dict(self.coil_dict))
        object.__setattr__(self, "cond_dict", _copy_mesh_dict(self.cond_dict))
        self.validate()

    @property
    def n_nodes(self) -> int:
        """Number of mesh vertices."""

        return int(self.nodes.shape[0])

    @property
    def n_cells(self) -> int:
        """Number of triangular cells."""

        return int(self.triangles.shape[0])

    @property
    def region_ids(self) -> NDArray[np.int32]:
        """Sorted unique region ids."""

        return np.unique(self.regions)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(r_min, r_max, z_min, z_max)``."""

        return (
            float(np.min(self.nodes[:, 0])),
            float(np.max(self.nodes[:, 0])),
            float(np.min(self.nodes[:, 1])),
            float(np.max(self.nodes[:, 1])),
        )

    def validate(self) -> None:
        """Validate core topology and geometry invariants."""

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n_nodes, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (n_cells, 3)")
        if self.regions.shape != (self.triangles.shape[0],):
            raise ValueError("regions must have shape (n_cells,)")
        if self.nodes.shape[0] < 3:
            raise ValueError("mesh must contain at least 3 nodes")
        if self.triangles.shape[0] < 1:
            raise ValueError("mesh must contain at least 1 cell")
        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("nodes must all be finite")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= self.nodes.shape[0]:
            raise ValueError("triangles contain node indices outside the mesh")
        if np.min(self.regions) < 1:
            raise ValueError("regions must use one-based TokaMaker region ids")
        if np.any(self.cell_areas() <= 0.0):
            raise ValueError("all triangle areas must be positive")
        _validate_region_references("coil_dict", self.coil_dict, self.region_ids)
        _validate_region_references("cond_dict", self.cond_dict, self.region_ids)

    def cell_areas(self) -> NDArray[np.float64]:
        """Return signed-positive triangle areas."""

        points = self.nodes[self.triangles]
        edge_1 = points[:, 1, :] - points[:, 0, :]
        edge_2 = points[:, 2, :] - points[:, 0, :]
        return 0.5 * (edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])

    def cell_centers(self) -> NDArray[np.float64]:
        """Return cell center coordinates."""

        return np.mean(self.nodes[self.triangles], axis=1)

    def region_mask(self, region_id: int) -> NDArray[np.bool_]:
        """Return a cell mask for one region id."""

        return self.regions == int(region_id)

    def region_cell_counts(self) -> dict[int, int]:
        """Return cell count by region id."""

        ids, counts = np.unique(self.regions, return_counts=True)
        return {int(region_id): int(count) for region_id, count in zip(ids, counts, strict=True)}

    def region_areas(self) -> dict[int, float]:
        """Return total poloidal area by region id."""

        areas = self.cell_areas()
        return {
            int(region_id): float(np.sum(areas[self.regions == region_id]))
            for region_id in self.region_ids
        }

    def boundary_edges(self) -> NDArray[np.int32]:
        """Return deterministic exterior mesh edges as sorted node index pairs."""

        tri = self.triangles
        all_edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
        all_edges = np.sort(all_edges, axis=1)
        edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        boundary = edges[counts == 1]
        order = np.lexsort((boundary[:, 1], boundary[:, 0]))
        return np.asarray(boundary[order], dtype=np.int32)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for logs, tests, and GUI previews."""

        r_min, r_max, z_min, z_max = self.bounds
        areas = self.cell_areas()
        return {
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "n_regions": int(self.region_ids.size),
            "n_coils": len(self.coil_dict),
            "n_conductors": len(self.conductor_names()),
            "n_vacuum_regions": len(self.vacuum_names()),
            "n_boundary_edges": int(self.boundary_edges().shape[0]),
            "r_min": r_min,
            "r_max": r_max,
            "z_min": z_min,
            "z_max": z_max,
            "area_min": float(np.min(areas)),
            "area_max": float(np.max(areas)),
            "area_total": float(np.sum(areas)),
        }

    def conductor_names(self) -> tuple[str, ...]:
        """Return names in ``cond_dict`` that represent conducting structures."""

        return tuple(name for name, data in self.cond_dict.items() if "cond_id" in data)

    def vacuum_names(self) -> tuple[str, ...]:
        """Return names in ``cond_dict`` that represent vacuum regions."""

        return tuple(name for name, data in self.cond_dict.items() if "vac_id" in data)


def load_gs_mesh(path: str | Path, *, use_hdf5: bool | None = None) -> TriMesh:
    """Load a native TokaMaker Grad-Shafranov mesh.

    HDF5 files use the upstream datasets ``mesh/r``, ``mesh/lc``, ``mesh/reg``,
    ``mesh/coil_dict``, and ``mesh/cond_dict``. JSON files use the same logical
    schema as upstream ``save_gs_mesh(..., use_hdf5=False)``.
    """

    path = Path(path)
    use_hdf5 = path.suffix.lower() in {".h5", ".hdf5"} if use_hdf5 is None else use_hdf5
    if use_hdf5:
        with h5py.File(path, "r") as h5_file:
            group = h5_file["mesh"]
            nodes = np.asarray(group["r"], dtype=np.float64)
            triangles = np.asarray(group["lc"], dtype=np.int32)
            regions = np.asarray(group["reg"], dtype=np.int32)
            coil_dict = _decode_json_dataset(group["coil_dict"][()])
            cond_dict = _decode_json_dataset(group["cond_dict"][()])
    else:
        raw = json.loads(path.read_text())
        mesh = raw["mesh"]
        nodes = np.asarray(mesh["r"], dtype=np.float64)
        triangles = np.asarray(mesh["lc"], dtype=np.int32)
        regions = np.asarray(mesh["reg"], dtype=np.int32)
        coil_dict = mesh.get("coil_dict", {})
        cond_dict = mesh.get("cond_dict", {})
    return TriMesh(
        nodes=nodes,
        triangles=triangles,
        regions=regions,
        coil_dict=coil_dict,
        cond_dict=cond_dict,
        source_path=str(path),
    )


def save_gs_mesh(mesh: TriMesh, path: str | Path, *, use_hdf5: bool | None = None) -> Path:
    """Save a :class:`TriMesh` in TokaMaker native mesh format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_hdf5 = path.suffix.lower() in {".h5", ".hdf5"} if use_hdf5 is None else use_hdf5
    if use_hdf5:
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("mesh/r", data=mesh.nodes, dtype="f8")
            h5_file.create_dataset("mesh/lc", data=mesh.triangles, dtype="i4")
            h5_file.create_dataset("mesh/reg", data=mesh.regions, dtype="i4")
            string_datatype = h5py.string_dtype("ascii")
            h5_file.create_dataset(
                "mesh/coil_dict",
                data=json.dumps(mesh.coil_dict),
                dtype=string_datatype,
            )
            h5_file.create_dataset(
                "mesh/cond_dict",
                data=json.dumps(mesh.cond_dict),
                dtype=string_datatype,
            )
    else:
        path.write_text(
            json.dumps(
                {
                    "mesh": {
                        "r": mesh.nodes.tolist(),
                        "lc": mesh.triangles.tolist(),
                        "reg": mesh.regions.tolist(),
                        "coil_dict": mesh.coil_dict,
                        "cond_dict": mesh.cond_dict,
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
    return path.resolve()


def mesh_from_arrays(
    nodes: ArrayLike,
    triangles: ArrayLike,
    regions: ArrayLike,
    *,
    coil_dict: MeshDict | None = None,
    cond_dict: MeshDict | None = None,
) -> TriMesh:
    """Build a validated :class:`TriMesh` from array-like inputs."""

    return TriMesh(
        nodes=np.asarray(nodes, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int32),
        regions=np.asarray(regions, dtype=np.int32),
        coil_dict={} if coil_dict is None else coil_dict,
        cond_dict={} if cond_dict is None else cond_dict,
    )


def _decode_json_dataset(value: Any) -> MeshDict:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, np.bytes_):
        value = value.decode("ascii")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("mesh metadata JSON must decode to a dictionary")
    return decoded


def _copy_mesh_dict(value: MeshDict) -> MeshDict:
    return {str(name): dict(data) for name, data in value.items()}


def _validate_region_references(
    name: str, mapping: MeshDict, region_ids: NDArray[np.int32]
) -> None:
    valid_ids = {int(region_id) for region_id in region_ids}
    for entry_name, data in mapping.items():
        region_id = data.get("reg_id")
        if region_id is None:
            raise ValueError(f"{name}[{entry_name!r}] is missing reg_id")
        if int(region_id) not in valid_ids:
            raise ValueError(f"{name}[{entry_name!r}] references unknown region id {region_id}")
