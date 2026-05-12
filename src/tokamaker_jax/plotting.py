"""Plotting helpers for seed equilibria."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch

from tokamaker_jax.config import CoilConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.free_boundary import coil_flux_on_grid
from tokamaker_jax.geometry import Region, RegionSet
from tokamaker_jax.mesh import TriMesh
from tokamaker_jax.solver import EquilibriumSolution

RZ_AXES = {
    "x": {
        "label": "R",
        "units": "m",
        "convention": "Major radius in axisymmetric cylindrical coordinates.",
    },
    "y": {
        "label": "Z",
        "units": "m",
        "convention": "Vertical coordinate in axisymmetric cylindrical coordinates.",
    },
}


@dataclass(frozen=True)
class FigureRecipe:
    """JSON-friendly description of the data behind a reproducible figure."""

    name: str
    source: str | None = None
    citation: str | None = None
    command: str | None = None
    axes: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data representation suitable for JSON serialization."""

        recipe: dict[str, Any] = {
            "name": self.name,
            "axes": _json_ready(self.axes),
            "data": _json_ready(self.data),
            "metadata": _json_ready(self.metadata),
        }
        if self.source is not None:
            recipe["source"] = self.source
        if self.citation is not None:
            recipe["citation"] = self.citation
        if self.command is not None:
            recipe["command"] = self.command
        return recipe

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the recipe as a deterministic JSON document."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def equilibrium_metadata_summary(solution: EquilibriumSolution) -> dict[str, Any]:
    """Return compact validation metadata for an equilibrium solution."""

    residual = np.asarray(solution.residual_history)
    return {
        "iterations": int(solution.iterations),
        "grid": {
            "nr": int(solution.grid.nr),
            "nz": int(solution.grid.nz),
            "r_min": float(solution.grid.r_min),
            "r_max": float(solution.grid.r_max),
            "z_min": float(solution.grid.z_min),
            "z_max": float(solution.grid.z_max),
            "dr": float(solution.grid.dr),
            "dz": float(solution.grid.dz),
        },
        "psi": _field_summary(solution.psi),
        "source": _field_summary(solution.source),
        "residual": {
            "shape": list(residual.shape),
            "dtype": str(residual.dtype),
            "initial": _finite_scalar(residual[0]) if residual.size else None,
            "final": _finite_scalar(residual[-1]) if residual.size else None,
            "range": _finite_range(residual),
        },
    }


def region_table_data(regions: RegionSet | Sequence[Region]) -> list[dict[str, Any]]:
    """Return flat JSON-ready rows for displaying or exporting region metadata."""

    region_tuple = _region_tuple(regions)
    if not region_tuple:
        raise ValueError("regions must contain at least one region")
    rows = []
    for region in region_tuple:
        r_min, r_max, z_min, z_max = region.bounds
        centroid_r, centroid_z = region.centroid
        rows.append(
            {
                "id": int(region.id),
                "name": region.name,
                "kind": region.kind,
                "area": float(region.area),
                "centroid_r": float(centroid_r),
                "centroid_z": float(centroid_z),
                "r_min": float(r_min),
                "r_max": float(r_max),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": int(region.points.shape[0]),
                "n_holes": int(len(region.holes)),
                "target_size": None if region.target_size is None else float(region.target_size),
                "metadata": _json_ready(region.metadata),
            }
        )
    return rows


def equilibrium_figure_data(
    solution: EquilibriumSolution,
    *,
    name: str = "fixed_boundary_equilibrium",
    source: str | None = None,
    citation: str | None = None,
    command: str | None = None,
    include_source: bool = True,
) -> FigureRecipe:
    """Return structured figure data for :func:`plot_equilibrium`."""

    r, z = solution.grid.mesh(dtype=solution.psi.dtype)
    data = {
        "R": _array_payload(r),
        "Z": _array_payload(z),
        "psi": _array_payload(
            solution.psi,
            label="Poloidal flux",
            units=None,
            convention="Solver-native fixed-boundary flux values.",
        ),
    }
    if include_source:
        data["source"] = _array_payload(
            solution.source,
            label="Grad-Shafranov source",
            units=None,
            convention="Solver-native right-hand-side source values.",
        )
    return FigureRecipe(
        name=name,
        source=source,
        citation=citation,
        command=command,
        axes=dict(RZ_AXES),
        data=data,
        metadata={
            "plot_type": "equilibrium_contours",
            "grid": {
                "nr": solution.grid.nr,
                "nz": solution.grid.nz,
                "dr": solution.grid.dr,
                "dz": solution.grid.dz,
            },
            "iterations": solution.iterations,
            "summary": equilibrium_metadata_summary(solution),
        },
    )


def mesh_figure_data(
    mesh: TriMesh,
    *,
    name: str = "triangular_mesh",
    source: str | None = None,
    citation: str | None = None,
    command: str | None = None,
) -> FigureRecipe:
    """Return structured figure data for :func:`plot_mesh`."""

    return FigureRecipe(
        name=name,
        source=source if source is not None else mesh.source_path,
        citation=citation,
        command=command,
        axes=dict(RZ_AXES),
        data={
            "nodes": _array_payload(
                mesh.nodes,
                columns=["R", "Z"],
                units=["m", "m"],
                convention="Node coordinates in axisymmetric R-Z space.",
            ),
            "triangles": _array_payload(
                mesh.triangles,
                columns=["node_0", "node_1", "node_2"],
                convention="Zero-based triangle node indices.",
            ),
            "cell_regions": _array_payload(
                mesh.regions,
                label="Region id",
                convention="One-based TokaMaker region ids, one value per triangular cell.",
            ),
        },
        metadata={
            "plot_type": "triangular_mesh",
            "summary": mesh.summary(),
            "coil_names": sorted(mesh.coil_dict),
            "conductor_names": list(mesh.conductor_names()),
            "vacuum_names": list(mesh.vacuum_names()),
        },
    )


def region_figure_data(
    regions: RegionSet | tuple[Region, ...] | list[Region],
    *,
    name: str = "region_geometry",
    source: str | None = None,
    citation: str | None = None,
    command: str | None = None,
) -> FigureRecipe:
    """Return structured figure data for :func:`plot_regions`."""

    region_tuple = _region_tuple(regions)
    if not region_tuple:
        raise ValueError("regions must contain at least one region")
    all_points = np.vstack([region.points for region in region_tuple])
    return FigureRecipe(
        name=name,
        source=source,
        citation=citation,
        command=command,
        axes=dict(RZ_AXES),
        data={
            "regions": [
                {
                    "id": region.id,
                    "name": region.name,
                    "kind": region.kind,
                    "points": _array_payload(
                        region.points,
                        columns=["R", "Z"],
                        units=["m", "m"],
                        convention="Counterclockwise outer region loop.",
                    ),
                    "holes": [
                        _array_payload(
                            hole,
                            columns=["R", "Z"],
                            units=["m", "m"],
                            convention="Counterclockwise inner region loop.",
                        )
                        for hole in region.holes
                    ],
                    "area": region.area,
                    "centroid": region.centroid,
                    "target_size": region.target_size,
                    "metadata": region.metadata,
                }
                for region in region_tuple
            ],
        },
        metadata={
            "plot_type": "region_geometry",
            "n_regions": len(region_tuple),
            "bounds": {
                "R": _finite_range(all_points[:, 0]),
                "Z": _finite_range(all_points[:, 1]),
            },
            "table": region_table_data(region_tuple),
        },
    )


def coil_response_figure_data(
    grid: RectangularGrid,
    coils: tuple[CoilConfig, ...],
    *,
    name: str = "reduced_coil_green_response",
    source: str | None = None,
    citation: str | None = None,
    command: str | None = None,
) -> FigureRecipe:
    """Return structured figure data for reduced coil Green's response plots."""

    r, z = grid.mesh()
    flux = coil_flux_on_grid(grid, coils)
    return FigureRecipe(
        name=name,
        source=source,
        citation=citation,
        command=command,
        axes=dict(RZ_AXES),
        data={
            "R": _array_payload(r, label="R", units="m"),
            "Z": _array_payload(z, label="Z", units="m"),
            "coil_flux": _array_payload(
                flux,
                label="Reduced free-boundary coil flux",
                convention="Large-aspect-ratio logarithmic Green's-function fixture.",
            ),
            "coils": [
                {
                    "name": coil.name,
                    "r": coil.r,
                    "z": coil.z,
                    "current": coil.current,
                    "core_radius": coil.sigma,
                }
                for coil in coils
            ],
        },
        metadata={
            "plot_type": "reduced_coil_green_response",
            "n_coils": len(coils),
            "grid": {
                "nr": grid.nr,
                "nz": grid.nz,
                "dr": grid.dr,
                "dz": grid.dz,
            },
            "flux": _field_summary(flux),
        },
    )


def plot_equilibrium(
    solution: EquilibriumSolution,
    *,
    levels: int = 24,
    ax: plt.Axes | None = None,
    show_source: bool = False,
    show_metadata: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot flux contours, optionally with the source term as a background."""

    r, z = solution.grid.mesh(dtype=solution.psi.dtype)
    fig, ax = (
        plt.subplots(figsize=(6.5, 5.2), constrained_layout=True) if ax is None else (ax.figure, ax)
    )
    r_np = np.asarray(r)
    z_np = np.asarray(z)
    psi_np = np.asarray(solution.psi)
    if show_source:
        source = ax.contourf(r_np, z_np, np.asarray(solution.source), levels=levels, cmap="magma")
        fig.colorbar(source, ax=ax, label="source")
    contours = ax.contour(r_np, z_np, psi_np, levels=levels, colors="black", linewidths=0.75)
    ax.clabel(contours, inline=True, fontsize=7)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("tokamaker-jax fixed-boundary seed equilibrium")
    if show_metadata:
        _annotate_equilibrium_metadata(ax, equilibrium_metadata_summary(solution))
    return fig, ax


def plot_coil_green_response(
    grid: RectangularGrid,
    coils: tuple[CoilConfig, ...],
    *,
    levels: int = 28,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the reduced free-boundary coil Green's-function response."""

    r, z = grid.mesh()
    flux = np.asarray(coil_flux_on_grid(grid, coils))
    fig, ax = (
        plt.subplots(figsize=(6.5, 5.2), constrained_layout=True) if ax is None else (ax.figure, ax)
    )
    filled = ax.contourf(np.asarray(r), np.asarray(z), flux, levels=levels, cmap="viridis")
    fig.colorbar(filled, ax=ax, label="coil flux")
    contours = ax.contour(
        np.asarray(r), np.asarray(z), flux, levels=levels, colors="black", linewidths=0.55
    )
    ax.clabel(contours, inline=True, fontsize=7)
    if coils:
        ax.scatter(
            [coil.r for coil in coils],
            [coil.z for coil in coils],
            c=["tab:red" if coil.current >= 0.0 else "tab:blue" for coil in coils],
            marker="s",
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
            label="PF coils",
        )
        for coil in coils:
            ax.text(
                coil.r + 0.055,
                coil.z,
                coil.name,
                va="center",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "alpha": 0.72,
                    "linewidth": 0.0,
                },
                zorder=5,
            )
        ax.legend(loc="upper right")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("reduced free-boundary coil Green's response")
    return fig, ax


def save_coil_green_response_plot(
    grid: RectangularGrid,
    coils: tuple[CoilConfig, ...],
    path: str | Path,
) -> Path:
    """Save a reduced coil Green's-function response plot."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_coil_green_response(grid, coils)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.resolve()


def save_equilibrium_plot(solution: EquilibriumSolution, path: str | Path) -> Path:
    """Save a PNG/SVG/PDF plot and return the resolved path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_equilibrium(solution, show_source=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.resolve()


def plot_mesh(
    mesh: TriMesh,
    *,
    ax: plt.Axes | None = None,
    show_regions: bool = True,
    show_edges: bool = True,
    linewidth: float = 0.35,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a triangular mesh with optional region coloring."""

    fig, ax = (
        plt.subplots(figsize=(6.5, 5.2), constrained_layout=True) if ax is None else (ax.figure, ax)
    )
    triangulation = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)
    if show_regions:
        region_plot = ax.tripcolor(
            triangulation,
            mesh.regions.astype(float),
            shading="flat",
            cmap="tab20",
            alpha=0.75,
        )
        fig.colorbar(region_plot, ax=ax, label="region id")
    if show_edges:
        ax.triplot(triangulation, color="black", linewidth=linewidth, alpha=0.45)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("tokamaker-jax triangular mesh")
    return fig, ax


def save_mesh_plot(mesh: TriMesh, path: str | Path) -> Path:
    """Save a mesh preview plot and return the resolved path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_mesh(mesh)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.resolve()


def plot_regions(
    regions: RegionSet | tuple[Region, ...] | list[Region],
    *,
    ax: plt.Axes | None = None,
    show_labels: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot region geometry loops for machine-definition previews."""

    region_tuple = _region_tuple(regions)
    if not region_tuple:
        raise ValueError("regions must contain at least one region")
    fig, ax = (
        plt.subplots(figsize=(6.5, 5.2), constrained_layout=True) if ax is None else (ax.figure, ax)
    )
    colors = plt.get_cmap("tab20")
    for index, region in enumerate(region_tuple):
        patch = PolygonPatch(
            region.points,
            closed=True,
            facecolor=colors(index % 20),
            edgecolor="black",
            linewidth=1.0,
            alpha=0.45,
        )
        ax.add_patch(patch)
        for hole in region.holes:
            hole_patch = PolygonPatch(
                hole,
                closed=True,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=1.0,
            )
            ax.add_patch(hole_patch)
        if show_labels:
            r, z = _region_label_position(region)
            ax.text(r, z, region.name, ha="center", va="center", fontsize=8)
    all_points = np.vstack([region.points for region in region_tuple])
    margin = 0.08 * max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]), 1.0)
    ax.set_xlim(float(np.min(all_points[:, 0]) - margin), float(np.max(all_points[:, 0]) + margin))
    ax.set_ylim(float(np.min(all_points[:, 1]) - margin), float(np.max(all_points[:, 1]) + margin))
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("tokamaker-jax region geometry")
    return fig, ax


def save_region_plot(
    regions: RegionSet | tuple[Region, ...] | list[Region], path: str | Path
) -> Path:
    """Save a region preview plot and return the resolved path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_regions(regions)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.resolve()


def _region_label_position(region: Region) -> tuple[float, float]:
    if not region.holes:
        return region.centroid
    inner_max_r = max(float(np.max(hole[:, 0])) for hole in region.holes)
    outer_max_r = float(np.max(region.points[:, 0]))
    _, center_z = region.centroid
    return 0.5 * (inner_max_r + outer_max_r), center_z


def _region_tuple(regions: RegionSet | Sequence[Region]) -> tuple[Region, ...]:
    return regions.regions if isinstance(regions, RegionSet) else tuple(regions)


def _annotate_equilibrium_metadata(ax: plt.Axes, summary: dict[str, Any]) -> None:
    psi_range = summary["psi"]["range"]
    residual = summary["residual"]
    text = "\n".join(
        (
            f"grid {summary['grid']['nr']}x{summary['grid']['nz']}",
            f"iterations {summary['iterations']}",
            f"psi [{_format_range_endpoint(psi_range['min'])}, "
            f"{_format_range_endpoint(psi_range['max'])}]",
            f"residual {_format_range_endpoint(residual['final'])}",
        )
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "linewidth": 0.5},
    )


def _array_payload(
    values: Any,
    *,
    label: str | None = None,
    units: str | list[str] | None = None,
    columns: list[str] | None = None,
    convention: str | None = None,
) -> dict[str, Any]:
    array = np.asarray(values)
    payload: dict[str, Any] = {
        "values": array.tolist(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "range": _finite_range(array),
    }
    if label is not None:
        payload["label"] = label
    if units is not None:
        payload["units"] = units
    if columns is not None:
        payload["columns"] = columns
    if convention is not None:
        payload["convention"] = convention
    return payload


def _field_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "range": _finite_range(array),
    }


def _finite_range(values: Any) -> dict[str, float | bool | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": None, "max": None, "all_finite": False}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "all_finite": bool(finite.size == array.size),
    }


def _finite_scalar(value: Any) -> float | None:
    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _format_range_endpoint(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
