"""Plotting helpers for seed equilibria."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch

from tokamaker_jax.geometry import Region, RegionSet
from tokamaker_jax.mesh import TriMesh
from tokamaker_jax.solver import EquilibriumSolution


def plot_equilibrium(
    solution: EquilibriumSolution,
    *,
    levels: int = 24,
    ax: plt.Axes | None = None,
    show_source: bool = False,
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
    return fig, ax


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

    region_tuple = regions.regions if isinstance(regions, RegionSet) else tuple(regions)
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
