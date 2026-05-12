"""Plotting helpers for seed equilibria."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
