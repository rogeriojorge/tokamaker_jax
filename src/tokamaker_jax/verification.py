"""Verification cases for FEM and physics validation gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tokamaker_jax.assembly import (
    assemble_laplace_stiffness_matrix,
    assemble_load_vector,
    assemble_mass_matrix,
    boundary_nodes_from_coordinates,
    solve_dirichlet_system,
)


@dataclass(frozen=True)
class PoissonConvergenceResult:
    """Error metrics for one manufactured Poisson refinement level."""

    subdivisions: int
    n_nodes: int
    n_cells: int
    h: float
    l2_error: float
    h1_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "subdivisions": self.subdivisions,
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "h": self.h,
            "l2_error": self.l2_error,
            "h1_error": self.h1_error,
        }


@dataclass(frozen=True)
class PoissonConvergenceStudy:
    """Observed convergence rates for the manufactured Poisson case."""

    results: tuple[PoissonConvergenceResult, ...]
    l2_rates: tuple[float, ...]
    h1_rates: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "results": [result.to_dict() for result in self.results],
            "l2_rates": list(self.l2_rates),
            "h1_rates": list(self.h1_rates),
        }


def unit_square_triangles(subdivisions: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a uniform right-triangle mesh on the unit square."""

    if subdivisions < 2:
        raise ValueError("subdivisions must be at least 2")
    nodes = jnp.asarray(
        [
            [i / subdivisions, j / subdivisions]
            for j in range(subdivisions + 1)
            for i in range(subdivisions + 1)
        ],
        dtype=jnp.float64,
    )
    cells = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            lower_left = j * (subdivisions + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + subdivisions + 1
            upper_right = upper_left + 1
            cells.append([lower_left, lower_right, upper_right])
            cells.append([lower_left, upper_right, upper_left])
    return nodes, jnp.asarray(cells, dtype=jnp.int32)


def sine_poisson_exact(points: jnp.ndarray) -> jnp.ndarray:
    """Exact solution ``sin(pi x) sin(pi y)`` for the unit-square gate."""

    points = jnp.asarray(points, dtype=jnp.float64)
    return jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1])


def sine_poisson_forcing(points: jnp.ndarray) -> jnp.ndarray:
    """Forcing for ``-Delta u = f`` with :func:`sine_poisson_exact`."""

    return 2.0 * jnp.pi**2 * sine_poisson_exact(points)


def solve_sine_poisson(subdivisions: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the manufactured p=1 Poisson problem on one refinement level."""

    nodes, triangles = unit_square_triangles(subdivisions)
    stiffness = assemble_laplace_stiffness_matrix(nodes, triangles)
    rhs = assemble_load_vector(nodes, triangles, sine_poisson_forcing)
    boundary_nodes = boundary_nodes_from_coordinates(nodes)
    boundary_values = sine_poisson_exact(nodes[boundary_nodes])
    solution = solve_dirichlet_system(stiffness, rhs, boundary_nodes, boundary_values)
    return nodes, triangles, solution


def poisson_error_metrics(subdivisions: int) -> PoissonConvergenceResult:
    """Return L2 and H1-seminorm errors for one manufactured Poisson solve."""

    nodes, triangles, solution = solve_sine_poisson(subdivisions)
    exact = sine_poisson_exact(nodes)
    error = solution - exact
    mass = assemble_mass_matrix(nodes, triangles)
    stiffness = assemble_laplace_stiffness_matrix(nodes, triangles)
    return PoissonConvergenceResult(
        subdivisions=subdivisions,
        n_nodes=int(nodes.shape[0]),
        n_cells=int(triangles.shape[0]),
        h=1.0 / subdivisions,
        l2_error=float(jnp.sqrt(error @ mass @ error)),
        h1_error=float(jnp.sqrt(error @ stiffness @ error)),
    )


def run_poisson_convergence_study(
    subdivisions: tuple[int, ...] = (4, 8, 16),
) -> PoissonConvergenceStudy:
    """Run the manufactured Poisson convergence gate."""

    if len(subdivisions) < 2:
        raise ValueError("at least two refinement levels are required")
    results = tuple(poisson_error_metrics(level) for level in subdivisions)
    mesh_sizes = [result.h for result in results]
    return PoissonConvergenceStudy(
        results=results,
        l2_rates=observed_rates([result.l2_error for result in results], mesh_sizes),
        h1_rates=observed_rates([result.h1_error for result in results], mesh_sizes),
    )


def observed_rates(errors: list[float], mesh_sizes: list[float]) -> tuple[float, ...]:
    """Return observed rates between successive refinement levels."""

    if len(errors) != len(mesh_sizes):
        raise ValueError("errors and mesh_sizes must have the same length")
    if len(errors) < 2:
        raise ValueError("at least two error values are required")
    return tuple(
        float(
            jnp.log(errors[index] / errors[index + 1])
            / jnp.log(mesh_sizes[index] / mesh_sizes[index + 1])
        )
        for index in range(len(errors) - 1)
    )
