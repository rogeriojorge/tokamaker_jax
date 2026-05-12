"""Verification cases for FEM and physics validation gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tokamaker_jax.assembly import (
    assemble_grad_shafranov_stiffness_matrix,
    assemble_laplace_stiffness_matrix,
    assemble_load_vector,
    axisymmetric_inverse_radius,
    boundary_nodes_from_coordinates,
    solve_dirichlet_system,
)
from tokamaker_jax.fem import (
    linear_basis,
    map_to_physical,
    physical_basis_gradients,
    triangle_jacobian,
    triangle_quadrature,
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


@dataclass(frozen=True)
class GradShafranovConvergenceResult:
    """Error metrics for one manufactured axisymmetric refinement level."""

    subdivisions_r: int
    subdivisions_z: int
    n_nodes: int
    n_cells: int
    h: float
    l2_error: float
    weighted_h1_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "subdivisions_r": self.subdivisions_r,
            "subdivisions_z": self.subdivisions_z,
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "h": self.h,
            "l2_error": self.l2_error,
            "weighted_h1_error": self.weighted_h1_error,
        }


@dataclass(frozen=True)
class GradShafranovConvergenceStudy:
    """Observed rates for the manufactured axisymmetric weak-form case."""

    results: tuple[GradShafranovConvergenceResult, ...]
    l2_rates: tuple[float, ...]
    weighted_h1_rates: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "results": [result.to_dict() for result in self.results],
            "l2_rates": list(self.l2_rates),
            "weighted_h1_rates": list(self.weighted_h1_rates),
        }


def unit_square_triangles(subdivisions: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a uniform right-triangle mesh on the unit square."""

    return rectangular_triangles(0.0, 1.0, 0.0, 1.0, subdivisions, subdivisions)


def rectangular_triangles(
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    subdivisions_r: int,
    subdivisions_z: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a uniform right-triangle mesh on a rectangular ``(R, Z)`` domain."""

    if subdivisions_z is None:
        subdivisions_z = subdivisions_r
    if subdivisions_r < 2 or subdivisions_z < 2:
        raise ValueError("subdivisions must be at least 2")
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min")
    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min")
    nodes = jnp.asarray(
        [
            [
                r_min + (r_max - r_min) * i / subdivisions_r,
                z_min + (z_max - z_min) * j / subdivisions_z,
            ]
            for j in range(subdivisions_z + 1)
            for i in range(subdivisions_r + 1)
        ],
        dtype=jnp.float64,
    )
    cells = []
    for j in range(subdivisions_z):
        for i in range(subdivisions_r):
            lower_left = j * (subdivisions_r + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + subdivisions_r + 1
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


def sine_poisson_gradient(points: jnp.ndarray) -> jnp.ndarray:
    """Exact gradient of :func:`sine_poisson_exact`."""

    points = jnp.asarray(points, dtype=jnp.float64)
    return jnp.column_stack(
        (
            jnp.pi * jnp.cos(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1]),
            jnp.pi * jnp.sin(jnp.pi * points[:, 0]) * jnp.cos(jnp.pi * points[:, 1]),
        )
    )


def solve_sine_poisson(subdivisions: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the manufactured p=1 Poisson problem on one refinement level."""

    nodes, triangles = unit_square_triangles(subdivisions)
    stiffness = assemble_laplace_stiffness_matrix(nodes, triangles)
    rhs = assemble_load_vector(nodes, triangles, sine_poisson_forcing)
    boundary_nodes = boundary_nodes_from_coordinates(nodes)
    boundary_values = sine_poisson_exact(nodes[boundary_nodes])
    solution = solve_dirichlet_system(stiffness, rhs, boundary_nodes, boundary_values)
    return nodes, triangles, solution


def manufactured_grad_shafranov_exact(
    points: jnp.ndarray,
    *,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> jnp.ndarray:
    """Exact flux for the manufactured axisymmetric weak-form gate."""

    points = jnp.asarray(points, dtype=jnp.float64)
    kr, kz, phase_r, phase_z = _manufactured_grad_shafranov_terms(
        points,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )
    return jnp.sin(phase_r) * jnp.sin(phase_z)


def manufactured_grad_shafranov_gradient(
    points: jnp.ndarray,
    *,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> jnp.ndarray:
    """Exact ``(dpsi/dR, dpsi/dZ)`` for the axisymmetric gate."""

    points = jnp.asarray(points, dtype=jnp.float64)
    kr, kz, phase_r, phase_z = _manufactured_grad_shafranov_terms(
        points,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )
    return jnp.column_stack(
        (
            kr * jnp.cos(phase_r) * jnp.sin(phase_z),
            kz * jnp.sin(phase_r) * jnp.cos(phase_z),
        )
    )


def manufactured_grad_shafranov_source(
    points: jnp.ndarray,
    *,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> jnp.ndarray:
    """Weak source for ``-div((1/R) grad psi) = q``."""

    points = jnp.asarray(points, dtype=jnp.float64)
    r = points[:, 0]
    kr, kz, phase_r, phase_z = _manufactured_grad_shafranov_terms(
        points,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )
    psi = jnp.sin(phase_r) * jnp.sin(phase_z)
    psi_r = kr * jnp.cos(phase_r) * jnp.sin(phase_z)
    return ((kr**2 + kz**2) * psi) / r + psi_r / r**2


def solve_manufactured_grad_shafranov(
    subdivisions: int,
    *,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the manufactured axisymmetric weak-form problem."""

    nodes, triangles = rectangular_triangles(r_min, r_max, z_min, z_max, subdivisions)
    stiffness = assemble_grad_shafranov_stiffness_matrix(nodes, triangles)
    rhs = assemble_load_vector(
        nodes,
        triangles,
        lambda points: manufactured_grad_shafranov_source(
            points,
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
        ),
    )
    boundary_nodes = boundary_nodes_from_coordinates(nodes)
    boundary_values = manufactured_grad_shafranov_exact(
        nodes[boundary_nodes],
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )
    solution = solve_dirichlet_system(stiffness, rhs, boundary_nodes, boundary_values)
    return nodes, triangles, solution


def poisson_error_metrics(subdivisions: int) -> PoissonConvergenceResult:
    """Return L2 and H1-seminorm errors for one manufactured Poisson solve."""

    nodes, triangles, solution = solve_sine_poisson(subdivisions)
    l2_error, h1_error = integrated_p1_error_norms(
        nodes,
        triangles,
        solution,
        sine_poisson_exact,
        sine_poisson_gradient,
    )
    return PoissonConvergenceResult(
        subdivisions=subdivisions,
        n_nodes=int(nodes.shape[0]),
        n_cells=int(triangles.shape[0]),
        h=1.0 / subdivisions,
        l2_error=float(l2_error),
        h1_error=float(h1_error),
    )


def manufactured_grad_shafranov_error_metrics(
    subdivisions: int,
    *,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> GradShafranovConvergenceResult:
    """Return L2 and weighted H1 errors for the axisymmetric gate."""

    nodes, triangles, solution = solve_manufactured_grad_shafranov(
        subdivisions,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )
    l2_error, weighted_h1_error = integrated_p1_error_norms(
        nodes,
        triangles,
        solution,
        lambda points: manufactured_grad_shafranov_exact(
            points,
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
        ),
        lambda points: manufactured_grad_shafranov_gradient(
            points,
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
        ),
        gradient_weight=axisymmetric_inverse_radius,
    )
    return GradShafranovConvergenceResult(
        subdivisions_r=subdivisions,
        subdivisions_z=subdivisions,
        n_nodes=int(nodes.shape[0]),
        n_cells=int(triangles.shape[0]),
        h=max((r_max - r_min) / subdivisions, (z_max - z_min) / subdivisions),
        l2_error=float(l2_error),
        weighted_h1_error=float(weighted_h1_error),
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


def run_grad_shafranov_convergence_study(
    subdivisions: tuple[int, ...] = (4, 8, 16),
) -> GradShafranovConvergenceStudy:
    """Run the manufactured axisymmetric Grad-Shafranov convergence gate."""

    if len(subdivisions) < 2:
        raise ValueError("at least two refinement levels are required")
    results = tuple(manufactured_grad_shafranov_error_metrics(level) for level in subdivisions)
    mesh_sizes = [result.h for result in results]
    return GradShafranovConvergenceStudy(
        results=results,
        l2_rates=observed_rates([result.l2_error for result in results], mesh_sizes),
        weighted_h1_rates=observed_rates(
            [result.weighted_h1_error for result in results], mesh_sizes
        ),
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


def integrated_p1_error_norms(
    nodes: jnp.ndarray,
    triangles: jnp.ndarray,
    solution: jnp.ndarray,
    exact: Any,
    exact_gradient: Any,
    *,
    gradient_weight: Any | None = None,
    quadrature_degree: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return quadrature-integrated L2 and weighted H1 seminorm errors."""

    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    solution = jnp.asarray(solution, dtype=jnp.float64)
    if solution.shape != (nodes.shape[0],):
        raise ValueError("solution must have shape (n_nodes,)")

    def element_error(
        vertices: jnp.ndarray, nodal_values: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        quadrature = triangle_quadrature(quadrature_degree)
        physical_points = map_to_physical(vertices, quadrature.points)
        basis = linear_basis(quadrature.points)
        approximate = basis @ nodal_values
        approximate_gradient = nodal_values @ physical_basis_gradients(vertices)
        exact_values = jnp.asarray(exact(physical_points), dtype=jnp.float64)
        exact_gradients = jnp.asarray(exact_gradient(physical_points), dtype=jnp.float64)
        if exact_values.shape != quadrature.weights.shape:
            raise ValueError("exact must return one value per quadrature point")
        if exact_gradients.shape != (quadrature.weights.shape[0], 2):
            raise ValueError("exact_gradient must return shape (n_quadrature, 2)")
        det_jacobian = jnp.abs(jnp.linalg.det(triangle_jacobian(vertices)))
        value_error = approximate - exact_values
        gradient_error = approximate_gradient[None, :] - exact_gradients
        if gradient_weight is None:
            weights = jnp.ones_like(quadrature.weights)
        else:
            weights = jnp.asarray(gradient_weight(physical_points), dtype=jnp.float64)
            if weights.shape != quadrature.weights.shape:
                raise ValueError("gradient_weight must return one value per quadrature point")
        l2_squared = det_jacobian * jnp.sum(quadrature.weights * value_error**2)
        h1_squared = det_jacobian * jnp.sum(
            quadrature.weights * weights * jnp.sum(gradient_error**2, axis=1)
        )
        return l2_squared, h1_squared

    l2_squared, h1_squared = jax_vmap_element_error(
        element_error, nodes[triangles], solution[triangles]
    )
    return jnp.sqrt(jnp.sum(l2_squared)), jnp.sqrt(jnp.sum(h1_squared))


def jax_vmap_element_error(
    element_error: Any,
    vertices: jnp.ndarray,
    nodal_values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized wrapper kept separate for testable error messages."""

    import jax

    return jax.vmap(element_error)(vertices, nodal_values)


def _manufactured_grad_shafranov_terms(
    points: jnp.ndarray,
    *,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> tuple[float, float, jnp.ndarray, jnp.ndarray]:
    kr = jnp.pi / (r_max - r_min)
    kz = jnp.pi / (z_max - z_min)
    phase_r = kr * (points[:, 0] - r_min)
    phase_z = kz * (points[:, 1] - z_min)
    return kr, kz, phase_r, phase_z
