"""Small nonlinear triangular-FEM equilibrium iterations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from tokamaker_jax.assembly import (
    assemble_grad_shafranov_profile_load_vector,
    assemble_grad_shafranov_stiffness_matrix,
    boundary_nodes_from_coordinates,
    solve_dirichlet_system,
)
from tokamaker_jax.fem import (
    linear_basis,
    map_to_physical,
    triangle_jacobian,
    triangle_quadrature,
)
from tokamaker_jax.profiles import MU0, power_profile
from tokamaker_jax.verification import rectangular_triangles


@dataclass(frozen=True)
class PowerProfile:
    """Power-law derivative profile parameters."""

    scale: float
    alpha: float = 1.0
    gamma: float = 1.0


@dataclass(frozen=True)
class NonlinearProfileParameters:
    """Pressure and ``FF'`` profile derivative parameters."""

    pressure: PowerProfile = field(default_factory=lambda: PowerProfile(5.0e3, 1.0, 1.0))
    ffprime: PowerProfile = field(default_factory=lambda: PowerProfile(-0.35, 1.0, 1.0))


@dataclass(frozen=True)
class FemProfileIterationSolution:
    """Result of a fixed-boundary nonlinear p=1 FEM profile iteration."""

    nodes: jnp.ndarray
    triangles: jnp.ndarray
    psi: jnp.ndarray
    rhs: jnp.ndarray
    residual_history: jnp.ndarray
    update_history: jnp.ndarray
    iterations: int
    parameters: NonlinearProfileParameters

    def stats(self) -> dict[str, float | int]:
        """Return compact scalar diagnostics."""

        return {
            "iterations": self.iterations,
            "n_nodes": int(self.nodes.shape[0]),
            "n_cells": int(self.triangles.shape[0]),
            "psi_min": float(jnp.min(self.psi)),
            "psi_max": float(jnp.max(self.psi)),
            "residual_final": float(self.residual_history[-1]),
            "update_final": float(self.update_history[-1]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic payload."""

        return {
            "stats": self.stats(),
            "parameters": {
                "pressure": self.parameters.pressure.__dict__,
                "ffprime": self.parameters.ffprime.__dict__,
            },
            "residual_history": jnp.asarray(self.residual_history).tolist(),
            "update_history": jnp.asarray(self.update_history).tolist(),
        }


@dataclass(frozen=True)
class ProfileIterationValidation:
    """Validation metrics for the nonlinear p=1 profile iteration."""

    n_nodes: int
    n_cells: int
    load_oracle_error: float
    residual_initial: float
    residual_final: float
    update_final: float
    pressure_gradient: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "load_oracle_error": self.load_oracle_error,
            "residual_initial": self.residual_initial,
            "residual_final": self.residual_final,
            "update_final": self.update_final,
            "pressure_gradient": self.pressure_gradient,
        }


def normalize_profile_flux(psi: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Normalize nodal flux to ``[0, 1]`` for profile evaluation."""

    psi = jnp.asarray(psi, dtype=jnp.float64)
    psi_min = jnp.min(psi)
    psi_max = jnp.max(psi)
    return (psi - psi_min) / (psi_max - psi_min + eps)


def nonlinear_profile_source_density(
    points: jnp.ndarray,
    psin: jnp.ndarray,
    parameters: NonlinearProfileParameters,
) -> jnp.ndarray:
    """Return weak-form profile density from normalized flux values."""

    points = jnp.asarray(points, dtype=jnp.float64)
    psin = jnp.asarray(psin, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if psin.shape != (points.shape[0],):
        raise ValueError("psin must have one value per point")
    pressure_prime = parameters.pressure.scale * power_profile(
        psin,
        parameters.pressure.alpha,
        parameters.pressure.gamma,
    )
    ffprime = parameters.ffprime.scale * power_profile(
        psin,
        parameters.ffprime.alpha,
        parameters.ffprime.gamma,
    )
    radius = points[:, 0]
    return 0.5 * ffprime / radius + MU0 * radius * pressure_prime


def linear_nonlinear_profile_load_vector(
    vertices: jnp.ndarray,
    nodal_psin: jnp.ndarray,
    parameters: NonlinearProfileParameters,
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Integrate one p=1 load vector with flux-dependent profiles."""

    vertices = jnp.asarray(vertices, dtype=jnp.float64)
    nodal_psin = jnp.asarray(nodal_psin, dtype=jnp.float64)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")
    if nodal_psin.shape != (3,):
        raise ValueError("nodal_psin must have shape (3,)")
    quadrature = triangle_quadrature(quadrature_degree)
    physical_points = map_to_physical(vertices, quadrature.points)
    basis = linear_basis(quadrature.points)
    quadrature_psin = basis @ nodal_psin
    source_values = nonlinear_profile_source_density(physical_points, quadrature_psin, parameters)
    det_jacobian = jnp.abs(jnp.linalg.det(triangle_jacobian(vertices)))
    return det_jacobian * jnp.sum(
        quadrature.weights[:, None] * source_values[:, None] * basis,
        axis=0,
    )


def assemble_nonlinear_profile_load_vector(
    nodes: jnp.ndarray,
    triangles: jnp.ndarray,
    psin: jnp.ndarray,
    parameters: NonlinearProfileParameters,
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Assemble the p=1 nonlinear profile source vector for fixed topology."""

    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    psin = jnp.asarray(psin, dtype=jnp.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n_cells, 3)")
    if psin.shape != (nodes.shape[0],):
        raise ValueError("psin must have shape (n_nodes,)")
    element_vectors = jax.vmap(
        lambda vertices, nodal_values: linear_nonlinear_profile_load_vector(
            vertices,
            nodal_values,
            parameters,
            quadrature_degree=quadrature_degree,
        )
    )(nodes[triangles], psin[triangles])
    return (
        jnp.zeros((nodes.shape[0],), dtype=element_vectors.dtype)
        .at[triangles.reshape(-1)]
        .add(element_vectors.reshape(-1))
    )


def solve_profile_iteration(
    nodes: jnp.ndarray,
    triangles: jnp.ndarray,
    parameters: NonlinearProfileParameters | None = None,
    *,
    iterations: int = 4,
    relaxation: float = 0.8,
    initial_psi: jnp.ndarray | None = None,
    dirichlet_nodes: jnp.ndarray | None = None,
    dirichlet_values: jnp.ndarray | None = None,
) -> FemProfileIterationSolution:
    """Run a dense fixed-boundary Picard iteration for p=1 profiles."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must satisfy 0 < relaxation <= 1")
    if parameters is None:
        parameters = NonlinearProfileParameters()
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    if initial_psi is None:
        psi = jnp.zeros((nodes.shape[0],), dtype=jnp.float64)
    else:
        psi = jnp.asarray(initial_psi, dtype=jnp.float64)
    if psi.shape != (nodes.shape[0],):
        raise ValueError("initial_psi must have shape (n_nodes,)")
    if dirichlet_nodes is None:
        dirichlet_nodes = boundary_nodes_from_coordinates(nodes)
    else:
        dirichlet_nodes = jnp.asarray(dirichlet_nodes, dtype=jnp.int32)
    if dirichlet_values is None:
        dirichlet_values = jnp.zeros((dirichlet_nodes.shape[0],), dtype=jnp.float64)
    else:
        dirichlet_values = jnp.asarray(dirichlet_values, dtype=jnp.float64)
    if dirichlet_values.shape != dirichlet_nodes.shape:
        raise ValueError("dirichlet_values must match dirichlet_nodes")

    stiffness = assemble_grad_shafranov_stiffness_matrix(nodes, triangles)
    residuals = []
    updates = []
    rhs = jnp.zeros((nodes.shape[0],), dtype=jnp.float64)
    for _ in range(iterations):
        previous = psi
        psin = normalize_profile_flux(previous)
        rhs = assemble_nonlinear_profile_load_vector(nodes, triangles, psin, parameters)
        solved = solve_dirichlet_system(stiffness, rhs, dirichlet_nodes, dirichlet_values)
        psi = (1.0 - relaxation) * previous + relaxation * solved
        free_mask = jnp.ones((nodes.shape[0],), dtype=bool).at[dirichlet_nodes].set(False)
        free_nodes = jnp.where(free_mask)[0]
        free_residual = (stiffness @ psi - rhs)[free_nodes]
        residual = jnp.linalg.norm(free_residual) / (jnp.linalg.norm(rhs[free_nodes]) + 1.0e-30)
        update = jnp.linalg.norm(psi - previous) / (jnp.linalg.norm(psi) + 1.0e-30)
        residuals.append(residual)
        updates.append(update)

    return FemProfileIterationSolution(
        nodes=nodes,
        triangles=triangles,
        psi=psi,
        rhs=rhs,
        residual_history=jnp.asarray(residuals),
        update_history=jnp.asarray(updates),
        iterations=iterations,
        parameters=parameters,
    )


def solve_profile_iteration_on_rectangle(
    subdivisions: int = 8,
    parameters: NonlinearProfileParameters | None = None,
    *,
    iterations: int = 4,
    relaxation: float = 0.8,
    r_min: float = 1.0,
    r_max: float = 2.0,
    z_min: float = -0.5,
    z_max: float = 0.5,
) -> FemProfileIterationSolution:
    """Run the nonlinear profile iteration on a rectangular triangular mesh."""

    nodes, triangles = rectangular_triangles(r_min, r_max, z_min, z_max, subdivisions)
    return solve_profile_iteration(
        nodes,
        triangles,
        parameters,
        iterations=iterations,
        relaxation=relaxation,
    )


def constant_profile_load_oracle(
    nodes: jnp.ndarray,
    triangles: jnp.ndarray,
    parameters: NonlinearProfileParameters,
) -> jnp.ndarray:
    """Return the scalar-profile load using the existing callable assembly path."""

    return assemble_grad_shafranov_profile_load_vector(
        nodes,
        triangles,
        pressure_prime=parameters.pressure.scale,
        ffprime=parameters.ffprime.scale,
    )


def run_profile_iteration_validation() -> ProfileIterationValidation:
    """Run fast analytic checks for the nonlinear p=1 profile iteration."""

    nodes, triangles = rectangular_triangles(1.0, 2.0, -0.5, 0.5, 4)
    constant_parameters = NonlinearProfileParameters(
        pressure=PowerProfile(scale=2.5, alpha=0.0, gamma=1.0),
        ffprime=PowerProfile(scale=0.8, alpha=0.0, gamma=1.0),
    )
    psin = jnp.linspace(0.0, 1.0, nodes.shape[0], dtype=jnp.float64)
    load = assemble_nonlinear_profile_load_vector(nodes, triangles, psin, constant_parameters)
    oracle = constant_profile_load_oracle(nodes, triangles, constant_parameters)
    solution = solve_profile_iteration(
        nodes,
        triangles,
        constant_parameters,
        iterations=3,
        relaxation=0.9,
    )

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        parameters = NonlinearProfileParameters(
            pressure=PowerProfile(scale=scale, alpha=0.0, gamma=1.0),
            ffprime=PowerProfile(scale=0.1, alpha=0.0, gamma=1.0),
        )
        return jnp.mean(
            solve_profile_iteration(
                nodes,
                triangles,
                parameters,
                iterations=2,
                relaxation=0.8,
            ).psi
            ** 2
        )

    pressure_gradient = jax.grad(objective)(jnp.asarray(2.0, dtype=jnp.float64))
    return ProfileIterationValidation(
        n_nodes=int(nodes.shape[0]),
        n_cells=int(triangles.shape[0]),
        load_oracle_error=float(jnp.linalg.norm(load - oracle)),
        residual_initial=float(solution.residual_history[0]),
        residual_final=float(solution.residual_history[-1]),
        update_final=float(solution.update_history[-1]),
        pressure_gradient=float(pressure_gradient),
    )
