"""Verification cases for FEM and physics validation gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from tokamaker_jax.assembly import (
    assemble_grad_shafranov_stiffness_matrix,
    assemble_laplace_stiffness_matrix,
    assemble_load_vector,
    axisymmetric_inverse_radius,
    boundary_nodes_from_coordinates,
    solve_dirichlet_system,
)
from tokamaker_jax.config import CoilConfig
from tokamaker_jax.fem import (
    linear_basis,
    map_to_physical,
    physical_basis_gradients,
    triangle_jacobian,
    triangle_quadrature,
)
from tokamaker_jax.free_boundary import (
    circular_loop_elliptic_coil_flux,
    circular_loop_elliptic_flux,
    circular_loop_elliptic_flux_gradient,
    circular_loop_elliptic_response_matrix,
    circular_loop_flux_gradient,
    circular_loop_response_matrix,
    coil_flux,
    coil_flux_gradient,
    coil_response_matrix,
    regularized_log_green_function,
)
from tokamaker_jax.profiles import MU0
from tokamaker_jax.upstream_fixed_boundary import (
    DEFAULT_OPENFUSIONTOOLKIT_ROOT,
    FIXED_BOUNDARY_EQDSK,
    FIXED_BOUNDARY_RELATIVE_ROOT,
    parse_geqdsk,
    summarize_geqdsk,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXED_BOUNDARY_ARTIFACT = (
    _PROJECT_ROOT / "docs/_static/fixed_boundary_upstream_evidence.json"
)
_DEFAULT_FIXED_BOUNDARY_GEQDSK = (
    DEFAULT_OPENFUSIONTOOLKIT_ROOT / FIXED_BOUNDARY_RELATIVE_ROOT / FIXED_BOUNDARY_EQDSK
)
_FIXED_BOUNDARY_GEQDSK_EXPECTED = {
    "nr": 129,
    "nz": 129,
    "current": 7_799_300.71,
    "bcentr": 9.2,
    "rmaxis": 3.5226247,
    "zmaxis": -7.96984474e-06,
}
_FIXED_BOUNDARY_GEQDSK_TOLERANCES = {
    "current_relative": 1.0e-10,
    "bcentr_absolute": 1.0e-12,
    "axis_absolute_m": 1.0e-8,
}


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


@dataclass(frozen=True)
class CoilGreenFunctionValidation:
    """Validation metrics for the reduced free-boundary coil response."""

    n_points: int
    n_coils: int
    symmetry_error: float
    linearity_error: float
    gradient_error: float
    log_ratio_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "n_points": self.n_points,
            "n_coils": self.n_coils,
            "symmetry_error": self.symmetry_error,
            "linearity_error": self.linearity_error,
            "gradient_error": self.gradient_error,
            "log_ratio_error": self.log_ratio_error,
        }


@dataclass(frozen=True)
class CircularLoopGreenFunctionValidation:
    """Validation metrics for the circular-loop elliptic Green's function."""

    n_points: int
    n_coils: int
    elliptic_quadrature_relative_error: float
    linearity_error: float
    gradient_error: float
    quadrature_gradient_relative_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "n_points": self.n_points,
            "n_coils": self.n_coils,
            "elliptic_quadrature_relative_error": self.elliptic_quadrature_relative_error,
            "linearity_error": self.linearity_error,
            "gradient_error": self.gradient_error,
            "quadrature_gradient_relative_error": self.quadrature_gradient_relative_error,
        }


@dataclass(frozen=True)
class FreeBoundaryProfileCouplingValidation:
    """Validation metrics for coupling coil Green functions to profile iteration."""

    n_nodes: int
    n_cells: int
    n_coils: int
    boundary_error: float
    coil_linearity_relative_error: float
    current_gradient_error: float
    pressure_scale_gradient: float
    residual_final: float
    update_final: float
    psi_abs_max: float
    coil_flux_abs_max: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "n_coils": self.n_coils,
            "boundary_error": self.boundary_error,
            "coil_linearity_relative_error": self.coil_linearity_relative_error,
            "current_gradient_error": self.current_gradient_error,
            "pressure_scale_gradient": self.pressure_scale_gradient,
            "residual_final": self.residual_final,
            "update_final": self.update_final,
            "psi_abs_max": self.psi_abs_max,
            "coil_flux_abs_max": self.coil_flux_abs_max,
        }


@dataclass(frozen=True)
class FixedBoundaryGeqdskValidation:
    """Numeric diagnostics for the upstream fixed-boundary gEQDSK seed."""

    status: str
    source: str
    source_kind: str
    nr: int
    nz: int
    profile_length: int
    current_A: float
    bcentr_T: float
    rmaxis_m: float
    zmaxis_m: float
    psi_min: float
    psi_max: float
    qpsi_min: float
    qpsi_max: float
    current_relative_error: float
    bcentr_absolute_error: float
    axis_error_m: float
    shape_matches: bool
    q_positive: bool
    psi_range_positive: bool
    numeric_parity_claim: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "status": self.status,
            "source": self.source,
            "source_kind": self.source_kind,
            "nr": self.nr,
            "nz": self.nz,
            "profile_length": self.profile_length,
            "current_A": self.current_A,
            "bcentr_T": self.bcentr_T,
            "rmaxis_m": self.rmaxis_m,
            "zmaxis_m": self.zmaxis_m,
            "psi_min": self.psi_min,
            "psi_max": self.psi_max,
            "qpsi_min": self.qpsi_min,
            "qpsi_max": self.qpsi_max,
            "current_relative_error": self.current_relative_error,
            "bcentr_absolute_error": self.bcentr_absolute_error,
            "axis_error_m": self.axis_error_m,
            "shape_matches": self.shape_matches,
            "q_positive": self.q_positive,
            "psi_range_positive": self.psi_range_positive,
            "numeric_parity_claim": self.numeric_parity_claim,
            "tolerances": dict(_FIXED_BOUNDARY_GEQDSK_TOLERANCES),
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


def run_coil_green_function_validation() -> CoilGreenFunctionValidation:
    """Run analytic checks for the reduced free-boundary coil Green's function."""

    coils = (
        CoilConfig(name="PF_A", r=1.5, z=0.0, current=2.0, sigma=0.05),
        CoilConfig(name="PF_B", r=2.2, z=0.35, current=-0.75, sigma=0.08),
    )
    points = jnp.asarray([[1.8, 0.2], [1.8, -0.2], [2.4, 0.1], [1.2, -0.35]], dtype=jnp.float64)
    symmetric_coil = (CoilConfig(name="PF", r=1.5, z=0.0, current=1.0, sigma=0.05),)
    symmetric_flux = coil_flux(points[:2], symmetric_coil)
    symmetry_error = float(jnp.abs(symmetric_flux[0] - symmetric_flux[1]))

    response = coil_response_matrix(points, coils)
    currents = jnp.asarray([coil.current for coil in coils], dtype=jnp.float64)
    linearity_error = float(jnp.linalg.norm(response @ currents - coil_flux(points, coils)))

    derivative_point = jnp.asarray([1.85, 0.18], dtype=jnp.float64)
    ad_gradient = jax.grad(lambda point: coil_flux(point[None, :], symmetric_coil)[0])(
        derivative_point
    )
    analytic_gradient = coil_flux_gradient(derivative_point[None, :], symmetric_coil)[0]
    gradient_error = float(jnp.linalg.norm(ad_gradient - analytic_gradient))

    d1 = 0.3
    d2 = 0.6
    core = symmetric_coil[0].sigma
    ratio_points = jnp.asarray([[1.5 + d1, 0.0], [1.5 + d2, 0.0]], dtype=jnp.float64)
    values = regularized_log_green_function(
        ratio_points,
        symmetric_coil[0].r,
        symmetric_coil[0].z,
        core_radius=core,
    )
    expected_difference = -MU0 / (4.0 * jnp.pi) * jnp.log((d2**2 + core**2) / (d1**2 + core**2))
    log_ratio_error = float(jnp.abs((values[1] - values[0]) - expected_difference))

    return CoilGreenFunctionValidation(
        n_points=int(points.shape[0]),
        n_coils=len(coils),
        symmetry_error=symmetry_error,
        linearity_error=linearity_error,
        gradient_error=gradient_error,
        log_ratio_error=log_ratio_error,
    )


def run_circular_loop_green_function_validation() -> CircularLoopGreenFunctionValidation:
    """Run closed-form circular-loop checks against quadrature references."""

    coils = (
        CoilConfig(name="PF_A", r=1.52, z=0.03, current=1.3, sigma=0.015),
        CoilConfig(name="PF_B", r=2.25, z=-0.28, current=-0.7, sigma=0.02),
    )
    points = jnp.asarray([[1.72, 0.18], [2.05, -0.22], [1.35, 0.31]], dtype=jnp.float64)
    elliptic_response = circular_loop_elliptic_response_matrix(points, coils)
    quadrature_response = circular_loop_response_matrix(points, coils, n_phi=2048)
    response_norm = jnp.linalg.norm(quadrature_response)
    elliptic_quadrature_relative_error = float(
        jnp.linalg.norm(elliptic_response - quadrature_response) / response_norm
    )

    currents = jnp.asarray([coil.current for coil in coils], dtype=jnp.float64)
    linearity_error = float(
        jnp.linalg.norm(
            circular_loop_elliptic_coil_flux(points, coils) - elliptic_response @ currents
        )
    )

    coil = coils[0]
    point = jnp.asarray([1.83, 0.21], dtype=jnp.float64)
    ad_gradient = jax.grad(
        lambda x: circular_loop_elliptic_flux(
            x[None, :],
            coil.r,
            coil.z,
            core_radius=coil.sigma,
        )[0]
    )(point)
    elliptic_gradient = circular_loop_elliptic_flux_gradient(
        point[None, :],
        coil.r,
        coil.z,
        core_radius=coil.sigma,
    )[0]
    quadrature_gradient = circular_loop_flux_gradient(
        point[None, :],
        coil.r,
        coil.z,
        core_radius=coil.sigma,
        n_phi=2048,
    )[0]
    gradient_error = float(jnp.linalg.norm(ad_gradient - elliptic_gradient))
    quadrature_gradient_relative_error = float(
        jnp.linalg.norm(elliptic_gradient - quadrature_gradient)
        / jnp.linalg.norm(quadrature_gradient)
    )

    return CircularLoopGreenFunctionValidation(
        n_points=int(points.shape[0]),
        n_coils=len(coils),
        elliptic_quadrature_relative_error=elliptic_quadrature_relative_error,
        linearity_error=linearity_error,
        gradient_error=gradient_error,
        quadrature_gradient_relative_error=quadrature_gradient_relative_error,
    )


def run_free_boundary_profile_coupling_validation() -> FreeBoundaryProfileCouplingValidation:
    """Run a compact coil-boundary plus nonlinear-profile coupling gate."""

    from tokamaker_jax.fem_equilibrium import (
        NonlinearProfileParameters,
        PowerProfile,
        solve_profile_iteration,
    )

    nodes, triangles = rectangular_triangles(1.0, 2.0, -0.5, 0.5, 4)
    coils = (
        CoilConfig(name="PF_U", r=2.28, z=0.72, current=1.25e5, sigma=0.05),
        CoilConfig(name="PF_L", r=2.28, z=-0.72, current=1.25e5, sigma=0.05),
        CoilConfig(name="PF_C", r=0.72, z=0.0, current=-0.75e5, sigma=0.06),
    )
    response = circular_loop_elliptic_response_matrix(nodes, coils)
    currents = jnp.asarray([coil.current for coil in coils], dtype=jnp.float64)
    coil_flux_from_response = response @ currents
    coil_flux_direct = circular_loop_elliptic_coil_flux(nodes, coils)
    coil_linearity_relative_error = jnp.linalg.norm(coil_flux_from_response - coil_flux_direct) / (
        jnp.linalg.norm(coil_flux_direct) + 1.0e-30
    )

    boundary_nodes = boundary_nodes_from_coordinates(nodes)
    boundary_values = coil_flux_direct[boundary_nodes]
    parameters = NonlinearProfileParameters(
        pressure=PowerProfile(scale=2.5, alpha=1.0, gamma=1.0),
        ffprime=PowerProfile(scale=-0.05, alpha=1.0, gamma=1.0),
    )
    solution = solve_profile_iteration(
        nodes,
        triangles,
        parameters,
        iterations=3,
        relaxation=0.75,
        initial_psi=coil_flux_direct,
        dirichlet_nodes=boundary_nodes,
        dirichlet_values=boundary_values,
    )
    boundary_error = jnp.max(jnp.abs(solution.psi[boundary_nodes] - boundary_values))

    def current_objective(current_vector: jnp.ndarray) -> jnp.ndarray:
        flux = response @ current_vector
        return jnp.mean(flux**2)

    current_gradient = jax.grad(current_objective)(currents)
    current_gradient_oracle = 2.0 * response.T @ (response @ currents) / nodes.shape[0]
    current_gradient_error = jnp.linalg.norm(current_gradient - current_gradient_oracle)

    def pressure_objective(scale: jnp.ndarray) -> jnp.ndarray:
        pressure_parameters = NonlinearProfileParameters(
            pressure=PowerProfile(scale=scale, alpha=1.0, gamma=1.0),
            ffprime=PowerProfile(scale=-0.05, alpha=1.0, gamma=1.0),
        )
        return jnp.mean(
            solve_profile_iteration(
                nodes,
                triangles,
                pressure_parameters,
                iterations=2,
                relaxation=0.75,
                initial_psi=coil_flux_direct,
                dirichlet_nodes=boundary_nodes,
                dirichlet_values=boundary_values,
            ).psi
            ** 2
        )

    pressure_scale_gradient = jax.grad(pressure_objective)(jnp.asarray(2.5, dtype=jnp.float64))

    return FreeBoundaryProfileCouplingValidation(
        n_nodes=int(nodes.shape[0]),
        n_cells=int(triangles.shape[0]),
        n_coils=len(coils),
        boundary_error=float(boundary_error),
        coil_linearity_relative_error=float(coil_linearity_relative_error),
        current_gradient_error=float(current_gradient_error),
        pressure_scale_gradient=float(pressure_scale_gradient),
        residual_final=float(solution.residual_history[-1]),
        update_final=float(solution.update_history[-1]),
        psi_abs_max=float(jnp.max(jnp.abs(solution.psi))),
        coil_flux_abs_max=float(jnp.max(jnp.abs(coil_flux_direct))),
    )


def run_fixed_boundary_geqdsk_validation(
    source: str | Path | None = None,
) -> FixedBoundaryGeqdskValidation:
    """Validate fixed-boundary gEQDSK source diagnostics against explicit tolerances.

    The default path is CI-safe: it reads the committed docs artifact. When a
    local OpenFUSIONToolkit checkout is present, callers can pass the direct
    ``gNT_example`` path to validate parser output from the source file.
    """

    metrics = _load_fixed_boundary_geqdsk_metrics(source)
    expected = _FIXED_BOUNDARY_GEQDSK_EXPECTED
    tolerances = _FIXED_BOUNDARY_GEQDSK_TOLERANCES

    nr = int(metrics["nr"])
    nz = int(metrics["nz"])
    current = float(metrics["current"])
    bcentr = float(metrics["bcentr"])
    rmaxis = float(metrics["rmaxis"])
    zmaxis = float(metrics["zmaxis"])
    psi_min = float(metrics["psi_min"])
    psi_max = float(metrics["psi_max"])
    qpsi_min = float(metrics["qpsi_min"])
    qpsi_max = float(metrics["qpsi_max"])
    profile_length = int(metrics["profile_length"])

    current_relative_error = abs(current - expected["current"]) / abs(expected["current"])
    bcentr_absolute_error = abs(bcentr - expected["bcentr"])
    axis_error_m = float(
        jnp.linalg.norm(
            jnp.asarray(
                [rmaxis - expected["rmaxis"], zmaxis - expected["zmaxis"]],
                dtype=jnp.float64,
            )
        )
    )
    shape_matches = (nr, nz, profile_length) == (
        expected["nr"],
        expected["nz"],
        expected["nr"],
    )
    q_positive = qpsi_min > 0.0 and qpsi_max > qpsi_min
    psi_range_positive = psi_max > psi_min
    passed = (
        shape_matches
        and q_positive
        and psi_range_positive
        and current_relative_error <= tolerances["current_relative"]
        and bcentr_absolute_error <= tolerances["bcentr_absolute"]
        and axis_error_m <= tolerances["axis_absolute_m"]
    )
    return FixedBoundaryGeqdskValidation(
        status="pass" if passed else "fail",
        source=str(metrics["source"]),
        source_kind=str(metrics["source_kind"]),
        nr=nr,
        nz=nz,
        profile_length=profile_length,
        current_A=current,
        bcentr_T=bcentr,
        rmaxis_m=rmaxis,
        zmaxis_m=zmaxis,
        psi_min=psi_min,
        psi_max=psi_max,
        qpsi_min=qpsi_min,
        qpsi_max=qpsi_max,
        current_relative_error=float(current_relative_error),
        bcentr_absolute_error=float(bcentr_absolute_error),
        axis_error_m=axis_error_m,
        shape_matches=shape_matches,
        q_positive=q_positive,
        psi_range_positive=psi_range_positive,
        numeric_parity_claim=False,
    )


def _load_fixed_boundary_geqdsk_metrics(source: str | Path | None) -> dict[str, Any]:
    if source is None:
        if _DEFAULT_FIXED_BOUNDARY_ARTIFACT.exists():
            return _geqdsk_metrics_from_artifact(_DEFAULT_FIXED_BOUNDARY_ARTIFACT)
        if _DEFAULT_FIXED_BOUNDARY_GEQDSK.exists():
            return _geqdsk_metrics_from_file(_DEFAULT_FIXED_BOUNDARY_GEQDSK)
        raise FileNotFoundError(
            "No fixed-boundary gEQDSK source available; pass a gEQDSK file or "
            "a fixed_boundary_upstream_evidence.json artifact."
        )

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        return _geqdsk_metrics_from_artifact(path)
    return _geqdsk_metrics_from_file(path)


def _geqdsk_metrics_from_artifact(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    geqdsk = report.get("geqdsk")
    if not isinstance(geqdsk, dict):
        raise ValueError(f"artifact does not contain a geqdsk object: {path}")
    return {
        "source": str(path),
        "source_kind": "committed_artifact",
        "nr": geqdsk["nr"],
        "nz": geqdsk["nz"],
        "profile_length": geqdsk["profile_length"],
        "current": geqdsk["current"],
        "bcentr": geqdsk["bcentr"],
        "rmaxis": geqdsk["rmaxis"],
        "zmaxis": geqdsk["zmaxis"],
        "psi_min": geqdsk["psi_min"],
        "psi_max": geqdsk["psi_max"],
        "qpsi_min": geqdsk["qpsi_min"],
        "qpsi_max": geqdsk["qpsi_max"],
    }


def _geqdsk_metrics_from_file(path: Path) -> dict[str, Any]:
    parsed = parse_geqdsk(path)
    summary = summarize_geqdsk(path, root=path.parent)
    return {
        "source": str(path),
        "source_kind": "geqdsk_file",
        "nr": parsed["nr"],
        "nz": parsed["nz"],
        "profile_length": int(parsed["fpol"].shape[0]),
        "current": parsed["current"],
        "bcentr": parsed["bcentr"],
        "rmaxis": parsed["rmaxis"],
        "zmaxis": parsed["zmaxis"],
        "psi_min": summary["psi_min"],
        "psi_max": summary["psi_max"],
        "qpsi_min": summary["qpsi_min"],
        "qpsi_max": summary["qpsi_max"],
    }


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
