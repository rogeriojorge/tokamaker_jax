"""Reduced free-boundary coil Green's-function helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tokamaker_jax.config import CoilConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.profiles import MU0


@dataclass(frozen=True)
class CoilGreenFunctionResponse:
    """Green's-function response for a set of observation points and coils."""

    points: jnp.ndarray
    coil_names: tuple[str, ...]
    coil_positions: jnp.ndarray
    coil_currents: jnp.ndarray
    response_per_amp: jnp.ndarray
    flux: jnp.ndarray
    gradient: jnp.ndarray
    field: jnp.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "points": jnp.asarray(self.points).tolist(),
            "coil_names": list(self.coil_names),
            "coil_positions": jnp.asarray(self.coil_positions).tolist(),
            "coil_currents": jnp.asarray(self.coil_currents).tolist(),
            "response_per_amp": jnp.asarray(self.response_per_amp).tolist(),
            "flux": jnp.asarray(self.flux).tolist(),
            "gradient": jnp.asarray(self.gradient).tolist(),
            "field": jnp.asarray(self.field).tolist(),
        }


def regularized_log_green_function(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    reference_radius: float = 1.0,
) -> jnp.ndarray:
    """Return the reduced 2D free-space Green's function per ampere.

    This is the large-aspect-ratio logarithmic fixture

    ``G = -mu0/(4 pi) log(((R-Rc)^2 + (Z-Zc)^2 + eps^2) / a_ref^2)``.

    It is not the full circular-filament axisymmetric Green's function used by
    production TokaMaker; it is a differentiable validation fixture for early
    free-boundary coupling tests.
    """

    if reference_radius <= 0.0:
        raise ValueError("reference_radius must be positive")
    scalar_input = (
        jnp.ndim(jnp.asarray(coil_r)) == 0
        and jnp.ndim(jnp.asarray(coil_z)) == 0
        and jnp.ndim(jnp.asarray(core_radius)) == 0
    )
    if isinstance(core_radius, (int, float)) and core_radius < 0.0:
        raise ValueError("core_radius must be nonnegative")
    points = _as_points(points)
    delta_r = points[:, 0, None] - jnp.asarray(coil_r, dtype=points.dtype)
    delta_z = points[:, 1, None] - jnp.asarray(coil_z, dtype=points.dtype)
    core = jnp.asarray(core_radius, dtype=points.dtype)
    squared_radius = delta_r**2 + delta_z**2 + core**2
    values = -MU0 / (4.0 * jnp.pi) * jnp.log(squared_radius / reference_radius**2)
    return values[:, 0] if scalar_input else values


def regularized_log_green_gradient(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Return ``(dG/dR, dG/dZ)`` per ampere for the reduced Green's function."""

    scalar_input = (
        jnp.ndim(jnp.asarray(coil_r)) == 0
        and jnp.ndim(jnp.asarray(coil_z)) == 0
        and jnp.ndim(jnp.asarray(core_radius)) == 0
    )
    if isinstance(core_radius, (int, float)) and core_radius < 0.0:
        raise ValueError("core_radius must be nonnegative")
    points = _as_points(points)
    delta_r = points[:, 0, None] - jnp.asarray(coil_r, dtype=points.dtype)
    delta_z = points[:, 1, None] - jnp.asarray(coil_z, dtype=points.dtype)
    core = jnp.asarray(core_radius, dtype=points.dtype)
    squared_radius = delta_r**2 + delta_z**2 + core**2
    factor = -MU0 / (2.0 * jnp.pi) / squared_radius
    values = jnp.stack((factor * delta_r, factor * delta_z), axis=-1)
    return values[:, 0, :] if scalar_input else values


def coil_response_matrix(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    reference_radius: float = 1.0,
) -> jnp.ndarray:
    """Return the point-by-coil Green's matrix per ampere."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["r"].size == 0:
        return jnp.zeros((points.shape[0], 0), dtype=points.dtype)
    return regularized_log_green_function(
        points,
        arrays["r"][None, :],
        arrays["z"][None, :],
        core_radius=arrays["core"][None, :],
        reference_radius=reference_radius,
    )


def coil_flux(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    reference_radius: float = 1.0,
) -> jnp.ndarray:
    """Return the reduced free-boundary flux from all coils at each point."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0],), dtype=points.dtype)
    response = coil_response_matrix(points, coils, reference_radius=reference_radius)
    return response @ arrays["current"]


def coil_flux_gradient(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
) -> jnp.ndarray:
    """Return ``(dpsi/dR, dpsi/dZ)`` from all reduced Green's-function coils."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0], 2), dtype=points.dtype)
    gradients = regularized_log_green_gradient(
        points,
        arrays["r"][None, :],
        arrays["z"][None, :],
        core_radius=arrays["core"][None, :],
    )
    return jnp.einsum("ncd,c->nd", gradients, arrays["current"])


def coil_field(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
) -> jnp.ndarray:
    """Return reduced poloidal field components ``(B_R, B_Z)`` from coil flux."""

    points = _as_points(points)
    gradient = coil_flux_gradient(points, coils)
    radius = points[:, 0]
    return jnp.column_stack((-gradient[:, 1] / radius, gradient[:, 0] / radius))


def evaluate_coil_green_response(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    reference_radius: float = 1.0,
) -> CoilGreenFunctionResponse:
    """Evaluate response matrix, total flux, gradient, and field at points."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    response = coil_response_matrix(points, coils, reference_radius=reference_radius)
    flux = (
        response @ arrays["current"]
        if arrays["current"].size
        else jnp.zeros((points.shape[0],), dtype=points.dtype)
    )
    gradient = coil_flux_gradient(points, coils)
    return CoilGreenFunctionResponse(
        points=points,
        coil_names=tuple(coil.name for coil in coils),
        coil_positions=jnp.column_stack((arrays["r"], arrays["z"])),
        coil_currents=arrays["current"],
        response_per_amp=response,
        flux=flux,
        gradient=gradient,
        field=coil_field(points, coils),
    )


def coil_flux_on_grid(
    grid: RectangularGrid,
    coils: tuple[CoilConfig, ...],
    *,
    dtype: jnp.dtype = jnp.float64,
    reference_radius: float = 1.0,
) -> jnp.ndarray:
    """Return reduced coil flux on a rectangular grid."""

    r, z = grid.mesh(dtype=dtype)
    points = jnp.column_stack((r.reshape(-1), z.reshape(-1)))
    return coil_flux(points, coils, reference_radius=reference_radius).reshape(grid.nr, grid.nz)


def _as_points(points: jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(points, dtype=jnp.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    return array


def _coil_arrays(coils: tuple[CoilConfig, ...], *, dtype: jnp.dtype) -> dict[str, jnp.ndarray]:
    return {
        "r": jnp.asarray([coil.r for coil in coils], dtype=dtype),
        "z": jnp.asarray([coil.z for coil in coils], dtype=dtype),
        "current": jnp.asarray([coil.current for coil in coils], dtype=dtype),
        "core": jnp.asarray([coil.sigma for coil in coils], dtype=dtype),
    }
