"""Free-boundary coil Green's-function helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
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


def circular_loop_vector_potential(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return circular-filament toroidal vector potential ``A_phi`` per ampere.

    The prototype integrates the Biot-Savart kernel for a circular toroidal
    current loop with a fixed midpoint quadrature over source toroidal angle.
    It intentionally avoids SciPy elliptic-integral calls so the path remains
    differentiable and JAX-transformable. ``core_radius`` is an optional
    softening length for near-filament tests; set it to zero for the ideal
    circular filament away from the source point.
    """

    points = _as_points(points)
    n_phi = _validate_toroidal_quadrature(n_phi)
    (
        observation_r,
        observation_z,
        source_r,
        source_z,
        core,
        scalar_input,
    ) = _circular_loop_geometry(points, coil_r, coil_z, core_radius)
    cos_phi, dphi = _toroidal_midpoint_cosines(n_phi, dtype=points.dtype)
    cos_phi = cos_phi.reshape((1,) * source_r.ndim + (n_phi,))
    distance = _circular_loop_distance(
        observation_r, observation_z, source_r, source_z, core, cos_phi
    )
    integral = dphi * jnp.sum(cos_phi / distance, axis=-1)
    values = MU0 / (4.0 * jnp.pi) * source_r * integral
    return values[:, 0] if scalar_input else values


def circular_loop_flux(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return circular-loop flux function ``psi = R A_phi`` per ampere."""

    points = _as_points(points)
    vector_potential = circular_loop_vector_potential(
        points, coil_r, coil_z, core_radius=core_radius, n_phi=n_phi
    )
    radius_shape = (points.shape[0],) + (1,) * (vector_potential.ndim - 1)
    return points[:, 0].reshape(radius_shape) * vector_potential


def circular_loop_flux_gradient(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return ``(dpsi/dR, dpsi/dZ)`` per ampere for circular-loop flux."""

    points = _as_points(points)
    n_phi = _validate_toroidal_quadrature(n_phi)
    (
        observation_r,
        observation_z,
        source_r,
        source_z,
        core,
        scalar_input,
    ) = _circular_loop_geometry(points, coil_r, coil_z, core_radius)
    cos_phi, dphi = _toroidal_midpoint_cosines(n_phi, dtype=points.dtype)
    cos_phi = cos_phi.reshape((1,) * source_r.ndim + (n_phi,))
    distance = _circular_loop_distance(
        observation_r, observation_z, source_r, source_z, core, cos_phi
    )
    inverse_distance = 1.0 / distance
    inverse_distance_cubed = inverse_distance**3
    delta_z = observation_z - source_z

    integral = dphi * jnp.sum(cos_phi * inverse_distance, axis=-1)
    d_integral_d_r = dphi * jnp.sum(
        -cos_phi
        * (observation_r[..., None] - source_r[..., None] * cos_phi)
        * inverse_distance_cubed,
        axis=-1,
    )
    d_integral_d_z = dphi * jnp.sum(
        -cos_phi * delta_z[..., None] * inverse_distance_cubed,
        axis=-1,
    )

    factor = MU0 / (4.0 * jnp.pi) * source_r
    dpsi_d_r = factor * (integral + observation_r * d_integral_d_r)
    dpsi_d_z = factor * observation_r * d_integral_d_z
    values = jnp.stack((dpsi_d_r, dpsi_d_z), axis=-1)
    return values[:, 0, :] if scalar_input else values


def complete_elliptic_integrals_agm(
    parameter: jnp.ndarray | float,
    *,
    iterations: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return complete elliptic integrals ``K(m)`` and ``E(m)``.

    The implementation uses a fixed-iteration arithmetic-geometric mean
    formula so it remains JAX-transformable and does not depend on SciPy's
    elliptic special functions. ``parameter`` is the elliptic parameter
    ``m = k**2``.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if isinstance(parameter, (int, float)) and not 0.0 <= parameter < 1.0:
        raise ValueError("elliptic parameter must satisfy 0 <= m < 1")
    parameter_array = jnp.asarray(parameter, dtype=jnp.float64)
    upper = 1.0 - jnp.finfo(parameter_array.dtype).eps
    m = jnp.clip(parameter_array, 0.0, upper)
    a = jnp.ones_like(m)
    b = jnp.sqrt(1.0 - m)
    correction = 0.5 * m
    weight = 1.0
    for _ in range(iterations):
        c = 0.5 * (a - b)
        correction = correction + weight * c**2
        next_a = 0.5 * (a + b)
        b = jnp.sqrt(a * b)
        a = next_a
        weight *= 2.0
    complete_first = jnp.pi / (2.0 * a)
    complete_second = complete_first * (1.0 - correction)
    return complete_first, complete_second


def circular_loop_elliptic_vector_potential(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return closed-form circular-loop ``A_phi`` per ampere.

    This evaluates the complete-elliptic-integral form of the same circular
    filament response used by :func:`circular_loop_vector_potential`. A nonzero
    ``core_radius`` enters as a smooth distance softening in the same way as the
    quadrature reference.
    """

    points = _as_points(points)
    (
        observation_r,
        _observation_z,
        source_r,
        _source_z,
        _core,
        scalar_input,
        parameter,
    ) = _circular_loop_elliptic_geometry(points, coil_r, coil_z, core_radius)
    complete_first, complete_second = complete_elliptic_integrals_agm(
        parameter,
        iterations=elliptic_iterations,
    )
    modulus = jnp.sqrt(parameter)
    bracket = (1.0 - 0.5 * parameter) * complete_first - complete_second
    values = MU0 / (jnp.pi * modulus) * jnp.sqrt(source_r / observation_r) * bracket
    return values[:, 0] if scalar_input else values


def circular_loop_elliptic_flux(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return closed-form circular-loop flux ``psi = R A_phi`` per ampere."""

    points = _as_points(points)
    vector_potential = circular_loop_elliptic_vector_potential(
        points,
        coil_r,
        coil_z,
        core_radius=core_radius,
        elliptic_iterations=elliptic_iterations,
    )
    radius_shape = (points.shape[0],) + (1,) * (vector_potential.ndim - 1)
    return points[:, 0].reshape(radius_shape) * vector_potential


def circular_loop_elliptic_flux_gradient(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    *,
    core_radius: jnp.ndarray | float = 0.0,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return ``(dpsi/dR, dpsi/dZ)`` for the closed-form circular-loop flux."""

    points = _as_points(points)
    scalar_input = (
        jnp.ndim(jnp.asarray(coil_r)) == 0
        and jnp.ndim(jnp.asarray(coil_z)) == 0
        and jnp.ndim(jnp.asarray(core_radius)) == 0
    )
    arrays = jnp.broadcast_arrays(
        jnp.asarray(coil_r, dtype=points.dtype),
        jnp.asarray(coil_z, dtype=points.dtype),
        jnp.asarray(core_radius, dtype=points.dtype),
    )
    source_r, source_z, core = (array.reshape(-1) for array in arrays)

    def scalar_flux(
        point: jnp.ndarray,
        radius: jnp.ndarray,
        height: jnp.ndarray,
        softening: jnp.ndarray,
    ) -> jnp.ndarray:
        return circular_loop_elliptic_flux(
            point[None, :],
            radius,
            height,
            core_radius=softening,
            elliptic_iterations=elliptic_iterations,
        )[0]

    gradient_one_coil = jax.grad(scalar_flux, argnums=0)
    gradients = jax.vmap(
        lambda point: jax.vmap(
            lambda radius, height, softening: gradient_one_coil(
                point,
                radius,
                height,
                softening,
            )
        )(source_r, source_z, core)
    )(points)
    return gradients[:, 0, :] if scalar_input else gradients


def circular_loop_elliptic_response_matrix(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return the closed-form circular-loop point-by-coil flux matrix."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["r"].size == 0:
        return jnp.zeros((points.shape[0], 0), dtype=points.dtype)
    return circular_loop_elliptic_flux(
        points,
        arrays["r"],
        arrays["z"],
        core_radius=arrays["core"],
        elliptic_iterations=elliptic_iterations,
    )


def circular_loop_elliptic_coil_flux(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return total closed-form circular-loop coil flux at each point."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0],), dtype=points.dtype)
    response = circular_loop_elliptic_response_matrix(
        points,
        coils,
        elliptic_iterations=elliptic_iterations,
    )
    return response @ arrays["current"]


def circular_loop_elliptic_coil_flux_gradient(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    elliptic_iterations: int = 8,
) -> jnp.ndarray:
    """Return ``(dpsi/dR, dpsi/dZ)`` from all closed-form loop coils."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0], 2), dtype=points.dtype)
    gradients = circular_loop_elliptic_flux_gradient(
        points,
        arrays["r"],
        arrays["z"],
        core_radius=arrays["core"],
        elliptic_iterations=elliptic_iterations,
    )
    return jnp.einsum("ncd,c->nd", gradients, arrays["current"])


def circular_loop_response_matrix(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return the circular-loop point-by-coil flux matrix per ampere."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["r"].size == 0:
        return jnp.zeros((points.shape[0], 0), dtype=points.dtype)
    return circular_loop_flux(
        points,
        arrays["r"],
        arrays["z"],
        core_radius=arrays["core"],
        n_phi=n_phi,
    )


def circular_loop_coil_flux(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return total circular-loop coil flux at each point."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0],), dtype=points.dtype)
    response = circular_loop_response_matrix(points, coils, n_phi=n_phi)
    return response @ arrays["current"]


def circular_loop_coil_flux_gradient(
    points: jnp.ndarray,
    coils: tuple[CoilConfig, ...],
    *,
    n_phi: int = 128,
) -> jnp.ndarray:
    """Return ``(dpsi/dR, dpsi/dZ)`` from all circular-loop coils."""

    points = _as_points(points)
    arrays = _coil_arrays(coils, dtype=points.dtype)
    if arrays["current"].size == 0:
        return jnp.zeros((points.shape[0], 2), dtype=points.dtype)
    gradients = circular_loop_flux_gradient(
        points,
        arrays["r"],
        arrays["z"],
        core_radius=arrays["core"],
        n_phi=n_phi,
    )
    return jnp.einsum("ncd,c->nd", gradients, arrays["current"])


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


def _validate_toroidal_quadrature(n_phi: int) -> int:
    if not isinstance(n_phi, int):
        raise TypeError("n_phi must be a Python integer so quadrature shape is static")
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")
    return n_phi


def _toroidal_midpoint_cosines(n_phi: int, *, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    dphi = 2.0 * jnp.pi / n_phi
    phi = (jnp.arange(n_phi, dtype=dtype) + 0.5) * dphi
    return jnp.cos(phi), dphi


def _circular_loop_geometry(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    core_radius: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    scalar_input = (
        jnp.ndim(jnp.asarray(coil_r)) == 0
        and jnp.ndim(jnp.asarray(coil_z)) == 0
        and jnp.ndim(jnp.asarray(core_radius)) == 0
    )
    if isinstance(coil_r, (int, float)) and coil_r <= 0.0:
        raise ValueError("coil_r must be positive")
    if isinstance(core_radius, (int, float)) and core_radius < 0.0:
        raise ValueError("core_radius must be nonnegative")

    observation_r = points[:, 0, None]
    observation_z = points[:, 1, None]
    source_r = jnp.zeros_like(observation_r) + jnp.asarray(coil_r, dtype=points.dtype)
    source_z = jnp.zeros_like(observation_z) + jnp.asarray(coil_z, dtype=points.dtype)
    core = jnp.zeros_like(observation_r) + jnp.asarray(core_radius, dtype=points.dtype)
    broadcast = jnp.broadcast_arrays(observation_r, observation_z, source_r, source_z, core)
    return (*broadcast, scalar_input)


def _circular_loop_elliptic_geometry(
    points: jnp.ndarray,
    coil_r: jnp.ndarray | float,
    coil_z: jnp.ndarray | float,
    core_radius: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, jnp.ndarray]:
    (
        observation_r,
        observation_z,
        source_r,
        source_z,
        core,
        scalar_input,
    ) = _circular_loop_geometry(points, coil_r, coil_z, core_radius)
    separation_squared = (observation_z - source_z) ** 2 + core**2
    denominator = (observation_r + source_r) ** 2 + separation_squared
    parameter = 4.0 * observation_r * source_r / denominator
    parameter = jnp.clip(parameter, jnp.finfo(points.dtype).tiny, 1.0 - jnp.finfo(points.dtype).eps)
    return observation_r, observation_z, source_r, source_z, core, scalar_input, parameter


def _circular_loop_distance(
    observation_r: jnp.ndarray,
    observation_z: jnp.ndarray,
    source_r: jnp.ndarray,
    source_z: jnp.ndarray,
    core: jnp.ndarray,
    cos_phi: jnp.ndarray,
) -> jnp.ndarray:
    delta_z = observation_z - source_z
    distance_squared = (
        observation_r[..., None] ** 2
        + source_r[..., None] ** 2
        - 2.0 * observation_r[..., None] * source_r[..., None] * cos_phi
        + delta_z[..., None] ** 2
        + core[..., None] ** 2
    )
    return jnp.sqrt(distance_squared)


def _coil_arrays(coils: tuple[CoilConfig, ...], *, dtype: jnp.dtype) -> dict[str, jnp.ndarray]:
    return {
        "r": jnp.asarray([coil.r for coil in coils], dtype=dtype),
        "z": jnp.asarray([coil.z for coil in coils], dtype=dtype),
        "current": jnp.asarray([coil.current for coil in coils], dtype=dtype),
        "core": jnp.asarray([coil.sigma for coil in coils], dtype=dtype),
    }
