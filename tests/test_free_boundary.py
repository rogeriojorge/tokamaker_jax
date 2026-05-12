import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.config import CoilConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.free_boundary import (
    coil_field,
    coil_flux,
    coil_flux_gradient,
    coil_flux_on_grid,
    coil_response_matrix,
    evaluate_coil_green_response,
    regularized_log_green_function,
    regularized_log_green_gradient,
)
from tokamaker_jax.profiles import MU0


def sample_coils():
    return (
        CoilConfig(name="PF_A", r=1.5, z=0.0, current=2.0, sigma=0.05),
        CoilConfig(name="PF_B", r=2.2, z=0.35, current=-0.75, sigma=0.08),
    )


def test_regularized_log_green_symmetry_and_radial_ratio():
    points = jnp.asarray([[1.8, 0.2], [1.8, -0.2]], dtype=jnp.float64)
    values = regularized_log_green_function(points, 1.5, 0.0, core_radius=0.05)

    np.testing.assert_allclose(values[0], values[1], atol=1.0e-18)

    d1 = 0.3
    d2 = 0.6
    core = 0.05
    ratio_points = jnp.asarray([[1.5 + d1, 0.0], [1.5 + d2, 0.0]], dtype=jnp.float64)
    ratio_values = regularized_log_green_function(ratio_points, 1.5, 0.0, core_radius=core)
    expected = -MU0 / (4.0 * jnp.pi) * jnp.log((d2**2 + core**2) / (d1**2 + core**2))

    np.testing.assert_allclose(ratio_values[1] - ratio_values[0], expected, atol=1.0e-18)


def test_coil_response_matrix_flux_and_empty_coils():
    coils = sample_coils()
    points = jnp.asarray([[1.8, 0.2], [2.4, -0.1], [1.2, -0.35]], dtype=jnp.float64)
    matrix = coil_response_matrix(points, coils)
    currents = jnp.asarray([coil.current for coil in coils], dtype=jnp.float64)

    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(coil_flux(points, coils), matrix @ currents, atol=1.0e-18)
    np.testing.assert_allclose(coil_flux(points, ()), np.zeros(3), atol=1.0e-18)
    assert coil_response_matrix(points, ()).shape == (3, 0)


def test_coil_gradient_matches_jax_ad_and_field_convention():
    coils = (CoilConfig(name="PF", r=1.5, z=0.0, current=2.0, sigma=0.05),)
    point = jnp.asarray([1.85, 0.18], dtype=jnp.float64)

    ad_gradient = jax.grad(lambda x: coil_flux(x[None, :], coils)[0])(point)
    analytic_gradient = coil_flux_gradient(point[None, :], coils)[0]
    field = coil_field(point[None, :], coils)[0]

    np.testing.assert_allclose(analytic_gradient, ad_gradient, rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(
        field, [-analytic_gradient[1] / point[0], analytic_gradient[0] / point[0]]
    )
    np.testing.assert_allclose(
        regularized_log_green_gradient(point[None, :], 1.5, 0.0, core_radius=0.05)[0] * 2.0,
        analytic_gradient,
        rtol=1.0e-10,
        atol=1.0e-14,
    )


def test_response_report_and_grid_payload_are_json_ready():
    coils = sample_coils()
    points = jnp.asarray([[1.8, 0.2], [2.4, -0.1]], dtype=jnp.float64)
    response = evaluate_coil_green_response(points, coils)
    grid = RectangularGrid(1.0, 2.6, -0.6, 0.6, 9, 7)
    flux = coil_flux_on_grid(grid, coils)

    payload = response.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["coil_names"] == ["PF_A", "PF_B"]
    assert response.response_per_amp.shape == (2, 2)
    assert response.field.shape == (2, 2)
    assert flux.shape == (9, 7)
    assert bool(jnp.all(jnp.isfinite(flux)))


def test_free_boundary_helpers_validate_inputs():
    with pytest.raises(ValueError, match="points must have shape"):
        coil_flux(jnp.ones((2, 3)), sample_coils())
    with pytest.raises(ValueError, match="reference_radius"):
        regularized_log_green_function(jnp.ones((2, 2)), 1.0, 0.0, reference_radius=0.0)
    with pytest.raises(ValueError, match="core_radius"):
        regularized_log_green_gradient(jnp.ones((2, 2)), 1.0, 0.0, core_radius=-1.0)
