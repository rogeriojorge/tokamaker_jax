import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.config import CoilConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.free_boundary import (
    circular_loop_coil_flux,
    circular_loop_coil_flux_gradient,
    circular_loop_elliptic_coil_flux,
    circular_loop_elliptic_coil_flux_gradient,
    circular_loop_elliptic_flux,
    circular_loop_elliptic_flux_gradient,
    circular_loop_elliptic_response_matrix,
    circular_loop_elliptic_vector_potential,
    circular_loop_flux,
    circular_loop_flux_gradient,
    circular_loop_response_matrix,
    circular_loop_vector_potential,
    coil_field,
    coil_flux,
    coil_flux_gradient,
    coil_flux_on_grid,
    coil_response_matrix,
    complete_elliptic_integrals_agm,
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


def test_circular_loop_kernel_is_symmetric_about_coil_midplane():
    points = jnp.asarray([[1.8, 0.24], [1.8, -0.24]], dtype=jnp.float64)

    vector_potential = circular_loop_vector_potential(points, 1.55, 0.0, core_radius=0.02, n_phi=96)
    flux = circular_loop_flux(points, 1.55, 0.0, core_radius=0.02, n_phi=96)
    gradient = circular_loop_flux_gradient(points, 1.55, 0.0, core_radius=0.02, n_phi=96)

    np.testing.assert_allclose(vector_potential[0], vector_potential[1], rtol=1.0e-12)
    np.testing.assert_allclose(flux[0], flux[1], rtol=1.0e-12)
    np.testing.assert_allclose(gradient[0, 0], gradient[1, 0], rtol=1.0e-12)
    np.testing.assert_allclose(gradient[0, 1], -gradient[1, 1], rtol=1.0e-12)


def test_circular_loop_response_matrix_is_linear_in_currents():
    coils = sample_coils()
    points = jnp.asarray([[1.8, 0.2], [2.4, -0.1], [1.2, -0.35]], dtype=jnp.float64)
    matrix = circular_loop_response_matrix(points, coils, n_phi=64)
    currents = jnp.asarray([coil.current for coil in coils], dtype=jnp.float64)

    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(
        circular_loop_coil_flux(points, coils, n_phi=64),
        matrix @ currents,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        circular_loop_coil_flux(points, (), n_phi=64), np.zeros(3), atol=1.0e-18
    )
    assert circular_loop_response_matrix(points, (), n_phi=64).shape == (3, 0)


def test_circular_loop_quadrature_converges_to_high_resolution_reference():
    coils = (
        CoilConfig(name="PF_A", r=1.52, z=0.03, current=1.3, sigma=0.015),
        CoilConfig(name="PF_B", r=2.25, z=-0.28, current=-0.7, sigma=0.02),
    )
    points = jnp.asarray([[1.72, 0.18], [2.05, -0.22], [1.35, 0.31]], dtype=jnp.float64)

    reference = circular_loop_response_matrix(points, coils, n_phi=512)
    coarse = circular_loop_response_matrix(points, coils, n_phi=16)
    fine = circular_loop_response_matrix(points, coils, n_phi=64)
    coarse_error = jnp.linalg.norm(coarse - reference)
    fine_error = jnp.linalg.norm(fine - reference)

    assert fine_error < 0.02 * coarse_error
    np.testing.assert_allclose(fine, reference, rtol=2.0e-3, atol=1.0e-12)


def test_complete_elliptic_integrals_agm_matches_reference_values():
    parameters = jnp.asarray([0.0, 0.5, 0.9], dtype=jnp.float64)
    complete_first, complete_second = complete_elliptic_integrals_agm(parameters)

    np.testing.assert_allclose(
        complete_first,
        [np.pi / 2.0, 1.8540746773013719, 2.5780921133481733],
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        complete_second,
        [np.pi / 2.0, 1.350643881047675, 1.1047747327040733],
        rtol=1.0e-13,
    )

    with pytest.raises(ValueError, match="elliptic parameter"):
        complete_elliptic_integrals_agm(1.0)
    with pytest.raises(ValueError, match="iterations"):
        complete_elliptic_integrals_agm(0.5, iterations=0)


def test_circular_loop_elliptic_kernel_matches_high_resolution_quadrature():
    coils = (
        CoilConfig(name="PF_A", r=1.52, z=0.03, current=1.3, sigma=0.015),
        CoilConfig(name="PF_B", r=2.25, z=-0.28, current=-0.7, sigma=0.02),
    )
    points = jnp.asarray([[1.72, 0.18], [2.05, -0.22], [1.35, 0.31]], dtype=jnp.float64)

    elliptic = circular_loop_elliptic_response_matrix(points, coils)
    reference = circular_loop_response_matrix(points, coils, n_phi=2048)

    np.testing.assert_allclose(elliptic, reference, rtol=2.0e-11, atol=1.0e-14)
    np.testing.assert_allclose(
        circular_loop_elliptic_coil_flux(points, coils),
        elliptic @ jnp.asarray([coil.current for coil in coils]),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(circular_loop_elliptic_coil_flux(points, ()), np.zeros(3))
    assert circular_loop_elliptic_response_matrix(points, ()).shape == (3, 0)


def test_circular_loop_elliptic_vector_potential_and_gradient_match_quadrature():
    point = jnp.asarray([1.83, 0.21], dtype=jnp.float64)
    coil = CoilConfig(name="PF", r=1.55, z=0.02, current=2.5, sigma=0.03)

    elliptic_vector_potential = circular_loop_elliptic_vector_potential(
        point[None, :],
        coil.r,
        coil.z,
        core_radius=coil.sigma,
    )[0]
    quadrature_vector_potential = circular_loop_vector_potential(
        point[None, :],
        coil.r,
        coil.z,
        core_radius=coil.sigma,
        n_phi=2048,
    )[0]
    elliptic_flux = circular_loop_elliptic_flux(
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
    elliptic_gradient = circular_loop_elliptic_flux_gradient(
        point[None, :],
        coil.r,
        coil.z,
        core_radius=coil.sigma,
    )[0]
    ad_gradient = jax.grad(
        lambda x: circular_loop_elliptic_flux(
            x[None, :],
            coil.r,
            coil.z,
            core_radius=coil.sigma,
        )[0]
    )(point)

    np.testing.assert_allclose(elliptic_vector_potential, quadrature_vector_potential, rtol=2.0e-11)
    np.testing.assert_allclose(elliptic_flux, point[0] * elliptic_vector_potential)
    np.testing.assert_allclose(elliptic_gradient, quadrature_gradient, rtol=3.0e-9, atol=1.0e-14)
    np.testing.assert_allclose(elliptic_gradient, ad_gradient, rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(
        circular_loop_elliptic_coil_flux_gradient(point[None, :], (coil,))[0],
        coil.current * elliptic_gradient,
        rtol=1.0e-10,
    )


def test_circular_loop_gradient_matches_jax_ad_and_is_finite():
    point = jnp.asarray([1.83, 0.21], dtype=jnp.float64)
    coil = CoilConfig(name="PF", r=1.55, z=0.02, current=2.5, sigma=0.03)

    ad_gradient = jax.grad(
        lambda x: circular_loop_flux(x[None, :], coil.r, coil.z, core_radius=coil.sigma, n_phi=96)[
            0
        ]
    )(point)
    quadrature_gradient = circular_loop_flux_gradient(
        point[None, :], coil.r, coil.z, core_radius=coil.sigma, n_phi=96
    )[0]
    total_gradient = circular_loop_coil_flux_gradient(point[None, :], (coil,), n_phi=96)[0]

    assert bool(jnp.all(jnp.isfinite(ad_gradient)))
    assert bool(jnp.all(jnp.isfinite(quadrature_gradient)))
    np.testing.assert_allclose(quadrature_gradient, ad_gradient, rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(total_gradient, coil.current * quadrature_gradient)


def test_free_boundary_helpers_validate_inputs():
    with pytest.raises(ValueError, match="points must have shape"):
        coil_flux(jnp.ones((2, 3)), sample_coils())
    with pytest.raises(ValueError, match="reference_radius"):
        regularized_log_green_function(jnp.ones((2, 2)), 1.0, 0.0, reference_radius=0.0)
    with pytest.raises(ValueError, match="core_radius"):
        regularized_log_green_gradient(jnp.ones((2, 2)), 1.0, 0.0, core_radius=-1.0)
    with pytest.raises(ValueError, match="n_phi"):
        circular_loop_flux(jnp.ones((2, 2)), 1.0, 0.0, n_phi=4)
