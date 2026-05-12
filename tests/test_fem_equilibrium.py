import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.fem_equilibrium import (
    NonlinearProfileParameters,
    PowerProfile,
    assemble_nonlinear_profile_load_vector,
    constant_profile_load_oracle,
    linear_nonlinear_profile_load_vector,
    nonlinear_profile_source_density,
    normalize_profile_flux,
    run_profile_iteration_validation,
    solve_profile_iteration,
    solve_profile_iteration_on_rectangle,
)
from tokamaker_jax.verification import rectangular_triangles


def small_axisymmetric_mesh():
    return rectangular_triangles(1.0, 2.0, -0.5, 0.5, 3)


def constant_parameters():
    return NonlinearProfileParameters(
        pressure=PowerProfile(scale=2.5, alpha=0.0, gamma=1.0),
        ffprime=PowerProfile(scale=0.8, alpha=0.0, gamma=1.0),
    )


def test_normalize_profile_flux_maps_to_unit_interval():
    psi = jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64)
    psin = normalize_profile_flux(psi)

    np.testing.assert_allclose(psin, [0.0, 0.5, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(normalize_profile_flux(jnp.ones(3)), np.zeros(3), atol=1.0e-12)


def test_nonlinear_profile_density_matches_constant_profile_convention():
    points = jnp.asarray([[1.0, 0.0], [2.0, 0.1]], dtype=jnp.float64)
    psin = jnp.asarray([0.0, 0.7], dtype=jnp.float64)
    parameters = constant_parameters()

    density = nonlinear_profile_source_density(points, psin, parameters)
    expected = (
        0.5 * parameters.ffprime.scale / points[:, 0]
        + (4.0e-7 * jnp.pi) * points[:, 0] * parameters.pressure.scale
    )

    np.testing.assert_allclose(density, expected, atol=1.0e-14)


def test_nonlinear_profile_load_reduces_to_existing_constant_oracle():
    nodes, triangles = small_axisymmetric_mesh()
    parameters = constant_parameters()
    psin = jnp.linspace(0.0, 1.0, nodes.shape[0], dtype=jnp.float64)

    load = assemble_nonlinear_profile_load_vector(nodes, triangles, psin, parameters)
    oracle = constant_profile_load_oracle(nodes, triangles, parameters)

    np.testing.assert_allclose(load, oracle, atol=1.0e-14)

    element_load = linear_nonlinear_profile_load_vector(
        nodes[triangles[0]], psin[triangles[0]], parameters
    )
    assert element_load.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(element_load)))


def test_profile_iteration_runs_and_exports_json_ready_diagnostics():
    solution = solve_profile_iteration_on_rectangle(
        subdivisions=4,
        parameters=constant_parameters(),
        iterations=3,
        relaxation=0.9,
    )

    payload = solution.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert solution.psi.shape == (25,)
    assert solution.rhs.shape == (25,)
    assert solution.residual_history.shape == (3,)
    assert solution.update_history.shape == (3,)
    assert solution.stats()["n_cells"] == 32
    assert bool(jnp.all(jnp.isfinite(solution.psi)))
    assert float(solution.residual_history[-1]) < float(solution.residual_history[0])


def test_profile_iteration_is_differentiable_wrt_pressure_scale():
    nodes, triangles = small_axisymmetric_mesh()

    def objective(scale):
        parameters = NonlinearProfileParameters(
            pressure=PowerProfile(scale=scale, alpha=0.0, gamma=1.0),
            ffprime=PowerProfile(scale=0.1, alpha=0.0, gamma=1.0),
        )
        solution = solve_profile_iteration(
            nodes,
            triangles,
            parameters,
            iterations=2,
            relaxation=0.8,
        )
        return jnp.mean(solution.psi**2)

    grad = jax.grad(objective)(jnp.asarray(2.0, dtype=jnp.float64))

    assert bool(jnp.isfinite(grad))
    assert float(grad) > 0.0


def test_profile_iteration_validation_schema_and_tolerances():
    result = run_profile_iteration_validation()
    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert result.n_nodes == 25
    assert result.n_cells == 32
    assert result.load_oracle_error < 1.0e-14
    assert result.residual_final < result.residual_initial
    assert result.update_final < 1.0
    assert result.pressure_gradient > 0.0


def test_profile_iteration_validates_inputs():
    nodes, triangles = small_axisymmetric_mesh()
    parameters = constant_parameters()

    with pytest.raises(ValueError, match="iterations"):
        solve_profile_iteration(nodes, triangles, parameters, iterations=0)
    with pytest.raises(ValueError, match="relaxation"):
        solve_profile_iteration(nodes, triangles, parameters, relaxation=1.5)
    with pytest.raises(ValueError, match="initial_psi"):
        solve_profile_iteration(nodes, triangles, parameters, initial_psi=jnp.ones(3))
    with pytest.raises(ValueError, match="dirichlet_values"):
        solve_profile_iteration(
            nodes,
            triangles,
            parameters,
            dirichlet_nodes=jnp.asarray([0, 1]),
            dirichlet_values=jnp.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="psin"):
        assemble_nonlinear_profile_load_vector(nodes, triangles, jnp.ones(3), parameters)
    with pytest.raises(ValueError, match="points"):
        nonlinear_profile_source_density(jnp.ones((3, 3)), jnp.ones(3), parameters)
    with pytest.raises(ValueError, match="nodal_psin"):
        linear_nonlinear_profile_load_vector(nodes[triangles[0]], jnp.ones(4), parameters)
