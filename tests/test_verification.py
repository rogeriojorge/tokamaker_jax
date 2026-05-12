import json

import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.verification import (
    manufactured_grad_shafranov_error_metrics,
    manufactured_grad_shafranov_exact,
    manufactured_grad_shafranov_source,
    observed_rates,
    poisson_error_metrics,
    rectangular_triangles,
    run_coil_green_function_validation,
    run_grad_shafranov_convergence_study,
    run_poisson_convergence_study,
    sine_poisson_exact,
    sine_poisson_forcing,
    solve_manufactured_grad_shafranov,
    solve_sine_poisson,
    unit_square_triangles,
)


def test_unit_square_triangles_and_manufactured_fields():
    nodes, triangles = unit_square_triangles(3)
    rectangle_nodes, rectangle_triangles = rectangular_triangles(1.0, 2.0, -0.5, 0.5, 3, 2)

    assert nodes.shape == (16, 2)
    assert triangles.shape == (18, 3)
    assert rectangle_nodes.shape == (12, 2)
    assert rectangle_triangles.shape == (12, 3)
    np.testing.assert_allclose(sine_poisson_exact(jnp.asarray([[0.5, 0.5]])), [1.0])
    np.testing.assert_allclose(
        sine_poisson_forcing(jnp.asarray([[0.5, 0.5]])),
        [2.0 * np.pi**2],
    )
    np.testing.assert_allclose(
        manufactured_grad_shafranov_exact(jnp.asarray([[1.5, 0.0]])),
        [1.0],
        atol=1.0e-14,
    )
    assert float(manufactured_grad_shafranov_source(jnp.asarray([[1.5, 0.0]]))[0]) > 0.0


def test_sine_poisson_solution_and_convergence_schema():
    nodes, triangles, solution = solve_sine_poisson(4)
    result = poisson_error_metrics(4)
    study = run_poisson_convergence_study((4, 8))

    assert nodes.shape[0] == solution.shape[0]
    assert triangles.shape[0] == result.n_cells
    assert result.l2_error > 0.0
    assert result.h1_error > 0.0
    payload = study.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert len(study.results) == 2
    assert len(study.l2_rates) == 1
    assert len(study.h1_rates) == 1


def test_manufactured_grad_shafranov_solution_and_convergence_schema():
    nodes, triangles, solution = solve_manufactured_grad_shafranov(4)
    result = manufactured_grad_shafranov_error_metrics(4)
    study = run_grad_shafranov_convergence_study((4, 8, 16))

    assert nodes.shape[0] == solution.shape[0]
    assert triangles.shape[0] == result.n_cells
    assert result.l2_error > 0.0
    assert result.weighted_h1_error > 0.0
    payload = study.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert len(study.results) == 3
    assert len(study.l2_rates) == 2
    assert len(study.weighted_h1_rates) == 2
    assert min(study.l2_rates) > 1.75
    assert min(study.weighted_h1_rates) > 0.85


def test_reduced_coil_green_function_validation_schema_and_tolerances():
    result = run_coil_green_function_validation()
    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert result.n_points == 4
    assert result.n_coils == 2
    assert result.symmetry_error < 1.0e-18
    assert result.linearity_error < 1.0e-18
    assert result.gradient_error < 1.0e-14
    assert result.log_ratio_error < 1.0e-18


def test_verification_validation_errors():
    with pytest.raises(ValueError, match="at least 2"):
        unit_square_triangles(1)
    with pytest.raises(ValueError, match="at least 2"):
        rectangular_triangles(1.0, 2.0, -1.0, 1.0, 2, 1)
    with pytest.raises(ValueError, match="r_max"):
        rectangular_triangles(1.0, 1.0, -1.0, 1.0, 2)
    with pytest.raises(ValueError, match="z_max"):
        rectangular_triangles(1.0, 2.0, 1.0, 1.0, 2)
    with pytest.raises(ValueError, match="at least two"):
        run_poisson_convergence_study((4,))
    with pytest.raises(ValueError, match="at least two"):
        run_grad_shafranov_convergence_study((4,))
    with pytest.raises(ValueError, match="same length"):
        observed_rates([1.0, 0.5], [1.0])
    with pytest.raises(ValueError, match="at least two"):
        observed_rates([1.0], [1.0])
