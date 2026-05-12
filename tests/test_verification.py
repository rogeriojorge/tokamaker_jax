import json

import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.verification import (
    observed_rates,
    poisson_error_metrics,
    run_poisson_convergence_study,
    sine_poisson_exact,
    sine_poisson_forcing,
    solve_sine_poisson,
    unit_square_triangles,
)


def test_unit_square_triangles_and_manufactured_fields():
    nodes, triangles = unit_square_triangles(3)

    assert nodes.shape == (16, 2)
    assert triangles.shape == (18, 3)
    np.testing.assert_allclose(sine_poisson_exact(jnp.asarray([[0.5, 0.5]])), [1.0])
    np.testing.assert_allclose(
        sine_poisson_forcing(jnp.asarray([[0.5, 0.5]])),
        [2.0 * np.pi**2],
    )


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


def test_verification_validation_errors():
    with pytest.raises(ValueError, match="at least 2"):
        unit_square_triangles(1)
    with pytest.raises(ValueError, match="at least two"):
        run_poisson_convergence_study((4,))
    with pytest.raises(ValueError, match="same length"):
        observed_rates([1.0, 0.5], [1.0])
    with pytest.raises(ValueError, match="at least two"):
        observed_rates([1.0], [1.0])
