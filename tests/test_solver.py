import jax
import jax.numpy as jnp
import pytest

from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.profiles import (
    gaussian_coil_source,
    grad_shafranov_weak_source_density,
    normalized_flux,
    power_profile,
    solovev_source,
)
from tokamaker_jax.solver import apply_operator, solve_fixed_boundary, solve_from_config


def test_grid_validation():
    with pytest.raises(ValueError, match="r_min"):
        RectangularGrid(0.0, 1.0, -1.0, 1.0, 5, 5)
    with pytest.raises(ValueError, match="r_max"):
        RectangularGrid(1.0, 1.0, -1.0, 1.0, 5, 5)
    with pytest.raises(ValueError, match="z_max"):
        RectangularGrid(1.0, 2.0, 1.0, 1.0, 5, 5)
    with pytest.raises(ValueError, match="at least 3"):
        RectangularGrid(1.0, 2.0, -1.0, 1.0, 2, 5)


def test_fixed_boundary_shapes_and_boundary_values():
    grid = RectangularGrid(0.7, 1.3, -0.3, 0.3, 17, 19)
    source = solovev_source(grid, dtype=jnp.float64)
    solution = solve_fixed_boundary(grid, source, iterations=25, relaxation=0.7)

    assert solution.psi.shape == (17, 19)
    assert solution.source.shape == (17, 19)
    assert solution.residual_history.shape == (25,)
    assert jnp.all(jnp.isfinite(solution.psi))
    assert jnp.all(solution.psi[0, :] == 0.0)
    assert jnp.all(solution.psi[-1, :] == 0.0)
    assert solution.stats()["iterations"] == 25


def test_apply_operator_zero_field():
    grid = RectangularGrid(0.7, 1.3, -0.3, 0.3, 9, 9)
    psi = grid.zeros(dtype=jnp.float64)

    assert jnp.linalg.norm(apply_operator(grid, psi)) == 0.0


def test_solver_is_differentiable_wrt_source_scale():
    grid = RectangularGrid(0.8, 1.2, -0.25, 0.25, 13, 13)
    base = solovev_source(grid, dtype=jnp.float64)

    def objective(scale):
        solution = solve_fixed_boundary(grid, scale * base, iterations=12, relaxation=0.65)
        return jnp.mean(solution.psi**2)

    grad = jax.grad(objective)(jnp.asarray(1.0, dtype=jnp.float64))

    assert jnp.isfinite(grad)
    assert grad > 0.0


def test_normalized_flux_range():
    psi = jnp.asarray([[2.0, 3.0], [4.0, 6.0]])
    psin = normalized_flux(psi)

    assert jnp.isclose(jnp.min(psin), 0.0)
    assert jnp.isclose(jnp.max(psin), 1.0)


def test_power_profile():
    psin = jnp.asarray([0.0, 0.5, 1.0, 1.5])
    profile = power_profile(psin, alpha=2.0, gamma=1.5)

    assert jnp.isclose(profile[0], 1.0)
    assert jnp.isclose(profile[-1], 0.0)


def test_grad_shafranov_weak_source_density_accepts_scalars_and_callables():
    points = jnp.asarray([[1.0, 0.0], [2.0, 0.5]], dtype=jnp.float64)

    scalar_density = grad_shafranov_weak_source_density(
        points,
        pressure_prime=0.0,
        ffprime=2.0,
    )
    callable_density = grad_shafranov_weak_source_density(
        points,
        pressure_prime=lambda x: x[:, 0],
        ffprime=lambda x: 2.0 * x[:, 0],
    )

    assert scalar_density[0] == pytest.approx(1.0)
    assert scalar_density[1] == pytest.approx(0.5)
    assert callable_density.shape == (2,)
    assert jnp.all(jnp.isfinite(callable_density))

    with pytest.raises(ValueError, match="points must have shape"):
        grad_shafranov_weak_source_density(jnp.ones((2, 3)), 0.0, 0.0)
    with pytest.raises(ValueError, match="pressure_prime"):
        grad_shafranov_weak_source_density(points, lambda x: jnp.ones(x.shape[0] + 1), 0.0)


def test_gaussian_coil_source_and_config_solve():
    config = RunConfig(
        grid=GridConfig(nr=15, nz=15),
        source=SourceConfig(pressure_scale=2000.0, ffp_scale=-0.2),
        solver=SolverConfig(iterations=15, relaxation=0.7, dtype="float64"),
        coils=(CoilConfig(name="PF", r=0.8, z=0.2, current=1000.0, sigma=0.05),),
    )
    grid = RectangularGrid(**config.grid.__dict__)
    coil_source = gaussian_coil_source(grid, config.coils)
    solution = solve_from_config(config)

    assert jnp.linalg.norm(coil_source) > 0.0
    assert solution.psi.shape == (15, 15)


def test_solver_rejects_invalid_inputs():
    grid = RectangularGrid(0.8, 1.2, -0.25, 0.25, 7, 7)
    source = solovev_source(grid, dtype=jnp.float64)

    with pytest.raises(ValueError, match="iterations"):
        solve_fixed_boundary(grid, source, iterations=0)
    with pytest.raises(ValueError, match="relaxation"):
        solve_fixed_boundary(grid, source, relaxation=1.1)
    with pytest.raises(ValueError, match="source shape"):
        solve_fixed_boundary(grid, source[:-1, :])


def test_config_solve_rejects_unsupported_profile_and_dtype():
    config = RunConfig(source=SourceConfig(profile="unsupported"))
    with pytest.raises(ValueError, match="Unsupported"):
        solve_from_config(config)

    bad_dtype = RunConfig(solver=SolverConfig(dtype="complex128"))
    with pytest.raises(ValueError, match="dtype"):
        solve_from_config(bad_dtype)
