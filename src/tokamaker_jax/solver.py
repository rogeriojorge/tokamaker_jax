"""Differentiable seed solver for the Grad-Shafranov equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from tokamaker_jax.config import RunConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.profiles import gaussian_coil_source, solovev_source


@dataclass(frozen=True)
class EquilibriumSolution:
    """Computed seed equilibrium."""

    grid: RectangularGrid
    psi: jnp.ndarray
    source: jnp.ndarray
    residual_history: jnp.ndarray
    iterations: int

    def stats(self) -> dict[str, float | int]:
        """Return scalar diagnostics for CLI and docs examples."""

        return {
            "iterations": self.iterations,
            "psi_min": float(jnp.min(self.psi)),
            "psi_max": float(jnp.max(self.psi)),
            "residual_final": float(self.residual_history[-1]),
        }


def solve_from_config(config: RunConfig) -> EquilibriumSolution:
    """Solve the seed fixed-boundary problem from a TOML-backed config."""

    dtype = _dtype_from_name(config.solver.dtype)
    grid = RectangularGrid(**config.grid.__dict__)
    source = build_source(config, grid, dtype=dtype)
    return solve_fixed_boundary(
        grid,
        source,
        iterations=config.solver.iterations,
        relaxation=config.solver.relaxation,
        dtype=dtype,
    )


def build_source(config: RunConfig, grid: RectangularGrid, dtype: jnp.dtype) -> jnp.ndarray:
    """Build the total RHS source for the current seed solver."""

    if config.source.profile != "solovev":
        raise ValueError(f"Unsupported seed source profile: {config.source.profile!r}")
    source = solovev_source(
        grid,
        pressure_scale=config.source.pressure_scale,
        ffp_scale=config.source.ffp_scale,
        dtype=dtype,
    )
    return source + gaussian_coil_source(grid, config.coils, dtype=dtype)


def solve_fixed_boundary(
    grid: RectangularGrid,
    source: jnp.ndarray,
    *,
    iterations: int = 600,
    relaxation: float = 0.75,
    boundary: jnp.ndarray | None = None,
    dtype: jnp.dtype = jnp.float64,
) -> EquilibriumSolution:
    """Solve ``Delta* psi = source`` with fixed Dirichlet boundary values.

    The Jacobi iteration is intentionally expressed with ``jax.lax.scan`` so
    gradients can pass through the unrolled solve during the first porting
    phase. Later phases will replace this with differentiable sparse FEM
    operators and implicit custom VJPs where needed for performance.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must satisfy 0 < relaxation <= 1")
    psi0 = grid.zeros(dtype=dtype) if boundary is None else jnp.asarray(boundary, dtype=dtype)
    source = jnp.asarray(source, dtype=dtype)
    if source.shape != psi0.shape:
        raise ValueError(f"source shape {source.shape} does not match grid shape {psi0.shape}")

    def step(psi: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        updated = jacobi_update(grid, psi, source, relaxation=relaxation)
        return updated, residual_norm(grid, updated, source)

    psi, residual_history = jax.lax.scan(step, psi0, None, length=iterations)
    return EquilibriumSolution(
        grid=grid,
        psi=psi,
        source=source,
        residual_history=residual_history,
        iterations=iterations,
    )


def jacobi_update(
    grid: RectangularGrid,
    psi: jnp.ndarray,
    source: jnp.ndarray,
    *,
    relaxation: float,
) -> jnp.ndarray:
    """One weighted Jacobi update for the axisymmetric operator."""

    r, _ = grid.mesh(dtype=psi.dtype)
    center = psi[1:-1, 1:-1]
    east = psi[2:, 1:-1]
    west = psi[:-2, 1:-1]
    north = psi[1:-1, 2:]
    south = psi[1:-1, :-2]
    r_i = r[1:-1, 1:-1]
    dr = grid.dr
    dz = grid.dz
    denominator = 2.0 / dr**2 + 2.0 / dz**2
    numerator = (
        (east + west) / dr**2
        + (north + south) / dz**2
        - (east - west) / (2.0 * r_i * dr)
        - source[1:-1, 1:-1]
    )
    interior = (1.0 - relaxation) * center + relaxation * numerator / denominator
    return psi.at[1:-1, 1:-1].set(interior)


def apply_operator(grid: RectangularGrid, psi: jnp.ndarray) -> jnp.ndarray:
    """Apply the finite-difference Grad-Shafranov operator to ``psi``."""

    r, _ = grid.mesh(dtype=psi.dtype)
    result = jnp.zeros_like(psi)
    east = psi[2:, 1:-1]
    west = psi[:-2, 1:-1]
    center = psi[1:-1, 1:-1]
    north = psi[1:-1, 2:]
    south = psi[1:-1, :-2]
    r_i = r[1:-1, 1:-1]
    interior = (
        (east - 2.0 * center + west) / grid.dr**2
        - (east - west) / (2.0 * r_i * grid.dr)
        + (north - 2.0 * center + south) / grid.dz**2
    )
    return result.at[1:-1, 1:-1].set(interior)


def residual_norm(grid: RectangularGrid, psi: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 residual on the interior grid."""

    residual = apply_operator(grid, psi)[1:-1, 1:-1] - source[1:-1, 1:-1]
    denom = jnp.linalg.norm(source[1:-1, 1:-1]) + 1.0e-30
    return jnp.linalg.norm(residual) / denom


def _dtype_from_name(name: str) -> jnp.dtype:
    if name == "float64":
        jax.config.update("jax_enable_x64", True)
        return jnp.float64
    if name == "float32":
        return jnp.float32
    raise ValueError("dtype must be 'float32' or 'float64'")
