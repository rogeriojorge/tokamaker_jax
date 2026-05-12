"""Small benchmark helpers with JSON-friendly outputs."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from tokamaker_jax.assembly import (
    apply_grad_shafranov_stiffness_matrix,
    assemble_grad_shafranov_stiffness_matrix,
)
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.fem import linear_mass_matrix, linear_stiffness_matrix
from tokamaker_jax.profiles import solovev_source
from tokamaker_jax.solver import EquilibriumSolution, solve_fixed_boundary
from tokamaker_jax.verification import rectangular_triangles

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for a small reproducible benchmark."""

    name: str
    repeats: int
    warmups: int
    best_s: float
    median_s: float
    worst_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "name": self.name,
            "repeats": self.repeats,
            "warmups": self.warmups,
            "best_s": self.best_s,
            "median_s": self.median_s,
            "worst_s": self.worst_s,
            "metadata": dict(self.metadata),
        }


def benchmark_callable(
    name: str,
    function: Callable[[], T],
    *,
    repeats: int = 5,
    warmups: int = 1,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Time a callable after optional warmup runs."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")

    for _ in range(warmups):
        _block_until_ready(function())

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block_until_ready(function())
        timings.append(time.perf_counter() - start)

    return BenchmarkResult(
        name=name,
        repeats=repeats,
        warmups=warmups,
        best_s=min(timings),
        median_s=statistics.median(timings),
        worst_s=max(timings),
        metadata={} if metadata is None else metadata,
    )


def benchmark_seed_equilibrium(
    *,
    nr: int = 65,
    nz: int = 65,
    iterations: int = 120,
    repeats: int = 3,
    warmups: int = 1,
) -> BenchmarkResult:
    """Benchmark the current fixed-boundary seed equilibrium solve."""

    grid = RectangularGrid(0.5, 1.5, -0.5, 0.5, nr, nz)
    source = solovev_source(grid, dtype=jnp.float64)

    def run() -> EquilibriumSolution:
        return solve_fixed_boundary(
            grid,
            source,
            iterations=iterations,
            relaxation=0.75,
            dtype=jnp.float64,
        )

    return benchmark_callable(
        "seed_fixed_boundary_equilibrium",
        run,
        repeats=repeats,
        warmups=warmups,
        metadata={"nr": nr, "nz": nz, "iterations": iterations},
    )


def benchmark_local_fem_kernel(
    *,
    repeats: int = 20,
    warmups: int = 2,
) -> BenchmarkResult:
    """Benchmark JIT-compiled p=1 local mass/stiffness matrix kernels."""

    vertices = jnp.asarray([[0.5, -0.2], [1.7, -0.1], [0.7, 0.9]], dtype=jnp.float64)

    @jax.jit
    def run() -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_mass_matrix(vertices), linear_stiffness_matrix(vertices)

    return benchmark_callable(
        "local_p1_triangle_matrices",
        run,
        repeats=repeats,
        warmups=warmups,
        metadata={"element": "p1_triangle", "matrices": ["mass", "stiffness"]},
    )


def benchmark_axisymmetric_fem_apply(
    *,
    subdivisions: int = 16,
    repeats: int = 5,
    warmups: int = 1,
) -> BenchmarkResult:
    """Benchmark p=1 axisymmetric stiffness assembly plus matrix-free apply."""

    nodes, triangles = rectangular_triangles(1.0, 2.0, -0.5, 0.5, subdivisions)
    vector = jnp.sin(jnp.linspace(0.0, 1.0, nodes.shape[0], dtype=jnp.float64))

    @jax.jit
    def run() -> dict[str, jnp.ndarray]:
        matrix = assemble_grad_shafranov_stiffness_matrix(nodes, triangles)
        applied = apply_grad_shafranov_stiffness_matrix(nodes, triangles, vector)
        return {"matrix": matrix, "applied": applied}

    return benchmark_callable(
        "axisymmetric_p1_fem_assembly_apply",
        run,
        repeats=repeats,
        warmups=warmups,
        metadata={"subdivisions": subdivisions, "operator": "grad_shafranov_weak"},
    )


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, EquilibriumSolution):
        value.psi.block_until_ready()
        value.source.block_until_ready()
        value.residual_history.block_until_ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _block_until_ready(item)
