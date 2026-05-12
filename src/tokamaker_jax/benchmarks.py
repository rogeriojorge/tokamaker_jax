"""Small benchmark helpers with JSON-friendly outputs."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from tokamaker_jax.assembly import (
    apply_grad_shafranov_stiffness_matrix,
    assemble_grad_shafranov_stiffness_matrix,
)
from tokamaker_jax.config import CoilConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.fem import linear_mass_matrix, linear_stiffness_matrix
from tokamaker_jax.free_boundary import circular_loop_elliptic_response_matrix, coil_flux_on_grid
from tokamaker_jax.profiles import solovev_source
from tokamaker_jax.solver import EquilibriumSolution, solve_fixed_boundary
from tokamaker_jax.verification import rectangular_triangles

T = TypeVar("T")

BENCHMARK_REPORT_SCHEMA_VERSION = 1


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


def benchmark_coil_green_response(
    *,
    nr: int = 65,
    nz: int = 65,
    repeats: int = 5,
    warmups: int = 1,
) -> BenchmarkResult:
    """Benchmark reduced free-boundary coil Green's response on a grid."""

    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, nr, nz)
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )

    @jax.jit
    def run() -> jnp.ndarray:
        return coil_flux_on_grid(grid, coils)

    return benchmark_callable(
        "reduced_coil_green_response",
        run,
        repeats=repeats,
        warmups=warmups,
        metadata={"nr": nr, "nz": nz, "n_coils": len(coils)},
    )


def benchmark_circular_loop_elliptic_response(
    *,
    n_points: int = 256,
    repeats: int = 5,
    warmups: int = 1,
) -> BenchmarkResult:
    """Benchmark the closed-form circular-loop elliptic response matrix."""

    r_values = jnp.linspace(1.1, 2.6, n_points, dtype=jnp.float64)
    z_values = 0.45 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, n_points, dtype=jnp.float64))
    points = jnp.column_stack((r_values, z_values))
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )

    @jax.jit
    def run() -> jnp.ndarray:
        return circular_loop_elliptic_response_matrix(points, coils)

    return benchmark_callable(
        "circular_loop_elliptic_response",
        run,
        repeats=repeats,
        warmups=warmups,
        metadata={"n_points": n_points, "n_coils": len(coils), "kernel": "agm_elliptic"},
    )


def benchmark_baseline_report(
    *,
    repeats: int = 5,
    warmups: int = 1,
    seed_equilibrium: Mapping[str, Any] | None = None,
    local_fem: Mapping[str, Any] | None = None,
    axisymmetric_fem: Mapping[str, Any] | None = None,
    coil_green: Mapping[str, Any] | None = None,
    circular_loop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the baseline benchmark lanes and return a JSON-friendly report.

    The report intentionally records timings without interpreting them against
    pass/fail thresholds; CI jobs can upload the payload as an artifact and
    compare it with environment-specific tooling.
    """

    common_options = {"repeats": repeats, "warmups": warmups}
    entries = [
        _benchmark_report_entry(
            "seed",
            benchmark_seed_equilibrium(
                **_benchmark_options(common_options, seed_equilibrium),
            ),
        ),
        _benchmark_report_entry(
            "local_fem",
            benchmark_local_fem_kernel(
                **_benchmark_options(common_options, local_fem),
            ),
        ),
        _benchmark_report_entry(
            "axisymmetric_fem",
            benchmark_axisymmetric_fem_apply(
                **_benchmark_options(common_options, axisymmetric_fem),
            ),
        ),
        _benchmark_report_entry(
            "reduced_coil_green",
            benchmark_coil_green_response(
                **_benchmark_options(common_options, coil_green),
            ),
        ),
        _benchmark_report_entry(
            "circular_loop_elliptic",
            benchmark_circular_loop_elliptic_response(
                **_benchmark_options(common_options, circular_loop),
            ),
        ),
    ]

    return {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "suite": "tokamaker_jax_baseline_benchmarks",
        "generated_by": "tokamaker_jax.benchmarks.benchmark_baseline_report",
        "time_unit": "seconds",
        "benchmarks": entries,
    }


def benchmark_report_to_json(report: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a benchmark report as deterministic JSON text."""

    return json.dumps(report, indent=indent, sort_keys=True) + "\n"


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


def _benchmark_options(
    common_options: Mapping[str, int],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = dict(common_options)
    if overrides is not None:
        options.update(overrides)
    return options


def _benchmark_report_entry(lane: str, result: BenchmarkResult) -> dict[str, Any]:
    return {
        "lane": lane,
        "result": result.to_dict(),
    }
