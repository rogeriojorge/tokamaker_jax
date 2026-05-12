import json

import pytest

from tokamaker_jax.benchmarks import (
    _block_until_ready,
    benchmark_axisymmetric_fem_apply,
    benchmark_callable,
    benchmark_coil_green_response,
    benchmark_local_fem_kernel,
    benchmark_seed_equilibrium,
)


def test_benchmark_callable_schema_and_validation():
    result = benchmark_callable(
        "constant",
        lambda: 1,
        repeats=2,
        warmups=1,
        metadata={"case": "unit"},
    )

    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["name"] == "constant"
    assert payload["repeats"] == 2
    assert payload["warmups"] == 1
    assert payload["best_s"] <= payload["median_s"] <= payload["worst_s"]
    assert payload["metadata"] == {"case": "unit"}

    with pytest.raises(ValueError, match="repeats"):
        benchmark_callable("bad", lambda: None, repeats=0)
    with pytest.raises(ValueError, match="warmups"):
        benchmark_callable("bad", lambda: None, warmups=-1)


def test_local_fem_benchmark_runs_jitted_kernel():
    result = benchmark_local_fem_kernel(repeats=1, warmups=1)

    assert result.name == "local_p1_triangle_matrices"
    assert result.repeats == 1
    assert result.metadata["element"] == "p1_triangle"
    assert result.best_s >= 0.0


def test_seed_equilibrium_benchmark_runs_small_case():
    result = benchmark_seed_equilibrium(nr=9, nz=9, iterations=3, repeats=1, warmups=0)

    assert result.name == "seed_fixed_boundary_equilibrium"
    assert result.metadata == {"nr": 9, "nz": 9, "iterations": 3}
    assert result.best_s >= 0.0


def test_axisymmetric_fem_benchmark_runs_small_case():
    result = benchmark_axisymmetric_fem_apply(subdivisions=3, repeats=1, warmups=0)

    assert result.name == "axisymmetric_p1_fem_assembly_apply"
    assert result.metadata == {"subdivisions": 3, "operator": "grad_shafranov_weak"}
    assert result.best_s >= 0.0


def test_coil_green_response_benchmark_runs_small_case():
    result = benchmark_coil_green_response(nr=7, nz=7, repeats=1, warmups=0)

    assert result.name == "reduced_coil_green_response"
    assert result.metadata == {"nr": 7, "nz": 7, "n_coils": 3}
    assert result.best_s >= 0.0


def test_block_until_ready_accepts_nested_python_containers():
    _block_until_ready({"items": [1, (2, 3)]})
