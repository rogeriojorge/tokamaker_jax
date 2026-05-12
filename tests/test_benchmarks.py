import json

import pytest

from tokamaker_jax.benchmarks import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BENCHMARK_THRESHOLD_SCHEMA_VERSION,
    DEFAULT_BENCHMARK_THRESHOLDS,
    _block_until_ready,
    benchmark_axisymmetric_fem_apply,
    benchmark_baseline_report,
    benchmark_callable,
    benchmark_circular_loop_elliptic_response,
    benchmark_coil_green_response,
    benchmark_local_fem_kernel,
    benchmark_report_to_json,
    benchmark_seed_equilibrium,
    benchmark_threshold_report,
)


def test_benchmark_callable_schema_and_validation():
    result = benchmark_callable(
        "constant",
        lambda: 1,
        repeats=1,
        warmups=0,
        metadata={"case": "unit"},
    )

    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["name"] == "constant"
    assert payload["repeats"] == 1
    assert payload["warmups"] == 0
    assert payload["best_s"] <= payload["median_s"] <= payload["worst_s"]
    assert payload["metadata"] == {"case": "unit"}

    with pytest.raises(ValueError, match="repeats"):
        benchmark_callable("bad", lambda: None, repeats=0)
    with pytest.raises(ValueError, match="warmups"):
        benchmark_callable("bad", lambda: None, warmups=-1)


def test_local_fem_benchmark_runs_jitted_kernel():
    result = benchmark_local_fem_kernel(repeats=1, warmups=0)

    assert result.name == "local_p1_triangle_matrices"
    assert result.repeats == 1
    assert result.warmups == 0
    assert result.metadata["element"] == "p1_triangle"
    assert result.best_s >= 0.0


def test_seed_equilibrium_benchmark_runs_small_case():
    result = benchmark_seed_equilibrium(nr=9, nz=9, iterations=3, repeats=1, warmups=0)

    assert result.name == "seed_fixed_boundary_equilibrium"
    assert result.repeats == 1
    assert result.warmups == 0
    assert result.metadata == {"nr": 9, "nz": 9, "iterations": 3}
    assert result.best_s >= 0.0


def test_axisymmetric_fem_benchmark_runs_small_case():
    result = benchmark_axisymmetric_fem_apply(subdivisions=3, repeats=1, warmups=0)

    assert result.name == "axisymmetric_p1_fem_assembly_apply"
    assert result.repeats == 1
    assert result.warmups == 0
    assert result.metadata == {"subdivisions": 3, "operator": "grad_shafranov_weak"}
    assert result.best_s >= 0.0


def test_coil_green_response_benchmark_runs_small_case():
    result = benchmark_coil_green_response(nr=7, nz=7, repeats=1, warmups=0)

    assert result.name == "reduced_coil_green_response"
    assert result.repeats == 1
    assert result.warmups == 0
    assert result.metadata == {"nr": 7, "nz": 7, "n_coils": 3}
    assert result.best_s >= 0.0


def test_circular_loop_elliptic_benchmark_runs_small_case():
    result = benchmark_circular_loop_elliptic_response(n_points=8, repeats=1, warmups=0)

    assert result.name == "circular_loop_elliptic_response"
    assert result.repeats == 1
    assert result.warmups == 0
    assert result.metadata == {"n_points": 8, "n_coils": 3, "kernel": "agm_elliptic"}
    assert result.best_s >= 0.0


def test_baseline_report_runs_all_lanes_and_roundtrips_json():
    report = benchmark_baseline_report(
        repeats=1,
        warmups=0,
        seed_equilibrium={"nr": 9, "nz": 9, "iterations": 3},
        axisymmetric_fem={"subdivisions": 3},
        coil_green={"nr": 7, "nz": 7},
        circular_loop={"n_points": 8},
    )

    assert json.loads(json.dumps(report)) == report
    assert json.loads(benchmark_report_to_json(report)) == report
    assert report["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert report["suite"] == "tokamaker_jax_baseline_benchmarks"
    assert report["time_unit"] == "seconds"
    assert [entry["lane"] for entry in report["benchmarks"]] == [
        "seed",
        "local_fem",
        "axisymmetric_fem",
        "reduced_coil_green",
        "circular_loop_elliptic",
    ]
    assert [entry["result"]["name"] for entry in report["benchmarks"]] == [
        "seed_fixed_boundary_equilibrium",
        "local_p1_triangle_matrices",
        "axisymmetric_p1_fem_assembly_apply",
        "reduced_coil_green_response",
        "circular_loop_elliptic_response",
    ]

    for entry in report["benchmarks"]:
        assert set(entry) == {"lane", "result"}
        result = entry["result"]
        assert set(result) == {
            "name",
            "repeats",
            "warmups",
            "best_s",
            "median_s",
            "worst_s",
            "metadata",
        }
        assert result["repeats"] == 1
        assert result["warmups"] == 0
        assert isinstance(result["best_s"], float)
        assert isinstance(result["median_s"], float)
        assert isinstance(result["worst_s"], float)
        assert isinstance(result["metadata"], dict)


def test_benchmark_threshold_report_passes_and_fails_lanes():
    report = benchmark_baseline_report(
        repeats=1,
        warmups=0,
        seed_equilibrium={"nr": 9, "nz": 9, "iterations": 3},
        axisymmetric_fem={"subdivisions": 3},
        coil_green={"nr": 7, "nz": 7},
        circular_loop={"n_points": 8},
    )
    passing = benchmark_threshold_report(report, DEFAULT_BENCHMARK_THRESHOLDS)
    failing = benchmark_threshold_report(
        report,
        {entry["lane"]: {"max_median_s": 1.0e-30} for entry in report["benchmarks"]},
    )

    assert json.loads(json.dumps(passing)) == passing
    assert json.loads(benchmark_report_to_json(passing)) == passing
    assert passing["schema_version"] == BENCHMARK_THRESHOLD_SCHEMA_VERSION
    assert passing["passed"] is True
    assert [entry["status"] for entry in passing["comparisons"]] == ["pass"] * 5
    assert failing["passed"] is False
    assert {entry["status"] for entry in failing["comparisons"]} == {"fail"}


def test_benchmark_threshold_report_validates_schema():
    with pytest.raises(ValueError, match="benchmarks list"):
        benchmark_threshold_report({})
    with pytest.raises(ValueError, match="missing benchmark threshold"):
        benchmark_threshold_report(
            {"benchmarks": [{"lane": "seed", "result": {"median_s": 1.0}}]}, {}
        )
    with pytest.raises(ValueError, match="max_median_s"):
        benchmark_threshold_report(
            {"benchmarks": [{"lane": "seed", "result": {"median_s": 1.0}}]},
            {"seed": {}},
        )
    with pytest.raises(ValueError, match="positive"):
        benchmark_threshold_report(
            {"benchmarks": [{"lane": "seed", "result": {"median_s": 1.0}}]},
            {"seed": {"max_median_s": 0.0}},
        )


def test_block_until_ready_accepts_nested_python_containers():
    _block_until_ready({"items": [1, (2, 3)]})
