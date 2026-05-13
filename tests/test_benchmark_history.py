import json

import pytest

from tokamaker_jax.benchmark_history import (
    BENCHMARK_HISTORY_SCHEMA_VERSION,
    benchmark_history_document,
    benchmark_history_entry,
    benchmark_history_to_json,
    benchmark_history_to_jsonl,
    compare_benchmark_history,
    read_benchmark_history_jsonl,
    write_benchmark_history_jsonl,
)


def test_benchmark_history_entry_normalizes_report_and_thresholds():
    entry = benchmark_history_entry(
        _benchmark_report(median_s=2.0),
        timestamp="2026-05-13T10:00:00+00:00",
        threshold_report=_threshold_report(status="pass", ratio=0.5),
        environment={
            "platform": "test-platform",
            "python": "3.12.1",
            "machine": "test-machine",
            "processor": "test-cpu",
        },
    )

    assert json.loads(json.dumps(entry)) == entry
    assert entry["schema_version"] == BENCHMARK_HISTORY_SCHEMA_VERSION
    assert entry["suite"] == "tokamaker_jax_baseline_benchmarks"
    assert entry["source_schema_version"] == 1
    assert entry["time_unit"] == "seconds"
    assert entry["environment"] == {
        "platform": "test-platform",
        "python": "3.12.1",
        "machine": "test-machine",
        "processor": "test-cpu",
        "timestamp": "2026-05-13T10:00:00+00:00",
    }
    assert entry["lanes"] == [
        {
            "lane": "seed",
            "name": "seed_fixed_boundary_equilibrium",
            "metrics": {
                "best_s": 1.0,
                "median_s": 2.0,
                "worst_s": 3.0,
                "repeats": 3,
                "warmups": 1,
            },
            "metadata": {"nr": 9, "nz": 9},
            "threshold": {
                "status": "pass",
                "ratio": 0.5,
                "max_median_s": 4.0,
            },
        }
    ]


def test_benchmark_history_serializes_json_document_and_jsonl(tmp_path):
    first = benchmark_history_entry(
        _benchmark_report(median_s=1.0),
        timestamp="2026-05-13T10:00:00+00:00",
        environment=_environment(),
    )
    second = benchmark_history_entry(
        _benchmark_report(median_s=1.2),
        timestamp="2026-05-13T11:00:00+00:00",
        environment=_environment(),
    )

    document = benchmark_history_document([first, second])
    assert json.loads(benchmark_history_to_json(document)) == document
    assert document == {
        "schema_version": BENCHMARK_HISTORY_SCHEMA_VERSION,
        "entries": [first, second],
    }

    path = tmp_path / "history.jsonl"
    write_benchmark_history_jsonl(path, [first], append=False)
    write_benchmark_history_jsonl(path, [second])

    assert benchmark_history_to_jsonl([first, second]) == path.read_text(encoding="utf-8")
    assert read_benchmark_history_jsonl(path) == [first, second]

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="history line 2"):
        read_benchmark_history_jsonl(invalid_path)


def test_compare_benchmark_history_returns_ratios_and_threshold_status():
    baseline = benchmark_history_entry(
        _benchmark_report(best_s=1.0, median_s=2.0, worst_s=4.0),
        timestamp="2026-05-13T10:00:00+00:00",
        threshold_report=_threshold_report(status="pass", ratio=0.5),
        environment=_environment(),
    )
    current = benchmark_history_entry(
        _benchmark_report(best_s=1.5, median_s=3.0, worst_s=6.0),
        timestamp="2026-05-13T11:00:00+00:00",
        threshold_report=_threshold_report(status="fail", ratio=1.5),
        environment=_environment(),
    )

    comparison = compare_benchmark_history(current, baseline)

    assert comparison == {
        "schema_version": 1,
        "suite": "tokamaker_jax_baseline_benchmarks",
        "baseline_timestamp": "2026-05-13T10:00:00+00:00",
        "current_timestamp": "2026-05-13T11:00:00+00:00",
        "time_unit": "seconds",
        "comparisons": [
            {
                "lane": "seed",
                "current_median_s": 3.0,
                "baseline_median_s": 2.0,
                "median_ratio": 1.5,
                "current_best_s": 1.5,
                "baseline_best_s": 1.0,
                "best_ratio": 1.5,
                "current_worst_s": 6.0,
                "baseline_worst_s": 4.0,
                "worst_ratio": 1.5,
                "current_threshold_status": "fail",
                "baseline_threshold_status": "pass",
            }
        ],
    }


def test_compare_benchmark_history_allows_nonoverlapping_lanes_and_default_environment():
    current = benchmark_history_entry(
        _benchmark_report(lane="current-only", median_s=1.0),
        environment={"platform": "override-platform"},
    )
    baseline = benchmark_history_entry(
        _benchmark_report(lane="baseline-only", median_s=1.0),
        environment=_environment(),
    )

    comparison = compare_benchmark_history(current, baseline)

    assert comparison["comparisons"] == []
    assert current["environment"]["platform"] == "override-platform"
    assert "timestamp" in current["environment"]


def test_benchmark_history_validates_required_shapes():
    with pytest.raises(ValueError, match="benchmarks list"):
        benchmark_history_entry({})
    with pytest.raises(ValueError, match="comparisons list"):
        benchmark_history_entry(_benchmark_report(median_s=1.0), threshold_report={})
    with pytest.raises(ValueError, match="lanes list"):
        compare_benchmark_history({}, {})
    with pytest.raises(ValueError, match="baseline median_s"):
        compare_benchmark_history(
            benchmark_history_entry(
                _benchmark_report(median_s=1.0),
                timestamp="2026-05-13T11:00:00+00:00",
                environment=_environment(),
            ),
            benchmark_history_entry(
                _benchmark_report(median_s=0.0),
                timestamp="2026-05-13T10:00:00+00:00",
                environment=_environment(),
            ),
        )


def _benchmark_report(
    *,
    lane: str = "seed",
    best_s: float = 1.0,
    median_s: float,
    worst_s: float = 3.0,
) -> dict:
    return {
        "schema_version": 1,
        "suite": "tokamaker_jax_baseline_benchmarks",
        "time_unit": "seconds",
        "benchmarks": [
            {
                "lane": lane,
                "result": {
                    "name": "seed_fixed_boundary_equilibrium",
                    "repeats": 3,
                    "warmups": 1,
                    "best_s": best_s,
                    "median_s": median_s,
                    "worst_s": worst_s,
                    "metadata": {"nr": 9, "nz": 9},
                },
            }
        ],
    }


def _threshold_report(*, status: str, ratio: float) -> dict:
    return {
        "schema_version": 1,
        "suite": "tokamaker_jax_baseline_benchmarks",
        "time_unit": "seconds",
        "passed": status == "pass",
        "comparisons": [
            {
                "lane": "seed",
                "median_s": 2.0,
                "max_median_s": 4.0,
                "ratio": ratio,
                "status": status,
            }
        ],
    }


def _environment() -> dict:
    return {
        "platform": "test-platform",
        "python": "3.12.1",
        "machine": "test-machine",
        "processor": "test-cpu",
    }
