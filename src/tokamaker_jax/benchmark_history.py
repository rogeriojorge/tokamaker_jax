"""Benchmark history helpers for hardware-normalized comparisons."""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_HISTORY_SCHEMA_VERSION = 1
BENCHMARK_HISTORY_COMPARISON_SCHEMA_VERSION = 1


def benchmark_history_entry(
    report: Mapping[str, Any],
    *,
    timestamp: str | None = None,
    threshold_report: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one stable benchmark-history entry from a benchmark report."""

    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("benchmark report must contain a benchmarks list")

    threshold_by_lane = _thresholds_by_lane(threshold_report)
    lanes = [_history_lane(entry, threshold_by_lane) for entry in benchmarks]

    return {
        "schema_version": BENCHMARK_HISTORY_SCHEMA_VERSION,
        "suite": str(report.get("suite", "tokamaker_jax_baseline_benchmarks")),
        "source_schema_version": report.get("schema_version"),
        "time_unit": str(report.get("time_unit", "seconds")),
        "environment": _environment_metadata(timestamp, environment),
        "lanes": lanes,
    }


def benchmark_history_document(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Wrap benchmark-history entries in a deterministic JSON document."""

    return {
        "schema_version": BENCHMARK_HISTORY_SCHEMA_VERSION,
        "entries": [dict(entry) for entry in entries],
    }


def benchmark_history_to_json(payload: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a history entry or document as deterministic JSON."""

    return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def benchmark_history_to_jsonl(entries: Iterable[Mapping[str, Any]]) -> str:
    """Serialize history entries as deterministic JSON Lines."""

    return "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)


def write_benchmark_history_jsonl(
    path: str | Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    append: bool = True,
) -> None:
    """Write benchmark-history entries to a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as stream:
        stream.write(benchmark_history_to_jsonl(entries))


def read_benchmark_history_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read benchmark-history entries from a JSONL file."""

    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError(f"history line {line_number} must contain a JSON object")
        entries.append(entry)
    return entries


def compare_benchmark_history(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two history entries and return per-lane timing ratios."""

    current_lanes = _lanes_by_name(current)
    baseline_lanes = _lanes_by_name(baseline)
    comparisons = []
    for lane, current_lane in current_lanes.items():
        if lane not in baseline_lanes:
            continue
        baseline_lane = baseline_lanes[lane]
        current_metrics = current_lane["metrics"]
        baseline_metrics = baseline_lane["metrics"]
        comparisons.append(
            {
                "lane": lane,
                "current_median_s": float(current_metrics["median_s"]),
                "baseline_median_s": float(baseline_metrics["median_s"]),
                "median_ratio": _positive_ratio(
                    float(current_metrics["median_s"]),
                    float(baseline_metrics["median_s"]),
                    lane,
                    "median_s",
                ),
                "current_best_s": float(current_metrics["best_s"]),
                "baseline_best_s": float(baseline_metrics["best_s"]),
                "best_ratio": _positive_ratio(
                    float(current_metrics["best_s"]),
                    float(baseline_metrics["best_s"]),
                    lane,
                    "best_s",
                ),
                "current_worst_s": float(current_metrics["worst_s"]),
                "baseline_worst_s": float(baseline_metrics["worst_s"]),
                "worst_ratio": _positive_ratio(
                    float(current_metrics["worst_s"]),
                    float(baseline_metrics["worst_s"]),
                    lane,
                    "worst_s",
                ),
                "current_threshold_status": current_lane.get("threshold", {}).get("status"),
                "baseline_threshold_status": baseline_lane.get("threshold", {}).get("status"),
            }
        )

    return {
        "schema_version": BENCHMARK_HISTORY_COMPARISON_SCHEMA_VERSION,
        "suite": current.get("suite"),
        "baseline_timestamp": baseline.get("environment", {}).get("timestamp"),
        "current_timestamp": current.get("environment", {}).get("timestamp"),
        "time_unit": current.get("time_unit", "seconds"),
        "comparisons": comparisons,
    }


def _environment_metadata(
    timestamp: str | None,
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if environment is not None:
        metadata.update(environment)
        if timestamp is not None:
            metadata["timestamp"] = timestamp
    return metadata


def _history_lane(
    entry: Mapping[str, Any],
    threshold_by_lane: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lane = str(entry["lane"])
    result = entry["result"]
    metrics = {
        "best_s": float(result["best_s"]),
        "median_s": float(result["median_s"]),
        "worst_s": float(result["worst_s"]),
        "repeats": int(result["repeats"]),
        "warmups": int(result["warmups"]),
    }
    history_lane = {
        "lane": lane,
        "name": str(result["name"]),
        "metrics": metrics,
        "metadata": dict(result.get("metadata", {})),
    }
    threshold = threshold_by_lane.get(lane)
    if threshold is not None:
        history_lane["threshold"] = {
            "status": str(threshold["status"]),
            "ratio": float(threshold["ratio"]),
            "max_median_s": float(threshold["max_median_s"]),
        }
    return history_lane


def _thresholds_by_lane(
    threshold_report: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if threshold_report is None:
        return {}
    comparisons = threshold_report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("threshold report must contain a comparisons list")
    return {str(entry["lane"]): entry for entry in comparisons}


def _lanes_by_name(entry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    lanes = entry.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError("history entry must contain a lanes list")
    return {str(lane["lane"]): lane for lane in lanes}


def _positive_ratio(current: float, baseline: float, lane: str, metric: str) -> float:
    if baseline <= 0.0:
        raise ValueError(f"baseline {metric} for lane {lane!r} must be positive")
    return current / baseline
