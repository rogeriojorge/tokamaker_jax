"""Record an existing benchmark report as a benchmark-history JSONL entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokamaker_jax.benchmark_history import (
    benchmark_history_entry,
    benchmark_history_to_json,
    write_benchmark_history_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="input benchmark_report JSON path")
    parser.add_argument("history", type=Path, help="output benchmark-history JSONL path")
    parser.add_argument(
        "--threshold-report",
        type=Path,
        help="optional benchmark_threshold_report JSON path",
    )
    parser.add_argument(
        "--timestamp",
        help="optional ISO-8601 timestamp for deterministic history entries",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace the output file instead of appending one entry",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    threshold_report = None
    if args.threshold_report is not None:
        threshold_report = json.loads(args.threshold_report.read_text(encoding="utf-8"))

    entry = benchmark_history_entry(
        report,
        timestamp=args.timestamp,
        threshold_report=threshold_report,
    )
    write_benchmark_history_jsonl(args.history, [entry], append=not args.replace)
    print(benchmark_history_to_json(entry), end="")


if __name__ == "__main__":
    main()
