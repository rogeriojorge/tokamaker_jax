"""Emit a JSON benchmark report suitable for CI artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jax import config as jax_config

from tokamaker_jax.benchmarks import (
    benchmark_baseline_report,
    benchmark_report_to_json,
    benchmark_threshold_report,
)


def main() -> None:
    jax_config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions per lane")
    parser.add_argument("--warmups", type=int, default=1, help="untimed warmups per lane")
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    parser.add_argument("--thresholds", type=Path, help="optional benchmark threshold JSON")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="optional output path for threshold comparison JSON",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="exit nonzero if any threshold comparison fails",
    )
    args = parser.parse_args()

    report = benchmark_baseline_report(repeats=args.repeats, warmups=args.warmups)
    payload = benchmark_report_to_json(report)
    comparison = None
    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        comparison = benchmark_threshold_report(report, thresholds)
        if args.comparison_output is not None:
            args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
            args.comparison_output.write_text(
                benchmark_report_to_json(comparison),
                encoding="utf-8",
            )
    if args.output is None:
        print(payload, end="")
        if comparison is not None and args.comparison_output is None:
            print(benchmark_report_to_json(comparison), end="")
        if args.fail_on_threshold and comparison is not None and not comparison["passed"]:
            raise SystemExit(1)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.fail_on_threshold and comparison is not None and not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
