"""Emit a JSON benchmark report suitable for CI artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from jax import config as jax_config

from tokamaker_jax.benchmarks import benchmark_baseline_report, benchmark_report_to_json


def main() -> None:
    jax_config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions per lane")
    parser.add_argument("--warmups", type=int, default=1, help="untimed warmups per lane")
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    args = parser.parse_args()

    report = benchmark_baseline_report(repeats=args.repeats, warmups=args.warmups)
    payload = benchmark_report_to_json(report)
    if args.output is None:
        print(payload, end="")
        return

    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
