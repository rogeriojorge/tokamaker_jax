"""Generate a Hansen CPC/TokaMaker seed-family reproduction surrogate.

This is an executable literature-anchored fixture, not a claim of exact
OpenFUSIONToolkit/TokaMaker parity. It uses the current rectangular seed solver
to exercise the report, figure-recipe, citation, and artifact path machinery
that future OFT-backed reproductions will reuse.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import config as jax_config

from tokamaker_jax.config import GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.plotting import (
    FigureRecipe,
    equilibrium_figure_data,
    equilibrium_metadata_summary,
    plot_equilibrium,
)
from tokamaker_jax.solver import EquilibriumSolution, solve_from_config

ARTIFACT_ID = "tokamaker-cpc-seed-equilibrium-family"
CPC_SOURCE = "Hansen et al., Computer Physics Communications 298, 109111 (2024)"
CPC_CITATION = (
    "Hansen, C. et al. TokaMaker: An open-source time-dependent Grad-Shafranov "
    "tool for the design and modeling of axisymmetric fusion devices. "
    "Computer Physics Communications 298, 109111 (2024). "
    "doi:10.1016/j.cpc.2024.109111"
)
DEFAULT_PRESSURE_SCALES = (2500.0, 5000.0, 7500.0)


@dataclass(frozen=True)
class ReproductionArtifacts:
    """Paths and JSON payload emitted by the reproduction script."""

    report_path: Path
    png_path: Path
    report: dict[str, Any]


def generate_cpc_seed_family_artifacts(
    output_dir: str | Path,
    *,
    pressure_scales: tuple[float, ...] = DEFAULT_PRESSURE_SCALES,
    ffp_scale: float = -0.35,
    nr: int = 41,
    nz: int = 41,
    iterations: int = 180,
    relaxation: float = 0.75,
    dtype: str = "float64",
    report_name: str = "cpc_seed_family_report.json",
    png_name: str = "cpc_seed_family.png",
    command: str | None = None,
) -> ReproductionArtifacts:
    """Run the seed-family fixture and write the JSON report plus PNG artifact."""

    if not pressure_scales:
        raise ValueError("pressure_scales must contain at least one value")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = _output_child(output_path, report_name)
    png_path = _output_child(output_path, png_name)

    solutions = [
        _solve_case(
            pressure_scale=pressure_scale,
            ffp_scale=ffp_scale,
            nr=nr,
            nz=nz,
            iterations=iterations,
            relaxation=relaxation,
            dtype=dtype,
        )
        for pressure_scale in pressure_scales
    ]
    representative_index = len(solutions) // 2
    family = [
        _family_entry(
            solution,
            pressure_scale=pressure_scale,
            ffp_scale=ffp_scale,
            dtype=dtype,
            relaxation=relaxation,
        )
        for solution, pressure_scale in zip(solutions, pressure_scales, strict=True)
    ]

    _write_family_png(
        solutions,
        pressure_scales=pressure_scales,
        png_path=png_path,
    )

    recipe = _figure_recipe(
        solutions[representative_index],
        pressure_scale=pressure_scales[representative_index],
        family=family,
        command=command,
    )
    report = _report_payload(
        recipe=recipe,
        family=family,
        png_path=png_path,
        report_path=report_path,
        command=command,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReproductionArtifacts(
        report_path=report_path.resolve(),
        png_path=png_path.resolve(),
        report=report,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Hansen CPC/TokaMaker literature-anchored seed-family "
            "surrogate report and PNG artifact."
        )
    )
    parser.add_argument("output_dir", help="Directory where the JSON report and PNG are written.")
    parser.add_argument(
        "--pressure-scale",
        dest="pressure_scales",
        action="append",
        type=float,
        help=(
            "Pressure scale for one family member. Repeat to build a family. "
            "Defaults to 2500, 5000, and 7500."
        ),
    )
    parser.add_argument("--ffp-scale", type=float, default=-0.35)
    parser.add_argument("--nr", type=int, default=41, help="Number of radial grid points.")
    parser.add_argument("--nz", type=int, default=41, help="Number of vertical grid points.")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--relaxation", type=float, default=0.75)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--report-name", default="cpc_seed_family_report.json")
    parser.add_argument("--png-name", default="cpc_seed_family.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    jax_config.update("jax_enable_x64", True)
    parser = build_parser()
    args = parser.parse_args(argv)
    command_args = sys.argv[1:] if argv is None else argv
    command = "python examples/reproduce_cpc_seed_family.py " + " ".join(
        shlex.quote(item) for item in command_args
    )
    artifacts = generate_cpc_seed_family_artifacts(
        args.output_dir,
        pressure_scales=tuple(args.pressure_scales or DEFAULT_PRESSURE_SCALES),
        ffp_scale=args.ffp_scale,
        nr=args.nr,
        nz=args.nz,
        iterations=args.iterations,
        relaxation=args.relaxation,
        dtype=args.dtype,
        report_name=args.report_name,
        png_name=args.png_name,
        command=command,
    )
    print(
        json.dumps(
            {
                "report": str(artifacts.report_path),
                "png": str(artifacts.png_path),
                "artifact_id": ARTIFACT_ID,
            },
            sort_keys=True,
        )
    )
    return 0


def _solve_case(
    *,
    pressure_scale: float,
    ffp_scale: float,
    nr: int,
    nz: int,
    iterations: int,
    relaxation: float,
    dtype: str,
) -> EquilibriumSolution:
    config = RunConfig(
        grid=GridConfig(nr=nr, nz=nz),
        source=SourceConfig(pressure_scale=pressure_scale, ffp_scale=ffp_scale),
        solver=SolverConfig(iterations=iterations, relaxation=relaxation, dtype=dtype),
    )
    return solve_from_config(config)


def _output_child(output_dir: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("output file names must stay under output_dir")
    return output_dir / path


def _family_entry(
    solution: EquilibriumSolution,
    *,
    pressure_scale: float,
    ffp_scale: float,
    dtype: str,
    relaxation: float,
) -> dict[str, Any]:
    summary = equilibrium_metadata_summary(solution)
    residual = summary["residual"]
    initial = residual["initial"]
    final = residual["final"]
    return {
        "config": {
            "pressure_scale": float(pressure_scale),
            "ffp_scale": float(ffp_scale),
            "dtype": dtype,
            "relaxation": float(relaxation),
            "iterations": int(solution.iterations),
            "grid": summary["grid"],
        },
        "diagnostics": {
            "psi": summary["psi"],
            "source": summary["source"],
            "residual": residual,
            "residual_drop": None
            if initial in (None, 0.0) or final is None
            else float(final / initial),
        },
    }


def _figure_recipe(
    solution: EquilibriumSolution,
    *,
    pressure_scale: float,
    family: list[dict[str, Any]],
    command: str | None,
) -> FigureRecipe:
    base = equilibrium_figure_data(
        solution,
        name=ARTIFACT_ID,
        source=CPC_SOURCE,
        citation=CPC_CITATION,
        command=command,
        include_source=True,
    )
    return FigureRecipe(
        name=base.name,
        source=base.source,
        citation=base.citation,
        command=base.command,
        axes=base.axes,
        data=base.data,
        metadata={
            **base.metadata,
            "artifact_id": ARTIFACT_ID,
            "literature_anchor": _literature_anchor(),
            "representative_case": {
                "pressure_scale": float(pressure_scale),
                "selection": "middle pressure-scale member of the generated family",
            },
            "reproduction_scope": {
                "status": "surrogate_fixture",
                "parity_claim": "none",
                "comparison_rule": (
                    "Generated artifact and seed-solver diagnostics only; no "
                    "OFT/TokaMaker numeric tolerance is asserted."
                ),
            },
            "family": family,
        },
    )


def _report_payload(
    *,
    recipe: FigureRecipe,
    family: list[dict[str, Any]],
    png_path: Path,
    report_path: Path,
    command: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "artifact_id": ARTIFACT_ID,
        "status": "surrogate_fixture",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _literature_anchor(),
        "command": command,
        "runtime": {
            "network": "not_used",
            "implementation": "tokamaker-jax rectangular fixed-boundary seed solver",
        },
        "outputs": {
            "report": str(report_path),
            "png": str(png_path),
        },
        "comparison_rule": {
            "type": "surrogate_seed_family",
            "tolerance": None,
            "description": (
                "This artifact checks executable reproduction plumbing and "
                "seed diagnostics. It does not compare against a numbered CPC "
                "figure or OpenFUSIONToolkit output."
            ),
        },
        "limitations": [
            "Uses the current rectangular fixed-boundary seed solver, not TokaMaker's triangular FEM stack.",
            "Does not model free-boundary coils, passive structures, reconstruction constraints, or time dependence.",
            "Does not assert numeric parity with Hansen et al. CPC figures or OpenFUSIONToolkit examples.",
        ],
        "family": family,
        "figure_recipe": recipe.to_dict(),
    }


def _literature_anchor() -> dict[str, Any]:
    return {
        "source": CPC_SOURCE,
        "citation": CPC_CITATION,
        "title": (
            "TokaMaker: An open-source time-dependent Grad-Shafranov tool for "
            "the design and modeling of axisymmetric fusion devices"
        ),
        "journal": "Computer Physics Communications",
        "volume": 298,
        "article": "109111",
        "year": 2024,
        "doi": "10.1016/j.cpc.2024.109111",
        "relationship": (
            "Seed-family surrogate for the literature reproduction workflow; "
            "pending OFT-backed fixture parity."
        ),
    }


def _write_family_png(
    solutions: list[EquilibriumSolution],
    *,
    pressure_scales: tuple[float, ...],
    png_path: Path,
) -> None:
    width = max(4.5, 3.2 * len(solutions))
    fig, axes = plt.subplots(
        1,
        len(solutions),
        figsize=(width, 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, solution, pressure_scale in zip(axes.flat, solutions, pressure_scales, strict=True):
        plot_equilibrium(
            solution,
            levels=14,
            ax=ax,
            show_source=False,
            show_metadata=False,
            label_contours=False,
        )
        ax.set_title(f"pressure scale {pressure_scale:.0f}")
    fig.suptitle("Hansen CPC/TokaMaker seed-family surrogate")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
