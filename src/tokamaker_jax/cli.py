"""Command line interface."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tokamaker_jax.cases import (
    CaseManifest,
    case_manifest_to_json,
    case_table_rows,
    default_case_manifest,
)
from tokamaker_jax.config import (
    CoilConfig,
    GridConfig,
    OutputConfig,
    RunConfig,
    SolverConfig,
    load_config,
)
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.plotting import save_equilibrium_plot
from tokamaker_jax.solver import EquilibriumSolution, solve_from_config
from tokamaker_jax.upstream_fixed_boundary import (
    DEFAULT_OPENFUSIONTOOLKIT_ROOT,
    fixed_boundary_report_to_json,
    fixed_boundary_upstream_report,
)
from tokamaker_jax.upstream_fixtures import (
    summarize_upstream_fixtures,
    upstream_fixture_report_to_json,
    upstream_fixture_rows,
)


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a TOML configuration validated without solving."""

    config_path: Path
    grid_shape: tuple[int, int]
    region_count: int
    output_paths: tuple[tuple[str, Path], ...]

    def summary_lines(self) -> tuple[str, ...]:
        outputs = (
            ", ".join(f"{label}={path}" for label, path in self.output_paths)
            if self.output_paths
            else "none configured"
        )
        return (
            f"grid: nr={self.grid_shape[0]}, nz={self.grid_shape[1]}",
            f"regions: {self.region_count}",
            f"outputs: {outputs}",
        )


class ConfigValidationError(ValueError):
    """Raised when a loaded configuration is not runnable."""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    With no TOML path this launches the GUI. With a TOML path it runs a
    reproducible noninteractive solve.
    """

    args_list = sys.argv[1:] if argv is None else list(argv)
    if args_list[:1] == ["cases"]:
        return _main_cases(args_list[1:])
    if args_list[:1] == ["upstream-fixtures"]:
        return _main_upstream_fixtures(args_list[1:])
    if args_list[:1] == ["fixed-boundary-evidence"]:
        return _main_fixed_boundary_evidence(args_list[1:])
    if args_list[:1] == ["validate"]:
        return _main_validate(args_list[1:])
    if args_list[:1] == ["verify"]:
        return _main_verify(args_list[1:])

    parser = argparse.ArgumentParser(prog="tokamaker-jax")
    parser.add_argument("config", nargs="?", help="TOML run configuration. Omit to launch the GUI.")
    parser.add_argument("--output", "-o", help="Write solution arrays to this .npz file.")
    parser.add_argument("--plot", help="Write an equilibrium plot to this image path.")
    args = parser.parse_args(args_list)
    if args.config is None:
        from tokamaker_jax.gui import launch_gui

        launch_gui()
        return 0
    solution = run_config(args.config, output=args.output, plot=args.plot)
    print(json.dumps(solution.stats(), indent=2, sort_keys=True))
    return 0


def _main_cases(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tokamaker-jax cases")
    parser.add_argument("--json", action="store_true", help="Print the manifest as JSON.")
    parser.add_argument("--output", "-o", help="Write the selected manifest JSON to this path.")
    parser.add_argument(
        "--runnable-only",
        action="store_true",
        help="Only include cases with commands intended to run today.",
    )
    parser.add_argument(
        "--status",
        help="Filter to one status such as runnable, validation_gate, or planned_upstream_fixture.",
    )
    args = parser.parse_args(argv)

    manifest = default_case_manifest().filter(
        status=args.status,
        runnable_only=args.runnable_only,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(case_manifest_to_json(manifest), encoding="utf-8")
    if args.json:
        print(case_manifest_to_json(manifest), end="")
    else:
        for line in _case_manifest_summary_lines(manifest):
            print(line)
    return 0


def _case_manifest_summary_lines(manifest: CaseManifest) -> tuple[str, ...]:
    rows = case_table_rows(manifest)
    counts = ", ".join(f"{status}={count}" for status, count in manifest.status_counts().items())
    lines = [
        f"tokamaker-jax cases: {len(rows)} entries ({counts})",
        "Use --json for the full manifest, or --runnable-only to show executable cases.",
    ]
    for row in rows:
        command = row["command"] or row["validation_gate"] or "planned"
        path = f" [{row['path']}]" if row["path"] else ""
        lines.append(
            f"- {row['case_id']}: {row['status']} / {row['parity_level']}{path} -> {command}"
        )
    return tuple(lines)


def _main_upstream_fixtures(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tokamaker-jax upstream-fixtures")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OPENFUSIONTOOLKIT_ROOT),
        help="OpenFUSIONToolkit checkout root.",
    )
    parser.add_argument("--json", action="store_true", help="Print the fixture report as JSON.")
    parser.add_argument("--output", "-o", help="Write the fixture report JSON to this path.")
    args = parser.parse_args(argv)

    report = summarize_upstream_fixtures(root=args.root)
    text = upstream_fixture_report_to_json(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        for line in _upstream_fixture_summary_lines(report):
            print(line)
    return 0


def _upstream_fixture_summary_lines(report: dict[str, Any]) -> tuple[str, ...]:
    rows = upstream_fixture_rows(report)
    lines = [
        (
            "upstream TokaMaker fixtures: "
            f"{report['available_fixture_count']}/{report['fixture_count']} available "
            f"at {report['checkout_path']}"
        ),
        "Claim: mesh/geometry inventory only; no full equilibrium parity claim.",
    ]
    for row in rows:
        lines.append(
            f"- {row['fixture_id']}: available={row['available']}; "
            f"mesh={row['mesh']}; geometry={row['geometry']}"
        )
    return tuple(lines)


def _main_fixed_boundary_evidence(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tokamaker-jax fixed-boundary-evidence")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OPENFUSIONTOOLKIT_ROOT),
        help="OpenFUSIONToolkit checkout root.",
    )
    parser.add_argument("--json", action="store_true", help="Print the evidence report as JSON.")
    parser.add_argument("--output", "-o", help="Write the evidence report JSON to this path.")
    args = parser.parse_args(argv)

    report = fixed_boundary_upstream_report(root=args.root)
    text = fixed_boundary_report_to_json(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        for line in _fixed_boundary_evidence_summary_lines(report):
            print(line)
    return 0


def _fixed_boundary_evidence_summary_lines(report: dict[str, Any]) -> tuple[str, ...]:
    lines = [
        f"fixed-boundary evidence from {report['checkout_path']}",
        "Claim: source evidence only; no full fixed-boundary equilibrium parity claim.",
    ]
    for notebook in report["notebooks"]:
        lines.append(
            f"- {Path(notebook['path']).name}: exists={notebook['exists']}; "
            f"solve_calls={notebook.get('solve_calls', 0)}; "
            f"fixed_boundary_assignments={notebook.get('fixed_boundary_assignments', 0)}"
        )
    geqdsk = report.get("geqdsk")
    if isinstance(geqdsk, dict):
        lines.append(
            f"- {Path(geqdsk['path']).name}: {geqdsk['nr']}x{geqdsk['nz']} grid; "
            f"Ip={geqdsk['current']:.6g}; Bcentr={geqdsk['bcentr']:.6g}"
        )
    return tuple(lines)


def _main_verify(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tokamaker-jax verify")
    parser.add_argument(
        "--gate",
        choices=(
            "all",
            "poisson",
            "grad-shafranov",
            "coil-green",
            "circular-loop",
            "oft-parity",
            "profile-iteration",
            "free-boundary-profile",
        ),
        default="all",
        help="Manufactured validation gate to run.",
    )
    parser.add_argument(
        "--subdivisions",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Uniform refinement levels.",
    )
    parser.add_argument("--output", "-o", help="Write the JSON report to this path.")
    args = parser.parse_args(argv)

    payload = run_verification_gates(args.gate, tuple(args.subdivisions))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def run_verification_gates(
    gate: str = "all",
    subdivisions: tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    """Run manufactured validation gates and return a JSON-ready report."""

    if len(subdivisions) < 2:
        raise ValueError("at least two subdivisions are required")
    if any(level < 2 for level in subdivisions):
        raise ValueError("all subdivisions must be at least 2")

    from jax import config as jax_config

    from tokamaker_jax.comparison import run_openfusiontoolkit_green_comparison
    from tokamaker_jax.fem_equilibrium import run_profile_iteration_validation
    from tokamaker_jax.verification import (
        run_circular_loop_green_function_validation,
        run_coil_green_function_validation,
        run_free_boundary_profile_coupling_validation,
        run_grad_shafranov_convergence_study,
        run_poisson_convergence_study,
    )

    jax_config.update("jax_enable_x64", True)
    payload: dict[str, Any] = {"subdivisions": list(subdivisions), "gates": {}}
    if gate in {"all", "poisson"}:
        payload["gates"]["poisson"] = run_poisson_convergence_study(subdivisions).to_dict()
    if gate in {"all", "grad-shafranov"}:
        payload["gates"]["grad_shafranov"] = run_grad_shafranov_convergence_study(
            subdivisions
        ).to_dict()
    if gate in {"all", "coil-green"}:
        payload["gates"]["coil_green"] = run_coil_green_function_validation().to_dict()
    if gate in {"all", "circular-loop"}:
        payload["gates"]["circular_loop"] = run_circular_loop_green_function_validation().to_dict()
    if gate in {"all", "oft-parity"}:
        payload["gates"]["openfusiontoolkit"] = run_openfusiontoolkit_green_comparison().to_dict()
    if gate in {"all", "profile-iteration"}:
        payload["gates"]["profile_iteration"] = run_profile_iteration_validation().to_dict()
    if gate in {"all", "free-boundary-profile"}:
        payload["gates"]["free_boundary_profile"] = (
            run_free_boundary_profile_coupling_validation().to_dict()
        )
    if not payload["gates"]:
        raise ValueError(
            "gate must be one of: all, poisson, grad-shafranov, coil-green, "
            "circular-loop, oft-parity, profile-iteration, free-boundary-profile"
        )
    return payload


def _main_validate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tokamaker-jax validate")
    parser.add_argument("config", help="TOML run configuration to validate without solving.")
    parser.add_argument("--output", "-o", help="Validate this solution .npz output path.")
    parser.add_argument("--plot", help="Validate this equilibrium plot output path.")
    args = parser.parse_args(argv)
    try:
        report = validate_config(args.config, output=args.output, plot=args.plot)
    except Exception as exc:
        print(f"Validation failed: {args.config}", file=sys.stderr)
        print(f"- {exc}", file=sys.stderr)
        return 1

    print(f"Validation succeeded: {report.config_path}")
    for line in report.summary_lines():
        print(f"- {line}")
    return 0


def validate_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    plot: str | Path | None = None,
) -> ValidationReport:
    """Validate a TOML configuration without running a solve."""

    path = Path(config_path)
    try:
        config = load_config(path)
    except Exception as exc:
        raise ConfigValidationError(f"TOML/config parse error: {exc}") from exc

    errors: list[str] = []
    _validate_grid(config.grid, errors)
    _validate_source(config, errors)
    _validate_solver(config.solver, errors)
    _validate_coils(config.coils, errors)
    output_paths = _validate_outputs(config.output, output=output, plot=plot, errors=errors)

    if errors:
        details = "\n  - ".join(errors)
        raise ConfigValidationError(f"{len(errors)} validation error(s):\n  - {details}")

    return ValidationReport(
        config_path=path.resolve(),
        grid_shape=(config.grid.nr, config.grid.nz),
        region_count=0 if config.regions is None else len(config.regions.regions),
        output_paths=tuple(output_paths),
    )


def _validate_grid(grid: GridConfig, errors: list[str]) -> None:
    numeric_fields = ("r_min", "r_max", "z_min", "z_max")
    dimensions = ("nr", "nz")
    numeric_ok = True
    for field in numeric_fields:
        numeric_ok = (
            _require_finite_number(f"grid.{field}", getattr(grid, field), errors) and numeric_ok
        )
    dimensions_ok = True
    for field in dimensions:
        dimensions_ok = (
            _require_int(f"grid.{field}", getattr(grid, field), minimum=3, errors=errors)
            and dimensions_ok
        )
    if numeric_ok and dimensions_ok:
        try:
            RectangularGrid(**grid.__dict__)
        except ValueError as exc:
            errors.append(f"grid: {exc}")


def _validate_source(config: RunConfig, errors: list[str]) -> None:
    if config.source.profile != "solovev":
        errors.append("source.profile must be 'solovev' for the current seed solver")
    _require_finite_number("source.pressure_scale", config.source.pressure_scale, errors)
    _require_finite_number("source.ffp_scale", config.source.ffp_scale, errors)


def _validate_solver(solver: SolverConfig, errors: list[str]) -> None:
    _require_int("solver.iterations", solver.iterations, minimum=1, errors=errors)
    if _require_finite_number("solver.relaxation", solver.relaxation, errors) and not (
        0.0 < float(solver.relaxation) <= 1.0
    ):
        errors.append("solver.relaxation must satisfy 0 < relaxation <= 1")
    if solver.dtype not in {"float32", "float64"}:
        errors.append("solver.dtype must be 'float32' or 'float64'")


def _validate_coils(coils: tuple[CoilConfig, ...], errors: list[str]) -> None:
    names: set[str] = set()
    for index, coil in enumerate(coils):
        prefix = f"coil[{index}]"
        if not coil.name:
            errors.append(f"{prefix}.name must be nonempty")
        elif coil.name in names:
            errors.append(f"{prefix}.name {coil.name!r} is duplicated")
        names.add(coil.name)
        for field in ("r", "z", "current", "sigma"):
            _require_finite_number(f"{prefix}.{field}", getattr(coil, field), errors)
        if _is_number(coil.sigma) and float(coil.sigma) <= 0.0:
            errors.append(f"{prefix}.sigma must be positive")


def _validate_outputs(
    config: OutputConfig,
    *,
    output: str | Path | None,
    plot: str | Path | None,
    errors: list[str],
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for label, raw_path in (("npz", output or config.npz), ("plot", plot or config.plot)):
        if raw_path is None:
            continue
        path = _validate_output_path(label, raw_path, errors)
        if path is not None:
            paths.append((label, path))
    return paths


def _validate_output_path(label: str, raw_path: str | Path, errors: list[str]) -> Path | None:
    if not isinstance(raw_path, (str, Path)):
        errors.append(f"output.{label} must be a filesystem path")
        return None
    path = Path(raw_path)
    if str(path) == "":
        errors.append(f"output.{label} must not be empty")
        return None
    if path.exists() and path.is_dir():
        errors.append(f"output.{label} points to a directory, not a file: {path}")
        return path
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        errors.append(f"output.{label} parent is not a directory: {parent}")
        return path
    ancestor = parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.exists():
        errors.append(f"output.{label} has no existing ancestor directory: {path}")
    elif not os.access(ancestor, os.W_OK):
        errors.append(f"output.{label} ancestor is not writable: {ancestor}")
    return path


def _require_int(name: str, value: Any, *, minimum: int, errors: list[str]) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{name} must be an integer")
        return False
    if value < minimum:
        errors.append(f"{name} must be at least {minimum}")
        return False
    return True


def _require_finite_number(name: str, value: Any, errors: list[str]) -> bool:
    if not _is_number(value):
        errors.append(f"{name} must be a finite number")
        return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    plot: str | Path | None = None,
) -> EquilibriumSolution:
    """Run a TOML configuration and write optional artifacts."""

    config = load_config(config_path)
    solution = solve_from_config(config)
    output_path = output or config.output.npz
    if output_path:
        save_npz(solution, output_path)
    plot_path = plot or config.output.plot
    if plot_path:
        save_equilibrium_plot(solution, plot_path)
    return solution


def save_npz(solution: EquilibriumSolution, path: str | Path) -> Path:
    """Persist solution arrays in NumPy format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r, z = solution.grid.mesh(dtype=solution.psi.dtype)
    np.savez(
        path,
        r=np.asarray(r),
        z=np.asarray(z),
        psi=np.asarray(solution.psi),
        source=np.asarray(solution.source),
        residual_history=np.asarray(solution.residual_history),
    )
    return path.resolve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
