"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokamaker_jax.config import load_config
from tokamaker_jax.plotting import save_equilibrium_plot
from tokamaker_jax.solver import EquilibriumSolution, solve_from_config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    With no TOML path this launches the GUI. With a TOML path it runs a
    reproducible noninteractive solve.
    """

    parser = argparse.ArgumentParser(prog="tokamaker-jax")
    parser.add_argument("config", nargs="?", help="TOML run configuration. Omit to launch the GUI.")
    parser.add_argument("--output", "-o", help="Write solution arrays to this .npz file.")
    parser.add_argument("--plot", help="Write an equilibrium plot to this image path.")
    args = parser.parse_args(argv)
    if args.config is None:
        from tokamaker_jax.gui import launch_gui

        launch_gui()
        return 0
    solution = run_config(args.config, output=args.output, plot=args.plot)
    print(json.dumps(solution.stats(), indent=2, sort_keys=True))
    return 0


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
