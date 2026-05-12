"""Generate README and documentation visual assets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jax import config as jax_config
from matplotlib.animation import FuncAnimation, PillowWriter
from reproduce_cpc_seed_family import generate_cpc_seed_family_artifacts

from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.fem_equilibrium import (
    NonlinearProfileParameters,
    PowerProfile,
    solve_profile_iteration_on_rectangle,
)
from tokamaker_jax.geometry import sample_regions
from tokamaker_jax.plotting import (
    plot_equilibrium,
    save_coil_green_response_plot,
    save_equilibrium_plot,
    save_region_plot,
)
from tokamaker_jax.solver import solve_from_config
from tokamaker_jax.verification import (
    run_grad_shafranov_convergence_study,
    run_poisson_convergence_study,
)

ASSET_DIR = Path("docs/_static")

jax_config.update("jax_enable_x64", True)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    base = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=5000.0, ffp_scale=-0.35),
        solver=SolverConfig(iterations=700, relaxation=0.75, dtype="float64"),
    )
    save_equilibrium_plot(solve_from_config(base), ASSET_DIR / "fixed_boundary_seed.png")
    write_region_geometry_preview()
    write_manufactured_poisson_convergence()
    write_grad_shafranov_convergence()
    write_coil_green_response()
    write_profile_iteration()
    write_cpc_seed_family()
    write_pressure_sweep()


def write_region_geometry_preview() -> None:
    save_region_plot(sample_regions(), ASSET_DIR / "region_geometry_seed.png")


def write_manufactured_poisson_convergence() -> None:
    study = run_poisson_convergence_study((4, 8, 16))
    h = np.asarray([result.h for result in study.results])
    l2 = np.asarray([result.l2_error for result in study.results])
    h1 = np.asarray([result.h1_error for result in study.results])

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.loglog(h, l2, "o-", label="L2 error")
    ax.loglog(h, h1, "s-", label="H1 seminorm error")
    ax.loglog(h, l2[-1] * (h / h[-1]) ** 2, "--", color="C0", alpha=0.55, label="O(h^2)")
    ax.loglog(h, h1[-1] * (h / h[-1]), "--", color="C1", alpha=0.55, label="O(h)")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error")
    ax.set_title("p=1 manufactured Poisson convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(ASSET_DIR / "manufactured_poisson_convergence.png", dpi=180)
    plt.close(fig)


def write_grad_shafranov_convergence() -> None:
    study = run_grad_shafranov_convergence_study((4, 8, 16))
    h = np.asarray([result.h for result in study.results])
    l2 = np.asarray([result.l2_error for result in study.results])
    h1 = np.asarray([result.weighted_h1_error for result in study.results])

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.loglog(h, l2, "o-", label="L2 error")
    ax.loglog(h, h1, "s-", label="weighted H1 seminorm error")
    ax.loglog(h, l2[-1] * (h / h[-1]) ** 2, "--", color="C0", alpha=0.55, label="O(h^2)")
    ax.loglog(h, h1[-1] * (h / h[-1]), "--", color="C1", alpha=0.55, label="O(h)")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error")
    ax.set_title("axisymmetric Grad-Shafranov manufactured convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(ASSET_DIR / "manufactured_grad_shafranov_convergence.png", dpi=180)
    plt.close(fig)


def write_coil_green_response() -> None:
    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, 101, 101)
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )
    save_coil_green_response_plot(grid, coils, ASSET_DIR / "coil_green_response.png")


def write_profile_iteration() -> None:
    solution = solve_profile_iteration_on_rectangle(
        subdivisions=16,
        parameters=NonlinearProfileParameters(
            pressure=PowerProfile(scale=4.0e3, alpha=1.25, gamma=1.0),
            ffprime=PowerProfile(scale=-0.25, alpha=1.0, gamma=1.0),
        ),
        iterations=5,
        relaxation=0.85,
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    contour = ax.tricontourf(
        np.asarray(solution.nodes[:, 0]),
        np.asarray(solution.nodes[:, 1]),
        np.asarray(solution.triangles),
        np.asarray(solution.psi),
        levels=24,
        cmap="viridis",
    )
    ax.tricontour(
        np.asarray(solution.nodes[:, 0]),
        np.asarray(solution.nodes[:, 1]),
        np.asarray(solution.triangles),
        np.asarray(solution.psi),
        levels=12,
        colors="black",
        linewidths=0.45,
    )
    fig.colorbar(contour, ax=ax, label="psi")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("nonlinear p=1 FEM profile iteration")
    fig.savefig(ASSET_DIR / "profile_iteration.png", dpi=180)
    plt.close(fig)


def write_cpc_seed_family() -> None:
    generate_cpc_seed_family_artifacts(
        ASSET_DIR,
        nr=31,
        nz=31,
        iterations=120,
        report_name="cpc_seed_family_report.json",
        png_name="cpc_seed_family.png",
        command="python examples/reproduce_cpc_seed_family.py docs/_static",
    )


def write_pressure_sweep() -> None:
    pressures = np.linspace(1500.0, 8500.0, 12)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)

    def update(frame: int):
        ax.clear()
        config = RunConfig(
            grid=GridConfig(nr=41, nz=41),
            source=SourceConfig(pressure_scale=float(pressures[frame]), ffp_scale=-0.35),
            solver=SolverConfig(iterations=220, relaxation=0.75, dtype="float64"),
        )
        solution = solve_from_config(config)
        plot_equilibrium(solution, levels=18, ax=ax, show_source=False)
        ax.set_title(f"pressure scale = {pressures[frame]:.0f}")
        return []

    animation = FuncAnimation(fig, update, frames=len(pressures), blit=False)
    animation.save(ASSET_DIR / "pressure_sweep.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


if __name__ == "__main__":
    main()
