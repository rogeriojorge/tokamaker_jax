"""Generate README and documentation visual assets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from tokamaker_jax.config import GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.geometry import RegionSet, annulus_region, rectangle_region
from tokamaker_jax.plotting import plot_equilibrium, save_equilibrium_plot, save_region_plot
from tokamaker_jax.solver import solve_from_config

ASSET_DIR = Path("docs/_static")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    base = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=5000.0, ffp_scale=-0.35),
        solver=SolverConfig(iterations=700, relaxation=0.75, dtype="float64"),
    )
    save_equilibrium_plot(solve_from_config(base), ASSET_DIR / "fixed_boundary_seed.png")
    write_region_geometry_preview()
    write_pressure_sweep()


def write_region_geometry_preview() -> None:
    regions = RegionSet(
        (
            annulus_region(
                id=2,
                name="VV",
                kind="conductor",
                center_r=2.0,
                center_z=0.0,
                inner_radius=1.05,
                outer_radius=1.25,
                n=96,
                target_size=0.05,
            ),
            rectangle_region(
                id=1,
                name="PLASMA",
                kind="plasma",
                r_min=1.35,
                r_max=2.65,
                z_min=-0.75,
                z_max=0.75,
                target_size=0.08,
            ),
            rectangle_region(
                id=3,
                name="PF",
                kind="coil",
                r_min=3.25,
                r_max=3.55,
                z_min=-0.25,
                z_max=0.25,
                target_size=0.04,
            ),
        )
    )
    save_region_plot(regions, ASSET_DIR / "region_geometry_seed.png")


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
