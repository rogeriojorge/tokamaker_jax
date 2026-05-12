"""Optional NiceGUI frontend."""

from __future__ import annotations

import jax.numpy as jnp

from tokamaker_jax.config import GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.solver import solve_from_config


def launch_gui(host: str = "127.0.0.1", port: int = 8080, reload: bool = False) -> None:
    """Launch a small interactive GUI.

    The complete GUI is planned as a richer workflow builder; this seed UI keeps
    the default CLI behavior useful while the solver port matures.
    """

    try:
        from nicegui import ui
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit('Install GUI dependencies with: pip install "tokamaker-jax[gui]"') from exc

    ui.page_title("tokamaker-jax")
    ui.label("tokamaker-jax").classes("text-h4")
    ui.label("Differentiable fixed-boundary seed equilibrium").classes("text-subtitle1")

    with ui.row().classes("items-end"):
        pressure = ui.number("pressure scale", value=5.0e3, min=0.0, step=250.0)
        ffp = ui.number("dF2/dpsi scale", value=-0.35, step=0.05)
        iterations = ui.number("iterations", value=450, min=1, max=3000, step=50)

    plot = ui.plotly(
        _figure(float(pressure.value), float(ffp.value), int(iterations.value))
    ).classes("w-full h-[620px]")

    def update() -> None:
        plot.figure = _figure(float(pressure.value), float(ffp.value), int(iterations.value))
        plot.update()

    ui.button("Run", on_click=update)
    ui.run(host=host, port=port, reload=reload, show=True)


def _figure(pressure_scale: float, ffp_scale: float, iterations: int):
    import plotly.graph_objects as go

    config = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=pressure_scale, ffp_scale=ffp_scale),
        solver=SolverConfig(iterations=iterations, relaxation=0.75, dtype="float64"),
    )
    solution = solve_from_config(config)
    r, z = solution.grid.mesh(dtype=solution.psi.dtype)
    fig = go.Figure(
        data=[
            go.Contour(
                x=jnp.asarray(r[:, 0]),
                y=jnp.asarray(z[0, :]),
                z=jnp.asarray(solution.psi).T,
                contours_coloring="heatmap",
                colorbar={"title": "psi"},
            )
        ]
    )
    fig.update_layout(
        xaxis_title="R [m]",
        yaxis_title="Z [m]",
        yaxis_scaleanchor="x",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    return fig
