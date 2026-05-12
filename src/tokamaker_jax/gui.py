"""Optional NiceGUI frontend."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from tokamaker_jax.config import GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.geometry import Region, RegionSet, annulus_region, rectangle_region
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

    with ui.tabs().classes("w-full") as tabs:
        equilibrium_tab = ui.tab("Seed equilibrium")
        geometry_tab = ui.tab("Region geometry")

    with ui.tab_panels(tabs, value=equilibrium_tab).classes("w-full"):
        with ui.tab_panel(equilibrium_tab):
            with ui.row().classes("items-end"):
                pressure = ui.number("pressure scale", value=5.0e3, min=0.0, step=250.0)
                ffp = ui.number("dF2/dpsi scale", value=-0.35, step=0.05)
                iterations = ui.number("iterations", value=450, min=1, max=3000, step=50)

            plot = ui.plotly(
                _figure(float(pressure.value), float(ffp.value), int(iterations.value))
            ).classes("w-full h-[620px]")

            def update() -> None:
                plot.figure = _figure(
                    float(pressure.value), float(ffp.value), int(iterations.value)
                )
                plot.update()

            ui.button("Run", on_click=update)

        with ui.tab_panel(geometry_tab):
            ui.label("Sample machine regions").classes("text-subtitle2")
            ui.plotly(region_geometry_figure()).classes("w-full h-[620px]")
    ui.run(host=host, port=port, reload=reload, show=True)


def region_geometry_figure(
    regions: RegionSet | Sequence[Region] | None = None,
    *,
    show_labels: bool = True,
):
    """Return a Plotly figure for region geometry without launching the GUI."""

    import plotly.graph_objects as go

    region_tuple = _region_tuple(_sample_regions() if regions is None else regions)
    if not region_tuple:
        raise ValueError("regions must contain at least one region")

    fig = go.Figure()
    colors = {
        "plasma": "rgba(31, 119, 180, 0.42)",
        "vacuum": "rgba(148, 103, 189, 0.28)",
        "conductor": "rgba(255, 127, 14, 0.38)",
        "coil": "rgba(214, 39, 40, 0.42)",
        "boundary": "rgba(44, 160, 44, 0.34)",
        "limiter": "rgba(140, 86, 75, 0.38)",
        "unknown": "rgba(127, 127, 127, 0.32)",
    }

    for region in region_tuple:
        fig.add_trace(
            _loop_trace(
                go,
                region.points,
                name=f"{region.name} ({region.kind})",
                fillcolor=colors.get(region.kind, colors["unknown"]),
                line_color="black",
            )
        )
        for hole_index, hole in enumerate(region.holes, start=1):
            fig.add_trace(
                _loop_trace(
                    go,
                    hole,
                    name=f"{region.name} hole {hole_index}",
                    fillcolor="rgba(255, 255, 255, 1.0)",
                    line_color="black",
                    showlegend=False,
                )
            )
        if show_labels:
            r, z = _region_label_position(region)
            fig.add_annotation(x=r, y=z, text=region.name, showarrow=False, font={"size": 12})

    all_points = np.vstack([region.points for region in region_tuple])
    margin = 0.08 * max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1])), 1.0)
    fig.update_layout(
        title="tokamaker-jax region geometry",
        xaxis_title="R [m]",
        yaxis_title="Z [m]",
        yaxis={"scaleanchor": "x", "scaleratio": 1.0},
        xaxis_range=[
            float(np.min(all_points[:, 0]) - margin),
            float(np.max(all_points[:, 0]) + margin),
        ],
        yaxis_range=[
            float(np.min(all_points[:, 1]) - margin),
            float(np.max(all_points[:, 1]) + margin),
        ],
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        legend_title_text="Regions",
    )
    return fig


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


def _region_tuple(regions: RegionSet | Sequence[Region]) -> tuple[Region, ...]:
    return regions.regions if isinstance(regions, RegionSet) else tuple(regions)


def _sample_regions() -> RegionSet:
    import tokamaker_jax.geometry as geometry

    for name in ("sample_regions", "sample_region_set"):
        provider = getattr(geometry, name, None)
        if provider is not None:
            value = provider()
            return value if isinstance(value, RegionSet) else RegionSet(tuple(value))

    value = getattr(geometry, "SAMPLE_REGIONS", None)
    if value is not None:
        return value if isinstance(value, RegionSet) else RegionSet(tuple(value))

    return RegionSet(
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


def _loop_trace(
    go,
    points: np.ndarray,
    *,
    name: str,
    fillcolor: str,
    line_color: str,
    showlegend: bool = True,
):
    closed = np.vstack([points, points[0]])
    return go.Scatter(
        x=closed[:, 0],
        y=closed[:, 1],
        mode="lines",
        name=name,
        fill="toself",
        fillcolor=fillcolor,
        line={"color": line_color, "width": 1.2},
        showlegend=showlegend,
        hovertemplate="R=%{x:.3f} m<br>Z=%{y:.3f} m<extra>%{fullData.name}</extra>",
    )


def _region_label_position(region: Region) -> tuple[float, float]:
    if not region.holes:
        return region.centroid
    inner_max_r = max(float(np.max(hole[:, 0])) for hole in region.holes)
    outer_max_r = float(np.max(region.points[:, 0]))
    _, center_z = region.centroid
    return 0.5 * (inner_max_r + outer_max_r), center_z
