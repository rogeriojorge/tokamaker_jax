"""Optional NiceGUI frontend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.free_boundary import coil_flux_on_grid
from tokamaker_jax.geometry import Region, RegionSet, annulus_region, rectangle_region
from tokamaker_jax.plotting import equilibrium_metadata_summary, region_table_data
from tokamaker_jax.solver import solve_from_config
from tokamaker_jax.verification import (
    GradShafranovConvergenceStudy,
    PoissonConvergenceStudy,
    run_grad_shafranov_convergence_study,
    run_poisson_convergence_study,
)


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
        validation_tab = ui.tab("Validation")
        coil_tab = ui.tab("Coil response")

    with ui.tab_panels(tabs, value=equilibrium_tab).classes("w-full"):
        with ui.tab_panel(equilibrium_tab):
            with ui.row().classes("items-end"):
                pressure = ui.number("pressure scale", value=5.0e3, min=0.0, step=250.0)
                ffp = ui.number("dF2/dpsi scale", value=-0.35, step=0.05)
                iterations = ui.number("iterations", value=450, min=1, max=3000, step=50)

            figure, summary = _seed_equilibrium_payload(
                float(pressure.value), float(ffp.value), int(iterations.value)
            )
            plot = ui.plotly(figure).classes("w-full h-[620px]")
            summary_table = ui.table(
                columns=[
                    {"name": "metric", "label": "Metric", "field": "metric"},
                    {"name": "value", "label": "Value", "field": "value"},
                ],
                rows=seed_equilibrium_summary_rows(summary),
            ).classes("w-full")

            def update() -> None:
                figure, summary = _seed_equilibrium_payload(
                    float(pressure.value), float(ffp.value), int(iterations.value)
                )
                plot.figure = figure
                plot.update()
                summary_table.rows = seed_equilibrium_summary_rows(summary)
                summary_table.update()

            ui.button("Run", on_click=update)

        with ui.tab_panel(geometry_tab):
            ui.label("Sample machine regions").classes("text-subtitle2")
            regions = _sample_regions()
            ui.plotly(region_geometry_figure(regions)).classes("w-full h-[620px]")
            ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id"},
                    {"name": "name", "label": "Name", "field": "name"},
                    {"name": "kind", "label": "Kind", "field": "kind"},
                    {"name": "area", "label": "Area", "field": "area"},
                    {"name": "centroid", "label": "Centroid (R,Z)", "field": "centroid"},
                    {"name": "target_size", "label": "Target size", "field": "target_size"},
                ],
                rows=region_table_rows(regions),
            ).classes("w-full")
        with ui.tab_panel(validation_tab):
            ui.plotly(validation_convergence_figure("grad-shafranov")).classes("w-full h-[620px]")
        with ui.tab_panel(coil_tab):
            ui.plotly(coil_green_response_figure()).classes("w-full h-[620px]")
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
        meta={"regions": region_table_data(region_tuple)},
    )
    return fig


def seed_equilibrium_figure(
    pressure_scale: float,
    ffp_scale: float,
    iterations: int,
):
    """Return a Plotly seed-equilibrium figure with validation metadata attached."""

    figure, _ = _seed_equilibrium_payload(pressure_scale, ffp_scale, iterations)
    return figure


def validation_convergence_figure(gate: str = "grad-shafranov"):
    """Return a Plotly figure for an implemented manufactured validation gate."""

    import plotly.graph_objects as go

    if gate == "poisson":
        study = run_poisson_convergence_study((4, 8, 16))
        return _convergence_figure_from_study(
            go,
            "p=1 manufactured Poisson convergence",
            study,
            h1_key="h1_error",
            h1_label="H1 seminorm error",
            h1_rates=study.h1_rates,
        )
    if gate == "grad-shafranov":
        study = run_grad_shafranov_convergence_study((4, 8, 16))
        return _convergence_figure_from_study(
            go,
            "axisymmetric Grad-Shafranov weak-form convergence",
            study,
            h1_key="weighted_h1_error",
            h1_label="weighted H1 seminorm error",
            h1_rates=study.weighted_h1_rates,
        )
    raise ValueError("gate must be 'poisson' or 'grad-shafranov'")


def coil_green_response_figure():
    """Return a Plotly figure for the reduced free-boundary coil response."""

    import plotly.graph_objects as go

    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, 75, 75)
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )
    r, z = grid.mesh()
    flux = coil_flux_on_grid(grid, coils)
    fig = go.Figure(
        data=[
            go.Contour(
                x=np.asarray(r[:, 0]),
                y=np.asarray(z[0, :]),
                z=np.asarray(flux).T,
                contours_coloring="heatmap",
                colorbar={"title": "coil flux"},
                hovertemplate="R=%{x:.3f} m<br>Z=%{y:.3f} m<br>psi=%{z:.3e}<extra></extra>",
            ),
            go.Scatter(
                x=[coil.r for coil in coils],
                y=[coil.z for coil in coils],
                mode="markers+text",
                text=[coil.name for coil in coils],
                textposition="middle right",
                marker={
                    "symbol": "square",
                    "size": 12,
                    "color": ["#d62728" if coil.current >= 0.0 else "#1f77b4" for coil in coils],
                    "line": {"color": "black", "width": 1},
                },
                name="PF coils",
                hovertemplate="coil %{text}<br>R=%{x:.3f} m<br>Z=%{y:.3f} m<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        title="reduced free-boundary coil Green's response",
        xaxis_title="R [m]",
        yaxis_title="Z [m]",
        yaxis_scaleanchor="x",
        template="plotly_white",
        margin={"l": 44, "r": 20, "t": 48, "b": 42},
        meta={
            "n_coils": len(coils),
            "coils": [
                {"name": coil.name, "r": coil.r, "z": coil.z, "current": coil.current}
                for coil in coils
            ],
        },
    )
    return fig


def seed_equilibrium_summary_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Return GUI-ready rows for an equilibrium metadata summary."""

    psi_range = summary["psi"]["range"]
    source_range = summary["source"]["range"]
    residual = summary["residual"]
    grid = summary["grid"]
    return [
        {"metric": "grid", "value": f"{grid['nr']} x {grid['nz']}"},
        {"metric": "spacing", "value": f"dR={grid['dr']:.4g} m, dZ={grid['dz']:.4g} m"},
        {"metric": "iterations", "value": str(summary["iterations"])},
        {
            "metric": "psi range",
            "value": f"{_format_number(psi_range['min'])} to {_format_number(psi_range['max'])}",
        },
        {
            "metric": "source range",
            "value": f"{_format_number(source_range['min'])} to "
            f"{_format_number(source_range['max'])}",
        },
        {"metric": "final residual", "value": _format_number(residual["final"])},
    ]


def region_table_rows(regions: RegionSet | Sequence[Region]) -> list[dict[str, Any]]:
    """Return GUI-ready region table rows with rounded display fields."""

    rows = []
    for row in region_table_data(regions):
        rows.append(
            {
                **row,
                "area": _format_number(row["area"]),
                "centroid": f"({_format_number(row['centroid_r'])}, "
                f"{_format_number(row['centroid_z'])})",
                "target_size": ""
                if row["target_size"] is None
                else _format_number(row["target_size"]),
            }
        )
    return rows


def _seed_equilibrium_payload(
    pressure_scale: float,
    ffp_scale: float,
    iterations: int,
):
    import plotly.graph_objects as go

    config = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=pressure_scale, ffp_scale=ffp_scale),
        solver=SolverConfig(iterations=iterations, relaxation=0.75, dtype="float64"),
    )
    solution = solve_from_config(config)
    summary = equilibrium_metadata_summary(solution)
    r, z = solution.grid.mesh(dtype=solution.psi.dtype)
    fig = go.Figure(
        data=[
            go.Contour(
                x=jnp.asarray(r[:, 0]),
                y=jnp.asarray(z[0, :]),
                z=jnp.asarray(solution.psi).T,
                contours_coloring="heatmap",
                colorbar={"title": "psi"},
                hovertemplate="R=%{x:.3f} m<br>Z=%{y:.3f} m<br>psi=%{z:.3e}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="R [m]",
        yaxis_title="Z [m]",
        yaxis_scaleanchor="x",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        meta={"summary": summary},
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=_seed_summary_annotation(summary),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.84)",
        bordercolor="rgba(0,0,0,0.22)",
        borderwidth=1,
        font={"size": 12},
    )
    return fig, summary


def _convergence_figure_from_study(
    go,
    title: str,
    study: PoissonConvergenceStudy | GradShafranovConvergenceStudy,
    *,
    h1_key: str,
    h1_label: str,
    h1_rates: tuple[float, ...],
):
    results = study.to_dict()["results"]
    h = np.asarray([result["h"] for result in results])
    l2 = np.asarray([result["l2_error"] for result in results])
    h1 = np.asarray([result[h1_key] for result in results])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h, y=l2, mode="lines+markers", name="L2 error"))
    fig.add_trace(go.Scatter(x=h, y=h1, mode="lines+markers", name=h1_label))
    fig.add_trace(
        go.Scatter(
            x=h,
            y=l2[-1] * (h / h[-1]) ** 2,
            mode="lines",
            name="O(h^2)",
            line={"dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h,
            y=h1[-1] * (h / h[-1]),
            mode="lines",
            name="O(h)",
            line={"dash": "dash"},
        )
    )
    fig.update_xaxes(title="mesh size h", type="log", autorange="reversed")
    fig.update_yaxes(title="error", type="log")
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 48, "r": 20, "t": 48, "b": 42},
        legend_title_text="Metric",
        meta={
            "results": results,
            "l2_rates": list(study.l2_rates),
            "h1_rates": list(h1_rates),
        },
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


def _seed_summary_annotation(summary: dict[str, Any]) -> str:
    psi_range = summary["psi"]["range"]
    residual = summary["residual"]
    return "<br>".join(
        (
            f"grid {summary['grid']['nr']} x {summary['grid']['nz']}",
            f"iterations {summary['iterations']}",
            f"psi {_format_number(psi_range['min'])} to {_format_number(psi_range['max'])}",
            f"residual {_format_number(residual['final'])}",
        )
    )


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3g}"
