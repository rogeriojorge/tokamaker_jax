"""Optional NiceGUI frontend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from tokamaker_jax.cases import (
    CaseManifest,
    case_table_rows,
    default_case_manifest,
)
from tokamaker_jax.cli import (
    _validate_coils,
    _validate_grid,
    _validate_outputs,
    _validate_solver,
    _validate_source,
)
from tokamaker_jax.config import (
    CoilConfig,
    GridConfig,
    RunConfig,
    SolverConfig,
    SourceConfig,
    config_from_dict,
)
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.free_boundary import coil_flux_on_grid
from tokamaker_jax.geometry import Region, RegionSet, annulus_region, rectangle_region
from tokamaker_jax.plotting import equilibrium_metadata_summary, region_table_data
from tokamaker_jax.solver import solve_from_config
from tokamaker_jax.upstream_fixtures import upstream_fixture_rows
from tokamaker_jax.verification import (
    GradShafranovConvergenceStudy,
    PoissonConvergenceStudy,
    run_coil_green_function_validation,
    run_grad_shafranov_convergence_study,
    run_poisson_convergence_study,
)

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

_DEFAULT_VALIDATION_SUBDIVISIONS = (4, 8, 16)
_CONVERGENCE_RATE_THRESHOLDS = {"l2": 1.8, "h1": 0.85}
_COIL_GREEN_ERROR_TOLERANCE = 1.0e-10
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_ARTIFACTS = {
    "validation": Path("outputs/verify.json"),
    "openfusiontoolkit": Path("docs/_static/openfusiontoolkit_comparison_report.json"),
    "upstream_fixtures": Path("docs/_static/upstream_fixture_summary.json"),
    "benchmark": Path("docs/_static/benchmark_report.json"),
}


def launch_gui(host: str = "127.0.0.1", port: int = 8080, reload: bool = False) -> None:
    """Launch a small interactive GUI.

    The complete GUI is planned as a richer workflow builder; this seed UI keeps
    the default CLI behavior useful while the solver port matures.
    """

    try:
        from nicegui import ui
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("Install GUI dependencies with: pip install tokamaker-jax") from exc

    ui.page_title("tokamaker-jax")
    ui.label("tokamaker-jax").classes("text-h4")
    ui.label("Differentiable fixed-boundary seed equilibrium").classes("text-subtitle1")

    with ui.tabs().classes("w-full") as tabs:
        workflow_tab = ui.tab("Workflow")
        equilibrium_tab = ui.tab("Seed equilibrium")
        geometry_tab = ui.tab("Region geometry")
        validation_tab = ui.tab("Validation")
        coil_tab = ui.tab("Coil response")
        cases_tab = ui.tab("Cases")
        reports_tab = ui.tab("Reports")

    with ui.tab_panels(tabs, value=workflow_tab).classes("w-full"):
        with ui.tab_panel(workflow_tab):
            dashboard = workflow_dashboard_data(iterations=120)
            ui.label("Workflow state").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "section", "label": "Section", "field": "section"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "metric", "label": "Metric", "field": "metric"},
                    {"name": "command", "label": "Command", "field": "command"},
                ],
                rows=workflow_status_rows(dashboard),
            ).classes("w-full")
            ui.label("Validation gates").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "gate", "label": "Gate", "field": "gate"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "metric", "label": "Metric", "field": "metric"},
                    {"name": "command", "label": "Command", "field": "command"},
                ],
                rows=validation_gate_rows(dashboard),
            ).classes("w-full")
            ui.label("Open next steps").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "step", "label": "Step", "field": "step"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "command", "label": "Command", "field": "command"},
                ],
                rows=workflow_next_step_rows(dashboard),
            ).classes("w-full")
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
        with ui.tab_panel(cases_tab):
            manifest = default_case_manifest()
            ui.label("Runnable cases and upstream parity targets").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "case_id", "label": "Case", "field": "case_id"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "category", "label": "Category", "field": "category"},
                    {"name": "parity_level", "label": "Parity level", "field": "parity_level"},
                    {"name": "command", "label": "Command", "field": "command"},
                    {"name": "validation_gate", "label": "Gate", "field": "validation_gate"},
                ],
                rows=case_manifest_rows(manifest),
            ).classes("w-full")

            preview_options = [entry.case_id for entry in manifest.entries if entry.path]
            initial_case = preview_options[0]
            source_payload = load_case_source_text(
                initial_case,
                manifest=manifest,
                root=_PROJECT_ROOT,
            )
            validation = _case_editor_validation(source_payload)
            selector = ui.select(
                preview_options,
                value=initial_case,
                label="Case file",
            ).classes("w-96")
            source_box = (
                ui.textarea(
                    "Source editor",
                    value=str(source_payload["source"]),
                )
                .classes("w-full font-mono")
                .props("autogrow")
            )
            command_label = ui.label(_case_preview_label(initial_case, manifest))
            validation_table = ui.table(
                columns=[
                    {"name": "check", "label": "Check", "field": "check"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "detail", "label": "Detail", "field": "detail"},
                ],
                rows=toml_validation_rows(validation),
            ).classes("w-full")
            command_table = ui.table(
                columns=[
                    {"name": "action", "label": "Action", "field": "action"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "detail", "label": "Detail", "field": "detail"},
                    {"name": "command", "label": "Command", "field": "command"},
                ],
                rows=case_validation_run_rows(
                    initial_case,
                    manifest=manifest,
                    root=_PROJECT_ROOT,
                    validation=validation,
                ),
            ).classes("w-full")

            def update_preview() -> None:
                case_id = str(selector.value)
                source_payload = load_case_source_text(
                    case_id,
                    manifest=manifest,
                    root=_PROJECT_ROOT,
                )
                validation = _case_editor_validation(source_payload)
                source_box.value = str(source_payload["source"])
                source_box.update()
                command_label.text = _case_preview_label(case_id, manifest)
                command_label.update()
                validation_table.rows = toml_validation_rows(validation)
                validation_table.update()
                command_table.rows = case_validation_run_rows(
                    case_id,
                    manifest=manifest,
                    root=_PROJECT_ROOT,
                    validation=validation,
                )
                command_table.update()

            def validate_editor() -> None:
                case_id = str(selector.value)
                entry = manifest.by_id(case_id)
                path = entry.path or "<editor>"
                if not path.lower().endswith(".toml"):
                    validation = {
                        "source_name": path,
                        "status": "n/a",
                        "message": "source file is not a TOML config",
                        "errors": [],
                        "grid_shape": [],
                        "region_count": 0,
                        "output_paths": [],
                    }
                else:
                    validation = validate_toml_text(str(source_box.value or ""), source_name=path)
                validation_table.rows = toml_validation_rows(validation)
                validation_table.update()
                command_table.rows = case_validation_run_rows(
                    case_id,
                    manifest=manifest,
                    root=_PROJECT_ROOT,
                    validation=validation,
                )
                command_table.update()

            selector.on_value_change(lambda _: update_preview())
            ui.button("Validate editor text", on_click=validate_editor)
        with ui.tab_panel(reports_tab):
            artifacts = load_gui_report_artifacts()
            ui.label("Validation reports").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "gate", "label": "Gate", "field": "gate"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "metric", "label": "Metric", "field": "metric"},
                    {"name": "source", "label": "Source", "field": "source"},
                ],
                rows=validation_report_rows(artifacts),
            ).classes("w-full")
            ui.label("Benchmark report").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "lane", "label": "Lane", "field": "lane"},
                    {"name": "median_ms", "label": "Median [ms]", "field": "median_ms"},
                    {"name": "best_ms", "label": "Best [ms]", "field": "best_ms"},
                    {"name": "worst_ms", "label": "Worst [ms]", "field": "worst_ms"},
                    {"name": "metadata", "label": "Metadata", "field": "metadata"},
                ],
                rows=benchmark_report_rows(artifacts.get("benchmark")),
            ).classes("w-full")
            ui.label("Upstream fixture inventory").classes("text-subtitle2")
            ui.table(
                columns=[
                    {"name": "fixture_id", "label": "Fixture", "field": "fixture_id"},
                    {"name": "available", "label": "Available", "field": "available"},
                    {"name": "category", "label": "Category", "field": "category"},
                    {"name": "mesh", "label": "Mesh", "field": "mesh"},
                    {"name": "geometry", "label": "Geometry", "field": "geometry"},
                    {"name": "claim", "label": "Claim", "field": "claim"},
                ],
                rows=upstream_fixture_report_rows(artifacts.get("upstream_fixtures")),
            ).classes("w-full")
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

    grid, coils = _default_coil_response_inputs()
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


def workflow_dashboard_data(
    *,
    pressure_scale: float = 5.0e3,
    ffp_scale: float = -0.35,
    iterations: int = 120,
    regions: RegionSet | Sequence[Region] | None = None,
    validation_subdivisions: Sequence[int] = _DEFAULT_VALIDATION_SUBDIVISIONS,
) -> dict[str, Any]:
    """Return JSON-ready dashboard data for the GUI workflow."""

    subdivisions = _validation_subdivisions_tuple(validation_subdivisions)
    region_tuple = _region_tuple(_sample_regions() if regions is None else regions)
    if not region_tuple:
        raise ValueError("regions must contain at least one region")

    seed = _seed_equilibrium_dashboard(pressure_scale, ffp_scale, iterations)
    geometry = _region_geometry_dashboard(region_tuple)
    validation = _validation_dashboard(subdivisions)
    coil_response = _coil_response_dashboard()
    sections = [
        _section_summary(
            "seed_equilibrium",
            "Seed equilibrium",
            seed["status"],
            (
                f"residual {_format_number(seed['metrics']['residual_initial'])} -> "
                f"{_format_number(seed['metrics']['residual_final'])}"
            ),
            seed["command"],
        ),
        _section_summary(
            "region_geometry",
            "Region geometry",
            geometry["status"],
            (
                f"{geometry['metrics']['n_regions']} regions; "
                f"{', '.join(geometry['metrics']['kinds'])}"
            ),
            geometry["command"],
        ),
        _section_summary(
            "validation",
            "Validation",
            validation["status"],
            f"{len(validation['gates'])} gates; min status {validation['status']}",
            validation["command"],
        ),
        _section_summary(
            "coil_response",
            "Coil response",
            coil_response["status"],
            (
                f"{coil_response['metrics']['n_coils']} coils; "
                f"|psi|max {_format_number(coil_response['metrics']['flux_abs_max'])}"
            ),
            coil_response["command"],
        ),
    ]
    dashboard = {
        "schema_version": 1,
        "workflow": {
            "name": "tokamaker-jax GUI workflow",
            "status": _rollup_status(section["status"] for section in sections),
        },
        "commands": [
            {
                "id": "validate_config",
                "label": "Validate config",
                "command": "tokamaker-jax validate examples/fixed_boundary.toml",
            },
            {
                "id": "solve_seed",
                "label": "Run seed solve",
                "command": "tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png",
            },
            {
                "id": "verify_all",
                "label": "Run validation gates",
                "command": _validation_command("all", subdivisions),
            },
        ],
        "sections": sections,
        "seed_equilibrium": seed,
        "region_geometry": geometry,
        "validation": validation,
        "coil_response": coil_response,
    }
    dashboard["next_steps"] = _workflow_next_steps(dashboard)
    return dashboard


def workflow_status_rows(dashboard: dict[str, Any]) -> list[dict[str, str]]:
    """Return compact GUI table rows for dashboard section status."""

    return [
        {
            "section": section["label"],
            "status": section["status"],
            "metric": section["metric"],
            "command": section["command"],
        }
        for section in dashboard["sections"]
    ]


def validation_gate_rows(dashboard: dict[str, Any]) -> list[dict[str, str]]:
    """Return compact GUI table rows for validation gates."""

    return [
        {
            "gate": gate["label"],
            "status": gate["status"],
            "metric": gate["summary"],
            "command": gate["command"],
        }
        for gate in dashboard["validation"]["gates"]
    ]


def workflow_next_step_rows(dashboard: dict[str, Any]) -> list[dict[str, str]]:
    """Return compact GUI table rows for open workflow items."""

    return [
        {
            "step": step["label"],
            "status": step["status"],
            "command": step["command"],
        }
        for step in dashboard["next_steps"]
    ]


def case_manifest_rows(manifest: CaseManifest | None = None) -> list[dict[str, str]]:
    """Return compact GUI table rows for runnable and planned cases."""

    rows = []
    for row in case_table_rows(default_case_manifest() if manifest is None else manifest):
        rows.append(
            {
                **row,
                "command": row["command"] or "planned",
                "validation_gate": row["validation_gate"] or "",
                "outputs": row["outputs"] or "",
            }
        )
    return rows


def load_case_source_text(
    case_id: str,
    *,
    manifest: CaseManifest | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the full source text for a manifest case without truncating it."""

    if manifest is None:
        case_manifest = default_case_manifest(_PROJECT_ROOT if root is None else root)
    else:
        case_manifest = manifest
    root_path = Path(root) if root is not None else case_manifest.root
    entry = case_manifest.by_id(case_id)
    payload: dict[str, Any] = {
        "case_id": entry.case_id,
        "title": entry.title,
        "status": entry.status,
        "category": entry.category,
        "parity_level": entry.parity_level,
        "path": entry.path or "",
        "absolute_path": "",
        "exists": False,
        "source_kind": "",
        "source": "",
        "message": "",
    }
    if entry.path is None:
        payload["message"] = (
            "This case is represented by a command or upstream fixture, not a local file."
        )
        return payload

    relative_path = Path(entry.path)
    source_path = relative_path if relative_path.is_absolute() else root_path / relative_path
    payload["absolute_path"] = str(source_path.resolve(strict=False))
    payload["source_kind"] = source_path.suffix.lstrip(".") or "source"
    if not source_path.exists():
        payload["message"] = f"Case file is not present: {entry.path}"
        return payload

    try:
        payload["source"] = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        payload["message"] = f"Case file could not be read: {exc}"
        return payload
    payload["exists"] = True
    return payload


def validate_toml_text(
    toml_text: str,
    *,
    source_name: str = "<editor>",
    output: str | Path | None = None,
    plot: str | Path | None = None,
) -> dict[str, Any]:
    """Validate TOML text in memory without writing a temporary config file."""

    source_label = str(source_name)
    try:
        config = config_from_dict(tomllib.loads(toml_text))
    except Exception as exc:
        message = f"TOML/config parse error: {exc}"
        return {
            "source_name": source_label,
            "status": "fail",
            "message": message,
            "errors": [message],
            "grid_shape": [],
            "region_count": 0,
            "output_paths": [],
        }

    errors: list[str] = []
    _validate_grid(config.grid, errors)
    _validate_source(config, errors)
    _validate_solver(config.solver, errors)
    _validate_coils(config.coils, errors)
    output_paths = _validate_outputs(config.output, output=output, plot=plot, errors=errors)
    payload = {
        "source_name": source_label,
        "status": "pass" if not errors else "fail",
        "message": "TOML config is valid" if not errors else f"{len(errors)} validation error(s)",
        "errors": errors,
        "grid_shape": [config.grid.nr, config.grid.nz],
        "region_count": 0 if config.regions is None else len(config.regions.regions),
        "output_paths": [{"label": str(label), "path": str(path)} for label, path in output_paths],
    }
    return payload


def toml_validation_rows(validation: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return GUI-ready rows for an in-memory TOML validation report."""

    status = str(validation.get("status", "unknown"))
    errors = validation.get("errors")
    if status == "pass":
        grid_shape = validation.get("grid_shape", [])
        grid_detail = (
            f"{grid_shape[0]} x {grid_shape[1]}"
            if isinstance(grid_shape, Sequence) and len(grid_shape) == 2
            else "n/a"
        )
        return [
            {
                "check": "TOML/config",
                "status": "pass",
                "detail": str(validation.get("message", "")),
            },
            {"check": "grid", "status": "pass", "detail": grid_detail},
            {
                "check": "regions",
                "status": "pass",
                "detail": str(validation.get("region_count", 0)),
            },
            {
                "check": "outputs",
                "status": "pass",
                "detail": _validation_output_detail(validation.get("output_paths", [])),
            },
        ]
    if isinstance(errors, Sequence) and not isinstance(errors, str) and errors:
        return [
            {
                "check": "TOML/config" if index == 0 else "validation",
                "status": status,
                "detail": str(error),
            }
            for index, error in enumerate(errors)
        ]
    return [
        {
            "check": "TOML/config",
            "status": status,
            "detail": str(validation.get("message", "")),
        }
    ]


def case_validation_run_rows(
    case_id: str,
    *,
    manifest: CaseManifest | None = None,
    root: str | Path | None = None,
    validation: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return GUI-ready validation and run command rows for a manifest case."""

    if manifest is None:
        case_manifest = default_case_manifest(_PROJECT_ROOT if root is None else root)
    else:
        case_manifest = manifest
    source = load_case_source_text(case_id, manifest=case_manifest, root=root)
    entry = case_manifest.by_id(case_id)
    path = entry.path or ""
    is_toml = path.lower().endswith(".toml")
    if not path:
        validate_status = "n/a"
        validate_detail = "case has no local TOML config"
        validate_command = ""
    elif not is_toml:
        validate_status = "n/a"
        validate_detail = f"source file is {Path(path).suffix or 'not TOML'}"
        validate_command = ""
    elif not source["exists"]:
        validate_status = "missing"
        validate_detail = str(source["message"])
        validate_command = f"tokamaker-jax validate {path}"
    else:
        validate_status = str(validation.get("status", "ready")) if validation else "ready"
        validate_detail = (
            str(validation.get("message", "local TOML file")) if validation else "local TOML file"
        )
        validate_command = f"tokamaker-jax validate {path}"

    run_command = entry.command or ""
    run_status = "planned"
    run_detail = entry.title
    if run_command:
        run_status = "blocked" if validate_status in {"fail", "missing"} else "ready"
        if run_status == "blocked":
            run_detail = "TOML validation must pass before running"

    rows = [
        {
            "action": "Validate TOML",
            "status": validate_status,
            "detail": validate_detail,
            "command": validate_command,
        },
        {
            "action": "Run case",
            "status": run_status,
            "detail": run_detail,
            "command": run_command or "planned",
        },
    ]
    if entry.validation_gate:
        rows.append(
            {
                "action": "Run validation gate",
                "status": "ready" if entry.runnable else "planned",
                "detail": entry.parity_level,
                "command": entry.validation_gate,
            }
        )
    return rows


def _case_preview_label(case_id: str, manifest: CaseManifest) -> str:
    entry = manifest.by_id(case_id)
    command = entry.command or entry.validation_gate or "planned"
    return f"{entry.title}: {entry.status}; {entry.parity_level}; {command}"


def _validation_output_detail(output_paths: Any) -> str:
    if not isinstance(output_paths, Sequence) or isinstance(output_paths, str):
        return "none configured"
    parts = []
    for output in output_paths:
        if isinstance(output, Mapping):
            label = str(output.get("label", "output"))
            path = str(output.get("path", ""))
            if path:
                parts.append(f"{label}={path}")
    return ", ".join(parts) if parts else "none configured"


def _case_editor_validation(source: Mapping[str, Any]) -> dict[str, Any]:
    path = str(source.get("path", ""))
    if not source.get("exists"):
        message = str(source.get("message", "case source is unavailable"))
        return {
            "source_name": path or "<case>",
            "status": "missing",
            "message": message,
            "errors": [message],
            "grid_shape": [],
            "region_count": 0,
            "output_paths": [],
        }
    if not path.lower().endswith(".toml"):
        return {
            "source_name": path or "<case>",
            "status": "n/a",
            "message": "source file is not a TOML config",
            "errors": [],
            "grid_shape": [],
            "region_count": 0,
            "output_paths": [],
        }
    return validate_toml_text(str(source.get("source", "")), source_name=path)


def load_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report for GUI display."""

    report_path = Path(path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_gui_report_artifacts(
    root: str | Path = _PROJECT_ROOT,
    artifacts: Mapping[str, str | Path] = _DEFAULT_REPORT_ARTIFACTS,
) -> dict[str, Any]:
    """Load known GUI report artifacts when they exist."""

    root_path = Path(root)
    reports: dict[str, Any] = {}
    for key, relative_path in artifacts.items():
        path = root_path / relative_path
        if path.exists():
            payload = load_json_report(path)
            payload["_artifact_path"] = str(path)
            reports[key] = payload
        else:
            reports[key] = {"status": "missing", "_artifact_path": str(path)}
    return reports


def validation_report_rows(reports: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return compact rows for stored validation and comparison reports."""

    rows: list[dict[str, str]] = []
    validation = reports.get("validation")
    if isinstance(validation, Mapping) and isinstance(validation.get("gates"), Mapping):
        for gate_id, payload in validation["gates"].items():
            rows.append(
                {
                    "gate": str(gate_id),
                    "status": _stored_gate_status(gate_id, payload),
                    "metric": _stored_gate_metric(gate_id, payload),
                    "source": str(validation.get("_artifact_path", "")),
                }
            )
    openfusiontoolkit = reports.get("openfusiontoolkit")
    if isinstance(openfusiontoolkit, Mapping):
        rows.append(
            {
                "gate": "openfusiontoolkit",
                "status": str(openfusiontoolkit.get("status", "missing")),
                "metric": _stored_gate_metric("openfusiontoolkit", openfusiontoolkit),
                "source": str(openfusiontoolkit.get("_artifact_path", "")),
            }
        )
    if not rows:
        rows.append(
            {
                "gate": "validation",
                "status": "missing",
                "metric": "no stored validation report found",
                "source": "",
            }
        )
    return rows


def benchmark_report_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Return compact rows for a stored benchmark report."""

    if not isinstance(report, Mapping) or not isinstance(report.get("benchmarks"), list):
        return [
            {
                "lane": "benchmark",
                "median_ms": "",
                "best_ms": "",
                "worst_ms": "",
                "metadata": "no stored benchmark report found",
            }
        ]
    rows = []
    for entry in report["benchmarks"]:
        result = entry["result"]
        rows.append(
            {
                "lane": str(entry["lane"]),
                "median_ms": _format_number(1000.0 * float(result["median_s"])),
                "best_ms": _format_number(1000.0 * float(result["best_s"])),
                "worst_ms": _format_number(1000.0 * float(result["worst_s"])),
                "metadata": json.dumps(result.get("metadata", {}), sort_keys=True),
            }
        )
    return rows


def upstream_fixture_report_rows(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Return compact rows for stored upstream fixture summaries."""

    if not isinstance(report, Mapping) or not isinstance(report.get("entries"), list):
        return [
            {
                "fixture_id": "upstream-fixtures",
                "available": "no",
                "category": "",
                "mesh": "no stored fixture summary found",
                "geometry": "",
                "claim": "missing",
            }
        ]
    return upstream_fixture_rows(dict(report))


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


def _seed_equilibrium_dashboard(
    pressure_scale: float,
    ffp_scale: float,
    iterations: int,
) -> dict[str, Any]:
    config = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=pressure_scale, ffp_scale=ffp_scale),
        solver=SolverConfig(iterations=iterations, relaxation=0.75, dtype="float64"),
    )
    solution = solve_from_config(config)
    summary = equilibrium_metadata_summary(solution)
    residual = summary["residual"]
    initial = residual["initial"]
    final = residual["final"]
    residual_drop_fraction = (
        None
        if initial in (None, 0.0) or final is None
        else float((float(initial) - float(final)) / abs(float(initial)))
    )
    residual_finite = bool(residual["range"]["all_finite"])
    status = "pass" if residual_finite and residual_drop_fraction is not None else "fail"
    if status == "pass" and residual_drop_fraction < 0.0:
        status = "warn"
    psi_range = summary["psi"]["range"]
    return {
        "status": status,
        "command": "tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png",
        "inputs": {
            "pressure_scale": float(pressure_scale),
            "ffp_scale": float(ffp_scale),
            "iterations": int(iterations),
            "relaxation": 0.75,
            "dtype": "float64",
        },
        "metrics": {
            "iterations": int(summary["iterations"]),
            "grid": summary["grid"],
            "psi_min": psi_range["min"],
            "psi_max": psi_range["max"],
            "residual_initial": initial,
            "residual_final": final,
            "residual_drop_fraction": residual_drop_fraction,
        },
        "summary": summary,
    }


def _region_geometry_dashboard(regions: tuple[Region, ...]) -> dict[str, Any]:
    rows = region_table_data(regions)
    by_kind: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind_summary = by_kind.setdefault(row["kind"], {"count": 0, "area": 0.0})
        kind_summary["count"] += 1
        kind_summary["area"] += row["area"]

    kinds = sorted(by_kind)
    missing_required = [kind for kind in ("plasma", "coil") if kind not in by_kind]
    status = "pass" if not missing_required else "warn"
    total_area = float(sum(row["area"] for row in rows))
    return {
        "status": status,
        "command": "tokamaker-jax validate examples/fixed_boundary.toml",
        "metrics": {
            "n_regions": len(rows),
            "kinds": kinds,
            "total_area": total_area,
            "n_with_target_size": sum(row["target_size"] is not None for row in rows),
            "missing_required_kinds": missing_required,
        },
        "by_kind": {
            kind: {"count": int(value["count"]), "area": float(value["area"])}
            for kind, value in by_kind.items()
        },
        "regions": rows,
    }


def _validation_dashboard(subdivisions: tuple[int, ...]) -> dict[str, Any]:
    poisson = run_poisson_convergence_study(subdivisions)
    grad_shafranov = run_grad_shafranov_convergence_study(subdivisions)
    coil_green = run_coil_green_function_validation()
    gates = [
        _convergence_gate_summary(
            "poisson",
            "Poisson FEM",
            poisson,
            h1_rate_key="h1_rates",
            h1_metric_key="h1_error",
            h1_label="H1",
            command=_validation_command("poisson", subdivisions),
        ),
        _convergence_gate_summary(
            "grad_shafranov",
            "Grad-Shafranov",
            grad_shafranov,
            h1_rate_key="weighted_h1_rates",
            h1_metric_key="weighted_h1_error",
            h1_label="weighted H1",
            command=_validation_command("grad-shafranov", subdivisions),
        ),
        _coil_green_gate_summary(coil_green),
    ]
    return {
        "status": _rollup_status(gate["status"] for gate in gates),
        "command": _validation_command("all", subdivisions),
        "subdivisions": list(subdivisions),
        "gates": gates,
    }


def _coil_response_dashboard() -> dict[str, Any]:
    grid, coils = _default_coil_response_inputs()
    flux = np.asarray(coil_flux_on_grid(grid, coils))
    finite = bool(np.isfinite(flux).all())
    abs_flux = np.abs(flux)
    return {
        "status": "pass" if coils and finite else "fail",
        "command": "tokamaker-jax verify --gate coil-green",
        "grid": {
            "r_min": float(grid.r_min),
            "r_max": float(grid.r_max),
            "z_min": float(grid.z_min),
            "z_max": float(grid.z_max),
            "nr": int(grid.nr),
            "nz": int(grid.nz),
        },
        "coils": [
            {
                "name": coil.name,
                "r": float(coil.r),
                "z": float(coil.z),
                "current": float(coil.current),
                "sigma": float(coil.sigma),
            }
            for coil in coils
        ],
        "metrics": {
            "n_coils": len(coils),
            "current_total": float(sum(coil.current for coil in coils)),
            "current_abs_total": float(sum(abs(coil.current) for coil in coils)),
            "flux_min": _finite_array_scalar(flux, np.min),
            "flux_max": _finite_array_scalar(flux, np.max),
            "flux_abs_max": _finite_array_scalar(abs_flux, np.max),
            "flux_all_finite": finite,
        },
    }


def _convergence_gate_summary(
    gate_id: str,
    label: str,
    study: PoissonConvergenceStudy | GradShafranovConvergenceStudy,
    *,
    h1_rate_key: str,
    h1_metric_key: str,
    h1_label: str,
    command: str,
) -> dict[str, Any]:
    payload = study.to_dict()
    results = payload["results"]
    l2_rates = [float(rate) for rate in payload["l2_rates"]]
    h1_rates = [float(rate) for rate in payload[h1_rate_key]]
    min_l2_rate = min(l2_rates)
    min_h1_rate = min(h1_rates)
    rates_finite = bool(np.isfinite(l2_rates + h1_rates).all())
    status = (
        "pass"
        if rates_finite
        and min_l2_rate >= _CONVERGENCE_RATE_THRESHOLDS["l2"]
        and min_h1_rate >= _CONVERGENCE_RATE_THRESHOLDS["h1"]
        else "warn"
    )
    if not rates_finite:
        status = "fail"
    final = results[-1]
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "command": command,
        "summary": f"min rates L2={_format_number(min_l2_rate)}, {h1_label}={_format_number(min_h1_rate)}",
        "metrics": {
            "levels": len(results),
            "final_l2_error": float(final["l2_error"]),
            f"final_{h1_metric_key}": float(final[h1_metric_key]),
            "min_l2_rate": min_l2_rate,
            "min_h1_rate": min_h1_rate,
        },
        "thresholds": dict(_CONVERGENCE_RATE_THRESHOLDS),
        "rates": {"l2": l2_rates, h1_label: h1_rates},
        "results": results,
    }


def _coil_green_gate_summary(validation) -> dict[str, Any]:
    metrics = validation.to_dict()
    error_keys = [key for key in metrics if key.endswith("_error")]
    max_error = max(float(metrics[key]) for key in error_keys)
    errors_finite = bool(np.isfinite([metrics[key] for key in error_keys]).all())
    status = "pass" if errors_finite and max_error <= _COIL_GREEN_ERROR_TOLERANCE else "warn"
    if not errors_finite:
        status = "fail"
    return {
        "id": "coil_green",
        "label": "Coil Green",
        "status": status,
        "command": "tokamaker-jax verify --gate coil-green",
        "summary": f"max error {_format_number(max_error)}",
        "metrics": {**metrics, "max_error": max_error},
        "thresholds": {"max_error": _COIL_GREEN_ERROR_TOLERANCE},
        "results": [],
    }


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


def _validation_subdivisions_tuple(validation_subdivisions: Sequence[int]) -> tuple[int, ...]:
    subdivisions = tuple(int(level) for level in validation_subdivisions)
    if len(subdivisions) < 2 or any(level < 2 for level in subdivisions):
        raise ValueError("validation_subdivisions must contain at least two levels >= 2")
    return subdivisions


def _validation_command(gate: str, subdivisions: tuple[int, ...]) -> str:
    levels = " ".join(str(level) for level in subdivisions)
    return f"tokamaker-jax verify --gate {gate} --subdivisions {levels}"


def _section_summary(
    section_id: str,
    label: str,
    status: str,
    metric: str,
    command: str,
) -> dict[str, str]:
    return {
        "id": section_id,
        "label": label,
        "status": status,
        "metric": metric,
        "command": command,
    }


def _rollup_status(statuses: Sequence[str] | Any) -> str:
    status_tuple = tuple(statuses)
    for status in ("fail", "warn", "open"):
        if status in status_tuple:
            return status
    return "pass"


def _workflow_next_steps(dashboard: dict[str, Any]) -> list[dict[str, str]]:
    steps = [
        {
            "id": "validate_config",
            "label": "Validate TOML inputs",
            "status": "open",
            "command": "tokamaker-jax validate examples/fixed_boundary.toml",
        },
        {
            "id": "run_seed_solve",
            "label": "Run configured seed solve",
            "status": "open",
            "command": dashboard["seed_equilibrium"]["command"],
        },
    ]
    missing_regions = dashboard["region_geometry"]["metrics"]["missing_required_kinds"]
    if missing_regions:
        steps.append(
            {
                "id": "complete_region_set",
                "label": f"Add required regions: {', '.join(missing_regions)}",
                "status": "open",
                "command": dashboard["region_geometry"]["command"],
            }
        )
    if dashboard["validation"]["status"] == "pass":
        steps.append(
            {
                "id": "export_validation_report",
                "label": "Export validation JSON",
                "status": "open",
                "command": f"{dashboard['validation']['command']} --output outputs/verify.json",
            }
        )
    else:
        steps.append(
            {
                "id": "triage_validation",
                "label": "Triage validation gates",
                "status": "open",
                "command": dashboard["validation"]["command"],
            }
        )
    steps.append(
        {
            "id": "coil_response_compare",
            "label": "Compare coil response",
            "status": "open",
            "command": dashboard["coil_response"]["command"],
        }
    )
    return steps


def _stored_gate_status(gate_id: str, payload: Any) -> str:
    if isinstance(payload, Mapping) and isinstance(payload.get("status"), str):
        return str(payload["status"])
    metric = _stored_gate_metric(gate_id, payload)
    if "n/a" in metric or "missing" in metric:
        return "missing"
    return "recorded"


def _stored_gate_metric(gate_id: str, payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return "n/a"
    if gate_id in {"poisson", "grad_shafranov"}:
        l2_rates = payload.get("l2_rates", [])
        h1_rates = payload.get("h1_rates", payload.get("weighted_h1_rates", []))
        if l2_rates and h1_rates:
            return (
                f"min L2 rate {_format_number(min(l2_rates))}; "
                f"min H1 rate {_format_number(min(h1_rates))}"
            )
    if gate_id in {"coil_green", "circular_loop"}:
        error_values = [
            float(value)
            for key, value in payload.items()
            if key.endswith("_error") and isinstance(value, int | float)
        ]
        if error_values:
            return f"max error {_format_number(max(error_values))}"
    if gate_id == "profile_iteration":
        return (
            f"residual {_format_number(payload.get('residual_initial'))} -> "
            f"{_format_number(payload.get('residual_final'))}"
        )
    if gate_id == "free_boundary_profile":
        return (
            f"boundary {_format_number(payload.get('boundary_error'))}; "
            f"residual {_format_number(payload.get('residual_final'))}"
        )
    if gate_id == "openfusiontoolkit":
        status = payload.get("status", "missing")
        relative_error = payload.get("relative_error")
        if relative_error is None:
            reason = payload.get("reason") or payload.get("probe", {}).get("reason")
            return f"{status}: {reason or 'no numeric comparison'}"
        return f"relative error {_format_number(relative_error)}"
    if payload.get("status") == "missing":
        return "missing artifact"
    return "stored report"


def _default_coil_response_inputs() -> tuple[RectangularGrid, tuple[CoilConfig, ...]]:
    return (
        RectangularGrid(1.0, 2.8, -0.9, 0.9, 75, 75),
        (
            CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
            CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
            CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
        ),
    )


def _finite_array_scalar(values: np.ndarray, reducer) -> float | None:
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(reducer(values))


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
