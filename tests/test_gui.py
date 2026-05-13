import json

import pytest

import tokamaker_jax.geometry as geometry
from tokamaker_jax.geometry import RegionSet, annulus_region, rectangle_region
from tokamaker_jax.gui import (
    benchmark_report_rows,
    case_manifest_rows,
    coil_green_response_figure,
    load_gui_report_artifacts,
    load_json_report,
    region_geometry_figure,
    region_table_rows,
    seed_equilibrium_figure,
    seed_equilibrium_summary_rows,
    validation_convergence_figure,
    validation_gate_rows,
    validation_report_rows,
    workflow_dashboard_data,
    workflow_next_step_rows,
    workflow_status_rows,
)

pytest.importorskip("plotly")


def test_region_geometry_figure_uses_sample_regions():
    fig = region_geometry_figure()

    trace_names = {trace.name for trace in fig.data}
    assert "PLASMA (plasma)" in trace_names
    assert "VV (conductor)" in trace_names
    assert "PF (coil)" in trace_names
    assert fig.layout.xaxis.title.text == "R [m]"
    assert fig.layout.yaxis.title.text == "Z [m]"
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.meta["regions"][0]["name"] == "VV"


def test_region_geometry_figure_prefers_geometry_sample_regions(monkeypatch):
    def sample_regions():
        return RegionSet(
            (
                rectangle_region(
                    id=8,
                    name="CUSTOM",
                    kind="boundary",
                    r_min=1.0,
                    r_max=2.0,
                    z_min=-0.5,
                    z_max=0.5,
                ),
            )
        )

    monkeypatch.setattr(geometry, "sample_regions", sample_regions, raising=False)

    fig = region_geometry_figure()

    assert [trace.name for trace in fig.data] == ["CUSTOM (boundary)"]


def test_region_geometry_figure_accepts_regions_and_holes():
    regions = RegionSet(
        (
            annulus_region(
                id=1,
                name="WALL",
                kind="conductor",
                center_r=2.0,
                center_z=0.0,
                inner_radius=0.5,
                outer_radius=1.0,
                n=24,
            ),
            rectangle_region(
                id=2,
                name="PLASMA",
                kind="plasma",
                r_min=1.5,
                r_max=2.5,
                z_min=-0.4,
                z_max=0.4,
            ),
        )
    )

    fig = region_geometry_figure(regions, show_labels=False)

    assert [trace.name for trace in fig.data] == [
        "WALL (conductor)",
        "WALL hole 1",
        "PLASMA (plasma)",
    ]
    assert not fig.layout.annotations
    assert fig.data[0].fill == "toself"
    assert fig.data[1].showlegend is False

    rows = region_table_rows(regions)
    assert rows[0]["name"] == "WALL"
    assert rows[0]["n_holes"] == 1
    assert rows[0]["centroid"].startswith("(")


def test_region_geometry_figure_rejects_empty_regions():
    with pytest.raises(ValueError, match="at least one"):
        region_geometry_figure(())


def test_seed_equilibrium_figure_attaches_summary_metadata():
    fig = seed_equilibrium_figure(pressure_scale=1.0e3, ffp_scale=-0.1, iterations=2)

    summary = fig.layout.meta["summary"]
    rows = seed_equilibrium_summary_rows(summary)

    assert summary["iterations"] == 2
    assert summary["grid"]["nr"] == 65
    assert any("residual" in annotation.text for annotation in fig.layout.annotations)
    assert rows[0] == {"metric": "grid", "value": "65 x 65"}
    assert rows[-1]["metric"] == "final residual"


def test_validation_convergence_figure_attaches_rates_metadata():
    fig = validation_convergence_figure("grad-shafranov")

    assert "Grad-Shafranov" in fig.layout.title.text
    assert len(fig.data) == 4
    assert len(fig.layout.meta["results"]) == 3
    assert min(fig.layout.meta["h1_rates"]) > 0.85

    with pytest.raises(ValueError, match="poisson"):
        validation_convergence_figure("bad")


def test_coil_green_response_figure_attaches_metadata():
    fig = coil_green_response_figure()

    assert "coil Green" in fig.layout.title.text
    assert fig.layout.meta["n_coils"] == 3
    assert len(fig.data) == 2
    assert fig.data[1].name == "PF coils"


def test_workflow_dashboard_data_schema_and_key_values():
    dashboard = workflow_dashboard_data(
        pressure_scale=1.0e3,
        ffp_scale=-0.1,
        iterations=2,
        validation_subdivisions=(4, 8),
    )

    json.dumps(dashboard)
    assert dashboard["schema_version"] == 1
    assert dashboard["workflow"]["status"] == "pass"
    assert [section["id"] for section in dashboard["sections"]] == [
        "seed_equilibrium",
        "region_geometry",
        "validation",
        "coil_response",
    ]

    seed = dashboard["seed_equilibrium"]
    assert seed["status"] == "pass"
    assert seed["metrics"]["iterations"] == 2
    assert seed["metrics"]["grid"]["nr"] == 65
    assert seed["metrics"]["residual_drop_fraction"] > 0.0

    geometry_summary = dashboard["region_geometry"]
    assert geometry_summary["status"] == "pass"
    assert geometry_summary["metrics"]["n_regions"] == 3
    assert geometry_summary["by_kind"]["coil"]["count"] == 1

    gates = {gate["id"]: gate for gate in dashboard["validation"]["gates"]}
    assert gates["poisson"]["status"] == "pass"
    assert gates["grad_shafranov"]["command"] == (
        "tokamaker-jax verify --gate grad-shafranov --subdivisions 4 8"
    )
    assert gates["coil_green"]["metrics"]["n_coils"] == 2
    assert gates["coil_green"]["metrics"]["max_error"] < 1.0e-10
    assert dashboard["coil_response"]["metrics"]["n_coils"] == 3

    status_rows = workflow_status_rows(dashboard)
    gate_rows = validation_gate_rows(dashboard)
    next_step_rows = workflow_next_step_rows(dashboard)

    assert status_rows[0]["section"] == "Seed equilibrium"
    assert gate_rows[-1]["gate"] == "Coil Green"
    assert next_step_rows[0] == {
        "step": "Validate TOML inputs",
        "status": "open",
        "command": "tokamaker-jax validate examples/fixed_boundary.toml",
    }


def test_workflow_dashboard_data_validates_subdivisions():
    with pytest.raises(ValueError, match="validation_subdivisions"):
        workflow_dashboard_data(iterations=1, validation_subdivisions=(4,))


def test_gui_report_helpers_load_validation_and_benchmark_rows(tmp_path):
    validation_path = tmp_path / "verify.json"
    validation_path.write_text(
        json.dumps(
            {
                "gates": {
                    "poisson": {"l2_rates": [1.9], "h1_rates": [0.95]},
                    "free_boundary_profile": {
                        "boundary_error": 0.0,
                        "residual_final": 0.02,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "lane": "seed",
                        "result": {
                            "median_s": 0.002,
                            "best_s": 0.001,
                            "worst_s": 0.003,
                            "metadata": {"nr": 9},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_json_report(validation_path)["gates"]["poisson"]["l2_rates"] == [1.9]
    reports = load_gui_report_artifacts(
        root=tmp_path,
        artifacts={
            "validation": "verify.json",
            "benchmark": "benchmark.json",
            "openfusiontoolkit": "missing.json",
        },
    )
    validation_rows = validation_report_rows(reports)
    benchmark_rows = benchmark_report_rows(reports["benchmark"])

    assert validation_rows[0]["gate"] == "poisson"
    assert validation_rows[0]["status"] == "recorded"
    assert "min L2 rate" in validation_rows[0]["metric"]
    assert validation_rows[1]["gate"] == "free_boundary_profile"
    assert validation_rows[-1]["status"] == "missing"
    assert benchmark_rows == [
        {
            "lane": "seed",
            "median_ms": "2",
            "best_ms": "1",
            "worst_ms": "3",
            "metadata": '{"nr": 9}',
        }
    ]


def test_case_manifest_rows_expose_runnable_and_planned_cases():
    rows = case_manifest_rows()
    by_id = {row["case_id"]: row for row in rows}

    assert by_id["fixed-boundary-seed"]["status"] == "runnable"
    assert "tokamaker-jax examples/fixed_boundary.toml" in by_id["fixed-boundary-seed"]["command"]
    assert by_id["iter-baseline-upstream"]["command"] == "planned"
    assert by_id["openfusiontoolkit-green-parity"]["parity_level"] == "kernel_parity"


def test_benchmark_report_rows_reports_missing_artifact():
    assert benchmark_report_rows(None) == [
        {
            "lane": "benchmark",
            "median_ms": "",
            "best_ms": "",
            "worst_ms": "",
            "metadata": "no stored benchmark report found",
        }
    ]
