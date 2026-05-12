import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.geometry import RegionSet, annulus_region, rectangle_region
from tokamaker_jax.mesh import mesh_from_arrays
from tokamaker_jax.plotting import (
    equilibrium_figure_data,
    equilibrium_metadata_summary,
    mesh_figure_data,
    plot_equilibrium,
    plot_mesh,
    region_figure_data,
    region_table_data,
)
from tokamaker_jax.profiles import solovev_source
from tokamaker_jax.solver import solve_fixed_boundary


def test_region_figure_data_is_json_ready_with_axes_and_ranges():
    regions = RegionSet(
        (
            rectangle_region(
                id=1,
                name="plasma",
                r_min=1.0,
                r_max=2.0,
                z_min=-0.5,
                z_max=0.5,
                kind="plasma",
            ),
            annulus_region(
                id=2,
                name="wall",
                center_r=1.5,
                center_z=0.0,
                inner_radius=0.7,
                outer_radius=0.9,
                n=16,
                kind="conductor",
            ),
        )
    )

    payload = region_figure_data(
        regions,
        source="synthetic test geometry",
        citation="test citation",
        command="tokamaker-jax plot-regions case.toml",
    ).to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["axes"]["x"]["label"] == "R"
    assert payload["axes"]["x"]["units"] == "m"
    assert payload["axes"]["y"]["label"] == "Z"
    assert payload["metadata"]["bounds"]["R"]["all_finite"] is True
    assert payload["metadata"]["bounds"]["Z"]["all_finite"] is True
    assert payload["metadata"]["bounds"]["R"]["min"] < payload["metadata"]["bounds"]["R"]["max"]
    assert payload["metadata"]["bounds"]["Z"]["min"] < payload["metadata"]["bounds"]["Z"]["max"]
    assert payload["data"]["regions"][0]["points"]["columns"] == ["R", "Z"]
    assert payload["data"]["regions"][0]["points"]["units"] == ["m", "m"]
    assert payload["metadata"]["table"][0]["name"] == "plasma"
    assert payload["metadata"]["table"][0]["n_holes"] == 0
    assert (
        json.loads(region_figure_data(regions).to_json()) == region_figure_data(regions).to_dict()
    )


def test_region_table_data_exports_flat_rows():
    regions = RegionSet(
        (
            rectangle_region(
                id=4,
                name="limiter",
                kind="limiter",
                r_min=1.0,
                r_max=1.2,
                z_min=-0.1,
                z_max=0.1,
                target_size=0.02,
                metadata={"role": "test"},
            ),
        )
    )

    rows = region_table_data(regions)

    assert json.loads(json.dumps(rows)) == rows
    assert rows[0]["id"] == 4
    assert rows[0]["name"] == "limiter"
    assert rows[0]["kind"] == "limiter"
    assert np.isclose(rows[0]["area"], 0.04)
    assert np.isclose(rows[0]["centroid_r"], 1.1)
    assert np.isclose(rows[0]["centroid_z"], 0.0)
    assert rows[0]["n_points"] == 4
    assert rows[0]["n_holes"] == 0
    assert rows[0]["target_size"] == 0.02
    assert rows[0]["metadata"] == {"role": "test"}

    with pytest.raises(ValueError, match="at least one"):
        region_table_data(())


def test_mesh_figure_data_exports_structured_numeric_payloads():
    mesh = mesh_from_arrays(
        nodes=np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]]),
        regions=np.asarray([1, 2]),
    )

    payload = mesh_figure_data(mesh, command="tokamaker-jax mesh-preview mesh.h5").to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["axes"]["x"]["units"] == "m"
    assert payload["data"]["nodes"]["columns"] == ["R", "Z"]
    assert payload["data"]["nodes"]["range"] == {"min": 0.0, "max": 2.0, "all_finite": True}
    assert payload["data"]["triangles"]["convention"] == "Zero-based triangle node indices."
    assert payload["data"]["cell_regions"]["range"] == {
        "min": 1.0,
        "max": 2.0,
        "all_finite": True,
    }
    assert payload["metadata"]["summary"]["n_cells"] == 2


def test_equilibrium_figure_data_exports_axes_fields_and_solution_ranges():
    grid = RectangularGrid(0.8, 1.2, -0.2, 0.2, 9, 9)
    source = solovev_source(grid, dtype=jnp.float32)
    solution = solve_fixed_boundary(grid, source, iterations=4, relaxation=0.7, dtype=jnp.float32)

    payload = equilibrium_figure_data(solution, include_source=True).to_dict()
    summary = equilibrium_metadata_summary(solution)

    assert json.loads(json.dumps(payload)) == payload
    assert (
        json.loads(equilibrium_figure_data(solution).to_json())
        == equilibrium_figure_data(solution).to_dict()
    )
    assert payload["axes"]["x"] == {
        "label": "R",
        "units": "m",
        "convention": "Major radius in axisymmetric cylindrical coordinates.",
    }
    assert payload["data"]["R"]["range"]["all_finite"] is True
    assert payload["data"]["Z"]["range"]["all_finite"] is True
    assert np.isclose(payload["data"]["R"]["range"]["min"], 0.8)
    assert np.isclose(payload["data"]["R"]["range"]["max"], 1.2)
    assert np.isclose(payload["data"]["Z"]["range"]["min"], -0.2)
    assert np.isclose(payload["data"]["Z"]["range"]["max"], 0.2)
    assert payload["data"]["psi"]["range"]["all_finite"] is True
    assert payload["data"]["source"]["range"]["all_finite"] is True
    assert payload["data"]["psi"]["shape"] == [9, 9]
    assert payload["metadata"]["grid"]["dr"] > 0.0
    assert payload["metadata"]["summary"] == summary
    assert summary["iterations"] == 4
    assert summary["residual"]["final"] is not None

    payload_without_source = equilibrium_figure_data(solution, include_source=False).to_dict()
    assert "source" not in payload_without_source["data"]


def test_plot_equilibrium_can_annotate_metadata():
    grid = RectangularGrid(0.8, 1.2, -0.2, 0.2, 9, 9)
    source = solovev_source(grid, dtype=jnp.float32)
    solution = solve_fixed_boundary(grid, source, iterations=3, relaxation=0.7, dtype=jnp.float32)

    fig, ax = plot_equilibrium(solution, levels=6)

    assert any("residual" in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_plot_equilibrium_and_mesh_can_disable_optional_layers():
    grid = RectangularGrid(0.8, 1.2, -0.2, 0.2, 9, 9)
    source = solovev_source(grid, dtype=jnp.float32)
    solution = solve_fixed_boundary(grid, source, iterations=3, relaxation=0.7, dtype=jnp.float32)
    fig, ax = plot_equilibrium(solution, levels=6, show_metadata=False)
    assert not any("residual" in text.get_text() for text in ax.texts)
    plt.close(fig)

    mesh = mesh_from_arrays(
        nodes=np.asarray([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]]),
        regions=np.asarray([1, 2]),
    )
    fig, ax = plot_mesh(mesh, show_regions=False, show_edges=False)
    assert not ax.collections
    assert not ax.lines
    plt.close(fig)
