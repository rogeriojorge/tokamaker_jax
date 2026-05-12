import json

import jax.numpy as jnp
import numpy as np

from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.geometry import RegionSet, annulus_region, rectangle_region
from tokamaker_jax.mesh import mesh_from_arrays
from tokamaker_jax.plotting import equilibrium_figure_data, mesh_figure_data, region_figure_data
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

    assert json.loads(json.dumps(payload)) == payload
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
