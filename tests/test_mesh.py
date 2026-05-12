from pathlib import Path

import h5py
import numpy as np
import pytest

from tokamaker_jax.mesh import TriMesh, load_gs_mesh, mesh_from_arrays, save_gs_mesh
from tokamaker_jax.plotting import save_mesh_plot


def tiny_mesh() -> TriMesh:
    nodes = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.5, 0.5],
        ]
    )
    triangles = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
    )
    regions = np.asarray([1, 2, 3, 4])
    return mesh_from_arrays(
        nodes,
        triangles,
        regions,
        coil_dict={"PF": {"reg_id": 3, "coil_id": 0, "nturns": 1, "coil_set": "PF"}},
        cond_dict={
            "AIR": {"reg_id": 2, "vac_id": 0},
            "VV": {"reg_id": 4, "cond_id": 0, "eta": 6.9e-7},
        },
    )


def test_mesh_summary_and_regions():
    mesh = tiny_mesh()

    assert mesh.n_nodes == 5
    assert mesh.n_cells == 4
    assert mesh.bounds == (1.0, 2.0, 0.0, 1.0)
    assert mesh.region_cell_counts() == {1: 1, 2: 1, 3: 1, 4: 1}
    assert set(mesh.region_areas()) == {1, 2, 3, 4}
    assert np.allclose(mesh.cell_areas(), 0.25)
    np.testing.assert_allclose(mesh.cell_centers()[0], [1.5, 1.0 / 6.0])
    np.testing.assert_array_equal(mesh.region_mask(3), [False, False, True, False])
    assert mesh.boundary_edges().shape == (4, 2)
    assert mesh.conductor_names() == ("VV",)
    assert mesh.vacuum_names() == ("AIR",)

    summary = mesh.summary()
    assert summary["n_boundary_edges"] == 4
    assert summary["n_coils"] == 1
    assert summary["n_conductors"] == 1
    assert summary["area_total"] == 1.0


def test_mesh_round_trips_hdf5_and_json(tmp_path: Path):
    mesh = tiny_mesh()

    h5_path = save_gs_mesh(mesh, tmp_path / "mesh.h5")
    json_path = save_gs_mesh(mesh, tmp_path / "mesh.json")
    h5_mesh = load_gs_mesh(h5_path)
    json_mesh = load_gs_mesh(json_path)

    for loaded in (h5_mesh, json_mesh):
        np.testing.assert_allclose(loaded.nodes, mesh.nodes)
        np.testing.assert_array_equal(loaded.triangles, mesh.triangles)
        np.testing.assert_array_equal(loaded.regions, mesh.regions)
        assert loaded.coil_dict == mesh.coil_dict
        assert loaded.cond_dict == mesh.cond_dict


def test_mesh_plot_writes_file(tmp_path: Path):
    path = save_mesh_plot(tiny_mesh(), tmp_path / "mesh.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_mesh_rejects_invalid_geometry_and_metadata():
    mesh = tiny_mesh()

    with pytest.raises(ValueError, match="nodes"):
        TriMesh(np.asarray([1.0, 2.0]), mesh.triangles, mesh.regions)

    with pytest.raises(ValueError, match="triangles"):
        TriMesh(mesh.nodes, np.asarray([0, 1, 2]), np.asarray([1]))

    with pytest.raises(ValueError, match="regions"):
        TriMesh(mesh.nodes, mesh.triangles, np.asarray([1, 2]))

    with pytest.raises(ValueError, match="at least 3 nodes"):
        TriMesh(mesh.nodes[:2], np.asarray([[0, 1, 1]]), np.asarray([1]))

    with pytest.raises(ValueError, match="at least 1 cell"):
        TriMesh(mesh.nodes, np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32))

    bad_nodes = mesh.nodes.copy()
    bad_nodes[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TriMesh(bad_nodes, mesh.triangles, mesh.regions)

    with pytest.raises(ValueError, match="positive"):
        TriMesh(mesh.nodes, mesh.triangles[:, [0, 2, 1]], mesh.regions)

    with pytest.raises(ValueError, match="outside"):
        TriMesh(mesh.nodes, np.asarray([[0, 1, 99]]), np.asarray([1]))

    with pytest.raises(ValueError, match="one-based"):
        TriMesh(mesh.nodes, np.asarray([[0, 1, 4]]), np.asarray([0]))

    with pytest.raises(ValueError, match="unknown region"):
        mesh_from_arrays(
            mesh.nodes,
            mesh.triangles,
            mesh.regions,
            coil_dict={"bad": {"reg_id": 99}},
        )

    with pytest.raises(ValueError, match="missing reg_id"):
        mesh_from_arrays(mesh.nodes, mesh.triangles, mesh.regions, cond_dict={"bad": {}})


def test_load_rejects_invalid_metadata_json(tmp_path: Path):
    path = tmp_path / "bad_metadata.h5"
    mesh = tiny_mesh()
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("mesh/r", data=mesh.nodes, dtype="f8")
        h5_file.create_dataset("mesh/lc", data=mesh.triangles, dtype="i4")
        h5_file.create_dataset("mesh/reg", data=mesh.regions, dtype="i4")
        string_datatype = h5py.string_dtype("ascii")
        h5_file.create_dataset("mesh/coil_dict", data="[]", dtype=string_datatype)
        h5_file.create_dataset("mesh/cond_dict", data="{}", dtype=string_datatype)

    with pytest.raises(ValueError, match="decode to a dictionary"):
        load_gs_mesh(path)


def test_loads_upstream_iter_mesh_when_available():
    upstream = Path(
        "/Users/rogeriojorge/local/OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_mesh.h5"
    )
    if not upstream.exists():
        pytest.skip("OpenFUSIONToolkit checkout is not available")

    mesh = load_gs_mesh(upstream)

    assert mesh.n_nodes == 4757
    assert mesh.n_cells == 9400
    assert int(mesh.region_ids[-1]) == 20
    assert "PF1" in mesh.coil_dict
    assert "VV1" in mesh.cond_dict
    assert mesh.summary()["n_boundary_edges"] > 0
