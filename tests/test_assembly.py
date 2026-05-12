import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.assembly import (
    assemble_global_matrix,
    assemble_laplace_stiffness_matrix,
    assemble_mass_matrix,
)
from tokamaker_jax.mesh import mesh_from_arrays


def square_mesh_arrays():
    nodes = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=jnp.float64,
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    return nodes, triangles


def test_square_mesh_mass_and_stiffness_properties():
    nodes, triangles = square_mesh_arrays()

    mass = assemble_mass_matrix(nodes, triangles)
    stiffness = assemble_laplace_stiffness_matrix(nodes, triangles)

    assert mass.shape == (4, 4)
    assert stiffness.shape == (4, 4)
    assert mass.dtype == jnp.float64
    assert stiffness.dtype == jnp.float64

    np.testing.assert_allclose(mass, mass.T, atol=1.0e-14)
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1.0e-14)
    np.testing.assert_allclose(jnp.sum(mass), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(jnp.sum(stiffness, axis=1), np.zeros(4), atol=1.0e-14)
    np.testing.assert_allclose(
        stiffness @ jnp.ones(4, dtype=nodes.dtype), np.zeros(4), atol=1.0e-14
    )

    assert float(jnp.min(jnp.linalg.eigvalsh(mass))) > 0.0
    assert float(jnp.min(jnp.linalg.eigvalsh(stiffness))) > -1.0e-12


def test_square_mesh_matches_exact_global_matrices():
    nodes, triangles = square_mesh_arrays()

    mass = assemble_mass_matrix(nodes, triangles)
    stiffness = assemble_laplace_stiffness_matrix(nodes, triangles)

    expected_mass = jnp.asarray(
        [
            [1.0 / 6.0, 1.0 / 24.0, 1.0 / 12.0, 1.0 / 24.0],
            [1.0 / 24.0, 1.0 / 12.0, 1.0 / 24.0, 0.0],
            [1.0 / 12.0, 1.0 / 24.0, 1.0 / 6.0, 1.0 / 24.0],
            [1.0 / 24.0, 0.0, 1.0 / 24.0, 1.0 / 12.0],
        ],
        dtype=jnp.float64,
    )
    expected_stiffness = jnp.asarray(
        [
            [1.0, -0.5, 0.0, -0.5],
            [-0.5, 1.0, -0.5, 0.0],
            [0.0, -0.5, 1.0, -0.5],
            [-0.5, 0.0, -0.5, 1.0],
        ],
        dtype=jnp.float64,
    )

    np.testing.assert_allclose(mass, expected_mass, atol=1.0e-14)
    np.testing.assert_allclose(stiffness, expected_stiffness, atol=1.0e-14)


def test_trimesh_and_array_inputs_use_same_assembly():
    nodes, triangles = square_mesh_arrays()
    mesh = mesh_from_arrays(np.asarray(nodes), np.asarray(triangles), np.asarray([1, 1]))

    np.testing.assert_allclose(assemble_mass_matrix(mesh), assemble_mass_matrix(nodes, triangles))
    np.testing.assert_allclose(
        assemble_laplace_stiffness_matrix(mesh),
        assemble_laplace_stiffness_matrix(nodes, triangles),
    )


def test_assemble_global_matrix_scatter_adds_shared_nodes():
    _, triangles = square_mesh_arrays()
    element_matrices = jnp.ones((2, 3, 3), dtype=jnp.float64)

    matrix = assemble_global_matrix(element_matrices, triangles, n_nodes=4)

    assert matrix.shape == (4, 4)
    assert matrix[0, 2] == pytest.approx(2.0)
    assert matrix[1, 3] == pytest.approx(0.0)


def test_assembly_is_jittable_and_differentiable_for_fixed_connectivity():
    nodes, triangles = square_mesh_arrays()

    mass = jax.jit(assemble_mass_matrix)(nodes, triangles)
    stiffness = jax.jit(assemble_laplace_stiffness_matrix)(nodes, triangles)
    mass_total_grad = jax.grad(lambda coords: jnp.sum(assemble_mass_matrix(coords, triangles)))(
        nodes
    )
    stiffness_energy_grad = jax.grad(
        lambda coords: jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=coords.dtype)
        @ assemble_laplace_stiffness_matrix(coords, triangles)
        @ jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=coords.dtype)
    )(nodes)

    np.testing.assert_allclose(jnp.sum(mass), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(jnp.sum(stiffness, axis=1), np.zeros(4), atol=1.0e-14)
    assert mass_total_grad.shape == nodes.shape
    assert stiffness_energy_grad.shape == nodes.shape
    assert bool(jnp.all(jnp.isfinite(mass_total_grad)))
    assert bool(jnp.all(jnp.isfinite(stiffness_energy_grad)))


def test_assembly_validates_input_shapes():
    nodes, triangles = square_mesh_arrays()
    mesh = mesh_from_arrays(np.asarray(nodes), np.asarray(triangles), np.asarray([1, 1]))

    with pytest.raises(ValueError, match="triangles must be provided"):
        assemble_mass_matrix(nodes)
    with pytest.raises(ValueError, match="triangles must be omitted"):
        assemble_mass_matrix(mesh, triangles)
    with pytest.raises(ValueError, match="nodes must have shape"):
        assemble_mass_matrix(jnp.ones((4, 3)), triangles)
    with pytest.raises(ValueError, match="at least 3"):
        assemble_mass_matrix(jnp.ones((2, 2)), jnp.asarray([[0, 1, 2]], dtype=jnp.int32))
    with pytest.raises(ValueError, match="triangles must have shape"):
        assemble_mass_matrix(nodes, jnp.ones((2, 4), dtype=jnp.int32))
    with pytest.raises(ValueError, match="element_matrices must have shape"):
        assemble_global_matrix(jnp.ones((2, 2, 2)), triangles, n_nodes=4)
    with pytest.raises(ValueError, match="triangles must have shape"):
        assemble_global_matrix(jnp.ones((2, 3, 3)), jnp.ones((2, 4), dtype=jnp.int32), n_nodes=4)
    with pytest.raises(ValueError, match="same number of cells"):
        assemble_global_matrix(jnp.ones((1, 3, 3)), triangles, n_nodes=4)
    with pytest.raises(ValueError, match="n_nodes"):
        assemble_global_matrix(jnp.ones((2, 3, 3)), triangles, n_nodes=2)
