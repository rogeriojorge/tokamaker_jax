"""Global finite-element assembly for triangular meshes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from tokamaker_jax.fem import linear_mass_matrix, linear_stiffness_matrix
from tokamaker_jax.mesh import TriMesh


def assemble_global_matrix(
    element_matrices: jnp.ndarray,
    triangles: jnp.ndarray,
    n_nodes: int,
) -> jnp.ndarray:
    """Assemble dense nodal matrix contributions from triangular elements.

    Parameters
    ----------
    element_matrices:
        Per-cell matrices with shape ``(n_cells, 3, 3)``.
    triangles:
        Node indices with shape ``(n_cells, 3)``.
    n_nodes:
        Number of global mesh nodes. This is a Python integer so the output
        shape stays static under ``jax.jit``.
    """

    element_matrices = jnp.asarray(element_matrices)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    _validate_element_matrix_shapes(element_matrices, triangles, n_nodes)

    rows = jnp.repeat(triangles, 3, axis=1).reshape(-1)
    cols = jnp.tile(triangles, (1, 3)).reshape(-1)
    values = element_matrices.reshape(-1)
    return jnp.zeros((int(n_nodes), int(n_nodes)), dtype=values.dtype).at[rows, cols].add(values)


def assemble_mass_matrix(
    mesh_or_nodes: TriMesh | Any, triangles: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Assemble the dense global p=1 triangular mass matrix.

    Pass either a :class:`~tokamaker_jax.mesh.TriMesh` or ``(nodes, triangles)``
    arrays. The array path is compatible with ``jax.jit`` and differentiation
    with respect to ``nodes`` for fixed connectivity.
    """

    nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles)
    element_matrices = jax.vmap(linear_mass_matrix)(nodes[triangles])
    return assemble_global_matrix(element_matrices, triangles, nodes.shape[0])


def assemble_laplace_stiffness_matrix(
    mesh_or_nodes: TriMesh | Any,
    triangles: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Assemble the dense global p=1 triangular Laplace stiffness matrix.

    Pass either a :class:`~tokamaker_jax.mesh.TriMesh` or ``(nodes, triangles)``
    arrays. The array path is compatible with ``jax.jit`` and differentiation
    with respect to ``nodes`` for fixed connectivity.
    """

    nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles)
    element_matrices = jax.vmap(linear_stiffness_matrix)(nodes[triangles])
    return assemble_global_matrix(element_matrices, triangles, nodes.shape[0])


def _mesh_arrays(
    mesh_or_nodes: TriMesh | Any,
    triangles: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if isinstance(mesh_or_nodes, TriMesh):
        if triangles is not None:
            raise ValueError("triangles must be omitted when passing a TriMesh")
        nodes = jnp.asarray(mesh_or_nodes.nodes)
        triangles = jnp.asarray(mesh_or_nodes.triangles, dtype=jnp.int32)
    else:
        if triangles is None:
            raise ValueError("triangles must be provided when passing node arrays")
        nodes = jnp.asarray(mesh_or_nodes)
        triangles = jnp.asarray(triangles, dtype=jnp.int32)
    _validate_mesh_array_shapes(nodes, triangles)
    return nodes, triangles


def _validate_mesh_array_shapes(nodes: jnp.ndarray, triangles: jnp.ndarray) -> None:
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n_cells, 3)")
    if nodes.shape[0] < 3:
        raise ValueError("nodes must contain at least 3 entries")


def _validate_element_matrix_shapes(
    element_matrices: jnp.ndarray,
    triangles: jnp.ndarray,
    n_nodes: int,
) -> None:
    if element_matrices.ndim != 3 or element_matrices.shape[1:] != (3, 3):
        raise ValueError("element_matrices must have shape (n_cells, 3, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n_cells, 3)")
    if element_matrices.shape[0] != triangles.shape[0]:
        raise ValueError("element_matrices and triangles must have the same number of cells")
    if int(n_nodes) < 3:
        raise ValueError("n_nodes must be at least 3")
