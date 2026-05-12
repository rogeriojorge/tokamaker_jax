"""Global finite-element assembly for triangular meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from tokamaker_jax.fem import (
    linear_basis,
    linear_mass_matrix,
    linear_stiffness_matrix,
    map_to_physical,
    triangle_jacobian,
    triangle_quadrature,
)
from tokamaker_jax.mesh import TriMesh


@dataclass(frozen=True)
class DirichletSystem:
    """Reduced dense linear system after applying nodal Dirichlet values."""

    matrix: jnp.ndarray
    rhs: jnp.ndarray
    free_nodes: jnp.ndarray
    dirichlet_nodes: jnp.ndarray
    dirichlet_values: jnp.ndarray
    n_nodes: int


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


def assemble_global_bcoo(
    element_matrices: jnp.ndarray,
    triangles: jnp.ndarray,
    n_nodes: int,
) -> BCOO:
    """Assemble sparse nodal matrix contributions as a JAX ``BCOO`` matrix."""

    element_matrices = jnp.asarray(element_matrices)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    _validate_element_matrix_shapes(element_matrices, triangles, n_nodes)

    rows = jnp.repeat(triangles, 3, axis=1).reshape(-1)
    cols = jnp.tile(triangles, (1, 3)).reshape(-1)
    values = element_matrices.reshape(-1)
    indices = jnp.column_stack((rows, cols))
    return BCOO((values, indices), shape=(int(n_nodes), int(n_nodes))).sum_duplicates()


def apply_global_matrix(
    element_matrices: jnp.ndarray,
    triangles: jnp.ndarray,
    vector: jnp.ndarray,
    n_nodes: int,
) -> jnp.ndarray:
    """Apply an assembled operator matrix-free using element scatter-adds."""

    element_matrices = jnp.asarray(element_matrices)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    vector = jnp.asarray(vector)
    _validate_element_matrix_shapes(element_matrices, triangles, n_nodes)
    if vector.shape != (int(n_nodes),):
        raise ValueError("vector must have shape (n_nodes,)")

    local_values = jnp.einsum("eij,ej->ei", element_matrices, vector[triangles])
    return (
        jnp.zeros((int(n_nodes),), dtype=local_values.dtype)
        .at[triangles.reshape(-1)]
        .add(local_values.reshape(-1))
    )


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


def assemble_mass_bcoo(mesh_or_nodes: TriMesh | Any, triangles: jnp.ndarray | None = None) -> BCOO:
    """Assemble the sparse global p=1 triangular mass matrix."""

    nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles)
    element_matrices = jax.vmap(linear_mass_matrix)(nodes[triangles])
    return assemble_global_bcoo(element_matrices, triangles, nodes.shape[0])


def apply_mass_matrix(
    mesh_or_nodes: TriMesh | Any,
    triangles_or_vector: jnp.ndarray,
    vector: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply the p=1 triangular mass matrix without materializing it densely."""

    nodes, triangles, vector = _mesh_arrays_and_vector(mesh_or_nodes, triangles_or_vector, vector)
    element_matrices = jax.vmap(linear_mass_matrix)(nodes[triangles])
    return apply_global_matrix(element_matrices, triangles, vector, nodes.shape[0])


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


def assemble_laplace_stiffness_bcoo(
    mesh_or_nodes: TriMesh | Any,
    triangles: jnp.ndarray | None = None,
) -> BCOO:
    """Assemble the sparse global p=1 triangular Laplace stiffness matrix."""

    nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles)
    element_matrices = jax.vmap(linear_stiffness_matrix)(nodes[triangles])
    return assemble_global_bcoo(element_matrices, triangles, nodes.shape[0])


def apply_laplace_stiffness_matrix(
    mesh_or_nodes: TriMesh | Any,
    triangles_or_vector: jnp.ndarray,
    vector: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply the p=1 triangular Laplace stiffness matrix matrix-free."""

    nodes, triangles, vector = _mesh_arrays_and_vector(mesh_or_nodes, triangles_or_vector, vector)
    element_matrices = jax.vmap(linear_stiffness_matrix)(nodes[triangles])
    return apply_global_matrix(element_matrices, triangles, vector, nodes.shape[0])


def linear_load_vector(
    vertices: jnp.ndarray,
    source: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Integrate one p=1 triangular load vector against a callable source."""

    quadrature = triangle_quadrature(quadrature_degree)
    physical_points = map_to_physical(vertices, quadrature.points)
    source_values = jnp.asarray(source(physical_points), dtype=jnp.float64)
    if source_values.shape != quadrature.weights.shape:
        raise ValueError("source must return one value per quadrature point")
    basis = linear_basis(quadrature.points)
    det_jacobian = jnp.abs(jnp.linalg.det(triangle_jacobian(vertices)))
    return det_jacobian * jnp.sum(
        quadrature.weights[:, None] * source_values[:, None] * basis,
        axis=0,
    )


def assemble_load_vector(
    mesh_or_nodes: TriMesh | Any,
    triangles_or_source: jnp.ndarray | Callable[[jnp.ndarray], jnp.ndarray],
    source: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Assemble a dense p=1 triangular load vector from a callable source."""

    nodes, triangles, source = _mesh_arrays_and_source(mesh_or_nodes, triangles_or_source, source)
    element_vectors = jax.vmap(
        lambda vertices: linear_load_vector(vertices, source, quadrature_degree=quadrature_degree)
    )(nodes[triangles])
    return (
        jnp.zeros((nodes.shape[0],), dtype=element_vectors.dtype)
        .at[triangles.reshape(-1)]
        .add(element_vectors.reshape(-1))
    )


def boundary_nodes_from_coordinates(nodes: jnp.ndarray, *, atol: float = 1.0e-12) -> jnp.ndarray:
    """Return nodes on the rectangular boundary of a coordinate cloud."""

    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    r = nodes[:, 0]
    z = nodes[:, 1]
    mask = (
        jnp.isclose(r, jnp.min(r), atol=atol)
        | jnp.isclose(r, jnp.max(r), atol=atol)
        | jnp.isclose(z, jnp.min(z), atol=atol)
        | jnp.isclose(z, jnp.max(z), atol=atol)
    )
    return jnp.where(mask)[0]


def apply_dirichlet_conditions(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    dirichlet_nodes: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
) -> DirichletSystem:
    """Return the reduced dense system for fixed nodal Dirichlet values."""

    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)
    dirichlet_nodes = jnp.asarray(dirichlet_nodes, dtype=jnp.int32)
    dirichlet_values = jnp.asarray(dirichlet_values, dtype=rhs.dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n_nodes = matrix.shape[0]
    if rhs.shape != (n_nodes,):
        raise ValueError("rhs must have shape (n_nodes,)")
    if dirichlet_nodes.ndim != 1:
        raise ValueError("dirichlet_nodes must be one-dimensional")
    if dirichlet_values.shape != dirichlet_nodes.shape:
        raise ValueError("dirichlet_values must match dirichlet_nodes")
    if dirichlet_nodes.size == 0:
        raise ValueError("at least one Dirichlet node is required")

    mask = jnp.ones((n_nodes,), dtype=bool).at[dirichlet_nodes].set(False)
    free_nodes = jnp.where(mask)[0]
    matrix_ff = matrix[jnp.ix_(free_nodes, free_nodes)]
    matrix_fd = matrix[jnp.ix_(free_nodes, dirichlet_nodes)]
    rhs_free = rhs[free_nodes] - matrix_fd @ dirichlet_values
    return DirichletSystem(
        matrix=matrix_ff,
        rhs=rhs_free,
        free_nodes=free_nodes,
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=dirichlet_values,
        n_nodes=n_nodes,
    )


def solve_dirichlet_system(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    dirichlet_nodes: jnp.ndarray,
    dirichlet_values: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a dense system with nodal Dirichlet values and return all nodes."""

    system = apply_dirichlet_conditions(matrix, rhs, dirichlet_nodes, dirichlet_values)
    free_values = jnp.linalg.solve(system.matrix, system.rhs)
    solution = jnp.zeros((system.n_nodes,), dtype=rhs.dtype)
    solution = solution.at[system.free_nodes].set(free_values)
    return solution.at[system.dirichlet_nodes].set(system.dirichlet_values)


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


def _mesh_arrays_and_vector(
    mesh_or_nodes: TriMesh | Any,
    triangles_or_vector: jnp.ndarray,
    vector: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if isinstance(mesh_or_nodes, TriMesh):
        if vector is not None:
            raise ValueError("vector must be the second positional argument when passing a TriMesh")
        nodes, triangles = _mesh_arrays(mesh_or_nodes, None)
        vector = jnp.asarray(triangles_or_vector)
    else:
        if vector is None:
            raise ValueError("vector must be provided when passing node arrays")
        nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles_or_vector)
        vector = jnp.asarray(vector)
    if vector.shape != (nodes.shape[0],):
        raise ValueError("vector must have shape (n_nodes,)")
    return nodes, triangles, vector


def _mesh_arrays_and_source(
    mesh_or_nodes: TriMesh | Any,
    triangles_or_source: jnp.ndarray | Callable[[jnp.ndarray], jnp.ndarray],
    source: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> tuple[jnp.ndarray, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
    if isinstance(mesh_or_nodes, TriMesh):
        if source is not None:
            raise ValueError("source must be the second positional argument when passing a TriMesh")
        if not callable(triangles_or_source):
            raise TypeError("source must be callable")
        nodes, triangles = _mesh_arrays(mesh_or_nodes, None)
        return nodes, triangles, triangles_or_source
    if source is None:
        raise ValueError("source must be provided when passing node arrays")
    if not callable(source):
        raise TypeError("source must be callable")
    nodes, triangles = _mesh_arrays(mesh_or_nodes, triangles_or_source)
    return nodes, triangles, source


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
