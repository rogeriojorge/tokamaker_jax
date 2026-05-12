"""Finite-element reference kernels for triangular TokaMaker meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TriangleQuadrature:
    """Quadrature rule on the reference triangle ``(0,0), (1,0), (0,1)``."""

    points: jnp.ndarray
    weights: jnp.ndarray
    degree: int


def reference_triangle_nodes(order: int = 1) -> jnp.ndarray:
    """Return nodal coordinates for the reference triangle."""

    if order != 1:
        raise NotImplementedError("only p=1 reference triangle nodes are implemented")
    return jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)


def triangle_quadrature(degree: int = 2) -> TriangleQuadrature:
    """Return a reference-triangle quadrature rule exact through ``degree``."""

    if degree <= 1:
        return TriangleQuadrature(
            points=jnp.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=jnp.float64),
            weights=jnp.asarray([0.5], dtype=jnp.float64),
            degree=1,
        )
    if degree <= 2:
        return TriangleQuadrature(
            points=jnp.asarray(
                [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
                dtype=jnp.float64,
            ),
            weights=jnp.asarray([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=jnp.float64),
            degree=2,
        )
    if degree <= 3:
        return TriangleQuadrature(
            points=jnp.asarray(
                [
                    [1.0 / 3.0, 1.0 / 3.0],
                    [0.6, 0.2],
                    [0.2, 0.6],
                    [0.2, 0.2],
                ],
                dtype=jnp.float64,
            ),
            weights=jnp.asarray(
                [-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0], dtype=jnp.float64
            ),
            degree=3,
        )
    raise NotImplementedError("triangle quadrature is currently implemented through degree 3")


def linear_basis(reference_points: jnp.ndarray) -> jnp.ndarray:
    """Evaluate p=1 Lagrange basis functions at reference points."""

    points = _as_points(reference_points)
    xi = points[:, 0]
    eta = points[:, 1]
    values = jnp.column_stack((1.0 - xi - eta, xi, eta))
    return values.reshape(reference_points.shape[:-1] + (3,))


def linear_basis_gradients() -> jnp.ndarray:
    """Return reference gradients of p=1 basis functions.

    Rows correspond to basis functions and columns correspond to ``xi, eta``.
    """

    return jnp.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)


def triangle_jacobian(vertices: jnp.ndarray) -> jnp.ndarray:
    """Return the affine Jacobian from reference to physical triangle."""

    vertices = _as_vertices(vertices)
    return jnp.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))


def triangle_area(vertices: jnp.ndarray) -> jnp.ndarray:
    """Return positive triangle area."""

    jacobian = triangle_jacobian(vertices)
    determinant = jnp.linalg.det(jacobian)
    return 0.5 * jnp.abs(determinant)


def map_to_physical(vertices: jnp.ndarray, reference_points: jnp.ndarray) -> jnp.ndarray:
    """Map reference points to physical ``R, Z`` coordinates."""

    vertices = _as_vertices(vertices)
    basis = linear_basis(reference_points)
    return basis @ vertices


def physical_basis_gradients(vertices: jnp.ndarray) -> jnp.ndarray:
    """Return physical-coordinate gradients for p=1 basis functions."""

    jacobian = triangle_jacobian(vertices)
    return linear_basis_gradients() @ jnp.linalg.inv(jacobian)


def linear_mass_matrix(vertices: jnp.ndarray) -> jnp.ndarray:
    """Return the exact p=1 element mass matrix."""

    area = triangle_area(vertices)
    return (
        area
        / 12.0
        * jnp.asarray([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=jnp.float64)
    )


def linear_stiffness_matrix(vertices: jnp.ndarray) -> jnp.ndarray:
    """Return the p=1 Laplace stiffness matrix for one triangle."""

    area = triangle_area(vertices)
    gradients = physical_basis_gradients(vertices)
    return area * (gradients @ gradients.T)


def linear_weighted_mass_matrix(
    vertices: jnp.ndarray,
    coefficient: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Return the p=1 mass matrix weighted by a physical-space coefficient.

    The matrix entries are

    ``M_ij = int_K coefficient(R, Z) phi_i(R, Z) phi_j(R, Z) dR dZ``.
    """

    quadrature = triangle_quadrature(quadrature_degree)
    physical_points = map_to_physical(vertices, quadrature.points)
    coefficient_values = _coefficient_values(coefficient, physical_points, quadrature.weights)
    basis = linear_basis(quadrature.points)
    det_jacobian = jnp.abs(jnp.linalg.det(triangle_jacobian(vertices)))
    return det_jacobian * jnp.einsum(
        "q,q,qi,qj->ij",
        quadrature.weights,
        coefficient_values,
        basis,
        basis,
    )


def linear_weighted_stiffness_matrix(
    vertices: jnp.ndarray,
    coefficient: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    quadrature_degree: int = 3,
) -> jnp.ndarray:
    """Return the p=1 stiffness matrix weighted by a coefficient.

    The matrix entries are

    ``A_ij = int_K coefficient(R, Z) grad(phi_i).grad(phi_j) dR dZ``.
    """

    quadrature = triangle_quadrature(quadrature_degree)
    physical_points = map_to_physical(vertices, quadrature.points)
    coefficient_values = _coefficient_values(coefficient, physical_points, quadrature.weights)
    gradients = physical_basis_gradients(vertices)
    det_jacobian = jnp.abs(jnp.linalg.det(triangle_jacobian(vertices)))
    coefficient_integral = det_jacobian * jnp.sum(quadrature.weights * coefficient_values)
    return coefficient_integral * (gradients @ gradients.T)


def _as_points(points: jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(points, dtype=jnp.float64)
    if array.shape[-1] != 2:
        raise ValueError("reference points must have final dimension 2")
    return array.reshape((-1, 2))


def _as_vertices(vertices: jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(vertices, dtype=jnp.float64)
    if array.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")
    return array


def _coefficient_values(
    coefficient: Callable[[jnp.ndarray], jnp.ndarray],
    physical_points: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    values = jnp.asarray(coefficient(physical_points), dtype=jnp.float64)
    if values.shape != weights.shape:
        raise ValueError("coefficient must return one value per quadrature point")
    return values
