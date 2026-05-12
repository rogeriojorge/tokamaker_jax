import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamaker_jax.fem import (
    linear_basis,
    linear_basis_gradients,
    linear_mass_matrix,
    linear_stiffness_matrix,
    linear_weighted_mass_matrix,
    linear_weighted_stiffness_matrix,
    map_to_physical,
    physical_basis_gradients,
    reference_triangle_nodes,
    triangle_area,
    triangle_jacobian,
    triangle_quadrature,
)


def analytic_reference_integral(power_x: int, power_y: int) -> float:
    return math.factorial(power_x) * math.factorial(power_y) / math.factorial(power_x + power_y + 2)


def test_reference_triangle_nodes_and_basis_partition():
    nodes = reference_triangle_nodes()
    values = linear_basis(nodes)

    np.testing.assert_allclose(nodes, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(values, np.eye(3), atol=1.0e-14)

    quadrature = triangle_quadrature(degree=2)
    basis = linear_basis(quadrature.points)
    np.testing.assert_allclose(np.sum(basis, axis=1), np.ones(3), atol=1.0e-14)


def test_linear_basis_gradients_match_finite_difference():
    point = jnp.asarray([0.23, 0.31], dtype=jnp.float64)
    gradients = linear_basis_gradients()

    for basis_index in range(3):
        finite_difference = jax.jacfwd(
            lambda x, basis_index=basis_index: linear_basis(x)[basis_index]
        )(point)
        np.testing.assert_allclose(finite_difference, gradients[basis_index], atol=1.0e-12)


def test_quadrature_exactness_for_degree_two_monomials():
    quadrature = triangle_quadrature(degree=2)

    for power_x in range(3):
        for power_y in range(3 - power_x):
            values = quadrature.points[:, 0] ** power_x * quadrature.points[:, 1] ** power_y
            numerical = float(jnp.sum(quadrature.weights * values))
            exact = analytic_reference_integral(power_x, power_y)
            assert numerical == pytest.approx(exact, abs=1.0e-14)


def test_quadrature_exactness_for_degree_three_monomials():
    quadrature = triangle_quadrature(degree=3)

    assert quadrature.degree == 3
    assert float(jnp.min(quadrature.weights)) < 0.0
    for power_x in range(4):
        for power_y in range(4 - power_x):
            values = quadrature.points[:, 0] ** power_x * quadrature.points[:, 1] ** power_y
            numerical = float(jnp.sum(quadrature.weights * values))
            exact = analytic_reference_integral(power_x, power_y)
            assert numerical == pytest.approx(exact, abs=1.0e-14)


def test_degree_one_quadrature_and_unsupported_orders():
    quadrature = triangle_quadrature(degree=1)

    assert quadrature.degree == 1
    assert float(jnp.sum(quadrature.weights)) == pytest.approx(0.5)

    with pytest.raises(NotImplementedError, match="p=1"):
        reference_triangle_nodes(order=2)
    with pytest.raises(NotImplementedError, match="degree 3"):
        triangle_quadrature(degree=4)


def test_affine_mapping_area_and_physical_gradients():
    vertices = jnp.asarray([[1.0, -0.5], [3.0, -0.25], [1.25, 1.0]], dtype=jnp.float64)
    jacobian = triangle_jacobian(vertices)
    area = triangle_area(vertices)
    mapped = map_to_physical(vertices, jnp.asarray([[0.0, 0.0], [0.25, 0.5]], dtype=jnp.float64))

    assert area == pytest.approx(1.46875)
    np.testing.assert_allclose(jacobian[:, 0], vertices[1] - vertices[0])
    np.testing.assert_allclose(jacobian[:, 1], vertices[2] - vertices[0])
    np.testing.assert_allclose(mapped[0], vertices[0])

    physical_gradients = physical_basis_gradients(vertices)
    np.testing.assert_allclose(jnp.sum(physical_gradients, axis=0), [0.0, 0.0], atol=1.0e-14)


def test_linear_element_matrices_are_symmetric_and_consistent():
    vertices = jnp.asarray([[0.5, -0.2], [1.7, -0.1], [0.7, 0.9]], dtype=jnp.float64)
    area = triangle_area(vertices)
    mass = linear_mass_matrix(vertices)
    stiffness = linear_stiffness_matrix(vertices)

    np.testing.assert_allclose(mass, mass.T, atol=1.0e-14)
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1.0e-14)
    np.testing.assert_allclose(jnp.sum(mass), area, atol=1.0e-14)
    np.testing.assert_allclose(jnp.sum(stiffness, axis=1), np.zeros(3), atol=1.0e-14)
    assert float(jnp.min(jnp.linalg.eigvalsh(mass))) > 0.0
    assert float(jnp.min(jnp.linalg.eigvalsh(stiffness))) > -1.0e-12


def test_weighted_element_matrices_reduce_to_constant_scaling():
    vertices = jnp.asarray([[0.5, -0.2], [1.7, -0.1], [0.7, 0.9]], dtype=jnp.float64)

    weighted_mass = linear_weighted_mass_matrix(
        vertices,
        lambda points: jnp.full(points.shape[0], 2.5, dtype=points.dtype),
    )
    weighted_stiffness = linear_weighted_stiffness_matrix(
        vertices,
        lambda points: jnp.full(points.shape[0], 2.5, dtype=points.dtype),
    )

    np.testing.assert_allclose(weighted_mass, 2.5 * linear_mass_matrix(vertices), atol=1.0e-14)
    np.testing.assert_allclose(
        weighted_stiffness,
        2.5 * linear_stiffness_matrix(vertices),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(weighted_mass, weighted_mass.T, atol=1.0e-14)
    np.testing.assert_allclose(weighted_stiffness, weighted_stiffness.T, atol=1.0e-14)


def test_fem_helpers_validate_input_shape():
    with pytest.raises(ValueError, match="final dimension 2"):
        linear_basis(jnp.asarray([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="shape"):
        triangle_area(jnp.asarray([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="one value per quadrature point"):
        linear_weighted_mass_matrix(
            jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
            lambda points: jnp.ones(points.shape[0] + 1),
        )
