"""JAX-native TokaMaker port scaffold."""

from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, load_config
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.fem import (
    TriangleQuadrature,
    linear_basis,
    linear_basis_gradients,
    linear_mass_matrix,
    linear_stiffness_matrix,
    reference_triangle_nodes,
    triangle_area,
    triangle_jacobian,
    triangle_quadrature,
)
from tokamaker_jax.geometry import (
    Region,
    RegionSet,
    annulus_region,
    polygon_region,
    rectangle_region,
)
from tokamaker_jax.mesh import TriMesh, load_gs_mesh, mesh_from_arrays, save_gs_mesh
from tokamaker_jax.plotting import FigureRecipe
from tokamaker_jax.profiles import MU0, gaussian_coil_source, normalized_flux, solovev_source
from tokamaker_jax.solver import (
    EquilibriumSolution,
    apply_operator,
    solve_fixed_boundary,
    solve_from_config,
)

__all__ = [
    "CoilConfig",
    "EquilibriumSolution",
    "FigureRecipe",
    "GridConfig",
    "MU0",
    "RectangularGrid",
    "Region",
    "RegionSet",
    "RunConfig",
    "SolverConfig",
    "TriMesh",
    "TriangleQuadrature",
    "annulus_region",
    "apply_operator",
    "gaussian_coil_source",
    "linear_basis",
    "linear_basis_gradients",
    "linear_mass_matrix",
    "linear_stiffness_matrix",
    "load_gs_mesh",
    "load_config",
    "mesh_from_arrays",
    "normalized_flux",
    "polygon_region",
    "rectangle_region",
    "reference_triangle_nodes",
    "save_gs_mesh",
    "solve_fixed_boundary",
    "solve_from_config",
    "solovev_source",
    "triangle_area",
    "triangle_jacobian",
    "triangle_quadrature",
]
