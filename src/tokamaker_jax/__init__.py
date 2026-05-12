"""JAX-native TokaMaker port scaffold."""

from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, load_config
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.mesh import TriMesh, load_gs_mesh, mesh_from_arrays, save_gs_mesh
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
    "GridConfig",
    "MU0",
    "RectangularGrid",
    "RunConfig",
    "SolverConfig",
    "TriMesh",
    "apply_operator",
    "gaussian_coil_source",
    "load_gs_mesh",
    "load_config",
    "mesh_from_arrays",
    "normalized_flux",
    "save_gs_mesh",
    "solve_fixed_boundary",
    "solve_from_config",
    "solovev_source",
]
