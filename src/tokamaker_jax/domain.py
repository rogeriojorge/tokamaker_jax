"""JAX-friendly computational domains."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class RectangularGrid:
    """Axisymmetric R-Z grid used by the initial differentiable seed solver."""

    r_min: float
    r_max: float
    z_min: float
    z_max: float
    nr: int
    nz: int

    def __post_init__(self) -> None:
        if self.r_min <= 0.0:
            raise ValueError("r_min must be positive for the Grad-Shafranov operator")
        if self.r_max <= self.r_min:
            raise ValueError("r_max must be greater than r_min")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min")
        if self.nr < 3 or self.nz < 3:
            raise ValueError("nr and nz must both be at least 3")

    @property
    def dr(self) -> float:
        return (self.r_max - self.r_min) / (self.nr - 1)

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)

    def axes(self, dtype: jnp.dtype = jnp.float64) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return one-dimensional R and Z coordinates."""

        r = jnp.linspace(self.r_min, self.r_max, self.nr, dtype=dtype)
        z = jnp.linspace(self.z_min, self.z_max, self.nz, dtype=dtype)
        return r, z

    def mesh(self, dtype: jnp.dtype = jnp.float64) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return two-dimensional ``R, Z`` arrays with indexing ``ij``."""

        return jnp.meshgrid(*self.axes(dtype=dtype), indexing="ij")

    def zeros(self, dtype: jnp.dtype = jnp.float64) -> jnp.ndarray:
        """Return a zero field on this grid."""

        return jnp.zeros((self.nr, self.nz), dtype=dtype)
