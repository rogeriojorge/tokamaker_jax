"""Differentiable profile and source helpers."""

from __future__ import annotations

import jax.numpy as jnp

from tokamaker_jax.domain import RectangularGrid

MU0 = 4.0e-7 * jnp.pi


def normalized_flux(psi: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Normalize flux to ``[0, 1]`` using JAX operations."""

    psi_min = jnp.min(psi)
    psi_max = jnp.max(psi)
    return (psi - psi_min) / (psi_max - psi_min + eps)


def power_profile(psin: jnp.ndarray, alpha: float, gamma: float) -> jnp.ndarray:
    """TokaMaker-style power-law flux profile."""

    clipped = jnp.clip(1.0 - psin, 0.0, 1.0)
    return (clipped**alpha) ** gamma


def solovev_source(
    grid: RectangularGrid,
    pressure_scale: float = 5.0e3,
    ffp_scale: float = -0.35,
    dtype: jnp.dtype = jnp.float64,
) -> jnp.ndarray:
    """Return a simple Solov'ev-like Grad-Shafranov source.

    This seed model mirrors the TokaMaker equation structure
    ``Delta* psi = -0.5 dF^2/dpsi - mu0 R^2 dP/dpsi`` with constant profile
    derivatives. It is intentionally small and differentiable while the full
    triangular FEM machinery is ported.
    """

    r, _ = grid.mesh(dtype=dtype)
    return -0.5 * ffp_scale - MU0 * r**2 * pressure_scale


def gaussian_coil_source(
    grid: RectangularGrid,
    coils: tuple[object, ...],
    dtype: jnp.dtype = jnp.float64,
) -> jnp.ndarray:
    """Approximate axisymmetric coil sources with normalized Gaussians."""

    r, z = grid.mesh(dtype=dtype)
    source = jnp.zeros((grid.nr, grid.nz), dtype=dtype)
    for coil in coils:
        sigma = jnp.asarray(coil.sigma, dtype=dtype)
        radius = (r - coil.r) ** 2 + (z - coil.z) ** 2
        jphi = coil.current * jnp.exp(-0.5 * radius / sigma**2) / (2.0 * jnp.pi * sigma**2)
        source = source - MU0 * r * jphi
    return source
