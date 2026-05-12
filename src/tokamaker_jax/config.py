"""Typed TOML configuration for tokamaker-jax."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


@dataclass(frozen=True)
class GridConfig:
    """Rectangular seed grid.

    The full TokaMaker port will replace this with differentiable triangular FEM meshes.
    """

    r_min: float = 0.5
    r_max: float = 1.5
    z_min: float = -0.5
    z_max: float = 0.5
    nr: int = 65
    nz: int = 65


@dataclass(frozen=True)
class CoilConfig:
    """Axisymmetric Gaussian coil source used by the seed solver."""

    name: str
    r: float
    z: float
    current: float
    sigma: float = 0.04


@dataclass(frozen=True)
class SourceConfig:
    """Flux-function source parameters for the seed solver."""

    profile: str = "solovev"
    pressure_scale: float = 5.0e3
    ffp_scale: float = -0.35


@dataclass(frozen=True)
class SolverConfig:
    """Nonlinear and linear solver controls."""

    iterations: int = 600
    relaxation: float = 0.75
    dtype: str = "float64"


@dataclass(frozen=True)
class OutputConfig:
    """Optional CLI output paths."""

    npz: str | None = None
    plot: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Complete TOML-backed run configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    coils: tuple[CoilConfig, ...] = ()
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary useful for logging and serialization."""

        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    """Load a TOML run configuration.

    Python 3.10 uses `tomli`; Python 3.11 and newer use the standard-library
    `tomllib`.
    """

    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Build :class:`RunConfig` from a parsed TOML dictionary."""

    grid = GridConfig(**_section(raw, "grid"))
    source = SourceConfig(**_section(raw, "source"))
    solver = SolverConfig(**_section(raw, "solver"))
    output = OutputConfig(**_section(raw, "output"))
    coil_entries = raw.get("coil", raw.get("coils", []))
    coils = tuple(CoilConfig(**entry) for entry in coil_entries)
    return RunConfig(grid=grid, source=source, solver=solver, coils=coils, output=output)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"TOML section [{key}] must be a table")
    return value
