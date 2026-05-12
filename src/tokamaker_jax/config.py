"""Typed TOML configuration for tokamaker-jax."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokamaker_jax.geometry import Region, RegionSet

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
    regions: RegionSet | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary useful for logging and serialization."""

        data = asdict(self)
        if self.regions is not None:
            data["regions"] = self.regions.to_dicts()
        return data


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
    regions = regions_from_dict(raw)
    return RunConfig(
        grid=grid,
        source=source,
        solver=solver,
        coils=coils,
        output=output,
        regions=regions,
    )


def regions_from_dict(raw: dict[str, Any]) -> RegionSet | None:
    """Build an optional :class:`RegionSet` from ``region`` or ``regions`` entries."""

    has_region = "region" in raw
    has_regions = "regions" in raw
    if has_region and has_regions:
        raise ValueError("TOML may define only one of [[region]] or [[regions]]")
    if not has_region and not has_regions:
        return None

    key = "region" if has_region else "regions"
    entries = raw[key]
    if not isinstance(entries, list):
        raise TypeError(f"TOML section [[{key}]] must be an array of tables")
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError(f"TOML section [[{key}]] entries must be tables")

    from tokamaker_jax.geometry import RegionSet

    return RegionSet(
        tuple(
            _region_from_entry(entry, index=index, key=key) for index, entry in enumerate(entries)
        )
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"TOML section [{key}] must be a table")
    return value


def _region_from_entry(entry: dict[str, Any], *, index: int, key: str) -> Region:
    from tokamaker_jax.geometry import Region, annulus_region, polygon_region, rectangle_region

    shape = entry.get("shape", entry.get("type"))
    if shape is None:
        if "points" not in entry:
            raise ValueError(f"[[{key}]] entry {index} must define shape/type or points")
        return Region.from_dict(entry)

    if not isinstance(shape, str):
        raise TypeError(f"[[{key}]] entry {index} shape/type must be a string")

    shape = shape.lower()
    common = _region_common(entry, key=key, index=index)
    try:
        if shape == "rectangle":
            _require_region_keys(entry, ("r_min", "r_max", "z_min", "z_max"), key=key, index=index)
            return rectangle_region(
                **common,
                r_min=entry["r_min"],
                r_max=entry["r_max"],
                z_min=entry["z_min"],
                z_max=entry["z_max"],
            )
        if shape == "polygon":
            _require_region_keys(entry, ("points",), key=key, index=index)
            return polygon_region(**common, points=entry["points"])
        if shape == "annulus":
            _require_region_keys(
                entry,
                ("center_r", "center_z", "inner_radius", "outer_radius"),
                key=key,
                index=index,
            )
            return annulus_region(
                **common,
                center_r=entry["center_r"],
                center_z=entry["center_z"],
                inner_radius=entry["inner_radius"],
                outer_radius=entry["outer_radius"],
                n=entry.get("n", 96),
            )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid [[{key}]] entry {index}: {exc}") from exc
    raise ValueError(f"[[{key}]] entry {index} has unsupported shape/type {shape!r}")


def _region_common(entry: dict[str, Any], *, key: str, index: int) -> dict[str, Any]:
    _require_region_keys(entry, ("id", "name"), key=key, index=index)
    return {
        "id": entry["id"],
        "name": entry["name"],
        "kind": entry.get("kind", "unknown"),
        "target_size": entry.get("target_size"),
        "metadata": entry.get("metadata"),
    }


def _require_region_keys(
    entry: dict[str, Any], required: tuple[str, ...], *, key: str, index: int
) -> None:
    missing = [name for name in required if name not in entry]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"[[{key}]] entry {index} missing required key(s): {names}")
