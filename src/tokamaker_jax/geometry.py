"""Machine-region geometry primitives for TokaMaker cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

RegionKind = Literal["plasma", "vacuum", "conductor", "coil", "boundary", "limiter", "unknown"]
VALID_REGION_KINDS = frozenset(
    ("plasma", "vacuum", "conductor", "coil", "boundary", "limiter", "unknown")
)


@dataclass(frozen=True)
class Region:
    """A named 2D region in axisymmetric ``R, Z`` coordinates."""

    id: int
    name: str
    kind: RegionKind = "unknown"
    points: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    holes: tuple[NDArray[np.float64], ...] = ()
    target_size: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.id) < 1:
            raise ValueError("region id must be positive and one-based")
        if not self.name:
            raise ValueError("region name must be nonempty")
        if self.kind not in VALID_REGION_KINDS:
            raise ValueError(f"unsupported region kind {self.kind!r}")
        if self.target_size is not None and self.target_size <= 0.0:
            raise ValueError("target_size must be positive when provided")

        points = ensure_counterclockwise(_as_loop(self.points, "points"))
        holes = tuple(ensure_counterclockwise(_as_loop(hole, "hole")) for hole in self.holes)
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "holes", holes)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(r_min, r_max, z_min, z_max)`` for the outer loop."""

        return bounds(self.points)

    @property
    def area(self) -> float:
        """Return poloidal area, subtracting any holes."""

        hole_area = sum(abs(polygon_area(hole)) for hole in self.holes)
        return float(abs(polygon_area(self.points)) - hole_area)

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the outer-loop centroid."""

        value = polygon_centroid(self.points)
        return float(value[0]), float(value[1])

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Return a mask for points contained by the region and outside holes."""

        query = np.asarray(points, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, 2)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        mask = points_in_polygon(query, self.points)
        for hole in self.holes:
            mask &= ~points_in_polygon(query, hole)
        return mask

    def to_dict(self) -> dict[str, Any]:
        """Return a TOML/JSON-friendly dictionary."""

        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "points": self.points.tolist(),
        }
        if self.holes:
            data["holes"] = [hole.tolist() for hole in self.holes]
        if self.target_size is not None:
            data["target_size"] = float(self.target_size)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        """Build a :class:`Region` from a TOML/JSON-friendly dictionary."""

        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            kind=data.get("kind", "unknown"),
            points=np.asarray(data["points"], dtype=np.float64),
            holes=tuple(np.asarray(hole, dtype=np.float64) for hole in data.get("holes", ())),
            target_size=data.get("target_size"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class RegionSet:
    """Validated collection of machine regions."""

    regions: tuple[Region, ...]

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions:
            raise ValueError("RegionSet must contain at least one region")
        ids = [region.id for region in regions]
        names = [region.name for region in regions]
        if len(set(ids)) != len(ids):
            raise ValueError("region ids must be unique")
        if len(set(names)) != len(names):
            raise ValueError("region names must be unique")
        object.__setattr__(self, "regions", regions)

    def by_kind(self, kind: RegionKind) -> tuple[Region, ...]:
        """Return regions of a given kind."""

        return tuple(region for region in self.regions if region.kind == kind)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all regions as TOML/JSON-friendly dictionaries."""

        return [region.to_dict() for region in self.regions]

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> RegionSet:
        """Build a :class:`RegionSet` from dictionaries."""

        return cls(tuple(Region.from_dict(entry) for entry in data))


def rectangle_region(
    *,
    id: int,
    name: str,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    kind: RegionKind = "unknown",
    target_size: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Region:
    """Create a rectangular region."""

    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min")
    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min")
    return Region(
        id=id,
        name=name,
        kind=kind,
        points=np.asarray(
            [
                [r_min, z_min],
                [r_max, z_min],
                [r_max, z_max],
                [r_min, z_max],
            ],
            dtype=np.float64,
        ),
        target_size=target_size,
        metadata={} if metadata is None else metadata,
    )


def polygon_region(
    *,
    id: int,
    name: str,
    points: ArrayLike,
    kind: RegionKind = "unknown",
    target_size: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Region:
    """Create a polygonal region."""

    return Region(
        id=id,
        name=name,
        kind=kind,
        points=np.asarray(points, dtype=np.float64),
        target_size=target_size,
        metadata={} if metadata is None else metadata,
    )


def annulus_region(
    *,
    id: int,
    name: str,
    center_r: float,
    center_z: float,
    inner_radius: float,
    outer_radius: float,
    n: int = 96,
    kind: RegionKind = "unknown",
    target_size: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Region:
    """Create an annular region approximated by polygon loops."""

    if inner_radius <= 0.0:
        raise ValueError("inner_radius must be positive")
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be greater than inner_radius")
    if n < 12:
        raise ValueError("n must be at least 12")
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    outer = _circle_points(center_r, center_z, outer_radius, angles)
    inner = _circle_points(center_r, center_z, inner_radius, angles)
    return Region(
        id=id,
        name=name,
        kind=kind,
        points=outer,
        holes=(inner,),
        target_size=target_size,
        metadata={} if metadata is None else metadata,
    )


def sample_regions() -> RegionSet:
    """Return a small canonical region set used by examples, tests, and the GUI."""

    return RegionSet(
        (
            annulus_region(
                id=2,
                name="VV",
                kind="conductor",
                center_r=2.0,
                center_z=0.0,
                inner_radius=1.05,
                outer_radius=1.25,
                n=96,
                target_size=0.05,
                metadata={"role": "vacuum_vessel"},
            ),
            rectangle_region(
                id=1,
                name="PLASMA",
                kind="plasma",
                r_min=1.35,
                r_max=2.65,
                z_min=-0.75,
                z_max=0.75,
                target_size=0.08,
                metadata={"role": "seed_plasma_domain"},
            ),
            rectangle_region(
                id=3,
                name="PF",
                kind="coil",
                r_min=3.25,
                r_max=3.55,
                z_min=-0.25,
                z_max=0.25,
                target_size=0.04,
                metadata={"role": "seed_pf_coil"},
            ),
        )
    )


def polygon_area(points: ArrayLike) -> float:
    """Return signed polygon area by the shoelace formula."""

    loop = _as_loop(points, "points")
    x = loop[:, 0]
    y = loop[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(points: ArrayLike) -> NDArray[np.float64]:
    """Return polygon centroid for a nondegenerate simple polygon."""

    loop = _as_loop(points, "points")
    signed_area = polygon_area(loop)
    if abs(signed_area) <= np.finfo(np.float64).eps:
        raise ValueError("polygon centroid is undefined for zero-area polygons")
    x = loop[:, 0]
    y = loop[:, 1]
    factor = x * np.roll(y, -1) - np.roll(x, -1) * y
    centroid = np.asarray(
        [
            np.sum((x + np.roll(x, -1)) * factor),
            np.sum((y + np.roll(y, -1)) * factor),
        ],
        dtype=np.float64,
    )
    return centroid / (6.0 * signed_area)


def ensure_counterclockwise(points: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of points oriented counterclockwise."""

    loop = _as_loop(points, "points")
    if polygon_area(loop) < 0.0:
        return np.ascontiguousarray(loop[::-1])
    return np.ascontiguousarray(loop)


def bounds(points: ArrayLike) -> tuple[float, float, float, float]:
    """Return ``(r_min, r_max, z_min, z_max)`` for a point loop."""

    loop = _as_loop(points, "points")
    return (
        float(np.min(loop[:, 0])),
        float(np.max(loop[:, 0])),
        float(np.min(loop[:, 1])),
        float(np.max(loop[:, 1])),
    )


def points_in_polygon(points: ArrayLike, polygon: ArrayLike) -> NDArray[np.bool_]:
    """Return point-in-polygon mask using a deterministic ray crossing test."""

    query = np.asarray(points, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, 2)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    loop = _as_loop(polygon, "polygon")
    x = query[:, 0]
    y = query[:, 1]
    x0 = loop[:, 0]
    y0 = loop[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    inside = np.zeros(query.shape[0], dtype=bool)
    for start_x, start_y, end_x, end_y in zip(x0, y0, x1, y1, strict=True):
        crosses = (start_y > y) != (end_y > y)
        denominator = end_y - start_y
        safe_denominator = denominator if abs(denominator) > 0.0 else np.finfo(np.float64).eps
        x_intersect = (end_x - start_x) * (y - start_y) / safe_denominator + start_x
        inside ^= crosses & (x < x_intersect)
    return inside


def _as_loop(points: ArrayLike, name: str) -> NDArray[np.float64]:
    loop = np.asarray(points, dtype=np.float64)
    if loop.ndim != 2 or loop.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if loop.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three points")
    if not np.all(np.isfinite(loop)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if abs(polygon_area_no_validate(loop)) <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must enclose nonzero area")
    return loop


def polygon_area_no_validate(points: NDArray[np.float64]) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _circle_points(
    center_r: float, center_z: float, radius: float, angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.column_stack((center_r + radius * np.cos(angles), center_z + radius * np.sin(angles)))
