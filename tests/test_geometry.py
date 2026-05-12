from pathlib import Path

import numpy as np
import pytest

from tokamaker_jax.geometry import (
    Region,
    RegionSet,
    annulus_region,
    bounds,
    ensure_counterclockwise,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_region,
    rectangle_region,
)
from tokamaker_jax.plotting import plot_regions, save_region_plot


def test_rectangle_region_serialization_and_contains_points():
    region = rectangle_region(
        id=2,
        name="VACUUM",
        kind="vacuum",
        r_min=1.0,
        r_max=3.0,
        z_min=-1.0,
        z_max=1.0,
        target_size=0.1,
        metadata={"allow_xpoints": True},
    )

    assert region.bounds == (1.0, 3.0, -1.0, 1.0)
    assert region.area == 4.0
    assert region.centroid == (2.0, 0.0)
    np.testing.assert_array_equal(region.contains_points([[2.0, 0.0], [4.0, 0.0]]), [True, False])

    loaded = Region.from_dict(region.to_dict())
    assert loaded.id == region.id
    assert loaded.name == region.name
    assert loaded.kind == region.kind
    assert loaded.target_size == region.target_size
    assert loaded.metadata == region.metadata
    np.testing.assert_allclose(loaded.points, region.points)


def test_polygon_helpers_and_orientation():
    clockwise = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    assert polygon_area(clockwise) < 0.0
    counterclockwise = ensure_counterclockwise(clockwise)
    assert polygon_area(counterclockwise) > 0.0
    np.testing.assert_allclose(polygon_centroid(counterclockwise), [0.5, 0.5])
    assert bounds(counterclockwise) == (0.0, 1.0, 0.0, 1.0)

    region = polygon_region(id=1, name="PLASMA", kind="plasma", points=clockwise)
    assert region.area == 1.0
    assert polygon_area(region.points) > 0.0


def test_annulus_region_area_and_hole_mask():
    region = annulus_region(
        id=4,
        name="VV",
        kind="conductor",
        center_r=2.0,
        center_z=0.0,
        inner_radius=0.5,
        outer_radius=1.0,
        n=128,
    )

    assert region.bounds[0] == pytest.approx(1.0)
    assert region.bounds[1] == pytest.approx(3.0)
    assert region.area == pytest.approx(np.pi * (1.0**2 - 0.5**2), rel=1.0e-3)
    np.testing.assert_array_equal(
        region.contains_points([[2.75, 0.0], [2.0, 0.0], [3.5, 0.0]]),
        [True, False, False],
    )


def test_region_set_and_plot(tmp_path: Path):
    regions = RegionSet(
        (
            rectangle_region(
                id=1,
                name="PLASMA",
                kind="plasma",
                r_min=1.2,
                r_max=2.8,
                z_min=-0.8,
                z_max=0.8,
            ),
            annulus_region(
                id=2,
                name="VV",
                kind="conductor",
                center_r=2.0,
                center_z=0.0,
                inner_radius=1.0,
                outer_radius=1.2,
                n=32,
            ),
        )
    )

    assert [region.name for region in regions.by_kind("conductor")] == ["VV"]
    assert RegionSet.from_dicts(regions.to_dicts()).regions[0].name == "PLASMA"

    path = save_region_plot(regions, tmp_path / "regions.png")
    assert path.exists()
    assert path.stat().st_size > 0

    with pytest.raises(ValueError, match="at least one"):
        plot_regions(())


def test_geometry_validation_errors():
    with pytest.raises(ValueError, match="one-based"):
        rectangle_region(id=0, name="bad", r_min=1.0, r_max=2.0, z_min=0.0, z_max=1.0)
    with pytest.raises(ValueError, match="nonempty"):
        rectangle_region(id=1, name="", r_min=1.0, r_max=2.0, z_min=0.0, z_max=1.0)
    with pytest.raises(ValueError, match="unsupported"):
        rectangle_region(
            id=1,
            name="bad",
            kind="not-a-kind",
            r_min=1.0,
            r_max=2.0,
            z_min=0.0,
            z_max=1.0,
        )
    with pytest.raises(ValueError, match="target_size"):
        rectangle_region(
            id=1,
            name="bad",
            r_min=1.0,
            r_max=2.0,
            z_min=0.0,
            z_max=1.0,
            target_size=0.0,
        )
    with pytest.raises(ValueError, match="r_max"):
        rectangle_region(id=1, name="bad", r_min=2.0, r_max=1.0, z_min=0.0, z_max=1.0)
    with pytest.raises(ValueError, match="z_max"):
        rectangle_region(id=1, name="bad", r_min=1.0, r_max=2.0, z_min=1.0, z_max=0.0)
    with pytest.raises(ValueError, match="at least three"):
        polygon_region(id=1, name="bad", points=[[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        polygon_region(id=1, name="bad", points=[[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="nonzero area"):
        polygon_region(id=1, name="bad", points=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="inner_radius"):
        annulus_region(
            id=1,
            name="bad",
            center_r=0.0,
            center_z=0.0,
            inner_radius=0.0,
            outer_radius=1.0,
        )
    with pytest.raises(ValueError, match="outer_radius"):
        annulus_region(
            id=1,
            name="bad",
            center_r=0.0,
            center_z=0.0,
            inner_radius=1.0,
            outer_radius=1.0,
        )
    with pytest.raises(ValueError, match="at least 12"):
        annulus_region(
            id=1,
            name="bad",
            center_r=0.0,
            center_z=0.0,
            inner_radius=0.5,
            outer_radius=1.0,
            n=8,
        )
    with pytest.raises(ValueError, match="shape"):
        points_in_polygon([1.0], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="at least one"):
        RegionSet(())
    with pytest.raises(ValueError, match="ids"):
        RegionSet(
            (
                rectangle_region(id=1, name="A", r_min=1.0, r_max=2.0, z_min=0.0, z_max=1.0),
                rectangle_region(id=1, name="B", r_min=2.0, r_max=3.0, z_min=0.0, z_max=1.0),
            )
        )
    with pytest.raises(ValueError, match="names"):
        RegionSet(
            (
                rectangle_region(id=1, name="A", r_min=1.0, r_max=2.0, z_min=0.0, z_max=1.0),
                rectangle_region(id=2, name="A", r_min=2.0, r_max=3.0, z_min=0.0, z_max=1.0),
            )
        )


def test_centroid_rejects_degenerate_polygon():
    with pytest.raises(ValueError, match="nonzero area"):
        polygon_centroid(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
