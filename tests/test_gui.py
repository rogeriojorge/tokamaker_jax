import pytest

import tokamaker_jax.geometry as geometry
from tokamaker_jax.geometry import RegionSet, annulus_region, rectangle_region
from tokamaker_jax.gui import region_geometry_figure

pytest.importorskip("plotly")


def test_region_geometry_figure_uses_sample_regions():
    fig = region_geometry_figure()

    trace_names = {trace.name for trace in fig.data}
    assert "PLASMA (plasma)" in trace_names
    assert "VV (conductor)" in trace_names
    assert "PF (coil)" in trace_names
    assert fig.layout.xaxis.title.text == "R [m]"
    assert fig.layout.yaxis.title.text == "Z [m]"
    assert fig.layout.yaxis.scaleanchor == "x"


def test_region_geometry_figure_prefers_geometry_sample_regions(monkeypatch):
    def sample_regions():
        return RegionSet(
            (
                rectangle_region(
                    id=8,
                    name="CUSTOM",
                    kind="boundary",
                    r_min=1.0,
                    r_max=2.0,
                    z_min=-0.5,
                    z_max=0.5,
                ),
            )
        )

    monkeypatch.setattr(geometry, "sample_regions", sample_regions, raising=False)

    fig = region_geometry_figure()

    assert [trace.name for trace in fig.data] == ["CUSTOM (boundary)"]


def test_region_geometry_figure_accepts_regions_and_holes():
    regions = RegionSet(
        (
            annulus_region(
                id=1,
                name="WALL",
                kind="conductor",
                center_r=2.0,
                center_z=0.0,
                inner_radius=0.5,
                outer_radius=1.0,
                n=24,
            ),
            rectangle_region(
                id=2,
                name="PLASMA",
                kind="plasma",
                r_min=1.5,
                r_max=2.5,
                z_min=-0.4,
                z_max=0.4,
            ),
        )
    )

    fig = region_geometry_figure(regions, show_labels=False)

    assert [trace.name for trace in fig.data] == [
        "WALL (conductor)",
        "WALL hole 1",
        "PLASMA (plasma)",
    ]
    assert not fig.layout.annotations
    assert fig.data[0].fill == "toself"
    assert fig.data[1].showlegend is False


def test_region_geometry_figure_rejects_empty_regions():
    with pytest.raises(ValueError, match="at least one"):
        region_geometry_figure(())
