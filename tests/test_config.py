import pytest
from conftest import REPO_ROOT

from tokamaker_jax.config import GridConfig, config_from_dict, load_config, regions_from_dict


def test_load_example_config():
    config = load_config(REPO_ROOT / "examples" / "fixed_boundary.toml")

    assert config.grid.nr == 65
    assert config.solver.dtype == "float64"
    assert [coil.name for coil in config.coils] == ["PF_upper", "PF_lower"]


def test_config_from_dict_defaults():
    config = config_from_dict({"grid": {"nr": 9, "nz": 11}})

    assert config.grid == GridConfig(nr=9, nz=11)
    assert config.source.profile == "solovev"
    assert config.coils == ()
    assert config.regions is None
    assert config.to_dict()["grid"]["nr"] == 9


def test_config_rejects_non_table_section():
    try:
        config_from_dict({"grid": []})
    except TypeError as exc:
        assert "[grid]" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_config_from_dict_parses_region_shapes():
    config = config_from_dict(
        {
            "region": [
                {
                    "shape": "rectangle",
                    "id": 1,
                    "name": "PLASMA",
                    "kind": "plasma",
                    "r_min": 1.0,
                    "r_max": 3.0,
                    "z_min": -1.0,
                    "z_max": 1.0,
                    "target_size": 0.08,
                },
                {
                    "shape": "polygon",
                    "id": 2,
                    "name": "LIMITER",
                    "kind": "limiter",
                    "points": [[1.0, 0.0], [2.0, 0.5], [2.5, -0.5]],
                },
                {
                    "shape": "annulus",
                    "id": 3,
                    "name": "VV",
                    "kind": "conductor",
                    "center_r": 2.0,
                    "center_z": 0.0,
                    "inner_radius": 0.4,
                    "outer_radius": 0.7,
                    "n": 24,
                    "metadata": {"material": "steel"},
                },
            ]
        }
    )

    assert config.regions is not None
    assert [region.name for region in config.regions.regions] == ["PLASMA", "LIMITER", "VV"]
    assert config.regions.regions[0].bounds == (1.0, 3.0, -1.0, 1.0)
    assert config.regions.regions[1].kind == "limiter"
    assert len(config.regions.regions[2].holes) == 1
    assert config.to_dict()["regions"][2]["metadata"] == {"material": "steel"}


def test_regions_from_dict_accepts_region_dicts_without_shape():
    regions = regions_from_dict(
        {
            "regions": [
                {
                    "id": 1,
                    "name": "POLY",
                    "kind": "plasma",
                    "points": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                }
            ]
        }
    )

    assert regions is not None
    assert regions.regions[0].name == "POLY"


def test_load_config_parses_region_toml(tmp_path):
    path = tmp_path / "regions.toml"
    path.write_text(
        """
[[region]]
shape = "rectangle"
id = 1
name = "VACUUM"
kind = "vacuum"
r_min = 1.0
r_max = 2.0
z_min = -0.5
z_max = 0.5

[[region]]
shape = "annulus"
id = 2
name = "VESSEL"
kind = "conductor"
center_r = 1.5
center_z = 0.0
inner_radius = 0.2
outer_radius = 0.3
n = 16
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.regions is not None
    assert [region.name for region in config.regions.regions] == ["VACUUM", "VESSEL"]


def test_region_config_validation_errors():
    with pytest.raises(TypeError, match=r"\[\[region\]\]"):
        regions_from_dict({"region": {"id": 1}})
    with pytest.raises(ValueError, match="only one"):
        regions_from_dict({"region": [], "regions": []})
    with pytest.raises(ValueError, match="missing required key"):
        regions_from_dict({"region": [{"shape": "rectangle", "id": 1, "name": "bad"}]})
    with pytest.raises(ValueError, match="unsupported shape"):
        regions_from_dict({"region": [{"shape": "ellipse", "id": 1, "name": "bad"}]})
