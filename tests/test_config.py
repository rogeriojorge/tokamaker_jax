from conftest import REPO_ROOT

from tokamaker_jax.config import GridConfig, config_from_dict, load_config


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
    assert config.to_dict()["grid"]["nr"] == 9


def test_config_rejects_non_table_section():
    try:
        config_from_dict({"grid": []})
    except TypeError as exc:
        assert "[grid]" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
