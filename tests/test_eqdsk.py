import json
from pathlib import Path

import numpy as np
import pytest

from tokamaker_jax.eqdsk import Eqdsk, diagnose_eqdsk, parse_eqdsk


def test_parse_eqdsk_extracts_core_fields_profiles_and_grid_coordinates(tmp_path: Path):
    path = tmp_path / "g_core"
    _write_eqdsk(
        path,
        nr=3,
        nz=2,
        values=[
            2.0,
            1.0,
            1.5,
            0.5,
            0.0,
            1.1,
            0.1,
            -0.2,
            0.8,
            2.5,
            120000.0,
            -0.2,
            0.0,
            1.1,
            0.0,
            0.1,
            0.0,
            0.8,
            0.0,
            0.0,
            1.0,
            1.1,
            1.2,
            10.0,
            11.0,
            12.0,
            -1.0,
            -1.1,
            -1.2,
            2.0,
            2.1,
            2.2,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            3.0,
            3.1,
            3.2,
        ],
    )

    parsed = parse_eqdsk(path)

    assert isinstance(parsed, Eqdsk)
    assert parsed.header == "TEST 0 3 2"
    assert parsed.nr == 3
    assert parsed.nz == 2
    assert parsed.current == 120000.0
    np.testing.assert_allclose(parsed.fpol, [1.0, 1.1, 1.2])
    np.testing.assert_allclose(parsed.pres, [10.0, 11.0, 12.0])
    np.testing.assert_allclose(parsed.ffprim, [-1.0, -1.1, -1.2])
    np.testing.assert_allclose(parsed.pprime, [2.0, 2.1, 2.2])
    np.testing.assert_allclose(parsed.psi_grid, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_allclose(parsed.qpsi, [3.0, 3.1, 3.2])
    np.testing.assert_allclose(parsed.r_grid, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(parsed.z_grid, [-0.5, 0.5])


def test_parse_eqdsk_accepts_d_exponent_floats(tmp_path: Path):
    path = tmp_path / "g_d_exp"
    values = _minimal_values(nr=2, nz=2)
    path.write_text(
        "D_EXP 0 2 2\n" + " ".join(f"{value:.9E}".replace("E", "D") for value in values),
        encoding="utf-8",
    )

    parsed = parse_eqdsk(path)

    assert parsed.nr == 2
    assert parsed.nz == 2
    assert parsed.rdim == 2.0
    np.testing.assert_allclose(parsed.qpsi, [9.0, 10.0])


def test_parse_eqdsk_extracts_boundary_and_limiter_coordinates(tmp_path: Path):
    path = tmp_path / "g_with_coords"
    values = [
        *_minimal_values(nr=2, nz=2),
        3,
        2,
        1.0,
        -0.5,
        1.5,
        0.0,
        1.0,
        0.5,
        0.6,
        -0.8,
        2.2,
        0.8,
    ]
    _write_eqdsk(path, nr=2, nz=2, values=values)

    parsed = parse_eqdsk(path)
    diagnostics = parsed.diagnostics()

    assert parsed.nbbbs == 3
    assert parsed.limitr == 2
    np.testing.assert_allclose(parsed.boundary, [[1.0, -0.5], [1.5, 0.0], [1.0, 0.5]])
    np.testing.assert_allclose(parsed.limiter, [[0.6, -0.8], [2.2, 0.8]])
    assert diagnostics["boundary_shape"] == [3, 2]
    assert diagnostics["limiter_shape"] == [2, 2]


def test_diagnose_eqdsk_reports_json_friendly_valid_payload(tmp_path: Path):
    path = tmp_path / "g_json"
    _write_eqdsk(path, nr=2, nz=2, values=_minimal_values(nr=2, nz=2))

    diagnostics = diagnose_eqdsk(path)

    assert diagnostics["valid"] is True
    assert diagnostics["nr"] == 2
    assert diagnostics["nz"] == 2
    assert diagnostics["psi_shape"] == [2, 2]
    assert json.loads(json.dumps(diagnostics)) == diagnostics


def test_to_json_dict_converts_arrays_to_lists(tmp_path: Path):
    path = tmp_path / "g_record"
    _write_eqdsk(path, nr=2, nz=2, values=_minimal_values(nr=2, nz=2))

    payload = parse_eqdsk(path).to_json_dict()

    assert payload["psi_grid"] == [[5.0, 6.0], [7.0, 8.0]]
    assert payload["qpsi"] == [9.0, 10.0]
    json.dumps(payload)


def test_parse_eqdsk_rejects_missing_and_invalid_files(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parse_eqdsk(missing)
    missing_diagnostics = diagnose_eqdsk(missing)
    assert missing_diagnostics["valid"] is False
    assert missing_diagnostics["exists"] is False
    assert missing_diagnostics["error_type"] == "FileNotFoundError"

    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty EQDSK"):
        parse_eqdsk(empty)

    no_dims = tmp_path / "no_dims"
    no_dims.write_text("NO DIMS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not parse"):
        parse_eqdsk(no_dims)

    too_short = tmp_path / "too_short"
    too_short.write_text("TEST 0 3 2\n1.0 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least"):
        parse_eqdsk(too_short)
    too_short_diagnostics = diagnose_eqdsk(too_short)
    assert too_short_diagnostics["valid"] is False
    assert too_short_diagnostics["exists"] is True
    assert too_short_diagnostics["error_type"] == "ValueError"


def test_parse_eqdsk_rejects_malformed_boundary_limiter_records(tmp_path: Path):
    incomplete_counts = tmp_path / "incomplete_counts"
    _write_eqdsk(
        incomplete_counts,
        nr=2,
        nz=2,
        values=[*_minimal_values(nr=2, nz=2), 3],
    )
    with pytest.raises(ValueError, match="incomplete boundary/limiter counts"):
        parse_eqdsk(incomplete_counts)

    incomplete_coordinates = tmp_path / "incomplete_coordinates"
    _write_eqdsk(
        incomplete_coordinates,
        nr=2,
        nz=2,
        values=[*_minimal_values(nr=2, nz=2), 1, 1, 1.0, 0.0],
    )
    with pytest.raises(ValueError, match="incomplete boundary/limiter coordinates"):
        parse_eqdsk(incomplete_coordinates)


def _write_eqdsk(path: Path, *, nr: int, nz: int, values: list[float]) -> None:
    path.write_text(
        f"TEST 0 {nr} {nz}\n" + " ".join(f"{value:.9E}" for value in values),
        encoding="utf-8",
    )


def _minimal_values(*, nr: int, nz: int) -> list[float]:
    assert nr == 2
    assert nz == 2
    return [
        2.0,
        1.0,
        1.5,
        0.5,
        0.0,
        1.1,
        0.1,
        -0.2,
        0.8,
        2.5,
        120000.0,
        -0.2,
        0.0,
        1.1,
        0.0,
        0.1,
        0.0,
        0.8,
        0.0,
        0.0,
        1.0,
        1.1,
        10.0,
        11.0,
        -1.0,
        -1.1,
        2.0,
        2.1,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
