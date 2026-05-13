import json
from pathlib import Path

import numpy as np
import pytest

from tokamaker_jax.upstream_fixed_boundary import (
    FIXED_BOUNDARY_RELATIVE_ROOT,
    default_fixed_boundary_root,
    fixed_boundary_report_to_json,
    fixed_boundary_upstream_report,
    parse_geqdsk,
    summarize_fixed_boundary_notebook,
    write_fixed_boundary_upstream_report,
)


def test_parse_geqdsk_extracts_core_arrays(tmp_path: Path):
    path = tmp_path / "g_test"
    values = [
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
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
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
    ]
    path.write_text("TEST 0 3 2\n" + " ".join(f"{value:.9E}" for value in values), encoding="utf-8")

    parsed = parse_geqdsk(path)

    assert parsed["nr"] == 3
    assert parsed["nz"] == 2
    assert parsed["current"] == 120000.0
    np.testing.assert_allclose(parsed["r_grid"], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(parsed["z_grid"], [-0.5, 0.5])
    np.testing.assert_allclose(parsed["psi_grid"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_allclose(parsed["qpsi"], [3.0, 3.1, 3.2])


def test_fixed_boundary_report_handles_synthetic_notebook_and_geqdsk(tmp_path: Path):
    root = tmp_path / "OpenFUSIONToolkit"
    source_root = root / "src/examples/TokaMaker/fixed_boundary"
    source_root.mkdir(parents=True)
    for name in ("fixed_boundary_ex1.ipynb", "fixed_boundary_ex2.ipynb"):
        (source_root / name).write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": ["TokaMaker Example: Fixed boundary equilibria\n"],
                        },
                        {
                            "cell_type": "code",
                            "source": [
                                "from OpenFUSIONToolkit.TokaMaker import TokaMaker\n",
                                "mesh_dx = 0.15\n",
                                "Ip_target = 120.E3\n",
                                "mygs.settings.free_boundary = False\n",
                                "mygs.init_psi()\n",
                                "_ = mygs.solve()\n",
                                "ffp_prof = {}; pp_prof = {}\n",
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
    (source_root / "gNT_example").write_text(
        "TEST 0 2 2\n" + " ".join(f"{value:.9E}" for value in range(34)),
        encoding="utf-8",
    )

    report = fixed_boundary_upstream_report(root=root)
    notebook = report["notebooks"][0]

    assert report["artifact_id"] == "upstream-fixed-boundary-evidence"
    assert report["claim"] == "source_evidence_only"
    assert report["parity_level"] == "source_audit"
    assert report["numeric_parity_claim"] is False
    assert notebook["exists"] is True
    assert notebook["imports_tokamaker"] is True
    assert notebook["fixed_boundary_assignments"] == 1
    assert notebook["solve_calls"] == 1
    assert notebook["uses_profile_matching"] is True
    assert report["geqdsk"]["nr"] == 2
    assert report["geqdsk"]["nz"] == 2
    assert fixed_boundary_report_to_json(report).endswith("\n")


def test_write_fixed_boundary_upstream_report_roundtrips_missing_sources(tmp_path: Path):
    output = write_fixed_boundary_upstream_report(tmp_path / "evidence.json", root=tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["checkout_exists"] is True
    assert payload["geqdsk"] is None
    assert all(not notebook["exists"] for notebook in payload["notebooks"])


def test_summarize_fixed_boundary_notebook_missing(tmp_path: Path):
    summary = summarize_fixed_boundary_notebook(tmp_path / "missing.ipynb", root=tmp_path)

    assert summary == {"path": "missing.ipynb", "exists": False, "sha256": None}


def test_parse_geqdsk_rejects_invalid_sources(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty gEQDSK"):
        parse_geqdsk(empty)

    no_dims = tmp_path / "no_dims"
    no_dims.write_text("NO DIMS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not parse"):
        parse_geqdsk(no_dims)

    too_short = tmp_path / "too_short"
    too_short.write_text("TEST 0 3 2\n1.0 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least"):
        parse_geqdsk(too_short)


def test_summarize_notebook_covers_setup_title_and_external_paths(tmp_path: Path):
    external = tmp_path.parent / f"{tmp_path.name}_external.ipynb"
    external.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["---\n", "===\n"]},
                    {
                        "cell_type": "code",
                        "source": [
                            "mygs.setup(order=2, F0=my_F0)\n",
                            "mygs.plot_constraints()\n",
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        summary = summarize_fixed_boundary_notebook(external, root=tmp_path)
    finally:
        external.unlink(missing_ok=True)

    assert summary["path"] == str(external)
    assert summary["title"] is None
    assert summary["f0_expressions"] == ["my_F0"]
    assert summary["referenced_outputs"] == ["plot_constraints"]


def test_default_fixed_boundary_root_uses_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENFUSIONTOOLKIT_ROOT", str(tmp_path))

    assert default_fixed_boundary_root() == tmp_path / FIXED_BOUNDARY_RELATIVE_ROOT
