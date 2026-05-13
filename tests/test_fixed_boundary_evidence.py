import importlib.util
import json
from pathlib import Path

from conftest import REPO_ROOT


def _load_evidence_module():
    module_path = REPO_ROOT / "docs" / "validation" / "build_fixed_boundary_evidence.py"
    spec = importlib.util.spec_from_file_location("build_fixed_boundary_evidence", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_fixed_boundary_evidence_records_bounded_claims():
    report_path = REPO_ROOT / "docs" / "validation" / "fixed_boundary_upstream_evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["artifact_id"] == "fixed-boundary-upstream-evidence"
    assert report["claim"] == "source_audit_plus_stored_upstream_output"
    assert report["parity_level"] == "source_audit"
    assert report["numeric_parity_claim"] is False
    assert all(row["parity_claim"] == "none" for row in report["evidence_matrix"])
    assert (
        "no fixed-boundary equilibrium vector parity against these notebooks"
        in report["bounded_local_status"]["not_claimed"]
    )


def test_committed_fixed_boundary_evidence_records_upstream_metrics():
    report_path = REPO_ROOT / "docs" / "validation" / "fixed_boundary_upstream_evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    notebooks = {Path(item["path"]).name: item for item in report["upstream_sources"]["notebooks"]}
    ex1 = notebooks["fixed_boundary_ex1.ipynb"]
    ex2 = notebooks["fixed_boundary_ex2.ipynb"]
    gfile = report["upstream_sources"]["gfile"]

    assert ex1["sha256"] == "a7e53f5a0e56ecd3a0a7ccb84f300a2ab517b21b09ff373c220ed931fcfd481f"
    assert ex2["sha256"] == "47825b5182f3dc51fb6c7f2395a0d7b475ee6b8eb49167872565e40ff5bc5521"
    assert gfile["sha256"] == "8e586dbbc296a66af9a493e7c2696a4408a43845dbcd2f7f38da88625fbf48ef"
    assert [mesh["points"] for mesh in ex1["mesh_outputs"]] == [700, 488]
    assert [mesh["cells"] for mesh in ex1["mesh_outputs"]] == [1322, 907]
    assert ex2["mesh_outputs"][1]["points"] == 3918
    assert ex2["mesh_outputs"][1]["cells"] == 7736
    assert ex1["equilibrium_statistics"][0]["toroidal_current_A"] == 119990.0
    assert ex1["equilibrium_statistics"][1]["toroidal_current_A"] == 7799300.0
    assert ex1["equilibrium_statistics"][2]["magnetic_axis_m"] == [3.522, -0.0]
    assert ex2["coil_currents_kA_turns"][0]["values"] == [
        30.4008,
        -22.2004,
        22.3988,
        -68.3032,
        86.3585,
        -35.2368,
        180.2722,
    ]
    assert gfile["header"]["nr"] == 129
    assert gfile["header"]["nz"] == 129
    assert gfile["header"]["current_A"] == 7799300.71
    assert gfile["header"]["nbbbs"] == 99
    assert gfile["array_ranges"]["qpsi"]["max"] == 7.31450488


def test_fixed_boundary_evidence_generator_parses_synthetic_upstream(tmp_path: Path):
    module = _load_evidence_module()
    fixed_dir = tmp_path / "src" / "examples" / "TokaMaker" / "fixed_boundary"
    fixed_dir.mkdir(parents=True)
    _write_notebook(
        fixed_dir / "fixed_boundary_ex1.ipynb",
        code=(
            "gs_mesh = gs_Domain()\n"
            "LCFS_contour = create_isoflux(8, 1.0, 0.0, 0.2, 1.1, 0.0)\n"
            "mygs.settings.free_boundary = False\n"
            "mygs.setup(order=2, F0=1.0)\n"
            "mygs.set_targets(Ip=1.0)\n"
            "mygs.set_profiles(ffp_prof={'type': 'flat'}, pp_prof={'type': 'flat'})\n"
            "psi, f, fp, p, pp = mygs.get_profiles()\n"
            "psi_q, qvals, *_ = mygs.get_q()\n"
            "EQ_in = read_eqdsk('gNT_example')\n"
        ),
        output=_mesh_output(points=5, cells=6)
        + "\n"
        + _solve_output()
        + "\n"
        + _stats_output(current=12.0, axis_r=1.25),
    )
    _write_notebook(
        fixed_dir / "fixed_boundary_ex2.ipynb",
        code=(
            "gs_mesh = gs_Domain()\n"
            "LCFS_contour = create_isoflux(8, 1.0, 0.0, 0.2, 1.1, 0.0)\n"
            "mygs.settings.free_boundary = False\n"
            "mygs.settings.free_boundary = True\n"
            "mygs.setup(order=2, F0=1.0)\n"
            "r_bnd, psi_bnd = mygs.get_vfixed()\n"
            "con[:, i] = eval_green(r_bnd, coil).sum(axis=0)\n"
            "currs = np.linalg.lstsq(con, rhs, rcond=None)[0]\n"
            "mygs.set_coil_vsc({'PF': 1.0})\n"
            "mygs.set_coil_currents({'PF': 3.0})\n"
            "mygs.set_targets(Ip=1.0)\n"
        ),
        output=_mesh_output(points=7, cells=8)
        + "\nCoil currents [kA-turns]:\n 1.0\n -2.5\n"
        + _stats_output(current=13.0, axis_r=1.35),
    )
    _write_tiny_gfile(fixed_dir / "gNT_example")

    report = module.build_fixed_boundary_evidence(tmp_path)
    notebooks = {Path(item["path"]).name: item for item in report["upstream_sources"]["notebooks"]}

    assert report["numeric_parity_claim"] is False
    assert notebooks["fixed_boundary_ex1.ipynb"]["mesh_outputs"][0]["points"] == 5
    assert notebooks["fixed_boundary_ex1.ipynb"]["solve_traces"][0]["iteration_count"] == 2
    assert notebooks["fixed_boundary_ex1.ipynb"]["workflow_markers"]["uses_read_eqdsk"] is True
    assert notebooks["fixed_boundary_ex2.ipynb"]["workflow_markers"]["uses_get_vfixed"] is True
    assert notebooks["fixed_boundary_ex2.ipynb"]["coil_currents_kA_turns"][0]["values"] == [
        1.0,
        -2.5,
    ]
    assert report["upstream_sources"]["gfile"]["header"]["nr"] == 2
    assert report["upstream_sources"]["gfile"]["header"]["nbbbs"] == 3
    assert all(row["parity_claim"] == "none" for row in report["evidence_matrix"])


def _write_notebook(path: Path, *, code: str, output: str) -> None:
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 4,
                "cells": [
                    {"cell_type": "markdown", "source": ["title"], "metadata": {}},
                    {
                        "cell_type": "code",
                        "source": code.splitlines(keepends=True),
                        "metadata": {},
                        "outputs": [{"name": "stdout", "output_type": "stream", "text": output}],
                    },
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )


def _mesh_output(*, points: int, cells: int) -> str:
    return f"""Assembling regions:
  # of unique points    = 3
  # of unique segments  = 1
Generating mesh with Triangle:
  # of points  = {points}
  # of cells   = {cells}
  # of regions = 1
"""


def _solve_output() -> str:
    return """Starting non-linear GS solver
     1  1.0E+00  2.0E+00
     2  5.0E-01  1.0E+00
 Timing:   1.0E-003
"""


def _stats_output(*, current: float, axis_r: float) -> str:
    return f"""Equilibrium Statistics:
  Topology                =   Limited
  Toroidal Current [A]    =    {current:.4E}
  Current Centroid [m]    =    1.000  0.000
  Magnetic Axis [m]       =    {axis_r:.3f}  0.000
  Elongation              =    1.200 (U:  1.200, L:  1.200)
  Triangularity           =    0.100 (U:  0.100, L:  0.100)
  Plasma Volume [m^3]     =    0.250
  q_0, q_95               =    1.000  2.000
  Plasma Pressure [Pa]    =   Axis:  1.0000E+03, Peak:  1.0000E+03
  Stored Energy [J]       =    1.0000E+02
  <Beta_pol> [%]          =   10.0000
  <Beta_tor> [%]          =   20.0000
  <Beta_n>   [%]          =    1.0000
  Diamagnetic flux [Wb]   =    1.0000E-03
  Toroidal flux [Wb]      =    2.0000E-03
  l_i                     =    0.9000
"""


def _write_tiny_gfile(path: Path) -> None:
    header = " TEST CASE 0 2 2\n"
    header_values = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        1.25,
        0.0,
        -0.1,
        0.2,
        6.0,
        12.0,
        -0.1,
        0.0,
        1.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    arrays = [
        1.0,
        2.0,
        10.0,
        20.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        0.1,
        0.2,
        0.3,
        0.4,
        1.1,
        1.2,
    ]
    contour_counts = [3.0, 1.0]
    rzout = [1.0, 0.0, 1.1, 0.1, 1.2, 0.0]
    rzlim = [0.9, 0.0]
    trailing = [0.0]
    values = header_values + arrays + contour_counts + rzout + rzlim + trailing
    lines = [
        " ".join(f"{value:.9E}" for value in values[i : i + 5]) for i in range(0, len(values), 5)
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
