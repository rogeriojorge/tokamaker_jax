"""Evidence extraction for upstream TokaMaker fixed-boundary examples."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from tokamaker_jax.upstream_fixtures import DEFAULT_OPENFUSIONTOOLKIT_ROOT

FIXED_BOUNDARY_RELATIVE_ROOT = Path("src/examples/TokaMaker/fixed_boundary")
FIXED_BOUNDARY_NOTEBOOKS = ("fixed_boundary_ex1.ipynb", "fixed_boundary_ex2.ipynb")
FIXED_BOUNDARY_EQDSK = "gNT_example"


def fixed_boundary_upstream_report(
    *,
    root: str | Path = DEFAULT_OPENFUSIONTOOLKIT_ROOT,
) -> dict[str, Any]:
    """Return a JSON-ready source-evidence report for upstream fixed-boundary cases."""

    root_path = Path(root)
    source_root = root_path / FIXED_BOUNDARY_RELATIVE_ROOT
    notebooks = [
        summarize_fixed_boundary_notebook(source_root / notebook_name, root=root_path)
        for notebook_name in FIXED_BOUNDARY_NOTEBOOKS
    ]
    geqdsk_path = source_root / FIXED_BOUNDARY_EQDSK
    geqdsk = summarize_geqdsk(geqdsk_path, root=root_path) if geqdsk_path.exists() else None
    return {
        "schema_version": 1,
        "artifact_id": "upstream-fixed-boundary-evidence",
        "checkout_path": str(root_path),
        "checkout_exists": root_path.exists(),
        "source_root": str(FIXED_BOUNDARY_RELATIVE_ROOT),
        "claim": "source_evidence_only",
        "parity_level": "source_audit",
        "numeric_parity_claim": False,
        "notebooks": notebooks,
        "geqdsk": geqdsk,
        "acceptance_gates": [
            "Reproduce fixed-boundary mesh and boundary geometry.",
            "Match solved flux diagnostics against upstream fixed_boundary_ex1 within documented tolerances.",
            "Match gEQDSK-profile case pressure, FF', q, Ip, magnetic-axis, and boundary diagnostics.",
        ],
    }


def summarize_fixed_boundary_notebook(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Summarize source-level evidence from one upstream fixed-boundary notebook."""

    notebook_path = Path(path)
    if not notebook_path.exists():
        return {
            "path": _relative_path(notebook_path, root),
            "exists": False,
            "sha256": None,
        }
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    markdown = "\n".join("".join(cell.get("source", [])) for cell in markdown_cells)
    return {
        "path": _relative_path(notebook_path, root),
        "exists": True,
        "sha256": _sha256_file(notebook_path),
        "n_cells": len(cells),
        "n_code_cells": len(code_cells),
        "n_markdown_cells": len(markdown_cells),
        "title": _first_markdown_title(markdown),
        "imports_tokamaker": "from OpenFUSIONToolkit.TokaMaker import TokaMaker" in code,
        "uses_gs_domain": "gs_Domain" in code,
        "uses_geqdsk": "read_eqdsk" in code or FIXED_BOUNDARY_EQDSK in code,
        "uses_profile_matching": "ffp_prof" in code and "pp_prof" in code,
        "fixed_boundary_assignments": code.count("settings.free_boundary = False"),
        "free_boundary_assignments": code.count("settings.free_boundary = True"),
        "solve_calls": len(re.findall(r"\.solve\(", code)),
        "init_psi_calls": len(re.findall(r"\.init_psi\(", code)),
        "mesh_dx_values": _assignment_values(code, "mesh_dx"),
        "ip_target_values": _assignment_values(code, "Ip_target"),
        "f0_expressions": _setup_argument_values(code, "F0"),
        "referenced_outputs": _referenced_plot_methods(code),
    }


def summarize_geqdsk(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Summarize an upstream gEQDSK file and its source flux grid."""

    parsed = parse_geqdsk(path)
    psi = parsed["psi_grid"]
    return {
        "path": _relative_path(Path(path), root),
        "exists": True,
        "sha256": _sha256_file(Path(path)),
        "nr": parsed["nr"],
        "nz": parsed["nz"],
        "rdim": parsed["rdim"],
        "zdim": parsed["zdim"],
        "rcentr": parsed["rcentr"],
        "rleft": parsed["rleft"],
        "zmid": parsed["zmid"],
        "rmaxis": parsed["rmaxis"],
        "zmaxis": parsed["zmaxis"],
        "simag": parsed["simag"],
        "sibry": parsed["sibry"],
        "bcentr": parsed["bcentr"],
        "current": parsed["current"],
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
        "psi_shape": list(psi.shape),
        "qpsi_min": float(np.min(parsed["qpsi"])),
        "qpsi_max": float(np.max(parsed["qpsi"])),
        "profile_length": int(parsed["fpol"].shape[0]),
        "numeric_value_count": parsed["numeric_value_count"],
    }


def parse_geqdsk(path: str | Path) -> dict[str, Any]:
    """Parse the core arrays from a standard gEQDSK file."""

    geqdsk_path = Path(path)
    lines = geqdsk_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty gEQDSK file: {geqdsk_path}")
    dims = [int(value) for value in re.findall(r"[-+]?\d+", lines[0])]
    if len(dims) < 2:
        raise ValueError(f"could not parse gEQDSK dimensions from first line: {geqdsk_path}")
    nr, nz = dims[-2], dims[-1]
    values = np.asarray(_floating_values("\n".join(lines[1:])), dtype=np.float64)
    minimum = 20 + 5 * nr + nr * nz
    if values.size < minimum:
        raise ValueError(
            f"gEQDSK file has {values.size} numeric values after header, expected at least {minimum}"
        )

    rdim, zdim, rcentr, rleft, zmid = values[0:5]
    rmaxis, zmaxis, simag, sibry, bcentr = values[5:10]
    current = values[10]
    offset = 20
    fpol = values[offset : offset + nr]
    offset += nr
    pres = values[offset : offset + nr]
    offset += nr
    ffprim = values[offset : offset + nr]
    offset += nr
    pprime = values[offset : offset + nr]
    offset += nr
    psi_grid = values[offset : offset + nr * nz].reshape((nz, nr))
    offset += nr * nz
    qpsi = values[offset : offset + nr]
    r_grid = np.linspace(rleft, rleft + rdim, nr)
    z_grid = np.linspace(zmid - 0.5 * zdim, zmid + 0.5 * zdim, nz)
    return {
        "nr": int(nr),
        "nz": int(nz),
        "rdim": float(rdim),
        "zdim": float(zdim),
        "rcentr": float(rcentr),
        "rleft": float(rleft),
        "zmid": float(zmid),
        "rmaxis": float(rmaxis),
        "zmaxis": float(zmaxis),
        "simag": float(simag),
        "sibry": float(sibry),
        "bcentr": float(bcentr),
        "current": float(current),
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprime": pprime,
        "psi_grid": psi_grid,
        "qpsi": qpsi,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "numeric_value_count": int(values.size),
    }


def fixed_boundary_report_to_json(report: dict[str, Any]) -> str:
    """Serialize a fixed-boundary evidence report with stable formatting."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def write_fixed_boundary_upstream_report(
    path: str | Path,
    *,
    root: str | Path = DEFAULT_OPENFUSIONTOOLKIT_ROOT,
) -> Path:
    """Write the fixed-boundary source-evidence JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        fixed_boundary_report_to_json(fixed_boundary_upstream_report(root=root)),
        encoding="utf-8",
    )
    return output_path.resolve()


def _floating_values(text: str) -> list[float]:
    return [
        float(value.replace("D", "E").replace("d", "E"))
        for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", text)
    ]


def _assignment_values(code: str, name: str) -> list[str]:
    values = set()
    for match in re.finditer(rf"^{re.escape(name)}\s*=\s*([^\n#]+)", code, flags=re.MULTILINE):
        values.add(match.group(1).strip())
    return sorted(values)


def _setup_argument_values(code: str, name: str) -> list[str]:
    values = set()
    pattern = rf"\.setup\([^\)]*{re.escape(name)}\s*=\s*([^\),]+)"
    for match in re.finditer(pattern, code):
        values.add(match.group(1).strip())
    return sorted(values)


def _referenced_plot_methods(code: str) -> list[str]:
    methods = set(re.findall(r"\.(plot_[A-Za-z0-9_]+)\(", code))
    return sorted(methods)


def _first_markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and not set(stripped) <= {"=", "-", "#"}:
            return stripped.lstrip("#").strip()
    return None


def _relative_path(path: Path, root: str | Path) -> str:
    try:
        return str(path.relative_to(Path(root)))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def default_fixed_boundary_root() -> Path:
    """Return the configured fixed-boundary source directory."""

    return (
        Path(os.environ.get("OPENFUSIONTOOLKIT_ROOT", str(DEFAULT_OPENFUSIONTOOLKIT_ROOT)))
        / FIXED_BOUNDARY_RELATIVE_ROOT
    )
