"""Build a bounded evidence report for upstream fixed-boundary examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_UPSTREAM_ROOT = Path("/Users/rogeriojorge/local/OpenFUSIONToolkit")
FIXED_BOUNDARY_DIR = Path("src/examples/TokaMaker/fixed_boundary")
DEFAULT_OUTPUT = Path("docs/validation/fixed_boundary_upstream_evidence.json")

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def build_fixed_boundary_evidence(
    upstream_root: str | Path = DEFAULT_UPSTREAM_ROOT,
) -> dict[str, Any]:
    """Return a JSON-ready bounded evidence report for fixed-boundary examples."""

    root = Path(upstream_root)
    example_dir = root / FIXED_BOUNDARY_DIR
    notebooks = [
        summarize_notebook(path, root=root)
        for path in (
            example_dir / "fixed_boundary_ex1.ipynb",
            example_dir / "fixed_boundary_ex2.ipynb",
        )
    ]
    gfile = summarize_gfile(example_dir / "gNT_example", root=root)
    return {
        "schema_version": 1,
        "artifact_id": "fixed-boundary-upstream-evidence",
        "claim": "source_audit_plus_stored_upstream_output",
        "parity_level": "source_audit",
        "numeric_parity_claim": False,
        "upstream_checkout": {
            "path": str(root),
            "exists": root.exists(),
            "commit": git_commit(root),
            "example_directory": str(FIXED_BOUNDARY_DIR),
            "example_directory_exists": example_dir.exists(),
        },
        "upstream_sources": {
            "notebooks": notebooks,
            "gfile": gfile,
        },
        "bounded_local_status": {
            "implemented_evidence": [
                "tokamaker-jax fixed_boundary.toml rectangular-grid seed solve",
                "manufactured axisymmetric Grad-Shafranov p=1 FEM convergence gate",
                "availability-gated OpenFUSIONToolkit eval_green kernel comparison",
            ],
            "not_claimed": [
                "no fixed-boundary equilibrium vector parity against these notebooks",
                "no EQDSK import/profile matching parity for gNT_example",
                "no order=2 triangular TokaMaker FEM solver parity",
                "no fixed-to-free-boundary workflow parity for fixed_boundary_ex2",
            ],
        },
        "evidence_matrix": fixed_boundary_evidence_matrix(notebooks, gfile),
        "limitations": [
            "Notebook metrics are read from committed/stored upstream notebook outputs.",
            "The report does not rerun OpenFUSIONToolkit or compare solved psi vectors.",
            "The current local fixed-boundary solver remains a seed path unless a separate "
            "numeric parity report says otherwise.",
        ],
    }


def summarize_notebook(path: Path, *, root: Path) -> dict[str, Any]:
    """Summarize static code markers and stored outputs from one notebook."""

    exists = path.exists()
    if not exists:
        return {
            "path": _relative(path, root),
            "exists": False,
            "sha256": None,
            "cell_count": 0,
            "code_cell_count": 0,
            "markdown_cell_count": 0,
            "workflow_markers": {},
            "mesh_outputs": [],
            "solve_traces": [],
            "equilibrium_statistics": [],
            "coil_currents_kA_turns": [],
        }

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_sources = [
        "".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"
    ]
    all_code = "\n".join(code_sources)
    mesh_outputs: list[dict[str, Any]] = []
    solve_traces: list[dict[str, Any]] = []
    equilibrium_statistics: list[dict[str, Any]] = []
    coil_currents: list[dict[str, Any]] = []

    for index, cell in enumerate(cells):
        text = cell_output_text(cell)
        mesh = parse_mesh_output(text)
        if mesh is not None:
            mesh_outputs.append({"cell_index": index, **mesh})
        solve_trace = parse_solve_trace(text)
        if solve_trace is not None:
            solve_traces.append({"cell_index": index, **solve_trace})
        for stats in parse_equilibrium_statistics(text):
            equilibrium_statistics.append({"cell_index": index, **stats})
        currents = parse_coil_currents(text)
        if currents:
            coil_currents.append({"cell_index": index, "values": currents})

    return {
        "path": _relative(path, root),
        "exists": True,
        "sha256": sha256_file(path),
        "nbformat": notebook.get("nbformat"),
        "nbformat_minor": notebook.get("nbformat_minor"),
        "cell_count": len(cells),
        "code_cell_count": sum(1 for cell in cells if cell.get("cell_type") == "code"),
        "markdown_cell_count": sum(1 for cell in cells if cell.get("cell_type") == "markdown"),
        "workflow_markers": notebook_workflow_markers(all_code),
        "mesh_outputs": mesh_outputs,
        "solve_traces": solve_traces,
        "equilibrium_statistics": equilibrium_statistics,
        "coil_currents_kA_turns": coil_currents,
    }


def summarize_gfile(path: Path, *, root: Path) -> dict[str, Any]:
    """Summarize the fixed-boundary gEQDSK seed used by upstream ex1."""

    if not path.exists():
        return {"path": _relative(path, root), "exists": False, "sha256": None}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return {
            "path": _relative(path, root),
            "exists": True,
            "sha256": sha256_file(path),
            "line_count": 0,
            "header": None,
        }
    header_parts = lines[0].split()
    nr = int(header_parts[-2])
    nz = int(header_parts[-1])
    case = " ".join(header_parts[:-3])
    tokens = FLOAT_RE.findall("\n".join(lines[1:]))
    cursor = 0

    def take(count: int) -> list[float]:
        nonlocal cursor
        values = [float(value) for value in tokens[cursor : cursor + count]]
        cursor += count
        return values

    header_values = take(20)
    arrays: dict[str, list[float]] = {
        "fpol": take(nr),
        "pres": take(nr),
        "ffprim": take(nr),
        "pprime": take(nr),
        "psirz": take(nr * nz),
        "qpsi": take(nr),
    }
    nbbbs = int(float(tokens[cursor]))
    cursor += 1
    limitr = int(float(tokens[cursor]))
    cursor += 1
    rzout = take(2 * nbbbs)
    rzlim = take(2 * limitr)
    return {
        "path": _relative(path, root),
        "exists": True,
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "header": {
            "case": case,
            "nr": nr,
            "nz": nz,
            "rdim_m": header_values[0],
            "zdim_m": header_values[1],
            "rcentr_m": header_values[2],
            "rleft_m": header_values[3],
            "zmid_m": header_values[4],
            "rmaxis_m": header_values[5],
            "zmaxis_m": header_values[6],
            "simag": header_values[7],
            "sibry": header_values[8],
            "bcentr_T": header_values[9],
            "current_A": header_values[10],
            "nbbbs": nbbbs,
            "limitr": limitr,
        },
        "array_ranges": {name: value_range(values) for name, values in arrays.items()},
        "boundary_bounds": rz_bounds(rzout),
        "limiter_bounds": rz_bounds(rzlim),
        "unconsumed_numeric_tokens": len(tokens) - cursor,
    }


def fixed_boundary_evidence_matrix(
    notebooks: list[dict[str, Any]],
    gfile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return bounded evidence rows for docs and tests."""

    by_name = {Path(item["path"]).name: item for item in notebooks}
    ex1 = by_name.get("fixed_boundary_ex1.ipynb", {})
    ex2 = by_name.get("fixed_boundary_ex2.ipynb", {})
    return [
        {
            "id": "analytic-lcfs-fixed-boundary",
            "upstream_source": "fixed_boundary_ex1.ipynb",
            "upstream_evidence": {
                "mesh_outputs": ex1.get("mesh_outputs", [])[:1],
                "equilibrium_statistics": ex1.get("equilibrium_statistics", [])[:1],
                "uses_fixed_boundary_false": ex1.get("workflow_markers", {}).get(
                    "uses_fixed_boundary_false", False
                ),
            },
            "local_status": "manufactured_validation_only",
            "parity_claim": "none",
        },
        {
            "id": "gnt-eqdsk-fixed-boundary",
            "upstream_source": "fixed_boundary_ex1.ipynb + gNT_example",
            "upstream_evidence": {
                "mesh_outputs": ex1.get("mesh_outputs", [])[1:],
                "equilibrium_statistics": ex1.get("equilibrium_statistics", [])[1:],
                "gfile_header": gfile.get("header"),
            },
            "local_status": "eqdsk_parity_not_implemented",
            "parity_claim": "none",
        },
        {
            "id": "fixed-to-free-boundary-bridge",
            "upstream_source": "fixed_boundary_ex2.ipynb",
            "upstream_evidence": {
                "mesh_outputs": ex2.get("mesh_outputs", []),
                "equilibrium_statistics": ex2.get("equilibrium_statistics", []),
                "coil_currents_kA_turns": ex2.get("coil_currents_kA_turns", []),
                "uses_get_vfixed": ex2.get("workflow_markers", {}).get("uses_get_vfixed", False),
                "uses_eval_green": ex2.get("workflow_markers", {}).get("uses_eval_green", False),
            },
            "local_status": "reduced_green_and_profile_coupling_only",
            "parity_claim": "none",
        },
    ]


def notebook_workflow_markers(code: str) -> dict[str, Any]:
    """Extract source-level workflow markers from notebook code cells."""

    return {
        "uses_gs_domain": "gs_Domain" in code,
        "uses_create_isoflux": "create_isoflux" in code,
        "uses_read_eqdsk": "read_eqdsk" in code,
        "uses_fixed_boundary_false": "settings.free_boundary = False" in code,
        "uses_fixed_boundary_true": "settings.free_boundary = True" in code,
        "uses_order_2_setup": bool(re.search(r"\.setup\(order=2\b", code)),
        "uses_set_targets": "set_targets" in code,
        "uses_set_profiles": "set_profiles" in code,
        "uses_get_profiles": "get_profiles" in code,
        "uses_get_q": "get_q" in code,
        "uses_get_vfixed": "get_vfixed" in code,
        "uses_eval_green": "eval_green" in code,
        "uses_lstsq_coil_fit": "np.linalg.lstsq" in code,
        "uses_set_coil_vsc": "set_coil_vsc" in code,
        "uses_set_coil_currents": "set_coil_currents" in code,
    }


def cell_output_text(cell: dict[str, Any]) -> str:
    """Return concatenated text/plain output from one notebook cell."""

    chunks: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            chunks.append("".join(output["text"]))
        data = output.get("data", {})
        if "text/plain" in data:
            chunks.append("".join(data["text/plain"]))
    return "\n".join(chunks)


def parse_mesh_output(text: str) -> dict[str, int] | None:
    """Parse Triangle mesh-count output from a notebook cell."""

    if "Generating mesh with Triangle" not in text:
        return None
    return {
        "unique_points": int(_required_match(r"# of unique points\s*=\s*(\d+)", text)),
        "unique_segments": int(_required_match(r"# of unique segments\s*=\s*(\d+)", text)),
        "points": int(_required_match(r"# of points\s*=\s*(\d+)", text)),
        "cells": int(_required_match(r"# of cells\s*=\s*(\d+)", text)),
        "regions": int(_required_match(r"# of regions\s*=\s*(\d+)", text)),
    }


def parse_solve_trace(text: str) -> dict[str, Any] | None:
    """Parse stored nonlinear solver trace rows from a notebook cell."""

    if "Starting non-linear GS solver" not in text:
        return None
    rows = []
    for line in text.splitlines():
        if re.match(r"^\s*\d+\s+", line):
            values = [float(value) for value in FLOAT_RE.findall(line)]
            if values:
                rows.append({"iteration": int(values[0]), "values": values[1:]})
    timing = _optional_float(r"Timing:\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)", text)
    return {
        "iteration_count": len(rows),
        "last_iteration": None if not rows else rows[-1],
        "timing_s": timing,
    }


def parse_equilibrium_statistics(text: str) -> list[dict[str, Any]]:
    """Parse TokaMaker print_info blocks stored in notebook outputs."""

    blocks = text.split("Equilibrium Statistics:")
    stats: list[dict[str, Any]] = []
    for block in blocks[1:]:
        stats.append(
            {
                "topology": _optional_string(r"Topology\s*=\s*([A-Za-z]+)", block),
                "toroidal_current_A": _optional_float(r"Toroidal Current \[A\]\s*=\s*(\S+)", block),
                "current_centroid_m": _optional_float_pair(
                    r"Current Centroid \[m\]\s*=\s*(\S+)\s+(\S+)", block
                ),
                "magnetic_axis_m": _optional_float_pair(
                    r"Magnetic Axis \[m\]\s*=\s*(\S+)\s+(\S+)", block
                ),
                "elongation": _optional_float(r"Elongation\s*=\s*(\S+)", block),
                "triangularity": _optional_float(r"Triangularity\s*=\s*(\S+)", block),
                "plasma_volume_m3": _optional_float(r"Plasma Volume \[m\^3\]\s*=\s*(\S+)", block),
                "q0_q95": _optional_float_pair(r"q_0, q_95\s*=\s*(\S+)\s+(\S+)", block),
                "pressure_axis_peak_Pa": _optional_float_pair(
                    r"Plasma Pressure \[Pa\]\s*=\s*Axis:\s*(\S+),\s*Peak:\s*(\S+)",
                    block,
                ),
                "stored_energy_J": _optional_float(r"Stored Energy \[J\]\s*=\s*(\S+)", block),
                "beta_pol_percent": _optional_float(r"<Beta_pol> \[%\]\s*=\s*(\S+)", block),
                "beta_tor_percent": _optional_float(r"<Beta_tor> \[%\]\s*=\s*(\S+)", block),
                "beta_n_percent": _optional_float(r"<Beta_n>\s+\[%\]\s*=\s*(\S+)", block),
                "diamagnetic_flux_Wb": _optional_float(
                    r"Diamagnetic flux \[Wb\]\s*=\s*(\S+)", block
                ),
                "toroidal_flux_Wb": _optional_float(r"Toroidal flux \[Wb\]\s*=\s*(\S+)", block),
                "li": _optional_float(r"l_i\s*=\s*(\S+)", block),
            }
        )
    return stats


def parse_coil_currents(text: str) -> list[float]:
    """Parse stored least-squares coil currents from fixed_boundary_ex2."""

    marker = "Coil currents [kA-turns]:"
    if marker not in text:
        return []
    block = text.split(marker, 1)[1].split("<Figure", 1)[0]
    block = block.split("Equilibrium Statistics:", 1)[0]
    return [float(value) for value in FLOAT_RE.findall(block)]


def value_range(values: list[float]) -> dict[str, float]:
    """Return min/max for a numeric array."""

    return {"min": min(values), "max": max(values)}


def rz_bounds(flat_rz: list[float]) -> dict[str, float]:
    """Return R/Z bounds from a flat R, Z, R, Z coordinate list."""

    r_values = flat_rz[0::2]
    z_values = flat_rz[1::2]
    return {
        "r_min": min(r_values),
        "r_max": max(r_values),
        "z_min": min(z_values),
        "z_max": max(z_values),
    }


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a file."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit(path: Path) -> str | None:
    """Return the git commit for a checkout when available."""

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_evidence(path: str | Path, *, upstream_root: str | Path = DEFAULT_UPSTREAM_ROOT) -> Path:
    """Write the fixed-boundary evidence JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_fixed_boundary_evidence(upstream_root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output.resolve()


def _required_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"required pattern not found: {pattern}")
    return match.group(1)


def _optional_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return None if match is None else float(match.group(1))


def _optional_string(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return None if match is None else match.group(1)


def _optional_float_pair(pattern: str, text: str) -> list[float] | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return [float(match.group(1)), float(match.group(2))]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        default=str(DEFAULT_UPSTREAM_ROOT),
        help="OpenFUSIONToolkit checkout root.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help="Evidence JSON output path.",
    )
    args = parser.parse_args(argv)
    output = write_evidence(args.output, upstream_root=args.upstream_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
