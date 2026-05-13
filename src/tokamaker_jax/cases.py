"""Case manifest for examples, validation gates, and upstream parity targets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaseManifestEntry:
    """One runnable or planned tokamaker-jax case."""

    case_id: str
    title: str
    status: str
    category: str
    description: str
    parity_level: str
    path: str | None = None
    command: str | None = None
    validation_gate: str | None = None
    upstream_sources: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        """Whether the entry has a command intended to run today."""

        return self.status in {"runnable", "validation_gate"} and self.command is not None

    def to_dict(self, *, root: str | Path | None = None) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "title": self.title,
            "status": self.status,
            "category": self.category,
            "description": self.description,
            "parity_level": self.parity_level,
            "path": self.path,
            "command": self.command,
            "validation_gate": self.validation_gate,
            "upstream_sources": list(self.upstream_sources),
            "outputs": list(self.outputs),
            "citations": list(self.citations),
            "notes": list(self.notes),
            "runnable": self.runnable,
        }
        if root is not None:
            payload["path_exists"] = _case_path_exists(root, self.path)
        return payload


@dataclass(frozen=True)
class CaseManifest:
    """Collection of example and parity-target cases."""

    entries: tuple[CaseManifestEntry, ...]
    root: Path = PROJECT_ROOT
    schema_version: int = 1
    artifact_id: str = "tokamaker-jax-case-manifest"

    def by_id(self, case_id: str) -> CaseManifestEntry:
        """Return a case by id."""

        for entry in self.entries:
            if entry.case_id == case_id:
                return entry
        raise KeyError(f"unknown case id: {case_id}")

    def runnable_entries(self) -> tuple[CaseManifestEntry, ...]:
        """Return entries that have runnable commands."""

        return tuple(entry for entry in self.entries if entry.runnable)

    def filter(
        self,
        *,
        status: str | None = None,
        runnable_only: bool = False,
    ) -> CaseManifest:
        """Return a filtered manifest."""

        entries = self.entries
        if status is not None:
            entries = tuple(entry for entry in entries if entry.status == status)
        if runnable_only:
            entries = tuple(entry for entry in entries if entry.runnable)
        return CaseManifest(
            entries=entries,
            root=self.root,
            schema_version=self.schema_version,
            artifact_id=self.artifact_id,
        )

    def status_counts(self) -> dict[str, int]:
        """Return case counts by status."""

        return dict(sorted(Counter(entry.status for entry in self.entries).items()))

    def category_counts(self) -> dict[str, int]:
        """Return case counts by category."""

        return dict(sorted(Counter(entry.category for entry in self.entries).items()))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest."""

        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "status_counts": self.status_counts(),
            "category_counts": self.category_counts(),
            "entries": [entry.to_dict(root=self.root) for entry in self.entries],
        }


def default_case_manifest(root: str | Path = PROJECT_ROOT) -> CaseManifest:
    """Return the built-in case manifest.

    The manifest intentionally separates runnable cases from planned upstream
    parity fixtures. That makes GUI/docs progress visible without claiming
    complete TokaMaker feature parity before numeric gates exist.
    """

    return CaseManifest(root=Path(root), entries=_DEFAULT_ENTRIES)


def case_table_rows(manifest: CaseManifest | None = None) -> list[dict[str, str]]:
    """Return GUI/CLI-friendly table rows for the manifest."""

    case_manifest = default_case_manifest() if manifest is None else manifest
    rows: list[dict[str, str]] = []
    for entry in case_manifest.entries:
        rows.append(
            {
                "case_id": entry.case_id,
                "title": entry.title,
                "status": entry.status,
                "category": entry.category,
                "parity_level": entry.parity_level,
                "path": entry.path or "",
                "command": entry.command or "",
                "validation_gate": entry.validation_gate or "",
                "sources": str(len(entry.upstream_sources)),
                "outputs": ", ".join(entry.outputs),
            }
        )
    return rows


def case_source_preview(
    case_id: str,
    *,
    manifest: CaseManifest | None = None,
    root: str | Path = PROJECT_ROOT,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """Return a bounded text preview for a local case file."""

    case_manifest = default_case_manifest(root) if manifest is None else manifest
    entry = case_manifest.by_id(case_id)
    if entry.path is None:
        return {
            "case_id": entry.case_id,
            "path": None,
            "exists": False,
            "truncated": False,
            "source": "",
            "message": "This case is represented by a command or upstream fixture, not a local file.",
        }

    path = Path(root) / entry.path
    if not path.exists():
        return {
            "case_id": entry.case_id,
            "path": entry.path,
            "exists": False,
            "truncated": False,
            "source": "",
            "message": f"Case file is not present: {entry.path}",
        }
    source = path.read_text(encoding="utf-8")
    return {
        "case_id": entry.case_id,
        "path": entry.path,
        "exists": True,
        "truncated": len(source) > max_chars,
        "source": source[:max_chars],
        "message": "",
    }


def case_manifest_to_json(manifest: CaseManifest | None = None) -> str:
    """Serialize a case manifest with stable formatting."""

    case_manifest = default_case_manifest() if manifest is None else manifest
    return json.dumps(case_manifest.to_dict(), indent=2, sort_keys=True) + "\n"


def write_case_manifest(
    path: str | Path,
    *,
    manifest: CaseManifest | None = None,
) -> Path:
    """Write the manifest JSON and return the resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(case_manifest_to_json(manifest), encoding="utf-8")
    return output_path.resolve()


def _case_path_exists(root: str | Path, path: str | None) -> bool:
    if path is None or path.startswith(("http://", "https://")):
        return False
    return (Path(root) / path).exists()


_DEFAULT_ENTRIES = (
    CaseManifestEntry(
        case_id="fixed-boundary-seed",
        title="Fixed-boundary seed equilibrium",
        status="runnable",
        category="fixed-boundary",
        description=(
            "Rectangular-grid seed Grad-Shafranov solve used by the README, CLI, "
            "GUI seed view, and docs figures."
        ),
        parity_level="manufactured_validation",
        path="examples/fixed_boundary.toml",
        command="tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png",
        validation_gate="tokamaker-jax verify --gate grad-shafranov --subdivisions 4 8 16",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/fixed_boundary/fixed_boundary_ex1.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/fixed_boundary/fixed_boundary_ex2.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/fixed_boundary/gNT_example",
        ),
        outputs=(
            "outputs/fixed_boundary.png",
            "docs/_static/fixed_boundary_seed.png",
            "docs/_static/manufactured_grad_shafranov_convergence.png",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=(
            "Current solver is a seed rectangular-grid path, not full upstream equilibrium parity.",
        ),
    ),
    CaseManifestEntry(
        case_id="case-manifest-browser",
        title="GUI/CLI case manifest browser",
        status="runnable",
        category="workflow",
        description=(
            "Single manifest used by the GUI, CLI, docs, and tests to show runnable "
            "cases and planned parity fixtures."
        ),
        parity_level="workflow_fixture",
        command="tokamaker-jax cases",
        validation_gate="tokamaker-jax cases --json",
        outputs=(
            "docs/_static/case_manifest.json",
            "docs/_static/case_manifest_status.png",
        ),
        notes=("This is an operational workflow artifact, not a physics parity claim.",),
    ),
    CaseManifestEntry(
        case_id="cpc-seed-family",
        title="TokaMaker CPC seed-family surrogate",
        status="runnable",
        category="literature-surrogate",
        description=(
            "Citation-linked seed-family artifact exercising the literature "
            "reproduction report/plot workflow while exact published case inputs "
            "remain a future gate."
        ),
        parity_level="surrogate_fixture",
        path="examples/reproduce_cpc_seed_family.py",
        command="python examples/reproduce_cpc_seed_family.py outputs/literature/cpc_seed_family",
        validation_gate="python -m pytest tests/test_literature_reproduction.py",
        upstream_sources=("OpenFUSIONToolkit/src/examples/TokaMaker",),
        outputs=(
            "docs/_static/cpc_seed_family.png",
            "docs/_static/cpc_seed_family_report.json",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=(
            "Marked as surrogate until exact source data and numeric figure tolerances are checked in.",
        ),
    ),
    CaseManifestEntry(
        case_id="openfusiontoolkit-green-parity",
        title="OpenFUSIONToolkit circular-loop eval_green parity",
        status="validation_gate",
        category="kernel-parity",
        description=(
            "Availability-gated numeric comparison between the JAX circular-loop "
            "elliptic kernel and local OpenFUSIONToolkit/TokaMaker eval_green."
        ),
        parity_level="kernel_parity",
        command="tokamaker-jax verify --gate oft-parity",
        validation_gate="tokamaker-jax verify --gate oft-parity",
        upstream_sources=(
            "OpenFUSIONToolkit/src/python/OpenFUSIONToolkit/TokaMaker/util.py",
            "OpenFUSIONToolkit/src/examples/TokaMaker",
        ),
        outputs=("docs/_static/openfusiontoolkit_comparison_report.json",),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=(
            "Runs as skipped_unavailable when the local OFT shared library has not been built.",
        ),
    ),
    CaseManifestEntry(
        case_id="free-boundary-target-schema",
        title="Free-boundary target TOML schema preview",
        status="schema_preview",
        category="free-boundary",
        description=(
            "Human-readable TOML shape for the future full free-boundary workflow "
            "with constraints and targets."
        ),
        parity_level="source_audit",
        path="examples/free_boundary_target.toml",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_baseline_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/DIIID/DIIID_baseline_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/HBT/HBT_eq_ex.ipynb",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=("Preview only; not accepted by the current solver loader.",),
    ),
    CaseManifestEntry(
        case_id="iter-baseline-upstream",
        title="ITER baseline upstream fixture",
        status="planned_upstream_fixture",
        category="free-boundary",
        description=(
            "Planned full-code parity fixture for ITER baseline and H-mode "
            "examples, including mesh, coils, profiles, constraints, and plots."
        ),
        parity_level="source_audit",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_baseline_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_Hmode_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_geom.json",
            "OpenFUSIONToolkit/src/examples/TokaMaker/ITER/ITER_mesh.h5",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=("Target acceptance level: workflow_parity after numeric tolerances exist.",),
    ),
    CaseManifestEntry(
        case_id="diiid-baseline-upstream",
        title="DIII-D baseline upstream fixture",
        status="planned_upstream_fixture",
        category="reconstruction",
        description=(
            "Planned parity fixture for a DIII-D baseline case with mesh, "
            "geometry, and g-file input."
        ),
        parity_level="source_audit",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/DIIID/DIIID_baseline_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/DIIID/DIIID_geom.json",
            "OpenFUSIONToolkit/src/examples/TokaMaker/DIIID/DIIID_mesh.h5",
            "OpenFUSIONToolkit/src/examples/TokaMaker/DIIID/g192185.02440",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=("Target acceptance level: equilibrium_parity then workflow_parity.",),
    ),
    CaseManifestEntry(
        case_id="hbt-equilibrium-upstream",
        title="HBT equilibrium upstream fixture",
        status="planned_upstream_fixture",
        category="free-boundary",
        description=(
            "Planned HBT fixture covering vacuum-coil and equilibrium notebooks "
            "from upstream TokaMaker."
        ),
        parity_level="source_audit",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/HBT/HBT_eq_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/HBT/HBT_vac_coils.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/HBT/HBT_geom.json",
            "OpenFUSIONToolkit/src/examples/TokaMaker/HBT/HBT_mesh.h5",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=("Use after coil/passive-structure data models are implemented.",),
    ),
    CaseManifestEntry(
        case_id="cute-vde-upstream",
        title="CUTE VDE upstream fixture",
        status="planned_upstream_fixture",
        category="time-dependent",
        description=(
            "Planned fixture for time-dependent and vertical displacement workflow parity."
        ),
        parity_level="source_audit",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/CUTE/CUTE_VDE_ex.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/CUTE/CUTE_geom.json",
            "OpenFUSIONToolkit/src/examples/TokaMaker/CUTE/CUTE_mesh.h5",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=(
            "Requires passive structures and time stepping before numeric parity can be claimed.",
        ),
    ),
    CaseManifestEntry(
        case_id="nstxu-isoflux-controller-upstream",
        title="NSTX-U isoflux controller upstream fixture",
        status="planned_upstream_fixture",
        category="control",
        description=("Planned advanced-workflow fixture for controller and isoflux target parity."),
        parity_level="source_audit",
        upstream_sources=(
            "OpenFUSIONToolkit/src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_shape_control_simulator.ipynb",
            "OpenFUSIONToolkit/src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_geom.json",
            "OpenFUSIONToolkit/src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_mesh.h5",
        ),
        citations=("Hansen et al. 2024, doi:10.1016/j.cpc.2024.109111",),
        notes=("Requires reconstruction/control APIs and fixture tolerances.",),
    ),
)
