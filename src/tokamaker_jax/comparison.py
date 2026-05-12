"""OpenFUSIONToolkit/TokaMaker comparison probes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from tokamaker_jax.free_boundary import circular_loop_elliptic_flux

DEFAULT_OFT_CHECKOUT = Path("/Users/rogeriojorge/local/OpenFUSIONToolkit")


@dataclass(frozen=True)
class OpenFUSIONToolkitProbe:
    """Availability and source inventory for a local OpenFUSIONToolkit checkout."""

    checkout_path: str
    exists: bool
    commit: str | None
    python_path: str | None
    library_path: str | None
    import_ok: bool
    tokamaker_import_ok: bool
    reason: str | None
    source_inventory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "checkout_path": self.checkout_path,
            "exists": self.exists,
            "commit": self.commit,
            "python_path": self.python_path,
            "library_path": self.library_path,
            "import_ok": self.import_ok,
            "tokamaker_import_ok": self.tokamaker_import_ok,
            "reason": self.reason,
            "source_inventory": self.source_inventory,
        }


@dataclass(frozen=True)
class OpenFUSIONToolkitComparison:
    """Result of an original TokaMaker comparison attempt."""

    status: str
    comparison: str
    probe: OpenFUSIONToolkitProbe
    n_points: int
    relative_error: float | None
    max_abs_error: float | None
    jax_flux: list[float]
    tokamaker_convention_jax_flux: list[float]
    oft_flux: list[float] | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "status": self.status,
            "comparison": self.comparison,
            "probe": self.probe.to_dict(),
            "n_points": self.n_points,
            "relative_error": self.relative_error,
            "max_abs_error": self.max_abs_error,
            "jax_flux": self.jax_flux,
            "tokamaker_convention_jax_flux": self.tokamaker_convention_jax_flux,
            "oft_flux": self.oft_flux,
            "notes": list(self.notes),
        }


def probe_openfusiontoolkit(
    checkout_path: str | Path = DEFAULT_OFT_CHECKOUT,
) -> OpenFUSIONToolkitProbe:
    """Probe whether the original OpenFUSIONToolkit/TokaMaker can run locally."""

    checkout = Path(checkout_path)
    exists = checkout.exists()
    if not exists:
        return OpenFUSIONToolkitProbe(
            checkout_path=str(checkout),
            exists=False,
            commit=None,
            python_path=None,
            library_path=None,
            import_ok=False,
            tokamaker_import_ok=False,
            reason="checkout_missing",
            source_inventory={},
        )

    commit = _git_commit(checkout)
    inventory = _source_inventory(checkout)
    candidates = _candidate_python_paths(checkout)
    if not candidates:
        return OpenFUSIONToolkitProbe(
            checkout_path=str(checkout),
            exists=True,
            commit=commit,
            python_path=str(checkout / "src" / "python"),
            library_path=None,
            import_ok=False,
            tokamaker_import_ok=False,
            reason="python_package_missing",
            source_inventory=inventory,
        )

    python_path = candidates[0]
    library_path = _shared_library_path(python_path)
    import_probe = _run_oft_python(
        python_path,
        """
import json
payload = {"import_ok": False, "tokamaker_import_ok": False, "reason": None}
try:
    import OpenFUSIONToolkit
    payload["import_ok"] = True
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    payload["tokamaker_import_ok"] = True
except Exception as exc:
    payload["reason"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
""",
    )
    if import_probe.returncode != 0:
        reason = import_probe.stderr.strip() or import_probe.stdout.strip() or "import_probe_failed"
        return OpenFUSIONToolkitProbe(
            checkout_path=str(checkout),
            exists=True,
            commit=commit,
            python_path=str(python_path),
            library_path=None if library_path is None else str(library_path),
            import_ok=False,
            tokamaker_import_ok=False,
            reason=reason,
            source_inventory=inventory,
        )

    try:
        payload = json.loads(import_probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {
            "import_ok": False,
            "tokamaker_import_ok": False,
            "reason": import_probe.stdout.strip() or "invalid_import_probe_output",
        }

    return OpenFUSIONToolkitProbe(
        checkout_path=str(checkout),
        exists=True,
        commit=commit,
        python_path=str(python_path),
        library_path=None if library_path is None else str(library_path),
        import_ok=bool(payload["import_ok"]),
        tokamaker_import_ok=bool(payload["tokamaker_import_ok"]),
        reason=payload["reason"],
        source_inventory=inventory,
    )


def run_openfusiontoolkit_green_comparison(
    checkout_path: str | Path = DEFAULT_OFT_CHECKOUT,
) -> OpenFUSIONToolkitComparison:
    """Compare circular-loop flux with original TokaMaker ``eval_green`` when available."""

    points = jnp.asarray([[1.72, 0.18], [2.05, -0.22], [1.35, 0.31]], dtype=jnp.float64)
    coil = (1.52, 0.03)
    jax_flux = circular_loop_elliptic_flux(points, coil[0], coil[1], core_radius=0.0)
    tokamaker_convention_jax_flux = -jax_flux
    probe = probe_openfusiontoolkit(checkout_path)
    notes = (
        "Compares the JAX closed-form circular filament with "
        "OpenFUSIONToolkit.TokaMaker.util.eval_green for unit current.",
        "OpenFUSIONToolkit eval_green uses the TokaMaker Grad-Shafranov Green's-function "
        "sign convention, so the comparison uses -circular_loop_elliptic_flux.",
        "The local OpenFUSIONToolkit shared library must be built for the numeric comparison to run.",
    )
    if not probe.tokamaker_import_ok or probe.python_path is None:
        return OpenFUSIONToolkitComparison(
            status="skipped_unavailable",
            comparison="circular_loop_eval_green",
            probe=probe,
            n_points=int(points.shape[0]),
            relative_error=None,
            max_abs_error=None,
            jax_flux=[float(value) for value in jax_flux],
            tokamaker_convention_jax_flux=[float(value) for value in tokamaker_convention_jax_flux],
            oft_flux=None,
            notes=notes,
        )

    payload = {
        "points": [[float(value) for value in row] for row in points],
        "coil": list(coil),
    }
    command = f"""
import json
import numpy as np
from OpenFUSIONToolkit.TokaMaker.util import eval_green
payload = {json.dumps(payload)}
points = np.asarray(payload["points"], dtype=np.float64)
coil = np.asarray(payload["coil"], dtype=np.float64)
print(json.dumps({{"flux": eval_green(points, coil).tolist()}}, sort_keys=True))
"""
    result = _run_oft_python(Path(probe.python_path), command)
    if result.returncode != 0:
        return OpenFUSIONToolkitComparison(
            status="failed",
            comparison="circular_loop_eval_green",
            probe=probe,
            n_points=int(points.shape[0]),
            relative_error=None,
            max_abs_error=None,
            jax_flux=[float(value) for value in jax_flux],
            tokamaker_convention_jax_flux=[float(value) for value in tokamaker_convention_jax_flux],
            oft_flux=None,
            notes=notes + (result.stderr.strip() or result.stdout.strip(),),
        )

    oft_payload = json.loads(result.stdout.strip().splitlines()[-1])
    oft_flux = jnp.asarray(oft_payload["flux"], dtype=jnp.float64)
    difference = tokamaker_convention_jax_flux - oft_flux
    relative_error = jnp.linalg.norm(difference) / jnp.linalg.norm(oft_flux)
    max_abs_error = jnp.max(jnp.abs(difference))
    return OpenFUSIONToolkitComparison(
        status="passed" if float(relative_error) < 1.0e-10 else "failed",
        comparison="circular_loop_eval_green",
        probe=probe,
        n_points=int(points.shape[0]),
        relative_error=float(relative_error),
        max_abs_error=float(max_abs_error),
        jax_flux=[float(value) for value in jax_flux],
        tokamaker_convention_jax_flux=[float(value) for value in tokamaker_convention_jax_flux],
        oft_flux=[float(value) for value in oft_flux],
        notes=notes,
    )


def _candidate_python_paths(checkout: Path) -> tuple[Path, ...]:
    paths = [
        path
        for path in sorted(checkout.glob("build*/python"))
        if (path / "OpenFUSIONToolkit").exists()
    ]
    source_path = checkout / "src" / "python"
    if source_path.exists():
        paths.append(source_path)
    return tuple(sorted(paths, key=lambda path: _shared_library_path(path) is None))


def _shared_library_path(python_path: Path) -> Path | None:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for candidate in (
        python_path / "OpenFUSIONToolkit" / f"liboftpy{suffix}",
        python_path.parent / "bin" / f"liboftpy{suffix}",
    ):
        if candidate.exists():
            return candidate
    return None


def _run_oft_python(python_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(python_path) if not existing_path else f"{python_path}:{existing_path}"
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _git_commit(checkout: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_inventory(checkout: Path) -> dict[str, Any]:
    examples = sorted(
        str(path.relative_to(checkout))
        for path in (checkout / "src" / "examples" / "TokaMaker").glob("*")
        if path.exists()
    )
    tests = checkout / "src" / "tests" / "physics" / "test_TokaMaker.py"
    source = checkout / "src" / "physics" / "grad_shaf.F90"
    return {
        "tokamaker_examples": examples,
        "test_tokamaker_py": str(tests.relative_to(checkout)) if tests.exists() else None,
        "grad_shaf_source": str(source.relative_to(checkout)) if source.exists() else None,
    }
