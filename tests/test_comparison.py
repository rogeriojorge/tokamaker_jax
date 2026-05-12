import json
import subprocess
from pathlib import Path

import jax.numpy as jnp

import tokamaker_jax.comparison as comparison_module
from tokamaker_jax.comparison import (
    OpenFUSIONToolkitProbe,
    probe_openfusiontoolkit,
    run_openfusiontoolkit_green_comparison,
)
from tokamaker_jax.free_boundary import circular_loop_elliptic_flux


def test_openfusiontoolkit_probe_reports_missing_checkout(tmp_path: Path):
    probe = probe_openfusiontoolkit(tmp_path / "missing")
    payload = probe.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert probe.exists is False
    assert probe.import_ok is False
    assert probe.tokamaker_import_ok is False
    assert probe.reason == "checkout_missing"


def test_openfusiontoolkit_comparison_is_json_ready_for_current_environment():
    comparison = run_openfusiontoolkit_green_comparison()
    payload = comparison.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert comparison.comparison == "circular_loop_eval_green"
    assert comparison.n_points == 3
    assert comparison.status in {"passed", "failed", "skipped_unavailable"}
    assert len(comparison.jax_flux) == 3
    assert len(comparison.tokamaker_convention_jax_flux) == 3
    if comparison.status == "passed":
        assert comparison.relative_error is not None
        assert comparison.relative_error < 1.0e-10
        assert comparison.oft_flux is not None
    else:
        assert comparison.probe.reason is not None


def test_openfusiontoolkit_probe_reports_missing_python_package(tmp_path: Path):
    checkout = tmp_path / "OpenFUSIONToolkit"
    checkout.mkdir()

    probe = probe_openfusiontoolkit(checkout)

    assert probe.exists is True
    assert probe.import_ok is False
    assert probe.tokamaker_import_ok is False
    assert probe.reason == "python_package_missing"


def test_openfusiontoolkit_probe_reports_failed_import_probe(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "OpenFUSIONToolkit"
    (checkout / "src" / "python").mkdir(parents=True)
    (checkout / "src" / "physics").mkdir(parents=True)
    (checkout / "src" / "physics" / "grad_shaf.F90").write_text("! source\n", encoding="utf-8")
    (checkout / "src" / "tests" / "physics").mkdir(parents=True)
    (checkout / "src" / "tests" / "physics" / "test_TokaMaker.py").write_text(
        "# tests\n", encoding="utf-8"
    )
    (checkout / "src" / "examples" / "TokaMaker" / "DIIID").mkdir(parents=True)

    monkeypatch.setattr(comparison_module, "_git_commit", lambda path: "abc123")
    monkeypatch.setattr(
        comparison_module,
        "_run_oft_python",
        lambda python_path, code: subprocess.CompletedProcess(
            args=["python"], returncode=1, stdout="", stderr="import failed"
        ),
    )

    probe = probe_openfusiontoolkit(checkout)

    assert probe.commit == "abc123"
    assert probe.reason == "import failed"
    assert probe.source_inventory["grad_shaf_source"] == "src/physics/grad_shaf.F90"
    assert probe.source_inventory["test_tokamaker_py"] == "src/tests/physics/test_TokaMaker.py"
    assert probe.source_inventory["tokamaker_examples"] == ["src/examples/TokaMaker/DIIID"]


def test_openfusiontoolkit_probe_reports_malformed_import_output(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "OpenFUSIONToolkit"
    (checkout / "src" / "python").mkdir(parents=True)

    monkeypatch.setattr(
        comparison_module,
        "_run_oft_python",
        lambda python_path, code: subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout="not-json\n", stderr=""
        ),
    )

    probe = probe_openfusiontoolkit(checkout)

    assert probe.exists is True
    assert probe.import_ok is False
    assert probe.tokamaker_import_ok is False
    assert probe.reason == "not-json"


def test_openfusiontoolkit_green_comparison_passes_with_matching_oft_flux(monkeypatch):
    probe = OpenFUSIONToolkitProbe(
        checkout_path="/tmp/oft",
        exists=True,
        commit="abc123",
        python_path="/tmp/oft/src/python",
        library_path="/tmp/oft/bin/liboftpy.dylib",
        import_ok=True,
        tokamaker_import_ok=True,
        reason=None,
    )
    points = jnp.asarray([[1.72, 0.18], [2.05, -0.22], [1.35, 0.31]], dtype=jnp.float64)
    expected_flux = circular_loop_elliptic_flux(points, 1.52, 0.03, core_radius=0.0)

    monkeypatch.setattr(comparison_module, "probe_openfusiontoolkit", lambda checkout_path: probe)
    monkeypatch.setattr(
        comparison_module,
        "_run_oft_python",
        lambda python_path, code: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=json.dumps({"flux": [float(value) for value in -expected_flux]}) + "\n",
            stderr="",
        ),
    )

    result = run_openfusiontoolkit_green_comparison("/tmp/oft")

    assert result.status == "passed"
    assert result.oft_flux == result.tokamaker_convention_jax_flux
    assert result.relative_error == 0.0
    assert result.max_abs_error == 0.0


def test_openfusiontoolkit_green_comparison_fails_on_subprocess_error(monkeypatch):
    probe = OpenFUSIONToolkitProbe(
        checkout_path="/tmp/oft",
        exists=True,
        commit="abc123",
        python_path="/tmp/oft/src/python",
        library_path="/tmp/oft/bin/liboftpy.dylib",
        import_ok=True,
        tokamaker_import_ok=True,
        reason=None,
    )

    monkeypatch.setattr(comparison_module, "probe_openfusiontoolkit", lambda checkout_path: probe)
    monkeypatch.setattr(
        comparison_module,
        "_run_oft_python",
        lambda python_path, code: subprocess.CompletedProcess(
            args=["python"], returncode=1, stdout="", stderr="eval failed"
        ),
    )

    result = run_openfusiontoolkit_green_comparison("/tmp/oft")

    assert result.status == "failed"
    assert result.oft_flux is None
    assert "eval failed" in result.notes[-1]


def test_openfusiontoolkit_green_comparison_fails_on_numeric_mismatch(monkeypatch):
    probe = OpenFUSIONToolkitProbe(
        checkout_path="/tmp/oft",
        exists=True,
        commit="abc123",
        python_path="/tmp/oft/src/python",
        library_path="/tmp/oft/bin/liboftpy.dylib",
        import_ok=True,
        tokamaker_import_ok=True,
        reason=None,
    )

    monkeypatch.setattr(comparison_module, "probe_openfusiontoolkit", lambda checkout_path: probe)
    monkeypatch.setattr(
        comparison_module,
        "_run_oft_python",
        lambda python_path, code: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=json.dumps({"flux": [1.0, 1.0, 1.0]}) + "\n",
            stderr="",
        ),
    )

    result = run_openfusiontoolkit_green_comparison("/tmp/oft")

    assert result.status == "failed"
    assert result.relative_error is not None
    assert result.relative_error > 1.0e-10
