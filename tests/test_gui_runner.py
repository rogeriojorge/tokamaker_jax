import subprocess
from pathlib import Path

import pytest

from tokamaker_jax.cases import CaseManifest, CaseManifestEntry
from tokamaker_jax.gui_runner import run_manifest_toml_case


def test_run_manifest_toml_case_dry_run_validates_without_subprocess(monkeypatch, tmp_path: Path):
    manifest = _single_case_manifest(tmp_path)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("dry run must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path, dry_run=True)

    assert result.status == "validated"
    assert result.validation_status == "pass"
    assert result.dry_run is True
    assert result.returncode is None
    assert result.command == ("tokamaker-jax", "case.toml", "--output", "out.npz")
    assert result.artifacts[0]["path"] == str(tmp_path / "out.npz")
    assert result.artifacts[0]["exists"] is False


def test_run_manifest_toml_case_runs_with_argv_cwd_timeout_and_capture(
    monkeypatch,
    tmp_path: Path,
):
    manifest = _single_case_manifest(tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        (kwargs["cwd"] / "out.npz").write_bytes(b"artifact")
        return subprocess.CompletedProcess(args, 0, stdout="ran\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path, timeout_s=3.5)

    args, kwargs = calls[0]
    assert args[-3:] == ("case.toml", "--output", "out.npz")
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] == 3.5
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["shell"] is False
    assert result.status == "pass"
    assert result.returncode == 0
    assert result.stdout == "ran\n"
    assert result.artifacts[0]["exists"] is True
    assert result.artifacts[0]["size_bytes"] == len(b"artifact")
    assert result.to_dict()["command"] == ["tokamaker-jax", "case.toml", "--output", "out.npz"]


def test_run_manifest_toml_case_preserves_non_tokamaker_argv(monkeypatch, tmp_path: Path):
    manifest = _single_case_manifest(tmp_path, command="python case.toml --output out.npz")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path)

    assert calls[0][0] == ("python", "case.toml", "--output", "out.npz")
    assert result.status == "pass"


def test_run_manifest_toml_case_blocks_invalid_toml_before_subprocess(
    monkeypatch,
    tmp_path: Path,
):
    manifest = _single_case_manifest(tmp_path, toml_text="[grid]\nnr = 2\n")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("invalid TOML must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path)

    assert result.status == "validation_failed"
    assert result.validation_status == "fail"
    assert (
        "validation error" in result.validation_message
        or "parse error" in result.validation_message
    )


def test_run_manifest_toml_case_reports_timeout(monkeypatch, tmp_path: Path):
    manifest = _single_case_manifest(tmp_path)

    def timeout_run(args, **_kwargs):
        raise subprocess.TimeoutExpired(args, timeout=0.01, output="partial", stderr=b"late")

    monkeypatch.setattr(subprocess, "run", timeout_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path, timeout_s=0.01)

    assert result.status == "timeout"
    assert result.timed_out is True
    assert result.stdout == "partial"
    assert result.stderr == "late"


def test_run_manifest_toml_case_reports_os_error(monkeypatch, tmp_path: Path):
    manifest = _single_case_manifest(tmp_path)

    def os_error_run(*_args, **_kwargs):
        raise OSError("cannot start process")

    monkeypatch.setattr(subprocess, "run", os_error_run)

    result = run_manifest_toml_case("case", manifest=manifest, root=tmp_path)

    assert result.status == "error"
    assert result.returncode is None
    assert result.stderr == "cannot start process"


def test_run_manifest_toml_case_rejects_non_toml_and_shell_tokens(tmp_path: Path):
    with pytest.raises(ValueError, match="does not define a runnable command"):
        run_manifest_toml_case(
            "case",
            manifest=_single_case_manifest(tmp_path, command=""),
        )

    with pytest.raises(ValueError, match="does not reference a local TOML config"):
        run_manifest_toml_case(
            "case",
            manifest=_single_case_manifest(tmp_path, path=""),
        )

    with pytest.raises(ValueError, match="not a TOML"):
        run_manifest_toml_case("case", manifest=_single_case_manifest(tmp_path, path="case.py"))

    with pytest.raises(ValueError, match="shell token"):
        run_manifest_toml_case(
            "case",
            manifest=_single_case_manifest(
                tmp_path,
                command="tokamaker-jax case.toml && rm out.npz",
            ),
        )


def _single_case_manifest(
    root: Path,
    *,
    path: str = "case.toml",
    command: str = "tokamaker-jax case.toml --output out.npz",
    toml_text: str | None = None,
) -> CaseManifest:
    if toml_text is None:
        toml_text = """
[grid]
nr = 9
nz = 9

[solver]
iterations = 1

[output]
npz = "out.npz"
""".strip()
    if path:
        (root / path).write_text(toml_text, encoding="utf-8")
    return CaseManifest(
        root=root,
        entries=(
            CaseManifestEntry(
                case_id="case",
                title="Case",
                status="runnable",
                category="test",
                description="test case",
                parity_level="test",
                path=path,
                command=command,
            ),
        ),
    )
