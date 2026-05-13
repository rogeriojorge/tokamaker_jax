"""Safe command runner helpers for GUI-launched TOML cases."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokamaker_jax.cases import CaseManifest, default_case_manifest
from tokamaker_jax.cli import ConfigValidationError, validate_config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHELL_TOKENS = frozenset({"|", "||", "&&", ";", "<", ">", ">>", "2>", "&"})


@dataclass(frozen=True)
class GuiCommandResult:
    """Structured result from a GUI case validation or run."""

    case_id: str
    command: tuple[str, ...]
    cwd: Path
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    dry_run: bool
    validation_status: str
    validation_message: str
    artifacts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return {
            "case_id": self.case_id,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "dry_run": self.dry_run,
            "validation_status": self.validation_status,
            "validation_message": self.validation_message,
            "artifacts": list(self.artifacts),
        }


def run_manifest_toml_case(
    case_id: str,
    *,
    manifest: CaseManifest | None = None,
    root: str | Path | None = None,
    timeout_s: float = 120.0,
    dry_run: bool = False,
) -> GuiCommandResult:
    """Validate and optionally run one manifest-backed TOML case.

    Commands are parsed into argv vectors and executed with ``shell=False``.
    The TOML config is validated before any subprocess is started.
    """

    if manifest is None:
        case_manifest = default_case_manifest(_PROJECT_ROOT if root is None else root)
    else:
        case_manifest = manifest
    root_path = Path(root) if root is not None else case_manifest.root
    entry = case_manifest.by_id(case_id)
    command = _entry_command_argv(entry.command)
    config_path = _entry_toml_path(entry.path, cwd=root_path)

    validation_status, validation_message, artifacts = _validate_case_config(
        config_path, cwd=root_path
    )
    if validation_status != "pass" or dry_run:
        status = "validated" if validation_status == "pass" else "validation_failed"
        return GuiCommandResult(
            case_id=case_id,
            command=command,
            cwd=root_path,
            status=status,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=0.0,
            timed_out=False,
            dry_run=dry_run,
            validation_status=validation_status,
            validation_message=validation_message,
            artifacts=artifacts,
        )

    started = time.monotonic()
    try:
        completed = subprocess.run(
            _executable_argv(command),
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return GuiCommandResult(
            case_id=case_id,
            command=command,
            cwd=root_path,
            status="timeout",
            returncode=None,
            stdout=_timeout_stream(exc.stdout),
            stderr=_timeout_stream(exc.stderr),
            duration_s=time.monotonic() - started,
            timed_out=True,
            dry_run=False,
            validation_status=validation_status,
            validation_message=validation_message,
            artifacts=_refresh_artifacts(artifacts),
        )
    except OSError as exc:
        return GuiCommandResult(
            case_id=case_id,
            command=command,
            cwd=root_path,
            status="error",
            returncode=None,
            stdout="",
            stderr=str(exc),
            duration_s=time.monotonic() - started,
            timed_out=False,
            dry_run=False,
            validation_status=validation_status,
            validation_message=validation_message,
            artifacts=_refresh_artifacts(artifacts),
        )

    return GuiCommandResult(
        case_id=case_id,
        command=command,
        cwd=root_path,
        status="pass" if completed.returncode == 0 else "fail",
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=time.monotonic() - started,
        timed_out=False,
        dry_run=False,
        validation_status=validation_status,
        validation_message=validation_message,
        artifacts=_refresh_artifacts(artifacts),
    )


def _entry_command_argv(command: str | None) -> tuple[str, ...]:
    if not command:
        raise ValueError("manifest entry does not define a runnable command")
    argv = tuple(shlex.split(command))
    if not argv:
        raise ValueError("manifest entry command is empty")
    disallowed = _SHELL_TOKENS.intersection(argv)
    if disallowed:
        token = sorted(disallowed)[0]
        raise ValueError(f"manifest entry command contains shell token: {token}")
    return argv


def _entry_toml_path(path: str | None, *, cwd: Path) -> Path:
    if not path:
        raise ValueError("manifest entry does not reference a local TOML config")
    config_path = Path(path)
    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"manifest entry is not a TOML config: {path}")
    return config_path if config_path.is_absolute() else cwd / config_path


def _validate_case_config(
    config_path: Path,
    *,
    cwd: Path,
) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    try:
        report = validate_config(config_path)
    except ConfigValidationError as exc:
        return "fail", str(exc), ()

    artifacts = tuple(
        _artifact_record(label, path if path.is_absolute() else cwd / path)
        for label, path in report.output_paths
    )
    message = (
        f"TOML config is valid; grid {report.grid_shape[0]} x {report.grid_shape[1]}; "
        f"{len(artifacts)} artifact target(s)"
    )
    return "pass", message, artifacts


def _artifact_record(label: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _refresh_artifacts(artifacts: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        _artifact_record(str(artifact.get("label", "artifact")), Path(str(artifact["path"])))
        for artifact in artifacts
        if "path" in artifact
    )


def _executable_argv(command: tuple[str, ...]) -> tuple[str, ...]:
    if command[0] != "tokamaker-jax":
        return command
    return (sys.executable, "-m", "tokamaker_jax.cli", *command[1:])


def _timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream
