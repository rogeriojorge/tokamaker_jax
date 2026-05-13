from pathlib import Path

import pytest
from conftest import REPO_ROOT

from tokamaker_jax import cli
from tokamaker_jax.cli import ConfigValidationError, main, run_verification_gates, validate_config


def test_validate_config_accepts_example():
    report = validate_config(REPO_ROOT / "examples" / "fixed_boundary.toml")

    assert report.grid_shape == (65, 65)
    assert report.region_count == 0
    assert [label for label, _ in report.output_paths] == ["npz", "plot"]


def test_main_validate_succeeds_without_solving_or_writing(monkeypatch, capsys, tmp_path: Path):
    def fail_solve(*args, **kwargs):
        raise AssertionError("validate must not run the solver")

    output = tmp_path / "solution.npz"
    monkeypatch.setattr(cli, "solve_from_config", fail_solve)

    exit_code = main(
        ["validate", str(REPO_ROOT / "examples" / "fixed_boundary.toml"), "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Validation succeeded" in captured.out
    assert "grid: nr=65, nz=65" in captured.out
    assert not output.exists()


def test_main_validate_reports_invalid_grid(capsys, tmp_path: Path):
    config_path = tmp_path / "bad_grid.toml"
    config_path.write_text(
        """
[grid]
nr = 2
nz = 9
""",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Validation failed" in captured.err
    assert "grid.nr must be at least 3" in captured.err


def test_main_verify_runs_manufactured_gate_and_writes_json(capsys, tmp_path: Path):
    output = tmp_path / "verify.json"

    exit_code = main(
        [
            "verify",
            "--gate",
            "grad-shafranov",
            "--subdivisions",
            "4",
            "8",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "grad_shafranov" in captured.out
    assert "weighted_h1_rates" in output.read_text(encoding="utf-8")


def test_main_verify_runs_reduced_coil_green_gate(capsys):
    exit_code = main(["verify", "--gate", "coil-green"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "coil_green" in captured.out
    assert "gradient_error" in captured.out


def test_main_verify_runs_circular_loop_gate(capsys):
    exit_code = main(["verify", "--gate", "circular-loop"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "circular_loop" in captured.out
    assert "elliptic_quadrature_relative_error" in captured.out


def test_main_verify_runs_openfusiontoolkit_parity_probe(capsys):
    exit_code = main(["verify", "--gate", "oft-parity"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "openfusiontoolkit" in captured.out
    assert "circular_loop_eval_green" in captured.out


def test_main_verify_runs_profile_iteration_gate(capsys):
    exit_code = main(["verify", "--gate", "profile-iteration"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "profile_iteration" in captured.out
    assert "load_oracle_error" in captured.out


def test_main_verify_runs_free_boundary_profile_gate(capsys):
    exit_code = main(["verify", "--gate", "free-boundary-profile"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "free_boundary_profile" in captured.out
    assert "coil_linearity_relative_error" in captured.out


def test_main_verify_runs_fixed_boundary_geqdsk_gate(capsys):
    exit_code = main(["verify", "--gate", "fixed-boundary-geqdsk"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fixed_boundary_geqdsk" in captured.out
    assert '"numeric_parity_claim": false' in captured.out
    assert '"status": "pass"' in captured.out


def test_main_cases_lists_manifest_and_writes_json(capsys, tmp_path: Path):
    output = tmp_path / "cases.json"

    exit_code = main(["cases", "--runnable-only", "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "tokamaker-jax cases" in captured.out
    assert "fixed-boundary-seed" in captured.out
    assert output.exists()
    assert "planned_upstream_fixture" not in output.read_text(encoding="utf-8")


def test_main_cases_prints_json(capsys):
    exit_code = main(["cases", "--status", "validation_gate", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"artifact_id": "tokamaker-jax-case-manifest"' in captured.out
    assert "openfusiontoolkit-green-parity" in captured.out
    assert "fixed-boundary-seed" not in captured.out


def test_main_upstream_fixtures_reports_json_without_requiring_checkout(capsys, tmp_path: Path):
    output = tmp_path / "upstream_fixtures.json"

    exit_code = main(
        [
            "upstream-fixtures",
            "--root",
            str(tmp_path / "missing_oft"),
            "--json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert '"artifact_id": "upstream-tokamaker-fixture-summary"' in captured.out
    assert '"checkout_exists": false' in captured.out


def test_main_upstream_fixtures_prints_summary(capsys, tmp_path: Path):
    exit_code = main(["upstream-fixtures", "--root", str(tmp_path / "missing_oft")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mesh/geometry inventory only" in captured.out
    assert "available=no" in captured.out


def test_main_fixed_boundary_evidence_reports_json_without_requiring_checkout(
    capsys,
    tmp_path: Path,
):
    output = tmp_path / "fixed_boundary_evidence.json"

    exit_code = main(
        [
            "fixed-boundary-evidence",
            "--root",
            str(tmp_path / "missing_oft"),
            "--json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert '"artifact_id": "upstream-fixed-boundary-evidence"' in captured.out
    assert '"claim": "source_evidence_only"' in captured.out
    assert '"checkout_exists": false' in captured.out
    assert '"numeric_parity_claim": false' in captured.out


def test_main_fixed_boundary_evidence_prints_summary(capsys, tmp_path: Path):
    root = tmp_path / "OpenFUSIONToolkit"
    fixed_dir = root / "src/examples/TokaMaker/fixed_boundary"
    fixed_dir.mkdir(parents=True)
    values = list(range(34))
    (fixed_dir / "gNT_example").write_text(
        "TEST 0 2 2\n" + " ".join(f"{value:.9E}" for value in values),
        encoding="utf-8",
    )

    exit_code = main(["fixed-boundary-evidence", "--root", str(root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "source evidence only" in captured.out
    assert "fixed_boundary_ex1.ipynb: exists=False" in captured.out
    assert "gNT_example: 2x2 grid" in captured.out


def test_run_verification_gates_validates_subdivisions():
    with pytest.raises(ValueError, match="at least two"):
        run_verification_gates("poisson", (4,))
    with pytest.raises(ValueError, match="at least 2"):
        run_verification_gates("poisson", (1, 2))
    with pytest.raises(ValueError, match="gate must be"):
        run_verification_gates("bad", (4, 8))


def test_validate_config_reports_physical_and_solver_errors(tmp_path: Path):
    config_path = tmp_path / "bad_controls.toml"
    config_path.write_text(
        """
[grid]
r_min = 1.0
r_max = 0.5
nr = 9
nz = 9

[source]
profile = "unsupported"
pressure_scale = "bad"
ffp_scale = 1.0

[solver]
iterations = "many"
relaxation = 2.0
dtype = "float16"

[[coil]]
name = "PF"
r = 1.0
z = 0.0
current = 1.0
sigma = 0.1

[[coil]]
name = "PF"
r = 1.0
z = 0.0
current = 1.0
sigma = -0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config_path)

    message = str(exc_info.value)
    assert "grid: r_max must be greater than r_min" in message
    assert "source.profile must be 'solovev'" in message
    assert "source.pressure_scale must be a finite number" in message
    assert "solver.iterations must be an integer" in message
    assert "solver.relaxation must satisfy" in message
    assert "solver.dtype must be 'float32' or 'float64'" in message
    assert "duplicated" in message
    assert "coil[1].sigma must be positive" in message


def test_validate_config_rejects_empty_and_bad_coil_fields(tmp_path: Path):
    config_path = tmp_path / "bad_coil.toml"
    config_path.write_text(
        """
[grid]
nr = 9
nz = 9

[[coil]]
name = ""
r = "bad"
z = 0.0
current = 1.0
sigma = 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config_path)

    message = str(exc_info.value)
    assert "coil[0].name must be nonempty" in message
    assert "coil[0].r must be a finite number" in message


def test_validate_config_rejects_invalid_region_geometry(tmp_path: Path):
    config_path = tmp_path / "bad_region.toml"
    config_path.write_text(
        """
[[region]]
shape = "rectangle"
id = 1
name = "BAD"
r_min = 2.0
r_max = 1.0
z_min = -0.5
z_max = 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match=r"TOML/config parse error"):
        validate_config(config_path)


def test_validate_config_rejects_output_parent_file(tmp_path: Path):
    config_path = tmp_path / "bad_output.toml"
    parent_file = tmp_path / "not_a_directory"
    parent_file.write_text("content", encoding="utf-8")
    config_path.write_text(
        f"""
[grid]
nr = 9
nz = 9

[output]
npz = "{parent_file / "solution.npz"}"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="parent is not a directory"):
        validate_config(config_path)


def test_validate_config_rejects_output_directory_and_non_path(tmp_path: Path):
    directory_config = tmp_path / "output_directory.toml"
    directory_config.write_text(
        f"""
[grid]
nr = 9
nz = 9

[output]
plot = "{tmp_path}"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="points to a directory"):
        validate_config(directory_config)

    non_path_config = tmp_path / "output_non_path.toml"
    non_path_config.write_text(
        """
[grid]
nr = 9
nz = 9

[output]
plot = [1, 2]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="must be a filesystem path"):
        validate_config(non_path_config)


def test_validate_output_path_reports_unwritable_ancestor(monkeypatch, tmp_path: Path):
    errors: list[str] = []
    monkeypatch.setattr(cli.os, "access", lambda *_args: False)

    cli._validate_output_path("plot", tmp_path / "figure.png", errors)

    assert any("ancestor is not writable" in error for error in errors)
