from pathlib import Path

import numpy as np
from conftest import REPO_ROOT

import tokamaker_jax.gui
from tokamaker_jax.cli import main, run_config


def test_run_config_writes_artifacts(tmp_path: Path):
    output = tmp_path / "solution.npz"
    plot = tmp_path / "solution.png"

    solution = run_config(REPO_ROOT / "examples" / "fixed_boundary.toml", output=output, plot=plot)

    assert solution.psi.shape == (65, 65)
    assert output.exists()
    assert plot.exists()
    with np.load(output) as data:
        assert {"r", "z", "psi", "source", "residual_history"} <= set(data.files)


def test_main_runs_toml(capsys, tmp_path: Path):
    output = tmp_path / "solution.npz"
    exit_code = main([str(REPO_ROOT / "examples" / "fixed_boundary.toml"), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "residual_final" in captured.out


def test_main_without_toml_launches_gui(monkeypatch):
    called = {"value": False}

    def fake_launch_gui():
        called["value"] = True

    monkeypatch.setattr(tokamaker_jax.gui, "launch_gui", fake_launch_gui)

    assert main([]) == 0
    assert called["value"] is True


def test_main_gui_subcommand_passes_host_port_and_browser_flags(monkeypatch):
    called = {}

    def fake_launch_gui(*, host, port, reload, show):
        called.update({"host": host, "port": port, "reload": reload, "show": show})

    monkeypatch.setattr(tokamaker_jax.gui, "launch_gui", fake_launch_gui)

    assert main(["gui", "--host", "0.0.0.0", "--port", "8091", "--reload", "--no-browser"]) == 0
    assert called == {"host": "0.0.0.0", "port": 8091, "reload": True, "show": False}


def test_run_config_uses_toml_output_defaults(tmp_path: Path):
    config_path = tmp_path / "case.toml"
    output_path = tmp_path / "from_config.npz"
    config_path.write_text(
        f"""
[grid]
nr = 9
nz = 9

[solver]
iterations = 3

[output]
npz = "{output_path}"
""".strip()
    )

    run_config(config_path)

    assert output_path.exists()
