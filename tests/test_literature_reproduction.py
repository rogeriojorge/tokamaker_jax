import importlib.util
import json
import sys
from pathlib import Path

from conftest import REPO_ROOT

SCRIPT_PATH = REPO_ROOT / "examples" / "reproduce_cpc_seed_family.py"


def test_cpc_seed_family_script_writes_report_recipe_and_png(tmp_path, capsys):
    module = _load_reproduction_script()

    exit_code = module.main(
        [
            str(tmp_path),
            "--pressure-scale",
            "2500",
            "--pressure-scale",
            "5000",
            "--nr",
            "7",
            "--nz",
            "7",
            "--iterations",
            "2",
            "--dtype",
            "float32",
        ]
    )

    assert exit_code == 0
    emitted = json.loads(capsys.readouterr().out)
    report_path = Path(emitted["report"])
    png_path = Path(emitted["png"])

    assert emitted["artifact_id"] == "tokamaker-cpc-seed-equilibrium-family"
    assert report_path.exists()
    assert png_path.exists()
    assert png_path.read_bytes().startswith(b"\x89PNG")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1.0"
    assert payload["artifact_id"] == "tokamaker-cpc-seed-equilibrium-family"
    assert payload["status"] == "surrogate_fixture"
    assert payload["runtime"]["network"] == "not_used"
    assert payload["source"]["doi"] == "10.1016/j.cpc.2024.109111"
    assert "Hansen" in payload["source"]["citation"]
    assert payload["comparison_rule"]["type"] == "surrogate_seed_family"
    assert payload["comparison_rule"]["tolerance"] is None
    assert len(payload["family"]) == 2
    assert payload["family"][0]["config"]["pressure_scale"] == 2500.0
    assert payload["family"][0]["diagnostics"]["residual"]["final"] is not None

    recipe = payload["figure_recipe"]
    assert recipe["source"] == payload["source"]["source"]
    assert recipe["citation"] == payload["source"]["citation"]
    assert recipe["command"].startswith("python examples/reproduce_cpc_seed_family.py")
    assert recipe["data"]["psi"]["shape"] == [7, 7]
    assert recipe["data"]["source"]["shape"] == [7, 7]
    assert recipe["metadata"]["artifact_id"] == payload["artifact_id"]
    assert recipe["metadata"]["reproduction_scope"]["status"] == "surrogate_fixture"
    assert recipe["metadata"]["reproduction_scope"]["parity_claim"] == "none"


def _load_reproduction_script():
    spec = importlib.util.spec_from_file_location("reproduce_cpc_seed_family", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
