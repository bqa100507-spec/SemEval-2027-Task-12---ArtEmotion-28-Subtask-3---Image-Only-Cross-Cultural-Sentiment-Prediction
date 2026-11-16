import ast
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT / "notebooks/EXP025_Colab.ipynb").read_text(encoding="utf-8"))


def test_single_run_all_pins_and_script_paths():
    nb = notebook()
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(codes) == 2
    namespace = {};exec(codes[0], namespace)
    assert namespace["AUTHORIZE_EXP025"] is False
    assert namespace["MAX_HOURS"] == 2.
    assert len(namespace["ZIP_SHA256"]) == 64
    for code in codes:ast.parse(code)
    assert "scripts/extract_exp025.py" in codes[1] and "scripts/run_exp025.py" in codes[1]
    assert "--full" in codes[1] and "check" in codes[1]
    assert (ROOT / "requirements-exp025.lock.txt").exists()
    assert not any(c.get("outputs") for c in nb["cells"])


def test_notebook_unauthorized_run_exports_and_releases_without_training(tmp_path, monkeypatch):
    events = []
    google = types.ModuleType("google");colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path:events.append("mount"))
    colab.files = types.SimpleNamespace(download=lambda path:events.append("download"))
    colab.runtime = types.SimpleNamespace(unassign=lambda:events.append("release"))
    google.colab = colab
    monkeypatch.setitem(sys.modules,"google",google);monkeypatch.setitem(sys.modules,"google.colab",colab)
    cells = ["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code"]
    namespace = {};exec(cells[0],namespace)
    namespace.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"))
    import subprocess
    monkeypatch.setattr(subprocess,"run",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("No command authorized")))
    exec(cells[1],namespace)
    assert events == ["mount","download","release"]
    assert namespace["result"]["persistent_readback"] == "PASS"
    assert "AUTHORIZE_EXP025" in namespace["result"]["error"]
