import ast
import json
from pathlib import Path
import sys
import types

import pytest


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
    assert "nf4_smoke" in codes[1] and "fp16_smoke" in codes[1]
    assert codes[1].count('input("ONE SMOKE REVIEW') == 1
    assert "five_fold_heads" in codes[1] and "--review-file" in codes[1]


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


def mock_probe_namespace(tmp_path,monkeypatch):
    google=types.ModuleType("google");colab=types.ModuleType("google.colab")
    colab.drive=types.SimpleNamespace(mount=lambda path:None)
    colab.files=types.SimpleNamespace(download=lambda path:None)
    colab.runtime=types.SimpleNamespace(unassign=lambda:None)
    google.colab=colab
    monkeypatch.setitem(sys.modules,"google",google);monkeypatch.setitem(sys.modules,"google.colab",colab)
    cells=["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"]=="code"]
    values={};exec(cells[0],values)
    values.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"))
    exec(cells[1],values)
    events=[]
    def run(command,name,env):
        events.append(name)
        if name=="smoke_comparison":
            output=Path(command[-1]);output.mkdir(parents=True,exist_ok=True)
            (output/"smoke_comparison.json").write_text(json.dumps({"technical_status":"PASS","fp16_smoke":{"identity":"fixed","records":["fixture"]}}))
        if name=="five_fold_heads":
            output=Path(command[-1]);output.mkdir(parents=True,exist_ok=True)
            (output/"completed.json").write_text(json.dumps({"status":"FIVE_FOLD_COMPLETE","folds":[1,2,3,4,5]}))
            (output/"summary.csv").write_text("CPU mock only")
            (output/"contribution_evidence.json").write_text("{}")
    values["run"]=run;values["show_comparison"]=lambda directory:events.append("gallery")
    return values,events


@pytest.mark.parametrize("answer,expected",[("APPROVE: isolated errors acceptable","FIVE_FOLD_COMPLETE"),("REJECT: scenes systematically wrong","STOPPED_SYSTEMATIC_HALLUCINATION")])
def test_one_smoke_review_then_automatic_completion_or_rejection(tmp_path,monkeypatch,answer,expected):
    values,events=mock_probe_namespace(tmp_path,monkeypatch)
    prompts=[]
    def respond(prompt):prompts.append(prompt);return answer
    monkeypatch.setattr("builtins.input",respond)
    args=("python",tmp_path/"source",tmp_path/"inputs",tmp_path/"drive",tmp_path/"attempt",{})
    result=values["probe_stages"](*args)
    assert result["status"]==expected and len(prompts)==1
    assert events[:4]==["nf4_smoke","fp16_smoke","smoke_comparison","gallery"]
    if answer.startswith("APPROVE"):
        assert events[-2:]==["full_extraction","five_fold_heads"]
    else:
        assert "full_extraction" not in events and "five_fold_heads" not in events
        assert result["decision"]=="ALTERNATIVE_VLM_REPRESENTATION_NEEDED"
    events.clear()
    result=values["probe_stages"](*args)
    assert result["status"]==expected and len(prompts)==1  # Exact-smoke review persists across resumes.


def test_technical_failure_never_requests_semantic_review_or_heads(tmp_path,monkeypatch):
    values,events=mock_probe_namespace(tmp_path,monkeypatch)
    monkeypatch.setattr("builtins.input",lambda *args:pytest.fail("Technical failure must stop before review"))
    def fail(*args):raise RuntimeError("Corrupt image/embedding fixture")
    values["run"]=fail
    with pytest.raises(RuntimeError,match="Corrupt"):
        values["probe_stages"]("python",tmp_path/"source",tmp_path/"inputs",tmp_path/"drive",tmp_path/"attempt",{})
