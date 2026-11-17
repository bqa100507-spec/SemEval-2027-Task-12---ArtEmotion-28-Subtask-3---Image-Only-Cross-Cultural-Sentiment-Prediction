"""EXP-026 exploratory notebook contract; cloud calls are mocked."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


def notebook_code():
    nb=json.loads((ROOT/"notebooks/EXP026_Exploratory_Colab.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and not cell["outputs"]
    settings={};exec("".join(nb["cells"][1]["source"]),settings)
    scope={};exec("".join(nb["cells"][2]["source"]),scope)
    return settings,scope


def test_fixed_stage_separate_artifacts_shared_frozen_cache_and_budget():
    settings,scope=notebook_code()
    assert settings["AUTHORIZE_EXP026"] is False
    assert settings["STAGE"]=="exploratory-fold" and settings["MAX_HOURS"]==2
    assert settings["PERSISTENT_ROOT"].endswith("/exp026/exploratory")
    assert settings["FEATURE_CACHE_ROOT"].endswith("/exp026/features")
    command=scope["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,settings,6000)
    assert command[command.index("--stage")+1]=="exploratory-fold"
    assert command[command.index("--cache")+1]==settings["FEATURE_CACHE_ROOT"]
    assert "exploratory_run_" in command[command.index("--run-dir")+1]
    for stage in ["pilot","one-fold","full-cv"]:
        with pytest.raises(ValueError):scope["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,{**settings,"STAGE":stage},6000)


@pytest.mark.parametrize("failure",["authorization","budget","stage","mount","download","export"])
def test_setup_failures_export_then_release_without_launching_gpu(tmp_path,monkeypatch,failure):
    settings,scope=notebook_code();events=[]
    settings.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"persistent"))
    settings["AUTHORIZE_EXP026"]=failure in ("budget","stage")
    if failure=="budget":settings["MAX_HOURS"]=2.01
    if failure=="stage":settings["STAGE"]="one-fold"
    def mount(*args):
        if failure=="mount":raise RuntimeError("CPU mock mount failure")
    def download(path):
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
        events.append("download")
        if failure=="download":raise RuntimeError("CPU mock blocked download")
    monkeypatch.setattr(scope["subprocess"],"run",lambda *a,**k:pytest.fail("No GPU/dependency execution allowed"))
    if failure=="export":
        monkeypatch.setattr(scope["zipfile"],"ZipFile",lambda *a,**k:(_ for _ in ()).throw(IOError("CPU mock export failure")))
    result=scope["run_notebook"](settings,SimpleNamespace(mount=mount),SimpleNamespace(download=download),SimpleNamespace(unassign=lambda:events.append("release")))
    assert result["status"]=="EXECUTION_STOP" and events[-1]=="release"
    assert result["browser_arrival"]=="UNVERIFIED"
    if failure not in ("export","mount"):assert events==["download","release"] and result["persistent_readback"]=="PASS"
    if failure=="budget":assert "at most2 hours" in result["error"]
