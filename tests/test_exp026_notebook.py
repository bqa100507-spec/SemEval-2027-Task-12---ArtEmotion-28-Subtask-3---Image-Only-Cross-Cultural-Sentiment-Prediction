"""Notebook orchestration is exercised without CUDA, downloads or private data."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT=Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT/"notebooks/EXP026_Colab.ipynb").read_text(encoding="utf-8"))


def code():
    cells=["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"]=="code"]
    settings={};exec(cells[0],settings)
    scope={};exec(cells[1],scope)
    return settings,scope


def test_syntax_schema_pins_and_only_exp026_stages():
    nb=notebook()
    for cell in nb["cells"]:
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and not cell["outputs"]
    settings,scope=code()
    assert settings["AUTHORIZE_EXP026"] is False and settings["STAGE"]=="pilot"
    assert settings["EXECUTION_SCOPE"]=="RESEARCH_ONLY_CV"
    assert len(settings["SOURCE_REVISION"])==40
    assert settings["ZIP_SHA256"]=="2db4a53679940c7eb63c7c8272b91be8ec0618bb6c04a69ebaa15909e20f64e0"
    assert settings["AUTO_BROWSER_DOWNLOAD"] and settings["AUTO_RELEASE_COLAB"]
    with pytest.raises(ValueError):scope["stage_command"]("python",ROOT,ROOT,ROOT,ROOT,{**settings,"STAGE":"full-cv"},300)
    command=scope["stage_command"]("python",ROOT,ROOT/"inputs",ROOT/"drive",ROOT/"local",settings,300)
    assert command[command.index("--stage")+1]=="pilot" and command[-1]=="--authorize-gpu"
    assert command[command.index("--max-seconds")+1]=="300"
    assert command[command.index("--cache")+1]==str(ROOT/"drive/features")


@pytest.mark.parametrize("failure",["authorization","mount","scope","input","download","export"])
def test_failure_exports_and_release_even_on_bootstrap_stop(tmp_path,monkeypatch,failure):
    settings,scope=code();events=[]
    settings.update(LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),INPUT_ZIP=str(tmp_path/"missing.zip"))
    settings["AUTHORIZE_EXP026"]=failure not in ("authorization","download","export")
    if failure=="scope":settings["EXECUTION_SCOPE"]="COMPETITION_READY"
    def mount(*args):
        events.append("mount")
        if failure=="mount":raise RuntimeError("MOCK mount failure")
    def download(path):
        with zipfile.ZipFile(path) as archive:
            assert "manifest_sha256.json" in archive.namelist()
            assert archive.testzip() is None
        events.append("download")
        if failure=="download":raise RuntimeError("MOCK download failure")
    def no_process(*args,**kwargs):pytest.fail("No GPU or dependency process may launch")
    monkeypatch.setattr(scope["subprocess"],"run",no_process)
    if failure=="export":
        monkeypatch.setattr(scope["zipfile"],"ZipFile",lambda *args,**kwargs:(_ for _ in ()).throw(IOError("MOCK export failure")))
    result=scope["run_notebook"](settings,SimpleNamespace(mount=mount),SimpleNamespace(download=download),SimpleNamespace(unassign=lambda:events.append("release")))
    assert events[-1]=="release" and result["status"]=="EXECUTION_STOP"
    assert result["physical_GPU_deallocation"]=="UNVERIFIED" and result["browser_arrival"]=="UNVERIFIED"
    if failure not in ("mount","export"):
        assert events==["mount","download","release"] and result["persistent_readback"]=="PASS"
        assert list((tmp_path/"drive").rglob("outputs.sha256"))
    if failure=="export":assert "export_error" in result and "download" not in events
    if failure=="download":assert result["download_request"].startswith("FAILED:")


def test_lock_matches_previously_used_runtime_without_new_backbone_packages():
    assert (ROOT/"requirements-exp026.lock.txt").read_text()==(ROOT/"requirements-exp025.lock.txt").read_text()
    configuration=json.loads((ROOT/"configs/exp026.json").read_text())
    assert configuration["vlm"]["revision"]=="66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert configuration["lora"]["expected_modules"]==72
