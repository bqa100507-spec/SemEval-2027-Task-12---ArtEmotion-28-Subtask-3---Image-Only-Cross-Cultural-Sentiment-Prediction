import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

from st3.experiment_workflow import validate_notebook_contract

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("exp023_notebook_builder", ROOT/"scripts/build_exp023_notebooks.py")
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fake_colab(monkeypatch, events, *, mount_failure=False, download_failure=False):
    def mount(path):
        events.append("mount")
        if mount_failure:raise RuntimeError("mount denied")
    def download(path):
        events.append("download")
        if download_failure:raise RuntimeError("browser blocked")
    google=types.ModuleType("google")
    colab=types.ModuleType("google.colab")
    colab.drive=types.SimpleNamespace(mount=mount)
    colab.files=types.SimpleNamespace(download=download)
    colab.runtime=types.SimpleNamespace(unassign=lambda:events.append("release"))
    google.colab=colab
    monkeypatch.setitem(sys.modules,"google",google)
    monkeypatch.setitem(sys.modules,"google.colab",colab)


@pytest.mark.parametrize("mount_failure,download_failure",[(False,False),(False,True),(True,False)])
def test_notebook_bootstrap_checksum_or_mount_failure_cleanup(tmp_path,monkeypatch,mount_failure,download_failure):
    events=[]
    fake_colab(monkeypatch,events,mount_failure=mount_failure,download_failure=download_failure)
    archive=tmp_path/"bad.zip";archive.write_bytes(b"bad checksum")
    values=dict(PLATFORM="colab", AUTO_RELEASE_COLAB=True, AUTO_BROWSER_DOWNLOAD=True,INPUT_ZIP=str(archive),
                ZIP_SHA256="0"*64,MANIFEST_SHA256="0"*64,SOURCE_REVISION="f"*40,ENTRY_SHA256="0"*64,
                PERSISTENT_ROOT=str(tmp_path/"drive"),LOCAL_ROOT=str(tmp_path/"local"),RUN_STARTED=0.)
    exec(compile(builder.ORCHESTRATION,"notebook_orchestration","exec"),values)
    assert events[-1]=="release"
    if mount_failure:
        assert "download" not in events
        assert values["RESULT"]["persistent_readback"]=="FAIL"
    else:
        assert events.index("download")<events.index("release")
        assert (tmp_path/"drive/export/bootstrap_failure_outputs.zip").exists()
        assert values["RESULT"]["browser_arrival"]=="UNVERIFIED"


def test_notebook_kaggle_has_no_colab_api_calls(tmp_path,monkeypatch):
    values=dict(PLATFORM="kaggle",AUTO_RELEASE_COLAB=True,AUTO_BROWSER_DOWNLOAD=True,INPUT_ZIP=str(tmp_path/"bad.zip"),
                ZIP_SHA256="0"*64,PERSISTENT_ROOT=str(tmp_path/"out"),RUN_STARTED=0.)
    (tmp_path/"bad.zip").write_bytes(b"invalid")
    monkeypatch.setitem(sys.modules,"google.colab",None)
    exec(compile(builder.ORCHESTRATION,"notebook_orchestration","exec"),values)
    assert values["RELEASE_ATTEMPTED"] is False
    assert values["RESULT"]["persistent_readback"]=="PASS"


def test_notebook_generation_syntax_authorization_and_pins(monkeypatch):
    entry=(ROOT/"scripts/exp023_notebook_entry.py").read_bytes()
    monkeypatch.setattr(builder.subprocess,"check_output",lambda *args,**kwargs:entry)
    receipt=dict(source_revision="f"*40,manifest_sha256="a"*64,zip_sha256="b"*64)
    for platform in ("colab","kaggle"):
        notebook=builder.make_notebook(receipt,platform)
        assert validate_notebook_contract(notebook)["status"]=="PASS"
        config="".join(notebook["cells"][1]["source"])
        namespace={}
        exec(compile(config,"owner_configuration","exec"),namespace)
        assert namespace["AUTHORIZE_EXP023"] is False
        assert namespace["EXECUTION_SCOPE"] == "RESEARCH_ONLY_CV"
        assert namespace["REQUIRE_SUBMISSION_ZIP"] is False
        assert namespace["AUTHORIZE_GUARDED_CONFIRMATION"] is False
        assert namespace["CAPTION_ELIGIBILITY_STATUS"]=="UNRESOLVED"
        assert namespace["MAX_GPU_HOURS"] == 4.0
        assert namespace["MAX_TOTAL_WALL_HOURS"] == 5.0
        assert namespace["SOURCE_REVISION"]==receipt["source_revision"]
        assert namespace["ENTRY_SHA256"]==hashlib.sha256(entry).hexdigest()
        assert not any(cell.get("outputs") for cell in notebook["cells"])


def test_trusted_entry_executes_only_after_all_pins_match(tmp_path,monkeypatch):
    events=[]
    fake_colab(monkeypatch,events)
    entry=b'def run_cloud(settings, **hooks):\n hooks["release"]()\n return {"status":"SYNTHETIC_TEST_ONLY"}\n'
    source=io.BytesIO()
    with zipfile.ZipFile(source,"w") as z:z.writestr("scripts/exp023_notebook_entry.py",entry)
    rawsource=source.getvalue()
    manifest=json.dumps({"source_revision":"f"*40,"files":{"source/source.zip":hashlib.sha256(rawsource).hexdigest()}}).encode()
    archive=tmp_path/"inputs.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("input_manifest.json",manifest);z.writestr("source/source.zip",rawsource)
    values=dict(PLATFORM="colab",AUTO_RELEASE_COLAB=True,AUTO_BROWSER_DOWNLOAD=True,INPUT_ZIP=str(archive),
                ZIP_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest(),MANIFEST_SHA256=hashlib.sha256(manifest).hexdigest(),
                ENTRY_SHA256=hashlib.sha256(entry).hexdigest(),SOURCE_REVISION="f"*40,
                LOCAL_ROOT=str(tmp_path/"local"),PERSISTENT_ROOT=str(tmp_path/"drive"),RUN_STARTED=0.)
    exec(compile(builder.ORCHESTRATION,"notebook_orchestration","exec"),values)
    assert values["RESULT"]["status"]=="SYNTHETIC_TEST_ONLY"
    assert events==["mount","release"]
