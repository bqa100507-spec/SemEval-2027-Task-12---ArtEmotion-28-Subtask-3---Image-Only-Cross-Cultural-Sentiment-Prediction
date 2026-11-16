"""Reusable CPU compliance suite. No real browser, cloud release or GPU is invoked."""

import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import pytest

from st3 import experiment_workflow as workflow
from st3.experiment_workflow import Budget, StopRun, execute_stages, export_artifacts, finalize, publish_model, restore_handoff, sha256, validate_notebook_contract, verify_model, write_json
from st3.exp023_cloud import child, run_cloud


def test_full_successful_synthetic_stage_sequence():
    events = []
    status = {"Verify": "PASS", "Decision": "CV_GO", "GuardedConfirmation": "CONFIRMATION_GO", "FinalModel": "PASS", "SubmissionValidation": "PASS"}
    callbacks = {stage: (lambda stage=stage: events.append(stage) or {"status": status.get(stage, "PASS")}) for stage in workflow.STAGES}
    result = execute_stages(callbacks, Budget(1, 2, cuda=False), lambda stage: None)
    assert tuple(events) == workflow.STAGES
    assert result["SubmissionValidation"]["status"] == "PASS"


@pytest.mark.parametrize("stage,status,expected", [("Verify", "FAIL", "FAILED_VERIFICATION"), ("Decision", "CV_NO_GO", "CV_NO_GO"),
    ("GuardedConfirmation", "CONFIRMATION_NO_GO", "CONFIRMATION_NO_GO"), ("FinalModel", "FAIL", "FAILED_VERIFICATION"),
    ("SubmissionValidation", "FAIL", "FAILED_SUBMISSION")])
def test_failed_gate_never_progresses(stage, status, expected):
    events = []
    successful = {"Verify": "PASS", "Decision": "CV_GO", "GuardedConfirmation": "CONFIRMATION_GO", "FinalModel": "PASS", "SubmissionValidation": "PASS"}
    successful[stage] = status
    callbacks = {name: (lambda name=name: events.append(name) or {"status":successful.get(name,"PASS")}) for name in workflow.STAGES}
    with pytest.raises(StopRun) as error:
        execute_stages(callbacks, Budget(1, 2, cuda=False), lambda stage:None)
    assert error.value.status == expected
    assert events[-1] == stage


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_budgets(value):
    with pytest.raises(ValueError):Budget(value, 2, cuda=True)


def test_gpu_wall_and_projection_budget():
    clock = [0.]
    b = Budget(1, 2, cuda=True, clock=lambda:clock[0])
    assert b.remaining() == 3600
    clock[0] = 3400
    b.check(10)
    with pytest.raises(StopRun):b.check(30)
    b = Budget(1, .5, cuda=False, clock=lambda:0.)
    assert b.remaining() == 1800


def test_hard_child_deadline_terminates_cpu_process(tmp_path):
    with pytest.raises(StopRun) as error:
        child([sys.executable, "-c", "import time;time.sleep(20)"], log=tmp_path/"child.log", deadline=time.time()+.1)
    assert error.value.status == "STOPPED_BUDGET"


def test_completed_checkpoint_resume_and_corruption(tmp_path):
    model = tmp_path/"model"
    identity = {"source":"pinned", "seed":2027}
    calls = []
    def writer(stage):
        calls.append(1)
        (stage/"weights.bin").write_bytes(b"synthetic")
    publish_model(model, identity, writer)
    publish_model(model, identity, writer)
    assert len(calls) == 1
    with pytest.raises(StopRun):verify_model(model,{**identity,"seed":2028})
    (model/"weights.bin").write_bytes(b"corrupt")
    with pytest.raises(StopRun):verify_model(model,identity)


def test_interrupted_fit_never_published(tmp_path):
    path = tmp_path/"model"
    def interrupted(stage):
        (stage/"checkpoint.pt.partial").write_bytes(b"not complete")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):publish_model(path,{"identity":1},interrupted)
    assert not path.exists()
    publish_model(path,{"identity":1},lambda stage:(stage/"checkpoint.pt").write_bytes(b"complete"))
    assert verify_model(path,{"identity":1})["complete"] is True


def make_zip(tmp_path, *, extra=None, corrupt=False):
    payload=b"fixture"
    manifest={"files":{"input.bin":hashlib.sha256(payload).hexdigest()}}
    raw=json.dumps(manifest).encode()
    path=tmp_path/"inputs.zip"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("input_manifest.json",raw)
        z.writestr("input.bin",b"bad" if corrupt else payload)
        if extra:z.writestr(extra,b"extra")
    return path,sha256(path),hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("extra", ["../escape", "/absolute", "x\\y", "C:/escape", "extra"])
def test_handoff_rejects_unsafe_or_extra_files(tmp_path,extra):
    path,digest,manifest=make_zip(tmp_path,extra=extra)
    with pytest.raises((StopRun,ValueError)):
        restore_handoff(path,tmp_path/"restored",digest,manifest)
    assert not (tmp_path/"restored").exists()


def test_handoff_clean_restore_resume_and_checksums(tmp_path):
    path,digest,manifest=make_zip(tmp_path)
    target=tmp_path/"restored"
    restore_handoff(path,target,digest,manifest)
    restore_handoff(path,target,digest,manifest)
    with pytest.raises(StopRun):restore_handoff(path,target,"0"*64,manifest)
    with pytest.raises(StopRun):restore_handoff(path,target,digest,"0"*64)
    (target/"input.bin").write_bytes(b"corrupt")
    with pytest.raises(StopRun):restore_handoff(path,target,digest,manifest)


def test_handoff_corruption_never_extracts(tmp_path):
    path,digest,manifest=make_zip(tmp_path,corrupt=True)
    with pytest.raises(StopRun):restore_handoff(path,tmp_path/"restored",digest,manifest)
    assert not (tmp_path/"restored").exists()


@pytest.mark.parametrize("status", ["SUCCESS_SUBMISSION_READY","CV_NO_GO","CONFIRMATION_NO_GO","BLOCKED_ELIGIBILITY","BLOCKED_INPUT","FAILED_HISTORICAL_PARITY","FAILED_TRAINING","FAILED_VERIFICATION","FAILED_SUBMISSION","STOPPED_BUDGET"])
def test_every_terminal_outcome_exports_downloads_then_releases(tmp_path,status):
    run=tmp_path/"run";run.mkdir();(run/"valid.log").write_text("diagnostic")
    if status=="SUCCESS_SUBMISSION_READY":(run/"submission.zip").write_bytes(b"validated synthetic placeholder")
    events=[]
    def exporter(*args):
        result=export_artifacts(*args)
        events.append("verified_export")
        return result
    def download(path):events.append("download:"+Path(path).name)
    report=finalize(run,tmp_path/"drive",{"status":status},platform="colab",download=download,release=lambda:events.append("release"),exporter=exporter)
    assert events[0]=="verified_export" and events[-1]=="release"
    assert "download:outputs.zip" in events
    assert ("download:submission.zip" in events)==(status=="SUCCESS_SUBMISSION_READY")
    assert report["browser_arrival"]==report["physical_runtime_termination"]=="UNVERIFIED"
    assert report["runtime_release"]=="API_RETURNED"


def test_download_failure_preserves_drive_and_cleanup(tmp_path):
    run=tmp_path/"run";run.mkdir();(run/"log").write_text("x")
    events=[]
    def blocked(path):events.append("download");raise RuntimeError("browser blocked")
    result=finalize(run,tmp_path/"drive",{"status":"CV_NO_GO"},platform="colab",download=blocked,release=lambda:events.append("release"))
    assert events==["download","release"]
    assert (tmp_path/"drive/outputs.zip").exists()
    saved=json.loads((tmp_path/"drive/final_report.json").read_text())
    assert saved["downloads"][-1]["status"]=="FAILED"
    assert result["status"]=="CV_NO_GO"


def test_export_failure_still_releases_and_skips_download(tmp_path):
    events=[]
    def bad(*args):raise OSError("Drive unavailable")
    result=finalize(tmp_path,tmp_path/"drive",{"status":"SUCCESS_SUBMISSION_READY"},platform="colab",download=lambda path:events.append("download"),release=lambda:events.append("release"),exporter=bad)
    assert result["status"]=="FAILED_EXPORT" and events==["release"]


def test_persistent_readback_corruption_fails_closed(tmp_path,monkeypatch):
    run=tmp_path/"run";run.mkdir();(run/"log").write_text("x")
    real=workflow.sha256
    def corrupt(path):
        if Path(path)==tmp_path/"drive/outputs.zip":return "bad"
        return real(path)
    monkeypatch.setattr(workflow,"sha256",corrupt)
    events=[]
    result=finalize(run,tmp_path/"drive",{"status":"CV_NO_GO"},platform="colab",download=lambda path:events.append("download"),release=lambda:events.append("release"))
    assert result["status"]=="FAILED_EXPORT" and events==["release"]


def test_kaggle_never_invokes_colab_callbacks(tmp_path):
    run=tmp_path/"run";run.mkdir();(run/"log").write_text("x")
    def forbidden(*args):pytest.fail("Colab API called on Kaggle")
    result=finalize(run,tmp_path/"output",{"status":"CV_NO_GO"},platform="kaggle",download=forbidden,release=forbidden)
    assert result["persistent_readback"]=="PASS"


@pytest.mark.parametrize("authorized,mount_failure", [(False,False),(True,False),(True,True)])
def test_cloud_bootstrap_failures_release(tmp_path,authorized,mount_failure):
    events=[]
    settings={"EXECUTION_SCOPE":"COMPETITION_READY","LOCAL_ROOT":str(tmp_path/"local"),"PERSISTENT_ROOT":str(tmp_path/"drive"),"AUTHORIZE_EXPERIMENT":authorized,
              "MAX_GPU_HOURS":1.,"MAX_TOTAL_WALL_HOURS":2.,"INPUT_ZIP":str(tmp_path/"missing.zip"),
              "ZIP_SHA256":"0"*64,"MANIFEST_SHA256":"0"*64,"SOURCE_REVISION":"x", "AUTO_BROWSER_DOWNLOAD":True,"AUTO_RELEASE_COLAB":True}
    def mount():
        if mount_failure:raise RuntimeError("mount failed")
    result=run_cloud(settings,platform="colab",mount=mount,download=lambda path:None,release=lambda:events.append("release"))
    assert events==["release"]
    assert result["status"] in (("FAILED_EXPORT",) if mount_failure else ("BLOCKED_INPUT","BLOCKED_AUTHORIZATION"))


def test_no_go_export_excludes_stale_promotion_archive(tmp_path):
    run=tmp_path/"run";run.mkdir();(run/"submission.zip").write_bytes(b"old")
    published=export_artifacts(run,tmp_path/"drive",{"status":"CV_NO_GO"})
    assert "submission.zip" not in published
    with zipfile.ZipFile(published["outputs.zip"]) as z:assert "submission.zip" not in z.namelist()


def test_single_configuration_and_one_entry_point_validation():
    config="# OWNER_CONFIGURATION\n"+"\n".join(f"{key} = None" for key in workflow.CONFIGURATION)
    nb={"cells":[{"cell_type":"code","source":[config]},{"cell_type":"code","source":["run_cloud(settings)"]}]}
    assert validate_notebook_contract(nb)["status"]=="PASS"
    nb["cells"].append(nb["cells"][0])
    with pytest.raises(ValueError):validate_notebook_contract(nb)


def test_eligibility_blocks_bootstrap_and_training(tmp_path,monkeypatch):
    from st3 import exp023_cloud
    source=tmp_path/"source";(source/"configs").mkdir(parents=True)
    (source/"configs/exp023_eligibility.json").write_text(json.dumps({"status":"ELIGIBILITY_UNRESOLVED"}))
    (source/"configs/exp023_runtime.json").write_text(json.dumps({"persistence_reserve_seconds":180}))
    monkeypatch.setattr(exp023_cloud,"restore_handoff",lambda *args:{"source_revision":"pinned"})
    monkeypatch.setattr(exp023_cloud,"restore_source",lambda *args:source)
    monkeypatch.setattr(exp023_cloud,"bootstrap",lambda *args:pytest.fail("Environment/model launched before eligibility"))
    settings={"EXECUTION_SCOPE":"COMPETITION_READY","LOCAL_ROOT":str(tmp_path/"local"),"PERSISTENT_ROOT":str(tmp_path/"drive"),"AUTHORIZE_EXPERIMENT":True,
              "MAX_GPU_HOURS":1.,"MAX_TOTAL_WALL_HOURS":2.,"INPUT_ZIP":"unused", "ZIP_SHA256":"x","MANIFEST_SHA256":"x",
              "SOURCE_REVISION":"pinned","CAPTION_ELIGIBILITY_STATUS":"VERIFIED","AUTO_BROWSER_DOWNLOAD":True,"AUTO_RELEASE_COLAB":True}
    events=[]
    result=run_cloud(settings,platform="colab",mount=lambda:None,download=lambda path:events.append("download"),release=lambda:events.append("release"))
    assert result["status"]=="BLOCKED_ELIGIBILITY"
    assert events==["download","release"]
